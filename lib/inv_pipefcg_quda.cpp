#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <iostream>

#include <functional>
#include <limits>


/***
* Experimental PipePCG algorithm
* Sources:
* P. Ghysels and W. Vanroose "Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm"
* S. Cools et al, "Analyzing the effect of local rounding error propagation on the maximal attainable accuracy of the
*                   pipelined Conjugate Gredient method ",  arXiv:1601.07068
***/

//#define FULL_PIPELINE


#include <mpi.h>

#define MPI_CHECK_(mpi_call) do {                   		\
  int status = comm_size() == 1 ? MPI_SUCCESS : mpi_call;	\
  if (status != MPI_SUCCESS) {                      		\
    char err_string[128];                           		\
    int err_len;                                    		\
    MPI_Error_string(status, err_string, &err_len); 		\
    err_string[127] = '\0';                         		\
    errorQuda("(MPI) %s", err_string);              		\
  }                                                 		\
} while (0)


namespace quda {

  using namespace blas;

  // set the required parameters for the inner solver
  static void fillInnerSolverParam(SolverParam &inner, const SolverParam &outer)
  {
    inner.tol = outer.tol_precondition;
    inner.maxiter = outer.maxiter_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver
    inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // used to tell the inner solver it is an inner solver

    if(outer.inv_type == QUDA_PIPEPCG_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  }


  PipeFCG::PipeFCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0), Kparam(param), init(false)
  {
    fillInnerSolverParam(Kparam, param);

    if(param.inv_type_precondition == QUDA_CG_INVERTER){
      K = new CG(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_MR_INVERTER){
      K = new MR(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_SD_INVERTER){
      K = new SD(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition != QUDA_INVALID_INVERTER){ // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
    }
  }

  PipeFCG::~PipeFCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {

      if (K) {
        delete p_pre;
        delete r_pre;
      }

      delete tmpp;
      delete rp;

      delete up;
      delete wp;
      delete mp;
      delete np;

      delete pp;
      delete sp;
      delete qp;
      delete zp;
    }

    if(K) delete K;

     profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void PipeFCG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    profile.TPSTART(QUDA_PROFILE_INIT);

    const double norm2b = sqrt(norm2(b));
    const int mmax      = param.Nkrylov;

    printfQuda("Running pipeFCG with %d.\n", mmax);

    if(norm2b == 0){
      printfQuda("Warning: inverting on zero-field source\n");
      blas::copy(x, b);
      param.true_res = 0.0;
      profile.TPSTOP(QUDA_PROFILE_INIT);
      return;
    }

    ColorSpinorParam csParam(b);

    if (!init) {
      // high precision fields:
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      tmpp = ColorSpinorField::Create(csParam); //temporary for sloppy mat-vec

      csParam.setPrecision(param.precision_sloppy);
      up = ColorSpinorField::Create(csParam);
      wp = ColorSpinorField::Create(csParam);
      mp = ColorSpinorField::Create(csParam);
      np = ColorSpinorField::Create(csParam);
      //
      csParam.is_composite  = true;          //working with array of spinor fields
      csParam.composite_dim = mmax+1;          //
      pp = ColorSpinorField::Create(csParam);
      sp = ColorSpinorField::Create(csParam);
      zp = ColorSpinorField::Create(csParam);
      qp = ColorSpinorField::Create(csParam);
      csParam.is_composite  =false;

      // these low precision fields are used by the inner solver
      if (K) {
        printfQuda("Allocated resources for the preconditioner.\n");
        csParam.setPrecision(param.precision_precondition);
        p_pre = ColorSpinorField::Create(csParam);
        r_pre = ColorSpinorField::Create(csParam);
      }

      init = true;
    }

    csParam.setPrecision(param.precision);
    ColorSpinorField *yp = ColorSpinorField::Create(csParam);

    ColorSpinorField *rp_sloppy = nullptr, *tmpp_sloppy = nullptr, *xp_sloppy = nullptr;

    if(param.precision_sloppy != param.precision) {
      csParam.setPrecision(param.precision_sloppy);
      rp_sloppy   = ColorSpinorField::Create(csParam);
      tmpp_sloppy = ColorSpinorField::Create(csParam);
      xp_sloppy   = ColorSpinorField::Create(csParam);
    } else {
      rp_sloppy   = rp;
      tmpp_sloppy = tmpp;
    }

    ColorSpinorField &r = *rp;

    ColorSpinorField &p = *pp;
    ColorSpinorField &s = *sp;
    ColorSpinorField &q = *qp;
    ColorSpinorField &z = *zp;
    //
    ColorSpinorField &u = *up;
    ColorSpinorField &w = *wp;
    ColorSpinorField &m = *mp;
    ColorSpinorField &n = *np;

    ColorSpinorField &y = *yp;

    ColorSpinorField &rSloppy = *rp_sloppy;
    ColorSpinorField &xSloppy = (/* param.use_sloppy_partial_accumulator == true &&*/ param.precision_sloppy != param.precision ) ? *xp_sloppy : x;

    ColorSpinorField &pPre = *p_pre;
    ColorSpinorField &rPre = *r_pre;

    ColorSpinorField &tmp        = *tmpp;
    ColorSpinorField &tmpSloppy  = *tmpp_sloppy;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);


    blas::flops = 0;

    std::unique_ptr<double[] > beta(new double[3+(mmax+1)]);
    std::unique_ptr<double[] > eta(new double[mmax+1]);
    std::unique_ptr<double[] > recvbuff(new double[mmax+1+3]);

    double alpha = 0.0;
    double &gamma = beta[0], &delta = beta[1], &rnorm = beta[2];
    const int beta_buffer_offset = 3;

    MPI_Request request_handle;

    // Compute residual
    mat(r, x, tmp);
    xpay(b, -1.0, r);
    rSloppy  = r;
    y = x;

    if(K) {
      rPre = rSloppy;
      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);
      u = pPre;
      blas::zero(m);
    } else {
      u = rSloppy;
    }

    matSloppy(w, u, tmpSloppy);
    zero(xSloppy);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // zero cycle
    commGlobalReductionSet(false);
    gamma  = reDotProduct(rSloppy, u);
    delta  = reDotProduct(w, u);
    rnorm  = norm2(rSloppy);
    commGlobalReductionSet(true);

    //Start async communications:
    MPI_CHECK_(MPI_Iallreduce(beta.get(), recvbuff.get(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));

    if(K) {
      rPre = w;
      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);
      m = pPre;
    } else {
      m = w;
    }
    matSloppy(n, m, tmpSloppy);

    MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
    if (comm_size() > 1) memcpy(beta.get(), recvbuff.get(), 3*sizeof(double));

    rnorm = sqrt(rnorm);
    printfQuda(" PipeFCG: Initial (relative) residual %le \n", rnorm / norm2b);
    //
    double stop = rnorm*rnorm*param.tol*param.tol;
    double heavy_quark_res = 0.0;    

    eta[0]   = delta; alpha  = gamma / delta;

    double theta  = 1.0;

    p[0] = u;
    s[0] = w;
    q[0] = m;
    z[0] = n;
    axpy( alpha, p[0], xSloppy);
    axpy(-alpha, s[0], rSloppy);
    axpy(-alpha, q[0], u);
    axpy(-alpha, z[0], w);

    commGlobalReductionSet(false);
    gamma   = reDotProduct(rSloppy, u);
    delta   = reDotProduct(w, u);
    rnorm   = norm2(rSloppy);

    beta[beta_buffer_offset+0] = reDotProduct(s[0], u); //beta[0]
    commGlobalReductionSet(true);

    //Start async communications:
    MPI_CHECK_(MPI_Iallreduce(beta.get(), recvbuff.get(), 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));

    //m = u + Precond(w-r)
    //Precond(m, n)
    if(K) {
      n = w;
      axpy(-theta, rSloppy, n);
      rPre = n;

      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);
      m = pPre;
      axpy(theta, u, m);

    } else {
      m = w;
    }
    matSloppy(n, m, tmpSloppy);

    MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
    if (comm_size() > 1) memcpy(beta.get(), recvbuff.get(), 4*sizeof(double));

    rnorm = sqrt(rnorm);
    printfQuda(" Before main loop : Initial residual = ", rnorm / norm2b);

    // Iteration control params:
    int k = 1;
    bool is_converged = false;

    //Truncation length (Notay version):
    int nu = 1;

    while (k < param.maxiter && is_converged == false) {

      int idx = k % (mmax + 1);
      // must finish all global comms here
      // eta[idx] = delta -sum beta[j]*beta[j]*eta[k]
      eta[idx] = 0.0;

      for (int i = std::max(0,k-nu), j = beta_buffer_offset; i < k; i++, j++) {
        int kdx = (i % (mmax+1));
        beta[j] /= -eta[kdx];
	eta[idx] -= ((beta[j])*(beta[j])) * eta[kdx];
      }
      // additional check
      eta[idx] += delta;
      if(eta[idx] <= 0.) {
        printfQuda("Restart due to square root breakdown or exact zero of eta at it = %d ", k);
        break;
      } else {
        alpha = gamma / eta[idx];
      }

      // project out stored search directions
      p[idx] = u;
      s[idx] = w;
      q[idx] = m;
      z[idx] = n;

      for (int i = std::max(0,k-nu), j = beta_buffer_offset; i < k; i++, j++) {
        int kdx = (i % (mmax+1));

        axpy( beta[j], p[kdx], p[idx]);
        axpy( beta[j], s[kdx], s[idx]);
        axpy( beta[j], q[kdx], q[idx]);
        axpy( beta[j], z[kdx], z[idx]);
      }

      // Update x, r, z, w
      axpy( +alpha, p[idx], xSloppy);
      axpy( -alpha, s[idx], rSloppy);
      axpy( -alpha, q[idx], u);
      axpy( -alpha, z[idx], w);

      nu = ((k) % mmax)+1;

      commGlobalReductionSet(false);
      gamma = reDotProduct(rSloppy, u);
      delta = reDotProduct(w, u);
      rnorm = norm2(rSloppy);

      for (int i = std::max(0,k-nu+1), j = beta_buffer_offset; i <= k; i++, j++) {
        int kdx = (i % (mmax+1));
        beta[j] = reDotProduct(s[kdx], u);
      }
      commGlobalReductionSet(true);

      //Start async communications:
      const auto buffer_size = 3+k-std::max(0,k-nu+1)+1;
      MPI_CHECK_(MPI_Iallreduce(beta.get(), recvbuff.get(), buffer_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));

      //m = u + Precond(w-r)
      //Precond(m, n)
      if(K) {
        n = w;
        axpy(-theta, rSloppy, n);
        rPre = n;

        commGlobalReductionSet(false);
        (*K)(pPre, rPre);
        commGlobalReductionSet(true);
        m = pPre;
        axpy(theta, u, m);

      } else {
        m = w;
      }
      matSloppy(n, m, tmpSloppy);

      MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
      if (comm_size() > 1) memcpy(beta.get(), recvbuff.get(), buffer_size*sizeof(double));

      rnorm = sqrt(rnorm);
      // Check for convergence:
      is_converged = ((rnorm*rnorm) < stop);

      printfQuda("PipeFCG(%d:%d): %d iteration, iter residual: %1.15e \n", mmax, nu, k, rnorm/norm2b);

      k += 1;
    } // while


    printfQuda("Finish PipeFCG: %d iterations, total restarst: ?? \n", k);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matPrecon.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k==param.maxiter)  warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual
    x = xSloppy;
    xpy(x, y);
    x = y;
    mat(r, y, tmp);
    param.true_res = sqrt(blas::xmyNorm(b, r)) / norm2b;

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    delete yp;

    if (param.precision_sloppy != param.precision) {
      delete rp_sloppy;
      delete xp_sloppy;
      delete tmpp_sloppy;
    }

    return;

  }

#undef MPI_CHECK_

} // namespace quda
