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


  PipePCG::PipePCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
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

  PipePCG::~PipePCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {

      if (K) {
        delete p_pre;
        delete r_pre;
      }

      delete tmpp;
      delete rp;

      delete work_space;
    }

    if(K) delete K;

     profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void PipePCG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    profile.TPSTART(QUDA_PROFILE_INIT);

    double norm2b = sqrt(norm2(b));

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
      csParam.is_composite  =true;
      csParam.composite_dim = 8;
      work_space = ColorSpinorField::Create(csParam);
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

    ColorSpinorField &p = (*work_space)[0];
    ColorSpinorField &s = (*work_space)[1];
    ColorSpinorField &u = (*work_space)[2];
    ColorSpinorField &w = (*work_space)[3];
    ColorSpinorField &q = (*work_space)[4];
    ColorSpinorField &n = (*work_space)[5];
    ColorSpinorField &m = (*work_space)[6];
    ColorSpinorField &z = (*work_space)[7];

    ColorSpinorField &y = *yp;

    ColorSpinorField &rSloppy = *rp_sloppy;
    ColorSpinorField &xSloppy = (/* param.use_sloppy_partial_accumulator == true &&*/ param.precision_sloppy != param.precision ) ? *xp_sloppy : x;

    ColorSpinorField &pPre = *p_pre;
    ColorSpinorField &rPre = *r_pre;

    ColorSpinorField &tmp        = *tmpp;
    ColorSpinorField &tmpSloppy  = *tmpp_sloppy;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    const double epsln     = param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon() : ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon() : pow(2.,-13));
    const double sqrteps   = param.delta*sqrt(epsln);

    double Dcr   = 0.0, Dcs      = 0.0, Dcw      = 0.0, Dcz      = 0.0;
    double errr  = 0.0, errrprev = 0.0, errs     = 0.0, errw     = 0.0, errz = 0.0;
    double alpha = 0.0, beta     = 0.0, alphaold = 0.0, gammaold = 0.0, eta  = 0.0;

    double3 *local_reduce = new double3[2];

    blas::flops = 0;

    double *recvbuff = new double[6];
    MPI_Request request_handle;

    double &gamma = local_reduce[0].x, &delta = local_reduce[0].y, &rnorm = local_reduce[0].z;
    double &sigma = local_reduce[1].x, &zeta  = local_reduce[1].y, &tau   = local_reduce[1].z;

    // Compute residual
    mat(r, x, tmp);
    rnorm    = sqrt(xmyNorm(b,r));
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

    double stop = rnorm*rnorm*param.tol*param.tol;
    double heavy_quark_res = 0.0;
    printfQuda(" PipePCG: Initial (relative) residual %le \n", rnorm / norm2b);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    // zero cycle
    commGlobalReductionSet(false);
    gamma  = reDotProduct(rSloppy, u);
    delta  = reDotProduct(w, u);
    commGlobalReductionSet(true);

    //Start async communications:
    MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));

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
    if (comm_size() > 1) memcpy(local_reduce, recvbuff, 2*sizeof(double));

    eta      = delta;
    alpha    = gamma / eta, beta = 0.0;
    alphaold = 0.0, gammaold = 0.0, tau = 0.0;

    double theta  = 1.0;

    z = n;
    q = m;
    p = u;
    s = w;
    axpy( alpha, p, xSloppy);
    axpy(-alpha, q, u);
    axpy(-alpha, z, w);
    axpy(-alpha, s, rSloppy);
    n = w;
    axpy(-theta, rSloppy, n);

    commGlobalReductionSet(false);
    gammaold = gamma;
    gamma   = reDotProduct(rSloppy, u);
    tau     = reDotProduct(s, u);
    delta   = reDotProduct(w, u);
    rnorm   = norm2(rSloppy);
    sigma   = norm2(s);
    zeta    = norm2(z);
    commGlobalReductionSet(true);

    //Start async communications:
    MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));

    if(K) {
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
    if (comm_size() > 1) memcpy(local_reduce, recvbuff, 6*sizeof(double));

    rnorm = sqrt(rnorm);
    sigma = sqrt(sigma);
    zeta  = sqrt(zeta);

    int k = 1, totResUpdates = 0;
    bool is_converged = false, rUpdate  = false;

    PrintStats( "PipePCG (before main loop)", k, rnorm*rnorm, norm2b*norm2b, heavy_quark_res);

    while (k < param.maxiter && !is_converged) {

      beta = -tau / eta;
      eta = delta - beta*beta*eta;
      alphaold = alpha; alpha = gamma / eta;
      gammaold = gamma;

      Dcr = (2.0*alphaold*sigma)*epsln;
      Dcs = (2.0*beta*sigma+2.0*alphaold*zeta)*epsln;
      Dcw = (2.0*alphaold*zeta)*epsln;
      Dcz = (2.0*beta*zeta)*epsln;

      //  Merged kernel:
      //  z = n + beta * z
      //  q = m + beta * q
      //  p = u + beta * p
      //  s = w + beta * s
      //  x = x + alpha * p
      //  u = u - alpha * q
      //  w = w - alpha * z
      //  r = r - alpha * s
      //  n = w - r
      //  local_reduce[0].x = (r, u) gamma
      //  local_reduce[0].y = (s, u) delta
      //  local_reduce[0].w = (r, r) rnorm
      //  local_reduce[1].x = (s, s) sigma
      //  local_reduce[1].y = (z, z) zeta
      //  local_reduce[1].w = (w, u) tau

      if (!rUpdate) {
        commGlobalReductionSet(false);
        pipePCGRRMergedOp(local_reduce,2,xSloppy,alpha,p,u,rSloppy,s,m,beta,q,w,n,z);
        commGlobalReductionSet(true);
        //
        MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
      } else {
        printfQuda("Do reliable update...\n");
        xpay(u, beta, p);
        axpy( alpha, p, xSloppy);
        //
        x = xSloppy;
        xpy(x,y);
        zero(xSloppy);

        mat(r,y, tmp);
        xpay(b,-1.0,r);
   	    rSloppy = r;

	      if(K) {
          rPre = r;
          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);
          u = pPre;
        } else {
          u = r;
        }

        matSloppy(w,u,tmpSloppy);
        matSloppy(s,p,tmpSloppy);

	      if(K) {
          rPre = s;
          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);
          q = pPre;
        } else {
          q = s;
        }
        matSloppy(z,q,tmpSloppy);

        commGlobalReductionSet(false);
        gamma   = reDotProduct(rSloppy, u);
        tau     = reDotProduct(s, u);
        delta   = reDotProduct(w, u);
        rnorm   = norm2(rSloppy);
        sigma   = norm2(s);
        zeta    = norm2(z);
        commGlobalReductionSet(true);

        MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));

        n = w;
        axpy(-theta, rSloppy, n);

        totResUpdates +=1;
      }

      if(K) {
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
      if (comm_size() > 1) memcpy(local_reduce, recvbuff, 6*sizeof(double));

      sigma = sqrt(sigma);
      zeta  = sqrt(zeta);
      rnorm = sqrt(rnorm);
      //
      if (k == 1 || rUpdate) {
        printfQuda("(Re-)initialize reliable parameters..\n");
        errrprev = errr;
        errr = Dcr;
        errs = Dcs;
        errw = Dcw;
        errz = Dcz;
        rUpdate = false;
      } else {
        errrprev = errr;
        errr = errr + alphaold*errs + Dcr;
        errs = beta*errs + errw + alphaold*errz + Dcs;
        errw = errw + alphaold*errz + Dcw;
        errz = beta*errz + Dcz;
      }

      // Do we need to refine iterative residual
      rUpdate = (gamma < 0.0) || ( (k > 1 && errrprev <= (sqrteps * sqrt(gammaold)) && errr > (sqrteps * sqrt(abs(gamma)))));

      // Check convergence:
      is_converged = ((rnorm*rnorm) < stop);

      PrintStats( "PipePCG", k, rnorm*rnorm, norm2b*norm2b, heavy_quark_res);

      // Update iter index
      k += 1;

    } //end while

    delete [] recvbuff;

    printfQuda("Finish PipeFCG: %d iterations, total restarst: %d \n", k, totResUpdates);

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

    delete[] local_reduce;

    if (param.precision_sloppy != param.precision) {
      delete rp_sloppy;
      delete xp_sloppy;
      delete tmpp_sloppy;
    }

    return;

  }

#undef MPI_CHECK_

} // namespace quda
