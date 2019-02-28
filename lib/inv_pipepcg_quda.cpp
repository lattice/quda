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

#define MPI_CHECK_(mpi_call) do {                    \
  int status = mpi_call;                            \
  if (status != MPI_SUCCESS) {                      \
    char err_string[128];                           \
    int err_len;                                    \
    MPI_Error_string(status, err_string, &err_len); \
    err_string[127] = '\0';                         \
    errorQuda("(MPI) %s", err_string);              \
  }                                                 \
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

      delete pp;

      delete tmpp;
      delete rp;

      delete zp;
      delete qp;
      delete sp;
      delete up;
      delete mp;
      delete np;

      delete wp;
    }

    if(K) delete K;

     profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void PipePCG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    profile.TPSTART(QUDA_PROFILE_INIT);

    ColorSpinorParam csParam(b);

    if (!init) {
      // high precision fields:
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      tmpp = ColorSpinorField::Create(csParam); //temporary for sloppy mat-vec

      csParam.setPrecision(param.precision_sloppy);
      pp = ColorSpinorField::Create(csParam);
      zp = ColorSpinorField::Create(csParam);
      wp = ColorSpinorField::Create(csParam);
      sp = ColorSpinorField::Create(csParam);
      qp = ColorSpinorField::Create(csParam);
      np = ColorSpinorField::Create(csParam);
      mp = ColorSpinorField::Create(csParam);
      up = ColorSpinorField::Create(csParam);

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
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField *lp = ColorSpinorField::Create(csParam);

    csParam.setPrecision(param.precision_sloppy);
    ColorSpinorField *yp = ColorSpinorField::Create(csParam);

    ColorSpinorField *rp_sloppy   = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : rp;
    ColorSpinorField *tmpp_sloppy = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : tmpp;
    ColorSpinorField *xp_sloppy   = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : yp;

    ColorSpinorField &r = *rp;
    ColorSpinorField &p = *pp;
    ColorSpinorField &s = *sp;
    ColorSpinorField &u = *up;
    ColorSpinorField &w = *wp;
    ColorSpinorField &q = *qp;
    ColorSpinorField &n = *np;
    ColorSpinorField &m = *mp;
    ColorSpinorField &z = *zp;

    ColorSpinorField &y = *yp;
    ColorSpinorField &l = *lp;

    ColorSpinorField &r_sloppy = *rp_sloppy;
    ColorSpinorField &x_sloppy = *xp_sloppy;

    ColorSpinorField &pPre = *p_pre;
    ColorSpinorField &rPre = *r_pre;
    ColorSpinorField &tmp  = *tmpp;
    ColorSpinorField &tmp_sloppy  = *tmpp_sloppy;

    // Check to see that we're not trying to invert on a zero-field source
    const double b2    = blas::norm2(b);
    y = x;

    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      blas::copy(x, b);
      param.true_res = 0.0;
      return;
    }

    //compute residual
    mat(r, x, tmp); // => r = A*x;
    blas::xpay(b,-1.0, r);
    r_sloppy = r;

    const double eps     = param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon() : ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon() : pow(2.,-13));
    const double sqrteps = param.delta*sqrt(eps);

    double Dcr = 0.0, Dcs = 0.0, Dcw = 0.0, Dcz = 0.0;
    double errr = 0.0, errrprev = 0.0, errs = 0.0, errw = 0.0, errz = 0.0;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    printfQuda("Stopping condition: %le\n", stop);

    double4 *local_reduce = new double3[2];//to keep double3 or double4 registers

    double alpha = 0.0, beta = 0.0, alpha_old = 0.0, gamma_old = 0.0, eta = 0.0;

    double &gamma = local_reduce[0].x, &delta = local_reduce[0].y, &mNorm = local_reduce[0].z;
    double &sigma = local_reduce[1].x, &zeta  = local_reduce[1].y, &tau   = local_reduce[1].z;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    if(K) {
      rPre = r_sloppy;
      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);
      u = pPre;
      blas::zero(m);
    } else {
      u = r_sloppy;
    }

    matSloppy(w, u, tmp_sloppy); // => w = A*u;

    blas::flops = 0;

    double *recvbuff   = new double[6];
    MPI_Request request_handle;
    //
    commGlobalReductionSet(false);

    gamma = blas::reDotProduct(r_sloppy,u);
    delta = blas::reDotProduct(w,u);
    mNorm = blas::norm2(u);

    if(K) {
      rPre = w;
      (*K)(pPre, rPre);
      m = pPre;
    } else {
      m = w;
    }

    commGlobalReductionSet(true);

    //Start async communications:
    MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
    matSloppy(n, m, tmp_sloppy);
    MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
    memcpy(local_reduce, recvbuff, 3*sizeof(double));

    // START zero iteration:
    eta   = delta;
    alpha = gamma / eta;

    z = n;           //  z <- n
  	q = m;           //  q <- m
  	p = u;           //  p <- u
  	s = w;           //  s <- w
  	blas::axpy( alpha, p, x);        //  x <- x + alpha * p
  	blas::axpy(-alpha, q, u);        //  u <- u - alpha * q
  	blas::axpy(-alpha, z, w);        //  w <- w - alpha * z
  	blas::axpy(-alpha, s, r_sloppy); //  r <- r - alpha * s
    n = w;
	  blas::mxpy(r_sloppy, n);          //  n <- w - r

    commGlobalReductionSet(false);

    gamma_old = gamma;
    gamma = blas::reDotProduct(r_sloppy,u);
    delta = blas::reDotProduct(w,u);
    mNorm = blas::norm2(u);
    sigma = blas::norm2(s);
    zeta  = blas::norm2(z);
    tau   = blas::reDotProduct(s,u);

    if(K) {
      rPre = n;
      (*K)(pPre, rPre);
      m = pPre;
      blas::xpy(u, m);
    } else {
      m = w;
    }

    commGlobalReductionSet(true);

    //Start async communications:
    MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
    matSloppy(n, m, tmp_sloppy);
    MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
    memcpy(local_reduce, recvbuff, 6*sizeof(double));

    int j = 1;
    double heavy_quark_res = 0.0;
    bool updateR = false;
    //
    PrintStats( "PipePCG (before main loop)", j, mNorm, b2, heavy_quark_res);

    while(!convergence(mNorm, heavy_quark_res, stop, param.tol_hq) && j < param.maxiter){

      beta = -tau / eta;
  	  eta  = delta - beta*beta*eta;
  	  alpha_old = alpha; alpha = gamma / eta;

      Dcr = (2.0*alpha_old*sqrt(zeta))*eps;
      Dcs = (2.0*beta*sqrt(sigma)+2.0*alpha_old*sqrt(zeta))*eps;
  	  Dcw = (2.0*alpha_old*sqrt(zeta))*eps;
  	  Dcz = (2.0*beta*sqrt(zeta))*eps;

      if (j == 1 || updateR) {
		    printfQuda("(Re-)initialize reliable parameters..");
        errrprev = errr;
        errr = Dcr;
        errs = Dcs;
        errw = Dcw;
        errz = Dcz;
        updateR = false;
      } else {
        errrprev = errr;
        errr = errr + alpha_old*errs + Dcr;
        errs = beta*errs + errw + alpha_old*errz + Dcs;
        errw = errw + alpha_old*errz + Dcw;
        errz = beta*errz + Dcz;
      }

      bool updateR = ((j > 1 && errrprev <= (sqrteps * sqrt(gamma_old)) && errr > (sqrteps * sqrt(gamma))) );

      gamma_old = gamma;

      if (!updateR){

        commGlobalReductionSet(false);//disable global reduction
        pipePCGRRMergedOp(local_reduce,2,y,alpha,p,u,r_sloppy,s,m,beta,q,w,n,z);
        commGlobalReductionSet(true);

      } else {

        x_sloppy = y;
        xpy(x_sloppy,x);
        zero(y);

        mat(r, x, tmp);
        xpay(b, -1.0, r);
        r_sloppy = r;

        if(K) {
          rPre = r_sloppy;

          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          u = pPre;
        } else {
          u = r_sloppy;
        }

        l = u;
        mat(r, l, tmp_sloppy);
        w = r;

        n = w;
    	  blas::mxpy(r_sloppy, n);          //  n <- w - r

        l = p;
        mat(r, l, tmp_sloppy);
        s = r;

        if(K) {
          rPre = s;

          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          q = pPre;
        } else {
          q = s;
        }

        l = q;
        mat(r, l, tmp_sloppy);
        z = r;

        commGlobalReductionSet(false);
        gamma = blas::reDotProduct(r_sloppy,u);
        delta = blas::reDotProduct(w,u);
        tau   = blas::reDotProduct(s,u);
        mNorm = blas::norm2(u);
        sigma = blas::norm2(s);
        zeta  = blas::norm2(z);
        commGlobalReductionSet(true);
      }

      if(K) {
        rPre = n;
        (*K)(pPre, rPre);
        m = pPre;
        blas::xpy(u, m);
      } else {
        m = w;
      }

      commGlobalReductionSet(true);

      //Start async communications:
      MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
      matSloppy(n, m, tmp_sloppy);
      MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
      memcpy(local_reduce, recvbuff, 6*sizeof(double));

      j += 1;

      PrintStats( "PipePCG", j, mNorm, b2, heavy_quark_res);
    }

    delete [] recvbuff;

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matPrecon.flops())*1e-9;
    param.gflops = gflops;
    param.iter += j;

    if (j==param.maxiter)  warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual
    x_sloppy = y;
    xpy(x_sloppy, x);
    mat(r, x, tmp);
    double true_res = blas::xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    delete lp;
    delete yp;

    delete[] local_reduce;

    if (param.precision_sloppy != param.precision) {
      delete rp_sloppy;
      delete xp_sloppy;
      delete tmpp_sloppy;
    }

    return;
  }

#ifdef FULL_PIPELINE
#undef FULL_PIPELINE
#endif

#ifdef NONBLOCK_REDUCE
#undef MPI_CHECK_
#endif

} // namespace quda
