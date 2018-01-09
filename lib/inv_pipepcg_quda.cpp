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

//#define USE_WORKER

#ifndef USE_WORKER

#include <mpi.h>

#define NONBLOCK_REDUCE
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

#endif

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

      if (K && (param.precision_precondition != param.precision_sloppy)) {
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

  class GlobalMPIallreduce : public Worker {
     double *buffer;
     int     size;
     int     vals_to_update;
     int     stages; 
    public:
    GlobalMPIallreduce(double *buffer, int size, int stages=4) : buffer(buffer), size(size), vals_to_update(size/4), stages(stages) { }

    virtual ~GlobalMPIallreduce() { }

    void apply(const cudaStream_t &stream) {  //this has nothing to do with GPU streaming but should work as well, we may need non-blocked MPI allreduce, see cooments below
      static int count = 0;
      if (count < (stages -1) ) {
        reduceDoubleArray( &buffer[count*vals_to_update] , vals_to_update );
        count += 1;
      } else {
        reduceDoubleArray( &buffer[count*vals_to_update] , (size-vals_to_update) );
        count = 0; 
      } 

      return;
    }
  };

  // this is the Worker pointer 
  namespace dslash {
    extern Worker* aux_worker;
  }

  void PipePCG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    profile.TPSTART(QUDA_PROFILE_INIT);

    ColorSpinorParam csParam(b);

    auto  MergedLocalReducer = K ? pipePCGRRFletcherReevesMergedOp : pipePCGRRMergedOp;

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
      if (K && (param.precision_precondition != param.precision_sloppy)) {
        printfQuda("Allocated resources for the preconditioner.\n");
        csParam.setPrecision(param.precision_precondition);
	p_pre = ColorSpinorField::Create(csParam);
	r_pre = ColorSpinorField::Create(csParam);
      } else {
	p_pre = mp;
	r_pre = wp;
      }

      init = true;
    }

    csParam.setPrecision(param.precision_sloppy);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    ColorSpinorField *yp = ColorSpinorField::Create(csParam);
    ColorSpinorField *lp = ColorSpinorField::Create(csParam);

    ColorSpinorField *rp_sloppy   = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : rp;
    ColorSpinorField *tmpp_sloppy = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : tmpp;
    csParam.setPrecision(param.precision);
    ColorSpinorField *xp_sloppy = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : yp;

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
    const double b2 = blas::norm2(b);
    const double bnorm = sqrt(b2);

    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      blas::copy(x, b);
      param.true_res = 0.0;
    }

    //Estimate A-norm:
//TEST:
    csParam.setPrecision(param.precision);
    ColorSpinorField *Ax_cuda = ColorSpinorField::Create(csParam);
    

    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam.location   = QUDA_CPU_FIELD_LOCATION;
    ColorSpinorField *Ax = ColorSpinorField::Create(csParam);
    Ax->Source(QUDA_RANDOM_SOURCE);
    *Ax_cuda = *Ax;

    const double Axnorm2 = sqrt(blas::norm2(*Ax_cuda));
    printfQuda("(Debug info) x-norm estimate: %1.15le\n", Axnorm2);
    blas::ax(1.0 / Axnorm2, *Ax_cuda);

    mat(r, *Ax_cuda, tmp);
    delete Ax;
    delete Ax_cuda;

    const double mu    = K ? 1000.0 : 100.0; 
    const double blen  = b.RealLength();
    const double Anorm = sqrt(blas::norm2(r));
    const double msqrn = mu*sqrt(blen); 
    const double msqrnA= msqrn*Anorm;
    printfQuda("(Debug info) A-norm estimate: %1.15le\n", Anorm);

    mat(r, x, tmp); // => r = A*x;
    blas::xpay(b,-1.0, r);

    r_sloppy = r;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    double4 *local_reduce = new double4[3];//to keep double3 or double4 registers

    double alpha, beta, alpha_old, beta_old, mNorm_old, mNorm_old2;
    double gammajm1 = 1.0, gamma_aux = 0.0;

    double &gamma = local_reduce[0].x, &delta = local_reduce[0].y, &mNorm = local_reduce[0].z;
    double &pi    = local_reduce[1].x, &sigma = local_reduce[1].y, &phi   = local_reduce[1].z, &psi = local_reduce[1].w;
    double &chi   = local_reduce[2].x, &ksi   = local_reduce[2].y, &omega = local_reduce[2].z, &nu  = local_reduce[2].w;

    double pi_old  = 0.0, sigma_old  = 0.0,  phi_old  = 0.0,  psi_old  = 0.0;
    double chi_old = 0.0, omega_old  = 0.0,  ksi_old  = 0.0,  nu_old   = 0.0;
    double pi_old2 = 0.0, sigma_old2 = 0.0,  phi_old2 = 0.0,  psi_old2 = 0.0;

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
      printfQuda("\nNo preconditioning...\n");
      u = r_sloppy;
    }

    matSloppy(w, u, tmp_sloppy); // => w = A*u;

    //Remark : no overlap here. 
    //double2 gammamNorm = reDotProductNormA(r_sloppy, u);
    //gamma = gammamNorm.y; 
    //mNorm = gammamNorm.x;

    gamma = blas::reDotProduct(r_sloppy,u); 
    mNorm = blas::norm2(r_sloppy);
    delta = blas::reDotProduct(w,u);

    mNorm_old = mNorm;
    gamma_aux = gamma;

    alpha = 1.0;
    beta  = 0.0;

    //const int maxResIncrease      = param.max_res_increase; // check if we reached the limit of our tolerance
    //const int maxResIncreaseTotal = param.max_res_increase_total;

    double ef = 0.0, eh = 0.0, eg = 0.0, ek = 0.0;
    double f = 0.0, g = 0.0, h = 0.0, k = 0.0, f_old = 0.0;

    const double uro   = param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon()/2. : ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon()/2. : pow(2.,-13));
/*
tau = sqrt(uro) will destroy convergence!
tau = 100000 * sqrt(uro) works! (212 iters)
tau = 10000 * sqrt(uro) works! (212 iters)
tau = 1000 * sqrt(uro) works! (212 iters)
tau = 100 * sqrt(uro) works! (214 iters)
tau = 10 * sqrt(uro) fails!
*/
    const double tau = 100.0*sqrt(uro);

    //int resIncrease = 0;
    //int resIncreaseTotal = 0;

    blas::flops = 0;

    // now create the worker class for updating the gradient vectors
#ifdef USE_WORKER
    GlobalMPIallreduce global_reduce((double*)&local_reduce, 12);
    dslash::aux_worker = nullptr;//&global_reduce;
#else
    //MsgHandle* allreduceHandle = comm_handle();
    MPI_Request request_handle;
    double *recvbuff           = new double[12];//mpi buffer for async global reduction
#endif

    if(K) {
      rPre = w;

      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);

      m = pPre;
    } else {
      m = w;
    }
    //
    matSloppy(n, m, tmp_sloppy);

    int j = 0;

    double heavy_quark_res = 0.0;
    //
    PrintStats( "PipePCG (before main loop)", j, mNorm, b2, heavy_quark_res);

    int updateR = 0;

    while(!convergence(mNorm, heavy_quark_res, stop, param.tol_hq) && j < param.maxiter){

      double ajm1, bjm1;

      beta_old  = beta;
      beta  = (gamma - gamma_aux) / gammajm1;
      double scal = (delta / gamma - beta / alpha);
      alpha_old = alpha;
      alpha = 1.0 / scal; 
      gammajm1 = gamma;

//start a cascade of updates:
      mNorm_old2 = mNorm_old;//sic! 
      mNorm_old  = mNorm;

      chi_old   = sqrt(chi);
      omega_old = sqrt(omega);
      ksi_old   = sqrt(ksi);
      nu_old    = sqrt(nu);

      pi_old2    = pi_old;
      sigma_old2 = sigma_old;
      phi_old2   = phi_old;
      psi_old2   = psi_old; 

      pi_old    = sqrt(pi);
      sigma_old = sqrt(sigma);
      phi_old   = sqrt(phi);
      psi_old   = sqrt(psi);

      commGlobalReductionSet(false);//disable global reduction
      MergedLocalReducer(local_reduce,y,alpha,p,u,r_sloppy,s,m,beta,q,w,n,z);
      commGlobalReductionSet(true);
//finish updates

      ajm1 = fabs(alpha_old);
      bjm1 = fabs(beta_old);

      if(j > 0) {
        PrintStats( "PipePCG", j, mNorm_old, b2, heavy_quark_res);

        ef = Anorm*(chi+2*ajm1*pi_old ) + sqrt(mNorm_old2) + 2*ajm1*sigma_old;
        eh = Anorm*(ksi+2*ajm1*phi_old) + omega_old + 2*ajm1*psi_old;

        if(j > 1) {
          eg = Anorm*(ksi+2*bjm1*pi_old2) + omega_old + 2*bjm1*sigma_old2;
          ek = Anorm*((msqrn+2)*nu_old+2*bjm1*phi_old2) + 2*bjm1*psi_old2;
        }

        if(j == 1 or updateR) {

          printfQuda("Do %s..\n", j == 1 ? "initialization" : "replace");

          f = uro * sqrt( (msqrnA + Anorm) * chi_old + bnorm ) + uro * sqrt( ajm1*msqrnA*pi_old  ) + sqrt(ef)*uro;
          g = uro * sqrt( msqrnA * pi_old );
          h = uro * sqrt( msqrnA * ksi_old ) + uro * sqrt( ajm1*msqrnA*phi_old ) + sqrt(eh)*uro;
          k = uro * sqrt( msqrnA * phi_old);

        } else { 
          f = f + ajm1*bjm1*g + ajm1*h + sqrt(ef)*uro + ajm1*sqrt(eg)*uro; 
          g = bjm1*g + h + sqrt(eg)*uro;
          h = h + ajm1*bjm1*k + sqrt(eh)*uro + ajm1*sqrt(ek)*uro;
          k = bjm1*k + sqrt(ek)*uro;
        }

      }

      updateR = ( j > 1 and f_old <= tau*sqrt(mNorm_old2) and f > tau*sqrt(mNorm_old) ) ? 1 : 0;
      f_old   = f;

      if( updateR ) { //trigger reliable updates:

        printfQuda("Start relibale update.. (f_old = %le , tau*mNorm_old = %le , f = %le, tau*mNorm = %le )\n",f_old, tau*sqrt(mNorm_old2) , f , tau*sqrt(mNorm_old));

        x_sloppy = y;
        xpy(x_sloppy,x);
        zero(y);

        mat(r, x, tmp);
        xpay(b, -1.0, r);
//Debug
        mNorm_old2 = norm2(r_sloppy);

        r_sloppy = r;
        mNorm = norm2(r_sloppy);

        printfQuda("Old residual: %1.15le vs new residual %1.15le\n", mNorm_old2, mNorm);

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

        matSloppy(w, u, tmp_sloppy);
        matSloppy(s, p, tmp_sloppy);

        zero(l);

        if(K) {
          rPre = s;
          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);
          q = pPre;
        } else {
          q = s;
        }

        matSloppy(z, q, tmp_sloppy);

        mNorm_old2 = mNorm_old;
        mNorm_old  = mNorm;
        gamma = blas::reDotProduct(r_sloppy,u); 
        delta = blas::reDotProduct(w,u);
        mNorm = blas::norm2(r_sloppy);
      }

#ifdef USE_WORKER
      {
//global_reduce.apply(0);
//Warning! ordering is critical here: fisrt preconditioner and then matvec, since worker uses blocking MPI allreduce
//in this approach all reduce is overlapped with matvec only.
//more robust way just to call non-blocking allreduce and then synchronize 
        if(K) {
          rPre = w;

          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          m = pPre;
        } else {
          m = w;
        }        
          //
        dslash::aux_worker = &global_reduce;
        matSloppy(n, m, tmp_sloppy);
        dslash::aux_worker = nullptr;

      }
#else
      {
#ifdef NONBLOCK_REDUCE
        MPI_CHECK_(MPI_Iallreduce((double*)&local_reduce, recvbuff, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
#else
        reduceDoubleArray((double*)&local_reduce, 12);
#endif
        if(K) {
          rPre = w;
         
          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          m = pPre;
        } else {
          m = w;
        }
        //
        matSloppy(n, m, tmp_sloppy);
#ifdef NONBLOCK_REDUCE//sync point
        MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
        memcpy(&local_reduce, recvbuff, 12*sizeof(double));
#endif
      }
#endif //end of USE_WORKER
      gamma_aux = local_reduce[0].w;//gamma, delta and mNorm are refs to local_reduce.[x|y|z]

      j += 1;
      //PrintStats( "PipePCG", j, mNorm, b2, heavy_quark_res);
    }

#ifdef NONBLOCK_REDUCE
    //host_free(allreduceHandle);
    delete [] recvbuff;
#endif

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    //param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matPrecon.flops())*1e-9;
    param.gflops = gflops;
    param.iter += j;

    if (j==param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

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

} // namespace quda
