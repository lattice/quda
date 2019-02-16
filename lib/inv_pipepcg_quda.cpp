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

     ColorSpinorField *xp;
     ColorSpinorField *up;
     ColorSpinorField *wp;
     ColorSpinorField *mp;
     ColorSpinorField *pp;
     ColorSpinorField *sp;
     ColorSpinorField *qp;
     ColorSpinorField *zp;

     double *buffer;
     int     n_update;


   public:
     GlobalMPIallreduce(ColorSpinorField *xp, ColorSpinorField *up, ColorSpinorField *wp, ColorSpinorField *mp, ColorSpinorField *pp, ColorSpinorField *sp, ColorSpinorField *qp, ColorSpinorField *zp, double *buffer) : xp(xp), up(up), wp(wp), mp(mp), pp(pp), sp(sp), qp(qp), zp(zp),  buffer(buffer), n_update( xp->Nspin()==4 ? 4 : 2 ) { }

     virtual ~GlobalMPIallreduce() { }

     void apply(const cudaStream_t &stream) {

       static int count = 0;

       if (count == 0 ) {
         //(x, x)
         //(u, u)
         buffer[4] = norm2 ( *xp );
         buffer[5] = norm2 ( *up );
       } else if ( count == 1 ) {
         //(w, w)
         //(m, m)
         buffer[6] = norm2 ( *wp );
         buffer[7] = norm2 ( *mp );
       } else if ( count == 2 ) {
         //(p, p)
         //(s, s)
         buffer[0] = norm2 ( *pp );
         buffer[1] = norm2 ( *sp );
       } else if ( count == 3 ) {
         //(p, p)
         //(s, s)
         buffer[2] = norm2 ( *qp );
         buffer[3] = norm2 ( *zp );
       }

       if (++count == n_update ) count = 0;

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
	r_pre = np;//wp->np
      }

      init = true;
    }

    csParam.setPrecision(param.precision);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    ColorSpinorField *lp = ColorSpinorField::Create(csParam);

    csParam.setPrecision(param.precision_sloppy);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    ColorSpinorField *yp = ColorSpinorField::Create(csParam);

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
    y = x;//copy initial guess

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

//?    const double mu    = K ? 1000.0 : 1000.0;
/*
32*64 : 1000
32*32 : 100 => better
*/
    const double mu    = K ? 100.0 : 1000.0;
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
    printfQuda("Stopping condition: %le\n", stop);

    double4 *local_reduce = new double4[3];//to keep double3 or double4 registers

    double alpha, beta, alpha_old, beta_old, mNorm_old, mNorm_old2;
    double gammajm1 = 1.0, gamma_aux = 0.0;

    double &gamma = local_reduce[0].x, &delta = local_reduce[0].y, &mNorm = local_reduce[0].z;
    double &pi    = local_reduce[1].x, &sigma = local_reduce[1].y, &phi   = local_reduce[1].z, &psi = local_reduce[1].w;
    double &chi   = local_reduce[2].x, &ksi   = local_reduce[2].y, &omega = local_reduce[2].z, &nu  = local_reduce[2].w;

    double pi_old  = 0.0, sigma_old  = 0.0,  phi_old  = 0.0,  psi_old  = 0.0;
    double chi_old = 0.0, omega_old  = 0.0,  ksi_old  = 0.0,  nu_old   = 0.0;
    double pi_old2 = 0.0, sigma_old2 = 0.0,  phi_old2 = 0.0,  psi_old2 = 0.0;
#ifdef FULL_PIPELINE
    double chi_old2= 0.0, omega_old2 = 0.0,  ksi_old2 = 0.0,  nu_old2 = 0.0;
#endif

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

    alpha = 1.0;
    beta  = 0.0;

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
32x32x2.38
tau = 5.0e+6*sqrt(uro);

*/
    const double tau = 5.0e+6*sqrt(uro);

    blas::flops = 0;

    // now create the worker class for updating the gradient vectors
#ifdef FULL_PIPELINE
    printfQuda("Running full pipeline.\n");
    GlobalMPIallreduce global_reduce(yp, up, wp, mp, pp, sp, qp, zp, reinterpret_cast<double*>(&local_reduce[1]));
    dslash::aux_worker = nullptr;
#endif

    double *recvbuff   = new double[12];
    MPI_Request request_handle;

    commGlobalReductionSet(false);
    //
    if(K) {
      rPre = w;
      (*K)(pPre, rPre);
      m = pPre;
    } else {
      m = w;
    }
    //
    gamma = blas::reDotProduct(r_sloppy,u);
    mNorm = blas::norm2(r_sloppy);
    delta = blas::reDotProduct(w,u);
    local_reduce[0].w = 0.0;

    local_reduce[1].x = 0.0;
    local_reduce[1].y = 0.0;
    local_reduce[1].z = 0.0;
    local_reduce[1].w = 0.0;

#ifdef FULL_PIPELINE
    local_reduce[2].x = norm2 ( y );
    local_reduce[2].y = norm2 ( u );
    local_reduce[2].z = norm2 ( w );
    local_reduce[2].w = norm2 ( m );
#else
    local_reduce[2].x = 0.0;
    local_reduce[2].y = 0.0;
    local_reduce[2].z = 0.0;
    local_reduce[2].w = 0.0;
#endif
    commGlobalReductionSet(true);

    //Start async communications:
    MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
    //
    matSloppy(n, m, tmp_sloppy);
    //
    MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
    memcpy(local_reduce, recvbuff, 12*sizeof(double));

    mNorm_old = mNorm;
    gamma_aux = gamma;//to set beta to zero

    int j = 0;

    double heavy_quark_res = 0.0;
    //
    PrintStats( "PipePCG (before main loop)", j, mNorm, b2, heavy_quark_res);

    int updateR = 0;
    int j_rupd_idx = 0;

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

#ifdef FULL_PIPELINE
      chi_old2   = chi_old;
      omega_old2 = omega_old;
      ksi_old2   = ksi_old;
      nu_old2    = nu_old;
#endif

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

#ifndef FULL_PIPELINE
      commGlobalReductionSet(false);//disable global reduction
      MergedLocalReducer(local_reduce,3,y,alpha,p,u,r_sloppy,s,m,beta,q,w,n,z);//K ? pipePCGRRFletcherReevesMergedOp : pipePCGRRMergedOp
      commGlobalReductionSet(true);
#else
      commGlobalReductionSet(false);//disable global reduction
      MergedLocalReducer(local_reduce,1,y,alpha,p,u,r_sloppy,s,m,beta,q,w,n,z);
      commGlobalReductionSet(true);
#endif
//finish recursion updates

      ajm1 = fabs(alpha_old);
      bjm1 = fabs(beta_old);

      constexpr double fma_factor = 1.0;//1.0 for FMA instructions and 2.0 for separate fp operations leading to classical relations

      if( j_rupd_idx > 0 ) {
        PrintStats( "PipePCG", j, mNorm_old, b2, heavy_quark_res);
#ifndef FULL_PIPELINE
        ef = Anorm*(chi_old+2*ajm1*pi_old ) + sqrt(mNorm_old2) + fma_factor*ajm1*sigma_old;
//!ef = Anorm*(chi_old+2*ajm1*pi_old ) + mNorm_old2 + fma_factor*ajm1*sigma_old;
        eh = Anorm*(ksi_old+2*ajm1*phi_old) + omega_old + fma_factor*ajm1*psi_old;

        if(j_rupd_idx > 1) {
          eg = Anorm*(ksi_old+2*bjm1*pi_old2) + omega_old + fma_factor*bjm1*sigma_old2;
          ek = Anorm*((msqrn+2)*nu_old+2*bjm1*phi_old2) + fma_factor*bjm1*psi_old2;
        }
#else
        ef = Anorm*(chi_old2+2*ajm1*pi_old ) + sqrt(mNorm_old2) + fma_factor*ajm1*sigma_old;
        eh = Anorm*(ksi_old2+2*ajm1*phi_old) + omega_old2 + fma_factor*ajm1*psi_old;

        if(j_rupd_idx > 1) {
          eg = Anorm*(ksi_old2+2*bjm1*pi_old2) + omega_old2 + fma_factor*bjm1*sigma_old2;
          ek = Anorm*((msqrn+2)*nu_old2+2*bjm1*phi_old2) + fma_factor*bjm1*psi_old2;
        }
#endif
        if( j_rupd_idx == 1 or updateR ) {
          printfQuda("Do %s..\n", j == 1 ? "initialization" : "replace");
#ifndef FULL_PIPELINE
          if(chi_old != 0.0) warningQuda("Something is wrong...?\n");
          f = uro * sqrt( (msqrnA + Anorm) * chi_old + bnorm ) + uro * sqrt( ajm1*msqrnA*pi_old  ) + sqrt(ef)*uro;
          h = uro * sqrt( msqrnA * ksi_old ) + uro * sqrt( ajm1*msqrnA*phi_old ) + sqrt(eh)*uro;
//f = uro * ( (msqrnA + Anorm) * chi_old + bnorm ) + uro * ( ajm1*msqrnA*pi_old  ) + (ef)*uro;
//h = uro * ( msqrnA * ksi_old ) + uro * ( ajm1*msqrnA*phi_old ) + (eh)*uro;
#else
          f = uro * sqrt( (msqrnA + Anorm) * chi_old2 + bnorm ) + uro * sqrt( ajm1*msqrnA*pi_old  ) + sqrt(ef)*uro;
          h = uro * sqrt( msqrnA * ksi_old2 ) + uro * sqrt( ajm1*msqrnA*phi_old ) + sqrt(eh)*uro;
#endif
          g = uro * sqrt( msqrnA * pi_old );
          k = uro * sqrt( msqrnA * phi_old);
//!g = uro * msqrnA * pi_old;
//!k = uro * msqrnA * phi_old;


          //updateR = 0;

        } else {
          f = f + ajm1*bjm1*g + ajm1*h + sqrt(ef)*uro + ajm1*sqrt(eg)*uro;
          g = bjm1*g + h + sqrt(eg)*uro;
          h = h + ajm1*bjm1*k + sqrt(eh)*uro + ajm1*sqrt(ek)*uro;
          k = bjm1*k + sqrt(ek)*uro;
        }

      }

//NB: 1.0 default
      const double Kscale_factor = 1.0;

      updateR = ( j_rupd_idx > 2 and f_old <= tau*sqrt(mNorm_old2) and f > tau*sqrt(mNorm_old) ) ? 1 : 0;
      f_old   = f;

      if( updateR ) { //trigger reliable updates:

        printfQuda("Start relibale update.. (f_old = %le , tau*mNorm_old = %le , f = %le, tau*mNorm = %le )\n",f_old, tau*sqrt(mNorm_old2) , f , tau*sqrt(mNorm_old));
#ifdef FULL_PIPELINE
        commGlobalReductionSet(false);
        //
        local_reduce[1].x = norm2 ( p );
        local_reduce[1].y = norm2 ( s );
        local_reduce[1].z = norm2 ( q );
        local_reduce[1].w = norm2 ( z );
        //
        commGlobalReductionSet(true);
#endif

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

        matSloppy(w, u, tmp_sloppy);
        matSloppy(s, p, tmp_sloppy);

#ifdef FULL_PIPELINE
        commGlobalReductionSet(false);
        //
        local_reduce[2].x = 0.0;
        local_reduce[2].y = norm2 ( u );
        local_reduce[2].z = norm2 ( w );
        local_reduce[2].w = norm2 ( m );//set m to zero
        //
        commGlobalReductionSet(true);
#endif

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


        commGlobalReductionSet(false);
        gamma = blas::reDotProduct(r_sloppy,u);
        delta = blas::reDotProduct(w,u);
        mNorm = blas::norm2(r_sloppy);
        commGlobalReductionSet(true);
      }

#ifdef FULL_PIPELINE
      {
        int recvbuff_size = !updateR ? 4 : 12;

        MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, recvbuff_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));

        if(K) {
          n = r_sloppy;
          xpay(w, -Kscale_factor, n);
          rPre = n;

          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          m = pPre;
          xpy(u, m);
        } else {
          m = w;
        }

        //
        dslash::aux_worker = !updateR ? &global_reduce : nullptr;
        matSloppy(n, m, tmp_sloppy);
        dslash::aux_worker = nullptr;

        MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
        memcpy(local_reduce, recvbuff, recvbuff_size*sizeof(double));
      }
#else //regular path
      {
#ifdef NONBLOCK_REDUCE
        MPI_CHECK_(MPI_Iallreduce((double*)local_reduce, recvbuff, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
#else
        reduceDoubleArray((double*)local_reduce, 12);
#endif
        if(K) {
          n = r_sloppy;
          xpay(w, -Kscale_factor, n);
          rPre = n;

          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          m = pPre;
          xpy(u, m);
        } else {
          m = w;
        }
        //
        matSloppy(n, m, tmp_sloppy);

#ifdef NONBLOCK_REDUCE//sync point
        MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
        memcpy(local_reduce, recvbuff, 12*sizeof(double));
#endif
      }
#endif //end of USE_WORKER
      gamma_aux = local_reduce[0].w;//gamma, delta and mNorm are refs to local_reduce.[x|y|z]

      j += 1;
      j_rupd_idx += 1;
      //PrintStats( "PipePCG", j, mNorm, b2, heavy_quark_res);
    }

#ifdef NONBLOCK_REDUCE
    delete [] recvbuff;
#endif

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
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

#ifdef FULL_PIPELINE
#undef FULL_PIPELINE
#endif

#ifdef NONBLOCK_REDUCE
#undef MPI_CHECK_
#endif

} // namespace quda
