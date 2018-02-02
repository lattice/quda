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
* Experimental Pipe2PCG algorithm
* Sources: 
* P. Eller and W. Gropp "Scalable non-blocking Preconditioned Conjugate Gradient Methods", SC2016  
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

    if(outer.inv_type == QUDA_PIPE2PCG_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  }


  Pipe2PCG::Pipe2PCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
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
  
  Pipe2PCG::~Pipe2PCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {

      if (K) {

        if (param.precision_precondition != param.precision_sloppy) {
          delete q_pre;
          delete c_pre;
          delete d_pre;
          delete g_pre;
        }

        delete zp;
        delete cp;
        delete pp;
        delete gp;

        delete K;
      }

      delete tmpp;
      delete rp;
      //delete yp;

      delete dp;
      delete qp;
      delete wp;
      delete hp;

      delete xp_sloppy;
      delete rp_sloppy;
    }

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

  namespace dslash {
    extern Worker* aux_worker;
  }

  void Pipe2PCG::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    profile.TPSTART(QUDA_PROFILE_INIT);

    ColorSpinorParam csParam(b);

    auto  MergedLocalReducer = K ? pipe2PCGMergedOp : pipe2CGMergedOp;

    if (!init) {
      // high precision fields:
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      tmpp = ColorSpinorField::Create(csParam); //temporary for full mat-vec

      csParam.setPrecision(param.precision_sloppy);

      hp   = ColorSpinorField::Create(csParam);

      csParam.is_composite  = true;
      csParam.composite_dim = 2;

      xp_sloppy = ColorSpinorField::Create(csParam);
      rp_sloppy = ColorSpinorField::Create(csParam);
      wp = ColorSpinorField::Create(csParam);
      qp = ColorSpinorField::Create(csParam);
      dp = ColorSpinorField::Create(csParam);

      zp = K ? ColorSpinorField::Create(csParam) : rp_sloppy;
      pp = K ? ColorSpinorField::Create(csParam) : wp;
      cp = K ? ColorSpinorField::Create(csParam) : qp;

      csParam.is_composite  = false;
      csParam.composite_dim = 1;

      gp = K ? ColorSpinorField::Create(csParam) : &dp->Component(0);


      // these low precision fields are used by the inner solver
      if ( K ) {
        printfQuda("Allocated resources for the preconditioner.\n");
        csParam.setPrecision(param.precision_precondition);
	q_pre = (param.precision_precondition != param.precision_sloppy) ? ColorSpinorField::Create(csParam) : &qp->Component(0);
	c_pre = (param.precision_precondition != param.precision_sloppy) ? ColorSpinorField::Create(csParam) : &cp->Component(0);

	d_pre = (param.precision_precondition != param.precision_sloppy) ? ColorSpinorField::Create(csParam) : &dp->Component(0);
	g_pre = (param.precision_precondition != param.precision_sloppy) ? ColorSpinorField::Create(csParam) : gp;
      } else { //no preconditioner
        printfQuda("\nNo preconditioning...\n");
        q_pre = nullptr;
        c_pre = &cp->Component(0);//qp
        d_pre = nullptr;
        g_pre = gp;//dp
      }

      init = true;
    }

    csParam.setPrecision(param.precision_sloppy);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.is_composite = false;

    ColorSpinorField *tmpp_sloppy = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : tmpp;

    ColorSpinorField &qPre = *q_pre;
    ColorSpinorField &cPre = *c_pre;

    ColorSpinorField &dPre = *d_pre;
    ColorSpinorField &gPre = *g_pre;

    ColorSpinorField &tmp  = *tmpp;
    ColorSpinorField &tmp_sloppy  = *tmpp_sloppy;

    ColorSpinorField &x1_sloppy = xp_sloppy->Component(0);
    ColorSpinorField &x2_sloppy = xp_sloppy->Component(1);
    ColorSpinorField &r1_sloppy = rp_sloppy->Component(0);
    ColorSpinorField &r2_sloppy = rp_sloppy->Component(1);

    ColorSpinorField &r = *rp;
    ColorSpinorField &z1 = zp->Component(0);
    ColorSpinorField &w1 = wp->Component(0);
    ColorSpinorField &q1 = qp->Component(0);
    ColorSpinorField &p1 = pp->Component(0);
    ColorSpinorField &c1 = cp->Component(0);
    ColorSpinorField &g = *gp;
    ColorSpinorField &d1 = dp->Component(0);
    ColorSpinorField &h = *hp;

    ColorSpinorField &z2 = zp->Component(1);
    ColorSpinorField &w2 = wp->Component(1);
    ColorSpinorField &q2 = qp->Component(1);
    ColorSpinorField &p2 = pp->Component(1);
    ColorSpinorField &c2 = cp->Component(1);
    ColorSpinorField &d2 = dp->Component(1);

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

    const double mu    = K ? 1000.0 : 10000.0; 
    const double blen  = b.RealLength();
    const double Anorm = sqrt(blas::norm2(r));
    const double msqrn = mu*sqrt(blen); 
    const double msqrnA= msqrn*Anorm;
    printfQuda("(Debug info) A-norm estimate: %1.15le\n", Anorm);

    mat(r, x, tmp); // => r = A*x;
    blas::xpay(b,-1.0, r);

    r1_sloppy = r;
    x1_sloppy = x;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    // now create the worker class for updating the gradient vectors
#ifdef USE_WORKER
    GlobalMPIallreduce global_reduce((double*)&local_reduce, 10);
    dslash::aux_worker = nullptr;//&global_reduce;
#else
    //MsgHandle* allreduceHandle = comm_handle();
    MPI_Request request_handle;
    double *recvbuff           = new double[10];//mpi buffer for async global reduction
#endif

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    double4 *local_reduce = new double4[3];//to keep double3 or double4 registers

    double betaj, betajp1, gammaj, gammajp1, deltaj, deltajp1, rhoj, rhojp1, mNorm;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    if(K) {
      qPre = r1_sloppy;
      commGlobalReductionSet(false);
      (*K)(cPre, qPre);
      commGlobalReductionSet(true);
      z1 = cPre;
    } 

    matSloppy(w1, z1, tmp_sloppy); // => w = A*u;

    deltaj   = blas::reDotProduct(z1,w1); 
    betaj    = blas::reDotProduct(z1,r1_sloppy); 

    gammaj  = betaj / deltaj;

    rhoj    = 1.0;

    const double uro   = param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon()/2. : ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon()/2. : pow(2.,-13));
    const double tau = 100.0*sqrt(uro);

    blas::flops = 0;

    if(K) {
      dPre = w1;
      commGlobalReductionSet(false);
      (*K)(gPre, dPre);
      commGlobalReductionSet(true);
      p1 = gPre;
    } 
    //
    matSloppy(q1, p1, tmp_sloppy);

    x2_sloppy = x1_sloppy;
    axpy(+gammaj, z1, x2_sloppy);
    r2_sloppy = r1_sloppy;
    axpy(-gammaj, w1, r2_sloppy);
    w2 = w1;
    axpy(-gammaj, q1, w2);

    deltajp1   = blas::reDotProduct(z2,w2); 
    betajp1    = blas::reDotProduct(z2,r2_sloppy); 
    mNorm      = blas::norm2(z2);

    gammajp1  = betajp1 / deltajp1;

    rhojp1    =  1.0 / (1 - (gammajp1*betajp1) / (gammaj*betaj));

    if(K) {
      qPre = q1;
      commGlobalReductionSet(false);
      (*K)(cPre, qPre);
      commGlobalReductionSet(true);
      c1 = cPre;
    } 
    //
    matSloppy(d1, c1, tmp_sloppy);

    if(K) {
      dPre = d1;

      commGlobalReductionSet(false);
      (*K)(gPre, dPre);
      commGlobalReductionSet(true);

      g = gPre;
    }      
    //
    matSloppy(h, g, tmp_sloppy);

    q2 = q1;
    axpy(-gammaj, d1, q2);
    d2 = d1;
    axpy(-gammaj,  h, d2);
   
    if(K) {
      z2 = z1;
      axpy(-gammaj, p1, z2);
      p2 = p1;
      axpy(-gammaj, c1, p2);
      c2 = c1;
      axpy(-gammaj,  g, c2);
    }

    int j = 1;

    PrintStats( "Pipe2PCG (before main loop)", j-1, mNorm, b2, 0.0);

    int updateR = 0;

    double phijp1[3] = {rhojp1, (rhojp1*gammajp1), (1-rhojp1)};
    double phij  [3] = {0.0, 0.0, 1.0};

    double &lambda0 = local_reduce[0].x, &lambda1 = local_reduce[0].y, &lambda2 = local_reduce[0].w, &lambda3 = local_reduce[0].z;
    double &lambda4 = local_reduce[1].x, &lambda5 = local_reduce[1].y, &lambda6 = local_reduce[1].w, &lambda7 = local_reduce[1].z;
    double &lambda8 = local_reduce[2].x, &lambda9 = local_reduce[2].y, &lambda10= local_reduce[2].w, &lambda11= local_reduce[2].z;

    while(!convergence(mNorm, 0.0, stop, param.tol_hq) && j < param.maxiter){

      commGlobalReductionSet(false);//disable global reduction
      MergedLocalReducer( local_reduce, phij[0], phij[1], phij[2], phijp1[0], phijp1[1], phijp1[2], x1_sloppy, r1_sloppy, w1, q1, d1, h, z1, p1, c1, g, x2_sloppy, r2_sloppy, w2, q2, d2, h, z2, p2, c2, g );
      commGlobalReductionSet(true);
#ifdef USE_WORKER
      {
        if(K) {
          qPre = q1;

          commGlobalReductionSet(false);
          (*K)(cPre, qPre);
          commGlobalReductionSet(true);

          c1 = cPre;
        }      
        //
        dslash::aux_worker = &global_reduce;
        matSloppy(d1, c1, tmp_sloppy);
        dslash::aux_worker = nullptr;

        if(K) {
          dPre = d1;

          commGlobalReductionSet(false);
          (*K)(gPre, dPre);
          commGlobalReductionSet(true);

          g = gPre;
        }      
        //
        dslash::aux_worker = &global_reduce;
        matSloppy(h, g, tmp_sloppy);
        dslash::aux_worker = nullptr;

      }
#else
      {
#ifdef NONBLOCK_REDUCE
        MPI_CHECK_(MPI_Iallreduce((double*)&local_reduce, recvbuff, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
#else
        reduceDoubleArray((double*)&local_reduce, 10);
#endif
        if(K) {
          qPre = q1;

          commGlobalReductionSet(false);
          (*K)(cPre, qPre);
          commGlobalReductionSet(true);

          c1 = cPre;
        }      
        //
        matSloppy(d1, c1, tmp_sloppy);

        if(K) {
          dPre = d1;

          commGlobalReductionSet(false);
          (*K)(gPre, dPre);
          commGlobalReductionSet(true);

          g = gPre;
        }      
        //
        matSloppy(h, g, tmp_sloppy);

#ifdef NONBLOCK_REDUCE//sync point
        MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
        memcpy(&local_reduce, recvbuff, 10*sizeof(double));
#endif
      }
#endif //end of USE_WORKER

      deltaj = lambda0;
      betaj  = lambda6;
      gammaj = betaj / deltaj;

      rhoj   = 1.0 / (1.0 - (gammaj*betaj) / (gammajp1*betajp1*rhojp1));

      phij[0]    = rhoj;
      phij[1]    = rhoj*gammaj;
      phij[2]    = (1 - rhoj);

      deltajp1 = 
		phij[0]*phij[0]*lambda0-
	      2*phij[0]*phij[1]*lambda1+
	      2*phij[0]*phij[2]*lambda2+
	        phij[1]*phij[1]*lambda3-
	      2*phij[1]*phij[2]*lambda4+
		phij[2]*phij[2]*lambda5;

      betajp1  = 
		phij[0]*phij[0]*lambda6-
	      2*phij[0]*phij[1]*lambda0+
	      2*phij[0]*phij[2]*lambda7+
		phij[1]*phij[1]*lambda1-
	      2*phij[1]*phij[2]*lambda2+
		phij[2]*phij[2]*lambda8;

      gammajp1 = betajp1 / deltajp1;
      rhojp1   = 1.0 / (1.0 - (gammajp1*betajp1) / (gammaj*betaj*rhoj));

      phijp1[0]    = rhojp1;
      phijp1[1]    = rhojp1*gammajp1;
      phijp1[2]    = (1 - rhojp1);

      mNorm = lambda6;

      PrintStats( "Pipe2PCG:", j, mNorm, b2, 0.0);

      j += 2;
    }

    delete [] local_reduce;
#ifdef NONBLOCK_REDUCE
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
    xpy(x1_sloppy, x);
    mat(r, x, tmp);
    double true_res = blas::xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    if (param.precision_sloppy != param.precision) {
      delete tmpp_sloppy;
    } 

    return;
  }

#ifdef USE_WORKER
#undef USE_WORKER
#endif

#ifdef NONBLOCK_REDUCE
#undef MPI_CHECK_
#endif 

} // namespace quda
