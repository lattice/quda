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

#include <face_quda.h>
#include <iostream>

/***
* Experimental PipePCG algorithm
* Source P. Ghysels and W. Vanroose "Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm" 
***/

#define USE_WORKER
//#define PIPECG_DEBUG 

#ifndef USE_WORKER
#include <comm_quda.h>
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

    if(outer.inv_type == QUDA_PCG_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  }


  PreconCG::PreconCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
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
  
  PreconCG::~PreconCG(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {

      if (param.precision_precondition != param.precision_sloppy) {
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
    public:
    GlobalMPIallreduce(double *buffer, int size) : buffer(buffer), size(size) { }

    virtual ~GlobalMPIallreduce() { }

    void apply(const cudaStream_t &stream) {  //this has nothing to do with GPU streaming but should work as well, we may need non-blocked MPI allreduce, see cooments below
      reduceDoubleArray((double*)&buffer, size);  
    }
  };

  // this is the Worker pointer 
  namespace dslash {
    extern Worker* aux_worker;
  }

  void PreconCG::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    profile.TPSTART(QUDA_PROFILE_INIT);

    ColorSpinorParam csParam(b);
    if (!init) {
      // high precision fields:
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);

      csParam.create = QUDA_ZERO_FIELD_CREATE;
      pp = ColorSpinorField::Create(csParam);
      zp = ColorSpinorField::Create(csParam);
      wp = ColorSpinorField::Create(csParam);
      sp = ColorSpinorField::Create(csParam);
      qp = ColorSpinorField::Create(csParam);
      np = ColorSpinorField::Create(csParam);
      mp = ColorSpinorField::Create(csParam);
      up = ColorSpinorField::Create(csParam);

      tmpp = ColorSpinorField::Create(csParam); //temporary for sloppy mat-vec

      // these low precision fields are used by the inner solver
      if (param.precision_precondition != param.precision) {
        csParam.setPrecision(param.precision_precondition);
	p_pre = ColorSpinorField::Create(csParam);
	r_pre = ColorSpinorField::Create(csParam);
      } else {
	p_pre = mp;
	r_pre = wp;
      }

      init = true;
    }

    ColorSpinorField &r = *rp;
    ColorSpinorField &p = *pp;
    ColorSpinorField &s = *sp;
    ColorSpinorField &u = *up;
    ColorSpinorField &w = *wp;
    ColorSpinorField &q = *qp;
    ColorSpinorField &n = *np;
    ColorSpinorField &m = *mp;
    ColorSpinorField &z = *zp;

    ColorSpinorField &pPre = *p_pre;
    ColorSpinorField &rPre = *r_pre;
    ColorSpinorField &tmp  = *tmpp;

    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = blas::norm2(b);
    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      blas::copy(x, b);
      param.true_res = 0.0;
    }

    mat(r, x, tmp); // => r = A*x;
    blas::xpay(b,-1.0, r);

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    const bool precMatch           = (param.precision_precondition != param.precision) ? false : true;

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    double alpha, beta;
    double gamma, gammajm1;
    double delta;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    if(K) {
      if( !precMatch )  rPre = r;
      else w = r;
 
      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);

      if( !precMatch ) u = pPre;
      else {
       u = m; 
       blas::zero(m);
      }
    } else {
      printfQuda("\nNo preconditioning...\n");
      u = r;
    }

    mat(w, u, tmp); // => w = A*u;

    //Remark : no overlap here. 
    gamma = blas::reDotProduct(r,u); 
    delta = blas::reDotProduct(w,u);
    double mNorm = blas::norm2(u);

    alpha = gamma / delta;
    beta  = 0.0;

    blas::flops = 0;

    double3 buffer;

    // now create the worker class for updating the gradient vectors
#ifdef USE_WORKER
    GlobalMPIallreduce global_reduce((double*)&buffer, 3);
    dslash::aux_worker = &global_reduce;
#else
    //MsgHandle* allreduceHandle = comm_handle();
#endif

    int j = 0;

    double heavy_quark_res = 0.0;

    //
    m = w;
    mat(n, m, tmp);

    PrintStats( "PreconCG", j, (mNorm*mNorm), b2, heavy_quark_res);

    while(!convergence(mNorm, heavy_quark_res, stop, param.tol_hq) && j < param.maxiter){

      if(j > 0) {
         beta  = gamma / gammajm1;
         alpha = gamma / (delta - beta / alpha * gamma); 
      } 

      buffer = pipePCGMergedOp(x,alpha,p,u,r,s,m,beta,q,w,n,z);
#ifdef USE_WORKER
      {
        //global_reduce.apply(0);
        //Warning! ordering is critical here: fisrt preconditioner and then matvec, since worker uses blocking MPI allreduce
        //in this approach all reduce is overlapped with matvec only.
        //more robust way just to call non-blocking allreduce and then synchronize 
        if(K) {
          if( !precMatch )  rPre = w;
         
          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          if( !precMatch ) u = pPre;
        } else {
          m = w;
        }        
        //
        dslash::aux_worker = &global_reduce;
        mat(n, m, tmp);
        dslash::aux_worker = nullptr;
      }
#else
      {
        //comm_allreduce_array_async((double*) &buffer, 3, allreduceHandle);
        reduceDoubleArray((double*)&buffer, 3);

        if(K) {
          if( !precMatch )  rPre = w;
         
          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          if( !precMatch ) u = pPre;
        } else {
          m = w;
        }
        //
        mat(n, m, tmp);

        //comm_wait(allreduceHandle);
      }
#endif

      gammajm1 = gamma;
      gamma = buffer.x;
      delta = buffer.y;
      mNorm = buffer.z;
      //
      m = w;
      mat(n, m, tmp);

      j += 1;
      PrintStats( "PreconCG", j, mNorm, b2, heavy_quark_res);
    }
#ifdef USE_WORKER
    dslash::aux_worker = nullptr;
#else
    //comm_free(allreduceHandle);
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
    mat(r, x, tmp);
    double true_res = blas::xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

#ifdef PIPECG_DEBUG
    delete tz;
    delete tu;
    delete ts;
    delete tq;
#endif


    return;
  }


} // namespace quda
