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

/***
* Experimental PipePCG algorithm
* Source P. Ghysels and W. Vanroose "Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm" 
***/

//#define USE_WORKER

#ifndef USE_WORKER

#define NONBLOCK_REDUCE
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
     int     elem;
     bool    stop; 
    public:
    GlobalMPIallreduce(double *buffer, int size) : buffer(buffer), size(size), elem(0), stop(false) { }

    virtual ~GlobalMPIallreduce() { }

    void reset() { elem = 0; stop = false; }

    void apply(const cudaStream_t &stream) {  //this has nothing to do with GPU streaming but should work as well, we may need non-blocked MPI allreduce, see cooments below
      if(stop) return;
      reduceDoubleArray(&buffer[elem], 1);
      elem +=1;
      if(elem == size) stop = true;  
      return;
    }
  };

  // this is the Worker pointer 
  namespace dslash {
    extern Worker* aux_worker;
  }

  void PipePCG::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    profile.TPSTART(QUDA_PROFILE_INIT);

    ColorSpinorParam csParam(b);

    ColorSpinorField *yp = nullptr;
    ColorSpinorField *lp = nullptr;

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

      yp = ColorSpinorField::Create(csParam);
      lp = ColorSpinorField::Create(csParam); 

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

    ColorSpinorField &y = *yp;
    ColorSpinorField &l = *lp;

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

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    double alpha, beta;
    double gamma, gammajm1, gamma_aux = 0.0;
    double delta;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    if(K) {
      rPre = r;

      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);

      u = pPre;
      blas::zero(m);
    } else {
      printfQuda("\nNo preconditioning...\n");
      u = r;
    }

    matSloppy(w, u, tmp); // => w = A*u;

    //Remark : no overlap here. 
    gamma = blas::reDotProduct(r,u); 
    delta = blas::reDotProduct(w,u);
    double mNorm = blas::norm2(u);

    alpha = 1.0;
    beta  = 0.0;

    double rNorm = sqrt(mNorm);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;

    const int maxResIncrease      = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    blas::flops = 0;

    double *local_buffer = nullptr;

    // now create the worker class for updating the gradient vectors
#ifdef USE_WORKER
    GlobalMPIallreduce global_reduce((double*)&buffer, 3);
    dslash::aux_worker = nullptr;//&global_reduce;
#else
    MsgHandle* allreduceHandle = comm_handle();
    double *recvbuff           = new double[4];//mpi buffer for async global reduction
#endif

    int j = 0;

    double heavy_quark_res = 0.0;
    //
    PrintStats( "PipePCG", j, mNorm, b2, heavy_quark_res);

    while(!convergence(mNorm, heavy_quark_res, stop, param.tol_hq) && j < param.maxiter){

      if(rNorm > maxrx) maxrx = mNorm;
      if(rNorm > maxrr) maxrr = mNorm;

      int updateX = (rNorm < param.delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < param.delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if( convergence(mNorm, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol) updateX = 1;

      if( ! (updateR || updateX)  ) {

        beta  = (gamma - gamma_aux) / gammajm1;
        alpha = gamma / (delta - beta / alpha * gamma); 

        commGlobalReductionSet(false);//disable global reduction

        if(K) {
          double4 res = pipePCGFletcherReevesMergedOp(x,alpha,p,u,r,s,m,beta,q,w,n,z);
          local_buffer= reinterpret_cast<double*>(&res);
        } else {
          double3 res = pipePCGMergedOp(x,alpha,p,u,r,s,m,beta,q,w,n,z);
          local_buffer= reinterpret_cast<double*>(&res);
        } 

        commGlobalReductionSet(true);
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
          matSloppy(n, m, tmp);
          dslash::aux_worker = nullptr;

          global_reduce.reset();
        }
#else
        {
#ifdef NONBLOCK_REDUCE
          comm_allreduce_array_async(recvbuff, local_buffer, (K ? 4 : 3), allreduceHandle);
#else
          reduceDoubleArray(local_buffer, (K ? 4 : 3));
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
          mat(n, m, tmp);
#ifdef NONBLOCK_REDUCE//sync point
          comm_wait(allreduceHandle);
          memcpy(&local_buffer, recvbuff, (K ? 4 : 3)*sizeof(double));
#endif
        }
#endif //end of USE_WORKER
        gammajm1 = gamma;
        gamma = local_buffer[0];
        delta = local_buffer[1];
        mNorm = local_buffer[2];
        gamma_aux = K ? local_buffer[3] : 0.0;
     } else { //trigger reliable updates:
        xpy(x,y);
        zero(x);

        mat(r, y, tmp);
        xpay(b, -1.0, r);

        mNorm = norm2(r);

        // break-out check if we have reached the limit of the precision
        if(sqrt(mNorm) > r0Norm && updateX) {
          resIncrease++;
          resIncreaseTotal++;
          // reuse r0Norm for this
          warningQuda("PCG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)", sqrt(mNorm), r0Norm, resIncreaseTotal);

//        if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) break;

        } else {
          resIncrease = 0;
        }

        rNorm = sqrt(mNorm);
        maxrr = rNorm;
        maxrx = rNorm;
        r0Norm = rNorm;

        if(K) {
          commGlobalReductionSet(false);
          (*K)(u, r);
          commGlobalReductionSet(true);
        } else {
          u = r;
        }

        matSloppy(w, u, tmp);
        matSloppy(s, p, tmp);

        zero(l);

        if(K) {
          commGlobalReductionSet(false);
          (*K)(q, s);
          commGlobalReductionSet(true);
        } else {
          q = s;
        }

        matSloppy(z, q, tmp);
      }

      j += 1;
      PrintStats( "PipePCG", j, mNorm, b2, heavy_quark_res);
    }

#ifdef NONBLOCK_REDUCE
    host_free(allreduceHandle);
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
    xpy(y, x);
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

    return;
  }


} // namespace quda
