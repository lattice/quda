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
* Experimental PipePCG3 algorithm
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

    if(outer.inv_type == QUDA_PIPEPCG3_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  }


  PipePCG3::PipePCG3(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
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
  
  PipePCG3::~PipePCG3(){
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {

      if (K && (param.precision_precondition != param.precision_sloppy)) {
        delete p_pre;
        delete r_pre;
      }

      delete pp;

      delete tmpp;
      delete rp;
      delete yp;

      delete zp;
      delete qp;
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

  namespace dslash {
    extern Worker* aux_worker;
  }

  void PipePCG3::operator()(ColorSpinorField &x, ColorSpinorField &b) {
#if 1
    profile.TPSTART(QUDA_PROFILE_INIT);

    ColorSpinorParam csParam(b);

    if (!init) {
      // high precision fields:
      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      tmpp = ColorSpinorField::Create(csParam); //temporary for sloppy mat-vec

      // these low precision fields are used by the inner solver
      if (K && (param.precision_precondition != param.precision_sloppy)) {
        printfQuda("Allocated resources for the preconditioner.\n");
        csParam.setPrecision(param.precision_precondition);
	p_pre = ColorSpinorField::Create(csParam);
	r_pre = ColorSpinorField::Create(csParam);
      } 

      csParam.setPrecision(param.precision_sloppy);

      qp = ColorSpinorField::Create(csParam);
      pp = K ? ColorSpinorField::Create(csParam) : nullptr;

      csParam.is_composite  = true;
      csParam.composite_dim = 2;

      wp = ColorSpinorField::Create(csParam);
      xp_sloppy = ColorSpinorField::Create(csParam);
      rp_sloppy = ColorSpinorField::Create(csParam);

      zp = K ? ColorSpinorField::Create(csParam) : rp_sloppy;

      init = true;
    }

    csParam.setPrecision(param.precision_sloppy);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.is_composite = false;

    ColorSpinorField *tmpp_sloppy = (param.precision_sloppy != param.precision) ? ColorSpinorField::Create(csParam) : tmpp;

    ColorSpinorField &pPre = *p_pre;
    ColorSpinorField &rPre = *r_pre;
    ColorSpinorField &tmp  = *tmpp;
    ColorSpinorField &tmp_sloppy  = *tmpp_sloppy;

    ColorSpinorField &r = *rp;
    ColorSpinorField &z = *zp;
    ColorSpinorField &w = *wp;
    ColorSpinorField &q = *qp;
    ColorSpinorField &x_sloppy = *xp_sloppy;
    ColorSpinorField &r_sloppy = *rp_sloppy;

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

    r_sloppy.Component(0) = r;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    double4 local_reduce;//to keep double3 or double4 registers

    double beta, gamma, delta, delta_old, rho = 1.0;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    if(K) {
      rPre = r_sloppy.Component(0);

      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);

      z.Component(0) = pPre;
    } else {
      printfQuda("\nNo preconditioning...\n");
    }

    matSloppy(w.Component(0), z.Component(0), tmp_sloppy); // => w = A*u;

    delta   = blas::reDotProduct(z.Component(0),w.Component(0)); 
    beta    = blas::reDotProduct(z.Component(0),r.Component(0)); 
    double mNorm   = blas::norm2(z.Component(0));

    gamma  = beta / delta; 

    const double uro   = param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon()/2. : ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon()/2. : pow(2.,-13));
    const double tau = 100.0*sqrt(uro);

    blas::flops = 0;

    // now create the worker class for updating the gradient vectors
#ifdef USE_WORKER
    GlobalMPIallreduce global_reduce((double*)&local_reduce, 3);
    dslash::aux_worker = nullptr;//&global_reduce;
#else
    //MsgHandle* allreduceHandle = comm_handle();
    MPI_Request request_handle;
    double *recvbuff           = new double[3];//mpi buffer for async global reduction
#endif

    if(K) {
      rPre = w.Component(0);

      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);

      *pp = pPre;
    } else {
      pp = &w.Component(0);
    } 
    //
    matSloppy(q, *pp, tmp_sloppy);

    int j = 1;

    PrintStats( "PipePCG3 (before main loop)", j, mNorm, b2, 0.0);

    int updateR = 0;

    while(!convergence(mNorm, 0.0, stop, param.tol_hq) && j < param.maxiter){

      double mu = 1 - rho;
      double nu = rho*gamma;
 
      int jm2   = j % 2;
      int jm1   = 1 - jm2;

      axpbypcz(rho, x_sloppy.Component(jm1), nu, z.Component(jm1), mu, x_sloppy.Component(jm2));
      axpbypcz(rho, r_sloppy.Component(jm1), nu, w.Component(jm1), mu, r_sloppy.Component(jm2));
      axpbypcz(rho, w.Component(jm1), nu, q, mu, w.Component(jm2));
      if (K) axpbypcz(rho, z.Component(jm1), nu, *pp, mu, z.Component(jm2));

      int &currentj   = jm2;

      delta   = blas::reDotProduct(z.Component(currentj),w.Component(currentj)); 
      beta    = blas::reDotProduct(z.Component(currentj),r.Component(currentj)); 
      mNorm   = blas::norm2(z.Component(currentj));

#ifdef USE_WORKER
      {
        if(K) {
          rPre = w.Component(currentj);

          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          *pp = pPre;
        } else {
          pp = &w.Component(currentj);
        }        
          //
        dslash::aux_worker = &global_reduce;
        matSloppy(q, *pp, tmp_sloppy);
        dslash::aux_worker = nullptr;

      }
#else
      {
#ifdef NONBLOCK_REDUCE
        MPI_CHECK_(MPI_Iallreduce((double*)&local_reduce, recvbuff, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request_handle));
#else
        reduceDoubleArray((double*)&local_reduce, 3);
#endif
        if(K) {
          rPre = w.Component(currentj);

          commGlobalReductionSet(false);
          (*K)(pPre, rPre);
          commGlobalReductionSet(true);

          *pp = pPre;
        } else {
          pp = &w.Component(currentj);
        } 
        //
        matSloppy(q, *pp, tmp_sloppy);
#ifdef NONBLOCK_REDUCE//sync point
        MPI_CHECK_(MPI_Wait(&request_handle, MPI_STATUS_IGNORE));
        memcpy(&local_reduce, recvbuff, 3*sizeof(double));
#endif
      }
#endif //end of USE_WORKER

      j += 1;
    }

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
    xpy(x_sloppy.Component(j%3), x);
    mat(r, x, tmp);
    double true_res = blas::xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    //delete[] local_reduce;

    if (param.precision_sloppy != param.precision) {
      delete tmpp_sloppy;
    } 
#endif
    return;
  }

#ifdef USE_WORKER
#undef USE_WORKER
#endif 

} // namespace quda
