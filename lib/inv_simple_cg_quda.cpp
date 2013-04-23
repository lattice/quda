#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <face_quda.h>
#include <domain_decomposition.h>
#include <resize_quda.h>

// I need to add functionality to the conjugate-gradient solver.
// Should I do this by simple inheritance, or should I use a decorator? 
// Is it even possible to use a decorator?

namespace quda {


  SimpleCG::SimpleCG(const DiracMatrix &mat,  QudaInvertParam &invParam, TimeProfile &profile) :
    Solver(invParam, profile), mat(mat), init(false)
  {
  }


  SimpleCG::~SimpleCG() {
    if(init){ 
      delete p;
      delete Ap;
      delete r;
      delete y;
    }
    init = false;
  }


  inline void accumulate_time(double* time_difference, const timeval& start, const timeval& stop)
  {
    long ds = stop.tv_sec - start.tv_sec;
    long dus = stop.tv_usec - start.tv_usec;
    *time_difference += ds + 0.000001*dus;
  }

  //void SimpleCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField &b, double* time)
  void SimpleCG::operator()(cudaColorSpinorField& x, cudaColorSpinorField& b,  double* time)
  {
    globalReduce = false;

    // profile[QUDA_PROFILE_INIT].Start();
    if(!init){
      r = new cudaColorSpinorField(b);
      p = new cudaColorSpinorField(b);
      Ap = new cudaColorSpinorField(b); 
      y = new cudaColorSpinorField(b);
      init = true;                        
    }


    // Assumes x = b
    /*
       mat(*p, b, *y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
    // Switching to a zero source would get rid of this operation. 
    // Will it affect the number of iterations
    // => r = A*x;
    double r2 = xmyNormCuda(b,*p);
    double alpha = 0.0, beta=0.0;
    double pAp;
    double r2_old;
    mat(*Ap, *p, *y);
    pAp = reDotProductCuda(*p, *Ap);
    alpha = r2/pAp;
    axpyzCuda(-alpha, *Ap, *p, *r); // r = p - alpha*Ap
    r2_old = r2;
    r2 = norm2(*r);
    beta = r2/r2_old;
    axpyzCuda(alpha, *p, b, x);   // This will do away with an additional copy
    axpyzCuda(beta, *p, *r, *p);  // Will it work.
    // x = x + alpha*p
    // p = r + beta*p
    int k=1;
    */

//    quda::blas_flops = 0; // start fresh

    zeroCuda(x);
    zeroCuda(*r);
    double r2 = xmyNormCuda(b,*r);
    double alpha = 0.0, beta = 0.0;
    double pAp, r2_old;
    *p = *r; 

    double b2;
    //if(invParam.verbosity >= QUDA_DEBUG_VERBOSE) b2 = norm2(b);
    b2 = norm2(b);


    int k=0;
    while( k < invParam.maxiter-1 ){
      mat(*Ap, *p, *y);
      pAp = reDotProductCuda(*p, *Ap);
      alpha = r2/pAp; 
      axpyCuda(-alpha, *Ap, *r); // r --> r - alpha*Ap
      r2_old = r2;
      r2 = norm2(*r);
      beta = r2/r2_old;
      axpyZpbxCuda(alpha, *p, x, *r, beta);
      // x = x + alpha*p
      // p = r + beta*p
//      if(invParam.verbosity >= QUDA_DEBUG_VERBOSE){
        printfQuda("Inner CG: %d iterations, |r| = %e, |r|/|b| = %e\n", k, sqrt(r2), sqrt(r2/b2));
//      } 
      ++k;
    }
    mat(*Ap, *p, *y);
    pAp = reDotProductCuda(*p, *Ap);
    alpha  = r2/pAp;
    axpyCuda(alpha, *p, x); // x --> x + alpha*p

/*
    double gflops = (quda::blas_flops + mat.flops())*1e-9;
    reduceDouble(gflops);
    invParam.gflops += gflops;

    quda::blas_flops = 0;
    mat.flops();
*/
//    if(invParam.verbosity >= QUDA_DEBUG_VERBOSE){
      axpyCuda(-alpha, *Ap, *r); 
      r2 = norm2(*r);
      // Compute the true residual
      mat(*r, x, *y);
      double true_r2 = xmyNormCuda(b,*r);
      printfQuda("Inner CG: %d iterations, accumulated |r| = %e, true |r| = %e,  |r|/|b| = %e\n", k, sqrt(r2), sqrt(true_r2), sqrt(true_r2/b2));
//    } 

    globalReduce = true;
    return;
  }


} // namespace quda
