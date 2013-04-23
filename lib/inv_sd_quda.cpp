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


  SD::SD(const DiracMatrix &mat,  QudaInvertParam &invParam, TimeProfile &profile) :
    Solver(invParam, profile), mat(mat), init(false)
  {
  }


  SD::~SD() {
    if(init){ 
      delete Ar;
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

  void SD::operator()(cudaColorSpinorField& x, cudaColorSpinorField& b,  double* time)
  {
    globalReduce = false;
    if(!init){
      r = new cudaColorSpinorField(b);
      Ar = new cudaColorSpinorField(b); 
      y = new cudaColorSpinorField(b);
      init = true;                        
    }

    double b2;
    //if(invParam.verbosity >= QUDA_DEBUG_VERBOSE) b2 = norm2(b);
    b2 = norm2(b);

    // Assumes x = b
    //mat(*r, b, *y); // operator()(cudaColorSpinorField& out, cudaColorSpinorField& in,
    // Switching to a zero source would get rid of this operation. 
    // Will it affect the number of iterations
    // => r = A*x;
    // I can make this much more efficient!
    zeroCuda(*r);
    zeroCuda(x); 
    double r2 = xmyNormCuda(b,*r);
    double alpha = 0.0, beta=0.0;
    double2 Ar2;

    int k=0;
    while(k < invParam.maxiter-1){

      mat(*Ar, *r, *y);
      Ar2 = reDotProductNormACuda(*r, *Ar); 
      alpha = Ar2.y/Ar2.x;
      axpyCuda(alpha, *r, x); // better way to do this!
      axpyCuda(-alpha, *Ar, *r);

      if(invParam.verbosity >= QUDA_DEBUG_VERBOSE){
        printfQuda("Steepest Descent: %d iterations, |r| = %e, |r|/|b| = %e\n", k, sqrt(r2), sqrt(r2/b2));
      }

      ++k;
    }

    mat(*Ar, *r, *y);
    Ar2 = reDotProductNormACuda(*r, *Ar);
    alpha = Ar2.y/Ar2.x;
    axpyCuda(alpha, *r, x);
    // x += alpha*r
//    if(invParam.verbosity >= QUDA_DEBUG_VERBOSE){
      axpyCuda(-alpha, *Ar, *r);
      r2 = norm2(*r);
      // Compute the true residual
      mat(*r, x, *y);
      double true_r2 = xmyNormCuda(b,*r);
      printfQuda("Steepest Descent: %d iterations, accumulated |r| = %e, true |r| = %e,  |r|/|b| = %e\n", k, sqrt(r2), sqrt(true_r2), sqrt(true_r2/b2));
//    }

    globalReduce = true;
    return;
  }


} // namespace quda
