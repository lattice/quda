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

namespace quda {

  using namespace blas;
  
  SD::SD(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param,profile), mat(mat), init(false)
  {

  }

  SD::~SD(){
    if(!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);
    if(init){
      delete r;
      delete Ar; 
      delete y;
    }
    if(!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }


  void SD::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    commGlobalReductionSet(param.global_reduction);

    if(!init){
      r = new cudaColorSpinorField(b);
      Ar = new cudaColorSpinorField(b);
      y = new cudaColorSpinorField(b);
      init = true;
    }

    double b2 = norm2(b);

    zero(*r), zero(x);
    double r2 = xmyNorm(b,*r);
    double alpha=0.; 
    double2 rAr;

    int k=0;
    while(k < param.maxiter-1){

      mat(*Ar, *r, *y);
      rAr = reDotProductNormA(*r, *Ar);
      alpha = rAr.y/rAr.x;
      axpy(alpha, *r, x);
      axpy(-alpha, *Ar, *r);

      if(getVerbosity() >= QUDA_VERBOSE){
        r2 = norm2(*r);
        printfQuda("Steepest Descent: %d iterations, |r| = %e, |r|/|b| = %e\n", k, sqrt(r2), sqrt(r2/b2));
      }

      ++k;
    }


    rAr = reDotProductNormA(*r, *Ar);
    alpha = rAr.y/rAr.x;
    axpy(alpha, *r, x);
    if(getVerbosity() >= QUDA_VERBOSE){
      axpy(-alpha, *Ar, *r);
      r2 = norm2(*r);
      printfQuda("Steepest Descent: %d iterations, |r| = %e, |r|/|b| = %e\n", k, sqrt(r2), sqrt(r2/b2));
      ++k;
    }

    if(getVerbosity() >= QUDA_DEBUG_VERBOSE){
      // Compute the true residual
      mat(*r, x, *y);
      double true_r2 = xmyNorm(b,*r);
      printfQuda("Steepest Descent: %d iterations, accumulated |r| = %e, true |r| = %e,  |r|/|b| = %e\n", k, sqrt(r2), sqrt(true_r2), sqrt(true_r2/b2));
    } // >= QUDA_DEBUG_VERBOSITY

    commGlobalReductionSet(true);
    return;
  }

} // namespace quda
