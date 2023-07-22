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

  SD::SD(const DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(mat, mat, mat, mat, param, profile)
  {

  }

  SD::~SD(){
    if(!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);
    if(init){
      delete r;
      delete Ar; 
    }
    if(!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }


  void SD::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    commGlobalReductionPush(param.global_reduction);

    if (!init) {
      r = new ColorSpinorField(b);
      Ar = new ColorSpinorField(b);
      init = true;
    }

    real_t b2 = norm2(b);

    zero(*r), zero(x);
    real_t r2 = xmyNorm(b,*r);
    real_t alpha=0.; 

    int k=0;
    while (k < param.maxiter - 1) {
      mat(*Ar, *r);
      auto rAr = cDotProductNormA(*r, *Ar);
      alpha = rAr[2]/rAr[0];
      axpy(alpha, *r, x);
      axpy(-alpha, *Ar, *r);

      if(getVerbosity() >= QUDA_VERBOSE){
        r2 = norm2(*r);
        printfQuda("Steepest Descent: %d iterations, |r| = %e, |r|/|b| = %e\n", k, double(sqrt(r2)), double(sqrt(r2/b2)));
      }

      ++k;
    }

    auto rAr = cDotProductNormA(*r, *Ar);
    alpha = rAr[2] / rAr[0];
    axpy(alpha, *r, x);
    if(getVerbosity() >= QUDA_VERBOSE){
      axpy(-alpha, *Ar, *r);
      r2 = norm2(*r);
      printfQuda("Steepest Descent: %d iterations, |r| = %e, |r|/|b| = %e\n", k, double(sqrt(r2)), double(sqrt(r2/b2)));
      ++k;
    }

    if(getVerbosity() >= QUDA_DEBUG_VERBOSE){
      // Compute the true residual
      mat(*r, x);
      real_t true_r2 = xmyNorm(b,*r);
      printfQuda("Steepest Descent: %d iterations, accumulated |r| = %e, true |r| = %e,  |r|/|b| = %e\n", k, double(sqrt(r2)), double(sqrt(true_r2)), double(sqrt(true_r2/b2)));
    } // >= QUDA_DEBUG_VERBOSITY

    commGlobalReductionPop();
  }

} // namespace quda
