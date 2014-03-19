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


namespace quda {

  SD::SD(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param,profile), mat(mat), init(false)
  {

  }

  SD::~SD(){
    if(param.inv_type_precondition != QUDA_CG_INVERTER) profile.Start(QUDA_PROFILE_FREE);
    if(init){
      delete r;
      delete Ar; 
      delete y;
    }
    if(param.inv_type_precondition != QUDA_CG_INVERTER) profile.Stop(QUDA_PROFILE_FREE);
  }


  void SD::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b)
  {
    globalReduce = false;
    if(!init){
      r = new cudaColorSpinorField(b);
      Ar = new cudaColorSpinorField(b);
      y = new cudaColorSpinorField(b);
      init = true;
    }

    double b2 = norm2(b);

    zeroCuda(*r), zeroCuda(x);
    double r2 = xmyNormCuda(b,*r);
    double alpha=0.; 
    double beta=0;
    double2 Ar2;

    int k=0;
    while(k < param.maxiter-1){

      mat(*Ar, *r, *y);
      Ar2 = reDotProductNormACuda(*r, *Ar);
      alpha = Ar2.y/Ar2.x;
      axpyCuda(alpha, *r, x);
      axpyCuda(-alpha, *Ar, *r);

      if(getVerbosity() >= QUDA_VERBOSE){
        printfQuda("Steepest Descent: %d iterations, |r| = %e, |r|/|b| = %e\n", k, sqrt(r2), sqrt(r2/b2));
      }

      ++k;
    }


    Ar2 = reDotProductNormACuda(*r, *Ar);
    alpha = Ar2.y/Ar2.x;
    axpyCuda(alpha, *r, x);
    if(getVerbosity() >= QUDA_DEBUG_VERBOSE){
      axpyCuda(-alpha, *Ar, *r);
      r2 = norm2(*r);
      // Compute the true residual
      mat(*r, x, *y);
      double true_r2 = xmyNormCuda(b,*r);
      printfQuda("Steepest Descent: %d iterations, accumulated |r| = %e, true |r| = %e,  |r|/|b| = %e\n", k, sqrt(r2), sqrt(true_r2), sqrt(true_r2/b2));
    } // >= QUDA_DEBUG_VERBOSITY
    globalReduce = true;
    return;
  }

} // namespace quda
