#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

  CG3::CG3(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat)
  {
  }

  CG3::~CG3() {
  }

  void CG3::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
  
    
    cudaColorSpinorField x_prev(b, csParam);  
    cudaColorSpinorField r_prev(b, csParam);
    cudaColorSpinorField temp(b, csParam);

    cudaColorSpinorField r(b);
    cudaColorSpinorField w(b);


    mat(r, x, temp);  // r = Mx
    double r2 = xmyNormCuda(b,r); // r = b - Mx
    PrintStats("CG3", 0, r2, b2, 0.0);


    double stop = stopping(param.tol, b2, param.residual_type);
    if(convergence(r2, 0.0, stop, 0.0)) return;
    // First iteration 
    mat(w, r, temp);
    double rAr = reDotProductCuda(r,w);
    double rho = 1.0;
    double gamma_prev = 0.0;
    double gamma = r2/rAr;


    cudaColorSpinorField x_new(x);
    cudaColorSpinorField r_new(r);
    axpyCuda(gamma, r, x_new);  // x_new += gamma*r
    axpyCuda(-gamma, w, r_new); // r_new -= gamma*w
    // end of first iteration  

    // axpbyCuda(a,b,x,y) => y = a*x + b*y

    int k = 1; // First iteration performed above

    double r2_prev;
    while(!convergence(r2, 0.0, stop, 0.0) && k<param.maxiter){
      x_prev = x; x = x_new;
      r_prev = r; r = r_new;
      mat(w, r, temp);
      rAr = reDotProductCuda(r,w);
      r2_prev = r2;
      r2 = norm2(r);

      // Need to rearrange this!
      PrintStats("CG3", k, r2, b2, 0.0);

      gamma_prev = gamma;
      gamma = r2/rAr;
      rho = 1.0/(1. - (gamma/gamma_prev)*(r2/r2_prev)*(1.0/rho));
      
      x_new = x;
      axCuda(rho,x_new); 
      axpyCuda(rho*gamma,r,x_new);
      axpyCuda((1. - rho),x_prev,x_new);

      r_new = r;
      axCuda(rho,r_new);
      axpyCuda(-rho*gamma,w,r_new);
      axpyCuda((1.-rho),r_prev,r_new);


       double rr_old = reDotProductCuda(r_new,r);
      printfQuda("rr_old = %1.14lf\n", rr_old);


 
      k++;
    }


    if(k == param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual
    mat(r, x, temp);
    param.true_res = sqrt(xmyNormCuda(b, r)/b2);

    PrintSummary("CG3", k, r2, b2);

    return;
  }

} // namespace quda
