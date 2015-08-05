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

  SimpleBiCGstab::SimpleBiCGstab(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat)
  {
  }

  SimpleBiCGstab::~SimpleBiCGstab() {
  }

  void SimpleBiCGstab::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);
    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    
    cudaColorSpinorField temp(b, csParam);

    cudaColorSpinorField r(b);
    


    mat(r, x, temp);  // r = Ax
    double r2 = xmyNormCuda(b,r); // r = b - Ax

    cudaColorSpinorField r0(r);
    cudaColorSpinorField p(r);
    cudaColorSpinorField Ap(r);
    cudaColorSpinorField A2p(r);
    cudaColorSpinorField Ar(r);
    cudaColorSpinorField r_new(r);
    cudaColorSpinorField p_new(r);
    Complex r0r;
    Complex alpha;
    Complex omega;
    Complex beta;


    double p2 = norm2(p);
    double stop = stopping(param.tol, b2, param.residual_type);
    int k=0;
    while(!convergence(r2, 0.0, stop, 0.0) && k<param.maxiter){

      PrintStats("SimpleBiCGstab", k, r2, b2, 0.0);
   
      mat(Ap,p,temp); 
      mat(A2p,Ap,temp);
      mat(Ar,r,temp);   

 
      r0r   = cDotProductCuda(r0,r);
      alpha = r0r/cDotProductCuda(r0,Ap); 


      Complex omega_num    =  cDotProductCuda(r,Ar) 
                           -  alpha*cDotProductCuda(r,A2p)
                           -  conj(alpha)*cDotProductCuda(Ap,Ar)
                           +  conj(alpha)*alpha*cDotProductCuda(Ap,A2p);


      Complex omega_denom  = cDotProductCuda(Ar,Ar) 
                           - alpha*cDotProductCuda(Ar,A2p) 
                           - conj(alpha)*cDotProductCuda(A2p,Ar)
                           + conj(alpha)*alpha*cDotProductCuda(A2p,A2p);


      omega = omega_num/omega_denom;


    
      // x ---> x + alpha p + omega s 
      caxpyCuda(alpha,p,x);
      caxpyCuda(omega,r,x);
      caxpyCuda(-alpha*omega,Ap,x);


      // r_new = r - omega*Ar - alpha*Ap + alpha*omega*A2p
      r_new = r;
      caxpyCuda(-omega,Ar,r_new);
      caxpyCuda(-alpha,Ap,r_new);
      caxpyCuda(alpha*omega,A2p,r_new);

      beta = (cDotProductCuda(r0,r_new)/r0r)*(alpha/omega);

      
      // p = r_new + beta p - omega*beta Ap
      p_new = r_new;
      caxpyCuda(beta, p, p_new);
      caxpyCuda(-beta*omega, Ap, p_new); 
   
      p = p_new; 
      r = r_new;
      r2 = norm2(r);
      p2 = norm2(p);
      k++;
    }

  
    if(k == param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residual
    mat(r, x, temp);
    param.true_res = sqrt(xmyNormCuda(b, r)/b2);

    PrintSummary("SimpleBiCGstab", k, r2, b2);

    return;
  }

} // namespace quda
