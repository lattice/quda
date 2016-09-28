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

  BiCGstabL::BiCGstabL(DiracMatrix &mat, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), nKrylov(param.Nkrylov), init(false)
  {
    r.resize(nKrylov+1);
    u.resize(nKrylov+1);
    
    gamma = new Complex[nKrylov+1];
    gamma_prime = new Complex[nKrylov+1];
    gamma_prime_prime = new Complex[nKrylov+1];
    sigma = new double[nKrylov+1];
    
    tau = new Complex*[nKrylov+1];
    for (int i = 0; i < nKrylov+1; i++) { tau[i] = new Complex[nKrylov+1]; }
  }

  BiCGstabL::~BiCGstabL() {
    profile.TPSTART(QUDA_PROFILE_FREE);
    delete[] gamma;
    delete[] gamma_prime;
    delete[] gamma_prime_prime;
    delete[] sigma;
    
    for (int i = 0; i < nKrylov+1; i++) { delete[] tau[i]; }
    delete[] tau; 
    
    if (init) {
      for (int i = 0; i < nKrylov+1; i++) {
        delete r[i];
        delete u[i];
      }
      
      delete r0p;
      delete tempp;
      
      init = false;
    }
    
    profile.TPSTOP(QUDA_PROFILE_FREE);
    
  }

  void BiCGstabL::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {
    // BiCGstab-l is based on the algorithm outlined in
    // BICGSTAB(L) FOR LINEAR EQUATIONS INVOLVING UNSYMMETRIC MATRICES WITH COMPLEX SPECTRUM
    // G. Sleijpen, D. Fokkema, 1993.
    // My implementation is based on Kate Clark's implementation in CPS, to be found in
    // src/util/dirac_op/d_op_wilson_types/bicgstab.C
    
    // Begin profiling initialization.
    profile.TPSTART(QUDA_PROFILE_INIT);
    
    if (!init) {
      // Initialize fields.
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      
      // Shadow residual.
      r0p = ColorSpinorField::Create(csParam);
      
      // Create temporary.
      tempp = ColorSpinorField::Create(csParam);
      
      // Residual (+ extra residuals for BiCG steps), Search directions.
      // Remark: search directions are sloppy in GCR. I wonder if we can
      //           get away with that here.
      for (int i = 0; i < nKrylov; i++) {
        r[i] = ColorSpinorField::Create(csParam);
        u[i] = ColorSpinorField::Create(csParam);
      }
    }
    
    // Folowing the GCR inverter...
    ColorSpinorField &r0 = *r0p;
    ColorSpinorField &temp = *tempp;
    blas::zero(y); 
    
    double b2 = blas::norm2(b); // norm sq of source.
    double r2;                  // norm sq of residual
    
    // Compute initial residual depending on whether we have an initial guess or not.
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r[0], x, temp); // r[0] = Ax
      r2 = blas::xmyNorm(b, *r[0]); // r = b - Ax, return norm.
    } else {
      blas::copy(*r[0], b); // r[0] = b
      r2 = b2;
      blas::zero(x); // defensive measure in case solution isn't already zero
    }
    
    // Set some initial values.
    sigma[0] = blas::norm2(*r[0]);
    blas::copy(r0, *r[0]);
    
    // Check to see that we're not trying to invert on a zero-field source    
    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      warningQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    // Initialize values.
    rho0 = 1.0;
    alpha = 0.0;
    omega = 1.0;
    
    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver.

    // done with the initialization, start preamble.
    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    
    blas::flops = 0;
    bool l2_converge = false;
    double r2_old = r2;
    
    // done with preamble, begin computing.
    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    
    int k = 0;
    PrintStats("BiCGstab-l", k, r2, b2, 0.0); // 0.0 is heavy quark residual.
    while(!convergence(r2, 0.0, stop, 0.0) && k < param.maxiter) {

      //PrintStats("BiCGstab-l", k, r2, b2, 0.0);
      
      // rho0 = -omega*rho0;
      rho0 *= -omega;
      
      // BiCG part of calculation.
      for (int j = 0; j < nKrylov; j++) {
        
        // rho1 = <r0, r_j>, beta = alpha*rho1/rho0, rho0 = rho1;
        rho1 = blas::cDotProduct(r0, *r[j]);
        beta = alpha*rho1/rho0;
        rho0 = rho1;
        
        // for i = 0 .. j, u[i] = r[i] - beta*u[i]
        for (int i = 0; i <= j; i++)
        {
          blas::xpay(*r[i], -beta, *u[i]);
        }
        
        // u[j+1] = A ( u[j] )
        mat(*u[j+1], *u[j], temp);
        
        // alpha = rho0/<r0, u[j+1]>
        alpha = rho0/blas::cDotProduct(r0, *u[j+1]);
        
        // for i = 0 .. j, r[i] = r[i] - alpha u[i+1]
        for (int i = 0; i <= j; i++)
        {
          blas::axpy(-alpha, *u[i+1], *r[i]);
        }
        
        // r[j+1] = A r[j], x = x + alpha*u[0]
        mat(*r[j+1], *r[j], temp);
        blas::axpy(alpha, *u[0], x);
      } // End BiCG part.
      
      // MR part. Really just modified Gram-Schmidt.
      // The algorithm uses the byproducts of the Gram-Schmidt to update x
      //   and other such niceties. One day I'll read the paper more closely.
      for (int j = 1; j <= nKrylov; j++)
      {
        for (int i = 1; i < j; i++)
        {
          // tau_ij = <r_i,r_j>/sigma_i.
          // This doesn't break on the first iteration because i < j is true.
          // (I was confused about this.)
          tau[i][j] = blas::cDotProduct(*r[i], *r[j])/sigma[i];
          
          // r_j = r_j - tau_ij r_i;
          blas::axpy(-tau[i][j], *r[i], *r[j]);
        }
        
        // sigma_j = r_j^2, gamma'_j = <r_0, r_j>/sigma_j
        sigma[j] = blas::norm2(*r[j]);
        gamma_prime[j] = blas::cDotProduct(*r[j], *r[0])/sigma[j];
      }
      
      // gamma[nKrylov] = gamma'[nKrylov], omega = gamma[nKrylov]
      gamma[nKrylov] = gamma_prime[nKrylov];
      omega = gamma[nKrylov];
      
      
          
   
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

    PrintSummary("BiCGstabL", k, r2, b2);

    return;
  }

} // namespace quda
