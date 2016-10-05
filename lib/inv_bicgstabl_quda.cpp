#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>

#include <face_quda.h>

#include <iostream>
#include <sstream>

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
    
    std::stringstream ss;
    ss << "BiCGstab-" << nKrylov;
    solver_name = ss.str();
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

  void BiCGstabL::operator()(ColorSpinorField &x, ColorSpinorField &b) 
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
      for (int i = 0; i <= nKrylov; i++) {
        r[i] = ColorSpinorField::Create(csParam);
        u[i] = ColorSpinorField::Create(csParam);
      }
    }
    
    // Folowing the GCR inverter...
    ColorSpinorField &r0 = *r0p;
    ColorSpinorField &temp = *tempp;
    
    double b2 = blas::norm2(b); // norm sq of source.
    double r2;                  // norm sq of residual
    
    // Compute initial residual depending on whether we have an initial guess or not.
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(*r[0], x, temp); // r[0] = Ax
      r2 = blas::xmyNorm(b, *r[0]); // r = b - Ax, return norm.
    } else {
      blas::copy(*r[0], b); // r[0] = b
      r2 = b2;
      blas::zero(x); // defensive measure in case solution isn't already zero
    }
    
    std::cout << "r2 " << r2 << " b2 " << b2 << "\n" << std::flush;
    
    // Set some initial values.
    sigma[0] = blas::norm2(*r[0]);
    blas::copy(r0, *r[0]);
    blas::zero(*u[0]);
    
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
    //bool l2_converge = false;
    //double r2_old = r2;
    
    // done with preamble, begin computing.
    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    
    int k = 0;
    PrintStats(solver_name.c_str(), k, r2, b2, 0.0); // 0.0 is heavy quark residual.
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
        
        std::cout << "j " << j << " rho0 " << rho0 << "\n" << std::flush;
        
        // for i = 0 .. j, u[i] = r[i] - beta*u[i]
        for (int i = 0; i <= j; i++)
        {
          //blas::xpay(*r[i], -beta, *u[i]);
		      blas::caxpby(1.0, *r[i], -beta, *u[i]);
        }
        
        // u[j+1] = A ( u[j] )
        mat(*u[j+1], *u[j], temp);
        
        // alpha = rho0/<r0, u[j+1]>
        alpha = rho0/blas::cDotProduct(r0, *u[j+1]);
        
        // for i = 0 .. j, r[i] = r[i] - alpha u[i+1]
        for (int i = 0; i <= j; i++)
        {
          blas::caxpy(-alpha, *u[i+1], *r[i]);
        }
        
        // r[j+1] = A r[j], x = x + alpha*u[0]
        mat(*r[j+1], *r[j], temp);
        blas::caxpy(alpha, *u[0], x);
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
          blas::caxpy(-tau[i][j], *r[i], *r[j]);
        }
        
        // sigma_j = r_j^2, gamma'_j = <r_0, r_j>/sigma_j
        sigma[j] = blas::norm2(*r[j]);
        gamma_prime[j] = blas::cDotProduct(*r[j], *r[0])/sigma[j];
      }
      
      // gamma[nKrylov] = gamma'[nKrylov], omega = gamma[nKrylov]
      gamma[nKrylov] = gamma_prime[nKrylov];
      omega = gamma[nKrylov];
      
      // gamma = T^(-1) gamma_prime. It's in the paper, I promise.
      for (int j = nKrylov-1; j > 0; j--)
      {
        // Internal def: gamma[j] = gamma'_j - \sum_{i = j+1 to nKrylov} tau_ji gamma_i
        gamma[j] = gamma_prime[j];
        for (int i = j+1; i <= nKrylov; i++)
        {
          gamma[j] = gamma[j] - tau[j][i]*gamma[i];
        }
      }
      
      // gamma'' = T S gamma. Check paper for defn of S.
      for (int j = 1; j < nKrylov; j++)
      {
        gamma_prime_prime[j] = gamma[j+1];
        for (int i = j+1; i < nKrylov; i++)
        {
          gamma_prime_prime[j] = gamma_prime_prime[j] + tau[j][i]*gamma[i+1];
        }
      }
      
      // Update x, r, u.
      // x = x+ gamma_1 r_0, r_0 = r_0 - gamma'_l r_l, u_0 = u_0 - gamma_l u_l, where l = nKrylov.
      // I bet there's some fancy blas for this.
      blas::caxpy(gamma[1], *r[0], x);
      blas::caxpy(-gamma[nKrylov], *u[nKrylov], *u[0]);
      blas::caxpy(-gamma_prime[nKrylov], *r[nKrylov], *r[0]);
      
      // for j = 1 .. nKrylov-1: u[0] -= gamma_j u[j], x += gamma''_j r[j], r[0] -= gamma'_j r[j]
      for (int j = 1; j < nKrylov; j++)
      {
        blas::caxpy(-gamma[j], *u[j], *u[0]);
        blas::caxpy(gamma_prime_prime[j], *r[j], x);
        blas::caxpy(-gamma_prime[j], *r[j], *r[0]);
      }
      
      // sigma[0] = r_0^2
      sigma[0] = blas::norm2(*r[0]);
      r2 = sigma[0];
      
      // Check convergence.
      k += nKrylov;
      PrintStats(solver_name.c_str(), k, r2, b2, 0.0); // last thing should be heavy quark res...
    } // Done iterating.
    
    // Done with compute, begin the epilogue.
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);
    
    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops()/* + matSloppy.flops()*/)*1e-9;
    param.gflops = gflops;
    param.iter += k;
    
    if (k >= param.maxiter) // >= if nKrylov doesn't divide max iter.
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // Compute true residual
    

    // compute the true residual
    if (param.compute_true_res) {
      mat(*r[0], x, temp);
      double true_res = blas::xmyNorm(b, *r[0]);
      param.true_res = sqrt(true_res);
      // Probably some heavy quark stuff...
      param.true_res_hq = 0.0;
    }
    
    // Reset flops counters.
    blas::flops = 0;
    mat.flops();
    
    // Done with epilogue, begin free.
    
    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);
    
    // ...yup...
    PrintSummary(solver_name.c_str(), k, r2, b2);
    
    // Done!
    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }

} // namespace quda
