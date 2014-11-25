#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include<face_quda.h>

#include <color_spinor_field.h>

namespace quda {

  // set the required parameters for the inner solver
  void fillInnerSolveParam(SolverParam &inner, const SolverParam &outer);

  double resNorm(const DiracMatrix &mat, cudaColorSpinorField &b, cudaColorSpinorField &x) {  
    cudaColorSpinorField r(b);
    mat(r, x);
    return xmyNormCuda(b, r);
  }


  BiCGstab::BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), init(false) {

  }

  BiCGstab::~BiCGstab() {
    profile.Start(QUDA_PROFILE_FREE);

    if(init) {
      delete yp;
      delete rp;
      delete pp;
      delete vp;
      delete tmpp;
      delete tp;
    }

    profile.Stop(QUDA_PROFILE_FREE);
  }

  int reliable(double &rNorm, double &maxrx, double &maxrr, const double &r2, const double &delta) {
    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrx) maxrx = rNorm;
    if (rNorm > maxrr) maxrr = rNorm;
    //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
    //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0
    int updateR = (rNorm < delta*maxrr) ? 1 : 0;
    
    //printf("reliable %d %e %e %e %e\n", updateR, rNorm, maxrx, maxrr, r2);

    return updateR;
  }

  void BiCGstab::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {
    profile.Start(QUDA_PROFILE_PREAMBLE);

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = new cudaColorSpinorField(x, csParam);
      rp = new cudaColorSpinorField(x, csParam); 
      csParam.setPrecision(param.precision_sloppy);
      pp = new cudaColorSpinorField(x, csParam);
      vp = new cudaColorSpinorField(x, csParam);
      tmpp = new cudaColorSpinorField(x, csParam);
      tp = new cudaColorSpinorField(x, csParam);

      init = true;
    }

    cudaColorSpinorField &y = *yp;
    cudaColorSpinorField &r = *rp; 
    cudaColorSpinorField &p = *pp;
    cudaColorSpinorField &v = *vp;
    cudaColorSpinorField &tmp = *tmpp;
    cudaColorSpinorField &t = *tp;

    cudaColorSpinorField *x_sloppy, *r_sloppy, *r_0;

    double b2 = normCuda(b); // norm sq of source
    double r2;               // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, y);
      r2 = xmyNormCuda(b, r);
      copyCuda(y, x);
    } else {
      copyCuda(r, b);
      r2 = b2;
      zeroCuda(x); // defensive measure in case solution isn't already zero
    }

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0) {
      profile.Stop(QUDA_PROFILE_INIT);
      warningQuda("inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    // set field aliasing according to whether we are doing mixed precision or not
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
      r_0 = &b;
    } else {
      ColorSpinorParam csParam(x);
      csParam.setPrecision(param.precision_sloppy);
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(r, csParam);
      r_0 = new cudaColorSpinorField(b, csParam);
    }

    // set field aliasing according to whether we are doing mixed precision or not
    if (param.precision_sloppy == x.Precision() ||
	!param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
      zeroCuda(*x_sloppy);
    } else {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      x_sloppy = new cudaColorSpinorField(x, csParam);
    }

    // Syntatic sugar
    cudaColorSpinorField &rSloppy = *r_sloppy;
    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &r0 = *r_0;

    SolverParam solve_param_inner(param);
    fillInnerSolveParam(solve_param_inner, param);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_res = use_heavy_quark_res ? sqrt(HeavyQuarkResidualNormCuda(x,r).z) : 0.0;
    int heavy_quark_check = 10; // how often to check the heavy quark residual

    double delta = param.delta;

    int k = 0;
    int rUpdate = 0;
  
    Complex rho(1.0, 0.0);
    Complex rho0 = rho;
    Complex alpha(1.0, 0.0);
    Complex omega(1.0, 0.0);
    Complex beta;

    double3 rho_r2;
    double3 omega_t2;
  
    double rNorm = sqrt(r2);
    //double r0Norm = rNorm;
    double maxrr = rNorm;
    double maxrx = rNorm;

    PrintStats("BiCGstab", k, r2, b2, heavy_quark_res);
    
    if (param.inv_type_precondition != QUDA_GCR_INVERTER) { // do not do the below if we this is an inner solver
      quda::blas_flops = 0;    
    }

    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);
    
    rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
    copyCuda(p, rSloppy);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
      printfQuda("BiCGstab debug: x2=%e, r2=%e, v2=%e, p2=%e, tmp2=%e r0=%e t2=%e\n", 
		 norm2(x), norm2(rSloppy), norm2(v), norm2(p), norm2(tmp), norm2(r0), norm2(t));

    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && 
	    k < param.maxiter) {
    
      matSloppy(v, p, tmp);

      Complex r0v;
      if (param.pipeline) {
	r0v = cDotProductCuda(r0, v);
	if (k>0) rho = cDotProductCuda(r0, r);
      } else {
	r0v = cDotProductCuda(r0, v);
      }
      if (abs(rho) == 0.0) alpha = 0.0;
      else alpha = rho / r0v;

      // r -= alpha*v
      caxpyCuda(-alpha, v, rSloppy);

      matSloppy(t, rSloppy, tmp);
    
      int updateR = 0;
      if (param.pipeline) {
	// omega = (t, r) / (t, t)
	omega_t2 = cDotProductNormACuda(t, rSloppy);
	Complex tr = Complex(omega_t2.x, omega_t2.y);
	double t2 = omega_t2.z;
	omega = tr / t2;
	double s2 = norm2(rSloppy);
	Complex r0t = cDotProductCuda(r0, t);
	beta = -r0t / r0v;
	r2 = s2 - real(omega * conj(tr)) ;

	// now we can work out if we need to do a reliable update
        updateR = reliable(rNorm, maxrx, maxrr, r2, delta);
      } else {
	// omega = (t, r) / (t, t)
	omega_t2 = cDotProductNormACuda(t, rSloppy);
	omega = Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);
      }

      if (param.pipeline && !updateR) {
	//x += alpha*p + omega*r, r -= omega*t, p = r - beta*omega*v + beta*p
	caxpbypzYmbwCuda(alpha, p, omega, rSloppy, xSloppy, t);
	cxpaypbzCuda(rSloppy, -beta*omega, v, beta, p);
	//tripleBiCGstabUpdate(alpha, p, omega, rSloppy, xSloppy, t, -beta*omega, v, beta, p
      } else {
	//x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
	rho_r2 = caxpbypzYmbwcDotProductUYNormYCuda(alpha, p, omega, rSloppy, xSloppy, t, r0);

	rho0 = rho;
	rho = Complex(rho_r2.x, rho_r2.y);
	r2 = rho_r2.z;
      }

      if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	if (&x != &xSloppy) {
	  copyCuda(tmp,y);
	  heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
	} else {
	  copyCuda(r, rSloppy);
	  heavy_quark_res = sqrt(xpyHeavyQuarkResidualNormCuda(x, y, r).z);	  
	}
      }

      if (!param.pipeline) updateR = reliable(rNorm, maxrx, maxrr, r2, delta);

      if (updateR) {
	if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
	xpyCuda(x, y); // swap these around?

	mat(r, y, x);
	r2 = xmyNormCuda(b, r);

	if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
	zeroCuda(xSloppy);

	rNorm = sqrt(r2);
	maxrr = rNorm;
	maxrx = rNorm;
	//r0Norm = rNorm;      
	rUpdate++;
      }
    
      k++;

      PrintStats("BiCGstab", k, r2, b2, heavy_quark_res);
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
	printfQuda("BiCGstab debug: x2=%e, r2=%e, v2=%e, p2=%e, tmp2=%e r0=%e t2=%e\n", 
		   norm2(x), norm2(rSloppy), norm2(v), norm2(p), norm2(tmp), norm2(r0), norm2(t));

      // update p
      if (!param.pipeline || updateR) {// need to update if not pipeline or did a reliable update
	if (abs(rho*alpha) == 0.0) beta = 0.0;
	else beta = (rho/rho0) * (alpha/omega);      
	cxpaypbzCuda(rSloppy, -beta*omega, v, beta, p);
      }

    }

    if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
    xpyCuda(y, x);

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    reduceDouble(gflops);

    param.gflops += gflops;
    param.iter += k;

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("BiCGstab: Reliable updates = %d\n", rUpdate);
  
    if (param.inv_type_precondition != QUDA_GCR_INVERTER) { // do not do the below if we this is an inner solver
      // Calculate the true residual
      mat(r, x);
      param.true_res = sqrt(xmyNormCuda(b, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
      param.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
#else
      param.true_res_hq = 0.0;
#endif
 
      PrintSummary("BiCGstab", k, r2, b2);      
    }

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);

    profile.Start(QUDA_PROFILE_FREE);
    if (param.precision_sloppy != x.Precision()) {
      delete r_0;
      delete r_sloppy;
    }

    if (&x != &xSloppy) delete x_sloppy;

    profile.Stop(QUDA_PROFILE_FREE);
    
    return;
  }

} // namespace quda
