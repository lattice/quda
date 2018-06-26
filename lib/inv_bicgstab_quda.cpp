#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <color_spinor_field.h>

namespace quda {

  // set the required parameters for the inner solver
  void fillInnerSolveParam(SolverParam &inner, const SolverParam &outer);

  BiCGstab::BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), init(false) {

  }

  BiCGstab::~BiCGstab() {
    profile.TPSTART(QUDA_PROFILE_FREE);

    if(init) {
      delete yp;
      delete rp;
      delete pp;
      delete vp;
      delete tmpp;
      delete tp;
    }

    profile.TPSTOP(QUDA_PROFILE_FREE);
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

  void BiCGstab::operator()(ColorSpinorField &x, ColorSpinorField &b) 
  {
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(csParam);
      rp = ColorSpinorField::Create(csParam);
      csParam.setPrecision(param.precision_sloppy);
      pp = ColorSpinorField::Create(csParam);
      vp = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);
      tp = ColorSpinorField::Create(csParam);

      init = true;
    }

    ColorSpinorField &y = *yp;
    ColorSpinorField &r = *rp; 
    ColorSpinorField &p = *pp;
    ColorSpinorField &v = *vp;
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &t = *tp;

    ColorSpinorField *x_sloppy, *r_sloppy, *r_0;

    double b2 = blas::norm2(b); // norm sq of source
    double r2;               // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, y);
      r2 = blas::xmyNorm(b, r);
      blas::copy(y, x);
    } else {
      blas::copy(r, b);
      r2 = b2;
      blas::zero(x);
    }

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0) {
      if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
        warningQuda("inverting on zero-field source");
        x = b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
	profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
        return;
      } else if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
        b2 = r2;
      } else {
        errorQuda("Null vector computing requires non-zero guess!");
      }
    }

    // set field aliasing according to whether we are doing mixed precision or not
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;

      if(param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO)
      {
        r_0 = &b;
      }
      else
      {
        ColorSpinorParam csParam(r);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        r_0 = ColorSpinorField::Create(csParam);//remember to delete this pointer.
        *r_0 = r;
      }
    } else {
      ColorSpinorParam csParam(x);
      csParam.setPrecision(param.precision_sloppy);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      r_sloppy = ColorSpinorField::Create(csParam);
      *r_sloppy = r;
      r_0 = ColorSpinorField::Create(csParam);
      *r_0 = r;
    }

    if (param.precision_sloppy == x.Precision() || !param.use_sloppy_partial_accumulator) 
    {
      x_sloppy = &x;
      blas::zero(*x_sloppy);
    } 
    else 
    {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      x_sloppy = ColorSpinorField::Create(csParam);
    }

    // Syntatic sugar
    ColorSpinorField &rSloppy = *r_sloppy;
    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &r0 = *r_0;

    SolverParam solve_param_inner(param);
    fillInnerSolveParam(solve_param_inner, param);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_res = use_heavy_quark_res ? sqrt(blas::HeavyQuarkResidualNorm(x,r).z) : 0.0;
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

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
    
    if (!param.is_preconditioner) { // do not do the below if we this is an inner solver
      blas::flops = 0;    
    }

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    
    rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
    blas::copy(p, rSloppy);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) 
      printfQuda("BiCGstab debug: x2=%e, r2=%e, v2=%e, p2=%e, tmp2=%e r0=%e t2=%e\n", 
		 blas::norm2(x), blas::norm2(rSloppy), blas::norm2(v), blas::norm2(p), 
		 blas::norm2(tmp), blas::norm2(r0), blas::norm2(t));

    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && 
	    k < param.maxiter) {
    
      matSloppy(v, p, tmp);

      Complex r0v;
      if (param.pipeline) {
	r0v = blas::cDotProduct(r0, v);
	if (k>0) rho = blas::cDotProduct(r0, r);
      } else {
	r0v = blas::cDotProduct(r0, v);
      }
      if (abs(rho) == 0.0) alpha = 0.0;
      else alpha = rho / r0v;

      // r -= alpha*v
      blas::caxpy(-alpha, v, rSloppy);

      matSloppy(t, rSloppy, tmp);
    
      int updateR = 0;
      if (param.pipeline) {
	// omega = (t, r) / (t, t)
	omega_t2 = blas::cDotProductNormA(t, rSloppy);
	Complex tr = Complex(omega_t2.x, omega_t2.y);
	double t2 = omega_t2.z;
	omega = tr / t2;
	double s2 = blas::norm2(rSloppy);
	Complex r0t = blas::cDotProduct(r0, t);
	beta = -r0t / r0v;
	r2 = s2 - real(omega * conj(tr)) ;

	// now we can work out if we need to do a reliable update
        updateR = reliable(rNorm, maxrx, maxrr, r2, delta);
      } else {
	// omega = (t, r) / (t, t)
	omega_t2 = blas::cDotProductNormA(t, rSloppy);
	omega = Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);
      }

      if (param.pipeline && !updateR) {
	//x += alpha*p + omega*r, r -= omega*t, p = r - beta*omega*v + beta*p
	blas::caxpbypzYmbw(alpha, p, omega, rSloppy, xSloppy, t);
	blas::cxpaypbz(rSloppy, -beta*omega, v, beta, p);
	//tripleBiCGstabUpdate(alpha, p, omega, rSloppy, xSloppy, t, -beta*omega, v, beta, p
      } else {
	//x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
	rho_r2 = blas::caxpbypzYmbwcDotProductUYNormY(alpha, p, omega, rSloppy, xSloppy, t, r0);
	rho0 = rho;
	rho = Complex(rho_r2.x, rho_r2.y);
	r2 = rho_r2.z;
      }

      if (use_heavy_quark_res && k%heavy_quark_check==0) {
        if (&x != &xSloppy) {
           blas::copy(tmp,y);
           heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy, tmp, rSloppy).z);
        } else {
           blas::copy(r, rSloppy);
           heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(x, y, r).z);
        }
      }

      if (!param.pipeline) updateR = reliable(rNorm, maxrx, maxrr, r2, delta);

      if (updateR) {
	if (x.Precision() != xSloppy.Precision()) blas::copy(x, xSloppy);
      
	blas::xpy(x, y); // swap these around?

	mat(r, y, x);
	r2 = blas::xmyNorm(b, r);

	if (x.Precision() != rSloppy.Precision()) blas::copy(rSloppy, r);            
	blas::zero(xSloppy);

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
		   blas::norm2(x), blas::norm2(rSloppy), blas::norm2(v), blas::norm2(p), 
		   blas::norm2(tmp), blas::norm2(r0), blas::norm2(t));

      // update p
      if (!param.pipeline || updateR) {// need to update if not pipeline or did a reliable update
	if (abs(rho*alpha) == 0.0) beta = 0.0;
	else beta = (rho/rho0) * (alpha/omega);      
	blas::cxpaypbz(rSloppy, -beta*omega, v, beta, p);
      }

    }

    if (x.Precision() != xSloppy.Precision()) blas::copy(x, xSloppy);
    blas::xpy(y, x);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;

    param.gflops += gflops;
    param.iter += k;

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("BiCGstab: Reliable updates = %d\n", rUpdate);
  
    if (!param.is_preconditioner) { // do not do the below if we this is an inner solver
      // Calculate the true residual
      mat(r, x);
      param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
      param.true_res_hq = use_heavy_quark_res ? sqrt(blas::HeavyQuarkResidualNorm(x,r).z) : 0.0;
 
      PrintSummary("BiCGstab", k, r2, b2);      
    }

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    // copy the residual to b so we can use it outside of the solver
    if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO) blas::copy(b,r);

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

    profile.TPSTART(QUDA_PROFILE_FREE);
    if (param.precision_sloppy != x.Precision()) {
      delete r_0;
      delete r_sloppy;
    }
    else if(param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) 
    {
      delete r_0;
    }

    if (&x != &xSloppy) delete x_sloppy;

    profile.TPSTOP(QUDA_PROFILE_FREE);
    
    return;
  }

} // namespace quda
