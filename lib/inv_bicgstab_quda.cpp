#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

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

  BiCGstab::BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), init(false) {

  }

  BiCGstab::~BiCGstab() {
    profile.Start(QUDA_PROFILE_FREE);

    if(init) {
      if (wp && wp != pp) delete wp;
      if (zp && zp != pp) delete zp;
      delete yp;
      delete rp;
      delete pp;
      delete vp;
      delete tmpp;
      delete tp;
    }

    profile.Stop(QUDA_PROFILE_FREE);
  }

  void BiCGstab::operator()(ColorSpinorField &x, ColorSpinorField &b) 
  {
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");

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

      // MR preconditioner - we need extra vectors
      if (param.inv_type_precondition == QUDA_MR_INVERTER) {
	wp = new cudaColorSpinorField(x, csParam);
	zp = new cudaColorSpinorField(x, csParam);
      } else { // dummy assignments
	wp = pp;
	zp = pp;
      }

      init = true;
    }

    ColorSpinorField &y = *yp;
    ColorSpinorField &r = *rp; 
    ColorSpinorField &p = *pp;
    ColorSpinorField &v = *vp;
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &t = *tp;
    ColorSpinorField &w = *wp;
    ColorSpinorField &z = *zp;

    ColorSpinorField *x_sloppy, *r_sloppy, *r_0;

    double b2; // norm sq of source
    double r2; // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, y);
      r2 = blas::xmyNorm(b, r);
      b2 = blas::norm2(b);
      blas::copy(y, x);
    } else {
      blas::copy(r, b);
      r2 = blas::norm2(b);
      b2 = r2;
    }

    // Check to see that we're not trying to invert on a zero-field source
    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    // set field aliasing according to whether we are doing mixed precision or not
    if (param.precision_sloppy == x.Precision()) {
      x_sloppy = &static_cast<cudaColorSpinorField&>(x);
      r_sloppy = &r;
      r_0 = &static_cast<cudaColorSpinorField&>(b);
      blas::zero(*x_sloppy);
    } else {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);
      x_sloppy = new cudaColorSpinorField(x, csParam);
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(r, csParam);
      r_0 = new cudaColorSpinorField(b, csParam);
    }

    // Syntatic sugar
    ColorSpinorField &rSloppy = *r_sloppy;
    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &r0 = *r_0;

    SolverParam solve_param_inner(param);
    fillInnerSolveParam(solve_param_inner, param);

    double stop = b2*param.tol*param.tol; // stopping condition of solver

    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_res = use_heavy_quark_res ? sqrt(blas::HeavyQuarkResidualNorm(x,r).z) : 0.0;
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
      blas::flops = 0;    
    }

    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);

    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && 
	    k < param.maxiter) {
    
      if (k==0) {
	rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
	blas::copy(p, rSloppy);
      } else {
	if (abs(rho*alpha) == 0.0) beta = 0.0;
	else beta = (rho/rho0) * (alpha/omega);

	blas::cxpaypbz(rSloppy, -beta*omega, v, beta, p);
      }
    
      if (param.inv_type_precondition == QUDA_MR_INVERTER) {
	errorQuda("Temporary disabled");
	//invertMRCuda(*matPrecon, w, p, &invert_param_inner);
	matSloppy(v, w, tmp);
      } else {
	matSloppy(v, p, tmp);
      }

      if (abs(rho) == 0.0) alpha = 0.0;
      else alpha = rho / blas::cDotProduct(r0, v);

      // r -= alpha*v
      blas::caxpy(-alpha, v, rSloppy);

      if (param.inv_type_precondition == QUDA_MR_INVERTER) {
	errorQuda("Temporary disabled");
	//invertMRCuda(*matPrecon, z, rSloppy, &invert_param_inner);
	matSloppy(t, z, tmp);
      } else {
	matSloppy(t, rSloppy, tmp);
      }
    
      // omega = (t, r) / (t, t)
      omega_t2 = blas::cDotProductNormA(t, rSloppy);
      omega = quda::Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);

      if (param.inv_type_precondition == QUDA_MR_INVERTER) {
	//x += alpha*w + omega*z, r -= omega*t, r2 = (r,r), rho = (r0, r)
	blas::caxpy(alpha, w, xSloppy);
	blas::caxpy(omega, z, xSloppy);
	blas::caxpy(-omega, t, rSloppy);
	rho_r2 = blas::cDotProductNormB(r0, rSloppy);
      } else {
	//x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
	rho_r2 = blas::caxpbypzYmbwcDotProductUYNormY(alpha, p, omega, rSloppy, xSloppy, t, r0);
      }

      rho0 = rho;
      rho = quda::Complex(rho_r2.x, rho_r2.y);
      r2 = rho_r2.z;

      if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	blas::copy(tmp,y);
	heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy, tmp, rSloppy).z);
      }

      // reliable updates
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
      //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
      int updateR = (rNorm < delta*maxrr) ? 1 : 0;

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
    }

    if (x.Precision() != xSloppy.Precision()) blas::copy(x, xSloppy);
    blas::xpy(y, x);

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    reduceDouble(gflops);

    param.gflops += gflops;
    param.iter += k;

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (param.verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: Reliable updates = %d\n", rUpdate);
  
    if (param.inv_type_precondition != QUDA_GCR_INVERTER) { // do not do the below if we this is an inner solver
      // Calculate the true residual
      mat(r, x);
      param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x,r).z);
#else
      param.true_res_hq = 0.0;
#endif
 
      PrintSummary("BiCGstab", k, r2, b2);      
    }

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);

    profile.Start(QUDA_PROFILE_FREE);
    if (param.precision_sloppy != x.Precision()) {
      delete r_0;
      delete r_sloppy;
      delete x_sloppy;
    }
    profile.Stop(QUDA_PROFILE_FREE);
    
    return;
  }

} // namespace quda
