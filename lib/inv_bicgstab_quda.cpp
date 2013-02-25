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
  void fillInnerInvertParam(QudaInvertParam &inner, const QudaInvertParam &outer);

  double resNorm(const DiracMatrix &mat, cudaColorSpinorField &b, cudaColorSpinorField &x) {  
    cudaColorSpinorField r(b);
    mat(r, x);
    return xmyNormCuda(b, r);
  }


  BiCGstab::BiCGstab(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, QudaInvertParam &invParam, TimeProfile &profile) :
    Solver(invParam, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), init(false) {

  }

  BiCGstab::~BiCGstab() {
    profile[QUDA_PROFILE_FREE].Start();

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

    profile[QUDA_PROFILE_FREE].Stop();
  }

  void BiCGstab::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {
    profile[QUDA_PROFILE_PREAMBLE].Start();

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = new cudaColorSpinorField(x, csParam);
      rp = new cudaColorSpinorField(x, csParam); 
      csParam.setPrecision(invParam.cuda_prec_sloppy);
      pp = new cudaColorSpinorField(x, csParam);
      vp = new cudaColorSpinorField(x, csParam);
      tmpp = new cudaColorSpinorField(x, csParam);
      tp = new cudaColorSpinorField(x, csParam);

      // MR preconditioner - we need extra vectors
      if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
	wp = new cudaColorSpinorField(x, csParam);
	zp = new cudaColorSpinorField(x, csParam);
      } else { // dummy assignments
	wp = pp;
	zp = pp;
      }

      init = true;
    }

    cudaColorSpinorField &y = *yp;
    cudaColorSpinorField &r = *rp; 
    cudaColorSpinorField &p = *pp;
    cudaColorSpinorField &v = *vp;
    cudaColorSpinorField &tmp = *tmpp;
    cudaColorSpinorField &t = *tp;
    cudaColorSpinorField &w = *wp;
    cudaColorSpinorField &z = *zp;

    cudaColorSpinorField *x_sloppy, *r_sloppy, *r_0;

    double b2; // norm sq of source
    double r2; // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (invParam.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, y);
      r2 = xmyNormCuda(b, r);
      b2 = normCuda(b);
      copyCuda(y, x);
    } else {
      copyCuda(r, b);
      r2 = normCuda(b);
      b2 = r2;
    }

    // Check to see that we're not trying to invert on a zero-field source
    if(b2 == 0){
      profile[QUDA_PROFILE_INIT].Stop();
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      invParam.true_res = 0.0;
      invParam.true_res_hq = 0.0;
      return;
    }

    // set field aliasing according to whether we are doing mixed precision or not
    if (invParam.cuda_prec_sloppy == x.Precision()) {
      x_sloppy = &x;
      r_sloppy = &r;
      r_0 = &b;
      zeroCuda(*x_sloppy);
    } else {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(invParam.cuda_prec_sloppy);
      x_sloppy = new cudaColorSpinorField(x, csParam);
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(r, csParam);
      r_0 = new cudaColorSpinorField(b, csParam);
    }

    // Syntatic sugar
    cudaColorSpinorField &rSloppy = *r_sloppy;
    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &r0 = *r_0;

    QudaInvertParam invert_param_inner = newQudaInvertParam();
    fillInnerInvertParam(invert_param_inner, invParam);

    double stop = b2*invParam.tol*invParam.tol; // stopping condition of solver

    const bool use_heavy_quark_res = (invParam.residual_type == QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_residual = use_heavy_quark_res ? sqrt(HeavyQuarkResidualNormCuda(x,r).z) : 0.0;
    double & distance_to_solution   =  (use_heavy_quark_res) ? heavy_quark_residual : r2;
    double & convergence_threshold  =  (use_heavy_quark_res) ? invParam.tol : stop;
    int heavy_quark_check = 10; // how often to check the heavy quark residual

    double delta = invParam.reliable_delta;

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

    if (invParam.verbosity >= QUDA_VERBOSE) {
      if (use_heavy_quark_res) {
	printfQuda("BiCGstab: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n", 
		   k, r2, sqrt(r2/b2), heavy_quark_residual);
      } else {
	printfQuda("BiCGstab: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));
      }
    }
    
    if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) { // do not do the below if we this is an inner solver
      quda::blas_flops = 0;    
    }

    profile[QUDA_PROFILE_PREAMBLE].Stop();
    profile[QUDA_PROFILE_COMPUTE].Start();

    while (distance_to_solution > convergence_threshold  && k < invParam.maxiter) {
    
      if (k==0) {
	rho = r2; // cDotProductCuda(r0, r_sloppy); // BiCRstab
	copyCuda(p, rSloppy);
      } else {
	if (abs(rho*alpha) == 0.0) beta = 0.0;
	else beta = (rho/rho0) * (alpha/omega);

	cxpaypbzCuda(rSloppy, -beta*omega, v, beta, p);
      }
    
      if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
	errorQuda("Temporary disabled");
	//invertMRCuda(*matPrecon, w, p, &invert_param_inner);
	matSloppy(v, w, tmp);
      } else {
	matSloppy(v, p, tmp);
      }

      if (abs(rho) == 0.0) alpha = 0.0;
      else alpha = rho / cDotProductCuda(r0, v);

      // r -= alpha*v
      caxpyCuda(-alpha, v, rSloppy);

      if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
	errorQuda("Temporary disabled");
	//invertMRCuda(*matPrecon, z, rSloppy, &invert_param_inner);
	matSloppy(t, z, tmp);
      } else {
	matSloppy(t, rSloppy, tmp);
      }
    
      // omega = (t, r) / (t, t)
      omega_t2 = cDotProductNormACuda(t, rSloppy);
      omega = quda::Complex(omega_t2.x / omega_t2.z, omega_t2.y / omega_t2.z);

      if (invParam.inv_type_precondition == QUDA_MR_INVERTER) {
	//x += alpha*w + omega*z, r -= omega*t, r2 = (r,r), rho = (r0, r)
	caxpyCuda(alpha, w, xSloppy);
	caxpyCuda(omega, z, xSloppy);
	caxpyCuda(-omega, t, rSloppy);
	rho_r2 = cDotProductNormBCuda(r0, rSloppy);
      } else {
	//x += alpha*p + omega*r, r -= omega*t, r2 = (r,r), rho = (r0, r)
	rho_r2 = caxpbypzYmbwcDotProductUYNormYCuda(alpha, p, omega, rSloppy, xSloppy, t, r0);
      }

      rho0 = rho;
      rho = quda::Complex(rho_r2.x, rho_r2.y);
      r2 = rho_r2.z;

      if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	copyCuda(tmp,y);
	heavy_quark_residual = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
      }

      // reliable updates
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      //int updateR = (rNorm < delta*maxrr && r0Norm <= maxrr) ? 1 : 0;
      //int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
    
      int updateR = (rNorm < delta*maxrr) ? 1 : 0;

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

      if (invParam.verbosity >= QUDA_VERBOSE) {
	if (use_heavy_quark_res) {
	  printfQuda("BiCGstab: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n", 
		     k, r2, sqrt(r2/b2), heavy_quark_residual);
	} else {
	  printfQuda("BiCGstab: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));
	}
      }

    }

    if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
    xpyCuda(y, x);

    profile[QUDA_PROFILE_COMPUTE].Stop();
    profile[QUDA_PROFILE_EPILOGUE].Start();

    invParam.secs += profile[QUDA_PROFILE_COMPUTE].Last();
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    reduceDouble(gflops);

    invParam.gflops += gflops;
    invParam.iter += k;

    if (k==invParam.maxiter) warningQuda("Exceeded maximum iterations %d", invParam.maxiter);

    if (invParam.verbosity >= QUDA_VERBOSE) printfQuda("BiCGstab: Reliable updates = %d\n", rUpdate);
  
    if (invParam.inv_type_precondition != QUDA_GCR_INVERTER) { // do not do the below if we this is an inner solver
      // Calculate the true residual
      mat(r, x);
      invParam.true_res = sqrt(xmyNormCuda(b, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
      invParam.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
#else
    invParam.true_res_hq = 0.0;
#endif
 
      if (invParam.verbosity >= QUDA_SUMMARIZE) {
	if (use_heavy_quark_res) {
	  printfQuda("BiCGstab: Converged after %d iterations, relative residua: iterated = %e, true = %e, heavy-quark residual = %e\n", k, sqrt(r2/b2), invParam.true_res, invParam.true_res_hq);    
	} else {
	  printfQuda("BiCGstab: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
		     k, sqrt(r2/b2), invParam.true_res);
	}
      }
      
    }

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile[QUDA_PROFILE_EPILOGUE].Stop();

    profile[QUDA_PROFILE_FREE].Start();
    if (invParam.cuda_prec_sloppy != x.Precision()) {
      delete r_0;
      delete r_sloppy;
      delete x_sloppy;
    }
    profile[QUDA_PROFILE_FREE].Stop();
    
    return;
  }

} // namespace quda
