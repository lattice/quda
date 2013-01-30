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

  CG::CG(DiracMatrix &mat, DiracMatrix &matSloppy, QudaInvertParam &invParam, TimeProfile &profile) :
    Solver(invParam, profile), mat(mat), matSloppy(matSloppy)
  {

  }

  CG::~CG() {

  }

  void CG::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b) 
  {
    profile[QUDA_PROFILE_INIT].Start();

    // Check to see that we're not trying to invert on a zero-field source    
    const double b2 = norm2(b);
    if(b2 == 0){
      profile[QUDA_PROFILE_INIT].Stop();
      printfQuda("Warning: inverting on zero-field source\n");
      x=b;
      invParam.true_res = 0.0;
      invParam.true_res_hq = 0.0;
      return;
    }


    cudaColorSpinorField r(b);

    ColorSpinorParam param(x);
    param.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b, param); 
  
    mat(r, x, y);
    zeroCuda(y);

    double r2 = xmyNormCuda(b, r);
  
    param.setPrecision(invParam.cuda_prec_sloppy);
    cudaColorSpinorField Ap(x, param);
    cudaColorSpinorField tmp(x, param);

    cudaColorSpinorField *tmp2_p = &tmp;
    // tmp only needed for multi-gpu Wilson-like kernels
    if (mat.Type() != typeid(DiracStaggeredPC).name() && 
	mat.Type() != typeid(DiracStaggered).name()) {
      tmp2_p = new cudaColorSpinorField(x, param);
    }
    cudaColorSpinorField &tmp2 = *tmp2_p;

    cudaColorSpinorField *x_sloppy, *r_sloppy;
    if (invParam.cuda_prec_sloppy == x.Precision()) {
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      x_sloppy = &x;
      r_sloppy = &r;
    } else {
      param.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = new cudaColorSpinorField(x, param);
      r_sloppy = new cudaColorSpinorField(r, param);
    }

    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;
    cudaColorSpinorField p(rSloppy);
    
    const bool use_heavy_quark_res = (invParam.residual_type == QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    
    profile[QUDA_PROFILE_INIT].Stop();
    profile[QUDA_PROFILE_PREAMBLE].Start();

    double r2_old;


    double stop = b2*invParam.tol*invParam.tol; // stopping condition of solver

    double heavy_quark_residual;
    if(use_heavy_quark_res) heavy_quark_residual = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
    double & distance_to_solution   =  (use_heavy_quark_res) ? heavy_quark_residual : r2;
    double & convergence_threshold  =  (use_heavy_quark_res) ? invParam.tol : stop;
    int heavy_quark_check = 10; // how often to check the heavy quark residual

    double alpha=0.0, beta=0.0;
    double pAp;
    int rUpdate = 0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = invParam.reliable_delta;

    profile[QUDA_PROFILE_PREAMBLE].Stop();
    profile[QUDA_PROFILE_COMPUTE].Start();
    quda::blas_flops = 0;

    int k=0;
    
    if (invParam.verbosity >= QUDA_VERBOSE) {
      if (use_heavy_quark_res) {
	printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n", 
		   k, r2, sqrt(r2/b2), heavy_quark_residual);
      } else {
	printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));
      }
    }

    while (distance_to_solution > convergence_threshold  && k < invParam.maxiter) {
      matSloppy(Ap, p, tmp, tmp2); // tmp as tmp
    
      pAp = reDotProductCuda(p, Ap);
      alpha = r2 / pAp;        
      r2_old = r2;

      //r2 = axpyNormCuda(-alpha, Ap, rSloppy);
      // here we are deploying the alternative beta computation 
      Complex cg_norm = axpyCGNormCuda(-alpha, Ap, rSloppy);
      r2 = real(cg_norm); // (r_new, r_new)
      double zr = imag(cg_norm); // (r_new, r_new-r_old)

      // reliable update conditions
      rNorm = sqrt(r2);
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
    
      // force a reliable update if we are within target tolerance (experimental)
      //if (distance_to_solution < convergence_threshold) updateX = 1;

      if ( !(updateR || updateX)) {
	beta = zr / r2_old; // use the stabilized beta computation
	//beta = r2 / r2_old;
	axpyZpbxCuda(alpha, p, xSloppy, rSloppy, beta);

	if (use_heavy_quark_res && k%heavy_quark_check==0) { 
	  copyCuda(tmp,y);
	  heavy_quark_residual = sqrt(xpyHeavyQuarkResidualNormCuda(xSloppy, tmp, rSloppy).z);
	}

      } else {
	axpyCuda(alpha, p, xSloppy);
	if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
      
	xpyCuda(x, y); // swap these around?
	mat(r, y, x); // here we can use x as tmp
	r2 = xmyNormCuda(b, r);

	if (x.Precision() != rSloppy.Precision()) copyCuda(rSloppy, r);            
	zeroCuda(xSloppy);

	// break-out check if we have reached the limit of the precision
	if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
	  warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e", sqrt(r2), r0Norm);
	  k++;
	  rUpdate++;
	  break;
	}

	rNorm = sqrt(r2);
	maxrr = rNorm;
	maxrx = rNorm;
	r0Norm = rNorm;      
	rUpdate++;

	// this is an experiment where we restore orthogonality of the gradient vector
	//double rp = reDotProductCuda(rSloppy, p) / (r2);
	//axpyCuda(-rp, rSloppy, p);

	beta = r2 / r2_old; 
	xpayCuda(rSloppy, beta, p);

	if(use_heavy_quark_res) heavy_quark_residual = sqrt(HeavyQuarkResidualNormCuda(y,r).z);
      }

      k++;

      if (invParam.verbosity >= QUDA_VERBOSE) {
	if (use_heavy_quark_res) {
	  printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e, heavy-quark residual = %e\n", 
		     k, r2, sqrt(r2/b2), heavy_quark_residual);
	} else {
	  printfQuda("CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2, sqrt(r2/b2));
	}
      }

    }

    if (x.Precision() != xSloppy.Precision()) copyCuda(x, xSloppy);
    xpyCuda(y, x);

    profile[QUDA_PROFILE_COMPUTE].Stop();
    profile[QUDA_PROFILE_EPILOGUE].Start();

    invParam.secs = profile[QUDA_PROFILE_COMPUTE].Last();
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
    reduceDouble(gflops);
      invParam.gflops = gflops;
    invParam.iter += k;

    if (k==invParam.maxiter) 
      warningQuda("Exceeded maximum iterations %d", invParam.maxiter);

    if (invParam.verbosity >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residuals
    mat(r, x, y);
    invParam.true_res = sqrt(xmyNormCuda(b, r) / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
    invParam.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
#else
    invParam.true_res_hq = 0.0;
#endif      

    if (invParam.verbosity >= QUDA_SUMMARIZE) {
      if (use_heavy_quark_res) {
	printfQuda("CG: Converged after %d iterations, relative residua: iterated = %e, true = %e, heavy-quark residual = %e\n", k, sqrt(r2/b2), invParam.true_res, invParam.true_res_hq);    
      }else{
	printfQuda("CG: Converged after %d iterations, relative residua: iterated = %e, true = %e\n", 
		   k, sqrt(r2/b2), invParam.true_res);
      }

    }

    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();

    profile[QUDA_PROFILE_EPILOGUE].Stop();
    profile[QUDA_PROFILE_FREE].Start();

    if (&tmp2 != &tmp) delete tmp2_p;

    if (invParam.cuda_prec_sloppy != x.Precision()) {
      delete r_sloppy;
      delete x_sloppy;
    }

    profile[QUDA_PROFILE_FREE].Stop();

    return;
  }

} // namespace quda
