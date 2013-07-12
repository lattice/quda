#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <face_quda.h>

/*!
 * Generic Multi Shift Solver 
 * 
 * For staggered, the mass is folded into the dirac operator
 * Otherwise the matrix mass is 'unmodified'. 
 *
 * The lowest offset is in offsets[0]
 *
 */

namespace quda {

  MultiShiftCG::MultiShiftCG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param,
			     TimeProfile &profile) 
    : MultiShiftSolver(param, profile), mat(mat), matSloppy(matSloppy) {

  }

  MultiShiftCG::~MultiShiftCG() {

  }

  /**
     Compute the new values of alpha and zeta
   */
  void updateAlphaZeta(double *alpha, double *zeta, double *zeta_old, 
		       const double *r2, const double *beta, const double pAp, 
		       const double *offset, const int nShift, const int j_low) {
    double alpha_old[QUDA_MAX_MULTI_SHIFT];
    for (int j=0; j<nShift; j++) alpha_old[j] = alpha[j];

    alpha[0] = r2[0] / pAp;        
    zeta[0] = 1.0;
    for (int j=1; j<nShift; j++) {
      double c0 = zeta[j] * zeta_old[j] * alpha_old[j_low];
      double c1 = alpha[j_low] * beta[j_low] * (zeta_old[j]-zeta[j]);
      double c2 = zeta_old[j] * alpha_old[j_low] * (1.0+(offset[j]-offset[0])*alpha[j_low]);
      
      zeta_old[j] = zeta[j];
      zeta[j] = c0 / (c1 + c2); 
      alpha[j] = alpha[j_low] * zeta[j] / zeta_old[j];
    }	
  }

  void MultiShiftCG::operator()(cudaColorSpinorField **x, cudaColorSpinorField &b)
  {
    profile.Start(QUDA_PROFILE_INIT);

    int num_offset = param.num_offset;
    double *offset = param.offset;
 
    if (num_offset == 0) return;

    const double b2 = normCuda(b);
    // Check to see that we're not trying to invert on a zero-field source
    if(b2 == 0){
      profile.Stop(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      for(int i=0; i<num_offset; ++i){
        *(x[i]) = b;
	param.true_res_offset[i] = 0.0;
	param.true_res_hq_offset[i] = 0.0;
      }
      return;
    }
    

    double *zeta = new double[num_offset];
    double *zeta_old = new double[num_offset];
    double *alpha = new double[num_offset];
    double *beta = new double[num_offset];
  
    int j_low = 0;   
    int num_offset_now = num_offset;
    for (int i=0; i<num_offset; i++) {
      zeta[i] = zeta_old[i] = 1.0;
      beta[i] = 0.0;
      alpha[i] = 1.0;
    }
  
    // flag whether we will be using reliable updates or not
    bool reliable = false;
    for (int j=0; j<num_offset; j++) 
      if (param.tol_offset[j] < param.delta) reliable = true;

    cudaColorSpinorField *r = new cudaColorSpinorField(b);
    cudaColorSpinorField *r_sloppy;
    cudaColorSpinorField **x_sloppy = new cudaColorSpinorField*[num_offset];
    cudaColorSpinorField **y = reliable ? new cudaColorSpinorField*[num_offset] : NULL;
  
    ColorSpinorParam csParam(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    if (reliable)
      for (int i=0; i<num_offset; i++) y[i] = new cudaColorSpinorField(*r, csParam);

    csParam.setPrecision(param.precision_sloppy);
  
    if (param.precision_sloppy == x[0]->Precision()) {
      for (int i=0; i<num_offset; i++){
	x_sloppy[i] = x[i];
	zeroCuda(*x_sloppy[i]);
      }
      r_sloppy = r;
    } else {
      for (int i=0; i<num_offset; i++)
	x_sloppy[i] = new cudaColorSpinorField(*x[i], csParam);
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(*r, csParam);
    }
  
    cudaColorSpinorField **p = new cudaColorSpinorField*[num_offset];  
    for (int i=0; i<num_offset; i++) p[i]= new cudaColorSpinorField(*r_sloppy);    
  
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField* Ap = new cudaColorSpinorField(*r_sloppy, csParam);
  
    cudaColorSpinorField tmp1(*Ap, csParam);
    cudaColorSpinorField *tmp2_p = &tmp1;
    // tmp only needed for multi-gpu Wilson-like kernels
    if (mat.Type() != typeid(DiracStaggeredPC).name() && 
	mat.Type() != typeid(DiracStaggered).name()) {
      tmp2_p = new cudaColorSpinorField(*Ap, csParam);
    }
    cudaColorSpinorField &tmp2 = *tmp2_p;

    profile.Stop(QUDA_PROFILE_INIT);
    profile.Start(QUDA_PROFILE_PREAMBLE);

    // stopping condition of each shift
    double stop[QUDA_MAX_MULTI_SHIFT];
    double r2[QUDA_MAX_MULTI_SHIFT];
    for (int i=0; i<num_offset; i++) {
      r2[i] = b2;
      stop[i] = r2[i] * param.tol_offset[i] * param.tol_offset[i];
    }

    double r2_old;
    double pAp;

    double rNorm[QUDA_MAX_MULTI_SHIFT];
    double r0Norm[QUDA_MAX_MULTI_SHIFT];
    double maxrx[QUDA_MAX_MULTI_SHIFT];
    double maxrr[QUDA_MAX_MULTI_SHIFT];
    for (int i=0; i<num_offset; i++) {
      rNorm[i] = sqrt(r2[i]);
      r0Norm[i] = rNorm[i];
      maxrx[i] = rNorm[i];
      maxrr[i] = rNorm[i];
    }
    double delta = param.delta;
    
    int k = 0;
    int rUpdate = 0;
    quda::blas_flops = 0;

    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);

    if (param.verbosity >= QUDA_VERBOSE) 
      printfQuda("MultiShift CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0]/b2));
    
    while (r2[0] > stop[0] &&  k < param.maxiter) {
      matSloppy(*Ap, *p[0], tmp1, tmp2);
      // FIXME - this should be curried into the Dirac operator
      if (r->Nspin()==4) axpyCuda(offset[0], *p[0], *Ap); 

      pAp = reDotProductCuda(*p[0], *Ap);

      // compute zeta and alpha
      updateAlphaZeta(alpha, zeta, zeta_old, r2, beta, pAp, offset, num_offset_now, j_low);
	
      r2_old = r2[0];
      Complex cg_norm = axpyCGNormCuda(-alpha[j_low], *Ap, *r_sloppy);
      r2[0] = real(cg_norm);
      double zn = imag(cg_norm);

      // reliable update conditions
      rNorm[0] = sqrt(r2[0]);
      for (int j=1; j<num_offset_now; j++) rNorm[j] = rNorm[0] * zeta[j];

      int updateX=0, updateR=0;
      int reliable_shift = -1; // this is the shift that sets the reliable_shift
      for (int j=num_offset_now-1; j>=0; j--) {
	if (rNorm[j] > maxrx[j]) maxrx[j] = rNorm[j];
	if (rNorm[j] > maxrr[j]) maxrr[j] = rNorm[j];
	updateX = (rNorm[j] < delta*r0Norm[j] && r0Norm[j] <= maxrx[j]) ? 1 : updateX;
	updateR = ((rNorm[j] < delta*maxrr[j] && r0Norm[j] <= maxrr[j]) || updateX) ? 1 : updateR;
	if ((updateX || updateR) && reliable_shift == -1) reliable_shift = j;
      }

      if ( !(updateR || updateX) || !reliable) {
	//beta[0] = r2[0] / r2_old;	
	beta[0] = zn / r2_old;
	// update p[0] and x[0]
	axpyZpbxCuda(alpha[0], *p[0], *x_sloppy[0], *r_sloppy, beta[0]);	

	for (int j=1; j<num_offset_now; j++) {
	  beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
	  // update p[i] and x[i]
	  axpyBzpcxCuda(alpha[j], *p[j], *x_sloppy[j], zeta[j], *r_sloppy, beta[j]);
	}
      } else {
	for (int j=0; j<num_offset_now; j++) {
	  axpyCuda(alpha[j], *p[j], *x_sloppy[j]);
	  copyCuda(*x[j], *x_sloppy[j]);
	  xpyCuda(*x[j], *y[j]);
	}

	mat(*r, *y[0], *x[0]); // here we can use x as tmp
	if (r->Nspin()==4) axpyCuda(offset[0], *y[0], *r);

	r2[0] = xmyNormCuda(b, *r);
	for (int j=1; j<num_offset_now; j++) r2[j] = zeta[j] * zeta[j] * r2[0];
	for (int j=0; j<num_offset_now; j++) zeroCuda(*x_sloppy[j]);

	copyCuda(*r_sloppy, *r);            

	// break-out check if we have reached the limit of the precision
	if (sqrt(r2[reliable_shift]) > r0Norm[reliable_shift]) { // reuse r0Norm for this
	  warningQuda("MultiShiftCG: Shift %d, updated residual %e is greater than previous residual %e", 
		      reliable_shift, sqrt(r2[reliable_shift]), r0Norm[reliable_shift]);
	  k++;
	  rUpdate++;
	  if (reliable_shift == j_low) break;
	}

	// update beta and p
	beta[0] = r2[0] / r2_old; 
	xpayCuda(*r_sloppy, beta[0], *p[0]);
	for (int j=1; j<num_offset_now; j++) {
	  beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
	  axpbyCuda(zeta[j], *r_sloppy, beta[j], *p[j]);
	}    

	// update reliable update parameters for the system that triggered the update
	int m = reliable_shift;
	rNorm[m] = sqrt(r2[0]) * zeta[m];
	maxrr[m] = rNorm[m];
	maxrx[m] = rNorm[m];
	r0Norm[m] = rNorm[m];      
	rUpdate++;
      }    

      // now we can check if any of the shifts have converged and remove them
      for (int j=1; j<num_offset_now; j++) {
	r2[j] = zeta[j] * zeta[j] * r2[0];
	if (r2[j] < stop[j]) {
	  if (param.verbosity >= QUDA_VERBOSE)
	    printfQuda("MultiShift CG: Shift %d converged after %d iterations\n", j, k+1);
	  num_offset_now--;
	}
      }

      k++;
      
      if (param.verbosity >= QUDA_VERBOSE) 
	printfQuda("MultiShift CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0]/b2));
    }
    
    
    for (int i=0; i<num_offset; i++) {
      copyCuda(*x[i], *x_sloppy[i]);
      if (reliable) xpyCuda(*y[i], *x[i]);
    }

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d\n", param.maxiter);
    
    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (quda::blas_flops + mat.flops() + matSloppy.flops())*1e-9;
    reduceDouble(gflops);
    param.gflops = gflops;
    param.iter += k;

    for(int i=0; i < num_offset; i++) { 
      mat(*r, *x[i]); 
      if (r->Nspin()==4) {
	axpyCuda(offset[i], *x[i], *r); // Offset it.
      } else if (i!=0) {
	axpyCuda(offset[i]-offset[0], *x[i], *r); // Offset it.
      }
      double true_res = xmyNormCuda(b, *r);
      param.true_res_offset[i] = sqrt(true_res/b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
      param.true_res_hq_offset[i] = sqrt(HeavyQuarkResidualNormCuda(*x[i], *r).z);
#else
      param.true_res_hq_offset[i] = 0.0;
#endif   
    }

    if (param.verbosity >= QUDA_SUMMARIZE){
      printfQuda("MultiShift CG: Converged after %d iterations\n", k);
      for(int i=0; i < num_offset; i++) { 
	printfQuda(" shift=%d, relative residua: iterated = %e, true = %e\n", 
		   i, sqrt(r2[i]/b2), param.true_res_offset[i]);
      }
    }      
  
    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);
    profile.Start(QUDA_PROFILE_FREE);

    if (&tmp2 != &tmp1) delete tmp2_p;

    delete r;
    for (int i=0; i<num_offset; i++) delete p[i];
    delete []p;

    if (reliable) {
      for (int i=0; i<num_offset; i++) delete y[i];
      delete []y;
    }

    delete Ap;
  
    if (param.precision_sloppy != x[0]->Precision()) {
      for (int i=0; i<num_offset; i++) delete x_sloppy[i];
      delete r_sloppy;
    }
    delete []x_sloppy;
  
    delete []zeta_old;
    delete []zeta;
    delete []alpha;
    delete []beta;

    profile.Stop(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
