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

#include <worker.h>

namespace quda {

  /**
     This worker class is used to update the shifted p and x vectors.
     These updates take place in the subsequent dslash application in
     the next iteration, while we're waiting on communication to
     complete.  This results in improved strong scaling of the
     multi-shift solver.

     Since the natrix-vector consists of multiple dslash applications,
     we partition the shifts between these successive dslash
     applicaitons for optimal communications hiding.
   */
  class ShiftUpdate : public Worker {

    cudaColorSpinorField *r;
    cudaColorSpinorField **p;
    cudaColorSpinorField **x;

    double *alpha;
    double *beta;
    double *zeta;
    double *zeta_old;

    const int j_low;
    int n_shift;

    /**
       How much to partition the shifted update.  Assuming the
       operator is (M^\dagger M), this means four applications of
       dslash for Wilson type operators and two applications for
       staggered
    */
    int n_update; 

  public:
    ShiftUpdate(cudaColorSpinorField *r, cudaColorSpinorField **p, cudaColorSpinorField **x,
		double *alpha, double *beta, double *zeta, double *zeta_old, int j_low, int n_shift) :
      r(r), p(p), x(x), alpha(alpha), beta(beta), zeta(zeta), zeta_old(zeta_old), j_low(j_low), 
      n_shift(n_shift), n_update( (r->Nspin()==4) ? 4 : 2 ) {
      
    }
    virtual ~ShiftUpdate() { }
    
    void updateNshift(int new_n_shift) { n_shift = new_n_shift; }
    void updateNupdate(int new_n_update) { n_update = 1; }
    
    // note that we can't set the stream parameter here so it is
    // ignored.  This is more of a future design direction to consider
    void apply(const cudaStream_t &stream) {      
      static int count = 0;

      // on the first call do the first half of the update
      for (int j= (count*n_shift)/n_update+1; j<=((count+1)*n_shift)/n_update && j<n_shift; j++) {
	beta[j] = beta[j_low] * zeta[j] * alpha[j] /  ( zeta_old[j] * alpha[j_low] );
	// update p[i] and x[i]
	axpyBzpcxCuda(alpha[j], *(p[j]), *(x[j]), zeta[j], *r, beta[j]);
      }
      
      if (++count == n_update) count = 0;
    }
    
  };

  // this is the Worker pointer that the dslash uses to launch the shifted updates
  namespace dslash {
    extern Worker* aux_worker;
  }  

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
      if (c1+c2 != 0.0){
        zeta[j] = c0 / (c1 + c2);
      }
      else {
        zeta[j] = 0.0;
      }
      if (zeta[j] != 0.0){
        alpha[j] = alpha[j_low] * zeta[j] / zeta_old[j];
      }
      else {
        alpha[j] = 0.0;    
      }
    }  
  }

  void MultiShiftCG::operator()(cudaColorSpinorField **x, cudaColorSpinorField &b)
  {
    profile.TPSTART(QUDA_PROFILE_INIT);

    int num_offset = param.num_offset;
    double *offset = param.offset;
 
    if (num_offset == 0) return;

    const double b2 = normCuda(b);
    // Check to see that we're not trying to invert on a zero-field source
    if(b2 == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      for(int i=0; i<num_offset; ++i){
        *(x[i]) = b;
	param.true_res_offset[i] = 0.0;
	param.true_res_hq_offset[i] = 0.0;
      }
      return;
    }
    
    // this is the limit of precision possible
    const double prec_tol = pow(10.,(-2*(int)b.Precision()+1));

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
    cudaColorSpinorField **y = reliable ? new cudaColorSpinorField*[num_offset] : NULL;
  
    ColorSpinorParam csParam(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    if (reliable)
      for (int i=0; i<num_offset; i++) y[i] = new cudaColorSpinorField(*r, csParam);

    csParam.setPrecision(param.precision_sloppy);
  
    cudaColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x[0]->Precision()) {
      r_sloppy = r;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(*r, csParam);
    }
  
    cudaColorSpinorField **x_sloppy = new cudaColorSpinorField*[num_offset];
    if (param.precision_sloppy == x[0]->Precision() ||
	!param.use_sloppy_partial_accumulator) {
      for (int i=0; i<num_offset; i++) x_sloppy[i] = x[i];
    } else {
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      for (int i=0; i<num_offset; i++)
	x_sloppy[i] = new cudaColorSpinorField(*x[i], csParam);
    }
  
    cudaColorSpinorField **p = new cudaColorSpinorField*[num_offset];  
    for (int i=0; i<num_offset; i++) p[i]= new cudaColorSpinorField(*r_sloppy);    
  
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField* Ap = new cudaColorSpinorField(*r_sloppy, csParam);
  
    cudaColorSpinorField tmp1(*Ap, csParam);

    // tmp2 only needed for multi-gpu Wilson-like kernels
    cudaColorSpinorField *tmp2_p = !mat.isStaggered() ?
      new cudaColorSpinorField(*Ap, csParam) : &tmp1;
    cudaColorSpinorField &tmp2 = *tmp2_p;

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);
    cudaColorSpinorField *tmp3_p =
      (param.precision != param.precision_sloppy && !mat.isStaggered()) ?
      new cudaColorSpinorField(*r, csParam) : &tmp1;
    cudaColorSpinorField &tmp3 = *tmp3_p;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    // stopping condition of each shift
    double stop[QUDA_MAX_MULTI_SHIFT];
    double r2[QUDA_MAX_MULTI_SHIFT];
    for (int i=0; i<num_offset; i++) {
      r2[i] = b2;
      stop[i] = Solver::stopping(param.tol_offset[i], b2, param.residual_type);
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

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease =  param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;
    
    int resIncrease = 0;
    int resIncreaseTotal[QUDA_MAX_MULTI_SHIFT];
    for (int i=0; i<num_offset; i++) {
      resIncreaseTotal[i]=0;
    }

    int k = 0;
    int rUpdate = 0;
    quda::blas_flops = 0;

    bool aux_update = false;

    // now create the worker class for updating the shifted solutions and gradient vectors
    ShiftUpdate shift_update(r_sloppy, p, x_sloppy, alpha, beta, zeta, zeta_old, j_low, num_offset_now);
    
    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    if (getVerbosity() >= QUDA_VERBOSE) 
      printfQuda("MultiShift CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0]/b2));
    
    while (r2[0] > stop[0] &&  k < param.maxiter) {

      if (aux_update) dslash::aux_worker = &shift_update;
      matSloppy(*Ap, *p[0], tmp1, tmp2);
      dslash::aux_worker = NULL;
      aux_update = false;

      // update number of shifts now instead of end of previous
      // iteration so that all shifts are updated during the dslash
      shift_update.updateNshift(num_offset_now);

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
      //fixme: with the current implementation of the reliable update it is sufficient to trigger it only for shift 0
      //fixme: The loop below is unnecessary but I don't want to delete it as we still might find a better reliable update
      int reliable_shift = -1; // this is the shift that sets the reliable_shift
      for (int j=0; j>=0; j--) {
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

	// this should trigger the shift update in the subsequent sloppy dslash
	aux_update = true;
	//shift_update.apply(0);
	//shift_update.apply(0);
	/*for (int j=1; j<num_offset_now; j++) {
	  beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
	  // update p[i] and x[i]
	  axpyBzpcxCuda(alpha[j], *p[j], *x_sloppy[j], zeta[j], *r_sloppy, beta[j]);
	  }*/
      } else {
	for (int j=0; j<num_offset_now; j++) {
	  axpyCuda(alpha[j], *p[j], *x_sloppy[j]);
	  copyCuda(*x[j], *x_sloppy[j]);
	  xpyCuda(*x[j], *y[j]);
	}

	mat(*r, *y[0], *x[0], tmp3); // here we can use x as tmp
	if (r->Nspin()==4) axpyCuda(offset[0], *y[0], *r);

	r2[0] = xmyNormCuda(b, *r);
	for (int j=1; j<num_offset_now; j++) r2[j] = zeta[j] * zeta[j] * r2[0];
	for (int j=0; j<num_offset_now; j++) zeroCuda(*x_sloppy[j]);

	copyCuda(*r_sloppy, *r);            

	// break-out check if we have reached the limit of the precision
	if (sqrt(r2[reliable_shift]) > r0Norm[reliable_shift]) { // reuse r0Norm for this
	  resIncrease++;
	  resIncreaseTotal[reliable_shift]++;
	  warningQuda("MultiShiftCG: Shift %d, updated residual %e is greater than previous residual %e (total #inc %i)", 
		      reliable_shift, sqrt(r2[reliable_shift]), r0Norm[reliable_shift], resIncreaseTotal[reliable_shift]);


	  if (resIncrease > maxResIncrease or resIncreaseTotal[reliable_shift] > maxResIncreaseTotal) break; // check if we reached the limit of our tolerancebreak;
	} else {
	  resIncrease = 0;
	}

	// explicitly restore the orthogonality of the gradient vector
	for (int j=0; j<num_offset_now; j++) {
	  double rp = reDotProductCuda(*r_sloppy, *p[j]) / (r2[0]);
	  axpyCuda(-rp, *r_sloppy, *p[j]);
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
      int converged = 0;
      for (int j=1; j<num_offset_now; j++) {
        if (zeta[j] == 0.0) {
          converged++;
          if (getVerbosity() >= QUDA_VERBOSE)
              printfQuda("MultiShift CG: Shift %d converged after %d iterations\n", j, k+1);
        } else {
	  r2[j] = zeta[j] * zeta[j] * r2[0];
	  if (r2[j] < stop[j] || sqrt(r2[j] / b2) < prec_tol) {
	    converged++;
	    if (getVerbosity() >= QUDA_VERBOSE)
	      printfQuda("MultiShift CG: Shift %d converged after %d iterations\n", j, k+1);
          }
	}
      }
      num_offset_now -= converged;

      // this ensure we do the update on any shifted systems that
      // happen to converge when the un-shifted system converges
      if ( (r2[0] <= stop[0] ||  k == param.maxiter) && aux_update == true) {
	if (getVerbosity() >= QUDA_VERBOSE) 
	  printfQuda("Convergence of unshifted system so trigger shiftUpdate\n");
	
	// set worker to do all updates at once
	shift_update.updateNupdate(1);
	shift_update.apply(0);
      }
      
      k++;

      if (getVerbosity() >= QUDA_VERBOSE) 
	printfQuda("MultiShift CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0]/b2));
    }
    
    
    for (int i=0; i<num_offset; i++) {
      copyCuda(*x[i], *x_sloppy[i]);
      if (reliable) xpyCuda(*y[i], *x[i]);
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("MultiShift CG: Reliable updates = %d\n", rUpdate);

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
      param.iter_res_offset[i] = sqrt(r2[i]/b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
      param.true_res_hq_offset[i] = sqrt(HeavyQuarkResidualNormCuda(*x[i], *r).z);
#else
      param.true_res_hq_offset[i] = 0.0;
#endif   
    }

    if (getVerbosity() >= QUDA_SUMMARIZE){
      printfQuda("MultiShift CG: Converged after %d iterations\n", k);
      for(int i=0; i < num_offset; i++) { 
	printfQuda(" shift=%d, relative residual: iterated = %e, true = %e\n", 
		   i, param.iter_res_offset[i], param.true_res_offset[i]);
      }
    }

  
    // reset the flops counters
    quda::blas_flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp1) delete tmp3_p;
    if (&tmp2 != &tmp1) delete tmp2_p;

    if (r_sloppy->Precision() != r->Precision()) delete r_sloppy;
    for (int i=0; i<num_offset; i++) 
       if (x_sloppy[i]->Precision() != x[i]->Precision()) delete x_sloppy[i];
    delete []x_sloppy;
  
    delete r;
    for (int i=0; i<num_offset; i++) delete p[i];
    delete []p;

    if (reliable) {
      for (int i=0; i<num_offset; i++) delete y[i];
      delete []y;
    }

    delete Ap;
  
    delete []zeta_old;
    delete []zeta;
    delete []alpha;
    delete []beta;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
