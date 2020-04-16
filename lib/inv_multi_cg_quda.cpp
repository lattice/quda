#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

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

     Since the matrix-vector consists of multiple dslash applications,
     we partition the shifts between these successive dslash
     applications for optimal communications hiding.

     In general, when using a Worker class to hide communication in
     the dslash, one must be aware whether auto-tuning on the dslash
     policy that envelops the dslash will occur.  If so, then the
     Worker class instance will be called multiple times during this
     tuning, potentially rendering the results wrong.  This isn't a
     problem in the multi-shift solve, since we are guaranteed to not
     run the worker class on the first iteration (when the dslash
     policy tuning will take place), but this is something that will
     need to be addressed in the future as the Worker idea to applied
     elsewhere.
   */
  class ShiftUpdate : public Worker {

    ColorSpinorField *r;
    std::vector<ColorSpinorField*> p;
    std::vector<ColorSpinorField*> x;

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
    ShiftUpdate(ColorSpinorField *r, std::vector<ColorSpinorField*> p, std::vector<ColorSpinorField*> x,
                double *alpha, double *beta, double *zeta, double *zeta_old, int j_low, int n_shift) :
      r(r), p(p), x(x), alpha(alpha), beta(beta), zeta(zeta), zeta_old(zeta_old), j_low(j_low), 
      n_shift(n_shift), n_update( (r->Nspin()==4) ? 4 : 2 ) {
      
    }
    virtual ~ShiftUpdate() { }
    
    void updateNshift(int new_n_shift) { n_shift = new_n_shift; }
    void updateNupdate(int new_n_update) { n_update = new_n_update; }
    
    // note that we can't set the stream parameter here so it is
    // ignored.  This is more of a future design direction to consider
    void apply(const qudaStream_t &stream)
    {
      static int count = 0;

#if 0
      // on the first call do the first half of the update
      for (int j= (count*n_shift)/n_update+1; j<=((count+1)*n_shift)/n_update && j<n_shift; j++) {
	beta[j] = beta[j_low] * zeta[j] * alpha[j] /  ( zeta_old[j] * alpha[j_low] );
	// update p[i] and x[i]
	blas::axpyBzpcx(alpha[j], *(p[j]), *(x[j]), zeta[j], *r, beta[j]);
      }
#else
      int zero = (count*n_shift)/n_update+1;
      std::vector<ColorSpinorField*> P, X;
      for (int j= (count*n_shift)/n_update+1; j<=((count+1)*n_shift)/n_update && j<n_shift; j++) {
	beta[j] = beta[j_low] * zeta[j] * alpha[j] /  ( zeta_old[j] * alpha[j_low] );
	P.push_back(p[j]);
	X.push_back(x[j]);
      }
      if (P.size()) blas::axpyBzpcx(&alpha[zero], P, X, &zeta[zero], *r, &beta[zero]);
#endif
      if (++count == n_update) count = 0;
    }
  };

  // this is the Worker pointer that the dslash uses to launch the shifted updates
  namespace dslash {
    extern Worker* aux_worker;
  }  

  MultiShiftCG::MultiShiftCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param,
			     TimeProfile &profile) :
    MultiShiftSolver(mat, matSloppy, param, profile) { }

  MultiShiftCG::~MultiShiftCG() { }

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

  void MultiShiftCG::operator()(std::vector<ColorSpinorField*>x, ColorSpinorField &b, std::vector<ColorSpinorField*> &p, double* r2_old_array )
  {
    if (checkLocation(*(x[0]), b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    int num_offset = param.num_offset;
    double *offset = param.offset;
 
    if (num_offset == 0) return;

    const double b2 = blas::norm2(b);
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
    
    bool exit_early = false;
    bool mixed = param.precision_sloppy != param.precision;
    // whether we will switch to refinement on unshifted system after other shifts have converged
    bool zero_refinement = param.precision_refinement_sloppy != param.precision;

    // this is the limit of precision possible
    const double sloppy_tol= param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon() :
      ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon() : pow(2.,-17));
    const double fine_tol = pow(10.,(-2*(int)b.Precision()+1));
    std::unique_ptr<double[]> prec_tol(new double[num_offset]);

    prec_tol[0] = mixed ? sloppy_tol : fine_tol;
    for (int i=1; i<num_offset; i++) {
      prec_tol[i] = std::min(sloppy_tol,std::max(fine_tol,sqrt(param.tol_offset[i]*sloppy_tol)));
    }

    double zeta[QUDA_MAX_MULTI_SHIFT];
    double zeta_old[QUDA_MAX_MULTI_SHIFT];
    double alpha[QUDA_MAX_MULTI_SHIFT];
    double beta[QUDA_MAX_MULTI_SHIFT];
  
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


    auto *r = new cudaColorSpinorField(b);
    std::vector<ColorSpinorField*> x_sloppy;
    x_sloppy.resize(num_offset);
    std::vector<ColorSpinorField*> y;
  
    ColorSpinorParam csParam(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    if (reliable) {
      y.resize(num_offset);
      for (int i=0; i<num_offset; i++) y[i] = new cudaColorSpinorField(*r, csParam);
    }

    csParam.setPrecision(param.precision_sloppy);
  
    cudaColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x[0]->Precision()) {
      r_sloppy = r;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(*r, csParam);
    }
  
    if (param.precision_sloppy == x[0]->Precision() ||
	!param.use_sloppy_partial_accumulator) {
      for (int i=0; i<num_offset; i++){
	x_sloppy[i] = x[i];
	blas::zero(*x_sloppy[i]);
      }
    } else {
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      for (int i=0; i<num_offset; i++)
	x_sloppy[i] = new cudaColorSpinorField(*x[i], csParam);
    }
  
    p.resize(num_offset);
    for (int i=0; i<num_offset; i++) p[i] = new cudaColorSpinorField(*r_sloppy);    
  
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    auto* Ap = new cudaColorSpinorField(*r_sloppy, csParam);
  
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
    int iter[QUDA_MAX_MULTI_SHIFT+1];     // record how many iterations for each shift
    for (int i=0; i<num_offset; i++) {
      r2[i] = b2;
      stop[i] = Solver::stopping(param.tol_offset[i], b2, param.residual_type);
      iter[i] = 0;
    }
    // this initial condition ensures that the heaviest shift can be removed
    iter[num_offset] = 1;

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
    blas::flops = 0;

    bool aux_update = false;

    // now create the worker class for updating the shifted solutions and gradient vectors
    ShiftUpdate shift_update(r_sloppy, p, x_sloppy, alpha, beta, zeta, zeta_old, j_low, num_offset_now);
    
    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    if (getVerbosity() >= QUDA_VERBOSE) 
      printfQuda("MultiShift CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0]/b2));
    
    while ( !convergence(r2, stop, num_offset_now) &&  !exit_early && k < param.maxiter) {

      if (aux_update) dslash::aux_worker = &shift_update;
      matSloppy(*Ap, *p[0], tmp1, tmp2);
      dslash::aux_worker = nullptr;
      aux_update = false;

      // update number of shifts now instead of end of previous
      // iteration so that all shifts are updated during the dslash
      shift_update.updateNshift(num_offset_now);

      // at some point we should curry these into the Dirac operator
      if (r->Nspin()==4) pAp = blas::axpyReDot(offset[0], *p[0], *Ap);
      else pAp = blas::reDotProduct(*p[0], *Ap);

      // compute zeta and alpha
      for (int j=1; j<num_offset_now; j++) r2_old_array[j] = zeta[j] * zeta[j] * r2[0];
      updateAlphaZeta(alpha, zeta, zeta_old, r2, beta, pAp, offset, num_offset_now, j_low);
	
      r2_old = r2[0];
      r2_old_array[0] = r2_old;
      
      Complex cg_norm = blas::axpyCGNorm(-alpha[j_low], *Ap, *r_sloppy);
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
	blas::axpyZpbx(alpha[0], *p[0], *x_sloppy[0], *r_sloppy, beta[0]);	

	// this should trigger the shift update in the subsequent sloppy dslash
	aux_update = true;
	/*
	  for (int j=1; j<num_offset_now; j++) {
	  beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
	  // update p[i] and x[i]
	  blas::axpyBzpcx(alpha[j], *p[j], *x_sloppy[j], zeta[j], *r_sloppy, beta[j]);
	  }
	*/
      } else {
	for (int j=0; j<num_offset_now; j++) {
	  blas::axpy(alpha[j], *p[j], *x_sloppy[j]);
	  blas::xpy(*x_sloppy[j], *y[j]);
	}

	mat(*r, *y[0], *x[0], tmp3); // here we can use x as tmp
	if (r->Nspin()==4) blas::axpy(offset[0], *y[0], *r);

	r2[0] = blas::xmyNorm(b, *r);
	for (int j=1; j<num_offset_now; j++) r2[j] = zeta[j] * zeta[j] * r2[0];
	for (int j=0; j<num_offset_now; j++) blas::zero(*x_sloppy[j]);

	blas::copy(*r_sloppy, *r);            

	// break-out check if we have reached the limit of the precision
	if (sqrt(r2[reliable_shift]) > r0Norm[reliable_shift]) { // reuse r0Norm for this
	  resIncrease++;
	  resIncreaseTotal[reliable_shift]++;
	  warningQuda("MultiShiftCG: Shift %d, updated residual %e is greater than previous residual %e (total #inc %i)", 
		      reliable_shift, sqrt(r2[reliable_shift]), r0Norm[reliable_shift], resIncreaseTotal[reliable_shift]);

	  if (resIncrease > maxResIncrease or resIncreaseTotal[reliable_shift] > maxResIncreaseTotal) {
	    warningQuda("MultiShiftCG: solver exiting due to too many true residual norm increases");
	    break;
	  }
	} else {
	  resIncrease = 0;
	}

	// explicitly restore the orthogonality of the gradient vector
	for (int j=0; j<num_offset_now; j++) {
	  Complex rp = blas::cDotProduct(*r_sloppy, *p[j]) / (r2[0]);
	  blas::caxpy(-rp, *r_sloppy, *p[j]);
	}

	// update beta and p
	beta[0] = r2[0] / r2_old; 
	blas::xpay(*r_sloppy, beta[0], *p[0]);
	for (int j=1; j<num_offset_now; j++) {
	  beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
	  blas::axpby(zeta[j], *r_sloppy, beta[j], *p[j]);
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
      for (int j=num_offset_now-1; j>=1; j--) {
        if (zeta[j] == 0.0 && r2[j+1] < stop[j+1]) {
          converged++;
          if (getVerbosity() >= QUDA_VERBOSE)
              printfQuda("MultiShift CG: Shift %d converged after %d iterations\n", j, k+1);
        } else {
	  r2[j] = zeta[j] * zeta[j] * r2[0];
	  // only remove if shift above has converged
	  if ((r2[j] < stop[j] || sqrt(r2[j] / b2) < prec_tol[j]) && iter[j+1] ) {
	    converged++;
	    iter[j] = k+1;
	    if (getVerbosity() >= QUDA_VERBOSE)
	      printfQuda("MultiShift CG: Shift %d converged after %d iterations\n", j, k+1);
          }
	}
      }
      num_offset_now -= converged;

      // exit early so that we can finish of shift 0 using CG and allowing for mixed precison refinement
      if ( (mixed || zero_refinement) and param.compute_true_res and num_offset_now==1) {
        exit_early=true;
        num_offset_now--;
      }

      // this ensure we do the update on any shifted systems that
      // happen to converge when the un-shifted system converges
      if ( (convergence(r2, stop, num_offset_now) || exit_early || k == param.maxiter) && aux_update == true) {
	if (getVerbosity() >= QUDA_VERBOSE) 
	  printfQuda("Convergence of unshifted system so trigger shiftUpdate\n");
	
	// set worker to do all updates at once
	shift_update.updateNupdate(1);
	shift_update.apply(0);

	for (int j=0; j<num_offset_now; j++) iter[j] = k+1;
      }
      
      k++;

      if (getVerbosity() >= QUDA_VERBOSE) 
	printfQuda("MultiShift CG: %d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0]/b2));
    }
    
    for (int i=0; i<num_offset; i++) {
      if (iter[i] == 0) iter[i] = k;
      blas::copy(*x[i], *x_sloppy[i]);
      if (reliable) blas::xpy(*y[i], *x[i]);
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("MultiShift CG: Reliable updates = %d\n", rUpdate);

    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d\n", param.maxiter);
    
    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (param.compute_true_res){
      // only allocate temporaries if necessary
      csParam.setPrecision(param.precision);
      ColorSpinorField *tmp4_p = reliable ? y[0] : tmp1.Precision() == x[0]->Precision() ? &tmp1 : ColorSpinorField::Create(csParam);
      ColorSpinorField *tmp5_p = mat.isStaggered() ? tmp4_p :
      reliable ? y[1] : (tmp2.Precision() == x[0]->Precision() && &tmp1 != tmp2_p) ? tmp2_p : ColorSpinorField::Create(csParam);

      for (int i = 0; i < num_offset; i++) {
        // only calculate true residual if we need to:
        // 1.) For higher shifts if we did not use mixed precision
        // 2.) For shift 0 if we did not exit early  (we went to the full solution)
        if ( (i > 0 and not mixed) or (i == 0 and not exit_early) ) {
          mat(*r, *x[i], *tmp4_p, *tmp5_p);
          if (r->Nspin() == 4) {
            blas::axpy(offset[i], *x[i], *r); // Offset it.
          } else if (i != 0) {
            blas::axpy(offset[i] - offset[0], *x[i], *r); // Offset it.
          }
          double true_res = blas::xmyNorm(b, *r);
          param.true_res_offset[i] = sqrt(true_res / b2);
          param.iter_res_offset[i] = sqrt(r2[i] / b2);
          param.true_res_hq_offset[i] = sqrt(blas::HeavyQuarkResidualNorm(*x[i], *r).z);
        } else {
          param.iter_res_offset[i] = sqrt(r2[i] / b2);
          param.true_res_offset[i] = std::numeric_limits<double>::infinity();
          param.true_res_hq_offset[i] = std::numeric_limits<double>::infinity();
        }
      }

      if (getVerbosity() >= QUDA_SUMMARIZE) {
        printfQuda("MultiShift CG: Converged after %d iterations\n", k);
        for (int i = 0; i < num_offset; i++) {
          if(std::isinf(param.true_res_offset[i])){
            printfQuda(" shift=%d, %d iterations, relative residual: iterated = %e\n",
                       i, iter[i], param.iter_res_offset[i]);
          } else {
            printfQuda(" shift=%d, %d iterations, relative residual: iterated = %e, true = %e\n",
                       i, iter[i], param.iter_res_offset[i], param.true_res_offset[i]);
          }
        }
      }

      if (tmp5_p != tmp4_p && tmp5_p != tmp2_p && (reliable ? tmp5_p != y[1] : 1)) delete tmp5_p;
      if (tmp4_p != &tmp1 && (reliable ? tmp4_p != y[0] : 1)) delete tmp4_p;
    } else {
      if (getVerbosity() >= QUDA_SUMMARIZE)
      {
        printfQuda("MultiShift CG: Converged after %d iterations\n", k);
        for (int i = 0; i < num_offset; i++) {
          param.iter_res_offset[i] = sqrt(r2[i] / b2);
          printfQuda(" shift=%d, %d iterations, relative residual: iterated = %e\n",
          i, iter[i], param.iter_res_offset[i]);
        }
      }
    }

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp1) delete tmp3_p;
    if (&tmp2 != &tmp1) delete tmp2_p;

    if (r_sloppy->Precision() != r->Precision()) delete r_sloppy;
    for (int i=0; i<num_offset; i++) 
       if (x_sloppy[i]->Precision() != x[i]->Precision()) delete x_sloppy[i];
  
    delete r;

    if (reliable) for (int i=0; i<num_offset; i++) delete y[i];

    delete Ap;
  
    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
