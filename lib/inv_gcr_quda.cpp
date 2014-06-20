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

#include <sys/time.h>

namespace quda {

  double timeInterval(struct timeval start, struct timeval end) {
    long ds = end.tv_sec - start.tv_sec;
    long dus = end.tv_usec - start.tv_usec;
    return ds + 0.000001*dus;
  }

  // set the required parameters for the inner solver
  void fillInnerSolveParam(SolverParam &inner, const SolverParam &outer) {
    inner.tol = outer.tol_precondition;
    inner.maxiter = outer.maxiter_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver
  
    inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;
  
    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_GCR_INVERTER; // used to tell the inner solver it is an inner solver

    if (outer.inv_type == QUDA_GCR_INVERTER && outer.precision_sloppy != outer.precision_precondition) 
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;

  }

  void orthoDir(Complex **beta, cudaColorSpinorField *Ap[], int k) {
    int type = 1;

    switch (type) {
    case 0: // no kernel fusion
      for (int i=0; i<k; i++) { // 5 (k-1) memory transactions here
	beta[i][k] = cDotProductCuda(*Ap[i], *Ap[k]);
	caxpyCuda(-beta[i][k], *Ap[i], *Ap[k]);
      }
      break;
    case 1: // basic kernel fusion
      if (k==0) break;
      beta[0][k] = cDotProductCuda(*Ap[0], *Ap[k]);
      for (int i=0; i<k-1; i++) { // 4 (k-1) memory transactions here
	beta[i+1][k] = caxpyDotzyCuda(-beta[i][k], *Ap[i], *Ap[k], *Ap[i+1]);
      }
      caxpyCuda(-beta[k-1][k], *Ap[k-1], *Ap[k]);
      break;
    case 2: // 
      for (int i=0; i<k-2; i+=3) { // 5 (k-1) memory transactions here
	for (int j=i; j<i+3; j++) beta[j][k] = cDotProductCuda(*Ap[j], *Ap[k]);
	caxpbypczpwCuda(-beta[i][k], *Ap[i], -beta[i+1][k], *Ap[i+1], -beta[i+2][k], *Ap[i+2], *Ap[k]);
      }
    
      if (k%3 != 0) { // need to update the remainder
	if ((k - 3*(k/3)) % 2 == 0) {
	  beta[k-2][k] = cDotProductCuda(*Ap[k-2], *Ap[k]);
	  beta[k-1][k] = cDotProductCuda(*Ap[k-1], *Ap[k]);
	  caxpbypzCuda(beta[k-2][k], *Ap[k-2], beta[k-1][k], *Ap[k-1], *Ap[k]);
	} else {
	  beta[k-1][k] = cDotProductCuda(*Ap[k-1], *Ap[k]);
	  caxpyCuda(beta[k-1][k], *Ap[k-1], *Ap[k]);
	}
      }

      break;
    case 3:
      for (int i=0; i<k-1; i+=2) {
	for (int j=i; j<i+2; j++) beta[j][k] = cDotProductCuda(*Ap[j], *Ap[k]);
	caxpbypzCuda(-beta[i][k], *Ap[i], -beta[i+1][k], *Ap[i+1], *Ap[k]);
      }
    
      if (k%2 != 0) { // need to update the remainder
	beta[k-1][k] = cDotProductCuda(*Ap[k-1], *Ap[k]);
	caxpyCuda(beta[k-1][k], *Ap[k-1], *Ap[k]);
      }
      break;
    default:
      errorQuda("Orthogonalization type not defined");
    }

  }   

  void backSubs(const Complex *alpha, Complex** const beta, const double *gamma, Complex *delta, int n) {
    for (int k=n-1; k>=0;k--) {
      delta[k] = alpha[k];
      for (int j=k+1;j<n; j++) {
	delta[k] -= beta[k][j]*delta[j];
      }
      delta[k] /= gamma[k];
    }
  }

  void updateSolution(cudaColorSpinorField &x, const Complex *alpha, Complex** const beta, 
		      double *gamma, int k, cudaColorSpinorField *p[]) {

    Complex *delta = new Complex[k];

    // Update the solution vector
    backSubs(alpha, beta, gamma, delta, k);
  
    //for (int i=0; i<k; i++) caxpyCuda(delta[i], *p[i], x);
  
    for (int i=0; i<k-2; i+=3) 
      caxpbypczpwCuda(delta[i], *p[i], delta[i+1], *p[i+1], delta[i+2], *p[i+2], x); 
  
    if (k%3 != 0) { // need to update the remainder
      if ((k - 3*(k/3)) % 2 == 0) caxpbypzCuda(delta[k-2], *p[k-2], delta[k-1], *p[k-1], x);
      else caxpyCuda(delta[k-1], *p[k-1], x);
    }

    delete []delta;
  }

  GCR::GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param,
	   TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0), Kparam(param)
  {

    fillInnerSolveParam(Kparam, param);

    if (param.inv_type_precondition == QUDA_CG_INVERTER) // inner CG preconditioner
      K = new CG(matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
      K = new BiCGstab(matPrecon, matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_MR_INVERTER) // inner MR preconditioner
      K = new MR(matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_SD_INVERTER) // inner MR preconditioner
      K = new SD(matPrecon, Kparam, profile);
    else if (param.inv_type_precondition != QUDA_INVALID_INVERTER) // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);

  }

  /*
  GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon,
      SolverParam &param, TimeProfile &profile);

  */

  GCR::~GCR() {
    profile.Start(QUDA_PROFILE_FREE);

    if (K) delete K;

    profile.Stop(QUDA_PROFILE_FREE);
  }

  void GCR::operator()(cudaColorSpinorField &x, cudaColorSpinorField &b)
  {
    profile.Start(QUDA_PROFILE_INIT);

    int Nkrylov = param.Nkrylov; // size of Krylov space

    ColorSpinorParam csParam(x);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField r(x, csParam); 
    cudaColorSpinorField y(x, csParam); // high precision accumulator

    // create sloppy fields used for orthogonalization
    csParam.setPrecision(param.precision_sloppy);
    cudaColorSpinorField **p = new cudaColorSpinorField*[Nkrylov];
    cudaColorSpinorField **Ap = new cudaColorSpinorField*[Nkrylov];
    for (int i=0; i<Nkrylov; i++) {
      p[i] = new cudaColorSpinorField(x, csParam);
      Ap[i] = new cudaColorSpinorField(x, csParam);
    }

    cudaColorSpinorField tmp(x, csParam); //temporary for sloppy mat-vec

    cudaColorSpinorField *x_sloppy, *r_sloppy;
    if (param.precision_sloppy != param.precision) {
      csParam.setPrecision(param.precision_sloppy);
      x_sloppy = new cudaColorSpinorField(x, csParam);
      r_sloppy = new cudaColorSpinorField(x, csParam);
    } else {
      x_sloppy = &x;
      r_sloppy = &r;
    }

    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;

    // these low precision fields are used by the inner solver
    bool precMatch = true;
    cudaColorSpinorField *r_pre, *p_pre;
    if (param.precision_precondition != param.precision_sloppy || param.precondition_cycle > 1) {
      csParam.setPrecision(param.precision_precondition);
      p_pre = new cudaColorSpinorField(x, csParam);
      r_pre = new cudaColorSpinorField(x, csParam);
      precMatch = false;
    } else {
      p_pre = NULL;
      r_pre = r_sloppy;
    }
    cudaColorSpinorField &rPre = *r_pre;

    cudaColorSpinorField *rM = param.precondition_cycle > 1 ? new cudaColorSpinorField(rSloppy) : 0;

    Complex *alpha = new Complex[Nkrylov];
    Complex **beta = new Complex*[Nkrylov];
    for (int i=0; i<Nkrylov; i++) beta[i] = new Complex[Nkrylov];
    double *gamma = new double[Nkrylov];

    // compute parity of the node
    int parity = 0;
    for (int i=0; i<4; i++) parity += commCoords(i);
    parity = parity % 2;

    double b2 = normCuda(b);  // norm sq of source
    double r2;                // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, y);
      r2 = xmyNormCuda(b, r);
      copyCuda(y, x);
      if (&x == &xSloppy) zeroCuda(x); // need to zero x when doing uni-precision solver
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

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_res = 0.0; // heavy quark residual
    if(use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(x,r).z);

    profile.Stop(QUDA_PROFILE_INIT);
    profile.Start(QUDA_PROFILE_PREAMBLE);

    blas_flops = 0;

    copyCuda(rSloppy, r);

    int total_iter = 0;
    int restart = 0;
    double r2_old = r2;
    bool l2_converge = false;

    profile.Stop(QUDA_PROFILE_PREAMBLE);
    profile.Start(QUDA_PROFILE_COMPUTE);

    int k = 0;
    PrintStats("GCR", total_iter+k, r2, b2, heavy_quark_res);
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && 
	    total_iter < param.maxiter) {
    
      for (int m=0; m<param.precondition_cycle; m++) {
	if (param.inv_type_precondition != QUDA_INVALID_INVERTER) {
	  cudaColorSpinorField &pPre = (precMatch ? *p[k] : *p_pre);
	
	  if (m==0) { // residual is just source
	    copyCuda(rPre, rSloppy);
	  } else { // compute residual
	    copyCuda(*rM, rSloppy);
	    axpyCuda(-1.0, *Ap[k], *rM);
	    copyCuda(rPre, *rM);
	  }
	
	  if ((parity+m)%2 == 0 || param.schwarz_type == QUDA_ADDITIVE_SCHWARZ) (*K)(pPre, rPre);
	  else copyCuda(pPre, rPre);
	
	  // relaxation p = omega*p + (1-omega)*r
	  //if (param.omega!=1.0) axpbyCuda((1.0-param.omega), rPre, param.omega, pPre);
	
	  if (m==0) { copyCuda(*p[k], pPre); }
	  else { copyCuda(tmp, pPre); xpyCuda(tmp, *p[k]); }

	} else { // no preconditioner
	  *p[k] = rSloppy;
	} 
      
	matSloppy(*Ap[k], *p[k], tmp);
	if (getVerbosity()>= QUDA_DEBUG_VERBOSE)
	  printfQuda("GCR debug iter=%d: Ap2=%e, p2=%e, rPre2=%e\n", total_iter, norm2(*Ap[k]), norm2(*p[k]), norm2(rPre));
      }

      orthoDir(beta, Ap, k);

      double3 Apr = cDotProductNormACuda(*Ap[k], rSloppy);

      if (getVerbosity()>= QUDA_DEBUG_VERBOSE) {
	printfQuda("GCR debug iter=%d: Apr=(%e,%e,%e)\n", total_iter, Apr.x, Apr.y, Apr.z);
	for (int i=0; i<k; i++)
	  for (int j=0; j<=k; j++)
	    printfQuda("GCR debug iter=%d: beta[%d][%d] = (%e,%e)\n", 
		       total_iter, i, j, real(beta[i][j]), imag(beta[i][j]));
      }

      gamma[k] = sqrt(Apr.z); // gamma[k] = Ap[k]
      if (gamma[k] == 0.0) errorQuda("GCR breakdown\n");
      alpha[k] = Complex(Apr.x, Apr.y) / gamma[k]; // alpha = (1/|Ap|) * (Ap, r)

      // r -= (1/|Ap|^2) * (Ap, r) r, Ap *= 1/|Ap|
      r2 = cabxpyAxNormCuda(1.0/gamma[k], -alpha[k], *Ap[k], rSloppy); 

      k++;
      total_iter++;

      PrintStats("GCR", total_iter, r2, b2, heavy_quark_res);
   
      // update since Nkrylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every Nkrylov steps
      if (k==Nkrylov || total_iter==param.maxiter || (r2 < stop && !l2_converge) || r2/r2_old < param.delta) { 

	// update the solution vector
	updateSolution(xSloppy, alpha, beta, gamma, k, p);

	// recalculate residual in high precision
	copyCuda(x, xSloppy);
	xpyCuda(x, y);
	mat(r, y, x);
	r2 = xmyNormCuda(b, r);  

	if (use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNormCuda(y, r).z);

	k = 0;

	if ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) ) {
	  restart++; // restarting if residual is still too great

	  PrintStats("GCR (restart)", restart, r2, b2, heavy_quark_res);
	  copyCuda(rSloppy, r);
	  zeroCuda(xSloppy);

	  r2_old = r2;

	  // prevent ending the Krylov space prematurely if other convergence criteria not met 
	  if (r2 < stop) l2_converge = true; 
	}

      }

    }

    if (total_iter > 0) copyCuda(x, y);

    profile.Stop(QUDA_PROFILE_COMPUTE);
    profile.Start(QUDA_PROFILE_EPILOGUE);

    param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
  
    double gflops = (blas_flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    reduceDouble(gflops);

    if (k>=param.maxiter && getVerbosity() >= QUDA_SUMMARIZE) 
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("GCR: number of restarts = %d\n", restart);
  
    // Calculate the true residual
    mat(r, x);
    double true_res = xmyNormCuda(b, r);
    param.true_res = sqrt(true_res / b2);
#if (__COMPUTE_CAPABILITY__ >= 200)
    param.true_res_hq = sqrt(HeavyQuarkResidualNormCuda(x,r).z);
#else
    param.true_res_hq = 0.0;
#endif   

    param.gflops += gflops;
    param.iter += total_iter;
  
    // reset the flops counters
    blas_flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile.Stop(QUDA_PROFILE_EPILOGUE);
    profile.Start(QUDA_PROFILE_FREE);

    PrintSummary("GCR", total_iter, r2, b2);

    if (param.precondition_cycle > 1) delete rM;

    if (param.precision_sloppy != param.precision) {
      delete x_sloppy;
      delete r_sloppy;
    }

    if (param.precision_precondition != param.precision_sloppy || param.precondition_cycle > 1) {
      delete p_pre;
      delete r_pre;
    }

    for (int i=0; i<Nkrylov; i++) {
      delete p[i];
      delete Ap[i];
    }
    delete[] p;
    delete[] Ap;

    delete []alpha;
    for (int i=0; i<Nkrylov; i++) delete []beta[i];
    delete []beta;
    delete []gamma;

    profile.Stop(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
