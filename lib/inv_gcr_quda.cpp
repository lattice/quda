#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <complex>

#include <quda_internal.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
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

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // tell inner solver it is a preconditioner

    inner.global_reduction = false;

    inner.use_init_guess = QUDA_USE_INIT_GUESS_NO;

    if (outer.inv_type == QUDA_GCR_INVERTER && outer.precision_sloppy != outer.precision_precondition) 
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;

  }

  void computeBeta(Complex **beta, std::vector<ColorSpinorField*> Ap, int i, int N, int k) {
    Complex *Beta = new Complex[N];
    std::vector<ColorSpinorField*> a(N), b(1);
    for (int j=0; j<N; j++) {
      a[j] = Ap[i+j];
      Beta[j] = 0;
    }
    b[0] = Ap[k];
    blas::cDotProduct(Beta, a, b); // vectorized dot product
#if 0
    for (int j=0; j<N; j++) {
      printfQuda("%d/%d vectorized %e %e, regular %e %e\n", j+1, N, Beta[j].real(), Beta[j].imag(),
		 blas::cDotProduct(*a[j], *b[j]).real(), blas::cDotProduct(*a[j], *b[j]).imag());
      }
#endif

    for (int j=0; j<N; j++) beta[i+j][k] = Beta[j];
    delete [] Beta;
  }

  void updateAp(Complex **beta, std::vector<ColorSpinorField*> Ap, int begin, int size, int k) {

    Complex *beta_ = new Complex[size];
    for (int i=0; i<size; i++) beta_[i] = -beta[i+begin][k];

    std::vector<ColorSpinorField*> Ap_(Ap.begin() + begin, Ap.begin() + begin + size);
    std::vector<ColorSpinorField*> Apk(Ap.begin() + k, Ap.begin() + k + 1);

    blas::caxpy(beta_, Ap_, Apk);

    delete []beta_;
  }

  void orthoDir(Complex **beta, std::vector<ColorSpinorField*> Ap, int k, int pipeline) {

    switch (pipeline) {
    case 0: // no kernel fusion
      for (int i=0; i<k; i++) { // 5 (k-1) memory transactions here
	beta[i][k] = blas::cDotProduct(*(Ap[i]), *(Ap[k]));
	blas::caxpy(-beta[i][k], *Ap[i], *Ap[k]);
      }
      break;
    case 1: // basic kernel fusion
      if (k==0) break;
      beta[0][k] = blas::cDotProduct(*Ap[0], *Ap[k]);
      for (int i=0; i<k-1; i++) { // 4 (k-1) memory transactions here
	beta[i+1][k] = blas::caxpyDotzy(-beta[i][k], *Ap[i], *Ap[k], *Ap[i+1]);
      }
      blas::caxpy(-beta[k-1][k], *Ap[k-1], *Ap[k]);
      break;
    case 2: // two-way pipelining
    case 3: // three-way pipelining
    case 4: // four-way pipelining
    case 5: // five-way pipelining
    case 6: // six-way pipelining
    case 7: // seven-way pipelining
    case 8: // eight-way pipelining
      {
	const int N = pipeline;
	for (int i=0; i<k-(N-1); i+=N) {
	  computeBeta(beta, Ap, i, N, k);
	  updateAp(beta, Ap, i, N, k);
	}

	if (k%N != 0) { // need to update the remainder
	  for (int r = N-1; r>0; r--) {
	    if ((k%N) % r == 0) { // if true this is the remainder
	      computeBeta(beta, Ap, k-r, r, k);
	      updateAp(beta, Ap, k-r, r, k);
	      break;
	    }
	  }
	}
      }
      break;
    default:
      errorQuda("Pipeline length %d type not defined", pipeline);
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

  void updateSolution(ColorSpinorField &x, const Complex *alpha, Complex** const beta, 
		      double *gamma, int k, std::vector<ColorSpinorField*> p) {

    Complex *delta = new Complex[k];

    // Update the solution vector
    backSubs(alpha, beta, gamma, delta, k);
  
    std::vector<ColorSpinorField*> X;
    X.push_back(&x);

    std::vector<ColorSpinorField*> P;
    for (int i=0; i<k; i++) P.push_back(p[i]);
    blas::caxpy(delta, P, X);

    delete []delta;
  }

  GCR::GCR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param,
	   TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(0), Kparam(param),
    nKrylov(param.Nkrylov), init(false),  rp(nullptr), yp(nullptr), tmpp(nullptr), x_sloppy(nullptr),
    r_sloppy(nullptr), r_pre(nullptr), p_pre(nullptr), rM(nullptr)
  {

    fillInnerSolveParam(Kparam, param);

    if (param.inv_type_precondition == QUDA_CG_INVERTER) // inner CG preconditioner
      K = new CG(matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
      K = new BiCGstab(matPrecon, matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_MR_INVERTER) // inner MR preconditioner
      K = new MR(matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_SD_INVERTER) // inner MR preconditioner
      K = new SD(matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_INVALID_INVERTER) // unknown preconditioner
      K = NULL;
    else 
      errorQuda("Unsupported preconditioner %d\n", param.inv_type_precondition);

    p.resize(nKrylov);
    Ap.resize(nKrylov);

    alpha = new Complex[nKrylov];
    beta = new Complex*[nKrylov];
    for (int i=0; i<nKrylov; i++) beta[i] = new Complex[nKrylov];
    gamma = new double[nKrylov];
  }

  GCR::GCR(DiracMatrix &mat, Solver &K, DiracMatrix &matSloppy, DiracMatrix &matPrecon, 
	   SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(&K), Kparam(param),
    nKrylov(param.Nkrylov), init(false), rp(nullptr), yp(nullptr), tmpp(nullptr), x_sloppy(nullptr),
    r_sloppy(nullptr), r_pre(nullptr), p_pre(nullptr), rM(nullptr)
  {
    p.resize(nKrylov);
    Ap.resize(nKrylov);

    alpha = new Complex[nKrylov];
    beta = new Complex*[nKrylov];
    for (int i=0; i<nKrylov; i++) beta[i] = new Complex[nKrylov];
    gamma = new double[nKrylov];
  }

  GCR::~GCR() {
    profile.TPSTART(QUDA_PROFILE_FREE);
    delete []alpha;
    for (int i=0; i<nKrylov; i++) delete []beta[i];
    delete []beta;
    delete []gamma;

    if (K && param.inv_type_precondition != QUDA_MG_INVERTER) delete K;

    if (param.precondition_cycle > 1) delete rM;

    if (param.precision_sloppy != param.precision) {
      if (x_sloppy) delete x_sloppy;
      if (r_sloppy) delete r_sloppy;
    }

    if (param.precision_precondition != param.precision_sloppy || param.precondition_cycle > 1) {
      if (p_pre) delete p_pre;
      if (r_pre) delete r_pre;
    }

    for (int i=0; i<nKrylov; i++) {
      if (p[i]) delete p[i];
      if (Ap[i]) delete Ap[i];
    }

    if (tmpp) delete tmpp;
    if (rp) delete rp;
    if (yp) delete yp;
    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void GCR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    profile.TPSTART(QUDA_PROFILE_INIT);

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      rp = ColorSpinorField::Create(csParam);

      // high precision accumulator
      yp = ColorSpinorField::Create(csParam);

      // create sloppy fields used for orthogonalization
      csParam.setPrecision(param.precision_sloppy);
      for (int i=0; i<nKrylov; i++) {
	p[i] = ColorSpinorField::Create(csParam);
	Ap[i] = ColorSpinorField::Create(csParam);
      }

      tmpp = ColorSpinorField::Create(csParam); //temporary for sloppy mat-vec

      if (param.precision_sloppy != param.precision) {
	csParam.setPrecision(param.precision_sloppy);
	x_sloppy = ColorSpinorField::Create(csParam);
	r_sloppy = ColorSpinorField::Create(csParam);
      } else {
	x_sloppy = &x;
	r_sloppy = rp;
      }

      // these low precision fields are used by the inner solver
      if (param.precision_precondition != param.precision_sloppy || param.precondition_cycle > 1) {
	csParam.setPrecision(param.precision_precondition);
	p_pre = ColorSpinorField::Create(csParam);
	r_pre = ColorSpinorField::Create(csParam);
      } else {
	p_pre = NULL;
	r_pre = r_sloppy;
      }

      if (param.precondition_cycle > 1) {
	ColorSpinorParam rParam(*r_sloppy);
	rM = ColorSpinorField::Create(rParam);
      }
      init = true;
    }

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;
    ColorSpinorField &rPre = *r_pre;
    ColorSpinorField &tmp = *tmpp;
    blas::zero(y);

    bool precMatch = (param.precision_precondition != param.precision_sloppy || param.precondition_cycle > 1) ? false : true;

    // compute parity of the node
    int parity = 0;
    for (int i=0; i<4; i++) parity += commCoords(i);
    parity = parity % 2;

    double b2 = blas::norm2(b);  // norm sq of source
    double r2;                // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, y);
      r2 = blas::xmyNorm(b, r);
      blas::copy(y, x);
      if (&x == &xSloppy) blas::zero(x); // need to zero x when doing uni-precision solver
    } else {
      blas::copy(r, b);
      r2 = b2;
      blas::zero(x); // defensive measure in case solution isn't already zero
      if (&x != &xSloppy) blas::zero(xSloppy);
    }

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0) {
      if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
	profile.TPSTOP(QUDA_PROFILE_INIT);
	warningQuda("inverting on zero-field source\n");
	x = b;
	param.true_res = 0.0;
	param.true_res_hq = 0.0;
	return;
      } else {
	b2 = r2;
      }
    }

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver

    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    double heavy_quark_res = 0.0; // heavy quark residual
    if(use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x,r).z);

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    blas::flops = 0;

    blas::copy(rSloppy, r);

    int total_iter = 0;
    int restart = 0;
    double r2_old = r2;
    bool l2_converge = false;

    int pipeline = param.pipeline;
    // Vectorized dot product only has limited support so work around
    if (Ap[0]->Location() == QUDA_CPU_FIELD_LOCATION || pipeline == 0) pipeline = 1;

    if (pipeline > 1)
      warningQuda("GCR with pipeline length %d is experimental", pipeline);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;
    PrintStats("GCR", total_iter+k, r2, b2, heavy_quark_res);
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && 
	    total_iter < param.maxiter) {
    
      for (int m=0; m<param.precondition_cycle; m++) {
	if (param.inv_type_precondition != QUDA_INVALID_INVERTER) {
	  ColorSpinorField &pPre = (precMatch ? *p[k] : *p_pre);
	
	  if (m==0) { // residual is just source
	    blas::copy(rPre, rSloppy);
	  } else { // compute residual
	    blas::copy(*rM, rSloppy);
	    blas::axpy(-1.0, *Ap[k], *rM);
	    blas::copy(rPre, *rM);
	  }

	  pushVerbosity(param.verbosity_precondition);
	  if ((parity+m)%2 == 0 || param.schwarz_type == QUDA_ADDITIVE_SCHWARZ) (*K)(pPre, rPre);
	  else blas::copy(pPre, rPre);
	  popVerbosity();

	  // relaxation p = omega*p + (1-omega)*r
	  //if (param.omega!=1.0) blas::axpby((1.0-param.omega), rPre, param.omega, pPre);
	
	  if (m==0) { blas::copy(*p[k], pPre); }
	  else { blas::copy(tmp, pPre); blas::xpy(tmp, *p[k]); }

	} else { // no preconditioner
	  *p[k] = rSloppy;
	}
	matSloppy(*Ap[k], *p[k], tmp);
	if (getVerbosity()>= QUDA_DEBUG_VERBOSE)
	  printfQuda("GCR debug iter=%d: Ap2=%e, p2=%e, rPre2=%e\n", 
		     total_iter, blas::norm2(*Ap[k]), blas::norm2(*p[k]), blas::norm2(rPre));
      }

      orthoDir(beta, Ap, k, pipeline);

      double3 Apr = blas::cDotProductNormA(*Ap[k], rSloppy);

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
      r2 = blas::cabxpyAxNorm(1.0/gamma[k], -alpha[k], *Ap[k], rSloppy); 

      k++;
      total_iter++;

      PrintStats("GCR", total_iter, r2, b2, heavy_quark_res);
   
      // update since nKrylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every nKrylov steps
      if (k==nKrylov || total_iter==param.maxiter || (r2 < stop && !l2_converge) || sqrt(r2/r2_old) < param.delta) {

	// update the solution vector
	updateSolution(xSloppy, alpha, beta, gamma, k, p);

	// recalculate residual in high precision
	blas::copy(x, xSloppy);
	blas::xpy(x, y);
	mat(r, y, x);
	r2 = blas::xmyNorm(b, r);  

	if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(y, r).z);

	// break-out check if we have reached the limit of the precision
	if (r2 > r2_old) {
	  resIncrease++;
	  resIncreaseTotal++;
	  warningQuda("GCR: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
		      sqrt(r2), sqrt(r2_old), resIncreaseTotal);
	  if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
	    warningQuda("GCR: solver exiting due to too many true residual norm increases");
	    break;
	  }
	} else {
	  resIncrease = 0;
	}

	k = 0;

	if ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) ) {
	  restart++; // restarting if residual is still too great

	  PrintStats("GCR (restart)", restart, r2, b2, heavy_quark_res);
	  blas::copy(rSloppy, r);
	  blas::zero(xSloppy);

	  r2_old = r2;

	  // prevent ending the Krylov space prematurely if other convergence criteria not met 
	  if (r2 < stop) l2_converge = true; 
	}

	r2_old = r2;

      }

    }

    if (total_iter > 0) blas::copy(x, y);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
  
    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops())*1e-9;
    if (K) gflops += K->flops()*1e-9;

    if (k>=param.maxiter && getVerbosity() >= QUDA_SUMMARIZE) 
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("GCR: number of restarts = %d\n", restart);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r, x, y);
      double true_res = blas::xmyNorm(b, r);
      param.true_res = sqrt(true_res / b2);
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL)
	param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x,r).z);
      else
	param.true_res_hq = 0.0;
    }

    param.gflops += gflops;
    param.iter += total_iter;

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    PrintSummary("GCR", total_iter, r2, b2);

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
