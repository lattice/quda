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
    inner.delta = 1e-20; // no reliable updates within the inner solver
  
    inner.precision = outer.precision_sloppy;
    inner.precision_sloppy = outer.precision_precondition;

    // this sets a fixed iteration count if we're using the MR solver
    inner.residual_type = (outer.inv_type_precondition == QUDA_MR_INVERTER) ? QUDA_INVALID_RESIDUAL : QUDA_L2_RELATIVE_RESIDUAL;
  
    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // tell inner solver it is a preconditioner
    inner.pipeline = true;

    inner.schwarz_type = outer.schwarz_type;
    inner.global_reduction = inner.schwarz_type == QUDA_INVALID_SCHWARZ ? true : false;

    inner.use_init_guess = QUDA_USE_INIT_GUESS_NO;

    inner.maxiter = outer.maxiter_precondition;
    if (outer.inv_type_precondition == QUDA_CA_GCR_INVERTER) {
      inner.Nkrylov = inner.maxiter / outer.precondition_cycle;
    } else {
      inner.Nsteps = outer.precondition_cycle;
    }

    inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;

    inner.verbosity_precondition = outer.verbosity_precondition;

    inner.compute_true_res = false;
    inner.sloppy_converge = true;
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
    default:
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
		      double *gamma, int k, std::vector<ColorSpinorField*> p)
  {
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

  GCR::GCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, param, profile),
    matMdagM(DiracMdagM(matPrecon.Expose())),
    K(0),
    Kparam(param),
    nKrylov(param.Nkrylov),
    init(false),
    rp(nullptr),
    tmpp(nullptr),
    tmp_sloppy(nullptr),
    r_sloppy(nullptr)
  {
    fillInnerSolveParam(Kparam, param);

    if (param.inv_type_precondition == QUDA_CG_INVERTER) // inner CG solver
      K = new CG(matSloppy, matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab solver
      K = new BiCGstab(matSloppy, matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_MR_INVERTER) // inner MR solver
      K = new MR(matSloppy, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_SD_INVERTER) // inner SD solver
      K = new SD(matSloppy, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_CA_GCR_INVERTER) // inner CA-GCR solver
      K = new CAGCR(matSloppy, matPrecon, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_INVALID_INVERTER) // unsupported
      K = NULL;
    else 
      errorQuda("Unsupported preconditioner %d\n", param.inv_type_precondition);

    p.resize(nKrylov+1);
    Ap.resize(nKrylov);

    alpha = new Complex[nKrylov];
    beta = new Complex*[nKrylov];
    for (int i=0; i<nKrylov; i++) beta[i] = new Complex[nKrylov];
    gamma = new double[nKrylov];
  }

  GCR::GCR(const DiracMatrix &mat, Solver &K, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param,
           TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, param, profile),
    matMdagM(matPrecon.Expose()),
    K(&K),
    Kparam(param),
    nKrylov(param.Nkrylov),
    init(false),
    rp(nullptr),
    tmpp(nullptr),
    tmp_sloppy(nullptr),
    r_sloppy(nullptr)
  {
    p.resize(nKrylov+1);
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

    if (init && param.precision_sloppy != tmpp->Precision()) {
      if (r_sloppy && r_sloppy != rp) delete r_sloppy;
    }

    for (int i=0; i<nKrylov+1; i++) if (p[i]) delete p[i];
    for (int i=0; i<nKrylov; i++) if (Ap[i]) delete Ap[i];

    if (tmp_sloppy != tmpp) delete tmp_sloppy;
    if (tmpp) delete tmpp;
    if (rp) delete rp;

    destroyDeflationSpace();

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void GCR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (nKrylov == 0) {
      // Krylov space is zero-dimensional so return doing no work
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    profile.TPSTART(QUDA_PROFILE_INIT);

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      rp = (K || x.Precision() != param.precision_sloppy) ? ColorSpinorField::Create(csParam) : nullptr;

      // high precision temporary
      tmpp = ColorSpinorField::Create(csParam);

      // create sloppy fields used for orthogonalization
      csParam.setPrecision(param.precision_sloppy);
      for (int i = 0; i < nKrylov + 1; i++) p[i] = ColorSpinorField::Create(csParam);
      for (int i=0; i<nKrylov; i++) Ap[i] = ColorSpinorField::Create(csParam);

      csParam.setPrecision(param.precision_sloppy);
      if (param.precision_sloppy != x.Precision()) {
        tmp_sloppy = tmpp->CreateAlias(csParam);
      } else {
        tmp_sloppy = tmpp;
      }

      if (param.precision_sloppy != x.Precision()) {
        r_sloppy = K ? ColorSpinorField::Create(csParam) : nullptr;
      } else {
        r_sloppy = K ? rp : nullptr;
      }

      init = true;
    }

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      constructDeflationSpace(b, matMdagM);
      if (deflate_compute) {
        // compute the deflation space.
        profile.TPSTOP(QUDA_PROFILE_INIT);
        (*eig_solve)(evecs, evals);
        // double the size of the Krylov space
        extendSVDDeflationSpace();
        // populate extra memory with L/R singular vectors
        eig_solve->computeSVD(matMdagM, evecs, evals);
        profile.TPSTART(QUDA_PROFILE_INIT);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(matMdagM, evecs, evals);
        eig_solve->computeSVD(matMdagM, evecs, evals);
        recompute_evals = false;
      }
    }

    ColorSpinorField &r = rp ? *rp : *p[0];
    ColorSpinorField &rSloppy = r_sloppy ? *r_sloppy : *p[0];
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &tmpSloppy = *tmp_sloppy;

    double b2 = blas::norm2(b);  // norm sq of source
    double r2;                // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x, tmp);
      r2 = blas::xmyNorm(b, r);
      // x contains the original guess.
    } else {
      blas::copy(r, b);
      r2 = b2;
      blas::zero(x);
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate: Hardcoded to SVD. If maxiter == 1, this is a dummy solve
      eig_solve->deflateSVD(x, r, evecs, evals, true);

      // Compute r_defl = RHS - A * LHS
      mat(r, x, tmp);
      r2 = blas::xmyNorm(b, r);
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
    double maxr_deflate = sqrt(r2);
    bool l2_converge = false;

    int pipeline = param.pipeline;
    // Vectorized dot product only has limited support so work around
    if (Ap[0]->Location() == QUDA_CPU_FIELD_LOCATION || pipeline == 0) pipeline = 1;
    if (pipeline > nKrylov) pipeline = nKrylov;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;
    int k_break = 0;

    PrintStats("GCR", total_iter+k, r2, b2, heavy_quark_res);
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && total_iter < param.maxiter) {

      if (K) {
	pushVerbosity(param.verbosity_precondition);
	(*K)(*p[k], rSloppy);
	popVerbosity();
	// relaxation p = omega*p + (1-omega)*r
	//if (param.omega!=1.0) blas::axpby((1.0-param.omega), rPre, param.omega, pPre);
      }

      matSloppy(*Ap[k], *p[k], tmpSloppy);

      if (getVerbosity()>= QUDA_DEBUG_VERBOSE)
	printfQuda("GCR debug iter=%d: Ap2=%e, p2=%e, r2=%e\n",
		   total_iter, blas::norm2(*Ap[k]), blas::norm2(*p[k]), blas::norm2(rSloppy));

      orthoDir(beta, Ap, k, pipeline);

      double3 Apr = blas::cDotProductNormA(*Ap[k], K ? rSloppy : *p[k]);

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
      r2 = blas::cabxpyzAxNorm(1.0 / gamma[k], -alpha[k], *Ap[k], K ? rSloppy : *p[k], K ? rSloppy : *p[k + 1]);

      k++;
      total_iter++;

      PrintStats("GCR", total_iter, r2, b2, heavy_quark_res);
   
      // update since nKrylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every nKrylov steps
      if (k==nKrylov || total_iter==param.maxiter || (r2 < stop && !l2_converge) || sqrt(r2/r2_old) < param.delta) {

        // update the solution vector
        updateSolution(x, alpha, beta, gamma, k, p);

        if ( (r2 < stop || total_iter==param.maxiter) && param.sloppy_converge) break;
        mat(r, x, tmp);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflateSVD(x, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, x, tmp);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2);
        }

        if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

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

        k_break = k;
        k = 0;

        if ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) ) {
          restart++; // restarting if residual is still too great

          PrintStats("GCR (restart)", restart, r2, b2, heavy_quark_res);
          blas::copy(rSloppy, r);

          r2_old = r2;

          // prevent ending the Krylov space prematurely if other convergence criteria not met 
          if (r2 < stop) l2_converge = true; 
        }

        r2_old = r2;

      }

    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops() + matMdagM.flops()) * 1e-9;
    if (K) gflops += K->flops()*1e-9;

    if (k>=param.maxiter && getVerbosity() >= QUDA_SUMMARIZE) 
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("GCR: number of restarts = %d\n", restart);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r, x, tmp);
      double true_res = blas::xmyNorm(b, r);
      param.true_res = sqrt(true_res / b2);
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL)
	param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x,r).z);
      else
	param.true_res_hq = 0.0;

      if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO) blas::copy(b, r);
    } else {
      if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO) blas::copy(b, K ? rSloppy : *p[k_break]);
    }

    param.gflops += gflops;
    param.iter += total_iter;

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();
    matMdagM.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    PrintSummary("GCR", total_iter, r2, b2, stop, param.tol_hq);

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }

} // namespace quda
