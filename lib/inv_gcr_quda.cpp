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

    inner.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  
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

    inner.verbosity_precondition = outer.verbosity_precondition;

    inner.compute_true_res = false;
    inner.sloppy_converge = true;
  }

  void GCR::computeBeta(std::vector<Complex> &beta, std::vector<ColorSpinorField> &Ap, int i, int N, int k)
  {
    std::vector<Complex> Beta(N, 0.0);
    blas::cDotProduct(Beta, {Ap.begin() + i, Ap.begin() + i + N}, Ap[k]); // vectorized dot product

#if 0
    for (int j=0; j<N; j++) {
      printfQuda("%d/%d vectorized %e %e, regular %e %e\n", j+1, N, Beta[j].real(), Beta[j].imag(),
		 blas::cDotProduct(a[j], b[j]).real(), blas::cDotProduct(a[j], b[j]).imag());
      }
#endif

    for (int j = 0; j < N; j++) beta[(i + j) * n_krylov + k] = Beta[j];
  }

  void GCR::updateAp(std::vector<Complex> &beta, std::vector<ColorSpinorField> &Ap, int begin, int size, int k)
  {
    std::vector<Complex> beta_(size);
    for (int i = 0; i < size; i++) beta_[i] = -beta[(i + begin) * n_krylov + k];
    blas::caxpy(beta_, {Ap.begin() + begin, Ap.begin() + begin + size}, Ap[k]);
  }

  void GCR::orthoDir(std::vector<Complex> &beta, std::vector<ColorSpinorField> &Ap, int k, int pipeline)
  {
    switch (pipeline) {
    case 0: // no kernel fusion
      for (int i=0; i<k; i++) { // 5 (k-1) memory transactions here
        beta[i * n_krylov + k] = blas::cDotProduct(Ap[i], Ap[k]);
        blas::caxpy(-beta[i * n_krylov + k], Ap[i], Ap[k]);
      }
      break;
    case 1: // basic kernel fusion
      if (k==0) break;
      beta[0 * n_krylov + k] = blas::cDotProduct(Ap[0], Ap[k]);
      for (int i=0; i<k-1; i++) { // 4 (k-1) memory transactions here
        beta[(i + 1) * n_krylov + k] = blas::caxpyDotzy(-beta[i * n_krylov + k], Ap[i], Ap[k], Ap[i + 1]);
      }
      blas::caxpy(-beta[(k - 1) * n_krylov + k], Ap[k - 1], Ap[k]);
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

  void GCR::backSubs(const std::vector<Complex> &alpha, const std::vector<Complex> &beta,
                     const std::vector<double> &gamma, std::vector<Complex> &delta, int n)
  {
    for (int k=n-1; k>=0;k--) {
      delta[k] = alpha[k];
      for (int j = k + 1; j < n; j++) { delta[k] -= beta[k * n_krylov + j] * delta[j]; }
      delta[k] /= gamma[k];
    }
  }

  void GCR::updateSolution(ColorSpinorField &x, const std::vector<Complex> &alpha, const std::vector<Complex> &beta,
                           std::vector<double> &gamma, int k, std::vector<ColorSpinorField> &p)
  {
    std::vector<Complex> delta(k);

    // Update the solution vector
    backSubs(alpha, beta, gamma, delta, k);

    blas::caxpy(delta, {p.begin(), p.begin() + k}, x);
  }

  GCR::GCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matEig, param, profile),
    matMdagM(DiracMdagM(matEig.Expose())),
    K(0),
    Kparam(param),
    n_krylov(param.Nkrylov),
    alpha(n_krylov),
    beta(n_krylov * n_krylov),
    gamma(n_krylov)
  {
    fillInnerSolveParam(Kparam, param);

    if (param.inv_type_precondition == QUDA_CG_INVERTER) // inner CG solver
      K = new CG(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab solver
      K = new BiCGstab(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_MR_INVERTER) // inner MR solver
      K = new MR(matSloppy, matPrecon, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_SD_INVERTER) // inner SD solver
      K = new SD(matSloppy, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_CA_GCR_INVERTER) // inner CA-GCR solver
      K = new CAGCR(matSloppy, matPrecon, matPrecon, matEig, Kparam, profile);
    else if (param.inv_type_precondition == QUDA_INVALID_INVERTER) // unsupported
      K = NULL;
    else 
      errorQuda("Unsupported preconditioner %d\n", param.inv_type_precondition);
  }

  GCR::GCR(const DiracMatrix &mat, Solver &K, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matEig, param, profile),
    matMdagM(matEig.Expose()),
    K(&K),
    Kparam(param),
    n_krylov(param.Nkrylov),
    alpha(n_krylov),
    beta(n_krylov * n_krylov),
    gamma(n_krylov)
  {
  }

  GCR::~GCR() {
    profile.TPSTART(QUDA_PROFILE_FREE);
    if (K && param.inv_type_precondition != QUDA_MG_INVERTER) delete K;
    destroyDeflationSpace();
    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void GCR::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);

    if (!init) {
      profile.TPSTART(QUDA_PROFILE_INIT);
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      // create sloppy fields used for orthogonalization
      csParam.setPrecision(param.precision_sloppy);
      resize(p, n_krylov + 1, QUDA_NULL_FIELD_CREATE, csParam);
      resize(Ap, n_krylov, QUDA_NULL_FIELD_CREATE, csParam);

      csParam.setPrecision(param.precision);
      if (K || mixed()) {
        r = ColorSpinorField(csParam);
      } else {
        r = p[0].create_alias();
      }

      csParam.setPrecision(param.precision_sloppy);
      if (!K) {
        r_sloppy = p[0].create_alias();
      } else {
        r_sloppy = mixed() ? ColorSpinorField(csParam) : r.create_alias();
      }

      profile.TPSTOP(QUDA_PROFILE_INIT);
      init = true;
    }
  }

  void GCR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (n_krylov == 0) {
      // Krylov space is zero-dimensional so return doing no work
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    profile.TPSTART(QUDA_PROFILE_INIT);
    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      if (param.eig_param.eig_type == QUDA_EIG_TR_LANCZOS || param.eig_param.eig_type == QUDA_EIG_BLK_TR_LANCZOS) {
        constructDeflationSpace(b, matMdagM);
      } else {
        // Use Arnoldi to inspect the space only and turn off deflation
        constructDeflationSpace(b, mat);
        param.deflate = false;
      }
      if (deflate_compute) {
        // compute the deflation space.
        profile.TPSTOP(QUDA_PROFILE_INIT);
        (*eig_solve)(evecs, evals);
        if (param.deflate) {
          // double the size of the Krylov space
          extendSVDDeflationSpace();
          // populate extra memory with L/R singular vectors
          eig_solve->computeSVD(evecs, evals);
        }
        profile.TPSTART(QUDA_PROFILE_INIT);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(evecs, evals);
        eig_solve->computeSVD(evecs, evals);
        recompute_evals = false;
      }
    }

    double b2 = blas::norm2(b);  // norm sq of source
    double r2;                // norm sq of residual

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x);
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
      mat(r, x);
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

    blas::copy(r_sloppy, r);

    int total_iter = 0;
    int restart = 0;
    double r2_old = r2;
    double maxr_deflate = sqrt(r2);
    bool l2_converge = false;

    int pipeline = param.pipeline;
    // Vectorized dot product only has limited support so work around
    if (Ap[0].Location() == QUDA_CPU_FIELD_LOCATION || pipeline == 0) pipeline = 1;
    if (pipeline > n_krylov) pipeline = n_krylov;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;
    int k_break = 0;

    PrintStats("GCR", total_iter+k, r2, b2, heavy_quark_res);
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && total_iter < param.maxiter) {

      if (K) {
	pushVerbosity(param.verbosity_precondition);
	(*K)(p[k], r_sloppy);
	popVerbosity();
	// relaxation p = omega*p + (1-omega)*r
	//if (param.omega!=1.0) blas::axpby((1.0-param.omega), rPre, param.omega, pPre);
      }

      matSloppy(Ap[k], p[k]);

      orthoDir(beta, Ap, k, pipeline);

      double3 Apr = blas::cDotProductNormA(Ap[k], K ? r_sloppy : p[k]);

      gamma[k] = sqrt(Apr.z); // gamma[k] = Ap[k]
      if (gamma[k] == 0.0) errorQuda("GCR breakdown");
      alpha[k] = Complex(Apr.x, Apr.y) / gamma[k]; // alpha = (1/|Ap|) * (Ap, r)

      // r -= (1/|Ap|^2) * (Ap, r) r, Ap *= 1/|Ap|
      r2 = blas::cabxpyzAxNorm(1.0 / gamma[k], -alpha[k], Ap[k], K ? r_sloppy : p[k], K ? r_sloppy : p[k + 1]);

      k++;
      total_iter++;

      PrintStats("GCR", total_iter, r2, b2, heavy_quark_res);

      // update since n_krylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every n_krylov steps
      if (k == n_krylov || total_iter == param.maxiter || (r2 < stop && !l2_converge) || sqrt(r2 / r2_old) < param.delta) {

        // update the solution vector
        updateSolution(x, alpha, beta, gamma, k, p);

        if ( (r2 < stop || total_iter==param.maxiter) && param.sloppy_converge) break;
        mat(r, x);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < maxr_deflate * param.tol_restart) {
          // Deflate: Hardcoded to SVD.
          eig_solve->deflateSVD(x, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, x);
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
          blas::copy(r_sloppy, r);

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

    if (k >= param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    logQuda(QUDA_VERBOSE, "GCR: number of restarts = %d\n", restart);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r, x);
      double true_res = blas::xmyNorm(b, r);
      param.true_res = sqrt(true_res / b2);
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL)
	param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x,r).z);
      else
	param.true_res_hq = 0.0;
      //if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO) blas::copy(b, r);
    } else {
      // reuse this when we add the get_residual method to GCR
      if (0) blas::copy(b, K ? r_sloppy : p[k_break]);
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
  }

} // namespace quda
