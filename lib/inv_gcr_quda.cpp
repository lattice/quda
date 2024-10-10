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

  void GCR::computeBeta(std::vector<Complex> &beta, cvector_ref<ColorSpinorField> &Ap, int i, int N, int k)
  {
    std::vector<Complex> Beta(N, 0.0);
    blas::block::cDotProduct(Beta, {Ap.begin() + i, Ap.begin() + i + N}, Ap[k]); // vectorized dot product

#if 0
    for (int j=0; j<N; j++) {
      printfQuda("%d/%d vectorized %e %e, regular %e %e\n", j+1, N, Beta[j].real(), Beta[j].imag(),
		 blas::cDotProduct(a[j], b[j]).real(), blas::cDotProduct(a[j], b[j]).imag());
      }
#endif

    for (int j = 0; j < N; j++) beta[(i + j) * n_krylov + k] = Beta[j];
  }

  void GCR::updateAp(std::vector<Complex> &beta, cvector_ref<ColorSpinorField> &Ap, int begin, int size, int k)
  {
    std::vector<Complex> beta_(size);
    for (int i = 0; i < size; i++) beta_[i] = -beta[(i + begin) * n_krylov + k];
    blas::block::caxpy(beta_, {Ap.begin() + begin, Ap.begin() + begin + size}, Ap[k]);
  }

  void GCR::orthoDir(std::vector<Complex> &beta, cvector_ref<ColorSpinorField> &Ap, int k, int pipeline)
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
                           std::vector<double> &gamma, int k, cvector_ref<ColorSpinorField> &p)
  {
    std::vector<Complex> delta(k);

    // Update the solution vector
    backSubs(alpha, beta, gamma, delta, k);

    blas::block::caxpy(delta, {p.begin(), p.begin() + k}, x);
  }

  GCR::GCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param),
    matMdagM(DiracMdagM(matEig.Expose())),
    K(0),
    Kparam(param),
    n_krylov(param.Nkrylov)
  {
    fillInnerSolverParam(Kparam, param);

    // Preconditioners do not need a deflation space (for now?) so we explicily set this here.
    Kparam.deflate = false;

    K = createPreconditioner(matSloppy, matPrecon, matPrecon, matEig, param, Kparam);
  }

  GCR::GCR(const DiracMatrix &mat, Solver &K_, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
           const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param),
    matMdagM(matEig.Expose()),
    K(nullptr),
    Kparam(param),
    n_krylov(param.Nkrylov)
  {
    fillInnerSolverParam(Kparam, param);
    K = wrapExternalPreconditioner(K_);
  }

  GCR::~GCR() {
    extractInnerSolverParam(param, Kparam);
    destroyDeflationSpace();
  }

  void GCR::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      getProfile().TPSTART(QUDA_PROFILE_INIT);
      ColorSpinorParam csParam(x[0]);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      // create sloppy fields used for orthogonalization
      csParam.setPrecision(param.precision_sloppy);
      p.resize(n_krylov + 1);
      Ap.resize(n_krylov);
      for (auto &p_ : p) resize(p_, b.size(), QUDA_NULL_FIELD_CREATE, csParam);
      for (auto &ap : Ap) resize(ap, b.size(), QUDA_NULL_FIELD_CREATE, csParam);

      csParam.setPrecision(param.precision);
      if (K || mixed()) {
        resize(r, b.size(), csParam);
      } else {
        create_alias(r, p[0]);
      }

      csParam.setPrecision(param.precision_sloppy);
      if (!K) {
        create_alias(r_sloppy, p[0]);
      } else if (!mixed()) {
        create_alias(r_sloppy, r);
      } else {
        resize(r_sloppy, b.size(), csParam);
      }

      getProfile().TPSTOP(QUDA_PROFILE_INIT);

      alpha.resize(b.size());
      beta.resize(b.size());
      gamma.resize(b.size());
      for (auto i = 0u; i < b.size(); i++) {
        alpha[i].resize(n_krylov);
        beta[i].resize(n_krylov * n_krylov);
        gamma[i].resize(n_krylov);
      }

      init = true;
    }
  }

  cvector_ref<const ColorSpinorField> GCR::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (param.compute_true_res)
      return r;
    else
      return K ? r_sloppy : p[k_break];
  }

  void GCR::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    if (n_krylov == 0) {
      // Krylov space is zero-dimensional so return doing no work
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    getProfile().TPSTART(QUDA_PROFILE_INIT);
    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      if (param.eig_param.eig_type == QUDA_EIG_TR_LANCZOS || param.eig_param.eig_type == QUDA_EIG_BLK_TR_LANCZOS) {
        constructDeflationSpace(b[0], matMdagM);
      } else {
        // Use Arnoldi to inspect the space only and turn off deflation
        constructDeflationSpace(b[0], mat);
        param.deflate = false;
      }
      if (deflate_compute) {
        // compute the deflation space.
        (*eig_solve)(evecs, evals);
        if (param.deflate) {
          // double the size of the Krylov space
          extendSVDDeflationSpace();
          // populate extra memory with L/R singular vectors
          eig_solve->computeSVD(evecs, evals);
        }
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(evecs, evals);
        eig_solve->computeSVD(evecs, evals);
        recompute_evals = false;
      }
    }

    vector<double> b2 = blas::norm2(b); // norm sq of source
    vector<double> r2;                  // norm sq of residual

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
    if (is_zero_src(x, b, b2)) {
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      return;
    }

    auto stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    auto stop_hq = vector(b.size(), param.tol_hq);

    const bool use_heavy_quark_res = 
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    std::vector<double> heavy_quark_res(b.size()); // heavy quark residual
    if (use_heavy_quark_res) {
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
    }

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    getProfile().TPSTOP(QUDA_PROFILE_INIT);
    getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    blas::copy(r_sloppy, r);

    int total_iter = 0;
    int restart = 0;
    auto r2_old = r2;
    double maxr_deflate = sqrt(r2[0]);
    bool l2_converge = false;

    int pipeline = param.pipeline;
    if (pipeline > n_krylov) pipeline = n_krylov;

    getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

    int k = 0;
    k_break = 0;

    PrintStats("GCR", total_iter+k, r2, b2, heavy_quark_res);
    while (!convergence(r2, heavy_quark_res, stop, stop_hq) && total_iter < param.maxiter) {

      if (K) {
	pushVerbosity(param.verbosity_precondition);
	(*K)(p[k], r_sloppy);
	popVerbosity();
	// relaxation p = omega*p + (1-omega)*r
	//if (param.omega!=1.0) blas::axpby((1.0-param.omega), rPre, param.omega, pPre);
      }

      matSloppy(Ap[k], p[k]);

      auto get_i = [](std::vector<std::vector<ColorSpinorField>> &p, int i) {
        vector_ref<ColorSpinorField> p_i;
        p_i.reserve(p.size());
        for (auto &pi : p) p_i.push_back(pi[i]);
        return p_i;
      };

      for (auto i = 0u; i < b.size(); i++) orthoDir(beta[i], get_i(Ap, i), k, pipeline);

      auto Apr = blas::cDotProductNormA(Ap[k], K ? r_sloppy : p[k]);

      for (auto i = 0u; i < b.size(); i++) {
        gamma[i][k] = sqrt(Apr[i].z); // gamma[k] = Ap[k]
        if (gamma[i][k] == 0.0) errorQuda("GCR breakdown");
        alpha[i][k] = Complex(Apr[i].x, Apr[i].y) / gamma[i][k]; // alpha = (1/|Ap|) * (Ap, r)
      }

      // r -= (1/|Ap|^2) * (Ap, r) r, Ap *= 1/|Ap|
      std::vector<double> gamma_k_inv(b.size());
      for (auto i = 0u; i < gamma_k_inv.size(); i++) gamma_k_inv[i] = 1.0 / gamma[i][k];
      std::vector<Complex> alpha_k(b.size());
      for (auto i = 0u; i < alpha_k.size(); i++) alpha_k[i] = -alpha[i][k];
      r2 = blas::cabxpyzAxNorm(gamma_k_inv, alpha_k, Ap[k], K ? r_sloppy : p[k], K ? r_sloppy : p[k + 1]);

      k++;
      total_iter++;

      PrintStats("GCR", total_iter, r2, b2, heavy_quark_res);

      // update since n_krylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every n_krylov steps
      if (k == n_krylov || total_iter == param.maxiter || (r2[0] < stop[0] && !l2_converge)
          || sqrt(r2[0] / r2_old[0]) < param.delta) {

        // update the solution vector
        for (auto i = 0u; i < b.size(); i++) updateSolution(x[i], alpha[i], beta[i], gamma[i], k, get_i(p, i));

        if ( (r2 < stop || total_iter==param.maxiter) && param.sloppy_converge) break;
        mat(r, x);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2[0]) < maxr_deflate * param.tol_restart) {
          // Deflate: Hardcoded to SVD.
          eig_solve->deflateSVD(x, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, x);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2[0]);
        }

        if (use_heavy_quark_res) {
          auto hq = blas::HeavyQuarkResidualNorm(x, r);
          for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
        }

        // break-out check if we have reached the limit of the precision
        if (r2 > r2_old) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda(
            "GCR: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2[0]), sqrt(r2_old[0]), resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            warningQuda("GCR: solver exiting due to too many true residual norm increases");
            break;
          }
        } else {
          resIncrease = 0;
        }

        k_break = k;
        k = 0;

        if (!convergence(r2, heavy_quark_res, stop, stop_hq)) {
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

    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    getProfile().TPSTART(QUDA_PROFILE_EPILOGUE);

    if (k >= param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    logQuda(QUDA_VERBOSE, "GCR: number of restarts = %d\n", restart);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r, x);
      auto true_r2 = blas::xmyNorm(b, r);
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) {
        param.true_res[i] = sqrt(true_r2[i] / b2[i]);
        param.true_res_hq[i] = sqrt(hq[i].z);
      }
    }

    param.iter += total_iter;

    getProfile().TPSTOP(QUDA_PROFILE_EPILOGUE);

    PrintSummary("GCR", total_iter, r2, b2, stop, stop_hq);
  }

} // namespace quda
