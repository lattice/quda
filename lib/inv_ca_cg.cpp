#include <invert_quda.h>
#include <blas_quda.h>
#include <eigen_helper.h>
#include <solver.hpp>

/**
   @file inv_ca_cg.cpp

   Implementation of the communication-avoiding CG algorithm.  Based
   on the description here:
   http://research.nvidia.com/sites/default/files/pubs/2016-04_S-Step-and-Communication-Avoiding/nvr-2016-003.pdf
*/

namespace quda
{

  CACG::CACG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
             const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param), lambda_init(false), basis(param.ca_basis)
  {
  }

  CACG::~CACG()
  {
    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_FREE);
    destroyDeflationSpace();
    if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_FREE);
  }

  void CACG::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);
    if (!init || r.size() != b.size()) {
      if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_INIT);

      Q_AQandg.resize(b.size());
      Q_AS.resize(b.size());
      alpha.resize(b.size());
      beta.resize(b.size());
      for (auto i = 0u; i < b.size(); i++) {
        Q_AQandg[i].resize(param.Nkrylov * (param.Nkrylov + 1));
        Q_AS[i].resize(param.Nkrylov * param.Nkrylov);
        alpha[i].resize(param.Nkrylov);
        beta[i].resize(param.Nkrylov * param.Nkrylov);
      }

      ColorSpinorParam csParam(b[0]);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      if (mixed()) resize(r, b.size(), csParam);

      // now allocate sloppy fields
      csParam.setPrecision(param.precision_sloppy);

      AS.resize(param.Nkrylov);
      Q.resize(param.Nkrylov);
      AQ.resize(param.Nkrylov);
      Qtmp.resize(param.Nkrylov); // only used as an intermediate for pointer swaps
      S.resize(param.Nkrylov);
      for (int i = 0; i < param.Nkrylov; i++) {
        resize(AS[i], b.size(), csParam);
        resize(Q[i], b.size(), csParam);
        resize(AQ[i], b.size(), csParam);
        resize(Qtmp[i], b.size(), csParam);
        // in the power basis we can alias AS[k] to S[k+1]
        if (basis == QUDA_POWER_BASIS && i > 0)
          create_alias(S[i], AS[i - 1]);
        else
          resize(S[i], b.size(), csParam);
      }

      if (!mixed()) create_alias(r, S[0]);

      if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_INIT);

      init = true;
    } // init
  }

  // template!
  template <int N> void compute_alpha_N(const std::vector<double> &Q_AQandg, std::vector<double> &alpha)
  {
    typedef Matrix<double, N, N, RowMajor> matrix;
    typedef Matrix<double, N, 1> vector;

    matrix matQ_AQ(N, N);
    vector vecg(N);
    for (int i = 0; i < N; i++) {
      vecg(i) = Q_AQandg[i * (N + 1) + N];
      for (int j = 0; j < N; j++) { matQ_AQ(i, j) = Q_AQandg[i * (N + 1) + j]; }
    }
    Map<vector> vecalpha(alpha.data(), N);

    vecalpha = matQ_AQ.fullPivLu().solve(vecg);

    // JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
    // psi = svd.solve(phi);
  }

  void CACG::compute_alpha(int b)
  {
    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      getProfile().TPSTART(QUDA_PROFILE_EIGEN);
    }

    const int N = Q.size();
    switch (N) {
#if 0 // since CA-CG is not used anywhere at the moment, no point paying for this compilation cost
      case 1: compute_alpha_N<1>(Q_AQandg[b], alpha[b]); break;
      case 2: compute_alpha_N<2>(Q_AQandg[b], alpha[b]); break;
      case 3: compute_alpha_N<3>(Q_AQandg[b], alpha[b]); break;
      case 4: compute_alpha_N<4>(Q_AQandg[b], alpha[b]); break;
      case 5: compute_alpha_N<5>(Q_AQandg[b], alpha[b]); break;
      case 6: compute_alpha_N<6>(Q_AQandg[b], alpha[b]); break;
      case 7: compute_alpha_N<7>(Q_AQandg[b], alpha[b]); break;
      case 8: compute_alpha_N<8>(Q_AQandg[b], alpha[b]); break;
      case 9: compute_alpha_N<9>(Q_AQandg[b], alpha[b]); break;
      case 10: compute_alpha_N<10>(Q_AQandg[b], alpha[b]); break;
      case 11: compute_alpha_N<11>(Q_AQandg[b], alpha[b]); break;
      case 12: compute_alpha_N<12>(Q_AQandg[b], alpha[b]); break;
#endif
    default: // failsafe
      typedef Matrix<double, Dynamic, Dynamic, RowMajor> matrix;
      typedef Matrix<double, Dynamic, 1> vector;

      matrix matQ_AQ(N, N);
      vector vecg(N);
      for (int i = 0; i < N; i++) {
        vecg(i) = Q_AQandg[b][i * (N + 1) + N];
        for (int j = 0; j < N; j++) { matQ_AQ(i, j) = Q_AQandg[b][i * (N + 1) + j]; }
      }
      Map<vector> vecalpha(alpha[b].data(), N);

      vecalpha = matQ_AQ.fullPivLu().solve(vecg);

      // JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
      // psi = svd.solve(phi);
      break;
    }

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_EIGEN);
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  // template!
  template <int N>
  void compute_beta_N(const std::vector<double> &Q_AQandg, std::vector<double> &Q_AS, std::vector<double> &beta)
  {
    typedef Matrix<double, N, N, RowMajor> matrix;

    matrix matQ_AQ(N, N);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) { matQ_AQ(i, j) = Q_AQandg[i * (N + 1) + j]; }
    }
    Map<matrix> matQ_AS(Q_AS.data(), N, N), matbeta(beta.data(), N, N);

    matQ_AQ = -matQ_AQ;
    matbeta = matQ_AQ.fullPivLu().solve(matQ_AS);

    // JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
    // psi = svd.solve(phi);
  }

  void CACG::compute_beta(int b)
  {
    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      getProfile().TPSTART(QUDA_PROFILE_EIGEN);
    }

    const int N = Q.size();
    switch (N) {
#if 0 // since CA-CG is not used anywhere at the moment, no point paying for this compilation cost
      case 1: compute_beta_N<1>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 2: compute_beta_N<2>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 3: compute_beta_N<3>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 4: compute_beta_N<4>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 5: compute_beta_N<5>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 6: compute_beta_N<6>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 7: compute_beta_N<7>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 8: compute_beta_N<8>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 9: compute_beta_N<9>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 10: compute_beta_N<10>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 11: compute_beta_N<11>(Q_AQandg[b], Q_AS[b], beta[b]); break;
      case 12: compute_beta_N<12>(Q_AQandg[b], Q_AS[b], beta[b]); break;
#endif
    default: // failsafe
      typedef Matrix<double, Dynamic, Dynamic, RowMajor> matrix;

      matrix matQ_AQ(N, N);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) { matQ_AQ(i, j) = Q_AQandg[b][i * (N + 1) + j]; }
      }
      Map<matrix> matQ_AS(Q_AS[b].data(), N, N), matbeta(beta[b].data(), N, N);

      matQ_AQ = -matQ_AQ;
      matbeta = matQ_AQ.fullPivLu().solve(matQ_AS);

      // JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
      // psi = svd.solve(phi);
    }

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_EIGEN);
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  // Code to check for reliable updates
  int CACG::reliable(double &rNorm, double &maxrr, int &rUpdate, const double &r2, const double &delta)
  {
    // reliable updates
    rNorm = sqrt(r2);
    if (rNorm > maxrr) maxrr = rNorm;

    int updateR = (rNorm < delta * maxrr) ? 1 : 0;

    if (updateR) {
      rUpdate++;
      rNorm = sqrt(r2);
      maxrr = rNorm;
    }

    return updateR;
  }

  cvector_ref<const ColorSpinorField> CACG::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");

    if (param.compute_true_res) {
      // true residual was computed and left in r
      return r;
    } else if (mixed()) {
      // iterated residual needs to be converted before copy back
      r = Q[0];
      return r;
    } else {
      // return iterated residual directly
      return Q[0];
    }
  }

  /*
    The main CA-CG algorithm, which consists of three main steps:
    1. Build basis vectors q_k = A p_k for k = 1..Nkrlylov
    2. Steepest descent minmization of the residual in this basis
    3. Update solution and residual vectors
  */
  void CACG::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    if (param.is_preconditioner) commGlobalReductionPush(param.global_reduction);

    const int n_krylov = param.Nkrylov;

    if (param.maxiter == 0 || n_krylov == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);

    // compute b2, but only if we need to
    bool fixed_iteration = param.sloppy_converge && n_krylov == param.maxiter && !param.compute_true_res;
    auto b2 = !fixed_iteration ? blas::norm2(b) : vector<double>(b.size(), 1.0);
    vector<double> r2(b.size(), 0.0); // if zero source then we will exit immediately doing no work

    if (param.deflate) {
      // Construct the eigensolver and deflation space.
      constructDeflationSpace(b[0], matEig);
      if (deflate_compute) {
        // compute the deflation space.
        if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
        (*eig_solve)(evecs, evals);
        if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(evecs, evals);
        recompute_evals = false;
      }
    }

    // compute initial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x);
      // r = b - Ax0
      if (!fixed_iteration) {
        r2 = blas::xmyNorm(b, r);
      } else {
        blas::xpay(b, -1.0, r);
        r2 = b2; // dummy setting
      }
    } else {
      r2 = b2;
      blas::copy(r, b);
      blas::zero(x);
    }

    // Check to see that we're not trying to invert on a zero-field source
    if (is_zero_src(x, b, b2)) {
      if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
      return;
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate and add solution to accumulator
      eig_solve->deflate(x, r, evecs, evals, true);

      mat(r, x);
      if (!fixed_iteration) {
        r2 = blas::xmyNorm(b, r);
      } else {
        blas::xpay(b, -1.0, r);
        r2 = b2; // dummy setting
      }
    }

    // Use power iterations to approx lambda_max
    auto &lambda_min = param.ca_lambda_min;
    auto &lambda_max = param.ca_lambda_max;

    if (basis == QUDA_CHEBYSHEV_BASIS && lambda_max < lambda_min && !lambda_init) {
      if (!param.is_preconditioner) {
        getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
        getProfile().TPSTART(QUDA_PROFILE_INIT);
      }

      // Perform 100 power iterations, normalizing every 10 mat-vecs, using r as an initial seed
      // and Q[0]/AQ[0] as temporaries for the power iterations
      lambda_max = 1.1 * Solver::performPowerIterations(matSloppy, r[0], Q[0][0], AQ[0][0], 100, 10);
      logQuda(QUDA_SUMMARIZE, "CA-CG Approximate lambda max = 1.1 x %e\n", lambda_max / 1.1);

      lambda_init = true;

      if (!param.is_preconditioner) {
        getProfile().TPSTOP(QUDA_PROFILE_INIT);
        getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);
      }
    }

    auto stop = !fixed_iteration ? stopping(param.tol, b2, param.residual_type) :
                                   vector<double>(b2.size(), 0.0); // stopping condition of solver

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    vector<double> heavy_quark_res(b.size(), 0.0); // heavy quark residual
    if (use_heavy_quark_res) {
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
    }

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    }
    int total_iter = 0;
    auto r2_old = r2;
    bool l2_converge = false;

    // Various variables related to reliable updates.
    int rUpdate = 0;             // count reliable updates.
    double delta = param.delta;  // delta for reliable updates.
    double rNorm = sqrt(r2[0]);  // The current residual norm.
    double maxrr = rNorm;        // The maximum residual norm since the last reliable update.
    double maxr_deflate = rNorm; // The maximum residual since the last deflation

    // Factors which map linear operator onto [-1,1]
    double m_map = 2. / (lambda_max - lambda_min);
    double b_map = -(lambda_max + lambda_min) / (lambda_max - lambda_min);

    auto get_i = [](std::vector<std::vector<ColorSpinorField>> &p, int i) {
      vector_ref<ColorSpinorField> p_i;
      p_i.reserve(p.size());
      for (auto &pi : p) p_i.push_back(pi[i]);
      return p_i;
    };

    blas::copy(S[0], r); // no op if uni-precision

    PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res);
    while (!convergenceL2(r2, stop) && total_iter < param.maxiter) {

      // build up a space of size n_krylov, assumes S[0] is in place
      computeCAKrylovSpace(matSloppy, AS, S, n_krylov, basis, m_map, b_map);

      // we can greatly simplify the workflow for fixed iterations
      if (!fixed_iteration) {
        // first iteration, copy S and AS into Q and AQ
        if (total_iter == 0) {
          // first iteration Q = S
          for (int i = 0; i < n_krylov; i++) Q[i] = S[i];
          for (int i = 0; i < n_krylov; i++) AQ[i] = AS[i];

        } else {

          // Compute the beta coefficients for updating Q, AQ
          // 1. compute matrix Q_AS = -Q^\dagger AS
          // 2. Solve Q_AQ beta = Q_AS
          for (auto i = 0u; i < b.size(); i++) {
            auto AQi = get_i(AQ, i);
            auto Si = get_i(S, i);
            blas::block::reDotProduct(Q_AS[i], AQi, Si);
            compute_beta(i);
          }

          // update direction vectors
          for (auto i = 0u; i < b.size(); i++) {
            auto Qi = get_i(Q, i);
            auto Si = get_i(S, i);
            auto Qtmpi = get_i(Qtmp, i);
            blas::block::axpyz(beta[i], Qi, Si, Qtmpi);
          }
          for (int i = 0; i < n_krylov; i++) std::swap(Q[i], Qtmp[i]);

          for (auto i = 0u; i < b.size(); i++) {
            auto AQi = get_i(AQ, i);
            auto ASi = get_i(AS, i);
            auto Qtmpi = get_i(Qtmp, i);
            blas::block::axpyz(beta[i], AQi, ASi, Qtmpi);
          }
          for (int i = 0; i < n_krylov; i++) std::swap(AQ[i], Qtmp[i]);
        }

        // compute the alpha coefficients
        // 1. Compute Q_AQ = Q^\dagger AQ and g = Q^dagger r = Q^dagger S[0]
        // 2. Solve Q_AQ alpha = g
        {
          for (auto i = 0u; i < b.size(); i++) {
            auto Qi = get_i(Q, i);
            auto AQi = get_i(AQ, i);
            auto Si = get_i(S, i);
            blas::block::reDotProduct(Q_AQandg[i], Qi, {AQi, Si[0]});
            compute_alpha(i);
          }
        }

        // update the solution vector
        for (auto i = 0u; i < b.size(); i++) {
          auto Qi = get_i(Q, i);
          blas::block::axpy(alpha[i], Qi, x[i]);
        }

        for (auto i = 0u; i < b.size(); i++)
          for (auto j = 0; j < n_krylov; j++) { alpha[i][j] = -alpha[i][j]; }

        // Can we fuse these? We don't need this reduce in all cases...
        for (auto i = 0u; i < b.size(); i++) {
          auto AQi = get_i(AQ, i);
          auto Si = get_i(S, i);
          blas::block::axpy(alpha[i], AQi, Si[0]);
        }
        // if (getVerbosity() >= QUDA_VERBOSE) r2 = blas::norm2(S[0]);
        /*else*/
        // actually the old r2... so we do one more iter than needed...
        for (auto i = 0u; i < r2.size(); i++) r2[i] = Q_AQandg[i][param.Nkrylov];
      } else {
        // fixed iterations
        // On the first pass, Q = S; AQ = AQ. We can just skip that.

        // We don't compute beta on the first iteration

        // We do compute the alpha coefficients: this is the same code as above
        // 1. Compute "Q_AQ" = S^\dagger AS and g = S^dagger r = S^dagger S[0]
        // 2. Solve "Q_AQ" alpha = g
        for (auto i = 0u; i < b.size(); i++) {
          auto Si = get_i(S, i);
          auto ASi = get_i(AS, i);
          blas::block::reDotProduct(Q_AQandg[i], Si, {ASi, Si[0]});
          compute_alpha(i);
        }

        // update the solution vector
        for (auto i = 0u; i < b.size(); i++) {
          auto Si = get_i(S, i);
          blas::block::axpy(alpha[i], Si, x[i]);
        }

        // no need to update AS
        for (auto i = 0u; i < r2.size(); i++) r2[i] = Q_AQandg[i][param.Nkrylov];
      }

      // NOTE: Because we always carry around the residual from an iteration before, we
      // "lie" about which iteration we're on so the printed residual matches with the
      // printed iteration number.
      if (total_iter > 0) { PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res); }

      total_iter += n_krylov;

      // update since n_krylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every n_krylov steps
      // Note: this won't reliable update when the norm _increases_.
      if (total_iter >= param.maxiter || (r2 < stop && !l2_converge) || reliable(rNorm, maxrr, rUpdate, r2[0], delta)) {

        if ((r2 < stop || total_iter >= param.maxiter) && param.sloppy_converge) break;
        mat(r, x);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2[0]) < maxr_deflate * param.tol_restart) {
          // Deflate and add solution to accumulator
          eig_solve->deflate(x, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, x);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2[0]);
        }

        blas::copy(S[0], r);

        if (use_heavy_quark_res) {
          auto hq = blas::HeavyQuarkResidualNorm(x, r);
          for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
        }

        // break-out check if we have reached the limit of the precision
        if (r2 > r2_old) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda(
            "CA-CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2[0]), sqrt(r2_old[0]), resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            warningQuda("CA-CG: solver exiting due to too many true residual norm increases");
            break;
          }
        } else {
          resIncrease = 0;
        }

        r2_old = r2;
      }
    }

    if (total_iter > param.maxiter && getVerbosity() >= QUDA_SUMMARIZE)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // Print number of reliable updates.
    logQuda(QUDA_VERBOSE, "%s: Reliable updates = %d\n", "CA-CG", rUpdate);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r, x);
      auto true_res = blas::xmyNorm(b, r);
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) {
        param.true_res[i] = sqrt(true_res[i] / b2[i]);
        param.true_res_hq[i] = sqrt(hq[i].z);
      }
    }

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      param.iter += total_iter;
    }

    PrintSummary("CA-CG", total_iter, r2, b2, stop, param.tol_hq);

    if (param.is_preconditioner) commGlobalReductionPop();
  }

} // namespace quda
