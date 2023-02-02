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
             const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matEig, param, profile), lambda_init(false), basis(param.ca_basis)
  {
  }

  CACG::~CACG()
  {
    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);
    destroyDeflationSpace();
    if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  CACGNE::CACGNE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                 const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    CACG(mmdag, mmdagSloppy, mmdagPrecon, mmdagEig, param, profile),
    mmdag(mat.Expose()),
    mmdagSloppy(matSloppy.Expose()),
    mmdagPrecon(matPrecon.Expose()),
    mmdagEig(matEig.Expose())
  {
  }

  void CACGNE::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);
    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      xp = ColorSpinorField(csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField(csParam);
      init = true;
    }
  }

  ColorSpinorField &CACGNE::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    // CG residual will match the CGNE residual (FIXME: but only with zero initial guess?)
    return param.use_init_guess ? xp : CACG::get_residual();
  }

  // CACGNE: M Mdag y = b is solved; x = Mdag y is returned as solution.
  void CACGNE::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    const int iter0 = param.iter;
    double b2 = param.compute_true_res ? blas::norm2(b) : 0.0;

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // compute initial residual
      mmdag.Expose()->M(xp, x);
      if (param.compute_true_res && b2 == 0.0)
        b2 = blas::xmyNorm(b, xp);
      else
        blas::xpay(b, -1.0, xp);

      // compute solution to residual equation
      CACG::operator()(yp, xp);

      mmdag.Expose()->Mdag(xp, yp);

      // compute full solution
      blas::xpy(xp, x);
    } else {
      CACG::operator()(yp, b);
      mmdag.Expose()->Mdag(x, yp);
    }

    if (param.compute_true_res || (param.use_init_guess && param.return_residual)) {
      // compute the true residual
      mmdag.Expose()->M(xp, x);
      blas::xpay(b, -1.0, xp); // xp now holds the residual

      double r2;
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        double3 h3 = blas::HeavyQuarkResidualNorm(x, xp);
        r2 = h3.y;
        param.true_res_hq = sqrt(h3.z);
      } else {
        r2 = blas::norm2(xp);
      }
      param.true_res = sqrt(r2 / b2);
      PrintSummary("CA-CGNE", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);
    }
  }

  CACGNR::CACGNR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                 const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    CACG(mdagm, mdagmSloppy, mdagmPrecon, mdagmEig, param, profile),
    mdagm(mat.Expose()),
    mdagmSloppy(matSloppy.Expose()),
    mdagmPrecon(matPrecon.Expose()),
    mdagmEig(matEig.Expose())
  {
  }

  void CACGNR::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);
    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      br = ColorSpinorField(csParam);
      init = true;
    }
  }

  ColorSpinorField &CACGNR::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    return br;
  }

  // CACGNR: Mdag M x = Mdag b is solved.
  void CACGNR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    const int iter0 = param.iter;
    double b2 = 0.0;
    if (param.compute_true_res) {
      b2 = blas::norm2(b);
      if (b2 == 0.0) { // compute initial residual vector
        mdagm.Expose()->M(br, x);
        b2 = blas::norm2(br);
      }
    }

    mdagm.Expose()->Mdag(br, b);
    CACG::operator()(x, br);

    if (param.compute_true_res || param.return_residual) {
      // compute the true residual
      mdagm.Expose()->M(br, x);
      blas::xpay(b, -1.0, br); // br now holds the residual

      if (param.compute_true_res) {
        double r2;
        if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
          double3 h3 = blas::HeavyQuarkResidualNorm(x, br);
          r2 = h3.y;
          param.true_res_hq = sqrt(h3.z);
        } else {
          r2 = blas::norm2(br);
        }
        param.true_res = sqrt(r2 / b2);
        PrintSummary("CA-CGNR", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);
      }
    }
  }

  void CACG::create(ColorSpinorField &x, const ColorSpinorField &b)
  {
    Solver::create(x, b);
    if (!init) {
      if (!param.is_preconditioner) {
        blas::flops = 0;
        profile.TPSTART(QUDA_PROFILE_INIT);
      }

      Q_AQandg.resize(param.Nkrylov * (param.Nkrylov + 1));
      Q_AS.resize(param.Nkrylov * param.Nkrylov);
      alpha.resize(param.Nkrylov);
      beta.resize(param.Nkrylov * param.Nkrylov);

      ColorSpinorParam csParam(b);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      if (mixed()) r = ColorSpinorField(csParam);

      // now allocate sloppy fields
      csParam.setPrecision(param.precision_sloppy);

      AS.resize(param.Nkrylov);
      Q.resize(param.Nkrylov);
      AQ.resize(param.Nkrylov);
      Qtmp.resize(param.Nkrylov); // only used as an intermediate for pointer swaps
      S.resize(param.Nkrylov);
      for (int i = 0; i < param.Nkrylov; i++) {
        AS[i] = ColorSpinorField(csParam);
        Q[i] = ColorSpinorField(csParam);
        AQ[i] = ColorSpinorField(csParam);
        Qtmp[i] = ColorSpinorField(csParam);
        // in the power basis we can alias AS[k] to S[k+1]
        S[i] = (basis == QUDA_POWER_BASIS && i > 0) ? AS[i - 1] : ColorSpinorField(csParam);
      }

      if (!mixed()) r = S[0].create_alias(csParam);

      if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_INIT);

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

  void CACG::compute_alpha()
  {
    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EIGEN);
    }

    const int N = Q.size();
    switch (N) {
#if 0 // since CA-CG is not used anywhere at the moment, no point paying for this compilation cost
      case 1: compute_alpha_N<1>(Q_AQandg, alpha); break;
      case 2: compute_alpha_N<2>(Q_AQandg, alpha); break;
      case 3: compute_alpha_N<3>(Q_AQandg, alpha); break;
      case 4: compute_alpha_N<4>(Q_AQandg, alpha); break;
      case 5: compute_alpha_N<5>(Q_AQandg, alpha); break;
      case 6: compute_alpha_N<6>(Q_AQandg, alpha); break;
      case 7: compute_alpha_N<7>(Q_AQandg, alpha); break;
      case 8: compute_alpha_N<8>(Q_AQandg, alpha); break;
      case 9: compute_alpha_N<9>(Q_AQandg, alpha); break;
      case 10: compute_alpha_N<10>(Q_AQandg, alpha); break;
      case 11: compute_alpha_N<11>(Q_AQandg, alpha); break;
      case 12: compute_alpha_N<12>(Q_AQandg, alpha); break;
#endif
    default: // failsafe
      typedef Matrix<double, Dynamic, Dynamic, RowMajor> matrix;
      typedef Matrix<double, Dynamic, 1> vector;

      const int N = Q.size();
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
      break;
    }

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_EIGEN);
      param.secs += profile.Last(QUDA_PROFILE_EIGEN);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
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

  void CACG::compute_beta()
  {
    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EIGEN);
    }

    const int N = Q.size();
    switch (N) {
#if 0 // since CA-CG is not used anywhere at the moment, no point paying for this compilation cost
      case 1: compute_beta_N<1>(Q_AQandg, Q_AS, beta); break;
      case 2: compute_beta_N<2>(Q_AQandg, Q_AS, beta); break;
      case 3: compute_beta_N<3>(Q_AQandg, Q_AS, beta); break;
      case 4: compute_beta_N<4>(Q_AQandg, Q_AS, beta); break;
      case 5: compute_beta_N<5>(Q_AQandg, Q_AS, beta); break;
      case 6: compute_beta_N<6>(Q_AQandg, Q_AS, beta); break;
      case 7: compute_beta_N<7>(Q_AQandg, Q_AS, beta); break;
      case 8: compute_beta_N<8>(Q_AQandg, Q_AS, beta); break;
      case 9: compute_beta_N<9>(Q_AQandg, Q_AS, beta); break;
      case 10: compute_beta_N<10>(Q_AQandg, Q_AS, beta); break;
      case 11: compute_beta_N<11>(Q_AQandg, Q_AS, beta); break;
      case 12: compute_beta_N<12>(Q_AQandg, Q_AS, beta); break;
#endif
    default: // failsafe
      typedef Matrix<double, Dynamic, Dynamic, RowMajor> matrix;

      const int N = Q.size();
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

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_EIGEN);
      param.secs += profile.Last(QUDA_PROFILE_EIGEN);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
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

  ColorSpinorField &CACG::get_residual()
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
  void CACG::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    if (param.is_preconditioner) commGlobalReductionPush(param.global_reduction);

    const int n_krylov = param.Nkrylov;

    if (param.maxiter == 0 || n_krylov == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    // compute b2, but only if we need to
    bool fixed_iteration = param.sloppy_converge && n_krylov == param.maxiter && !param.compute_true_res;
    double b2 = !fixed_iteration ? blas::norm2(b) : 1.0;
    double r2 = 0.0; // if zero source then we will exit immediately doing no work

    if (param.deflate) {
      // Construct the eigensolver and deflation space.
      constructDeflationSpace(b, matEig);
      if (deflate_compute) {
        // compute the deflation space.
        if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
        (*eig_solve)(evecs, evals);
        if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_PREAMBLE);
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
        profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
        profile.TPSTART(QUDA_PROFILE_INIT);
      }

      // Perform 100 power iterations, normalizing every 10 mat-vecs, using r as an initial seed
      // and Q[0]/AQ[0] as temporaries for the power iterations
      lambda_max = 1.1 * Solver::performPowerIterations(matSloppy, r, Q[0], AQ[0], 100, 10);
      logQuda(QUDA_SUMMARIZE, "CA-CG Approximate lambda max = 1.1 x %e\n", lambda_max / 1.1);

      lambda_init = true;

      if (!param.is_preconditioner) {
        profile.TPSTOP(QUDA_PROFILE_INIT);
        profile.TPSTART(QUDA_PROFILE_PREAMBLE);
      }
    }

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0) {
      if (param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
        warningQuda("inverting on zero-field source\n");
        x = b;
        param.true_res = 0.0;
        param.true_res_hq = 0.0;
        return;
      } else {
        b2 = r2;
      }
    }

    double stop = !fixed_iteration ? stopping(param.tol, b2, param.residual_type) : 0.0; // stopping condition of solver

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    double heavy_quark_res = 0.0; // heavy quark residual
    if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    if (!param.is_preconditioner) {
      blas::flops = 0;
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
    int total_iter = 0;
    double r2_old = r2;
    bool l2_converge = false;

    // Various variables related to reliable updates.
    int rUpdate = 0;             // count reliable updates.
    double delta = param.delta;  // delta for reliable updates.
    double rNorm = sqrt(r2);     // The current residual norm.
    double maxrr = rNorm;        // The maximum residual norm since the last reliable update.
    double maxr_deflate = rNorm; // The maximum residual since the last deflation

    // Factors which map linear operator onto [-1,1]
    double m_map = 2. / (lambda_max - lambda_min);
    double b_map = -(lambda_max + lambda_min) / (lambda_max - lambda_min);

    blas::copy(S[0], r); // no op if uni-precision

    PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res);
    while (!convergence(r2, heavy_quark_res, stop, param.tol_hq) && total_iter < param.maxiter) {

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
          blas::reDotProduct(Q_AS, AQ, S);
          compute_beta();

          // update direction vectors
          blas::axpyz(beta, Q, S, Qtmp);
          for (int i = 0; i < n_krylov; i++) std::swap(Q[i], Qtmp[i]);

          blas::axpyz(beta, AQ, AS, Qtmp);
          for (int i = 0; i < n_krylov; i++) std::swap(AQ[i], Qtmp[i]);
        }

        // compute the alpha coefficients
        // 1. Compute Q_AQ = Q^\dagger AQ and g = Q^dagger r = Q^dagger S[0]
        // 2. Solve Q_AQ alpha = g
        {
          blas::reDotProduct(Q_AQandg, Q, {AQ, S[0]});
          compute_alpha();
        }

        // update the solution vector
        blas::axpy(alpha, Q, x);

        for (int i = 0; i < param.Nkrylov; i++) { alpha[i] = -alpha[i]; }

        // Can we fuse these? We don't need this reduce in all cases...
        blas::axpy(alpha, AQ, S[0]);
        // if (getVerbosity() >= QUDA_VERBOSE) r2 = blas::norm2(S[0]);
        /*else*/ r2 = Q_AQandg[param.Nkrylov]; // actually the old r2... so we do one more iter than needed...
      } else {
        // fixed iterations
        // On the first pass, Q = S; AQ = AQ. We can just skip that.

        // We don't compute beta on the first iteration

        // We do compute the alpha coefficients: this is the same code as above
        // 1. Compute "Q_AQ" = S^\dagger AS and g = S^dagger r = S^dagger S[0]
        // 2. Solve "Q_AQ" alpha = g
        blas::reDotProduct(Q_AQandg, S, {AS, S[0]});
        compute_alpha();

        // update the solution vector
        blas::axpy(alpha, S, x);

        // no need to update AS
        r2 = Q_AQandg[param.Nkrylov]; // actually the old r2... so we do one more iter than needed...
      }

      // NOTE: Because we always carry around the residual from an iteration before, we
      // "lie" about which iteration we're on so the printed residual matches with the
      // printed iteration number.
      if (total_iter > 0) { PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res); }

      total_iter += n_krylov;

      // update since n_krylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every n_krylov steps
      // Note: this won't reliable update when the norm _increases_.
      if (total_iter >= param.maxiter || (r2 < stop && !l2_converge) || reliable(rNorm, maxrr, rUpdate, r2, delta)) {

        if ((r2 < stop || total_iter >= param.maxiter) && param.sloppy_converge) break;
        mat(r, x);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < maxr_deflate * param.tol_restart) {
          // Deflate and add solution to accumulator
          eig_solve->deflate(x, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, x);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2);
        }

        blas::copy(S[0], r);

        if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

        // break-out check if we have reached the limit of the precision
        if (r2 > r2_old) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda(
            "CA-CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2), sqrt(r2_old), resIncreaseTotal);
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
      double true_res = blas::xmyNorm(b, r);
      param.true_res = sqrt(true_res / b2);
      param.true_res_hq
        = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? sqrt(blas::HeavyQuarkResidualNorm(x, r).z) : 0.0;
    }

    if (!param.is_preconditioner) {
      qudaDeviceSynchronize(); // ensure solver is complete before ending timing
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

      // store flops and reset counters
      double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops() + matEig.flops()) * 1e-9;

      param.gflops += gflops;
      param.iter += total_iter;

      // reset the flops counters
      blas::flops = 0;
      mat.flops();
      matSloppy.flops();
      matPrecon.flops();
      matEig.flops();

      profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    }

    PrintSummary("CA-CG", total_iter, r2, b2, stop, param.tol_hq);

    if (param.is_preconditioner) commGlobalReductionPop();
  }

} // namespace quda
