#include <invert_quda.h>
#include <blas_quda.h>
#include <eigen_helper.h>
#include <solver.hpp>

namespace quda {

  CAGCR::CAGCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
               const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matEig, param, profile),
    matMdagM(matEig.Expose()),
    init(false),
    use_source(param.preserve_source == QUDA_PRESERVE_SOURCE_NO && param.precision == param.precision_sloppy
               && param.use_init_guess == QUDA_USE_INIT_GUESS_NO && !param.deflate),
    lambda_init(false),
    basis(param.ca_basis),
    alpha(nullptr),
    rp(nullptr),
    tmpp(nullptr),
    tmp_sloppy(nullptr)
  {
  }

  CAGCR::~CAGCR() {
    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {
      if (alpha) delete []alpha;
      if (basis == QUDA_POWER_BASIS) {
        for (int i=0; i<param.Nkrylov+1; i++) if (i>0 || !use_source) delete p[i];
      } else {
        for (int i=0; i<param.Nkrylov; i++) if (i>0 || !use_source) delete p[i];
        for (int i=0; i<param.Nkrylov; i++) delete q[i];
      }
      if (tmp_sloppy) delete tmp_sloppy;
      if (tmpp) delete tmpp;
      if (rp) delete rp;
    }

    destroyDeflationSpace();

    if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void CAGCR::create(ColorSpinorField &b)
  {
    if (!init) {
      if (!param.is_preconditioner) {
        blas::flops = 0;
        profile.TPSTART(QUDA_PROFILE_INIT);
      }

      alpha = new Complex[param.Nkrylov];

      bool mixed = param.precision != param.precision_sloppy;

      ColorSpinorParam csParam(b);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      // Source needs to be preserved if we're computing the true residual
      rp = (mixed && !use_source) ? ColorSpinorField::Create(csParam) : nullptr;
      tmpp = ColorSpinorField::Create(csParam);

      // now allocate sloppy fields
      csParam.setPrecision(param.precision_sloppy);

      if (basis == QUDA_POWER_BASIS) {
        // in power basis q[k] = p[k+1], so we don't need a separate q array
        p.resize(param.Nkrylov+1);
        q.resize(param.Nkrylov);
        for (int i=0; i<param.Nkrylov+1; i++) {
          p[i] = (i==0 && use_source) ? &b : ColorSpinorField::Create(csParam);
          if (i>0) q[i-1] = p[i];
        }
      } else {
        p.resize(param.Nkrylov);
        q.resize(param.Nkrylov);
        for (int i=0; i<param.Nkrylov; i++) {
          p[i] = (i==0 && use_source) ? &b : ColorSpinorField::Create(csParam);
          q[i] = ColorSpinorField::Create(csParam);
        }
      }

      //sloppy temporary for mat-vec
      tmp_sloppy = mixed ? tmpp->CreateAlias(csParam) : nullptr;

      if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_INIT);

      init = true;
    } // init
  }

  void CAGCR::solve(Complex *psi_, std::vector<ColorSpinorField*> &q, ColorSpinorField &b)
  {
    typedef Matrix<Complex, Dynamic, Dynamic> matrix;
    typedef Matrix<Complex, Dynamic, 1> vector;

    const int N = q.size();
    vector phi(N), psi(N);
    matrix A(N,N);

#if 1
    // only a single reduction but requires using the full dot product 
    // compute rhs vector phi = Q* b = (q_i, b)
    std::vector<ColorSpinorField *> Q(N + 1);
    for (int i = 0; i < N; i++) Q[i] = q[i];
    Q[N] = &b;

    // Construct the matrix Q* Q = (A P)* (A P) = (q_i, q_j) = (A p_i, A p_j)
    Complex *A_ = new Complex[N*(N+1)];
    blas::cDotProduct(A_, q, Q);
    for (int i=0; i<N; i++) {
      phi(i) = A_[i*(N+1)+N];
      for (int j=0; j<N; j++) {
        A(i,j) = A_[i*(N+1)+j];
      }
    }
    delete[] A_;
#else
    // two reductions but uses the Hermitian block dot product
    // compute rhs vector phi = Q* b = (q_i, b)
    std::vector<ColorSpinorField *> B = {&b};
    Complex *phi_ = new Complex[N];
    blas::cDotProduct(phi_,q, B);
    for (int i=0; i<N; i++) phi(i) = phi_[i];
    delete phi_;

    // Construct the matrix Q* Q = (A P)* (A P) = (q_i, q_j) = (A p_i, A p_j)
    Complex *A_ = new Complex[N*N];
    blas::hDotProduct(A_, q, q);
    for (int i=0; i<N; i++)
      for (int j=0; j<N; j++)
        A(i,j) = A_[i*N+j];
    delete[] A_;
#endif

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EIGEN);
    }

    // use Cholesky LDL since this seems plenty stable
    LDLT<matrix> cholesky(A);
    psi = cholesky.solve(phi);

    for (int i=0; i<N; i++) psi_[i] = psi(i);

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_EIGEN);
      param.secs += profile.Last(QUDA_PROFILE_EIGEN);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }

  }

  /*
    The main CA-GCR algorithm, which consists of three main steps:
    1. Build basis vectors q_k = A p_k for k = 1..Nkrlylov
    2. Minimize the residual in this basis
    3. Update solution and residual vectors
    4. (Optional) restart if convergence or maxiter not reached
  */
  void CAGCR::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    const int n_krylov = param.Nkrylov;

    if (checkPrecision(x,b) != param.precision) errorQuda("Precision mismatch %d %d", checkPrecision(x,b), param.precision);
    if (param.return_residual && param.preserve_source == QUDA_PRESERVE_SOURCE_YES) errorQuda("Cannot preserve source and return the residual");

    if (param.maxiter == 0 || n_krylov == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(b);

    ColorSpinorField &r = rp ? *rp : *p[0];
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &tmpSloppy = tmp_sloppy ? *tmp_sloppy : tmp;

    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    // compute b2, but only if we need to
    bool fixed_iteration = param.sloppy_converge && n_krylov == param.maxiter && !param.compute_true_res;
    double b2 = !fixed_iteration ? blas::norm2(b) : 1.0;
    double r2 = 0.0; // if zero source then we will exit immediately doing no work

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
        if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
        (*eig_solve)(evecs, evals);
        if (param.deflate) {
          // double the size of the Krylov space
          extendSVDDeflationSpace();
          // populate extra memory with L/R singular vectors
          eig_solve->computeSVD(matMdagM, evecs, evals);
        }
        if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_PREAMBLE);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(matMdagM, evecs, evals);
        eig_solve->computeSVD(matMdagM, evecs, evals);
        recompute_evals = false;
      }
    }

    // compute intitial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x, tmp);
      //r = b - Ax0
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
      // Deflate: Hardcoded to SVD. If maxiter == 1, this is a dummy solve
      eig_solve->deflateSVD(x, r, evecs, evals, true);

      // Compute r_defl = RHS - A * LHS
      mat(r, x, tmp);
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

    if (basis == QUDA_CHEBYSHEV_BASIS && n_krylov > 1 && lambda_max < lambda_min && !lambda_init) {
      if (!param.is_preconditioner) {
        profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
        profile.TPSTART(QUDA_PROFILE_INIT);
      }

      // Perform 100 power iterations, normalizing every 10 mat-vecs, using r_ as an initial seed
      // and q[0]/q[1] as temporaries for the power iterations. tmpSloppy get passed in a temporary
      // for matSloppy. Technically illegal if n_krylov == 1, but in that case lambda_max isn't used anyway.
      lambda_max = 1.1 * Solver::performPowerIterations(matSloppy, r, q[0], q[1], 100, 10, tmpSloppy);
      if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("CA-GCR Approximate lambda max = 1.1 x %e\n", lambda_max / 1.1);

      lambda_init = true;

      if (!param.is_preconditioner) {
        profile.TPSTOP(QUDA_PROFILE_INIT);
        profile.TPSTART(QUDA_PROFILE_PREAMBLE);
      }
    }

    // Factors which map linear operator onto [-1,1]
    double m_map = 2. / (lambda_max - lambda_min);
    double b_map = -(lambda_max + lambda_min) / (lambda_max - lambda_min);

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
    if(use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x,r).z);

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    if (!param.is_preconditioner) {
      blas::flops = 0;
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
    int total_iter = 0;
    int restart = 0;
    double r2_old = r2;
    double maxr_deflate = sqrt(r2);
    bool l2_converge = false;

    blas::copy(*p[0], r); // no op if uni-precision

    PrintStats("CA-GCR", total_iter, r2, b2, heavy_quark_res);
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && total_iter < param.maxiter) {

      // build up a space of size n_krylov
      computeCAKrylovSpace(matSloppy, q, p, n_krylov, basis, m_map, b_map, tmpSloppy);

      solve(alpha, q, *p[0]);

      // update the solution vector
      std::vector<ColorSpinorField *> X = {&x};
      // need to make sure P is only length n_krylov
      std::vector<ColorSpinorField *> P(n_krylov);
      for (int i = 0; i < n_krylov; i++) P[i] = p[i];
      blas::caxpy(alpha, P, X);

      // no need to compute residual vector if not returning
      // residual vector and only doing a single fixed iteration
      if (!fixed_iteration || param.return_residual) {
        // update the residual vector
        std::vector<ColorSpinorField *> R = {&r};
        for (int i = 0; i < n_krylov; i++) alpha[i] = -alpha[i];
        blas::caxpy(alpha, q, R);
      }

      total_iter += n_krylov;
      if ( !fixed_iteration || getVerbosity() >= QUDA_DEBUG_VERBOSE) {
        // only compute the residual norm if we need to
        r2 = blas::norm2(r);
      }

      PrintStats("CA-GCR", total_iter, r2, b2, heavy_quark_res);

      // update since n_krylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every n_krylov steps
      if (total_iter>=param.maxiter || (r2 < stop && !l2_converge) || sqrt(r2/r2_old) < param.delta) {

        if ( (r2 < stop || total_iter>=param.maxiter) && param.sloppy_converge) break;
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
          warningQuda("CA-GCR: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
          sqrt(r2), sqrt(r2_old), resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            warningQuda("CA-GCR: solver exiting due to too many true residual norm increases");
            break;
          }
        } else {
          resIncrease = 0;
        }

        r2_old = r2;
      }

      // No matter what, if we haven't converged, we do a restart.
      if ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) ) {
        restart++; // restarting if residual is still too great

        PrintStats("CA-GCR (restart)", restart, r2, b2, heavy_quark_res);
        blas::copy(*p[0],r); // no-op if uni-precision

        r2_old = r2;

        // prevent ending the Krylov space prematurely if other convergence criteria not met 
        if (r2 < stop) l2_converge = true; 
      }

    }

    if (total_iter>param.maxiter && getVerbosity() >= QUDA_SUMMARIZE)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("CA-GCR: number of restarts = %d\n", restart);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r, x, tmp);
      double true_res = blas::xmyNorm(b, r);
      param.true_res = sqrt(true_res / b2);
      param.true_res_hq = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? sqrt(blas::HeavyQuarkResidualNorm(x,r).z) : 0.0;
      if (param.return_residual) blas::copy(b, r);
    } else {
      if (param.return_residual) blas::copy(b, r);
    }

    if (!param.is_preconditioner) {
      qudaDeviceSynchronize(); // ensure solver is complete before ending timing
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

      // store flops and reset counters
      double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops() + matMdagM.flops()) * 1e-9;

      param.gflops += gflops;
      param.iter += total_iter;

      // reset the flops counters
      blas::flops = 0;
      mat.flops();
      matSloppy.flops();
      matPrecon.flops();
      matMdagM.flops();

      profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    }

    PrintSummary("CA-GCR", total_iter, r2, b2, stop, param.tol_hq);
  }

} // namespace quda
