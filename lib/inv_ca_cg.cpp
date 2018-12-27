#include <invert_quda.h>
#include <blas_quda.h>
#include <Eigen/Dense>

/**
   @file inv_ca_cg.cpp

   Implementation of the communication -avoiding CG algorithm.  Based
   on the description here:
   http://research.nvidia.com/sites/default/files/pubs/2016-04_S-Step-and-Communication-Avoiding/nvr-2016-003.pdf
*/

namespace quda {

  enum Basis {
    POWER_BASIS,
    CHEBYSHEV_BASIS,
    INVALID_BASIS
  };

  const static Basis basis = CHEBYSHEV_BASIS;

  CACG::CACG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile)
    : Solver(param, profile), mat(mat), matSloppy(matSloppy), init(false),
      Q_AQandg(nullptr), Q_AS(nullptr), alpha(nullptr), beta(nullptr), rp(nullptr),
      tmpp(nullptr), tmpp2(nullptr), tmp_sloppy(nullptr), tmp_sloppy2(nullptr) { }

  CACG::~CACG() {
    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {
      if (Q_AQandg) delete []Q_AQandg;
      if (Q_AS) delete []Q_AS;
      if (alpha) delete []alpha;
      if (beta) delete []beta;
      bool use_source = (param.preserve_source == QUDA_PRESERVE_SOURCE_NO &&
                         param.precision == param.precision_sloppy &&
                         param.use_init_guess == QUDA_USE_INIT_GUESS_NO);
      if (basis == POWER_BASIS) {
        for (int i=0; i<param.Nkrylov+1; i++) if (i>0 || !use_source) delete S[i];
      } else {
        for (int i=0; i<param.Nkrylov; i++) if (i>0 || !use_source) delete S[i];
        for (int i=0; i<param.Nkrylov; i++) delete AS[i];
      }
      for (int i=0; i<param.Nkrylov; i++) delete Q[i];
      for (int i=0; i<param.Nkrylov; i++) delete Qtmp[i];
      for (int i=0; i<param.Nkrylov; i++) delete AQ[i];

      if (tmp_sloppy) delete tmp_sloppy;
      if (tmp_sloppy2) delete tmp_sloppy2;
      if (tmpp) delete tmpp;
      if (tmpp2) delete tmpp2;
      if (rp) delete rp;
    }

    if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void CACG::create(ColorSpinorField &b)
  {
    if (!init) {
      if (!param.is_preconditioner) {
        blas::flops = 0;
        profile.TPSTART(QUDA_PROFILE_INIT);
      }

      Q_AQandg = new Complex[param.Nkrylov*(param.Nkrylov+1)];
      Q_AS = new Complex[param.Nkrylov*param.Nkrylov];
      alpha = new Complex[param.Nkrylov];
      beta = new Complex[param.Nkrylov*param.Nkrylov];

      bool mixed = param.precision != param.precision_sloppy;
      bool use_source = false; // need to preserve source for residual computation
      //(param.preserve_source == QUDA_PRESERVE_SOURCE_NO && !mixed &&
      //                 param.use_init_guess == QUDA_USE_INIT_GUESS_NO);

      ColorSpinorParam csParam(b);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      // Source needs to be preserved if we're computing the true residual
      rp = (mixed && !use_source) ? ColorSpinorField::Create(csParam) : nullptr;
      tmpp = ColorSpinorField::Create(csParam);
      tmpp2 = ColorSpinorField::Create(csParam);

      // now allocate sloppy fields
      csParam.setPrecision(param.precision_sloppy);

      if (basis == POWER_BASIS) {
        // in power basis AS[k] = S[k+1], so we don't need a separate array
        S.resize(param.Nkrylov+1);
        AS.resize(param.Nkrylov);
        Q.resize(param.Nkrylov);
        AQ.resize(param.Nkrylov);
        Qtmp.resize(param.Nkrylov); // for pointer swap
        for (int i=0; i<param.Nkrylov+1; i++) {
          S[i] = (i==0 && use_source) ? &b : ColorSpinorField::Create(csParam);
          if (i>0) AS[i-1] = S[i];
        }
      } else {
        S.resize(param.Nkrylov);
        AS.resize(param.Nkrylov);
        Q.resize(param.Nkrylov);
        AQ.resize(param.Nkrylov);
        Qtmp.resize(param.Nkrylov);
        for (int i=0; i<param.Nkrylov; i++) {
          S[i] = (i==0 && use_source) ? &b : ColorSpinorField::Create(csParam);
          AS[i] = ColorSpinorField::Create(csParam);
        }
      }

      for (int i=0; i<param.Nkrylov; i++) Q[i] = ColorSpinorField::Create(csParam);
      for (int i=0; i<param.Nkrylov; i++) Qtmp[i] = ColorSpinorField::Create(csParam);
      for (int i=0; i<param.Nkrylov; i++) AQ[i] = ColorSpinorField::Create(csParam);

      //sloppy temporary for mat-vec
      tmp_sloppy = mixed ? ColorSpinorField::Create(csParam) : nullptr;
      tmp_sloppy2 = mixed ? ColorSpinorField::Create(csParam) : nullptr;

      if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_INIT);

      init = true;
    } // init
  }

  void CACG::compute_alpha()
  {
    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EIGEN);
    }

    using namespace Eigen;
    typedef Matrix<Complex, Dynamic, Dynamic, RowMajor> matrix;
    typedef Matrix<Complex, Dynamic, 1> vector;

    const int N = Q.size();
    matrix matQ_AQ(N,N);
    vector vecg(N);
    for (int i=0; i<N; i++) {
      vecg(i) = Q_AQandg[i*(N+1)+N];
      for (int j=0; j<N; j++) {
        matQ_AQ(i,j) = Q_AQandg[i*(N+1)+j];
      }
    }
    Map<vector> vecalpha(alpha,N);

    vecalpha = matQ_AQ.fullPivLu().solve(vecg);
    
    //JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
    //psi = svd.solve(phi);

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_EIGEN);
      param.secs += profile.Last(QUDA_PROFILE_EIGEN);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  void CACG::compute_beta()
  {
    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EIGEN);
    }

    using namespace Eigen;
    typedef Matrix<Complex, Dynamic, Dynamic, RowMajor> matrix;

    const int N = Q.size();
    matrix matQ_AQ(N,N);
    for (int i=0; i<N; i++) {
      for (int j=0; j<N; j++) {
        matQ_AQ(i,j) = Q_AQandg[i*(N+1)+j];
      }
    }
    Map<matrix> matQ_AS(Q_AS,N,N), matbeta(beta,N,N);

    matQ_AQ = -matQ_AQ;
    matbeta = matQ_AQ.fullPivLu().solve(matQ_AS);

    //JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
    //psi = svd.solve(phi);

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_EIGEN);
      param.secs += profile.Last(QUDA_PROFILE_EIGEN);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  /*
    The main CA-CG algorithm, which consists of three main steps:
    1. Build basis vectors q_k = A p_k for k = 1..Nkrlylov
    2. Steepest descent minmization of the residual in this basis
    3. Update solution and residual vectors
    4. (Optional) restart if convergence or maxiter not reached
  */
  void CACG::operator()(ColorSpinorField &x, ColorSpinorField &b)
  {
    const int nKrylov = param.Nkrylov;

    // Hard code this for now...
    const double lambda_min = 0.;
    const double lambda_max = 20.;
    
    // Factors which map linear operator onto [-1,1]    
    double m_map = 2./(lambda_max-lambda_min);
    double b_map = -(lambda_max+lambda_min)/(lambda_max-lambda_min);

    if (checkPrecision(x,b) != param.precision) errorQuda("Precision mismatch %d %d", checkPrecision(x,b), param.precision);
    if (param.return_residual && param.preserve_source == QUDA_PRESERVE_SOURCE_YES) errorQuda("Cannot preserve source and return the residual");

    if (param.maxiter == 0 || nKrylov == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(b);

    ColorSpinorField &r_ = rp ? *rp : *S[0];
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &tmp2 = *tmpp2;
    ColorSpinorField &tmpSloppy = tmp_sloppy ? *tmp_sloppy : tmp;
    ColorSpinorField &tmpSloppy2 = tmp_sloppy2 ? *tmp_sloppy2 : tmp2;

    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    // compute b2, but only if we need to
    bool fixed_iteration = param.sloppy_converge && nKrylov==param.maxiter && !param.compute_true_res;
    double b2 = !fixed_iteration ? blas::norm2(b) : 1.0;
    double r2 = 0.0; // if zero source then we will exit immediately doing no work

    // compute intitial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r_, x, tmp, tmp2);
      //r = b - Ax0
      if (!fixed_iteration) {
        r2 = blas::xmyNorm(b, r_);
      } else {
        blas::xpay(b, -1.0, r_);
        r2 = b2; // dummy setting
      }
    } else {
      r2 = b2;
      blas::copy(r_, b);
      blas::zero(x);
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
    if(use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x,r_).z);

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
    bool l2_converge = false;

    blas::copy(*S[0], r_); // no op if uni-precision

    PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res);
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && total_iter < param.maxiter) {

      // build up a space of size nKrylov
      if (basis == POWER_BASIS) {
        for (int k=0; k<nKrylov; k++) {
          matSloppy(*AS[k], *S[k], tmpSloppy, tmpSloppy2);
        }
      } else { // chebyshev basis

        matSloppy(*AS[0], *S[0], tmpSloppy, tmpSloppy2);

        if (nKrylov > 1) {
          Complex facs1[2] = { m_map, b_map };
          std::vector<ColorSpinorField*> recur1{AS[0],S[0]};
          std::vector<ColorSpinorField*> S1{S[1]};
          blas::zero(*S[1]);
          blas::caxpy(facs1,recur1,S1);
          matSloppy(*AS[1], *S[1], tmpSloppy, tmpSloppy2);

          // Enter recursion relation
          if (nKrylov > 2) {
            Complex factors[3] = { 2.*m_map, 2.*b_map, -1. };
            for (int k = 2; k < nKrylov; k++) {
              std::vector<ColorSpinorField*> recurse;
              recurse.push_back(AS[k-1]);
              recurse.push_back(S[k-1]);
              recurse.push_back(S[k-2]);
              std::vector<ColorSpinorField*> Sk;
              Sk.push_back(S[k]);
              blas::zero(*S[k]);
              blas::caxpy(factors, recurse, Sk);
              matSloppy(*AS[k], *S[k], tmpSloppy, tmpSloppy2);
            }
          }
        }

      }

      // first iteration, copy S and AS into Q and AQ
      if (total_iter == 0) {
        // first iteration Q = S
        for (int i=0; i<nKrylov; i++) *Q[i] = *S[i];
        for (int i=0; i<nKrylov; i++) *AQ[i] = *AS[i];

      } else {


        // Compute the beta coefficients for updating Q, AQ
        // 1. compute matrix Q_AS = -Q^\dagger AS
        // 2. Solve Q_AQ beta = Q_AS
        blas::cDotProduct(Q_AS, Q, AS);
        for (int i = 0; i < param.Nkrylov*param.Nkrylov; i++) { Q_AS[i] = real(Q_AS[i]); }

        compute_beta();

        // update direction vectors
        std::vector<ColorSpinorField*> R;
        for (int i=0; i<nKrylov; i++) R.push_back(S[i]); // needed because S is size N+1 in power basis
        blas::caxpyz(beta, Q, R, Qtmp);
        for (int i=0; i<nKrylov; i++) std::swap(Q[i],Qtmp[i]);

        blas::caxpyz(beta, AQ, AS, Qtmp);
        for (int i=0; i<nKrylov; i++) std::swap(AQ[i],Qtmp[i]);
      }

      // compute the alpha coefficients
      // 1. Compute Q_AQ = Q^\dagger AQ and g = Q^dagger r = Q^dagger S[0]
      // 2. Solve Q_AQ alpha = g
      {
        std::vector<ColorSpinorField*> Q2;
        for (int i=0; i<nKrylov; i++) Q2.push_back(AQ[i]);
        Q2.push_back(S[0]);
        blas::cDotProduct(Q_AQandg, Q, Q2);

        for (int i = 0; i < param.Nkrylov*(param.Nkrylov+1); i++) { Q_AQandg[i] = real(Q_AQandg[i]); }

        compute_alpha();
      }

      // update the solution vector
      std::vector<ColorSpinorField*> X;
      X.push_back(&x);
      blas::caxpy(alpha, Q, X);

      // no need to compute residual vector if only doing a single fixed iteration
      // perhaps we could save the mat-vec here if we compute "Ap"
      // vectors when we update p?
      if (!fixed_iteration) {
        for (int i = 0; i < param.Nkrylov; i++) { alpha[i] = -alpha[i]; }
        std::vector<ColorSpinorField*> S0;
        S0.push_back(S[0]);

        // Can we fuse these? We don't need this reduce in all cases...
        blas::caxpy(alpha, AQ, S0);
        r2 = blas::norm2(*S[0]);
      }

      total_iter+=nKrylov;

      PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res);

      // update since nKrylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every nKrylov steps
      // Note: this won't reliable update when the norm _increases_.
      if (total_iter>=param.maxiter || (r2 < stop && !l2_converge) || sqrt(r2/r2_old) < param.delta) {
        if ( (r2 < stop || total_iter>=param.maxiter) && param.sloppy_converge) break;
        mat(r_, x, tmp, tmp2);
        r2 = blas::xmyNorm(b, r_);
        blas::copy(*S[0], r_);

        if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r_).z);

        // break-out check if we have reached the limit of the precision
        if (r2 > r2_old) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("CA-CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
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

    if (total_iter>param.maxiter && getVerbosity() >= QUDA_SUMMARIZE)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("CA-CG: number of restarts = %d\n", restart);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r_, x, tmp, tmp2);
      double true_res = blas::xmyNorm(b, r_);
      param.true_res = sqrt(true_res / b2);
      param.true_res_hq = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? sqrt(blas::HeavyQuarkResidualNorm(x,r_).z) : 0.0;
      if (param.return_residual) blas::copy(b, r_);
    } else {
      if (param.return_residual) blas::copy(b, *Q[0]);
    }

    if (!param.is_preconditioner) {
      qudaDeviceSynchronize(); // ensure solver is complete before ending timing
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);

      // store flops and reset counters
      double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;

      param.gflops += gflops;
      param.iter += total_iter;

      // reset the flops counters
      blas::flops = 0;
      mat.flops();
      matSloppy.flops();

      profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    }

    PrintSummary("CA-CG", total_iter, r2, b2, stop, param.tol_hq);
  }

} // namespace quda
