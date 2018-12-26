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
    INVALID_BASIS
  };

  const static Basis basis = POWER_BASIS;

  CACG::CACG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile)
    : Solver(param, profile), mat(mat), matSloppy(matSloppy), init(false),
      W(nullptr), C(nullptr), alpha(nullptr), beta(nullptr), phi(nullptr), rp(nullptr),
      tmpp(nullptr), tmpp2(nullptr), tmp_sloppy(nullptr), tmp_sloppy2(nullptr) { }

  CACG::~CACG() {
    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);

    if (init) {
      if (W) delete []W;
      if (C) delete []C;
      if (alpha) delete []alpha;
      if (beta) delete []beta;
      if (phi) delete []phi;
      bool use_source = (param.preserve_source == QUDA_PRESERVE_SOURCE_NO &&
                         param.precision == param.precision_sloppy &&
                         param.use_init_guess == QUDA_USE_INIT_GUESS_NO);
      if (basis == POWER_BASIS) {
        for (int i=0; i<param.Nkrylov+1; i++) if (i>0 || !use_source) delete r[i];
      } else {
        for (int i=0; i<param.Nkrylov; i++) if (i>0 || !use_source) delete r[i];
        for (int i=0; i<param.Nkrylov; i++) delete q[i];
      }
      for (int i=0; i<param.Nkrylov; i++) delete p[i];
      for (int i=0; i<param.Nkrylov; i++) delete p2[i];
      if (x_sloppy) delete x_sloppy;
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

      alpha = new Complex[param.Nkrylov];
      beta = new Complex[param.Nkrylov*param.Nkrylov];
      phi = new Complex[param.Nkrylov];
      W = new Complex[param.Nkrylov*param.Nkrylov];
      C = new Complex[param.Nkrylov*param.Nkrylov];

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
        // in power basis q[k] = r[k+1], so we don't need a separate q array
        r.resize(param.Nkrylov+1);
        q.resize(param.Nkrylov);
        p.resize(param.Nkrylov);
        p2.resize(param.Nkrylov); // for pointer swap
        for (int i=0; i<param.Nkrylov+1; i++) {
          r[i] = (i==0 && use_source) ? &b : ColorSpinorField::Create(csParam);
          if (i>0) q[i-1] = r[i];
        }
      } else {
        r.resize(param.Nkrylov);
        q.resize(param.Nkrylov);
        p.resize(param.Nkrylov);
        p2.resize(param.Nkrylov); // for pointer swap
        for (int i=0; i<param.Nkrylov; i++) {
          r[i] = (i==0 && use_source) ? &b : ColorSpinorField::Create(csParam);
          q[i] = ColorSpinorField::Create(csParam);
        }
      }

      for (int i=0; i<param.Nkrylov; i++) p[i] = ColorSpinorField::Create(csParam);
      for (int i=0; i<param.Nkrylov; i++) p2[i] = ColorSpinorField::Create(csParam);

      //sloppy temporary for mat-vec
      tmp_sloppy = mixed ? ColorSpinorField::Create(csParam) : nullptr;
      tmp_sloppy2 = mixed ? ColorSpinorField::Create(csParam) : nullptr;

      // sloppy solution for sloppy mat-vec
      x_sloppy = mixed ? ColorSpinorField::Create(csParam) : nullptr;

      if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_INIT);

      init = true;
    } // init
  }

  void CACG::compute_alpha(Complex *psi_, Complex *A_, Complex *phi_)
  {
    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EIGEN);
    }

    using namespace Eigen;
    typedef Matrix<Complex, Dynamic, Dynamic, RowMajor> matrix;
    typedef Matrix<Complex, Dynamic, 1> vector;

    const int N = q.size();
    Map<matrix> A(W,N,N);
    Map<vector> phi(phi_,N), psi(psi_,N);

    JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
    psi = svd.solve(phi);

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_EIGEN);
      param.secs += profile.Last(QUDA_PROFILE_EIGEN);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  void CACG::compute_beta(Complex *psi_, Complex *A_, Complex *phi_)
  {
    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      param.secs += profile.Last(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EIGEN);
    }

    using namespace Eigen;
    typedef Matrix<Complex, Dynamic, Dynamic, RowMajor> matrix;

    const int N = q.size();
    Map<matrix> A(A_,N,N), phi(phi_,N,N), psi(psi_,N,N);

    phi = -phi;

    JacobiSVD<matrix> svd(A, ComputeThinU | ComputeThinV);
    psi = svd.solve(phi);

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

    if (checkPrecision(x,b) != param.precision) errorQuda("Precision mismatch %d %d", checkPrecision(x,b), param.precision);
    if (param.return_residual && param.preserve_source == QUDA_PRESERVE_SOURCE_YES) errorQuda("Cannot preserve source and return the residual");

    if (param.maxiter == 0 || nKrylov == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(b);

    ColorSpinorField &r_ = rp ? *rp : *r[0];
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &tmp2 = *tmpp2;
    ColorSpinorField &tmpSloppy = tmp_sloppy ? *tmp_sloppy : tmp;
    ColorSpinorField &tmpSloppy2 = tmp_sloppy2 ? *tmp_sloppy2 : tmp2;
    ColorSpinorField &xSloppy = x_sloppy ? *x_sloppy : x;

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

    blas::copy(*r[0], r_); // no op if uni-precision

    PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res);
    while ( !convergence(r2, heavy_quark_res, stop, param.tol_hq) && total_iter < param.maxiter) {

      // build up a space of size nKrylov
      for (int k=0; k<nKrylov; k++) {
        matSloppy(*q[k], *r[k], tmpSloppy, tmpSloppy2);
        if (k<nKrylov-1 && basis != POWER_BASIS) blas::copy(*r[k+1], *q[k]);
      }

      // for now just copy R into P since the beta computation is
      // buggy - this means we end up doing CG within the s-steps and
      // steepest descent between them
      if (total_iter == 0) {
        // first iteration P = R
        for (int i=0; i<nKrylov; i++) *p[i] = *r[i];
      } else {

        // compute the beta coefficients for updating P
        // 1. Compute matrix C = -Q^\dagger P
        // 2. Solve W beta = phi
        blas::cDotProduct(C, p, q);
        for (int i = 0; i < param.Nkrylov*param.Nkrylov; i++) { C[i] = real(C[i]); }

        compute_beta(beta, W, C);

        // update direction vectors
        std::vector<ColorSpinorField*> R;
        for (int i=0; i<nKrylov; i++) R.push_back(r[i]);
        blas::caxpyz(beta, p, R, p2);
        for (int i=0; i<nKrylov; i++) std::swap(p[i],p2[i]);
      }

      // compute the alpha coefficients
      // 1. Compute W = Q^\dagger P and phi = P^\dagger r
      // 2. Solve W alpha = phi
      {
        /*
          // optimization we can make once we have p A-orthogonality fixed
          std::vector<ColorSpinorField*> Q;
          for (int i=0; i<nKrylov; i++) Q.push_back(q[i]);
          Q.push_back(r[0]);

          // only a single reduction but requires using the full dot product
          // compute rhs vector phi = P* r = (p_i, r)
          // Construct the matrix P* Q = P* (A R) = (p_i, q_j) = (p_i, A r_j)
          //blas::cDotProduct(W, q, Q);
        */
        blas::cDotProduct(W, p, q);
        for (int i = 0; i < param.Nkrylov*param.Nkrylov; i++) { W[i] = real(W[i]); }
        std::vector<ColorSpinorField*> R;
        R.push_back(r[0]);
        blas::cDotProduct(phi, p, R);
        for (int i = 0; i < param.Nkrylov; i++) { phi[i] = real(phi[i]); }

        compute_alpha(alpha, W, phi);
      }

      // update the solution vector
      std::vector<ColorSpinorField*> X;
      X.push_back(&x);
      blas::caxpy(alpha, p, X);

      // no need to compute residual vector if not returning residual
      // vector and only doing a single fixed iteration
      // perhaps we could save the mat-vec here if we compute "Ap"
      // vectors when we update p?
      if (!fixed_iteration || param.return_residual) {
        // update the residual vector
        blas::copy(xSloppy, x); // no op if uni-precision
        matSloppy(*r[0], xSloppy, tmpSloppy, tmpSloppy2);
        if (getVerbosity() >= QUDA_VERBOSE) r2 = blas::xmyNorm(b, *r[0]);
        else blas::xpay(b, -1.0, *r[0]);
      }

      total_iter+=nKrylov;

      PrintStats("CA-CG", total_iter, r2, b2, heavy_quark_res);

      // update since nKrylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every nKrylov steps
      if (total_iter>=param.maxiter || (r2 < stop && !l2_converge) || sqrt(r2/r2_old) < param.delta) {

        if ( (r2 < stop || total_iter>=param.maxiter) && param.sloppy_converge) break;
        mat(r_, x, tmp, tmp2);
        r2 = blas::xmyNorm(b, r_);
        blas::copy(*r[0], r_);

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
      if (param.return_residual) blas::copy(b, *p[0]);
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
