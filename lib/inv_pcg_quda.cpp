#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#include <madwf_ml.h>

namespace quda
{

  using namespace blas;

  // set the required parameters for the inner solver
  static void fillInnerSolverParam(SolverParam &inner, const SolverParam &outer)
  {
    inner.tol = outer.tol_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver

    // most preconditioners are uni-precision solvers, with CG being an exception
    inner.precision
      = (outer.inv_type_precondition == QUDA_CG_INVERTER && !outer.precondition_no_advanced_feature) ?
        outer.precision_sloppy : outer.precision_precondition;
    inner.precision_sloppy = outer.precision_precondition;

    // this sets a fixed iteration count if we're using the MR solver
    inner.residual_type
      = (outer.inv_type_precondition == QUDA_MR_INVERTER) ? QUDA_INVALID_RESIDUAL : QUDA_L2_RELATIVE_RESIDUAL;

    inner.iter = 0;
    inner.gflops = 0;
    inner.secs = 0;

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // used to tell the inner solver it is an inner solver
    inner.pipeline = true;

    inner.schwarz_type = outer.schwarz_type;
    inner.global_reduction = inner.schwarz_type == QUDA_INVALID_SCHWARZ ? true : false;

    inner.maxiter = outer.maxiter_precondition;
    if (outer.inv_type_precondition == QUDA_CA_GCR_INVERTER) {
      inner.Nkrylov = inner.maxiter / outer.precondition_cycle;
    } else {
      inner.Nsteps = outer.precondition_cycle;
    }

    if (outer.inv_type == QUDA_PCG_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else
      inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;

    inner.verbosity_precondition = outer.verbosity_precondition;

    inner.compute_true_res = false;
    inner.sloppy_converge = true;
  }

  PreconCG::PreconCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                     const DiracMatrix &matEig, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matEig, param, profile),
    K(0),
    Kparam(param)
  {
    fillInnerSolverParam(Kparam, param);
    // Preconditioners do not need a deflation space,
    // so we explicily set this here.
    Kparam.deflate = false;

    if (param.schwarz_type == QUDA_ADDITIVE_MADWF_SCHWARZ) {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new Acc<MadwfAcc, CG>(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else { // unknown preconditioner
        errorQuda("Unknown inner solver %d for MADWF", param.inv_type_precondition);
      }
    } else {
      if (param.inv_type_precondition == QUDA_CG_INVERTER) {
        K = new CG(matPrecon, matPrecon, matPrecon, matEig, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_MR_INVERTER) {
        K = new MR(matPrecon, matPrecon, Kparam, profile);
      } else if (param.inv_type_precondition == QUDA_SD_INVERTER) {
        K = new SD(matPrecon, Kparam, profile);
      } else if (param.inv_type_precondition != QUDA_INVALID_INVERTER) { // unknown preconditioner
        errorQuda("Unknown inner solver %d", param.inv_type_precondition);
      }
    }
  }

  PreconCG::~PreconCG()
  {
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (K) delete K;
    destroyDeflationSpace();

    profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  void PreconCG::solve_and_collect(ColorSpinorField &x, ColorSpinorField &b, std::vector<ColorSpinorField *> &v_r,
                            int collect_miniter, double collect_tol)
  {
    if (param.schwarz_type == QUDA_ADDITIVE_MADWF_SCHWARZ) { K->train_param(*this, b); }

    profile.TPSTART(QUDA_PROFILE_INIT);

    // whether to select alternative reliable updates
    bool alternative_reliable = param.use_alternative_reliable;

    double b2 = blas::norm2(b);

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0 && param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    int k = 0;
    int rUpdate = 0;

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      constructDeflationSpace(b, matEig);
      if (deflate_compute) {
        // compute the deflation space.
        (*eig_solve)(evecs, evals);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(matEig, evecs, evals);
        recompute_evals = false;
      }
    }
    cudaColorSpinorField *minvrPre = nullptr;
    cudaColorSpinorField *rPre = nullptr;
    cudaColorSpinorField *minvr = nullptr;
    cudaColorSpinorField *minvrSloppy = nullptr;
    cudaColorSpinorField *p = nullptr;

    ColorSpinorParam csParam(b);
    cudaColorSpinorField r(b);
    if (K) minvr = new cudaColorSpinorField(b);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b, csParam);

    csParam.setPrecision(param.precision_sloppy);

    // temporary fields
    ColorSpinorField *tmpp = ColorSpinorField::Create(csParam);
    ColorSpinorField *tmp2p = nullptr;
    ColorSpinorField *tmp3p = nullptr;
    if (!mat.isStaggered()) {
      // tmp2 only needed for multi-gpu Wilson-like kernels
      tmp2p = ColorSpinorField::Create(csParam);
      // additional high-precision temporary if Wilson and mixed-precision
      csParam.setPrecision(param.precision);
      tmp3p = (param.precision != param.precision_sloppy) ? ColorSpinorField::Create(csParam) : tmpp;
      csParam.setPrecision(param.precision_sloppy);
    } else {
      tmp3p = tmp2p = tmpp;
    }

    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &tmp2 = *tmp2p;
    ColorSpinorField &tmp3 = *tmp3p;

    // alternative reliable updates
    // alternative reliable updates - set precision - does not hurt performance here

    const double u = param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon() / 2. :
      param.precision_sloppy == 4                ? std::numeric_limits<float>::epsilon() / 2. :
      param.precision_sloppy == 2                ? pow(2., -13) :
                                                   pow(2., -6);
    const double uhigh = param.precision == 8 ? std::numeric_limits<double>::epsilon() / 2. :
      param.precision == 4                    ? std::numeric_limits<float>::epsilon() / 2. :
      param.precision == 2                    ? pow(2., -13) :
                                                pow(2., -6);

    const double deps = sqrt(u);
    constexpr double dfac = 1.1;
    double d_new = 0;
    double d = 0;
    double dinit = 0;
    double xNorm = 0;
    double xnorm = 0;
    double pnorm = 0;
    double ppnorm = 0;
    double Anorm = 0;

    // for alternative reliable updates
    if (alternative_reliable) {
      // estimate norm for reliable updates
      mat(r, b, y, tmp3);
      Anorm = sqrt(blas::norm2(r) / b2);
    }

    // compute initial residual
    double r2 = 0.0;
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x, y, tmp3);
      r2 = blas::xmyNorm(b, r);
      if (b2 == 0) b2 = r2;
      // y contains the original guess.
      blas::copy(y, x);
    } else {
      if (&r != &b) blas::copy(r, b);
      r2 = b2;
      blas::zero(y);
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate and accumulate to solution vector
      eig_solve->deflate(y, r, evecs, evals, true);
      mat(r, y, x, tmp3);
      r2 = blas::xmyNorm(b, r);
    }

    cudaColorSpinorField Ap(x, csParam);

    cudaColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
      minvrSloppy = minvr;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = new cudaColorSpinorField(r, csParam);
      if (K) minvrSloppy = new cudaColorSpinorField(*minvr, csParam);
    }

    cudaColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() || !param.use_sloppy_partial_accumulator) {
      csParam.create = QUDA_REFERENCE_FIELD_CREATE;
      x_sloppy = &static_cast<cudaColorSpinorField &>(x);
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = new cudaColorSpinorField(x, csParam);
    }

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    blas::zero(x);
    if (&x != &xSloppy) blas::zero(xSloppy);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    if (K) {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      csParam.setPrecision(Kparam.precision);
      rPre = new cudaColorSpinorField(rSloppy, csParam);
      // Create minvrPre
      minvrPre = new cudaColorSpinorField(*rPre);
      pushVerbosity(param.verbosity_precondition);
      (*K)(*minvrPre, *rPre);
      popVerbosity();
      *minvrSloppy = *minvrPre;
      p = new cudaColorSpinorField(*minvrSloppy);
    } else {
      p = new cudaColorSpinorField(rSloppy);
    }

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
    double heavy_quark_res = 0.0;                               // heavy quark residual
    if (use_heavy_quark_res) heavy_quark_res = sqrt(HeavyQuarkResidualNorm(x, r).z);

    double alpha = 0.0, beta = 0.0;
    double pAp;
    double rMinvr = 0;
    double rMinvr_old = 0.0;
    double r_new_Minvr_old = 0.0;
    double r2_old = 0;
    r2 = norm2(r);

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double maxr_deflate = rNorm; // The maximum residual since the last deflation
    double delta = param.delta;

    if (K) rMinvr = reDotProduct(rSloppy, *minvrSloppy);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    blas::flops = 0;

    PrintStats("PCG", k, r2, b2, heavy_quark_res);

    int steps_since_reliable = 1;

    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    int collect = v_r.size();

    // alternative reliable updates
    if (alternative_reliable) {
      dinit = uhigh * (rNorm + Anorm * xNorm);
      d = dinit;
    }

    while (!convergence(r2, heavy_quark_res, stop, param.tol_hq) && k < param.maxiter) {

      matSloppy(Ap, *p, tmp, tmp2);

      double sigma;
      // alternative reliable updates,
      if (alternative_reliable) {
        double3 pAppp = blas::cDotProductNormA(*p, Ap);
        pAp = pAppp.x;
        ppnorm = pAppp.z;
      } else {
        pAp = reDotProduct(*p, Ap);
      }

      alpha = (K) ? rMinvr / pAp : r2 / pAp;
      Complex cg_norm = axpyCGNorm(-alpha, Ap, rSloppy);
      // r --> r - alpha*A*p
      r2_old = r2;
      r2 = real(cg_norm);

      sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2; // use r2 if (r_k+1, r_k-1 - r_k) breaks

      if (K) rMinvr_old = rMinvr;

      rNorm = sqrt(r2);
      int updateX;
      int updateR;

      if (alternative_reliable) { // alternative reliable updates
        updateX = ((d <= deps * sqrt(r2_old)) or (dfac * dinit > deps * r0Norm)) and (d_new > deps * rNorm)
          and (d_new > dfac * dinit);
        updateR = 0;
      } else {
        if (rNorm > maxrx) maxrx = rNorm;
        if (rNorm > maxrr) maxrr = rNorm;

        updateX = (rNorm < delta * r0Norm && r0Norm <= maxrx) ? 1 : 0;
        updateR = ((rNorm < delta * maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
      }

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if (convergence(r2, heavy_quark_res, stop, param.tol_hq) && delta >= param.tol) updateX = 1;

      if (collect > 0 && k > collect_miniter && r2 < collect_tol * collect_tol * b2) {
        *v_r[v_r.size() - collect] = rSloppy;
        printfQuda("Collecting r %2d: r2 / b2 = %12.8e, k = %5d.\n", collect, sqrt(r2 / b2), k);
        collect--;
      }

      if (!(updateR || updateX)) {

        if (K) {
          // can fuse these two kernels
          r_new_Minvr_old = reDotProduct(rSloppy, *minvrSloppy);
          *rPre = rSloppy;

          pushVerbosity(param.verbosity_precondition);
          (*K)(*minvrPre, *rPre);
          popVerbosity();

          // can fuse these two kernels
          *minvrSloppy = *minvrPre;
          rMinvr = reDotProduct(rSloppy, *minvrSloppy);

          beta = (rMinvr - r_new_Minvr_old) / rMinvr_old;
          axpyZpbx(alpha, *p, xSloppy, *minvrSloppy, beta);
        } else {
          beta = sigma / r2_old; // use the alternative beta computation
          axpyZpbx(alpha, *p, xSloppy, rSloppy, beta);
        }

        // alternative reliable updates
        if (alternative_reliable) {
          d = d_new;
          pnorm = pnorm + alpha * alpha * ppnorm;
          xnorm = sqrt(pnorm);
          d_new = d + u * rNorm + uhigh * Anorm * xnorm;
          if (steps_since_reliable == 0 && getVerbosity() >= QUDA_DEBUG_VERBOSE) {
            printfQuda("New dnew: %e (r %e , y %e)\n", d_new, u * rNorm, uhigh * Anorm * sqrt(blas::norm2(y)));
          }
        }

        steps_since_reliable++;

      } else { // reliable update

        axpy(alpha, *p, xSloppy); // xSloppy += alpha*p
        xpy(xSloppy, y);          // y += x
        // Now compute r
        mat(r, y, x, tmp3); // x is just a temporary here
        r2 = xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y, x, tmp3);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2);
        }

        copy(rSloppy, r); // copy r to rSloppy
        zero(xSloppy);

        // break-out check if we have reached the limit of the precision
        if (sqrt(r2) > r0Norm && updateX) {
          resIncrease++;
          resIncreaseTotal++;
          // reuse r0Norm for this
          warningQuda(
            "PCG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2), r0Norm, resIncreaseTotal);

          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) break;

        } else {
          resIncrease = 0;
        }

        // alternative reliable updates
        if (alternative_reliable) {
          dinit = uhigh * (sqrt(r2) + Anorm * sqrt(blas::norm2(y)));
          d = d_new;
          xnorm = 0; // sqrt(norm2(x));
          pnorm = 0; // pnorm + alpha * sqrt(norm2(p));
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
            printfQuda("New dinit: %e (r %e , y %e)\n", dinit, uhigh * sqrt(r2), uhigh * Anorm * sqrt(blas::norm2(y)));
          }
          d_new = dinit;
        } else {
          rNorm = sqrt(r2);
          maxrr = rNorm;
          maxrx = rNorm;
        }

        steps_since_reliable = 0;
        r0Norm = sqrt(r2);
        ++rUpdate;

        if (K) {
          // can fuse these two kernels
          r_new_Minvr_old = reDotProduct(rSloppy, *minvrSloppy);
          *rPre = rSloppy;

          pushVerbosity(param.verbosity_precondition);
          (*K)(*minvrPre, *rPre);
          popVerbosity();

          // can fuse these two kernels
          *minvrSloppy = *minvrPre;
          rMinvr = reDotProduct(rSloppy, *minvrSloppy);

          beta = (rMinvr - r_new_Minvr_old) / rMinvr_old;
          xpay(*minvrSloppy, beta, *p); // p = minvrSloppy + beta*p
        } else {                        // standard CG - no preconditioning

          // explicitly restore the orthogonality of the gradient vector
          double rp = reDotProduct(rSloppy, *p) / (r2);
          axpy(-rp, rSloppy, *p);

          beta = r2 / r2_old;
          xpay(rSloppy, beta, *p);
        }
      }

      ++k;
      PrintStats("PCG", k, r2, b2, heavy_quark_res);
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    if (x.Precision() != param.precision_sloppy) copy(x, xSloppy);
    xpy(y, x); // x += y

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops() + matEig.flops()) * 1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (collect > 0) { warningQuda("%d r vectors still to be collected ...\n", collect); }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("PCG: Reliable updates = %d\n", rUpdate);

    // compute the true residual
    mat(r, x, y, tmp3);
    double true_res = xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();
    matPrecon.flops();
    matEig.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (tmpp) delete tmpp;
    if (!mat.isStaggered()) {
      if (tmp2p && tmpp != tmp2p) delete tmp2p;
      if (tmp3p && tmpp != tmp3p && param.precision != param.precision_sloppy) delete tmp3p;
    }

    if (K) { // These are only needed if preconditioning is used
      delete minvrPre;
      delete rPre;
      delete minvr;
      if (x.Precision() != param.precision_sloppy) delete minvrSloppy;
    }
    delete p;

    if (param.precision_sloppy != x.Precision()) {
      delete r_sloppy;
      if (param.use_sloppy_partial_accumulator) { delete x_sloppy; }
    }

    profile.TPSTOP(QUDA_PROFILE_FREE);
    return;
  }

  std::unordered_map<std::string, std::vector<float>> MadwfAcc::host_training_param_cache; // empty map

} // namespace quda
