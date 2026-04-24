/**
 * @file low_mode_force.cpp
 * @brief Low-mode fermion force using coarse-grid deflation + optional MR smoothing.
 *
 * Implements the "cheap" inner force for the nested force-gradient integrator:
 *   1. Restrict the pseudofermion to the coarse grid via the MG transfer.
 *   2. Project onto the tracked coarse eigenvectors (deflation).
 *   3. Prolong the low-mode correction back to the fine grid.
 *   4. Optional MR smoothing iterations on the fine-grid operator.
 *   5. Feed the resulting approximate solution to the standard fermion-force
 *      kernel (computeEOFermionForce for Wilson, computeCloverForce for clover).
 *
 * This is the force component that is refreshed at the inner (fast) timescale
 * in the nested FGI integrator, while the "expensive" F_full - F_low
 * correction is applied only at the outer (slow) timescale.
 */
#include <nested_fgi.h>
#include <hmc_quda.h>
#include <blas_quda.h>
#include <invert_quda.h>
#include <timer.h>

namespace quda
{

  static TimeProfile profileLowModeForce("LowModeForce");

  LowModeForce::LowModeForce(CoarseDeflationManager &deflManager_, const DiracMatrix &matFine_, int nMRSmooth_,
                              double mrOmega_) :
    deflManager(deflManager_), nMRSmooth(nMRSmooth_), mrOmega(mrOmega_), matFine(&matFine_)
  {
  }

  void LowModeForce::projectLowModes(ColorSpinorField &xLow, const ColorSpinorField &src)
  {
    auto profile = pushProfile(profileLowModeForce);
    const auto &evecs = deflManager.getEvecs();
    const auto &evals = deflManager.getEvals();
    const Transfer &T = deflManager.getTransfer();
    int nDefl = deflManager.getNDefl();

    // Lazily allocate coarse workspace fields from the first eigenvector's metadata
    if (coarseTmp.empty()) {
      ColorSpinorParam csParam{evecs[0]};
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      coarseTmp = ColorSpinorField(csParam);
      coarseSol = ColorSpinorField(csParam);
    }

    // The restrict/prolong kernels are instantiated at the null-space precision.
    // If src (and xLow) are at a different precision than the transfer expects,
    // bounce through a fine-grid workspace at T.NullPrecision().
    QudaPrecision nullPrec = T.NullPrecision();
    const ColorSpinorField *srcForRestrict = &src;
    if (src.Precision() != nullPrec) {
      if (fineSrcBuf.empty() || fineSrcBuf.Precision() != nullPrec) {
        ColorSpinorParam p(src);
        p.setPrecision(nullPrec);
        p.create = QUDA_NULL_FIELD_CREATE;
        fineSrcBuf = ColorSpinorField(p);
      }
      blas::copy(fineSrcBuf, src);
      srcForRestrict = &fineSrcBuf;
    }

    // 1. Restrict source to coarse grid: phi_c = R * src
    T.R(coarseTmp, *srcForRestrict);

    // 2. Project onto coarse eigenvectors: x_c = sum_k <v_k|phi_c>/lambda_k * v_k
    //    Uses the block deflation pattern from EigenSolver::deflate()
    std::vector<Complex> dots(nDefl);
    blas::block::cDotProduct(dots, {evecs.begin(), evecs.begin() + nDefl}, coarseTmp);

    for (int k = 0; k < nDefl; k++) { dots[k] /= evals[k].real(); }

    blas::zero(coarseSol);
    blas::block::caxpy(dots, {evecs.begin(), evecs.begin() + nDefl}, coarseSol);

    // 3. Prolong back to fine grid: x_low = P * x_c.
    // The prolong kernel writes at T.NullPrecision(); convert back if xLow differs.
    if (xLow.Precision() != nullPrec) {
      if (fineProlongBuf.empty() || fineProlongBuf.Precision() != nullPrec) {
        ColorSpinorParam p(xLow);
        p.setPrecision(nullPrec);
        p.create = QUDA_NULL_FIELD_CREATE;
        fineProlongBuf = ColorSpinorField(p);
      }
      T.P(fineProlongBuf, coarseSol);
      blas::copy(xLow, fineProlongBuf);
    } else {
      T.P(xLow, coarseSol);
    }

    // 4. Optional MR smoothing to improve the approximation
    if (nMRSmooth > 0) {
      // Lazily allocate fine workspace
      if (fineSol.empty()) {
        ColorSpinorParam fineParam{xLow};
        fineParam.create = QUDA_ZERO_FIELD_CREATE;
        fineSol = ColorSpinorField(fineParam);
      }

      // Use xLow as initial guess, apply a few MR iterations of A x = src
      blas::copy(fineSol, xLow);

      // Lazily allocate MR workspace
      if (mrResid.empty()) {
        ColorSpinorParam tmpParam{xLow};
        tmpParam.create = QUDA_ZERO_FIELD_CREATE;
        mrResid = ColorSpinorField(tmpParam);
        mrAr = ColorSpinorField(tmpParam);
      }

      // Simple MR relaxation: x_{k+1} = x_k + omega * r * <Ar, r> / <Ar, Ar>
      for (int iter = 0; iter < nMRSmooth; iter++) {
        // r = src - A * x
        (*matFine)(mrAr, fineSol);
        blas::copy(mrResid, src);
        blas::axpy(-1.0, mrAr, mrResid);

        // Ar = A * r
        (*matFine)(mrAr, mrResid);

        // alpha = <Ar, r> / <Ar, Ar>
        Complex ArDotR = blas::cDotProduct(mrAr, mrResid);
        double ArNorm = blas::norm2(mrAr);
        if (ArNorm == 0.0) break;
        double alpha = mrOmega * ArDotR.real() / ArNorm;

        // x = x + alpha * r
        blas::axpy(alpha, mrResid, fineSol);
      }

      blas::copy(xLow, fineSol);
    }
  }

  void LowModeForce::computeForce(GaugeField &mom, const ColorSpinorField &src, double coeff, GaugeField &gauge,
                                   const CloverField *clover, QudaGaugeParam &, QudaInvertParam &invParam)
  {
    auto profile = pushProfile(profileLowModeForce);
    // Lazily allocate fine solution workspace
    if (fineSol.empty()) {
      ColorSpinorParam fineParam{src};
      fineParam.create = QUDA_ZERO_FIELD_CREATE;
      fineSol = ColorSpinorField(fineParam);
    }

    // Project source onto low modes
    projectLowModes(fineSol, src);

    // Compute fermion force using the projected solution. The low-mode fine-grid
    // vector is the approximate (D†D)^{-1} φ, structurally equivalent to the CG
    // solution used by the standard fermion force kernel.
    if (invParam.dslash_type == QUDA_WILSON_DSLASH) {
      // Wilson: internal EO force (matches the other integrators' convention).
      computeEOFermionForce(mom, fineSol, invParam, coeff);
    } else {
      // Clover: preserve the established computeCloverForce call path.
      std::vector<ColorSpinorField> xVec = {fineSol};
      std::vector<ColorSpinorField> x0Vec(1);
      std::vector<double> forceCoeff = {coeff};
      std::vector<array<double, 2>> fermEpsilon = {{0.0, 0.0}};

      lat_dim_t R;
      for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * commDimPartitioned(d);
      GaugeField *gaugeEx = createExtendedGauge(gauge, R, getProfile());

      computeCloverForce(mom, *gaugeEx, gauge, *clover, xVec, x0Vec, forceCoeff, fermEpsilon, 0.0, false, invParam);

      delete gaugeEx;
    }
  }

} // namespace quda
