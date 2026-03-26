#include <nested_fgi.h>
#include <blas_quda.h>
#include <invert_quda.h>

namespace quda
{

  LowModeForce::LowModeForce(CoarseDeflationManager &deflManager_, const DiracMatrix &matFine_, int nMRSmooth_,
                              double mrOmega_) :
    deflManager(deflManager_), nMRSmooth(nMRSmooth_), mrOmega(mrOmega_), matFine(&matFine_)
  {
  }

  void LowModeForce::projectLowModes(ColorSpinorField &xLow, const ColorSpinorField &src)
  {
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

    // 1. Restrict source to coarse grid: phi_c = R * src
    T.R(coarseTmp, src);

    // 2. Project onto coarse eigenvectors: x_c = sum_k <v_k|phi_c>/lambda_k * v_k
    //    Uses the block deflation pattern from EigenSolver::deflate()
    std::vector<Complex> dots(nDefl);
    blas::block::cDotProduct(dots, {evecs.begin(), evecs.begin() + nDefl}, coarseTmp);

    for (int k = 0; k < nDefl; k++) { dots[k] /= evals[k].real(); }

    blas::zero(coarseSol);
    blas::block::caxpy(dots, {evecs.begin(), evecs.begin() + nDefl}, coarseSol);

    // 3. Prolong back to fine grid: x_low = P * x_c
    T.P(xLow, coarseSol);

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

      // Simple MR relaxation: x_{k+1} = x_k + omega * r * <Ar, r> / <Ar, Ar>
      ColorSpinorParam tmpParam{xLow};
      tmpParam.create = QUDA_ZERO_FIELD_CREATE;
      ColorSpinorField r{tmpParam};
      ColorSpinorField Ar{tmpParam};

      for (int iter = 0; iter < nMRSmooth; iter++) {
        // r = src - A * x
        (*matFine)(Ar, fineSol);
        blas::copy(r, src);
        blas::axpy(-1.0, Ar, r);

        // Ar = A * r
        (*matFine)(Ar, r);

        // alpha = <Ar, r> / <Ar, Ar>
        Complex ArDotR = blas::cDotProduct(Ar, r);
        double ArNorm = blas::norm2(Ar);
        if (ArNorm == 0.0) break;
        double alpha = mrOmega * ArDotR.real() / ArNorm;

        // x = x + alpha * r
        blas::axpy(alpha, r, fineSol);
      }

      blas::copy(xLow, fineSol);
    }
  }

  void LowModeForce::computeForce(GaugeField &mom, const ColorSpinorField &src, double coeff, GaugeField &gauge,
                                   const CloverField *clover, QudaGaugeParam &, QudaInvertParam &invParam)
  {
    // Lazily allocate fine solution workspace
    if (fineSol.empty()) {
      ColorSpinorParam fineParam{src};
      fineParam.create = QUDA_ZERO_FIELD_CREATE;
      fineSol = ColorSpinorField(fineParam);
    }

    // Project source onto low modes
    projectLowModes(fineSol, src);

    // Compute fermion force using the projected solution via existing QUDA force infrastructure.
    // The solution vector fineSol is the low-mode approximation to (D†D)^{-1} phi.
    // The force kernel computes dS/dU using this solution.
    std::vector<ColorSpinorField> xVec = {fineSol};
    std::vector<ColorSpinorField> x0Vec(1);
    std::vector<double> forceCoeff = {coeff};
    std::vector<array<double, 2>> fermEpsilon = {{0.0, 0.0}};

    // Build extended gauge if needed
    lat_dim_t R;
    for (int d = 0; d < 4; d++) R[d] = (d == 0 ? 2 : 1) * commDimPartitioned(d);
    GaugeField *gaugeEx = createExtendedGauge(gauge, R, getProfile());

    computeCloverForce(mom, *gaugeEx, gauge, *clover, xVec, x0Vec, forceCoeff, fermEpsilon, 0.0, false, invParam);

    delete gaugeEx;
  }

} // namespace quda
