#include <transfer.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

// This define controls which kernels get compiled in `coarse_op.cuh`.
// This ensures only kernels relevant for coarsening a coarse operator
// get built, saving compile time.
#define COARSECOARSE
#include <coarse_op.cuh>

namespace quda
{

  template <bool use_mma, typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  std::enable_if_t<!use_mma, void>
  calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                   const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                   double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                   bool need_bidirectional)
  {
    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {
      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using V = typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat>;
      using F = typename colorspinor::FieldOrderCB<Float, 2 * fineSpin, fineColor, coarseColor, csOrder, vFloat>;
      using gFine = typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat>;
      using cFine = typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat>;
      using gCoarse = typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, vFloat>;
      using gCoarseAtomic =
        typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, storeType>;

      const ColorSpinorField &v = T.Vectors(Y.Location());

      V vAccessor(const_cast<ColorSpinorField &>(v));
      F uvAccessor(const_cast<ColorSpinorField &>(uv));
      gFine gAccessor(const_cast<GaugeField &>(g));
      cFine cAccessor(const_cast<GaugeField &>(clover));
      cFine cInvAccessor(const_cast<GaugeField &>(cloverInv));
      gCoarse yAccessor(const_cast<GaugeField &>(Y));
      gCoarse xAccessor(const_cast<GaugeField &>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField &>(Yatomic));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField &>(Xatomic));

      calculateY<use_mma, QUDA_CPU_FIELD_LOCATION, true, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor, cAccessor,
        cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v), v, kappa, mass, mu, mu_factor,
        dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()), T.coarseToFine(Y.Location()));

    } else {

      constexpr QudaFieldOrder csOrder = QUDA_FLOAT2_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using V = typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat>;
      using F = typename colorspinor::FieldOrderCB<Float, 2 * fineSpin, fineColor, coarseColor, csOrder, vFloat>;
      using gFine = typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat>;
      using cFine = typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat>;
      using gCoarse = typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, vFloat>;
      using gCoarseAtomic =
        typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, storeType>;

      const ColorSpinorField &v = T.Vectors(Y.Location());

      V vAccessor(const_cast<ColorSpinorField &>(v));
      F uvAccessor(const_cast<ColorSpinorField &>(uv));
      gFine gAccessor(const_cast<GaugeField &>(g));
      cFine cAccessor(const_cast<GaugeField &>(clover));
      cFine cInvAccessor(const_cast<GaugeField &>(cloverInv));
      gCoarse yAccessor(const_cast<GaugeField &>(Y));
      gCoarse xAccessor(const_cast<GaugeField &>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField &>(Yatomic));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField &>(Xatomic));

      // create a dummy clover field to allow us to call the external clover reduction routines elsewhere
      calculateY<use_mma, QUDA_CUDA_FIELD_LOCATION, true, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor, cAccessor,
        cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v), v, kappa, mass, mu, mu_factor,
        dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()), T.coarseToFine(Y.Location()));
    }
  }

  template <bool use_mma, typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  std::enable_if_t<use_mma, void> calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic,
                                                   ColorSpinorField &uv, const Transfer &T, const GaugeField &g,
                                                   const GaugeField &clover, const GaugeField &cloverInv, double kappa,
                                                   double mass, double mu, double mu_factor, QudaDiracType dirac,
                                                   QudaMatPCType matpc, bool need_bidirectional)
  {
    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {
      errorQuda("use_mma not supported on the CPU");
    } else {

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

      using V = typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat>;
      using F = typename colorspinor::FieldOrderCB<Float, 2 * fineSpin, fineColor, coarseColor, csOrder, vFloat>;
      using gFine = typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat>;
      using cFine = typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat>;
      using gCoarse = typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, vFloat>;
      using gCoarseAtomic =
        typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, storeType>;

      const ColorSpinorField &v = T.Vectors(Y.Location());
      ColorSpinorParam param_v(v);
      param_v.fieldOrder = csOrder;
      param_v.setPrecision(v.Precision());
      cudaColorSpinorField v_(param_v);
      v_.copy(v);

      V vAccessor(v_);
      F uvAccessor(const_cast<ColorSpinorField &>(uv));
      gFine gAccessor(const_cast<GaugeField &>(g));
      cFine cAccessor(const_cast<GaugeField &>(clover));
      cFine cInvAccessor(const_cast<GaugeField &>(cloverInv));

      gCoarse yAccessor(const_cast<GaugeField &>(Y));
      gCoarse xAccessor(const_cast<GaugeField &>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField &>(Yatomic));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField &>(Xatomic));

      // create a dummy clover field to allow us to call the external clover reduction routines elsewhere
      calculateY<use_mma, QUDA_CUDA_FIELD_LOCATION, true, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor, cAccessor,
        cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<cudaColorSpinorField &>(v_), v_, kappa, mass, mu,
        mu_factor, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()), T.coarseToFine(Y.Location()));
    }
  }

  // template on fine colors
  template <bool use_mma, typename Float, typename vFloat, int fineSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac,
                        QudaMatPCType matpc, bool need_bidirectional)
  {
    if (T.Vectors().Nspin() / T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d\n", T.Vectors().Nspin() / T.Spin_bs());
    const int fineColor = g.Ncolor() / fineSpin;
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;
#ifdef NSPIN4
    if (fineColor == 6) { // free field Wilson
      if (coarseColor == 6) {
        calculateYcoarse<use_mma, Float, vFloat, 6, fineSpin, 6, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                             cloverInv, kappa, mass, mu, mu_factor,
                                                                             dirac, matpc, need_bidirectional);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
    } else
#endif
      if (fineColor == 24) { // coarsened Wilson or free field staggered
      if (coarseColor == 24) {
        calculateYcoarse<use_mma, Float, vFloat, 24, fineSpin, 24, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                               cloverInv, kappa, mass, mu, mu_factor,
                                                                               dirac, matpc, need_bidirectional);
      } else
#ifdef NSPIN4
        if (coarseColor == 32) {
        calculateYcoarse<use_mma, Float, vFloat, 24, fineSpin, 32, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                               cloverInv, kappa, mass, mu, mu_factor,
                                                                               dirac, matpc, need_bidirectional);
      } else
#endif // NSPIN4
#ifdef NSPIN1
        if (coarseColor == 64) {
        calculateYcoarse<use_mma, Float, vFloat, 24, fineSpin, 64, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                               cloverInv, kappa, mass, mu, mu_factor,
                                                                               dirac, matpc, need_bidirectional);
      } else // --- note, coarsening Nc == 24 -> Nc == 96 for staggered is worth revisiting in the future
#endif
      {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
#ifdef NSPIN4
    } else if (fineColor == 32) {
      if (coarseColor == 32) {
        calculateYcoarse<use_mma, Float, vFloat, 32, fineSpin, 32, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                               cloverInv, kappa, mass, mu, mu_factor,
                                                                               dirac, matpc, need_bidirectional);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
#endif // NSPIN4
#ifdef NSPIN1
    } else if (fineColor == 64) {
      if (coarseColor == 64) {
        calculateYcoarse<use_mma, Float, vFloat, 64, fineSpin, 64, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                               cloverInv, kappa, mass, mu, mu_factor,
                                                                               dirac, matpc, need_bidirectional);
      } else if (coarseColor == 96) {
        calculateYcoarse<use_mma, Float, vFloat, 64, fineSpin, 96, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                               cloverInv, kappa, mass, mu, mu_factor,
                                                                               dirac, matpc, need_bidirectional);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      } // --- note, revisit Nc == 96 -> Nc == 96 in the future
#endif // NSPIN1
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  // template on fine spin
  template <bool use_mma, typename Float, typename vFloat>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac,
                        QudaMatPCType matpc, bool need_bidirectional)
  {
    if (T.Vectors().Nspin() == 2) {
      calculateYcoarse<use_mma, Float, vFloat, 2>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu,
                                                  mu_factor, dirac, matpc, need_bidirectional);
    } else {
      errorQuda("Unsupported number of spins %d\n", T.Vectors().Nspin());
    }
  }

  // Does the heavy lifting of creating the coarse color matrices Y
#ifdef GPU_MULTIGRID
  template <bool use_mma>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac,
                        QudaMatPCType matpc, bool need_bidirectional)
  {
    checkPrecision(X, Y, g, clover, cloverInv, uv, T.Vectors(X.Location()));
    checkPrecision(Xatomic, Yatomic);

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (use_mma) errorQuda("MG-MMA does not support double precision, yet.");
      if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
        calculateYcoarse<use_mma, double, double>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu,
                                                  mu_factor, dirac, matpc, need_bidirectional);
      } else {
        errorQuda("Unsupported precision %d\n", Y.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
      if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
        calculateYcoarse<use_mma, float, float>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu,
                                                mu_factor, dirac, matpc, need_bidirectional);
      } else {
        errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
#else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
    } else if (Y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
        calculateYcoarse<use_mma, float, short>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu,
                                                mu_factor, dirac, matpc, need_bidirectional);
      } else {
        errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("....done computing Y field\n");
  }
#else
  template <bool use_mma>
  void calculateYcoarse(GaugeField &, GaugeField &, GaugeField &, GaugeField &, ColorSpinorField &, const Transfer &,
                        const GaugeField &, const GaugeField &, const GaugeField &, double, double, double, double,
                        QudaDiracType, QudaMatPCType, bool)
  {
    errorQuda("Multigrid has not been built");
  }
#endif // GPU_MULTIGRID

} // namespace quda
