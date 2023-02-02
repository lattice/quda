#include <transfer.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

// This define controls which kernels get compiled in `coarse_op.cuh`.
// This ensures only kernels relevant for coarsening a coarse operator
// get built, saving compile time.
#define COARSECOARSE
#include <coarse_op.cuh>
#include "multigrid.h"

namespace quda
{

  template <bool use_mma, typename Float, typename vFloat, int fineColor, int coarseColor>
  std::enable_if_t<!use_mma, void>
  calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                   const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                   double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                   bool need_bidirectional)
  {
    constexpr int fineSpin = 2;
    constexpr int coarseSpin = 2;
    constexpr bool allow_truncation = false;

    if (g.Ncolor() / fineSpin != fineColor)
      errorQuda("Unexpected fine color %d doesn't match template %d", g.Ncolor() / fineSpin, fineColor);
    if (Y.Ncolor() / coarseSpin != coarseColor)
      errorQuda("Unexpected coarse color %d doesn't match template %d", Y.Ncolor() / coarseSpin, coarseColor);
    if (T.Vectors().Nspin() != 2) errorQuda("Unsupported number of fine spins %d", T.Vectors().Nspin());
    if (T.Vectors().Nspin() / T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d", T.Vectors().Nspin() / T.Spin_bs());

    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {
      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d", g.FieldOrder());

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
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor, gAccessor,
        gAccessor, cAccessor, cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v), v, kappa,
        mass, mu, mu_factor, allow_truncation, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
        T.coarseToFine(Y.Location()));

    } else {

      constexpr QudaFieldOrder csOrder = colorspinor::getNative<vFloat>(fineSpin);
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d", g.FieldOrder());

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
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor, gAccessor,
        gAccessor, cAccessor, cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v), v, kappa,
        mass, mu, mu_factor, allow_truncation, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
        T.coarseToFine(Y.Location()));
    }
  }

  template <bool use_mma, typename Float, typename vFloat, int fineColor, int coarseColor>
  std::enable_if_t<use_mma, void> calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic,
                                                   ColorSpinorField &uv, const Transfer &T, const GaugeField &g,
                                                   const GaugeField &clover, const GaugeField &cloverInv, double kappa,
                                                   double mass, double mu, double mu_factor, QudaDiracType dirac,
                                                   QudaMatPCType matpc, bool need_bidirectional)
  {
    constexpr int fineSpin = 2;
    constexpr int coarseSpin = 2;
    constexpr bool allow_truncation = false;

    if (g.Ncolor() / fineSpin != fineColor)
      errorQuda("Unexpected fine color %d doesn't match template %d", g.Ncolor() / fineSpin, fineColor);
    if (Y.Ncolor() / coarseSpin != coarseColor)
      errorQuda("Unexpected coarse color %d doesn't match template %d", Y.Ncolor() / coarseSpin, coarseColor);
    if (T.Vectors().Nspin() != 2) errorQuda("Unsupported number of fine spins %d", T.Vectors().Nspin());
    if (T.Vectors().Nspin() / T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d", T.Vectors().Nspin() / T.Spin_bs());

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
      ColorSpinorField v_(param_v);
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
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor, gAccessor,
        gAccessor, cAccessor, cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v_), v_, kappa,
        mass, mu, mu_factor, allow_truncation, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
        T.coarseToFine(Y.Location()));
    }
  }

  // Does the heavy lifting of creating the coarse color matrices Y
  template <bool use_mma, int fineColor, int coarseColor>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac,
                        QudaMatPCType matpc, bool need_bidirectional)
  {
    if constexpr (is_enabled_multigrid()) {
      checkPrecision(X, Y, g, clover, cloverInv, uv, T.Vectors(X.Location()));
      checkPrecision(Xatomic, Yatomic);
      if (!is_enabled(Y.Precision()))
        errorQuda("QUDA_PRECISION=%d does not enable %d precision", QUDA_PRECISION, Y.Precision());

      logQuda(QUDA_SUMMARIZE, "Computing Y field......\n");
      if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double()) {
          if (use_mma) errorQuda("MG-MMA does not support double precision, yet.");
          if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
            calculateYcoarse<use_mma, double, double, fineColor, coarseColor>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                              cloverInv, kappa, mass, mu, mu_factor,
                                                                              dirac, matpc, need_bidirectional);
          } else {
            errorQuda("Unsupported precision %d", Y.Precision());
          }
        } else {
          errorQuda("Double precision multigrid has not been enabled");
        }
      } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) {
          if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
            calculateYcoarse<use_mma, float, float, fineColor, coarseColor>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                            cloverInv, kappa, mass, mu, mu_factor,
                                                                            dirac, matpc, need_bidirectional);
          } else {
            errorQuda("Unsupported precision %d", T.Vectors(X.Location()).Precision());
          }
        }
      } else if (Y.Precision() == QUDA_HALF_PRECISION) {
        if constexpr (is_enabled(QUDA_HALF_PRECISION)) {
          if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
            calculateYcoarse<use_mma, float, short, fineColor, coarseColor>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                            cloverInv, kappa, mass, mu, mu_factor,
                                                                            dirac, matpc, need_bidirectional);
          } else {
            errorQuda("Unsupported precision %d", T.Vectors(X.Location()).Precision());
          }
        }
      } else {
        errorQuda("Unsupported precision %d", Y.Precision());
      }
      logQuda(QUDA_SUMMARIZE, "....done computing Y field\n");
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
