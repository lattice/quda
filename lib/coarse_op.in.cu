#include <transfer.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>

// This define controls which kernels get compiled in `coarse_op.cuh`.
// This ensures only kernels relevant for coarsening Wilson-type
// operators get built, saving compile time.
#define WILSONCOARSE
#include <coarse_op.cuh>
#include "multigrid.h"

namespace quda {

  template <typename Float, typename vFloat, int fineColor, int coarseColor>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic,
                  ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc)
  {
    QudaFieldLocation location = Y.Location();
    constexpr int fineSpin = 4;
    constexpr int coarseSpin = 2;
    constexpr bool use_mma = false;
    constexpr bool staggered_allow_truncation = false;

    if (uv.Nspin() != 4) errorQuda("Unsupported number of spins %d", uv.Nspin());
    if (Y.Ncolor() / coarseSpin != coarseColor)
      errorQuda("Unexpected coarse color %d doesn't match template %d", Y.Ncolor() / coarseSpin, coarseColor);
    if (T.Vectors().Nspin() / T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d", T.Vectors().Nspin() / T.Spin_bs());

    bool need_bidirectional = false;
    if (dirac == QUDA_CLOVERPC_DIRAC || dirac == QUDA_TWISTED_MASSPC_DIRAC || dirac == QUDA_TWISTED_CLOVERPC_DIRAC) { need_bidirectional = true; }

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;
      constexpr QudaCloverFieldOrder clOrder = QUDA_PACKED_CLOVER_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d", g.FieldOrder());
      if (c.Order() != clOrder && c.Bytes()) errorQuda("Unsupported field order %d", c.Order());

      using V = typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat>;
      using F = typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat>;
      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder>;
      using gCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;
      using gCoarseAtomic = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,storeType>;
      using cFine = typename clover::FieldOrder<Float,fineColor,fineSpin,clOrder>;

      const ColorSpinorField &v = T.Vectors(g.Location());

      V vAccessor(const_cast<ColorSpinorField&>(v));
      F uvAccessor(const_cast<ColorSpinorField&>(uv));
      F avAccessor(const_cast<ColorSpinorField&>(av));
      gFine gAccessor(const_cast<GaugeField&>(g));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField&>(Yatomic));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField&>(Xatomic));
      cFine cAccessor(const_cast<CloverField&>(c), false);
      cFine cInvAccessor(const_cast<CloverField&>(c), true);

      calculateY<use_mma, QUDA_CPU_FIELD_LOCATION, false,Float,fineSpin,fineColor,coarseSpin,coarseColor>
	(yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor,
	 avAccessor, vAccessor, gAccessor, gAccessor, gAccessor, cAccessor, cInvAccessor, Y, X, Yatomic, Xatomic, uv, av, v,
         kappa, mass, mu, mu_factor, staggered_allow_truncation, dirac, matpc, need_bidirectional,
	 T.fineToCoarse(location), T.coarseToFine(location));

    } else {

      constexpr QudaFieldOrder csOrder = colorspinor::getNative<vFloat>(fineSpin);
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;
      constexpr QudaCloverFieldOrder clOrder = QUDA_FLOAT4_CLOVER_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d", g.FieldOrder());
      if (c.Order() != clOrder && c.Bytes()) errorQuda("Unsupported field order %d", c.Order());

      using F = typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat,vFloat,false,false>;
      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float>;
      using gCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;
      using gCoarseAtomic = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,storeType>;
      using cFine = typename clover::FieldOrder<Float,fineColor,fineSpin,clOrder>;

      const ColorSpinorField &v = T.Vectors(g.Location());

      F vAccessor(const_cast<ColorSpinorField&>(v));
      F uvAccessor(const_cast<ColorSpinorField&>(uv));
      F avAccessor(const_cast<ColorSpinorField&>(av));
      gFine gAccessor(const_cast<GaugeField&>(g));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField&>(Yatomic));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField&>(Xatomic));
      cFine cAccessor(const_cast<CloverField&>(c), false);
      cFine cInvAccessor(const_cast<CloverField&>(c), true);

      calculateY<use_mma, QUDA_CUDA_FIELD_LOCATION, false,Float,fineSpin,fineColor,coarseSpin,coarseColor>
        (yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor,
         avAccessor, vAccessor, gAccessor, gAccessor, gAccessor, cAccessor, cInvAccessor, Y, X, Yatomic, Xatomic, uv, av, v,
         kappa, mass, mu, mu_factor, staggered_allow_truncation, dirac, matpc, need_bidirectional,
         T.fineToCoarse(location), T.coarseToFine(location));
    }

  }

  template <int fineColor, int coarseColor>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic,
                  ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T, const GaugeField &g,
                  const CloverField &c, double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if constexpr (is_enabled_multigrid()) {
      checkPrecision(Xatomic, Yatomic, g);
      checkPrecision(uv, av, T.Vectors(X.Location()), X, Y);
      logQuda(QUDA_SUMMARIZE, "Computing Y field......\n");

      if (!is_enabled(Y.Precision())) errorQuda("QUDA_PRECISION=%d does not enable %d precision", QUDA_PRECISION, Y.Precision());
      if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
        if constexpr (is_enabled_multigrid_double()) {
          calculateY<double, double, fineColor, coarseColor>(Y, X, Yatomic, Xatomic, uv, av, T, g, c, kappa, mass, mu, mu_factor, dirac, matpc);
        } else {
          errorQuda("Double precision multigrid has not been enabled");
        }
      } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
        if constexpr(is_enabled(QUDA_SINGLE_PRECISION)) {
          calculateY<float, float, fineColor, coarseColor>(Y, X, Yatomic, Xatomic, uv, av, T, g, c, kappa, mass, mu, mu_factor, dirac, matpc);
        }
      } else if (Y.Precision() == QUDA_HALF_PRECISION) {
        if constexpr(is_enabled(QUDA_HALF_PRECISION)) {
          calculateY<float, short, fineColor, coarseColor>(Y, X, Yatomic, Xatomic, uv, av, T, g, c, kappa, mass, mu, mu_factor, dirac, matpc);
        }
      } else {
        errorQuda("Unsupported precision %d", Y.Precision());
      }
      logQuda(QUDA_SUMMARIZE, "....done computing Y field\n");
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

  constexpr int fineColor = 3;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC@;

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  template <>
  void CoarseOp<fineColor, coarseColor>(GaugeField &Y, GaugeField &X, const Transfer &T,
                                        const GaugeField &gauge, const CloverField *clover,
                                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc)
  {
    QudaPrecision precision = Y.Precision();
    QudaFieldLocation location = checkLocation(Y, X);

    GaugeField *U = location == QUDA_CUDA_FIELD_LOCATION ? const_cast<GaugeField*>(&gauge) : nullptr;
    CloverField *C = location == QUDA_CUDA_FIELD_LOCATION ? const_cast<CloverField*>(clover) : nullptr;

    if (location == QUDA_CPU_FIELD_LOCATION) {
      //First make a cpu gauge field from the cuda gauge field
      int pad = 0;
      GaugeFieldParam gf_param(gauge.X(), precision, QUDA_RECONSTRUCT_NO, pad, gauge.Geometry());
      gf_param.order = QUDA_QDP_GAUGE_ORDER;
      gf_param.fixed = gauge.GaugeFixed();
      gf_param.link_type = gauge.LinkType();
      gf_param.t_boundary = gauge.TBoundary();
      gf_param.anisotropy = gauge.Anisotropy();
      gf_param.gauge = NULL;
      gf_param.create = QUDA_NULL_FIELD_CREATE;
      gf_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      gf_param.nFace = 1;
      gf_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

      U = new cpuGaugeField(gf_param);

      //Copy the cuda gauge field to the cpu
      static_cast<const cudaGaugeField&>(gauge).saveCPUField(*static_cast<cpuGaugeField*>(U));
    } else if (location == QUDA_CUDA_FIELD_LOCATION && gauge.Reconstruct() != QUDA_RECONSTRUCT_NO) {
      //Create a copy of the gauge field with no reconstruction, required for fine-grained access
      GaugeFieldParam gf_param(gauge);
      gf_param.reconstruct = QUDA_RECONSTRUCT_NO;
      gf_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      gf_param.setPrecision(gf_param.Precision());
      U = new cudaGaugeField(gf_param);

      U->copy(gauge);
    }

    CloverFieldParam cf_param;
    cf_param.nDim = 4;
    cf_param.pad = 0;
    cf_param.setPrecision(clover ? clover->Precision() : QUDA_SINGLE_PRECISION);

    // if we have no clover term then create an empty clover field
    for (int i = 0; i < cf_param.nDim; i++) cf_param.x[i] = clover ? clover->X()[i] : 0;

    // only create inverse if not doing dynamic clover and one already exists
    cf_param.inverse = !clover::dynamic_inverse() && clover && clover->V(true);
    cf_param.clover = nullptr;
    cf_param.cloverInv = nullptr;
    cf_param.create = QUDA_NULL_FIELD_CREATE;
    cf_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    cf_param.location = location;
    cf_param.reconstruct = false;

    if (location == QUDA_CUDA_FIELD_LOCATION && !clover) {
      // create a dummy CloverField if one is not defined
      cf_param.order = QUDA_INVALID_CLOVER_ORDER;
      C = new CloverField(cf_param);
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      // Create a host field from the device one
      cf_param.order = QUDA_PACKED_CLOVER_ORDER;
      C = new CloverField(cf_param);
      if (clover) C->copy(*clover);
    }

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors(location));
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.setPrecision(T.Vectors(location).Precision());
    UVparam.mem_type = Y.MemType(); // allocate temporaries to match coarse-grid link field

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    // if we are coarsening a preconditioned clover or twisted-mass operator we need
    // an additional vector to store the cloverInv * V field, else just alias v
    ColorSpinorField *av = ((matpc != QUDA_MATPC_INVALID && clover) || (dirac == QUDA_TWISTED_MASSPC_DIRAC)) ?
      ColorSpinorField::Create(UVparam) : &const_cast<ColorSpinorField&>(T.Vectors(location));

    GaugeField *Yatomic = &Y;
    GaugeField *Xatomic = &X;
    if (Y.Precision() < QUDA_SINGLE_PRECISION) {
      // we need to coarsen into single precision fields (float or int), so we allocate temporaries for this purpose
      // else we can just coarsen directly into the original fields
      GaugeFieldParam param(X); // use X since we want scalar geometry
      param.location = location;
      param.setPrecision(QUDA_SINGLE_PRECISION, location == QUDA_CUDA_FIELD_LOCATION ? true : false);

      Yatomic = GaugeField::Create(param);
      Xatomic = GaugeField::Create(param);
    }

    calculateY<fineColor, coarseColor>(Y, X, *Yatomic, *Xatomic, *uv, *av, T, *U, *C, kappa, mass, mu, mu_factor, dirac, matpc);

    if (Yatomic != &Y) delete Yatomic;
    if (Xatomic != &X) delete Xatomic;

    if (&T.Vectors(location) != av) delete av;
    delete uv;

    if (C != clover) delete C;
    if (U != &gauge) delete U;
  }

} //namespace quda
