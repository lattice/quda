#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <blas_cublas.h>

// this is the storage type used when computing the coarse link variables
// by using integers we have deterministic atomics
typedef int storeType;

#include <staggered_coarse_op.cuh>

namespace quda {

  template <typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T,
		  const GaugeField &g, double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    QudaFieldLocation location = Y.Location();

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;
      constexpr QudaCloverFieldOrder clOrder = QUDA_PACKED_CLOVER_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
	errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat> V;
      typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat> F;
      typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,storeType> gCoarseAtomic;

      const ColorSpinorField &v = T.Vectors(g.Location());

      V vAccessor(const_cast<ColorSpinorField&>(v));
      F uvAccessor(const_cast<ColorSpinorField&>(uv));
      gFine gAccessor(const_cast<GaugeField&>(g));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField&>(Y));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField&>(X));

      calculateY<Float,fineSpin,fineColor,coarseSpin,coarseColor>
	(yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor,
	 vAccessor, gAccessor, Y, X, uv, v, mass, dirac, matpc,
	 T.fineToCoarse(location), T.coarseToFine(location));

    } else {

      constexpr QudaFieldOrder csOrder = QUDA_FLOAT2_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
	errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat> F;
      typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,storeType> gCoarseAtomic;

      const ColorSpinorField &v = T.Vectors(g.Location());

      F vAccessor(const_cast<ColorSpinorField&>(v));
      F uvAccessor(const_cast<ColorSpinorField&>(uv));
      gFine gAccessor(const_cast<GaugeField&>(g));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField&>(Y));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField&>(X));

      calculateStaggeredY<Float,fineSpin,fineColor,coarseSpin,coarseColor>
	(yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor,
	 vAccessor, gAccessor, Y, X, uv, v, mass, dirac, matpc,
	 T.fineToCoarse(location), T.coarseToFine(location));

    }

  }

  // template on the number of coarse degrees of freedom
  template <typename Float, typename vFloat, int fineColor, int fineSpin>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T,
		  const GaugeField &g, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().Nspin() != 1)
      errorQuda("Unsupported fine spin %d\n",T.Vectors().Nspin());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

#if 0
    } else if (coarseCoor == 4) {
      calculateY<Float,vFloat,fineColor,fineSpin,4,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
#endif
    if (coarseColor == 24) { // free field staggered
      calculateStaggeredY<Float,vFloat,fineColor,fineSpin,24,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
#if 0
    } else if (coarseColor == 8) {
      calculateY<Float,vFloat,fineColor,fineSpin,8,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
    } else if (coarseColor == 12) {
      calculateY<Float,vFloat,fineColor,fineSpin,12,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
    } else if (coarseColor == 16) {
      calculateY<Float,vFloat,fineColor,fineSpin,16,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
    } else if (coarseColor == 20) {
      calculateY<Float,vFloat,fineColor,fineSpin,20,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
#endif
    } else if (coarseColor == 96) {
      calculateStaggeredY<Float,vFloat,fineColor,fineSpin,96,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
    } else if (coarseColor == 128) {
      calculateStaggeredY<Float,vFloat,fineColor,fineSpin,128,coarseSpin>(Y, X, uv, av, T, g, c, kappa, mu, mu_factor, dirac, matpc);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, typename vFloat, int fineColor>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (uv.Nspin() == 1) {
      calculateStaggeredY<Float,vFloat,fineColor,1>(Y, X, uv, T, g, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of spins %d\n", uv.Nspin());
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T,
		  const GaugeField &g, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (g.Ncolor() == 3) {
      calculateStaggeredY<Float,vFloat,3>(Y, X, uv, T, g, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, ColorSpinorField &uv, const Transfer &T,
		  const GaugeField &g, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    checkPrecision(X, Y, g);
    checkPrecision(uv, T.Vectors(X.Location()));

    printfQuda("Computing Y field......\n");

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
        calculateStaggeredY<double,double>(Y, X, uv, T, g, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", Y.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
        calculateStaggeredY<float,float>(Y, X, uv, T, g, mass, dirac, matpc);
      } else if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
        calculateStaggeredY<float,short>(Y, X, uv, T, g, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void StaggeredCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T,
		const cudaGaugeField &gauge,
		double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    QudaPrecision precision = Y.Precision();
    QudaFieldLocation location = checkLocation(Y, X);

    GaugeField *U = location == QUDA_CUDA_FIELD_LOCATION ? const_cast<cudaGaugeField*>(&gauge) : nullptr;

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
      gauge.saveCPUField(*static_cast<cpuGaugeField*>(U));
    } else if (location == QUDA_CUDA_FIELD_LOCATION && gauge.Reconstruct() != QUDA_RECONSTRUCT_NO) {
      //Create a copy of the gauge field with no reconstruction, required for fine-grained access
      GaugeFieldParam gf_param(gauge);
      gf_param.reconstruct = QUDA_RECONSTRUCT_NO;
      gf_param.setPrecision(gf_param.precision);
      U = new cudaGaugeField(gf_param);

      U->copy(gauge);
    }

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors(location));
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.precision = T.Vectors(location).Precision();

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    calculateStaggeredY(Y, X, *uv, T, *U, mass, dirac, matpc);

    delete uv;

    if (U != &gauge) delete U;
  }

} //namespace quda
