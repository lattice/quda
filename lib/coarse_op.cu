#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <blas_magma.h>
#include <coarse_op.cuh>

namespace quda {

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder,
            int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {

    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;
    typedef typename clover::FieldOrder<Float,fineColor,fineSpin,clOrder> cFine;

    const ColorSpinorField &v = T.Vectors();
    int dummy = 0;
    v.exchangeGhost(QUDA_INVALID_PARITY, dummy);

    F vAccessor(const_cast<ColorSpinorField&>(v));
    F uvAccessor(const_cast<ColorSpinorField&>(uv));
    F avAccessor(const_cast<ColorSpinorField&>(av));
    gFine gAccessor(const_cast<GaugeField&>(g));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X));
    gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));
    cFine cAccessor(const_cast<CloverField&>(c), false);
    cFine cInvAccessor(const_cast<CloverField&>(cI), true);

    calculateY<false,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>
      (yAccessor, xAccessor, xInvAccessor, uvAccessor, avAccessor, vAccessor, gAccessor, cAccessor, cInvAccessor, Y, X, Xinv, Yhat, av, v, kappa, mu, dirac, matpc);
  }

  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder,
	    int fineColor, int fineSpin>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,2,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else if (coarseColor == 4) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,4,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else if (coarseColor == 8) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,8,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else if (coarseColor == 12) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,12,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else if (coarseColor == 16) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,16,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else if (coarseColor == 20) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,20,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else if (coarseColor == 24) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,24,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else if (coarseColor == 32) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,32,coarseSpin>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder, int fineColor>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    if (uv.Nspin() == 4) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,4>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      errorQuda("Unsupported number of spins %d\n", uv.Nspin());
    }
  }

  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    if (g.Ncolor() == 3) {
      calculateY<Float,csOrder,gOrder,clOrder,3>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    //If c == NULL, then this is standard Wilson.  csOrder is dummy and will not matter      
    if (c.Order() == QUDA_PACKED_CLOVER_ORDER) {
      calculateY<Float,csOrder,gOrder,QUDA_PACKED_CLOVER_ORDER>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      errorQuda("Unsupported field order %d\n", c.Order());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    if (g.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      calculateY<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      errorQuda("Unsupported field order %d\n", g.FieldOrder());
    }
  }

 template <typename Float>
 void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		 const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateY<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv, ColorSpinorField &av, const Transfer &T,
		  const GaugeField &g, const CloverField &c, const CloverField &cI, double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() ||
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != g.Precision())
      errorQuda("Unsupported precision mix");

    printfQuda("Computing Y field......\n");

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      calculateY<double>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateY<float>(Y, X, Xinv, Yhat, uv, av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, const Transfer &T,
		const cudaGaugeField &gauge, const cudaCloverField *clover, const cudaCloverField *cloverInv,
		double kappa, double mu, QudaDiracType dirac, QudaMatPCType matpc) {
    QudaPrecision precision = Y.Precision();
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

    cpuGaugeField g(gf_param);

    //Copy the cuda gauge field to the cpu
    gauge.saveCPUField(g, QUDA_CPU_FIELD_LOCATION);

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = QUDA_CPU_FIELD_LOCATION;

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    // if we are coarsening a preconditioned clover or twisted-mass operator we need
    // an additional vector to store the cloverInv * V field, else just alias v
    ColorSpinorField *av = ((matpc != QUDA_MATPC_INVALID && clover) || (dirac == QUDA_TWISTED_MASSPC_DIRAC)) ? ColorSpinorField::Create(UVparam) :
      &const_cast<ColorSpinorField&>(T.Vectors());

    //If the fine lattice operator is the clover operator, copy the cudaCloverField to cpuCloverField

    //Create a cpuCloverField from the cudaCloverField
    CloverFieldParam cf_param;
    cf_param.nDim = 4;
    cf_param.pad = pad;
    cf_param.precision = clover ? clover->Precision() : QUDA_INVALID_PRECISION;

    // if we have no clover term then create an empty clover field
    for(int i = 0; i < cf_param.nDim; i++) cf_param.x[i] = clover ? clover->X()[i] : 0;

    cf_param.order = QUDA_PACKED_CLOVER_ORDER;
    cf_param.direct = true;
    cf_param.inverse = true;
    cf_param.clover = NULL;
    cf_param.norm = 0;
    cf_param.cloverInv = NULL;
    cf_param.invNorm = 0;
    cf_param.create = QUDA_NULL_FIELD_CREATE;
    cf_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    if (cloverInv && (dirac == QUDA_TWISTED_CLOVERPC_DIRAC)) {
      cf_param.direct = false;
      cpuCloverField cI(cf_param);
      cloverInv->saveCPUField(cI);
      cf_param.direct = true;
      cpuCloverField c(cf_param);
      clover->saveCPUField(c);
      calculateY(Y, X, Xinv, Yhat, *uv, *av, T, g, c, cI, kappa, mu, dirac, matpc);
    } else {
      cpuCloverField c(cf_param);
      if (clover) clover->saveCPUField(c);
      calculateY(Y, X, Xinv, Yhat, *uv, *av, T, g, c, c, kappa, mu, dirac, matpc);
    }

    if (&T.Vectors() != av) delete av;
    delete uv;
  }

} //namespace quda
