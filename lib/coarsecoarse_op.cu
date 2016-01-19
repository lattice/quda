#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <blas_magma.h>
#include <coarse_op.cuh>

namespace quda {

  /*
    To do setup when the source is preconditioned links we have to run
    setup twice: once for forwards links, and once for backward links.
    Then need to combine the resulting X fields, which would replace
    the createCoarseLocal function that's currently done.
   */

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, 
            int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {

    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename colorspinor::FieldOrderCB<Float,2*fineSpin,fineColor,coarseColor,csOrder> F2;
    typedef typename gauge::FieldOrder<Float,fineColor*fineSpin,fineSpin,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;

    const ColorSpinorField &v = T.Vectors();

    F vAccessor(const_cast<ColorSpinorField&>(v));
    F2 uvAccessor(const_cast<ColorSpinorField&>(uv));
    gFine gAccessor(const_cast<GaugeField&>(g));
    gFine cAccessor(const_cast<GaugeField&>(clover));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X)); 

    calculateY<true,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>
      (yAccessor, xAccessor, uvAccessor, vAccessor, gAccessor, cAccessor, Y, X, Xinv, Yhat, v, kappa);
  }


  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor, int fineSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2) 
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) { 
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,2,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else if (coarseColor == 8) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,8,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else if (coarseColor == 16) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,16,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else if (coarseColor == 24) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,24,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (T.Vectors().Nspin() == 2) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,2>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported number of spins %d\n", T.Vectors().Nspin());
    }
  }

  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (g.Ncolor()/T.Vectors().Nspin() == 24) {
      calculateYcoarse<Float,csOrder,gOrder,24>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else if (g.Ncolor()/T.Vectors().Nspin() == 2) {
      calculateYcoarse<Float,csOrder,gOrder,2>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else if (g.Ncolor()/T.Vectors().Nspin() == 8) {
      calculateYcoarse<Float,csOrder,gOrder,8>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else if (g.Ncolor()/T.Vectors().Nspin() == 16) {
      calculateYcoarse<Float,csOrder,gOrder,16>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (g.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      calculateYcoarse<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", g.FieldOrder());
    }
  }

  template <typename Float>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateYcoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv,
			const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() || 
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != g.Precision())
      errorQuda("Unsupported precision mix");

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateYcoarse<double>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateYcoarse<float>(Y, X, Xinv, Yhat, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
		      const Transfer &T, const cpuGaugeField &gauge, const cpuGaugeField &clover, double kappa) {

    QudaPrecision precision = Y.Precision();
    //First make a cpu gauge field from the cuda gauge field

#if 0
    GaugeFieldParam gf_param(gauge.X(), precision, gauge.Reconstruct(), pad, gauge.Geometry());
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
#endif


    //Create a field UV which holds U*V.  Has the same similar
    //structure to V but double the number of spins so we can store
    //the four distinct block chiral multiplications in a single UV
    //computation.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.nSpin *= 2; // so nSpin == 4
    cpuColorSpinorField uv(UVparam);

    calculateYcoarse(Y, X, Xinv, Yhat, uv, T, gauge, clover, kappa);

  }
  
} //namespace quda
