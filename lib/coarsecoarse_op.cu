#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <coarse_op.cuh>
#include <blas_magma.h>

namespace quda {

  template<typename Float, int coarseSpin, int coarseColor,
	   typename F, typename coarseGauge, typename fineGauge>
  void calculateYcoarse(coarseGauge &Y, coarseGauge &X, F &UV, F &V, fineGauge &G, fineGauge &C,
			const int *xx_size, const int *xc_size, double kappa) {
    if (UV.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) errorQuda("Gamma basis not supported");

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int x_size[5];
    for (int i=0; i<4; i++) x_size[i] = xx_size[i];
    x_size[4] = 1;

    int comm_dim[nDim];
    for (int i=0; i<nDim; i++) comm_dim[i] = comm_dim_partitioned(i);

    int geo_bs[QUDA_MAX_DIM]; 
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];
    int spin_bs = V.Nspin()/Y.NspinCoarse();

    Gamma<Float, QUDA_INVALID_GAMMA_BASIS, 0> dummy;

    for(int d = 0; d < nDim; d++) {
      for(int s = 0; s < V.Nspin(); s++) {
        //First calculate UV
        setZero<Float,F>(UV);

        printfQuda("Computing %d UV and VUV s=%d\n", d, s);
        //Calculate UV and then VUV for this direction, accumulating directly into the coarse gauge field Y
        if (d==0) {
          computeUV<true,Float,0>(UV, V, G, nDim, x_size, comm_dim, s);
          computeVUV<true,Float,0>(Y, X, UV, V, dummy, G, x_size, xc_size, geo_bs, spin_bs, s);
        } else if (d==1) {
          computeUV<true,Float,1>(UV, V, G, nDim, x_size, comm_dim, s);
          computeVUV<true,Float,1>(Y, X, UV, V, dummy, G, x_size, xc_size, geo_bs, spin_bs, s);
        } else if (d==2) {
          computeUV<true,Float,2>(UV, V, G, nDim, x_size, comm_dim, s);
          computeVUV<true,Float,2>(Y, X, UV, V, dummy, G, x_size, xc_size, geo_bs, spin_bs, s);
        } else {
          computeUV<true,Float,3>(UV, V, G, nDim, x_size, comm_dim, s);
          computeVUV<true,Float,3>(Y, X, UV, V, dummy, G, x_size, xc_size, geo_bs, spin_bs, s);
        }
      }
      printfQuda("UV2[%d] = %e\n", d, UV.norm2());
      printfQuda("Y2[%d] = %e\n", d, Y.norm2(d));
    }

    printfQuda("Computing coarse diagonal\n");
    createCoarseLocal<Float,coarseSpin,coarseColor>(X, kappa);

    createCoarseCloverFromCoarse<Float,nDim>(X, V, C, x_size, xc_size, geo_bs, spin_bs);
    printfQuda("X2 = %e\n", X.norm2(0));
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, 
            int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor*fineSpin,fineSpin,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;

    const ColorSpinorField &v = T.Vectors();
    int dummy = 0;
    v.exchangeGhost(QUDA_INVALID_PARITY, dummy);

    F vAccessor(const_cast<ColorSpinorField&>(v));
    F uvAccessor(const_cast<ColorSpinorField&>(uv));
    gFine gAccessor(const_cast<GaugeField&>(g));
    gFine cloverAccessor(const_cast<GaugeField&>(clover));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X)); 

    calculateYcoarse<Float,coarseSpin,coarseColor>
      (yAccessor, xAccessor, uvAccessor, vAccessor, gAccessor, cloverAccessor, g.X(), Y.X(), kappa);

    {
      cpuGaugeField *X_h = static_cast<cpuGaugeField*>(&X);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv);

      // invert the clover matrix field
      const int n = X_h->Ncolor();
      BlasMagmaArgs magma(X_h->Precision());
      magma.BatchInvertMatrix(((float**)Xinv_h->Gauge_p())[0], ((float**)X_h->Gauge_p())[0], n, X_h->Volume());
    }
  }


  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor, int fineSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2) 
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) { 
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,2,coarseSpin>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else if (coarseColor == 8) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,8,coarseSpin>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else if (coarseColor == 16) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,16,coarseSpin>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else if (coarseColor == 24) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,24,coarseSpin>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (uv.Nspin() == 2) {
      calculateYcoarse<Float,csOrder,gOrder,fineColor,2>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported number of spins %d\n", uv.Nspin());
    }
  }

  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (g.Ncolor()/uv.Nspin() == 24) {
      calculateYcoarse<Float,csOrder,gOrder,24>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else if (g.Ncolor()/uv.Nspin() == 2) {
      calculateYcoarse<Float,csOrder,gOrder,2>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else if (g.Ncolor()/uv.Nspin() == 8) {
      calculateYcoarse<Float,csOrder,gOrder,8>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else if (g.Ncolor()/uv.Nspin() == 16) {
      calculateYcoarse<Float,csOrder,gOrder,16>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (g.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      calculateYcoarse<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", g.FieldOrder());
    }
  }

  template <typename Float>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateYcoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover, double kappa) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() || 
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != g.Precision())
      errorQuda("Unsupported precision mix");

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateYcoarse<double>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateYcoarse<float>(Y, X, Xinv, uv, T, g, clover, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseCoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, const Transfer &T, const cpuGaugeField &gauge, const cpuGaugeField &clover, double kappa) {
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


    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField uv(UVparam);

    calculateYcoarse(Y, X, Xinv, uv, T, gauge, clover, kappa);

    // now exchange Y halos for multi-process dslash
    Y.exchangeGhost();
  }
  
} //namespace quda
