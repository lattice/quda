#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <coarse_op.cuh>
#include <blas_magma.h>

namespace quda {

  extern bool preconditioned_links;

  //Calculates the coarse gauge field
  template<typename Float, int coarseSpin, int coarseColor,
	   typename F, typename coarseGauge, typename fineGauge, typename fineClover>
  void calculateY(coarseGauge &Y, coarseGauge &X, F &UV, F &V, fineGauge &G, fineClover *C,
		  const int *xx_size, const int *xc_size, double kappa) {
    if (UV.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) errorQuda("Gamma basis not supported");
    const QudaGammaBasis basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

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

    for(int d = 0; d < nDim; d++) {
      //First calculate UV
      setZero<Float,F>(UV);

      printfQuda("Computing %d UV and VUV\n", d);
      //Calculate UV and then VUV for this direction, accumulating directly into the coarse gauge field Y
      if (d==0) {
        computeUV<false,Float,0>(UV, V, G, nDim, x_size, comm_dim);
        Gamma<Float, basis, 0> gamma;
        computeVUV<false,Float,0>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      } else if (d==1) {
        computeUV<false,Float,1>(UV, V, G, nDim, x_size, comm_dim);
        Gamma<Float, basis, 1> gamma;
        computeVUV<false,Float,1>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      } else if (d==2) {
        computeUV<false,Float,2>(UV, V, G, nDim, x_size, comm_dim);
        Gamma<Float, basis, 2> gamma;
        computeVUV<false,Float,2>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      } else {
        computeUV<false,Float,3>(UV, V, G, nDim, x_size, comm_dim);
        Gamma<Float, basis, 3> gamma;
        computeVUV<false,Float,3>(Y, X, UV, V, gamma, G, x_size, xc_size, geo_bs, spin_bs);
      }

      printfQuda("UV2[%d] = %e\n", d, UV.norm2());
      printfQuda("Y2[%d] = %e\n", d, Y.norm2(d));
    }
    createYreverse<Float,coarseSpin,coarseColor>(Y);

    printfQuda("X2 = %e\n", X.norm2(0));
    printfQuda("Computing coarse diagonal\n");
    createCoarseLocal<Float, coarseSpin, coarseColor>(X, kappa);

    //If C!=NULL we have to coarsen the fine clover term and add it in.
    if (C != NULL) {
      printfQuda("Computing fine->coarse clover term\n");
      createCoarseCloverFromFine<Float,nDim>(X, V, *C, x_size, xc_size, geo_bs, spin_bs);
      printfQuda("X2 = %e\n", X.norm2(0));
    }
    //Otherwise, we have a fine Wilson operator.  The "clover" term for the Wilson operator
    //is just the identity matrix.
    else {
      addCoarseDiagonal<Float>(X);
    }
    printfQuda("X2 = %e\n", X.norm2(0));
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder,
            int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {

    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;
    typedef typename clover::FieldOrder<Float,fineColor,fineSpin,clOrder> cFine;

    const ColorSpinorField &v = T.Vectors();
    int dummy = 0;
    v.exchangeGhost(QUDA_INVALID_PARITY, dummy);

    F vAccessor(const_cast<ColorSpinorField&>(v));
    F uvAccessor(const_cast<ColorSpinorField&>(uv));
    gFine gAccessor(const_cast<GaugeField&>(g));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X));

    if(c != NULL) {
      cFine cAccessor(const_cast<CloverField&>(*c));
      calculateY<Float,coarseSpin,coarseColor>(yAccessor, xAccessor, uvAccessor, vAccessor, gAccessor, &cAccessor, g.X(), Y.X(), kappa);
    }
    else {
      cFine *cAccessor = NULL;
      calculateY<Float,coarseSpin,coarseColor>(yAccessor, xAccessor, uvAccessor, vAccessor, gAccessor, cAccessor, g.X(), Y.X(), kappa);
    }    

    {
      cpuGaugeField *X_h = static_cast<cpuGaugeField*>(&X);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv);

      // invert the clover matrix field
      const int n = X_h->Ncolor();
      BlasMagmaArgs magma(X_h->Precision());
      magma.BatchInvertMatrix(((float**)Xinv_h->Gauge_p())[0], ((float**)X_h->Gauge_p())[0], n, X_h->Volume());
    }

    // now exchange Y halos for multi-process dslash
    Y.exchangeGhost();

    if (preconditioned_links) {
      // create the preconditioned links
      // Y_back(x-\mu) = Y_back(x-\mu) * Xinv^dagger(x) (positive projector)
      // Y_fwd(x) = Xinv(x) * Y_fwd(x)                  (negative projector)

      // use spin-ignorant accessor to make multiplication simpler
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,1,gOrder> gCoarse;
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));
      int comm_dim[4];
      for (int i=0; i<4; i++) comm_dim[i] = comm_dim_partitioned(i);
      createYpreconditioned<Float,coarseSpin*coarseColor>(yAccessor, xInvAccessor, X.X(), 1, comm_dim);
    }

  }

  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder, int fineColor, int fineSpin>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,2,coarseSpin>(Y, X, Xinv, uv, T, g, c, kappa);
    } else if (coarseColor == 4) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,4,coarseSpin>(Y, X, Xinv, uv, T, g, c, kappa);
    } else if (coarseColor == 8) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,8,coarseSpin>(Y, X, Xinv, uv, T, g, c, kappa);
    } else if (coarseColor == 12) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,12,coarseSpin>(Y, X, Xinv, uv, T, g, c, kappa);
    } else if (coarseColor == 16) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,16,coarseSpin>(Y, X, Xinv, uv, T, g, c, kappa);
    } else if (coarseColor == 20) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,20,coarseSpin>(Y, X, Xinv, uv, T, g, c, kappa);
    } else if (coarseColor == 24) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,fineSpin,24,coarseSpin>(Y, X, Xinv, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder, int fineColor>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (uv.Nspin() == 4) {
      calculateY<Float,csOrder,gOrder,clOrder,fineColor,4>(Y, X, Xinv, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported number of spins %d\n", uv.Nspin());
    }
  }

  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, QudaCloverFieldOrder clOrder>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (g.Ncolor() == 3) {
      calculateY<Float,csOrder,gOrder,clOrder,3>(Y, X, Xinv, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    //If c == NULL, then this is standard Wilson.  csOrder is dummy and will not matter      
    if (c==NULL || c->Order() == QUDA_PACKED_CLOVER_ORDER) {
      calculateY<Float,csOrder,gOrder,QUDA_PACKED_CLOVER_ORDER>(Y, X, Xinv, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", c->Order());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (g.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      calculateY<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, Xinv, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", g.FieldOrder());
    }
  }

 template <typename Float>
 void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateY<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, Xinv, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField &uv, const Transfer &T, GaugeField &g, CloverField *c, double kappa) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() ||
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != g.Precision())
      errorQuda("Unsupported precision mix");

    printfQuda("Computing Y field......\n");

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateY<double>(Y, X, Xinv, uv, T, g, c, kappa);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateY<float>(Y, X, Xinv, uv, T, g, c, kappa);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, const Transfer &T,
		const cudaGaugeField &gauge, const cudaCloverField *clover, double kappa) {
    QudaPrecision precision = Y.Precision();
    //First make a cpu gauge field from the cuda gauge field

    int pad = 0;
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

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    cpuColorSpinorField uv(UVparam);

    //If the fine lattice operator is the clover operator, copy the cudaCloverField to cpuCloverField
    if(clover != NULL) {
      //Create a cpuCloverField from the cudaCloverField
      CloverFieldParam cf_param;
      cf_param.nDim = 4;
      cf_param.pad = pad;
      cf_param.precision = clover->Precision();
      for(int i = 0; i < cf_param.nDim; i++) {
        cf_param.x[i] = clover->X()[i];
      }

      cf_param.order = QUDA_PACKED_CLOVER_ORDER;
      cf_param.direct = true;
      cf_param.inverse = true;
      cf_param.clover = NULL;
      cf_param.norm = 0;
      cf_param.cloverInv = NULL;
      cf_param.invNorm = 0;
      cf_param.create = QUDA_NULL_FIELD_CREATE;
      cf_param.siteSubset = QUDA_FULL_SITE_SUBSET;

      cpuCloverField c(cf_param);
      clover->saveCPUField(c);

      calculateY(Y, X, Xinv, uv, T, g, &c, kappa);
    }
    else {
      calculateY(Y, X, Xinv, uv, T, g, NULL, kappa);
    }

  }

} //namespace quda
