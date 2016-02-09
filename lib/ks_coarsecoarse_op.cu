#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <blas_magma.h>
#include <ks_coarse_op.cuh>

namespace quda {

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, 
            int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateKSYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover,
			const GaugeField &cloverInv, double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename colorspinor::FieldOrderCB<Float,2*fineSpin,fineColor,coarseColor,csOrder> F2;
    typedef typename gauge::FieldOrder<Float,fineColor*fineSpin,fineSpin,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;

    const ColorSpinorField &v = T.Vectors();
    int dummy = 0;
    v.exchangeGhost(QUDA_INVALID_PARITY, dummy);

    F vAccessor(const_cast<ColorSpinorField&>(v));
    F2 uvAccessor(const_cast<ColorSpinorField&>(uv));
    gFine gAccessor(const_cast<GaugeField&>(g));
    gFine cAccessor(const_cast<GaugeField&>(clover));
    gFine cInvAccessor(const_cast<GaugeField&>(cloverInv));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X)); 
    gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));

    gFine *lAccessor = nullptr;
    F2 *uvlAccessor   = nullptr;

    calculateKSY<true,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>
      (yAccessor, xAccessor, xInvAccessor, &uvAccessor, uvlAccessor, vAccessor, &gAccessor, lAccessor, cAccessor, cInvAccessor,
       Y, X, Xinv, Yhat, v, mass, dirac, matpc);
  }


  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor, int fineSpin>
  void calculateKSYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover,
			const GaugeField &cloverInv, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2) 
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) { 
      calculateKSYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,2,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (coarseColor == 8) {
      calculateKSYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,8,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (coarseColor == 16) {
      calculateKSYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,16,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (coarseColor == 24) {
      calculateKSYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,24,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (coarseColor == 32) {
      calculateKSYcoarse<Float,csOrder,gOrder,fineColor,fineSpin,32,coarseSpin>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor>
  void calculateKSYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover,
			const GaugeField &cloverInv, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().Nspin() == 2) {
      calculateKSYcoarse<Float,csOrder,gOrder,fineColor,2>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of spins %d\n", T.Vectors().Nspin());
    }
  }

  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateKSYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover,
			const GaugeField &cloverInv, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (g.Ncolor()/T.Vectors().Nspin() == 2) {
      calculateKSYcoarse<Float,csOrder,gOrder,2>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (g.Ncolor()/T.Vectors().Nspin() == 8) {
      calculateKSYcoarse<Float,csOrder,gOrder,8>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (g.Ncolor()/T.Vectors().Nspin() == 16) {
      calculateKSYcoarse<Float,csOrder,gOrder,16>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (g.Ncolor()/T.Vectors().Nspin() == 24) {
      calculateKSYcoarse<Float,csOrder,gOrder,24>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else if (g.Ncolor()/T.Vectors().Nspin() == 32) {
      calculateKSYcoarse<Float,csOrder,gOrder,32>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateKSYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover,
			const GaugeField &cloverInv, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (g.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      calculateKSYcoarse<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported field order %d\n", g.FieldOrder());
    }
  }

  template <typename Float>
  void calculateKSYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
			ColorSpinorField &uv, const Transfer &T, const GaugeField &g, const GaugeField &clover,
			const GaugeField &cloverInv, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
      calculateKSYcoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateKSYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField &uv,
			const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
			double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv.Precision() || 
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != g.Precision())
      errorQuda("Unsupported precision mix");

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      calculateKSYcoarse<double>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateKSYcoarse<float>(Y, X, Xinv, Yhat, uv, T, g, clover, cloverInv, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    printfQuda("....done computing Y field\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseCoarseKSOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
		      const Transfer &T, const cpuGaugeField &gauge, const cpuGaugeField &clover, const cpuGaugeField &cloverInv,
		      double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    QudaPrecision precision = Y.Precision();
    //First make a cpu gauge field from the cuda gauge field

    //Create a field UV which holds U*V.  Has the same similar
    //structure to V but double the number of spins so we can store
    //the four distinct block chiral multiplications in a single UV
    //computation.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.nSpin *= 2; // so nSpin == 4
    cpuColorSpinorField uv(UVparam);

    calculateKSYcoarse(Y, X, Xinv, Yhat, uv, T, gauge, clover, cloverInv, mass, dirac, matpc);

  }
  
} //namespace quda
