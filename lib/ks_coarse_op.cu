#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <clover_field.h>

#include <blas_magma.h>
#include <ks_coarse_op.cuh> //remove it!

namespace quda {


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor, int coarseColor, int coarseSpin>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    const int fineSpin = 1;

    typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder> F;
    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
    typedef typename gauge::FieldOrder<Float,coarseSpin*coarseColor,coarseSpin,gOrder> gCoarse;

    const ColorSpinorField &v = T.Vectors();
    int dummy = 0;
    v.exchangeGhost(QUDA_INVALID_PARITY, dummy);

    F vAccessor(const_cast<ColorSpinorField&>(v));
    F uvAccessor(const_cast<ColorSpinorField&>(*uv));
    gFine fAccessor(const_cast<GaugeField&>(*f));
    gCoarse yAccessor(const_cast<GaugeField&>(Y));
    gCoarse xAccessor(const_cast<GaugeField&>(X));
    gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));

//Create fake clover fields (temporary hack)
    CloverFieldParam cf_param;
    cf_param.nDim = 4;
    cf_param.pad = 0;
    cf_param.precision = QUDA_INVALID_PRECISION;

    // if we have no clover term then create an empty clover field
    for(int i = 0; i < cf_param.nDim; i++) cf_param.x[i] = 0;

    cf_param.order = QUDA_PACKED_CLOVER_ORDER;
    cf_param.direct = true;
    cf_param.inverse = true;
    cf_param.clover = nullptr;
    cf_param.norm = 0;
    cf_param.cloverInv = nullptr;
    cf_param.invNorm = 0;
    cf_param.create = QUDA_NULL_FIELD_CREATE;
    cf_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    CloverField c(cf_param);

    typedef typename clover::FieldOrder<Float,fineColor,fineSpin,QUDA_PACKED_CLOVER_ORDER> cFine;
    cFine cAccessor(const_cast<CloverField&>(c), false);
    cFine cInvAccessor(const_cast<CloverField&>(c), true);

     gFine *lAccessor = nullptr;
     F *uvlAccessor  = nullptr;

    if(l != nullptr && uv_long != nullptr) {
      lAccessor   = new gFine(const_cast<GaugeField&>(*l));
      uvlAccessor = new F(const_cast<ColorSpinorField&>(*uv_long));
    }
    calculateKSY<false,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>(yAccessor, xAccessor, xInvAccessor, &uvAccessor, uvlAccessor, vAccessor, &fAccessor, lAccessor, cAccessor, cInvAccessor, Y, X, Xinv, Yhat, v, mass, dirac, matpc);
    
    if(lAccessor) delete lAccessor;
    if(uvlAccessor) delete uvlAccessor;

  }

  // template on the number of coarse degrees of freedom
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int fineColor>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    if ((T.Vectors().Nspin() != 1) && (T.Vectors().Nspin()/T.Spin_bs() != 2))  errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) {
      calculateKSY<Float,csOrder,gOrder,fineColor,2, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 4) {
      calculateKSY<Float,csOrder,gOrder,fineColor,4, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 8) {
      calculateKSY<Float,csOrder,gOrder,fineColor,8, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 12) {
      calculateKSY<Float,csOrder,gOrder,fineColor,12, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 16) {
      calculateKSY<Float,csOrder,gOrder,fineColor,16, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 20) {
      calculateKSY<Float,csOrder,gOrder,fineColor,20, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 24) {
      calculateKSY<Float,csOrder,gOrder,fineColor,24, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 48) {
      calculateKSY<Float,csOrder,gOrder,fineColor,48, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (coarseColor == 96) {
      calculateKSY<Float,csOrder,gOrder,fineColor,96, coarseSpin>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }


  // template on fine colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,  ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (f->Ncolor() == 3) {
      if( l ) if( f->Ncolor() != l->Ncolor() ) errorQuda("Unsupported number of colors %d\n", l->Ncolor());

      calculateKSY<Float,csOrder,gOrder, 3>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of colors %d\n", f->Ncolor());
    }
  }

  template <typename Float, QudaFieldOrder csOrder>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,  ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (f->FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      if( l ) if( l->FieldOrder() != QUDA_QDP_GAUGE_ORDER ) errorQuda("Unsupported field order for long links %d\n", l->FieldOrder());

      calculateKSY<Float,csOrder,QUDA_QDP_GAUGE_ORDER>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported field order %d\n", f->FieldOrder());
    }
  }

 template <typename Float>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {

      calculateKSY<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported field order %d\n", T.Vectors().FieldOrder());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T, GaugeField *f, GaugeField *l, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (X.Precision() != Y.Precision() || Y.Precision() != uv->Precision() ||
        Y.Precision() != T.Vectors().Precision() || Y.Precision() != f->Precision())
    {
      errorQuda("Unsupported precision mix");
    }

    if( l )
    { 
      if(Y.Precision() != l->Precision() || Y.Precision() != uv_long->Precision()) errorQuda("Unsupported precision mix for long links.");
    }

    printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateKSY<double>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      calculateKSY<float>(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    if(l)
     printfQuda("....done computing Y field for improved staggered operator\n");  
    else 
     printfQuda("....done computing Y field for naive staggered operator\n");
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseKSOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, GaugeField &Yhat,
		const Transfer &T, const cudaGaugeField *fat_links, const cudaGaugeField *long_links,  double mass, QudaDiracType dirac, QudaMatPCType matpc ) {
    QudaPrecision precision = Y.Precision();
    //First make a cpu gauge field from the cuda gauge field

    int pad = 0;
    GaugeFieldParam fat_param(fat_links->X(), precision, fat_links->Reconstruct(), pad, fat_links->Geometry());
    fat_param.order = QUDA_QDP_GAUGE_ORDER;
    fat_param.fixed = fat_links->GaugeFixed();
    fat_param.link_type = fat_links->LinkType();
    fat_param.t_boundary = fat_links->TBoundary();
    fat_param.anisotropy = fat_links->Anisotropy();
    fat_param.gauge = nullptr;
    fat_param.create = QUDA_NULL_FIELD_CREATE;
    fat_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    cpuGaugeField *f = new cpuGaugeField(fat_param);
    cpuGaugeField *l = nullptr;

    //Copy the cuda gauge field to the cpu
/* WARNING!
set QUDA_REORDER_LOCATION to CPU to perform re-ordering on CPU (GPU reordering does not work, bug??)
*/
    fat_links->saveCPUField(*f);

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors());
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = QUDA_CPU_FIELD_LOCATION;

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    ColorSpinorField *uv_long = nullptr;

    if(long_links)
    {
      GaugeFieldParam long_param(long_links->X(), precision, long_links->Reconstruct(), pad, long_links->Geometry());
      long_param.order = QUDA_QDP_GAUGE_ORDER;
      long_param.fixed = long_links->GaugeFixed();
      long_param.link_type = long_links->LinkType();
      long_param.t_boundary = long_links->TBoundary();
      long_param.anisotropy = long_links->Anisotropy();
      long_param.gauge = nullptr;
      long_param.create = QUDA_NULL_FIELD_CREATE;
      long_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      //
      l = new cpuGaugeField(long_param);
      //
      uv_long = ColorSpinorField::Create(UVparam);
      //Copy the cuda gauge field to the cpu
      long_links->saveCPUField(*l);
    }

    //If the fine lattice operator is the clover operator, copy the cudaCloverField to cpuCloverField
    calculateKSY(Y, X, Xinv, Yhat, uv, uv_long, T, f, l, mass, dirac, matpc);

    // now exchange Y halos for multi-process dslash
    Y.exchangeGhost();

    delete uv; 
    delete f;

    if(l)
    { 
      delete l;
      delete uv_long;
    }

  }

} //namespace quda
