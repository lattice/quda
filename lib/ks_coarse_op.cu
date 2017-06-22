#include <transfer.h>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <gauge_field.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <index_helper.cuh>
#include <blas_cublas.h>
#include <ks_coarse_op.cuh>

namespace quda {

  template <typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T,
		  const GaugeField *fl, const GaugeField *ll,  const CloverField &c, double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    QudaFieldLocation location = Y.Location();

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;
      constexpr QudaCloverFieldOrder clOrder = QUDA_PACKED_CLOVER_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
	errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (fl->FieldOrder() != gOrder) errorQuda("Unsupported field order for fat links%d\n", fl->FieldOrder());
      if ( ll ) if (ll->FieldOrder() != gOrder) errorQuda("Unsupported field order for long links%d\n", ll->FieldOrder());
      if (c.Order() != clOrder && c.Bytes()) errorQuda("Unsupported field order %d\n", c.Order());

      typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat> V;
      typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat> F;
      typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;
      typedef typename clover::FieldOrder<Float,fineColor,fineSpin,clOrder> cFine;

      const ColorSpinorField &v = T.Vectors(fl->Location());

      V vAccessor(const_cast<ColorSpinorField&>(v));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));
      cFine cAccessor(const_cast<CloverField&>(c), false);
      cFine cInvAccessor(const_cast<CloverField&>(c), true);

      F *uvAccessor    = new F(const_cast<ColorSpinorField&>(*uv));;
      gFine *fAccessor = new gFine(const_cast<GaugeField&>(*fl));

      F *uvlAccessor   = nullptr;
      gFine *lAccessor = nullptr;

      if( ll != nullptr ) {
        uvlAccessor = new F(const_cast<ColorSpinorField&>(*uv_long));
        lAccessor   = new gFine(const_cast<GaugeField&>(*ll));

        calculateKSY<false,true,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>
	  (yAccessor, xAccessor, xInvAccessor, uvAccessor, uvlAccessor, vAccessor, fAccessor, lAccessor, cAccessor, cInvAccessor, Y, X, Xinv, uv, uv_long, v, mass, dirac, matpc);

        delete uvlAccessor;
        delete lAccessor;

      } else {
      
        calculateKSY<false,false,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>
	  (yAccessor, xAccessor, xInvAccessor, uvAccessor, uvlAccessor, vAccessor, fAccessor, lAccessor, cAccessor, cInvAccessor, Y, X, Xinv, uv, uv_long, v, mass, dirac, matpc);
      }

      delete uvAccessor;
      delete fAccessor;

    } else {

      constexpr QudaFieldOrder csOrder = QUDA_FLOAT2_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;
      constexpr QudaCloverFieldOrder clOrder = QUDA_FLOAT4_CLOVER_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
	errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (fl->FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", fl->FieldOrder());
      if( ll ) if (ll->FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", ll->FieldOrder());
      if (c.Order() != clOrder && c.Bytes()) errorQuda("Unsupported field order %d\n", c.Order());

      typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat> F;
      typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> gCoarse;
      typedef typename clover::FieldOrder<Float,fineColor,fineSpin,clOrder> cFine;

      const ColorSpinorField &v = T.Vectors(fl->Location());

      F vAccessor(const_cast<ColorSpinorField&>(v));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarse xInvAccessor(const_cast<GaugeField&>(Xinv));
      cFine cAccessor(const_cast<CloverField&>(c), false);
      cFine cInvAccessor(const_cast<CloverField&>(c), true);

      F *uvAccessor    = new F(const_cast<ColorSpinorField&>(*uv));;
      gFine *fAccessor = new gFine(const_cast<GaugeField&>(*fl));

      F *uvlAccessor   = nullptr;
      gFine *lAccessor = nullptr;

      if(ll != nullptr && uv_long != nullptr) {
        uvlAccessor = new F(const_cast<ColorSpinorField&>(*uv_long));
        lAccessor   = new gFine(const_cast<GaugeField&>(*ll));

        calculateKSY<false,true,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>
	  (yAccessor, xAccessor, xInvAccessor, uvAccessor, uvlAccessor, vAccessor, fAccessor, lAccessor, cAccessor, cInvAccessor, Y, X, Xinv, uv, uv_long, v, mass, dirac, matpc);

        delete uvlAccessor;
        delete lAccessor;

      } else {

        calculateKSY<false,false,Float,fineSpin,fineColor,coarseSpin,coarseColor,gOrder>
	  (yAccessor, xAccessor, xInvAccessor, uvAccessor, uvlAccessor, vAccessor, fAccessor, lAccessor, cAccessor, cInvAccessor, Y, X, Xinv, uv, uv_long, v, mass, dirac, matpc);
      }

      delete uvAccessor;
      delete fAccessor;
    }
  }

  // template on the number of coarse degrees of freedom
  template <typename Float, typename vFloat, int fineColor, int fineSpin>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T,
		  const GaugeField *fl, const GaugeField *ll, const CloverField &c, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 2) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,2,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
#if 0
    } else if (coarseCoor == 4) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,4,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else if (coarseColor == 8) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,8,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else if (coarseColor == 12) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,12,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else if (coarseColor == 16) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,16,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else if (coarseColor == 20) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,20,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
#endif
    } else if (coarseColor == 24) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,24,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else if (coarseColor == 32) {
      calculateKSY<Float,vFloat,fineColor,fineSpin,32,coarseSpin>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, typename vFloat, int fineColor>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T,
		  const GaugeField *fl, const GaugeField *ll, const CloverField &c, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (uv->Nspin() == 1) {
      calculateKSY<Float,vFloat,fineColor,1>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of spins %d\n", uv->Nspin());
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat>
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T,
		  const GaugeField *fl, const GaugeField *ll, const CloverField &c, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    if (fl->Ncolor() == 3) {
      calculateKSY<Float,vFloat,3>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of colors %d\n", fl->Ncolor());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
   //?? calculateKSY(Y, X, Xinv, *uv, *uv_long, T, *Fl, *Ll, *C, mass, dirac, matpc);
  void calculateKSY(GaugeField &Y, GaugeField &X, GaugeField &Xinv, ColorSpinorField *uv, ColorSpinorField *uv_long, const Transfer &T,
		  const GaugeField *fl, const GaugeField *ll, const CloverField &c, double mass, QudaDiracType dirac, QudaMatPCType matpc) {
    Precision(X, Y, *fl);
    Precision(X, Y, *ll);
    Precision(*uv, *uv_long, T.Vectors(X.Location()));

    printfQuda("Computing Y field......\n");

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
	calculateKSY<double,double>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
      } else {
	errorQuda("Unsupported precision %d\n", Y.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
	calculateKSY<float,float>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
      } else if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
	//calculateKSY<float,short>(Y, X, Xinv, uv, uv_long, T, fl, ll, c, mass, dirac, matpc);
        errorQuda("Unsupported option.\n");
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
  //&gauge -> *fat_link, *long_link
  void CoarseKSOp(GaugeField &Y, GaugeField &X, GaugeField &Xinv, const Transfer &T,
		const cudaGaugeField *fat_links, const cudaGaugeField *long_links,
		double mass, QudaDiracType dirac, QudaMatPCType matpc) {

    QudaPrecision precision = Y.Precision();
    QudaFieldLocation location = Location(Y, X, Xinv);

    CloverField *C = nullptr;

    GaugeField *Fl = location == QUDA_CUDA_FIELD_LOCATION ? const_cast<cudaGaugeField*>(fat_links) : nullptr;
    GaugeField *Ll = location == QUDA_CUDA_FIELD_LOCATION && long_links ? const_cast<cudaGaugeField*>(long_links) : nullptr;

    if (location == QUDA_CPU_FIELD_LOCATION) {
      //First make a cpu gauge field from the cuda gauge field
      int pad = 0;
      GaugeFieldParam fat_param(fat_links->X(), precision, QUDA_RECONSTRUCT_NO, pad, fat_links->Geometry());
      fat_param.order = QUDA_QDP_GAUGE_ORDER;
      fat_param.fixed = fat_links->GaugeFixed();
      fat_param.link_type = fat_links->LinkType();
      fat_param.t_boundary = fat_links->TBoundary();
      fat_param.anisotropy = fat_links->Anisotropy();
      fat_param.gauge = nullptr;
      fat_param.create = QUDA_NULL_FIELD_CREATE;
      fat_param.siteSubset = QUDA_FULL_SITE_SUBSET;
      fat_param.nFace = 1;
      fat_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

      Fl = new cpuGaugeField(fat_param);

      //Copy the cuda gauge field to the cpu
      fat_links->saveCPUField(*static_cast<cpuGaugeField*>(Fl));

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

        long_param.nFace = 3;
        long_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
        //
        Ll = new cpuGaugeField(long_param);
        //Copy the cuda gauge field to the cpu
        long_links->saveCPUField(*static_cast<cpuGaugeField*>(Ll));
      }

    } else if (location == QUDA_CUDA_FIELD_LOCATION && fat_links->Reconstruct() != QUDA_RECONSTRUCT_NO) {
      //Create a copy of the gauge field with no reconstruction, required for fine-grained access
      GaugeFieldParam fat_param(*fat_links);
      fat_param.reconstruct = QUDA_RECONSTRUCT_NO;
      fat_param.setPrecision(fat_param.precision);
      Fl = new cudaGaugeField(fat_param);

      Fl->copy(*fat_links);
    }

    //check long links: 
    if (location == QUDA_CUDA_FIELD_LOCATION && long_links) {
      if (long_links->Reconstruct() != QUDA_RECONSTRUCT_NO) {
        //Create a copy of the gauge field with no reconstruction, required for fine-grained access
        GaugeFieldParam long_param(*long_links);
        long_param.reconstruct = QUDA_RECONSTRUCT_NO;
        long_param.setPrecision(long_param.precision);
        Ll = new cudaGaugeField(long_param);

        Ll->copy(*long_links);
      }
    }

    CloverFieldParam cf_param;
    cf_param.nDim = 4;
    cf_param.pad = 0;
    cf_param.precision = QUDA_INVALID_PRECISION;

    // if we have no clover term then create an empty clover field
    for(int i = 0; i < cf_param.nDim; i++) cf_param.x[i] = 0;

    cf_param.direct = true;
    cf_param.inverse = true;
    cf_param.clover = nullptr;
    cf_param.norm = 0;
    cf_param.cloverInv = nullptr;
    cf_param.invNorm = 0;
    cf_param.create = QUDA_NULL_FIELD_CREATE;
    cf_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    if (location == QUDA_CUDA_FIELD_LOCATION) {
      // create a dummy cudaCloverField since one is not defined
      cf_param.order = QUDA_INVALID_CLOVER_ORDER;
      C = new cudaCloverField(cf_param);
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      //Create a cpuCloverField from the cudaCloverField
      cf_param.order = QUDA_PACKED_CLOVER_ORDER;
      C = new cpuCloverField(cf_param);
    }

    //Create a field UV which holds U*V.  Has the same structure as V.
    ColorSpinorParam UVparam(T.Vectors(location));
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.precision = T.Vectors(location).Precision();

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    ColorSpinorField *uv_long = nullptr;
    if(long_links != nullptr)  uv_long = ColorSpinorField::Create(UVparam);

    calculateKSY(Y, X, Xinv, uv, uv_long, T, Fl, Ll, *C, mass, dirac, matpc);

    delete uv;
    if(uv_long) delete uv_long;

    delete C;
    if (Fl != fat_links) delete Fl;
    if(long_links) if (Ll != long_links) delete Ll;
  }

} //namespace quda
