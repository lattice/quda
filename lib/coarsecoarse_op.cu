#include <transfer.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

#define COARSECOARSE
#include <coarse_op.cuh>

namespace quda {

  /**
     @brief dummyClover is a helper function to allow us to create an
     empty clover object - this allows us to use the the externally
     linked reduction kernels when we do have a clover field.
   */
  inline std::unique_ptr<cudaCloverField> dummyClover()
  {
    CloverFieldParam cf_param;
    cf_param.nDim = 4;
    cf_param.pad = 0;
    cf_param.setPrecision(QUDA_SINGLE_PRECISION);

    for (int i = 0; i < cf_param.nDim; i++) cf_param.x[i] = 0;

    cf_param.direct = true;
    cf_param.inverse = true;
    cf_param.clover = nullptr;
    cf_param.norm = 0;
    cf_param.cloverInv = nullptr;
    cf_param.invNorm = 0;
    cf_param.create = QUDA_NULL_FIELD_CREATE;
    cf_param.siteSubset = QUDA_FULL_SITE_SUBSET;

    // create a dummy cudaCloverField if one is not defined
    cf_param.order = QUDA_INVALID_CLOVER_ORDER;
    return std::make_unique<cudaCloverField>(cf_param);
  }

  template <typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                        bool need_bidirectional, bool use_mma)
  {
    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {

      if (use_mma) { errorQuda("MMA intructions are not supported on the host."); }

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
	errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      typedef typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat> V;
      typedef typename colorspinor::FieldOrderCB<Float,2*fineSpin,fineColor,coarseColor,csOrder,vFloat> F;
      typedef typename gauge::FieldOrder<Float,fineColor*fineSpin,fineSpin,gOrder,true,vFloat> gFine;
      typedef typename gauge::FieldOrder<Float,fineColor*fineSpin,fineSpin,gOrder,true,vFloat> cFine;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat> gCoarse;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,storeType> gCoarseAtomic;

      const ColorSpinorField &v = T.Vectors(Y.Location());

      V vAccessor(const_cast<ColorSpinorField&>(v));
      F uvAccessor(const_cast<ColorSpinorField&>(uv));
      gFine gAccessor(const_cast<GaugeField&>(g));
      cFine cAccessor(const_cast<GaugeField&>(clover));
      cFine cInvAccessor(const_cast<GaugeField&>(cloverInv));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarseAtomic yAccessorAtomic(const_cast<GaugeField&>(Yatomic));
      gCoarseAtomic xAccessorAtomic(const_cast<GaugeField&>(Xatomic));

      calculateY<QUDA_CPU_FIELD_LOCATION, true, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor, cAccessor,
        cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v), v, g, *dummyClover(), kappa, mu,
        mu_factor, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()), T.coarseToFine(Y.Location()), use_mma);
    } else {

      // TODO: Change the !use_mma predicate to the field order one.
      if (!use_mma) {

        if (use_mma) { errorQuda("MMA intructions are not supported on for the AoS orders."); }

        constexpr QudaFieldOrder csOrder = QUDA_FLOAT2_FIELD_ORDER;
        constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

        if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
          errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
        if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

        typedef typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat> V;
        typedef typename colorspinor::FieldOrderCB<Float, 2 * fineSpin, fineColor, coarseColor, csOrder, vFloat> F;
        typedef typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat> gFine;
        typedef typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat> cFine;
        typedef typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, vFloat> gCoarse;
        typedef typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, storeType> gCoarseAtomic;

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
        calculateY<QUDA_CUDA_FIELD_LOCATION, true, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
          yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor,
          cAccessor, cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v), v, g, *dummyClover(),
          kappa, mu, mu_factor, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
          T.coarseToFine(Y.Location()), use_mma);

      } else {

        constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

        // XXX: This doesn't work for single precision since X/Yatomic are just references to X/Y.
        GaugeFieldParam param_Y(Y);
        GaugeFieldParam param_X(X);
        GaugeFieldParam param_Xatomic(Xatomic);
        GaugeFieldParam param_Yatomic(Yatomic);

        ColorSpinorParam param_uv(uv);

        GaugeFieldParam param_g(g);
        GaugeFieldParam param_clover(clover);
        GaugeFieldParam param_cloverInv(cloverInv);

        param_Y.order = gOrder;
        param_X.order = gOrder;
        param_Xatomic.order = gOrder;
        param_Yatomic.order = gOrder;

        param_uv.fieldOrder = csOrder;

        param_g.order = gOrder;
        param_clover.order = gOrder;
        param_cloverInv.order = gOrder;

        param_Y.setPrecision(X.Precision());
        param_X.setPrecision(Y.Precision());
        param_Xatomic.setPrecision(Xatomic.Precision());
        param_Yatomic.setPrecision(Yatomic.Precision());

        param_uv.setPrecision(uv.Precision());

        param_g.setPrecision(g.Precision());
        param_clover.setPrecision(clover.Precision());
        param_cloverInv.setPrecision(cloverInv.Precision());

        cudaGaugeField X_(param_X);
        cudaGaugeField Y_(param_Y);
        // cudaGaugeField Xatomic_(param_Xatomic);
        // cudaGaugeField Yatomic_(param_Yatomic);

        GaugeField *Yatomic_; // = &Y_;
        GaugeField *Xatomic_; // = &X_;

        if (Y.Precision() < QUDA_SINGLE_PRECISION) {
          // we need to coarsen into single precision fields (float or int), so we allocate temporaries for this purpose
          // else we can just coarsen directly into the original fields
          Yatomic_ = GaugeField::Create(param_Yatomic);
          Xatomic_ = GaugeField::Create(param_Xatomic);
        } else {
          Yatomic_ = &Y_;
          Xatomic_ = &X_;
        }

        cudaColorSpinorField uv_(param_uv);

        cudaGaugeField g_(param_g);
        cudaGaugeField clover_(param_clover);
        cudaGaugeField cloverInv_(param_cloverInv);

        g_.copy(g);
        clover_.copy(clover);
        cloverInv_.copy(cloverInv);

        uv_.copy(uv);
        typedef typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat> V;
        typedef typename colorspinor::FieldOrderCB<Float, 2 * fineSpin, fineColor, coarseColor, csOrder, vFloat> F;
        typedef typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat> gFine;
        typedef typename gauge::FieldOrder<Float, fineColor * fineSpin, fineSpin, gOrder, true, vFloat> cFine;
        typedef typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, vFloat> gCoarse;
        typedef typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, storeType> gCoarseAtomic;

        const ColorSpinorField &v = T.Vectors(Y.Location());

        ColorSpinorParam param_v(v);
        param_v.fieldOrder = csOrder;
        param_v.setPrecision(v.Precision());

        cudaColorSpinorField v_(param_v);

        v_.copy(v);

        V vAccessor(const_cast<cudaColorSpinorField &>(v_));
        F uvAccessor(const_cast<cudaColorSpinorField &>(uv_));
        gFine gAccessor(const_cast<cudaGaugeField &>(g_));
        cFine cAccessor(const_cast<cudaGaugeField &>(clover_));
        cFine cInvAccessor(const_cast<cudaGaugeField &>(cloverInv_));
        gCoarse yAccessor(const_cast<cudaGaugeField &>(Y_));
        gCoarse xAccessor(const_cast<cudaGaugeField &>(X_));
        gCoarseAtomic yAccessorAtomic(reinterpret_cast<cudaGaugeField &>(*Yatomic_));
        gCoarseAtomic xAccessorAtomic(reinterpret_cast<cudaGaugeField &>(*Xatomic_));

        // create a dummy clover field to allow us to call the external clover reduction routines elsewhere
        calculateY<QUDA_CUDA_FIELD_LOCATION, true, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
          yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, vAccessor, vAccessor, gAccessor,
          cAccessor, cInvAccessor, Y_, X_, *Yatomic_, *Xatomic_, uv_, const_cast<cudaColorSpinorField &>(v_), v_, g_,
          *dummyClover(), kappa, mu, mu_factor, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
          T.coarseToFine(Y.Location()), use_mma);

        X.copy(X_);
        Y.copy(Y_);
        if (Y.Precision() < QUDA_SINGLE_PRECISION) {
          Xatomic.copy(*Xatomic_);
          Yatomic.copy(*Yatomic_);
          delete Xatomic_;
          delete Yatomic_;
        }

        reinterpret_cast<cudaColorSpinorField &>(uv).copy(uv_);
      }
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat, int fineSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                        bool need_bidirectional, bool use_mma)
  {
    if (T.Vectors().Nspin()/T.Spin_bs() != 2) 
      errorQuda("Unsupported number of coarse spins %d\n",T.Vectors().Nspin()/T.Spin_bs());
    const int fineColor = g.Ncolor() / fineSpin;
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

#ifdef NSPIN4
    if (fineColor == 6) { // free field Wilson
      if (coarseColor == 6) {
        // calculateYcoarse<Float,vFloat,6,fineSpin,6,coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv,
        // kappa, mu, mu_factor, dirac, matpc, need_bidirectional, use_mma);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
    } else
#endif
    if (fineColor == 24) { // coarsened Wilson or free field staggered
      if (coarseColor == 24) {
        calculateYcoarse<Float, vFloat, 24, fineSpin, 24, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else
#ifdef NSPIN4
      if (coarseColor == 32) {
        calculateYcoarse<Float, vFloat, 24, fineSpin, 32, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else
#endif // NSPIN4
#ifdef NSPIN1
      if (coarseColor == 64) {
        calculateYcoarse<Float, vFloat, 24, fineSpin, 64, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else if (coarseColor == 96) {
        calculateYcoarse<Float, vFloat, 24, fineSpin, 96, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else
#endif
      {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
#ifdef NSPIN4
    } else if (fineColor == 32) {
      if (coarseColor == 32) {
        calculateYcoarse<Float, vFloat, 32, fineSpin, 32, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
#endif // NSPIN4
#ifdef NSPIN1
    } else if (fineColor == 64) {
      if (coarseColor == 64) {
        calculateYcoarse<Float, vFloat, 64, fineSpin, 64, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else if (coarseColor == 96) {
        calculateYcoarse<Float, vFloat, 64, fineSpin, 96, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
    } else if (fineColor == 96) {
      if (coarseColor == 96) {
        calculateYcoarse<Float, vFloat, 96, fineSpin, 96, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
#endif // NSPIN1
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, typename vFloat>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                        bool need_bidirectional, bool use_mma)
  {
    if (T.Vectors().Nspin() == 2) {
      calculateYcoarse<Float, vFloat, 2>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mu, mu_factor,
                                         dirac, matpc, need_bidirectional, use_mma);
    } else {
      errorQuda("Unsupported number of spins %d\n", T.Vectors().Nspin());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                        bool need_bidirectional, bool use_mma)
  {
#ifdef GPU_MULTIGRID
    checkPrecision(X, Y, g, clover, cloverInv, uv, T.Vectors(X.Location()));
    checkPrecision(Xatomic, Yatomic);

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
        calculateYcoarse<double, double>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mu, mu_factor,
                                         dirac, matpc, need_bidirectional, use_mma);
      } else {
	errorQuda("Unsupported precision %d\n", Y.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
        calculateYcoarse<float, float>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mu, mu_factor, dirac,
                                       matpc, need_bidirectional, use_mma);
      } else {
	errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
    } else if (Y.Precision() == QUDA_HALF_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
        calculateYcoarse<float, short>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mu, mu_factor, dirac,
                                       matpc, need_bidirectional, use_mma);
      } else {
	errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("....done computing Y field\n");
#else
    errorQuda("Multigrid has not been built");
#endif // GPU_MULTIGRID
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void CoarseCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &gauge,
                      const GaugeField &clover, const GaugeField &cloverInv, double kappa, double mu, double mu_factor,
                      QudaDiracType dirac, QudaMatPCType matpc, bool need_bidirectional, bool use_mma)
  {
    QudaPrecision precision = Y.Precision();
    QudaFieldLocation location = checkLocation(X, Y, gauge, clover, cloverInv);

    //Create a field UV which holds U*V.  Has the same similar
    //structure to V but double the number of spins so we can store
    //the four distinct block chiral multiplications in a single UV
    //computation.
    ColorSpinorParam UVparam(T.Vectors(location));
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.nSpin *= 2; // so nSpin == 4
    UVparam.setPrecision(T.Vectors(location).Precision());
    UVparam.mem_type = Y.MemType(); // allocate temporaries to match coarse-grid link field

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

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

    calculateYcoarse(Y, X, *Yatomic, *Xatomic, *uv, T, gauge, clover, cloverInv, kappa, mu, mu_factor, dirac, matpc,
                     need_bidirectional, use_mma);

    if (Yatomic != &Y) delete Yatomic;
    if (Xatomic != &X) delete Xatomic;

    delete uv;
  }
  
} //namespace quda
