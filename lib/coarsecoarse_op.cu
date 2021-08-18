#include <transfer.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

// This define controls which kernels get compiled in `coarse_op.cuh`.
// This ensures only kernels relevant for coarsening a coarse operator
// get built, saving compile time.
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
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
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
        cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<ColorSpinorField &>(v), v, g, *dummyClover(), kappa, mass, mu,
        mu_factor, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()), T.coarseToFine(Y.Location()), use_mma);
    } else {

      if (!use_mma) {

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
          kappa, mass, mu, mu_factor, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
          T.coarseToFine(Y.Location()), use_mma);

      } else {

        constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
        constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

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

        V vAccessor(v_);
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
          cAccessor, cInvAccessor, Y, X, Yatomic, Xatomic, uv, const_cast<cudaColorSpinorField &>(v_), v_, g,
          *dummyClover(), kappa, mass, mu, mu_factor, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
          T.coarseToFine(Y.Location()), use_mma);
      }
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat, int fineSpin>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
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
        calculateYcoarse<Float, vFloat, 6, fineSpin, 6, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv,
                                                                    kappa, mass, mu, mu_factor, dirac, matpc,
                                                                    need_bidirectional, use_mma);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
    } else
#endif
    if (fineColor == 24) { // coarsened Wilson or free field staggered
      if (coarseColor == 24) {
        calculateYcoarse<Float, vFloat, 24, fineSpin, 24, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else
#ifdef NSPIN4
        if (coarseColor == 32) {
        calculateYcoarse<Float, vFloat, 24, fineSpin, 32, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else
#endif // NSPIN4
#ifdef NSPIN1
        if (coarseColor == 64) {
        calculateYcoarse<Float, vFloat, 24, fineSpin, 64, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else // --- note, coarsening Nc == 24 -> Nc == 96 for staggered is worth revisiting in the future
#endif
      {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
#ifdef NSPIN4
    } else if (fineColor == 32) {
      if (coarseColor == 32) {
        calculateYcoarse<Float, vFloat, 32, fineSpin, 32, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      }
#endif // NSPIN4
#ifdef NSPIN1
    } else if (fineColor == 64) {
      if (coarseColor == 64) {
        calculateYcoarse<Float, vFloat, 64, fineSpin, 64, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else if (coarseColor == 96) {
        calculateYcoarse<Float, vFloat, 64, fineSpin, 96, coarseSpin>(Y, X, Yatomic, Xatomic, uv, T, g, clover,
                                                                      cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                                                                      need_bidirectional, use_mma);
      } else {
        errorQuda("Unsupported fineColor = %d coarseColor = %d\n", fineColor, coarseColor);
      } // --- note, revisit Nc == 96 -> Nc == 96 in the future
#endif // NSPIN1
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, typename vFloat>
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                        bool need_bidirectional, bool use_mma)
  {
    if (T.Vectors().Nspin() == 2) {
      calculateYcoarse<Float, vFloat, 2>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu, mu_factor,
                                         dirac, matpc, need_bidirectional, use_mma);
    } else {
      errorQuda("Unsupported number of spins %d\n", T.Vectors().Nspin());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateYcoarse(GaugeField &Y, GaugeField &X, GaugeField &Yatomic, GaugeField &Xatomic, ColorSpinorField &uv,
                        const Transfer &T, const GaugeField &g, const GaugeField &clover, const GaugeField &cloverInv,
                        double kappa, double mass, double mu, double mu_factor, QudaDiracType dirac, QudaMatPCType matpc,
                        bool need_bidirectional, bool use_mma)
  {
#ifdef GPU_MULTIGRID
    checkPrecision(X, Y, g, clover, cloverInv, uv, T.Vectors(X.Location()));
    checkPrecision(Xatomic, Yatomic);

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing Y field......\n");
    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (use_mma) errorQuda("MG-MMA does not support double precision, yet.");
      if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
        calculateYcoarse<double, double>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu, mu_factor,
                                         dirac, matpc, need_bidirectional, use_mma);
      } else {
	errorQuda("Unsupported precision %d\n", Y.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
      if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
        calculateYcoarse<float, float>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu, mu_factor, dirac,
                                       matpc, need_bidirectional, use_mma);
      } else {
	errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
#else
      errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
    } else if (Y.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
      if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
        calculateYcoarse<float, short>(Y, X, Yatomic, Xatomic, uv, T, g, clover, cloverInv, kappa, mass, mu, mu_factor, dirac,
                                       matpc, need_bidirectional, use_mma);
      } else {
	errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
#else
      errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
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
                      const GaugeField &clover, const GaugeField &cloverInv, double kappa, double mass, double mu, double mu_factor,
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

    if (!use_mma) {

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

      calculateYcoarse(Y, X, *Yatomic, *Xatomic, *uv, T, gauge, clover, cloverInv, kappa, mass, mu, mu_factor, dirac, matpc,
                       need_bidirectional, use_mma);

      if (Yatomic != &Y) delete Yatomic;
      if (Xatomic != &X) delete Xatomic;

    } else {

      if (location == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("use_mma = true does not go with QUDA_CPU_FIELD_LOCATION.");
      }

      // The MMA implementation requires fields to be in AoS order: we check each gauge field, and create a copy if not.
      // The UV field is an intermediate field, its order doesn't matter; The AoS copy of the V field will be created
      // later in the code path.
      constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

      auto create_gauge_copy = [](const GaugeField &X, QudaGaugeFieldOrder order, bool copy_content) -> auto
      {
        GaugeField *output = nullptr;
        if (X.Order() == order) {
          output = const_cast<GaugeField *>(&X);
        } else {
          GaugeFieldParam param(X);
          param.order = order;
          output = cudaGaugeField::Create(param);
          if (copy_content) output->copy(X);
        }
        return static_cast<cudaGaugeField *>(output);
      };

      auto Y_order = create_gauge_copy(Y, gOrder, false);
      auto X_order = create_gauge_copy(X, gOrder, false);
      auto G_order = create_gauge_copy(gauge, gOrder, true);
      auto C_order = create_gauge_copy(clover, gOrder, true);
      auto I_order = create_gauge_copy(cloverInv, gOrder, true);

      GaugeField *Yatomic = nullptr;
      GaugeField *Xatomic = nullptr;

      if (Y.Precision() < QUDA_SINGLE_PRECISION) {
        // we need to coarsen into single precision fields (float or int), so we allocate temporaries for this purpose
        // else we can just coarsen directly into the original fields
        GaugeFieldParam param(*X_order); // use X since we want scalar geometry
        param.location = location;
        param.order = gOrder;
        param.setPrecision(QUDA_SINGLE_PRECISION);

        Yatomic = GaugeField::Create(param);
        Xatomic = GaugeField::Create(param);
      } else {
        Yatomic = Y_order;
        Xatomic = X_order;
      }

      calculateYcoarse(*Y_order, *X_order, *Yatomic, *Xatomic, *uv, T, *G_order, *C_order, *I_order, kappa, mass, mu,
                       mu_factor, dirac, matpc, need_bidirectional, use_mma);

      if (Yatomic != Y_order) delete Yatomic;
      if (Xatomic != X_order) delete Xatomic;

      if (&Y != Y_order) {
        Y.copy(*Y_order);
        delete Y_order;
      }

      if (&X != X_order) {
        X.copy(*X_order);
        delete X_order;
      }

      if (&gauge != G_order) { delete G_order; }
      if (&clover != C_order) { delete C_order; }
      if (&cloverInv != I_order) { delete I_order; }
    }

    delete uv;
  }
} //namespace quda
