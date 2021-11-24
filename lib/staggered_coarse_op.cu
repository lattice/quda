#include <transfer.h>
#include <gauge_field.h>
#include <tunable_nd.h>

// For naive Kahler-Dirac coarsening
#include <kernels/staggered_coarse_op_kernel.cuh>

// This define controls which kernels get compiled in `coarse_op.cuh`.
// This ensures only kernels relevant for coarsening the staggered
// operator get built, saving compile time.
#define STAGGEREDCOARSE
#include <coarse_op.cuh>

namespace quda {

  template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredY : public TunableKernel3D {

    Arg &arg;
    const GaugeField &meta;
    GaugeField &Y;
    GaugeField &X;

    long long flops() const { return arg.coarseVolumeCB*coarseSpin*coarseColor; }

    long long bytes() const
    {
      // 2 from forwards / backwards contributions, Y and X are sparse - only needs to write non-zero elements, 2nd term is mass term
      return meta.Bytes() + (2 * meta.Bytes() * Y.Precision()) / meta.Precision() + 2 * 2 * coarseSpin * coarseColor * arg.coarseVolumeCB * X.Precision();
    }

    unsigned int minThreads() const { return arg.fineVolumeCB; }
    bool tuneSharedBytes() const { return false; } // don't tune the grid dimension

  public:
    CalculateStaggeredY(Arg &arg, const GaugeField &meta, GaugeField &Y, GaugeField &X) :
      TunableKernel3D(meta, fineColor*fineColor, 2),
      arg(arg),
      meta(meta),
      Y(Y),
      X(X)
    {
      strcat(aux,comm_dim_partitioned_string());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux,",computeStaggeredVUV");
      strcat(aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
             meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      strcat(aux,"coarse_vol=");
      strcat(aux,X.VolString());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        launch_host<ComputeStaggeredVUV>(tp, stream, arg);
      } else {
        launch_device<ComputeStaggeredVUV>(tp, stream, arg);
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }
  };

  /**
     @brief Calculate the coarse-link field, including the coarse clover field.

     @param Y[out] Coarse (fat-)link field accessor
     @param X[out] Coarse clover field accessor
     @param G[in] Fine grid link / gauge field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param G_[in] Fine gauge field
     @param mass[in] Kappa parameter
     @param matpc[in] The type of preconditioning of the source fine-grid operator
   */
  template<typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	   typename coarseGauge, typename fineGauge>
  void calculateStaggeredY(coarseGauge &Y, coarseGauge &X, fineGauge &G, GaugeField &Y_, GaugeField &X_,
                           const GaugeField &G_, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    // sanity checks
    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    // This is the last time we use fineSpin, since this file only coarsens
    // staggered-type ops, not wilson-type AND coarse-type.
    if (fineSpin != 1)
      errorQuda("Input Dirac operator %d should have nSpin=1, not nSpin=%d\n", dirac, fineSpin);
    if (fineColor != 3)
      errorQuda("Input Dirac operator %d should have nColor=3, not nColor=%d\n", dirac, fineColor);

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int x_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) x_size[i] = G_.X()[i];
    x_size[4] = 1;

    int xc_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) xc_size[i] = X_.X()[i];
    xc_size[4] = 1;

    int geo_bs[QUDA_MAX_DIM] = { };
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];

    // Calculate VUV in one pass (due to KD-transform) for each dimension,
    // accumulating directly into the coarse gauge field Y

    using Arg = CalculateStaggeredYArg<Float,coarseSpin,fineColor,coarseColor,coarseGauge,fineGauge>;
    Arg arg(Y, X, G, mass, x_size, xc_size, geo_bs);
    CalculateStaggeredY<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, G_, Y_, X_);

    QudaFieldLocation location = checkLocation(Y_, X_, G_);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Running link coarsening on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // We know exactly what the scale should be: the max of all of the (fat) links.
    double max_scale = G_.abs_max();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global U_max = %e\n", max_scale);

    if (coarseGauge::fixedPoint()) {
      arg.Y.resetScale(max_scale);
      arg.X.resetScale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
      Y_.Scale(max_scale);
      X_.Scale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
    }

    // We can technically do a uni-directional build, but becauase
    // the coarse link builds are just permutations plus lots of zeros,
    // it's faster to skip the flip!

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing VUV\n");
    y.apply(device::get_default_stream());

    if (getVerbosity() >= QUDA_VERBOSE) {
      for (int d = 0; d < nDim; d++) printfQuda("Y2[%d] = %e\n", 4+d, Y_.norm2( 4+d ));
      for (int d = 0; d < nDim; d++) printfQuda("Y2[%d] = %e\n", d, Y_.norm2( d ));
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X_.norm2(0));
  }

  template <typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    QudaFieldLocation location = Y.Location();

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder>;
      using gCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;

      gFine gAccessor(const_cast<GaugeField&>(g));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));

      calculateStaggeredY<Float,fineSpin,fineColor,coarseSpin,coarseColor>
        (yAccessor, xAccessor, gAccessor, Y, X, g, mass, dirac, matpc);
    } else {

      constexpr QudaFieldOrder csOrder = QUDA_FLOAT2_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float>;
      using gCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;

      gFine gAccessor(const_cast<GaugeField&>(g));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));

      calculateStaggeredY<Float,fineSpin,fineColor,coarseSpin,coarseColor>
        (yAccessor, xAccessor, gAccessor, Y, X, g, mass, dirac, matpc);
    }

  }

  template <typename Float, typename vFloat, typename vFloatXinv, int fineColor, int fineSpin, int xinvColor, int xinvSpin, int coarseColor, int coarseSpin>
  void aggregateStaggeredY(GaugeField &Y, GaugeField &X,
                        const Transfer &T, const GaugeField &g, const GaugeField &XinvKD, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    // Actually create the temporaries like UV, etc.
    auto location = Y.Location();

    //Create a field UV which holds U*V.  Has the same structure as V,
    // No need to double spin for the staggered operator, will need to double it for the K-D op.
    ColorSpinorParam UVparam(T.Vectors(location));
    UVparam.create = QUDA_ZERO_FIELD_CREATE;
    UVparam.location = location;
    UVparam.setPrecision(T.Vectors(location).Precision());
    UVparam.mem_type = Y.MemType(); // allocate temporaries to match coarse-grid link field

    ColorSpinorField *uv = ColorSpinorField::Create(UVparam);

    ColorSpinorField *av = (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) ? ColorSpinorField::Create(UVparam) : &const_cast<ColorSpinorField&>(T.Vectors(location));
    
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

    // Moving along to the build

    const double kappa = -1.; // cancels a minus sign factor for kappa w/in the dslash application
    const double mu_dummy = 0.; 
    const double mu_factor_dummy = 0.;
    constexpr bool use_mma = false;
    
    bool need_bidirectional = false;
    if (dirac == QUDA_STAGGEREDKD_DIRAC || dirac == QUDA_ASQTADKD_DIRAC) need_bidirectional = true;

    constexpr QudaGaugeFieldOrder xinvOrder = QUDA_MILC_GAUGE_ORDER;
    if (XinvKD.FieldOrder() != xinvOrder && XinvKD.Bytes()) errorQuda("unsupported field order %d\n", XinvKD.FieldOrder());

    if (Y.Location() == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaFieldOrder csOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using V = typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat>;
      using F = typename colorspinor::FieldOrderCB<Float,fineSpin,fineColor,coarseColor,csOrder,vFloat>; // will need 2x the spin components for the KD op
      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder>;
      using xinvFine = typename gauge::FieldOrder<typename mapper<vFloatXinv>::type,xinvColor,xinvSpin,xinvOrder,true,vFloatXinv>;
      using gCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;
      using gCoarseAtomic = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,storeType>;

      const ColorSpinorField &v = T.Vectors(Y.Location());

      V vAccessor(const_cast<ColorSpinorField&>(v));
      V uvAccessor(*uv); // will need 2x the spin components for the KD op
      F avAccessor(*av);
      gFine gAccessor(const_cast<GaugeField&>(g));
      xinvFine xinvAccessor(const_cast<GaugeField&>(XinvKD));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));
      gCoarseAtomic yAccessorAtomic(*Yatomic);
      gCoarseAtomic xAccessorAtomic(*Xatomic);
      
      // the repeated xinvAccessor is intentional
      calculateY<use_mma, QUDA_CPU_FIELD_LOCATION, false, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, avAccessor, vAccessor, gAccessor, xinvAccessor, xinvAccessor,
        Y, X, *Yatomic, *Xatomic, *uv, *av, v, kappa, mass, mu_dummy,
        mu_factor_dummy, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()), T.coarseToFine(Y.Location()));
    } else {

      constexpr QudaFieldOrder csOrder = QUDA_FLOAT2_FIELD_ORDER;
      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (T.Vectors(Y.Location()).FieldOrder() != csOrder)
        errorQuda("Unsupported field order %d\n", T.Vectors(Y.Location()).FieldOrder());
      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using V = typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat, vFloat, false, false>;
      using F = typename colorspinor::FieldOrderCB<Float, fineSpin, fineColor, coarseColor, csOrder, vFloat, vFloat, false, false>; // will need 2x the spin components for the KD op
      using gFine =  typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float>;
      using xinvFine = typename gauge::FieldOrder<typename mapper<vFloatXinv>::type,xinvColor,xinvSpin,xinvOrder,true,vFloatXinv>;
      using gCoarse = typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, vFloat>;
      using gCoarseAtomic = typename gauge::FieldOrder<Float, coarseColor * coarseSpin, coarseSpin, gOrder, true, storeType>;

      const ColorSpinorField &v = T.Vectors(Y.Location());

      V vAccessor(const_cast<ColorSpinorField &>(v));
      V uvAccessor(*uv); // will need 2x the spin components for the KD op
      F avAccessor(*av);
      gFine gAccessor(const_cast<GaugeField &>(g));
      xinvFine xinvAccessor(const_cast<GaugeField&>(XinvKD));
      gCoarse yAccessor(const_cast<GaugeField &>(Y));
      gCoarse xAccessor(const_cast<GaugeField &>(X));
      gCoarseAtomic yAccessorAtomic(*Yatomic);
      gCoarseAtomic xAccessorAtomic(*Xatomic);

      // create a dummy clover field to allow us to call the external clover reduction routines elsewhere
      calculateY<use_mma, QUDA_CUDA_FIELD_LOCATION, false, Float, fineSpin, fineColor, coarseSpin, coarseColor>(
        yAccessor, xAccessor, yAccessorAtomic, xAccessorAtomic, uvAccessor, avAccessor, vAccessor, gAccessor,
        xinvAccessor, xinvAccessor, Y, X, *Yatomic, *Xatomic, *uv, *av, v,
        kappa, mass, mu_dummy, mu_factor_dummy, dirac, matpc, need_bidirectional, T.fineToCoarse(Y.Location()),
        T.coarseToFine(Y.Location()));
    }

    // Clean up
    if (Yatomic != &Y) delete Yatomic;
    if (Xatomic != &X) delete Xatomic;

    if (av != nullptr && &T.Vectors(location) != av) delete av;
    if (uv != nullptr) delete uv;

  }

  // template on precision of XinvKD
  template <typename Float, typename vFloat, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void aggregateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           const GaugeField &XinvKD, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if (XinvKD.Ncolor() != 16*fineColor)
      errorQuda("Invalid Nc %d", XinvKD.Ncolor());

    constexpr int xinvColor = 16 * fineColor;
    constexpr int xinvSpin = 2;

    if (XinvKD.Precision() == QUDA_SINGLE_PRECISION) {
      aggregateStaggeredY<Float,vFloat,float,fineColor,fineSpin,xinvColor,xinvSpin,coarseColor,coarseSpin>(Y, X, T, g, XinvKD, mass, dirac, matpc);
    //} else if (XinvKD.Precision() == QUDA_HALF_PRECISION) { --- unnecessary until we add KD coarsening support
    //  aggregateStaggeredY<Float,vFloat,short,fineColor,fineSpin,xinvColor,xinvSpin,coarseColor,coarseSpin>(Y, X, T, g, XinvKD, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported precision %d", XinvKD.Precision());
    }
  }


  // template on the number of coarse degrees of freedom, branch between naive K-D 
  // and actual aggregation
  template <typename Float, typename vFloat, int fineColor, int fineSpin>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           const GaugeField &XinvKD, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 24) {
      if (T.getTransferType() == QUDA_TRANSFER_COARSE_KD)
        // the permutation routines don't need Yatomic, Xatomic, uv, av
        calculateStaggeredY<Float,vFloat,fineColor,fineSpin,24,coarseSpin>(Y, X, T, g, mass, dirac, matpc);
      else {
        // free field aggregation
        aggregateStaggeredY<Float,vFloat,fineColor,fineSpin,24,coarseSpin>(Y, X, T, g, XinvKD, mass, dirac, matpc);
      }
    } else if (coarseColor == 64) {
      aggregateStaggeredY<Float,vFloat,fineColor,fineSpin,64,coarseSpin>(Y, X, T, g, XinvKD, mass, dirac, matpc);
    } else { // note --- may revisit 3 -> 96 in the future
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, typename vFloat, int fineColor>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           const GaugeField &XinvKD, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if (T.Vectors().Nspin() == 1) {
      calculateStaggeredY<Float,vFloat,fineColor,1>(Y, X, T, g, XinvKD, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of spins %d\n", T.Vectors(X.Location()).Nspin());
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           const GaugeField &XinvKD, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if (g.Ncolor() == 3) {
      calculateStaggeredY<Float,vFloat,3>(Y, X, T, g, XinvKD, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

#if defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)
  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           const GaugeField &XinvKD, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    checkPrecision(T.Vectors(X.Location()), X, Y);

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing Y field......\n");

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
        calculateStaggeredY<double,double>(Y, X, T, g, XinvKD, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", Y.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else 
#if QUDA_PRECISION & 4
    if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
        calculateStaggeredY<float,float>(Y, X, T, g, XinvKD, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
    } else 
#endif
#if QUDA_PRECISION & 2
    if (Y.Precision() == QUDA_HALF_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
        calculateStaggeredY<float,short>(Y, X, T, g, XinvKD, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
    } else
#endif
    {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("....done computing Y field\n");
  }
#else
  void calculateStaggeredY(GaugeField &, GaugeField &, const Transfer &, const GaugeField &,
                           const GaugeField &, double, QudaDiracType, QudaMatPCType)
  {
    errorQuda("Staggered multigrid has not been built");
  }
#endif

//Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void StaggeredCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge,
                         const cudaGaugeField* XinvKD, double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    QudaPrecision precision = Y.Precision();
    QudaFieldLocation location = checkLocation(Y, X);

    GaugeField *U = location == QUDA_CUDA_FIELD_LOCATION ? const_cast<cudaGaugeField*>(&gauge) : nullptr;
    GaugeField *Xinv = location == QUDA_CUDA_FIELD_LOCATION ? const_cast<cudaGaugeField*>(XinvKD) : nullptr;

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
      gf_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      gf_param.setPrecision(gf_param.Precision());
      U = new cudaGaugeField(gf_param);

      U->copy(gauge);
    }

    // Create an Xinv in the spirit of what's done in coarse_op.cu
    
    GaugeFieldParam xinvParam;
    xinvParam.nDim = 4;
    auto precision_xinv = XinvKD ? XinvKD->Precision() : QUDA_SINGLE_PRECISION;
    if (precision_xinv < QUDA_HALF_PRECISION) { 
      precision_xinv = QUDA_HALF_PRECISION;
    } else if (precision_xinv > QUDA_SINGLE_PRECISION) {
      precision_xinv = QUDA_SINGLE_PRECISION;
    }
    xinvParam.setPrecision( precision_xinv );
    for (int i = 0; i < xinvParam.nDim; i++) xinvParam.x[i] = XinvKD ? XinvKD->X()[i] : 0;
    xinvParam.nColor = XinvKD ? XinvKD->Ncolor() : 16 * U->Ncolor();
    xinvParam.reconstruct = QUDA_RECONSTRUCT_NO;
    xinvParam.order = QUDA_MILC_GAUGE_ORDER;
    xinvParam.link_type = QUDA_COARSE_LINKS;
    xinvParam.t_boundary = QUDA_PERIODIC_T;
    xinvParam.create = QUDA_NULL_FIELD_CREATE;
    xinvParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    xinvParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    xinvParam.nFace = 0;
    xinvParam.geometry = QUDA_SCALAR_GEOMETRY;
    xinvParam.pad = 0;

    if (location == QUDA_CUDA_FIELD_LOCATION && !XinvKD) {
      // create a dummy GaugeField if one is not defined
      xinvParam.order = QUDA_INVALID_GAUGE_ORDER;
      Xinv = new cudaGaugeField(xinvParam);
      
      
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      // Create a cpuGaugeField from the cudaGaugeField
      Xinv = new cpuGaugeField(xinvParam);
      
      if (XinvKD) XinvKD->saveCPUField(*static_cast<cpuGaugeField*>(Xinv));
    }

    calculateStaggeredY(Y, X, T, *U, *Xinv, mass, dirac, matpc);

    if (U != &gauge) delete U;
    if (Xinv != XinvKD) delete Xinv;
  }

} //namespace quda
