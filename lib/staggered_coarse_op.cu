#include <tune_quda.h>
#include <transfer.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_coarse_op_kernel.cuh>

namespace quda {

  template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredY : public TunableVectorYZ {

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
    bool tuneGridDim() const { return false; } // don't tune the grid dimension
    bool tuneAuxDim() const { return false; }

  public:
    CalculateStaggeredY(Arg &arg, const GaugeField &meta, GaugeField &Y, GaugeField &X) :
      TunableVectorYZ(fineColor*fineColor, 2),
      arg(arg),
      meta(meta),
      Y(Y),
      X(X)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/staggered_coarse_op_kernel.cuh");
#endif
      }
      strcpy(aux, compile_type_str(meta));
      strcpy(aux, meta.AuxString());
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
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        ComputeStaggeredVUVCPU<Float,fineColor,coarseSpin,coarseColor>(arg);
      } else {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::ComputeStaggeredVUVGPU")
          .instantiate(Type<Float>(),fineColor,coarseSpin,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
        ComputeStaggeredVUVGPU<Float,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif // JITIFY
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
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
    int spin_bs = 0; // 0 -> spin-less types.

    // Calculate VUV in one pass (due to KD-transform) for each dimension,
    // accumulating directly into the coarse gauge field Y

    typedef CalculateStaggeredYArg<Float,coarseSpin,fineColor,coarseColor,coarseGauge,fineGauge> Arg;
    Arg arg(Y, X, G, mass, x_size, xc_size, geo_bs, spin_bs);
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
    y.apply(0);

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

      typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder> gFine;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat> gCoarse;

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

      constexpr bool use_tex = __COMPUTE_CAPABILITY__ < 520 ? true : false; // on pre-Maxwell-2 use textures/ldg to get caching
      typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float,use_tex> gFine;
      typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat> gCoarse;

      gFine gAccessor(const_cast<GaugeField&>(g));
      gCoarse yAccessor(const_cast<GaugeField&>(Y));
      gCoarse xAccessor(const_cast<GaugeField&>(X));

      calculateStaggeredY<Float,fineSpin,fineColor,coarseSpin,coarseColor>
        (yAccessor, xAccessor, gAccessor, Y, X, g, mass, dirac, matpc);
    }

  }

  // template on the number of coarse degrees of freedom
  template <typename Float, typename vFloat, int fineColor, int fineSpin>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    const int coarseSpin = 2;
    const int coarseColor = Y.Ncolor() / coarseSpin;

    if (coarseColor == 24) { // free field staggered
      calculateStaggeredY<Float,vFloat,fineColor,fineSpin,24,coarseSpin>(Y, X, T, g, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // template on fine spin
  template <typename Float, typename vFloat, int fineColor>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if (T.Vectors().Nspin() == 1) {
      calculateStaggeredY<Float,vFloat,fineColor,1>(Y, X, T, g, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of spins %d\n", T.Vectors(X.Location()).Nspin());
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat>
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    if (g.Ncolor() == 3) {
      calculateStaggeredY<Float,vFloat,3>(Y, X, T, g, mass, dirac, matpc);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  //Does the heavy lifting of creating the coarse color matrices Y
  void calculateStaggeredY(GaugeField &Y, GaugeField &X, const Transfer &T, const GaugeField &g,
                           double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
#if defined(GPU_MULTIGRID) && defined(GPU_STAGGERED_DIRAC)
    checkPrecision(T.Vectors(X.Location()), X, Y);

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing Y field......\n");

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      if (T.Vectors(X.Location()).Precision() == QUDA_DOUBLE_PRECISION) {
        calculateStaggeredY<double,double>(Y, X, T, g, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", Y.Precision());
      }
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_SINGLE_PRECISION) {
        calculateStaggeredY<float,float>(Y, X, T, g, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
    } else if (Y.Precision() == QUDA_HALF_PRECISION) {
      if (T.Vectors(X.Location()).Precision() == QUDA_HALF_PRECISION) {
        calculateStaggeredY<float,short>(Y, X, T, g, mass, dirac, matpc);
      } else {
        errorQuda("Unsupported precision %d\n", T.Vectors(X.Location()).Precision());
      }
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("....done computing Y field\n");
#else
    errorQuda("Staggered multigrid has not been built");
#endif
  }

  //Calculates the coarse color matrix and puts the result in Y.
  //N.B. Assumes Y, X have been allocated.
  void StaggeredCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, const cudaGaugeField &gauge,
                         double mass, QudaDiracType dirac, QudaMatPCType matpc)
  {
    QudaPrecision precision = Y.Precision();
    QudaFieldLocation location = checkLocation(Y, X);

    GaugeField *U = location == QUDA_CUDA_FIELD_LOCATION ? const_cast<cudaGaugeField*>(&gauge) : nullptr;

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

    calculateStaggeredY(Y, X, T, *U, mass, dirac, matpc);

    if (U != &gauge) delete U;
  }

} //namespace quda
