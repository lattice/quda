#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>
#include <tunable_nd.h>

#include <staggered_kd_build_xinv.h>
#include <kernels/staggered_coarse_op_kernel.cuh>

namespace quda {

  template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredKDBlock : public TunableKernel3D {

    Arg &arg;
    const GaugeField &meta;
    GaugeField &X;

    long long flops() const {
      // only real work is multiplying the mass by two
      return arg.coarseVolumeCB*coarseSpin*coarseColor;
    }

    long long bytes() const
    {
      // 1. meta.Bytes() / 2 b/c the Kahler-Dirac blocking is a dual decomposition: only
      //    half of the gauge field needs to be loaded.
      // 2. Storing X, extra factor of two b/c it stores forwards and backwards.
      // 3. Storing mass contribution
      return meta.Bytes() / 2 + (meta.Bytes() * X.Precision()) / meta.Precision() + 2 * coarseSpin * coarseColor * arg.coarseVolumeCB * X.Precision();
    }

    unsigned int minThreads() const { return arg.fineVolumeCB; }
    bool tuneSharedBytes() const { return false; } // FIXME don't tune the grid dimension

  public:
    CalculateStaggeredKDBlock(Arg &arg, const GaugeField &meta, GaugeField &X) :
      TunableKernel3D(meta, fineColor*fineColor, 2),
      arg(arg),
      meta(meta),
      X(X)
    {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux,",computeStaggeredKDBlock");
      strcat(aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && X.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
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
      if (X.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }
  };

  /**
     @brief Calculate the staggered Kahler-Dirac block (coarse clover)

     @param X[out] KD block (coarse clover field) accessor
     @param G[in] Fine grid link / gauge field accessor
     @param X_[out] KD block (coarse clover field)
     @param G_[in] Fine gauge field
     @param mass[in] mass
   */
  template<typename Float, int fineColor, int coarseSpin, int coarseColor, typename xGauge, typename fineGauge>
  void calculateStaggeredKDBlock(xGauge &X, fineGauge &G, GaugeField &X_, const GaugeField &G_, double mass)
  {
    // sanity checks
    if (fineColor != 3)
      errorQuda("Input gauge field should have nColor=3, not nColor=%d\n", fineColor);

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    if (fineColor * 16 != coarseColor*coarseSpin)
      errorQuda("Fine nColor=%d is not consistent with KD dof %d", fineColor, coarseColor*coarseSpin);

    int x_size[QUDA_MAX_DIM] = { };
    int xc_size[QUDA_MAX_DIM] = { };
    for (int i = 0; i < nDim; i++) {
      x_size[i] = G_.X()[i];
      xc_size[i] = X_.X()[i];
      // check that local volumes are consistent
      if (2 * xc_size[i] != x_size[i]) {
        errorQuda("Inconsistent fine dimension %d and coarse KD dimension %d", x_size[i], xc_size[i]);
      }
    }
    x_size[4] = xc_size[4] = 1;

    // Calculate X (KD block), which is really just a permutation of the gauge fields w/in a KD block
    using Arg = CalculateStaggeredYArg<Float, coarseSpin, fineColor, coarseColor, xGauge, fineGauge, true>;
    Arg arg(X, X, G, mass, x_size, xc_size);
    CalculateStaggeredKDBlock<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, G_, X_);

    QudaFieldLocation location = checkLocation(X_, G_);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Calculating the KD block on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // We know exactly what the scale should be: the max of all of the (fat) links.
    double max_scale = G_.abs_max();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global U_max = %e\n", max_scale);

    if (xGauge::fixedPoint()) {
      arg.X.resetScale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
      X_.Scale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing KD block\n");
    y.apply(device::get_default_stream());

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X_.norm2(0));
  }

  template <typename Float, typename vFloat, int fineColor, int coarseColor, int coarseSpin>
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {

    QudaFieldLocation location = X.Location();

    if (location == QUDA_CPU_FIELD_LOCATION) {

      constexpr QudaGaugeFieldOrder gOrder = QUDA_QDP_GAUGE_ORDER;

      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder>;
      using xCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;

      gFine gAccessor(const_cast<GaugeField&>(g));
      xCoarse xAccessor(const_cast<GaugeField&>(X));

      calculateStaggeredKDBlock<Float,fineColor,coarseSpin,coarseColor>(xAccessor, gAccessor, X, g, mass);

    } else {

      constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

      if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

      using gFine = typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float>;
      using xCoarse = typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat>;

      gFine gAccessor(const_cast<GaugeField&>(g));
      xCoarse xAccessor(const_cast<GaugeField&>(X));

      calculateStaggeredKDBlock<Float,fineColor,coarseSpin,coarseColor>(xAccessor, gAccessor, X, g, mass);
    }

  }

  // template on the number of KD (coarse) degrees of freedom
  template <typename Float, typename vFloat, int fineColor>
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField& g, const double mass)
  {
    constexpr int coarseSpin = 2;
    const int coarseColor = X.Ncolor() / coarseSpin;

    if (coarseColor == 24) { // half the dof w/in a KD-block
      calculateStaggeredKDBlock<Float,vFloat,fineColor,24,coarseSpin>(X, g, mass);
    } else {
      errorQuda("Unsupported number of Kahler-Dirac dof %d\n", X.Ncolor());
    }
  }

  // template on fine colors
  template <typename Float, typename vFloat>
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {
    if (g.Ncolor() == 3) {
      calculateStaggeredKDBlock<Float,vFloat,3>(X, g, mass);
    } else {
      errorQuda("Unsupported number of colors %d\n", g.Ncolor());
    }
  }

  //Does the heavy lifting of building X
#if defined(GPU_STAGGERED_DIRAC) && defined(GPU_MULTIGRID)
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {
    // FIXME remove when done
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing X for StaggeredKD...\n");

#if QUDA_PRECISION & 8 && defined(GPU_MULTIGRID_DOUBLE)
    if (X.Precision() == QUDA_DOUBLE_PRECISION) {
      calculateStaggeredKDBlock<double,double>(X, g, mass);
    } else
#endif
#if QUDA_PRECISION & 4
    if (X.Precision() == QUDA_SINGLE_PRECISION) {
      calculateStaggeredKDBlock<float,float>(X, g, mass);
    } else
#endif
#if QUDA_PRECISION & 2
    if (X.Precision() == QUDA_HALF_PRECISION) {
      calculateStaggeredKDBlock<float,short>(X, g, mass);
    } else
#endif
//#if QUDA_PRECISION & 1
//    if (X.Precision() == QUDA_QUARTER_PRECISION) {
//      calculateStaggeredKDBlock<float,int8_t>(X, g, mass);
//    } else
//#endif
    {
      errorQuda("Unsupported precision %d", X.Precision());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("....done computing X for StaggeredKD\n");
  }
#else
  void calculateStaggeredKDBlock(GaugeField &, const GaugeField &, const double)
  {
    errorQuda("Staggered fermion multigrid support has not been built");
  }
#endif

  // Calculates the inverse KD block and puts the result in Xinv. Assumes Xinv has been allocated
  void BuildStaggeredKahlerDiracInverse(GaugeField &Xinv, const cudaGaugeField &gauge, const double mass)
  {
    using namespace blas_lapack;
    auto invert = use_native() ? native::BatchInvertMatrix : generic::BatchInvertMatrix;

    QudaFieldLocation location = checkLocation(Xinv, gauge);
    QudaPrecision precision = checkPrecision(Xinv, gauge);

    if (Xinv.Geometry() != QUDA_KDINVERSE_GEOMETRY)
      errorQuda("Unsupported gauge geometry %d , expected %d for Xinv", Xinv.Geometry(), QUDA_KDINVERSE_GEOMETRY);

    // Note: the BatchInvertMatrix abstraction only supports single precision for now,
    // so we copy a lot of intermediates to single precision early on.

    // Step 1: build temporary Xinv field in QUDA_MILC_GAUGE_ORDER,
    // independent of field location. Xinv is always single precision
    // because it's an intermediate field.
    std::unique_ptr<GaugeField> xInvMilcOrder(nullptr);
    {
      const int ndim = 4;
      int xc[QUDA_MAX_DIM];
      for (int i = 0; i < ndim; i++) { xc[i] = gauge.X()[i]/2; }
      const int Nc_c = gauge.Ncolor() * 8; // 24
      const int Ns_c = 2; // staggered parity
      GaugeFieldParam gParam;
      memcpy(gParam.x, xc, QUDA_MAX_DIM*sizeof(int));
      gParam.nColor = Nc_c*Ns_c;
      gParam.reconstruct = QUDA_RECONSTRUCT_NO;
      gParam.order = QUDA_MILC_GAUGE_ORDER;
      gParam.link_type = QUDA_COARSE_LINKS;
      gParam.t_boundary = QUDA_PERIODIC_T;
      gParam.create = QUDA_ZERO_FIELD_CREATE;
      gParam.setPrecision( QUDA_SINGLE_PRECISION );
      gParam.nDim = ndim;
      gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      gParam.nFace = 0;
      gParam.geometry = QUDA_SCALAR_GEOMETRY;
      gParam.pad = 0;

      if (location == QUDA_CUDA_FIELD_LOCATION)
        xInvMilcOrder = std::make_unique<cudaGaugeField>(gParam);
      else if (location == QUDA_CPU_FIELD_LOCATION)
        xInvMilcOrder = std::make_unique<cpuGaugeField>(gParam);
      else
        errorQuda("Invalid field location %d", location);

    }

    // Step 2: build a host or device gauge field as appropriate, but
    // in any case change to reconstruct 18 so we can use fine-grained
    // accessors for constructing X. Logic copied from `staggered_coarse_op.cu`
    bool need_new_U = true;
    if (location == QUDA_CUDA_FIELD_LOCATION && gauge.Reconstruct() == QUDA_RECONSTRUCT_NO && gauge.Precision() == QUDA_SINGLE_PRECISION)
      need_new_U = false;

    std::unique_ptr<GaugeField> tmp_U(nullptr);

    if (need_new_U) {
      if (location == QUDA_CPU_FIELD_LOCATION) {

        //First make a cpu gauge field from the cuda gauge field
        int pad = 0;
        GaugeFieldParam gf_param(gauge.X(), QUDA_SINGLE_PRECISION, QUDA_RECONSTRUCT_NO, pad, gauge.Geometry());
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

        tmp_U = std::make_unique<cpuGaugeField>(gf_param);

        //Copy the cuda gauge field to the cpu
        gauge.saveCPUField(reinterpret_cast<cpuGaugeField&>(*tmp_U));

      } else if (location == QUDA_CUDA_FIELD_LOCATION) {

        // We can assume: gauge.Reconstruct() != QUDA_RECONSTRUCT_NO || gauge.Precision() != QUDA_SINGLE_PRECISION)
        GaugeFieldParam gf_param(gauge);
        gf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        gf_param.order = QUDA_FLOAT2_GAUGE_ORDER; // guaranteed for no recon
        gf_param.setPrecision( QUDA_SINGLE_PRECISION );
        tmp_U = std::make_unique<cudaGaugeField>(gf_param);

        tmp_U->copy(gauge);
      }
    }

    const GaugeField& U = need_new_U ? *tmp_U : reinterpret_cast<const GaugeField&>(gauge);

    // Step 3: Create the X field based on Xinv, but switch to a native ordering for a GPU setup.
    std::unique_ptr<GaugeField> tmp_X(nullptr);
    GaugeFieldParam x_param(*xInvMilcOrder);
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      x_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      x_param.setPrecision(x_param.Precision());
      tmp_X = std::make_unique<cudaGaugeField>(x_param);
    } else {
      tmp_X = std::make_unique<cpuGaugeField>(x_param);
    }
    GaugeField& X = *tmp_X;

    // Step 4: Calculate X from U
    calculateStaggeredKDBlock(X, U, mass);

    // FIXME: expose on command line, though empirically we've seen
    // this leads to poorer conditioned operator. Might be worth it because
    // we could apply the dagger approx. operator directly without
    // building the KD block, though
    constexpr bool dagger_approximation = false;

    if (dagger_approximation) {
      xInvMilcOrder->copy(X);
    } else {
      // Logic copied from `coarse_op_preconditioned.cu`
      const int n = xInvMilcOrder->Ncolor();
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        // FIXME: add support for double precision inverse
        // Reorder to MILC order for inversion, based on "coarse_op_preconditioned.cu"
        GaugeFieldParam param(*xInvMilcOrder);
        param.order = QUDA_MILC_GAUGE_ORDER; // MILC order == QDP order for Xinv
        param.setPrecision(QUDA_SINGLE_PRECISION);
        cudaGaugeField X_(param);
        
        X_.copy(X);

        blas::flops += invert((void*)xInvMilcOrder->Gauge_p(), (void*)X_.Gauge_p(), n, X_.Volume(), X_.Precision(), X.Location());

      } else if (location == QUDA_CPU_FIELD_LOCATION) {

        blas::flops += invert((void*)xInvMilcOrder->Gauge_p(), (void*)X.Gauge_p(), n, X.Volume(), X.Precision(), X.Location());
      }

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("xInvMilcOrder = %e\n", xInvMilcOrder->norm2(0));

    }

    // Step 6: reorder the KD inverse into a "gauge field" with a QUDA_KDINVERSE_GEOMETRY
    // last two parameters: dagger approximation, mass (which becomes a scale in the dagger approx)
    ReorderStaggeredKahlerDiracInverse(Xinv, *xInvMilcOrder, dagger_approximation, mass);    

  }


  // Allocates and calculates the inverse KD block, returning Xinv
  std::unique_ptr<GaugeField> AllocateAndBuildStaggeredKahlerDiracInverse(const cudaGaugeField &gauge, const double mass)
  {
    GaugeFieldParam gParam(gauge);
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.geometry = QUDA_KDINVERSE_GEOMETRY;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.pad = 0;

    // latter true is to force FLOAT2
    gParam.setPrecision(gauge.Precision(), true);

    std::unique_ptr<GaugeField> Xinv(reinterpret_cast<GaugeField*>(new cudaGaugeField(gParam)));

    BuildStaggeredKahlerDiracInverse(*Xinv, gauge, mass);

    return Xinv;
  }

} //namespace quda
