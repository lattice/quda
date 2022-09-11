#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>
#include <tunable_nd.h>
#include <instantiate.h>

#include <staggered_kd_build_xinv.h>
#include <kernels/staggered_coarse_op_kernel.cuh>

namespace quda {

  template <typename Float, int fineColor>
  class CalculateStaggeredKDBlock : public TunableKernel3D {

    GaugeField &X;
    const GaugeField &g;
    double mass;

    const int nDim = 4;

    long long flops() const {
      // no work, just a permutation
      return 0ll;
    }

    long long bytes() const
    {
      // 1. meta.Bytes() / 2 b/c the Kahler-Dirac blocking is a dual decomposition: only
      //    half of the gauge field needs to be loaded.
      // 2. Storing X, extra factor of two b/c it stores forwards and backwards.
      // 3. Storing mass contribution
      return g.Bytes() / 2 + (g.Bytes() * X.Precision()) / g.Precision() + X.Ncolor() * X.Volume() * X.Precision();
    }

    unsigned int minThreads() const { return g.VolumeCB(); }

  public:
    CalculateStaggeredKDBlock(const GaugeField &g, GaugeField &X, double mass) :
      TunableKernel3D(g, fineColor*fineColor, 2),
      X(X),
      g(g),
      mass(mass)
    {
      checkPrecision(X, g);
      checkLocation(X, g);
      if (g.Geometry() != QUDA_VECTOR_GEOMETRY)
        errorQuda("Unsupported geometry %d", g.Geometry());
      if (g.Ndim() != 4)
        errorQuda("Number of dimensions %d is not supported", g.Ndim());
      if (X.Geometry() != QUDA_SCALAR_GEOMETRY)
        errorQuda("Unsupported geometry %d", X.Geometry());

      strcat(aux,",computeStaggeredKDBlock");

      // X is relatively sparse; the kernel assumes the rest of X is already zero
      X.zero();

      // reset scales as appropriate
      if constexpr (sizeof(Float) < QUDA_SINGLE_PRECISION) {
        double max_scale = g.abs_max();
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global U_max = %e\n", max_scale);
        X.Scale(max_scale > 2.0*mass ? max_scale : 2.0*mass);
      }

      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      // We're only building X; we specify this because we're reusing the
      // code that builds both X and Y for the "coarse" KD operator
      constexpr bool kd_build_x = true;
      if (X.Location() == QUDA_CPU_FIELD_LOCATION) {
        constexpr QudaGaugeFieldOrder order = QUDA_QDP_GAUGE_ORDER;
        CalculateStaggeredYArg<Float,fineColor,order,kd_build_x> arg(X, X, g, mass);
        launch_host<ComputeStaggeredVUV>(tp, stream, arg);
      } else if (X.Location() == QUDA_CUDA_FIELD_LOCATION) {
        constexpr QudaGaugeFieldOrder order = QUDA_FLOAT2_GAUGE_ORDER;
        CalculateStaggeredYArg<Float,fineColor,order,kd_build_x> arg(X, X, g, mass);
        launch_device<ComputeStaggeredVUV>(tp, stream, arg);
      }
    }
  };


#if defined(GPU_STAGGERED_DIRAC) && defined(GPU_MULTIGRID)
  /**
     @brief Build the Kahler-Dirac term from the fine gauge fields

     @param X[out] supersite-local Kahler-Dirac term
     @param g[in] fine gauge field (fat links for asqtad)
     @param mass[in] Mass of staggered fermion
   */
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {
    // Instantiate based on precision, number of colors
    // need to swizzle `g` to the first argument to get the right fine nColor
    instantiate<CalculateStaggeredKDBlock>(g, X, mass);
  }
#else
  void calculateStaggeredKDBlock(GaugeField &, const GaugeField &, const double)
  {
    errorQuda("Staggered fermion multigrid support has not been built");
  }
#endif

  /**
     @brief Calculates the inverse KD block and puts the result in Xinv. Assumes Xinv has been allocated.

     @param Xinv[out] KD inverse fine gauge in KD geometry
     @param gauge[in] fine gauge field (fat links for asqtad)
     @param mass[in] Mass of staggered fermion
     @param dagger_approximation[in] Whether or not to use the dagger approximation, using the dagger of X instead of Xinv
   */
  void BuildStaggeredKahlerDiracInverse(GaugeField &Xinv, const cudaGaugeField &gauge, const double mass, const bool dagger_approximation)
  {
    using namespace blas_lapack;
    auto invert = use_native() ? native::BatchInvertMatrix : generic::BatchInvertMatrix;

    QudaFieldLocation location = checkLocation(Xinv, gauge);
    checkPrecision(Xinv, gauge);

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
      lat_dim_t xc;
      for (int i = 0; i < ndim; i++) { xc[i] = gauge.X()[i]/2; }
      const int Nc_c = gauge.Ncolor() * 8; // 24
      const int Ns_c = 2; // staggered parity
      GaugeFieldParam gParam;
      gParam.x = xc;
      gParam.nColor = Nc_c*Ns_c;
      gParam.location = location;
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
        gf_param.location = location;
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
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing the KD block on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    calculateStaggeredKDBlock(X, U, mass);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X.norm2(0));

    // Step 5: Calculate Xinv
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

    if (getVerbosity() >= QUDA_VERBOSE) {
      if (dagger_approximation) printfQuda("Using the dagger approximation to Xinv\n");
      printfQuda("xInvKdGeometry = %e\n", Xinv.norm2());
    }
  }


  // Allocates and calculates the inverse KD block, returning Xinv
  std::shared_ptr<GaugeField> AllocateAndBuildStaggeredKahlerDiracInverse(const cudaGaugeField &gauge, const double mass, const bool dagger_approximation)
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

    std::shared_ptr<GaugeField> Xinv(reinterpret_cast<GaugeField*>(new cudaGaugeField(gParam)));

    BuildStaggeredKahlerDiracInverse(*Xinv, gauge, mass, dagger_approximation);

    return Xinv;
  }

} //namespace quda
