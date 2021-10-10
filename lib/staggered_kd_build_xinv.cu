#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>
#include <tunable_nd.h>
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

  // Calculates the inverse KD block and puts the result in Xinv. Assumes Xinv has been allocated, in MILC data order
  void BuildStaggeredKahlerDiracInverse(GaugeField &Xinv, const cudaGaugeField &gauge, const double mass)
  {
    using namespace blas_lapack;
    auto invert = use_native() ? native::BatchInvertMatrix : generic::BatchInvertMatrix;

    // Xinv should have a MILC gauge order independent of being
    // on the CPU or GPU
    if (Xinv.FieldOrder() != QUDA_MILC_GAUGE_ORDER)
      errorQuda("Unsupported field order %d", Xinv.FieldOrder());

    QudaPrecision precision = Xinv.Precision();
    QudaFieldLocation location = Xinv.Location();

    // Logic copied from `staggered_coarse_op.cu`
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

    } else if (location == QUDA_CUDA_FIELD_LOCATION) {

      // no reconstruct not strictly necessary, for now we do this for simplicity so
      // we can take advantage of fine-grained access like in "staggered_coarse_op.cu"
      // technically don't need to require the precision check, but it should
      // generally be equal anyway

      // FIXME: make this work for any gauge precision
      if (gauge.Reconstruct() != QUDA_RECONSTRUCT_NO || gauge.Precision() != precision || gauge.Precision() < QUDA_SINGLE_PRECISION) {
        GaugeFieldParam gf_param(gauge);
        gf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        gf_param.order = QUDA_FLOAT2_GAUGE_ORDER; // guaranteed for no recon
        gf_param.setPrecision( precision == QUDA_DOUBLE_PRECISION ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION );
        U = new cudaGaugeField(gf_param);

        U->copy(gauge);
      }
    }

    // Create X based on Xinv, but switch to a native ordering
    // for a GPU setup.
    GaugeField *X = nullptr;
    GaugeFieldParam x_param(Xinv);
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      x_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      x_param.setPrecision(x_param.Precision());
      X = static_cast<GaugeField*>(new cudaGaugeField(x_param));
    } else {
      X = static_cast<GaugeField*>(new cpuGaugeField(x_param));
    }

    // Calculate X
    calculateStaggeredKDBlock(*X, *U, mass);

    // Invert X
    // Logic copied from `coarse_op_preconditioned.cu`
    const int n = Xinv.Ncolor();
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      // FIXME: add support for double precision inverse
      // Reorder to MILC order for inversion, based on "coarse_op_preconditioned.cu"
      GaugeFieldParam param(Xinv);
      param.order = QUDA_MILC_GAUGE_ORDER; // MILC order == QDP order for Xinv
      param.setPrecision(QUDA_SINGLE_PRECISION); // FIXME until double prec support is added
      cudaGaugeField X_(param);
      cudaGaugeField* Xinv_ = ( Xinv.Precision() == QUDA_SINGLE_PRECISION) ? static_cast<cudaGaugeField*>(&Xinv) : new cudaGaugeField(param);

      X_.copy(*X);

      blas::flops += invert((void*)Xinv_->Gauge_p(), (void*)X_.Gauge_p(), n, X_.Volume(), X_.Precision(), X->Location());
      
      if ( Xinv_ != &Xinv) {
        if (Xinv.Precision() < QUDA_SINGLE_PRECISION) Xinv.Scale( Xinv_->abs_max() );
        Xinv.copy(*Xinv_);
        delete Xinv_;
      }

    } else if (location == QUDA_CPU_FIELD_LOCATION) {

      if (Xinv.Precision() == QUDA_DOUBLE_PRECISION)
        errorQuda("Unsupported precision %d", Xinv.Precision());

      const cpuGaugeField *X_h = static_cast<const cpuGaugeField*>(X);
      cpuGaugeField *Xinv_h = static_cast<cpuGaugeField*>(&Xinv);
      blas::flops += invert((void*)Xinv_h->Gauge_p(), (void*)X_h->Gauge_p(), n, X_h->Volume(), X->Precision(), X->Location());

    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Xinv = %e\n", Xinv.norm2(0));

    // Clean up
    delete X;
    if (U != &gauge) delete U;

  }


  // Allocates and calculates the inverse KD block, returning Xinv
  cudaGaugeField* AllocateAndBuildStaggeredKahlerDiracInverse(const cudaGaugeField &gauge, const double mass, const QudaPrecision override_prec)
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
    gParam.setPrecision( override_prec );
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    gParam.pad = 0;

    cudaGaugeField* Xinv = new cudaGaugeField(gParam);

    BuildStaggeredKahlerDiracInverse(*Xinv, gauge, mass);

    return Xinv;
 }

} //namespace quda
