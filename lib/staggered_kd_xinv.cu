#include <tune_quda.h>
#include <transfer.h>
#include <gauge_field.h>
#include <blas_quda.h>
#include <blas_lapack.h>

#include <staggered_kd_xinv.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_kd_xinv_kernel.cuh>

namespace quda {

  template <typename Float, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredKDBlock : public TunableVectorYZ {

    Arg &arg;
    const GaugeField &meta;
    GaugeField &X;

    long long flops() const { 
      // only real work is adding mass
      return arg.coarseVolumeCB*coarseSpin*coarseColor;
    }

    long long bytes() const
    {
      // FIXME: this is from staggered_coarse_op which additionally builds Y. We only need X here.
      // 2 from forwards / backwards contributions, Y and X are sparse - only needs to write non-zero elements, 2nd term is mass term
      //return meta.Bytes() + (2 * meta.Bytes() * Y.Precision()) / meta.Precision() + 2 * 2 * coarseSpin * coarseColor * arg.coarseVolumeCB * X.Precision();
      return 0ll;
    }

    unsigned int minThreads() const { return arg.fineVolumeCB; }
    bool tuneSharedBytes() const { return false; } // FIXME don't tune the grid dimension
    bool tuneGridDim() const { return false; } // FIXME don't tune the grid dimension
    bool tuneAuxDim() const { return false; }

  public:
    CalculateStaggeredKDBlock(Arg &arg, const GaugeField &meta, GaugeField &X) :
      TunableVectorYZ(fineColor*fineColor, 2),
      arg(arg),
      meta(meta),
      X(X)
    {
#ifdef JITIFY
      create_jitify_program("kernels/staggered_kd_xinv_kernel.cuh");
#endif
      strcpy(aux, compile_type_str(meta));
      strcpy(aux, meta.AuxString());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
      strcat(aux,",computeStaggeredKDBlock");
      strcat(aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && X.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
             meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      strcat(aux,"coarse_vol=");
      strcat(aux,X.VolString());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("ComputeStaggeredKDBlock does not really support CPU execution yet");
        ComputeStaggeredKDBlockCPU<Float,fineColor,coarseSpin,coarseColor>(arg);
      } else {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::ComputeStaggeredKDBlockGPU")
          .instantiate(Type<Float>(),fineColor,coarseSpin,coarseColor,Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
        qudaLaunchKernel(ComputeStaggeredKDBlockGPU<Float,fineColor,coarseSpin,coarseColor,Arg>, tp, stream, arg);
#endif // JITIFY
      }
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (X.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
      else return false;
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
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
    typedef CalculateStaggeredKDBlockArg<Float,coarseSpin,fineColor,coarseColor,xGauge,fineGauge> Arg;
    Arg arg(X, G, mass, x_size, xc_size);
    CalculateStaggeredKDBlock<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, G_, X_);

    // We know exactly what the scale should be: the max of all of the (fat) links.
    double max_scale = G_.abs_max();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Global U_max = %e\n", max_scale);

    if (xGauge::fixedPoint()) {
      arg.X.resetScale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
      X_.Scale(max_scale > 2.0*mass ? max_scale : 2.0*mass); // To be safe
    }

    // We can technically do a uni-directional build, but becauase
    // the coarse link builds are just permutations plus lots of zeros,
    // it's faster to skip the flip!

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Computing KD Block\n");
    y.apply(0);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("X2 = %e\n", X_.norm2(0));
  }

  template <typename Float, typename vFloat, int fineColor, int coarseColor, int coarseSpin>
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {

    constexpr QudaGaugeFieldOrder gOrder = QUDA_FLOAT2_GAUGE_ORDER;

    if (g.FieldOrder() != gOrder) errorQuda("Unsupported field order %d\n", g.FieldOrder());

    typedef typename gauge::FieldOrder<Float,fineColor,1,gOrder,true,Float> gFine;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,vFloat> xCoarse;

    gFine gAccessor(const_cast<GaugeField&>(g));
    xCoarse xAccessor(const_cast<GaugeField&>(X));

    calculateStaggeredKDBlock<Float,fineColor,coarseSpin,coarseColor>(xAccessor, gAccessor, X, g, mass);

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
  void calculateStaggeredKDBlock(GaugeField &X, const GaugeField &g, const double mass)
  {
#if defined(GPU_STAGGERED_DIRAC)
    checkPrecision(X, g);

    // FIXME remove when done
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Computing X for StaggeredKD...\n");

#if QUDA_PRECISION & 8
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
#if QUDA_PRECISION & 1
    if (X.Precision() == QUDA_QUARTER_PRECISION) {
      calculateStaggeredKDBlock<float,char>(X, g, mass);
    } else
#endif
    {
      errorQuda("Unsupported precision %d\n", X.Precision());
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("....done computing X for StaggeredKD\n");
#else
    errorQuda("Staggered fermion support has not been built");
#endif
  }

  // Calculates the inverse KD block and puts the result in Xinv. Assumes Xinv has been allocated, in MILC data order
  void BuildStaggeredKahlerDiracInverse(cudaGaugeField &Xinv, const cudaGaugeField &gauge, const double mass)
  {
    using namespace blas_lapack;
    auto invert = use_native() ? native::BatchInvertMatrix : generic::BatchInvertMatrix;

    QudaPrecision precision = Xinv.Precision();
    QudaFieldLocation location = checkLocation(Xinv, gauge);

    if (location == QUDA_CUDA_FIELD_LOCATION && Xinv.FieldOrder() == QUDA_MILC_GAUGE_ORDER) {

      GaugeField *U = const_cast<cudaGaugeField*>(&gauge);

      // no reconstruct not strictly necessary, for now we do this for simplicity so
      // we can take advantage of fine-grained access like in "staggered_coarse_op.cu"
      if (gauge.Reconstruct() != QUDA_RECONSTRUCT_NO || gauge.Precision() != precision) {
        GaugeFieldParam gf_param(gauge);
        gf_param.reconstruct = QUDA_RECONSTRUCT_NO;
        gf_param.order = QUDA_FLOAT2_GAUGE_ORDER; // guaranteed for no recon
        gf_param.setPrecision(gf_param.Precision());
        U = new cudaGaugeField(gf_param);

        U->copy(gauge);
      }

      // Create X based on Xinv, remember Xinv is in QDP order
      GaugeFieldParam x_param(Xinv);
      x_param.order = QUDA_FLOAT2_GAUGE_ORDER;
      x_param.setPrecision(x_param.Precision());
      cudaGaugeField X(x_param);

      calculateStaggeredKDBlock(X, *U, mass);

      // FIXME: add support for double precision inverse
      // Reorder to MILC order for inversion, based on "coarse_op_preconditioned.cu"
      GaugeFieldParam param(Xinv);
      param.order = QUDA_MILC_GAUGE_ORDER;
      param.setPrecision( X.Precision() != QUDA_SINGLE_PRECISION ? QUDA_SINGLE_PRECISION : X.Precision());
      cudaGaugeField X_(param);
      cudaGaugeField* Xinv_ = ( Xinv.Precision() == QUDA_SINGLE_PRECISION) ? &Xinv : new cudaGaugeField(param);

      X_.copy(X);

      const int n = X.Ncolor();
      blas::flops += invert((void*)Xinv_->Gauge_p(), (void*)X_.Gauge_p(), n, X_.Volume(), X_.Precision(), X.Location());
      
      if ( Xinv_ != &Xinv) {

        if (Xinv.Precision() < QUDA_SINGLE_PRECISION) Xinv.Scale( Xinv_->abs_max() );
        Xinv.copy(*Xinv_);
        delete Xinv_;
      }

      if (U != &gauge) delete U;

    } else { 
      errorQuda("Unsupported field location %d", location);
    }


  }

} //namespace quda
