#include <tune_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <register_traits.h>

#include <jitify_helper.cuh>
#include <kernels/staggered_kd_apply_xinv_kernel.cuh>

namespace quda {

  template <typename Arg>
  class ApplyStaggeredKDBlock : public TunableVectorY {

    Arg &arg;
    const ColorSpinorField &meta;
    const GaugeField &Xinv;

    long long flops() const { 
      // a coarse volume number of 48x48 mat-vec
      return 2ll * arg.coarseVolumeCB * Arg::coarseDof * (8ll * Arg::coarseDof - 2);
    }

    long long bytes() const
    {
      return 2 * meta.Bytes() + Xinv.Bytes();
    }

    unsigned int sharedBytesPerThread() const {
      // 16 threads needs to store 16 ColorVectors in the compute
      // precision, times 2 for in vs out
      // -> each thread needs to store 2 x size of ColorVector
      // plus some padding to avoid bank conflicts: each KD block stores
      // 17 ColorVectors. (2 * 16 threads * 51 complex * 8 bytes per complex) / 256 threads
      // -> each thread needs 51 bytes
      return 2 * ((Arg::fineColor * 16 * Arg::paddedSizeKD * sizeof(complex<typename Arg::Float>)) / 256 + 1);
    }

    int blockStep() const { return 256; }
    int blockMin() const { return 256; }

    unsigned int minThreads() const { return arg.fineVolumeCB; }
    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    ApplyStaggeredKDBlock(Arg &arg, const ColorSpinorField &meta, const GaugeField &Xinv) :
      TunableVectorY(8),
      arg(arg),
      meta(meta),
      Xinv(Xinv)
    {
#ifdef JITIFY
      create_jitify_program("kernels/staggered_kd_apply_xinv_kernel.cuh");
#endif
      strcpy(aux, compile_type_str(meta));
      // should be all we need?
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        //ComputeStaggeredKDBlockCPU<Float,fineColor,coarseSpin,coarseColor>(arg);
      } else {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::ApplyStaggeredKDBlockGPU")
          .instantiate(Type<Arg>())
          .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else // not jitify
        qudaLaunchKernel(ApplyStaggeredKDBlockGPU<Arg>, tp, stream, arg);
#endif // JITIFY
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  /**
     @brief Apply the staggered Kahler-Dirac block inverse

     @param out[out] output staggered spinor accessor
     @param in[in] input staggered spinor accessor
     @param Xinv[in] KD block inverse accessor
     @param out_[out] output staggered spinor
     @param in_[in] input staggered spinor
     @param Xinv_[in] KD block inverse
  */
  template<typename vFloatSpinor, typename vFloatGauge, int fineColor, int coarseSpin, int coarseColor, bool dagger, typename fineColorSpinor, typename xInvGauge>
  void applyStaggeredKDBlock(fineColorSpinor &out, const fineColorSpinor &in, const xInvGauge &Xinv,
        ColorSpinorField &out_, const ColorSpinorField &in_, const GaugeField &Xinv_)
  {
    // sanity checks
    if (fineColor != 3)
      errorQuda("Input gauge field should have nColor=3, not nColor=%d\n", fineColor);

    if (Xinv.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    if (fineColor * 16 != coarseColor*coarseSpin)
      errorQuda("Fine nColor=%d is not consistent with KD dof %d", fineColor, coarseColor*coarseSpin);

    int x_size[QUDA_MAX_DIM] = { };
    int xc_size[QUDA_MAX_DIM] = { };
    for (int i = 0; i < nDim; i++) {
      x_size[i] = out_.X()[i];
      xc_size[i] = Xinv_.X()[i];
      // check that local volumes are consistent
      if (2 * xc_size[i] != x_size[i]) {
        errorQuda("Inconsistent fine dimension %d and coarse KD dimension %d", x_size[i], xc_size[i]);
      }
    }
    
    typedef ApplyStaggeredKDBlockArg<vFloatSpinor,vFloatGauge,coarseSpin,fineColor,coarseColor,dagger,fineColorSpinor,xInvGauge> Arg;
    Arg arg(out, in, Xinv, x_size, xc_size);

    ApplyStaggeredKDBlock<Arg> y(arg, out_, Xinv_);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Applying KD block\n");
    y.apply(0);
  }

  // create accessors, specify dagger vs non-dagger
  template <typename vFloatSpinor, typename vFloatGauge, int fineColor, int fineSpin, int coarseColor, int coarseSpin>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {

    // Create the accessor for Xinv
    constexpr QudaGaugeFieldOrder xOrder = QUDA_MILC_GAUGE_ORDER;
    if (Xinv.FieldOrder() != xOrder) errorQuda("Unsupported field order %d\n", Xinv.FieldOrder());
    typedef typename gauge::FieldOrder<typename mapper<vFloatGauge>::type,coarseColor*coarseSpin,coarseSpin,xOrder,true,vFloatGauge> xInvCoarse;
    xInvCoarse xInvAccessor(const_cast<GaugeField &>(Xinv));

    // Create the accessors for out, in
    constexpr bool spin_project = false;
    constexpr bool spinor_direct_load = false; // seems legacy? false means texture load
    typedef typename colorspinor_mapper<vFloatSpinor, fineSpin, fineColor, spin_project, spinor_direct_load>::type csFine;
    const csFine inAccessor(in);
    csFine outAccessor(out);

    if (dagger) applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, fineColor, coarseSpin, coarseColor, true>(outAccessor, inAccessor, xInvAccessor, out, in, Xinv);
    else applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, fineColor, coarseSpin, coarseColor, false>(outAccessor, inAccessor, xInvAccessor, out, in, Xinv);
  }

  // template on coarse color, spin
  template <typename vFloatSpinor, typename vFloatGauge, int fineColor, int fineSpin>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    constexpr int coarseSpin = 2;
    const int coarseColor = Xinv.Ncolor() / coarseSpin;

    if (coarseColor == 24) { // half the dof w/in a KD block
      applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, fineColor, fineSpin, 24, coarseSpin>(out, in, Xinv, dagger);
    } else {
      errorQuda("Unsupported number of Kahler-Dirac dof %d\n", Xinv.Ncolor());
    }
  }

  // template on fine colors, spin
  template <typename vFloatSpinor, typename vFloatGauge>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    if (out.Ncolor() != in.Ncolor()) 
      errorQuda("Ncolor %d and %d do not match", out.Ncolor(), in.Ncolor());

    if (out.Nspin() != in.Nspin())
      errorQuda("Nspin %d and %d do not match", out.Nspin(), in.Nspin());

    if (out.Ncolor() == 3 && out.Nspin() == 1) {
      applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, 3, 1>(out, in, Xinv, dagger);
    } else {
      errorQuda("Unsupported (color, spin) = (%d, %d)", out.Ncolor(), out.Nspin());
    }
  }

  // template on Xinv precision (only half and single for now)
  template <typename vFloatSpinor>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {

#if QUDA_PRECISION & 4
    if (Xinv.Precision() == QUDA_SINGLE_PRECISION) {
      applyStaggeredKDBlock<vFloatSpinor, float>(out, in, Xinv, dagger);
    } else
#endif
#if QUDA_PRECISION & 2
    if (Xinv.Precision() == QUDA_HALF_PRECISION) {
      applyStaggeredKDBlock<vFloatSpinor, short>(out, in, Xinv, dagger);
    } else
#endif
    {
      errorQuda("Unsupported precision %d", Xinv.Precision());
    }
  }

  // Applies the staggered KD block inverse to a staggered ColorSpinor
  void ApplyStaggeredKahlerDiracInverse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
#if defined(GPU_STAGGERED_DIRAC)
    // FIXME: This will eventually get replaced by a custom instantiate
    // I just don't want to think about it for now.
    auto location = checkLocation(out, in, Xinv);

    if (location == QUDA_CPU_FIELD_LOCATION)
      errorQuda("There is no support for applying the KD operator to CPU fields (yet)");
    
    // FIXME For first pass let's not think about the dagger op...
    if (dagger)
      errorQuda("There is no support for applying the KD inverse dagger (yet)");

    // the staggered KD block inverse can only be applied to a full field
    if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() != QUDA_FULL_SITE_SUBSET)
      errorQuda("There is no meaning to applying the KD inverse to a single parity field");

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Applying StaggeredKD...\n");

    // Instantiate based on ColorSpinor precision
    // We don't have a constraint on the precision of Xinv matching
    // the precision of the spinors.
    auto precision = checkPrecision(out, in);

#if QUDA_PRECISION & 8
    if (precision == QUDA_DOUBLE_PRECISION) {
      applyStaggeredKDBlock<double>(out, in, Xinv, dagger);
    } else
#endif
#if QUDA_PRECISION & 4
    if (precision == QUDA_SINGLE_PRECISION) {
      applyStaggeredKDBlock<float>(out, in, Xinv, dagger);
    } else
#endif
#if QUDA_PRECISION & 2
    if (precision == QUDA_HALF_PRECISION) {
      applyStaggeredKDBlock<short>(out, in, Xinv, dagger);
    } else
#endif
#if QUDA_PRECISION & 1
    if (precision == QUDA_QUARTER_PRECISION) {
      applyStaggeredKDBlock<int8_t>(out, in, Xinv, dagger);
    } else
#endif
    {
      errorQuda("Unsupported precision %d\n", precision);
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("...done applying StaggeredKD\n");

#else
    errorQuda("Staggered fermion support has not been built");
#endif
  }

} //namespace quda
