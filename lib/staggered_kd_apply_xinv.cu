#include <gauge_field.h>
#include <color_spinor_field.h>
#include <register_traits.h>
#include <dslash_quda.h>
#include <instantiate.h>

#include <tunable_nd.h>
#include <kernels/staggered_kd_apply_xinv_kernel.cuh>

namespace quda {

  template <typename Arg>
  class ApplyStaggeredKDBlock : public TunableKernel1D {

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
      return 2 * Arg::fineColor * 16 * Arg::paddedSpinorSizeKD * sizeof(complex<typename Arg::real>) / 256 +
              Arg::xinvPaddedColTileSize * sizeof(complex<typename Arg::real>);
    }

    int blockStep() const { return 256; }
    int blockMin() const { return 256; }

    unsigned int minThreads() const { return 2 * arg.fineVolumeCB; }

  public:
    ApplyStaggeredKDBlock(Arg &arg, const ColorSpinorField &meta, const GaugeField &Xinv) :
      TunableKernel1D(meta),
      arg(arg),
      meta(meta),
      Xinv(Xinv)
    {
      strcat(aux, ",Xinv:coarse_");
      strcat(aux, Xinv.AuxString());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Not implemented");
      } else {
        launch_device<StaggeredKDBlock>(tp, stream, arg);
      }
    }
  };

  /**
     @brief Apply the staggered Kahler-Dirac block inverse

     @param out[out] output staggered spinor accessor
     @param in[in] input staggered spinor accessor
     @param Xinv[in] KD block inverse accessor
     @param out_[out] output staggered spinor
     @param in_[in] input staggered spinor (unused)
     @param Xinv_[in] KD block inverse
  */
  template<typename vFloatSpinor, typename vFloatGauge, int fineColor, int coarseDof, bool dagger, typename fineColorSpinor, typename xInvGauge>
  void applyStaggeredKDBlock(fineColorSpinor &out, const fineColorSpinor &in, const xInvGauge &Xinv,
                             ColorSpinorField &out_, const ColorSpinorField &, const GaugeField &Xinv_)
  {
    // sanity checks
    if (fineColor != 3)
      errorQuda("Input gauge field should have nColor=3, not nColor=%d\n", fineColor);

    if (Xinv.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    if (fineColor * 16 != coarseDof)
      errorQuda("Fine nColor=%d is not consistent with KD dof %d", fineColor, coarseDof);

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
    
    using Arg = ApplyStaggeredKDBlockArg<vFloatSpinor,vFloatGauge,coarseDof,fineColor,dagger,fineColorSpinor,xInvGauge>;
    Arg arg(out, in, Xinv, x_size, xc_size);

    ApplyStaggeredKDBlock<Arg> y(arg, out_, Xinv_);

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("Applying KD block...\n");
    y.apply(device::get_default_stream());

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("... done applying KD block\n");
  }

  // create accessors, specify dagger vs non-dagger
  template <typename vFloatSpinor, typename vFloatGauge, int fineColor, int fineSpin, int coarseDof>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    // Create the accessor for Xinv
    constexpr QudaGaugeFieldOrder xOrder = QUDA_MILC_GAUGE_ORDER;
    if (Xinv.FieldOrder() != xOrder) errorQuda("Unsupported field order %d\n", Xinv.FieldOrder());
    using xInvCoarse = typename gauge::FieldOrder<typename mapper<vFloatGauge>::type,coarseDof,1,xOrder,true,vFloatGauge>;
    xInvCoarse xInvAccessor(const_cast<GaugeField &>(Xinv));

    // Create the accessors for out, in
    constexpr bool spin_project = false;
    constexpr bool spinor_direct_load = false; // seems legacy? false means texture load
    using csFine = typename colorspinor_mapper<vFloatSpinor, fineSpin, fineColor, spin_project, spinor_direct_load>::type;
    const csFine inAccessor(in);
    csFine outAccessor(out);

    if (dagger) applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, fineColor, coarseDof, true>(outAccessor, inAccessor, xInvAccessor, out, in, Xinv);
    else applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, fineColor, coarseDof, false>(outAccessor, inAccessor, xInvAccessor, out, in, Xinv);
  }

  // template on coarse color, spin
  template <typename vFloatSpinor, typename vFloatGauge, int fineColor, int fineSpin>
  void applyStaggeredKDBlock(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    //constexpr int coarseSpin = 2;
    const int coarseDof = Xinv.Ncolor(); // / coarseSpin;

    if (coarseDof == 48) { // dof w/in a KD block
      applyStaggeredKDBlock<vFloatSpinor, vFloatGauge, fineColor, fineSpin, 48>(out, in, Xinv, dagger);
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
  template <typename vFloatSpinor> struct StaggeredKDBlockApply {
    StaggeredKDBlockApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
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
  };

#if defined(GPU_STAGGERED_DIRAC) && defined(GPU_MULTIGRID)
  // Applies the staggered KD block inverse to a staggered ColorSpinor
  void ApplyStaggeredKahlerDiracInverse(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &Xinv, bool dagger)
  {
    auto location = checkLocation(out, in, Xinv);

    if (location == QUDA_CPU_FIELD_LOCATION)
      errorQuda("There is no support for applying the KD operator to CPU fields (yet)");

    // the staggered KD block inverse can only be applied to a full field
    if (out.SiteSubset() != QUDA_FULL_SITE_SUBSET || out.SiteSubset() != QUDA_FULL_SITE_SUBSET)
      errorQuda("There is no meaning to applying the KD inverse to a single parity field");
    
    checkPrecision(out, in);

    // Instantiate based on ColorSpinor precision
    // We don't have a constraint on the precision of Xinv matching
    // the precision of the spinors.
    instantiatePrecision<StaggeredKDBlockApply>(out, in, Xinv, dagger);
  }
#else
  // Applies the staggered KD block inverse to a staggered ColorSpinor
  void ApplyStaggeredKahlerDiracInverse(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, bool)
  {
    errorQuda("Staggered fermion multigrid support has not been built");
  }
#endif

} //namespace quda
