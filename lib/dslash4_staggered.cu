#ifndef USE_LEGACY_DSLASH


#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_staggered.cuh>

/**
   This is a staggered Dirac operator
*/

namespace quda
{

  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct StaggeredLaunch {
    static constexpr const char *kernel = "quda::staggeredGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      dslash.launch(staggeredGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class Staggered : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    Staggered(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash<Float>(arg, out, in, "kernels/dslash_staggered.cuh"), arg(arg), in(in) {}

    virtual ~Staggered() {}

    void apply(const cudaStream_t &stream)
    {
      if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
        errorQuda("Staggered Dslash not implemented on CPU");
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        Dslash<Float>::setParam(arg);
        if (arg.xpay)
          Dslash<Float>::template instantiate<StaggeredLaunch, nDim, nColor, true>(tp, arg, stream);
        else
          Dslash<Float>::template instantiate<StaggeredLaunch, nDim, nColor, false>(tp, arg, stream);
      }
    }

    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }
  };

  template <typename Float, int nColor, QudaReconstructType recon_u, QudaReconstructType recon_l>
  void ApplyStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, const ColorSpinorField &x,
                      int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4; // MWTODO: this probably should be 5 for mrhs Dslash
    constexpr bool improved = false;
    StaggeredArg<Float, nColor, recon_u, recon_l, improved> arg(out, in, U, U, a, x, parity, dagger, comm_override);
    Staggered<Float, nDim, nColor, decltype(arg)> staggered(arg, out, in);

    dslash::DslashPolicyTune<decltype(staggered)> policy(staggered, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, const ColorSpinorField &x,
                      int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
      ApplyStaggered<Float, nColor, QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_NO>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct() == QUDA_RECONSTRUCT_13) {
      ApplyStaggered<Float, nColor, QUDA_RECONSTRUCT_13, QUDA_RECONSTRUCT_NO>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else if (U.Reconstruct() == QUDA_RECONSTRUCT_9) {
      // errorQuda("Recon 8 not implemented for standard staggered.\n");
      ApplyStaggered<Float, nColor, QUDA_RECONSTRUCT_9, QUDA_RECONSTRUCT_NO>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, const ColorSpinorField &x,
                      int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyStaggered<Float, 3>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  void ApplyStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, const ColorSpinorField &x,
      int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {

#ifdef GPU_STAGGERED_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder()) errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyStaggered<double>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyStaggered<float>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyStaggered<short>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
#else
    errorQuda("Staggered dslash has not been built");
#endif
  }

} // namespace quda

#endif
