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

namespace quda
{
#include <dslash_events.cuh>
#include <dslash_policy.cuh>
} // namespace quda

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

    // TODO: fix flop / byte count?

    /*
        long long flops() const
        {
          return (2*nDim*(8*nColor*nColor)-2*nColor + (arg.xpay ? 2*2*nColor : 0) )*arg.nParity*(long long)in.VolumeCB();
        }
        long long bytes() const
        {
          return arg.out.Bytes() + 2*nDim*arg.in.Bytes() + arg.nParity*2*nDim*arg.U.Bytes()*in.VolumeCB() +
      (arg.xpay ? arg.x.Bytes() : 0);
        }

        bool tuneGridDim() const { return false; }
        unsigned int minThreads() const { return arg.volumeCB; }
    */
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

  template <typename Float, int nColor, QudaReconstructType recon_u, QudaReconstructType recon_l, bool improved>
  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, double a, const ColorSpinorField &x,
      int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    constexpr int nDim = 4; // MWTODO: this probably should be 5 for mrhs Dslash
    StaggeredArg<Float, nColor, recon_u, recon_l, improved> arg(out, in, U, L, a, x, parity, dagger, comm_override);
    Staggered<Float, nDim, nColor, decltype(arg)> staggered(arg, out, in);

    DslashPolicyTune<decltype(staggered)> policy(
        staggered, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(), in.GhostFaceCB(), profile);
    policy.apply(0);

    checkCudaError();
  }

  // template on the gauge reconstruction
  template <typename Float, int nColor>
  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, double a, const ColorSpinorField &x,
      int parity, bool dagger, bool improved, const int *comm_override, TimeProfile &profile)
  {
    if (improved) {
      if (L.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        ApplyDslashStaggered<Float, nColor, QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_NO, true>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
      } else if (L.Reconstruct() == QUDA_RECONSTRUCT_13) {
        ApplyDslashStaggered<Float, nColor, QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_13, true>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
      } else if (L.Reconstruct() == QUDA_RECONSTRUCT_9) {
        ApplyDslashStaggered<Float, nColor, QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_9, true>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
      } else {
        errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
      }
    } else {
      if (U.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        ApplyDslashStaggered<Float, nColor, QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_NO, false>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
      } else if (U.Reconstruct() == QUDA_RECONSTRUCT_13) {
        ApplyDslashStaggered<Float, nColor, QUDA_RECONSTRUCT_13, QUDA_RECONSTRUCT_NO, false>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
      } else if (U.Reconstruct() == QUDA_RECONSTRUCT_9) {
        // errorQuda("Recon 8 not implemented for standard staggered.\n");
        ApplyDslashStaggered<Float, nColor, QUDA_RECONSTRUCT_9, QUDA_RECONSTRUCT_NO, false>(out, in, U, L, a, x, parity, dagger, comm_override, profile);
      } else {
        errorQuda("Unsupported reconstruct type %d\n", U.Reconstruct());
      }
    }
  }

  // template on the number of colors
  template <typename Float>
  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, double a, const ColorSpinorField &x,
      int parity, bool dagger, bool improved, const int *comm_override, TimeProfile &profile)
  {
    if (in.Ncolor() == 3) {
      ApplyDslashStaggered<Float, 3>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } else {
      errorQuda("Unsupported number of colors %d\n", U.Ncolor());
    }
  }

  void ApplyDslashStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const GaugeField &L, double a, const ColorSpinorField &x,
      int parity, bool dagger, bool improved, const int *comm_override, TimeProfile &profile)
  {

#ifdef GPU_STAGGERED_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder()) errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U, L);

    // check all locations match
    checkLocation(out, in, U, L);

    if (dslash::aux_worker) dslash::aux_worker->apply(0);
    if (U.Precision() == QUDA_DOUBLE_PRECISION) {
      ApplyDslashStaggered<double>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } else if (U.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyDslashStaggered<float>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } else if (U.Precision() == QUDA_HALF_PRECISION) {
      ApplyDslashStaggered<short>(out, in, U, L, a, x, parity, dagger, improved, comm_override, profile);
    } else {
      errorQuda("Unsupported precision %d\n", U.Precision());
    }
#else
    errorQuda("Staggered dslash has not been built");
#endif
  }

} // namespace quda

#endif
