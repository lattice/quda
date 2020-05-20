#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson_clover.cuh>

/**
   This is the basic gauged twisted-clover operator
*/

namespace quda
{

  template <typename Arg> class TwistedClover : public Dslash<wilsonClover, Arg>
  {
    using Dslash = Dslash<wilsonClover, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    TwistedClover(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.xpay)
        this->template instantiate<packShmem, true>(tp, stream);
      else
        errorQuda("Twisted-clover operator only defined for xpay=true");
    }

    long long flops() const
    {
      int clover_flops = 504 + 48;
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: flops += clover_flops * in.Volume(); break;
      default: break; // all clover flops are in the interior kernel
      }
      return flops;
    }

    long long bytes() const
    {
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int clover_bytes = 72 * in.Precision() + (isFixed ? 2 * sizeof(float) : 0);

      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: bytes += clover_bytes * in.Volume(); break;
      default: break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct TwistedCloverApply {

    inline TwistedCloverApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                              const CloverField &C, double a, double b, const ColorSpinorField &x, int parity,
                              bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      WilsonCloverArg<Float, nColor, nDim, recon, true> arg(out, in, U, C, a, b, x, parity, dagger, comm_override);
      TwistedClover<decltype(arg)> twisted(arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(
        twisted, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
        in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the twisted-mass Dslash operator
  // out(x) = M*in = (A + i*b*gamma_5)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyTwistedClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &C,
                          double a, double b, const ColorSpinorField &x, int parity, bool dagger,
                          const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_TWISTED_CLOVER_DIRAC
    instantiate<TwistedCloverApply>(out, in, U, C, a, b, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Twisted-clover dslash has not been built");
#endif // GPU_TWISTED_CLOVEr_DIRAC
  }

} // namespace quda
