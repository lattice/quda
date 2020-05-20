#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_twisted_mass.cuh>

/**
   This is the basic gauged twisted-mass operator
*/

namespace quda
{

  template <typename Arg> class TwistedMass : public Dslash<twistedMass, Arg>
  {
    using Dslash = Dslash<twistedMass, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    TwistedMass(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.xpay)
        Dslash::template instantiate<packShmem, true>(tp, stream);
      else
        errorQuda("Twisted-mass operator only defined for xpay=true");
    }

    long long flops() const
    {
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        flops += 2 * in.Ncolor() * 4 * 2 * in.Volume(); // complex * Nc * Ns * fma * vol
        break;
      default: break; // twisted-mass flops are in the interior kernel
      }
      return flops;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct TwistedMassApply {

    inline TwistedMassApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
                            const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                            TimeProfile &profile)
    {
      constexpr int nDim = 4;
      TwistedMassArg<Float, nColor, nDim, recon> arg(out, in, U, a, b, x, parity, dagger, comm_override);
      TwistedMass<decltype(arg)> twisted(arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(
        twisted, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
        in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the twisted-mass Dslash operator
  // out(x) = M*in = (1 + i*b*gamma_5)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyTwistedMass(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double b,
                        const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                        TimeProfile &profile)
  {
#ifdef GPU_TWISTED_MASS_DIRAC
    instantiate<TwistedMassApply>(out, in, U, a, b, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Twisted-mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

} // namespace quda
