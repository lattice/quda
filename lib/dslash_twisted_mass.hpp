#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
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
    const GaugeField &U;

  public:
    TwistedMass(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                const ColorSpinorField &halo, const GaugeField U) :
      Dslash(arg, out, in, halo), U(U)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp, U);
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
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += in.size() * 2 * in.Ncolor() * 4 * 2 * in.Volume(); // complex * Nc * Ns * fma * vol
        break;
      default: break; // twisted-mass flops are in the interior kernel
      }
      return flops;
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct TwistedMassApply {
    TwistedMassApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                     cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double b, int parity,
                     bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      TwistedMassArg<Float, nColor, nDim, DDArg, recon> arg(out, in, halo, U, a, b, x, parity, dagger, comm_override);
      TwistedMass<decltype(arg)> twisted(arg, out, in, halo, U);
      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, out, in, halo, profile);
    }
  };

} // namespace quda
