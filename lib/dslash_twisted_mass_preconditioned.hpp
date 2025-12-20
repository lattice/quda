#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_twisted_mass_preconditioned.cuh>

/**
   This is the preconditioned gauged twisted-mass operator
*/

namespace quda
{

  // trait to ensure we don't instantiate asymmetric & xpay
  template <bool symmetric> constexpr bool xpay_() { return true; }
  template <> constexpr bool xpay_<true>() { return false; }

  // trait to ensure we don't instantiate asymmetric & !dagger
  template <bool symmetric> constexpr bool not_dagger_() { return false; }
  template <> constexpr bool not_dagger_<true>() { return true; }

  template <typename Arg> class TwistedMassPreconditioned : public Dslash<twistedMassPreconditioned, Arg>
  {
    using Dslash = Dslash<twistedMassPreconditioned, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    TwistedMassPreconditioned(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                              const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.asymmetric && !arg.dagger) errorQuda("asymmetric operator only defined for dagger");
      if (arg.asymmetric && arg.xpay) errorQuda("asymmetric operator not defined for xpay");
      if (arg.nParity != 1) errorQuda("Preconditioned twisted-mass operator not defined nParity=%d", arg.nParity);

      if (arg.dagger) {
        if (arg.xpay)
          Dslash::template instantiate<packShmem, true, xpay_<Arg::asymmetric>()>(tp, stream);
        else
          Dslash::template instantiate<packShmem, true, false>(tp, stream);
      } else {
        if (arg.xpay)
          Dslash::template instantiate<packShmem, not_dagger_<Arg::asymmetric>(), xpay_<Arg::asymmetric>()>(tp, stream);
        else
          Dslash::template instantiate<packShmem, not_dagger_<Arg::asymmetric>(), false>(tp, stream);
      }
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
      default: break;
      }
      return flops;
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct TwistedMassPreconditionedApply {
    TwistedMassPreconditionedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                   cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double b,
                                   bool xpay, int parity, bool dagger, bool asymmetric, const int *comm_override,
                                   TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      if (asymmetric) {
        TwistedMassArg<Float, nColor, nDim, DDArg, recon, true> arg(out, in, halo, U, a, b, xpay, x, parity, dagger,
                                                                    comm_override);
        TwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in, halo);

        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, out, in, halo, profile);
      } else {
        TwistedMassArg<Float, nColor, nDim, DDArg, recon, false> arg(out, in, halo, U, a, b, xpay, x, parity, dagger,
                                                                     comm_override);
        TwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in, halo);

        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, out, in, halo, profile);
      }
    }
  };

} // namespace quda
