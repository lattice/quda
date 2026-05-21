#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_ndeg_twisted_mass.cuh>

/**
   This is the gauged twisted-mass operator acting on a non-generate
   quark doublet.
*/

namespace quda
{

  template <typename Arg> class NdegTwistedMass : public Dslash<nDegTwistedMass, Arg>
  {
    using Dslash = Dslash<nDegTwistedMass, Arg>;
    using Dslash::arg;
    using Dslash::in;
    const GaugeField &U;

  public:
    NdegTwistedMass(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                    const ColorSpinorField &halo, const GaugeField &U) :
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
        errorQuda("Non-degenerate twisted-mass operator only defined for xpay=true");
    }

    long long flops() const
    {
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += in.size() * 2 * in.Ncolor() * 4 * 4 * in.Volume(); // complex * Nc * Ns * fma * vol
        break;
      default: break; // twisted-mass flops are in the interior kernel
      }
      return flops;
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct NdegTwistedMassApply {
    NdegTwistedMassApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         cvector_ref<const ColorSpinorField> &x, const GaugeField &U, real_t a, real_t b, real_t c,
                         int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      NdegTwistedMassArg<Float, nColor, nDim, DDArg, recon> arg(out, in, halo, U, a, b, c, x, parity, dagger,
                                                                comm_override);
      NdegTwistedMass<decltype(arg)> twisted(arg, out, in, halo, U);
      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, out, in, halo, profile);
    }
  };

} // namespace quda
