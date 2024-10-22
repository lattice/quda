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

  public:
    NdegTwistedMass(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                    const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
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

  template <bool distance_pc> struct DistanceType {
  };


  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct NdegTwistedMassApply {
    template <bool distance_pc>
    NdegTwistedMassApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double b, double c,
                         int parity, bool dagger, const int *comm_override,DistanceType<distance_pc>, TimeProfile &profile)
#ifdef SIGNATURE_ONLY
    ;
#else
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      NdegTwistedMassArg<Float, nColor, nDim, DDArg, recon> arg(out, in, halo, U, a, b, c, x, parity, dagger,
                                                                comm_override);
      NdegTwistedMass<decltype(arg)> twisted(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, in, halo, profile);
    }
#endif
  };
} // namespace quda