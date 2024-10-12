#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
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
    using Dslash::halo;
    using Dslash::in;

  public:
    TwistedClover(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
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
        errorQuda("Twisted-clover operator only defined for xpay=true");
    }

    long long flops() const
    {
      int clover_flops = 504 + 48;
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: flops += clover_flops * halo.Volume(); break;
      default: break; // all clover flops are in the interior kernel
      }
      return flops;
    }

    long long bytes() const
    {
      int clover_bytes = 72 * in.Precision() + (isFixed<typename Arg::Float>::value ? 2 * sizeof(float) : 0);

      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: bytes += clover_bytes * halo.Volume(); break;
      default: break;
      }

      return bytes;
    }
  };

  template <bool distance_pc> struct DistanceType {
  };


  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct TwistedCloverApply {
    template <bool distance_pc>
    TwistedCloverApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                       cvector_ref<const ColorSpinorField> &x, const GaugeField &U, const CloverField &C, double a,
                       double b, int parity, bool dagger, const int *comm_override, DistanceType<distance_pc>,TimeProfile &profile)
#ifdef SIGNATURE_ONLY
    ;
#else
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonCloverArg<Float, nColor, nDim, DDArg, recon, true,distance_pc> arg(out, in, halo, U, C, a, b, x, parity, dagger,
                                                                   comm_override);
      TwistedClover<decltype(arg)> twisted(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, in, halo, profile);
    }
#endif
  };

} // namespace quda
