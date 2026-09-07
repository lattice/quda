#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_wilson_clover_hasenbusch_twist.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  template <typename Arg> class WilsonCloverHasenbuschTwist : public Dslash<cloverHasenbusch, Arg>
  {
    using Dslash = Dslash<cloverHasenbusch, Arg>;
    using Dslash::arg;
    using Dslash::in;
    const GaugeField &U;

  public:
    WilsonCloverHasenbuschTwist(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
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
        errorQuda("Wilson-clover - Hasenbusch Twist operator only defined for xpay=true");
    }

    long long flops() const
    {
      int clover_flops = in.size() * 504;
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += clover_flops * in.Volume();

        // -mu * (i gamma_5 A) (A x)
        flops += ((clover_flops + 48) * in.Volume());

        break;
      default: break; // all clover flops are in the interior kernel
      }
      return flops;
    }

    long long bytes() const
    {
      int clover_bytes = in.size() * 72 * in.Precision() + (isFixed<typename Arg::Float>::value ? 2 * sizeof(float) : 0);

      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: bytes += clover_bytes * in.Volume(); break;
      default: break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct WilsonCloverHasenbuschTwistApply {
    WilsonCloverHasenbuschTwistApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                     cvector_ref<const ColorSpinorField> &x, const GaugeField &U, const CloverField &A,
                                     real_t a, real_t b, int parity, bool dagger, const int *comm_override,
                                     TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonCloverHasenbuschTwistArg<Float, nColor, nDim, DDArg, recon> arg(out, in, halo, U, A, a, b, x, parity,
                                                                            dagger, comm_override);
      WilsonCloverHasenbuschTwist<decltype(arg)> wilson(arg, out, in, halo, U);
      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, out, in, halo, profile);
    }
  };

} // namespace quda
