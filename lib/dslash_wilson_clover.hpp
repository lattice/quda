#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson_clover.cuh>

/**
   This is the Wilson-clover linear operator
*/

namespace quda
{

  template <typename Arg> class WilsonClover : public Dslash<wilsonClover, Arg>
  {
    using Dslash = Dslash<wilsonClover, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    WilsonClover(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.xpay)
        Dslash::template instantiate<packShmem, true>(tp, stream);
      else
        errorQuda("Wilson-clover operator only defined for xpay=true");
    }

    long long flops() const
    {
      int clover_flops = 504;
      long long flops = Dslash::flops();

      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: flops += clover_flops * in.Volume(); break;
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
      case KERNEL_POLICY: bytes += clover_bytes * in.Volume(); break;
      default: break;
      }

      return bytes;
    }
  };

  template <bool distance_pc> struct DistanceType {
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct WilsonCloverApply {

    template <bool distance_pc>
    inline WilsonCloverApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &A,
                             double a, double alpha0, int t0, const ColorSpinorField &x, int parity, bool dagger,
                             const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      WilsonCloverArg<Float, nColor, nDim, DDArg, recon, false, distance_pc> arg(out, in, U, A, a, 0.0, x, parity, dagger,
                                                                          comm_override, alpha0, t0);
      WilsonClover<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

} // namespace quda
