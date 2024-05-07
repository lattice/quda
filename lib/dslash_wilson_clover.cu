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
    using Dslash::halo;
    const CloverField &A;

  public:
    WilsonClover(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                 const ColorSpinorField &halo, const CloverField &A) :
      Dslash(arg, out, in, halo), A(A)
    {
    }

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
      int n = (in.Nspin() / 2) * in.Ncolor();
      int mv_flops = 8 * n * n - 2 * n;
      long long flops = Dslash::flops();

      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: flops += 2 * mv_flops * halo.Volume(); break;
      default: break; // all clover flops are in the interior kernel
      }
      return flops;
    }

    long long bytes() const
    {
      long long bytes = Dslash::bytes();

      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: bytes += in.size() * A.Bytes(); break;
      default: break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonCloverApply {

    WilsonCloverApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                      const CloverField &A, double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                      const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonCloverArg<Float, nColor, nDim, recon> arg(out, in, halo, U, A, a, 0.0, x, parity, dagger, comm_override);
      WilsonClover<decltype(arg)> wilson(arg, out, in, halo, A);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonCloverWithTwistApply {

    WilsonCloverWithTwistApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                               const GaugeField &U, const CloverField &A, double a, double b,
                               cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                               const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonCloverArg<Float, nColor, nDim, recon, true> arg(out, in, halo, U, A, a, b, x, parity, dagger, comm_override);
      WilsonClover<decltype(arg)> wilson(arg, out, in, halo, A);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x)*in(x) + a * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         const GaugeField &U, const CloverField &A, double a, cvector_ref<const ColorSpinorField> &x,
                         int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_WILSON_DSLASH>())
      instantiate<WilsonCloverApply>(out, in, U, A, a, x, parity, dagger, comm_override, profile);
  }

} // namespace quda
