#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson_clover_preconditioned.cuh>

/**
   This is the Wilson-clover preconditioned linear operator
*/

namespace quda
{

  template <typename Arg> class WilsonCloverPreconditioned : public Dslash<wilsonCloverPreconditioned, Arg>
  {
    using Dslash = Dslash<wilsonCloverPreconditioned, Arg>;
    using Dslash::arg;
    using Dslash::in;
    using Dslash::halo;
    const CloverField &A;

  public:
    WilsonCloverPreconditioned(Arg &arg, cvector_ref<ColorSpinorField> &out,
                               cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
                               const CloverField &A) :
      Dslash(arg, out, in, halo), A(A)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      // specialize here to constrain the template instantiation
      if (arg.nParity == 1) {
        if (arg.xpay) {
          if (arg.dagger) errorQuda("xpay operator only defined for not dagger");
          Dslash::template instantiate<packShmem, 1, false, true>(tp, stream);
        } else {
          if (arg.dagger)
            Dslash::template instantiate<packShmem, 1, true, false>(tp, stream);
          else
            Dslash::template instantiate<packShmem, 1, false, false>(tp, stream);
        }
      } else {
        errorQuda("Preconditioned Wilson-clover operator not defined nParity=%d", arg.nParity);
      }
    }

    long long flops() const
    {
      int n = (in.Nspin() / 2) * in.Ncolor();
      int mv_flops = 8 * n * n - 2 * n;
      int clover_flops = 2 * mv_flops; // 1 m-v product per chiral block
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: flops += clover_flops * 2 * halo.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL:
        flops
          += clover_flops * 2 * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += clover_flops * in.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * halo.GhostFace()[d];
        flops -= clover_flops * ghost_sites;

        break;
      }
      return flops;
    }

    long long bytes() const
    {
      int clover_bytes = A.Bytes() / A.Volume();
      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: bytes += clover_bytes * 2 * halo.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL:
        bytes
          += clover_bytes * 2 * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        bytes += clover_bytes * in.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in[0].GhostFace()[d];
        bytes -= clover_bytes * ghost_sites;

        break;
      }

      return bytes;
    }

  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonCloverPreconditionedApply {

    WilsonCloverPreconditionedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                    const GaugeField &U, const CloverField &A, double a,
                                    cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                    const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonCloverArg<Float, nColor, nDim, recon> arg(out, in, halo, U, A, a, x, parity, dagger, comm_override);
      WilsonCloverPreconditioned<decltype(arg)> wilson(arg, out, in, halo, A);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

  // Apply the preconditioned Wilson-clover operator
  // out(x) = M*in = a * A(x)^{-1} (\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                       const GaugeField &U, const CloverField &A, double a,
                                       cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                       const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_WILSON_DSLASH>())
      instantiate<WilsonCloverPreconditionedApply>(out, in, U, A, a, x, parity, dagger, comm_override, profile);
  }

} // namespace quda
