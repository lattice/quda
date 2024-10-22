#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_wilson_clover_hasenbusch_twist_preconditioned.cuh>

namespace quda
{

  /* ***************************
   * No Clov Inv:  1 - k^2 D - i mu gamma_5 A
   * **************************/
  template <typename Arg>
  class WilsonCloverHasenbuschTwistPCNoClovInv : public Dslash<cloverHasenbuschPreconditioned, Arg>
  {
    using Dslash = Dslash<cloverHasenbuschPreconditioned, Arg>;
    using Dslash::arg;
    using Dslash::halo;
    using Dslash::in;

  public:
    WilsonCloverHasenbuschTwistPCNoClovInv(Arg &arg, cvector_ref<ColorSpinorField> &out,
                                           cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);

      // specialize here to constrain the template instantiation
      if (arg.nParity == 1) {
        if (arg.xpay)
          Dslash::template instantiate<packShmem, 1, true>(tp, stream);
        else
          errorQuda("Operator only defined for xpay=true");
      } else {
        errorQuda("Operator not defined nParity=%d", arg.nParity);
      }
    }

    long long flops() const
    {
      int clover_flops = 504;
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        // 2 from fwd / back face * 1 clover terms:
        // there is no A^{-1}D only D
        // there is one clover_term and 48 is the - mu (igamma_5) A
        flops += 2 * (clover_flops + 48) * halo.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        flops += 2 * (clover_flops + 48)
          * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += (clover_flops + 48) * halo.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * halo.GhostFace()[d];
        flops -= (clover_flops + 48) * ghost_sites;

        break;
      }
      return flops;
    }

    long long bytes() const
    {
      int clover_bytes = 72 * in.Precision() + (isFixed<typename Arg::Float>::value ? 2 * sizeof(float) : 0);

      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        // Factor of 2 is from the fwd/back faces.
        bytes += clover_bytes * 2 * halo.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        // Factor of 2 is from the fwd/back faces
        bytes
          += clover_bytes * 2 * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:

        bytes += clover_bytes * halo.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * halo.GhostFace()[d];
        bytes -= clover_bytes * ghost_sites;

        break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct WilsonCloverHasenbuschTwistPCNoClovInvApply {

    WilsonCloverHasenbuschTwistPCNoClovInvApply(cvector_ref<ColorSpinorField> &out,
                                                cvector_ref<const ColorSpinorField> &in,
                                                cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                                                const CloverField &A, double a, double b, int parity, bool dagger,
                                                const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      using ArgType = WilsonCloverHasenbuschTwistPCArg<Float, nColor, nDim, DDArg, recon, false>;
      ArgType arg(out, in, halo, U, A, a, b, x, parity, dagger, comm_override);
      WilsonCloverHasenbuschTwistPCNoClovInv<ArgType> wilson(arg, out, in, halo);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverHasenbuschTwistPCNoClovInv(cvector_ref<ColorSpinorField> &out,
                                                   cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                                                   const CloverField &A, double a, double b,
                                                   cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                                   const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH>()) {
      instantiate<WilsonCloverHasenbuschTwistPCNoClovInvApply>(out, in, x, U, A, a, b, parity, dagger, comm_override,
                                                               profile);
    } else {
      errorQuda("Clover Hasenbusch Twist operator has not been built");
    }
  }

  /* ***************************
   * Clov Inv
   *
   * M = psi_p - k^2 A^{-1} D_p\not{p} - i mu gamma_5 A_{pp} psi_{p}
   * **************************/
  template <typename Arg>
  class WilsonCloverHasenbuschTwistPCClovInv : public Dslash<cloverHasenbuschPreconditioned, Arg>
  {
    using Dslash = Dslash<cloverHasenbuschPreconditioned, Arg>;
    using Dslash::arg;
    using Dslash::halo;
    using Dslash::in;

  public:
    WilsonCloverHasenbuschTwistPCClovInv(Arg &arg, cvector_ref<ColorSpinorField> &out,
                                         cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);

      // specialize here to constrain the template instantiation
      if (arg.nParity == 1) {
        if (arg.xpay)
          Dslash::template instantiate<packShmem, 1, true>(tp, stream);
        else
          errorQuda("Operator only defined for xpay=true");
      } else {
        errorQuda("Operator not defined nParity=%d", arg.nParity);
      }
    }

    long long flops() const
    {
      int clover_flops = 504;
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        // 2 from fwd / back face * 2 clover terms:
        // one clover_term from the A^{-1}D
        // second clover_term and 48 is the - mu (igamma_5) A
        flops += 2 * (2 * clover_flops + 48) * halo.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        flops += 2 * (2 * clover_flops + 48)
          * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += (2 * clover_flops + 48) * halo.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * halo.GhostFace()[d];
        flops -= (2 * clover_flops + 48) * ghost_sites;

        break;
      }
      return flops;
    }

    long long bytes() const
    {
      int clover_bytes = 72 * in.Precision() + (isFixed<typename Arg::Float>::value ? 2 * sizeof(float) : 0);

      // if we use dynamic clover we read only A (even for A^{-1}
      // otherwise we read both A and A^{-1}
      int dyn_factor = arg.dynamic_clover ? 1 : 2;

      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
        // Factor of 2 is from the fwd/back faces.
        bytes += dyn_factor * clover_bytes * 2 * halo.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        // Factor of 2 is from the fwd/back faces
        bytes += dyn_factor * clover_bytes * 2
          * (halo.GhostFace()[0] + halo.GhostFace()[1] + halo.GhostFace()[2] + halo.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:

        bytes += dyn_factor * clover_bytes * halo.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * halo.GhostFace()[d];
        bytes -= dyn_factor * clover_bytes * ghost_sites;

        break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct WilsonCloverHasenbuschTwistPCClovInvApply {

    WilsonCloverHasenbuschTwistPCClovInvApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                              cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                                              const CloverField &A, double kappa, double mu, int parity, bool dagger,
                                              const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      using ArgType = WilsonCloverHasenbuschTwistPCArg<Float, nColor, nDim, DDArg, recon, true>;
      ArgType arg(out, in, halo, U, A, kappa, mu, x, parity, dagger, comm_override);
      WilsonCloverHasenbuschTwistPCClovInv<ArgType> wilson(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverHasenbuschTwistPCClovInv(cvector_ref<ColorSpinorField> &out,
                                                 cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                                                 const CloverField &A, double a, double b,
                                                 cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                                 const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH>()) {
      instantiate<WilsonCloverHasenbuschTwistPCClovInvApply>(out, in, x, U, A, a, b, parity, dagger, comm_override,
                                                             profile);
    } else {
      errorQuda("Clover Hasenbusch Twist operator has not been built");
    }
  }

} // namespace quda
