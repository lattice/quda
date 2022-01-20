#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
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
    using Dslash::in;

  public:
    WilsonCloverHasenbuschTwistPCNoClovInv(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash(arg, out, in) {}

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
        flops += 2 * (clover_flops + 48) * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        flops
          += 2 * (clover_flops + 48) * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += (clover_flops + 48) * in.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
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
        bytes += clover_bytes * 2 * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        // Factor of 2 is from the fwd/back faces
        bytes += clover_bytes * 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:

        bytes += clover_bytes * in.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        bytes -= clover_bytes * ghost_sites;

        break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonCloverHasenbuschTwistPCNoClovInvApply {

    inline WilsonCloverHasenbuschTwistPCNoClovInvApply(ColorSpinorField &out, const ColorSpinorField &in,
                                                       const GaugeField &U, const CloverField &A, double a, double b,
                                                       const ColorSpinorField &x, int parity, bool dagger,
                                                       const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      using ArgType = WilsonCloverHasenbuschTwistPCArg<Float, nColor, nDim, recon, false>;

      ArgType arg(out, in, U, A, a, b, x, parity, dagger, comm_override);
      WilsonCloverHasenbuschTwistPCNoClovInv<ArgType> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
#ifdef GPU_CLOVER_HASENBUSCH_TWIST
  void ApplyWilsonCloverHasenbuschTwistPCNoClovInv(ColorSpinorField &out, const ColorSpinorField &in,
                                                   const GaugeField &U, const CloverField &A, double a, double b,
                                                   const ColorSpinorField &x, int parity, bool dagger,
                                                   const int *comm_override, TimeProfile &profile)
  {
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U, A);

    // check all locations match
    checkLocation(out, in, U, A);

    instantiate<WilsonCloverHasenbuschTwistPCNoClovInvApply>(out, in, U, A, a, b, x, parity, dagger, comm_override,
                                                             profile);
  }
#else
  void ApplyWilsonCloverHasenbuschTwistPCNoClovInv(ColorSpinorField &, const ColorSpinorField &,
                                                   const GaugeField &, const CloverField &, double, double,
                                                   const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Clover Hasenbusch Twist dslash has not been built");
  }
#endif

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
    using Dslash::in;

  public:
    WilsonCloverHasenbuschTwistPCClovInv(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash(arg, out, in) {}

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
        flops += 2 * (2 * clover_flops + 48) * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        flops += 2 * (2 * clover_flops + 48)
          * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += (2 * clover_flops + 48) * in.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
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
        bytes += dyn_factor * clover_bytes * 2 * in.GhostFace()[arg.kernel_type];
        break;
      case EXTERIOR_KERNEL_ALL:
        // Factor of 2 is from the fwd/back faces
        bytes += dyn_factor * clover_bytes * 2
          * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY:

        bytes += dyn_factor * clover_bytes * in.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for bytes done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        bytes -= dyn_factor * clover_bytes * ghost_sites;

        break;
      }

      return bytes;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonCloverHasenbuschTwistPCClovInvApply {

    inline WilsonCloverHasenbuschTwistPCClovInvApply(ColorSpinorField &out, const ColorSpinorField &in,
                                                     const GaugeField &U, const CloverField &A, double kappa, double mu,
                                                     const ColorSpinorField &x, int parity, bool dagger,
                                                     const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      using ArgType = WilsonCloverHasenbuschTwistPCArg<Float, nColor, nDim, recon, true>;
      ArgType arg(out, in, U, A, kappa, mu, x, parity, dagger, comm_override);
      WilsonCloverHasenbuschTwistPCClovInv<ArgType> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + kappa * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
#ifdef GPU_CLOVER_HASENBUSCH_TWIST
  void ApplyWilsonCloverHasenbuschTwistPCClovInv(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
                                                 const CloverField &A, double a, double b, const ColorSpinorField &x,
                                                 int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    instantiate<WilsonCloverHasenbuschTwistPCClovInvApply>(out, in, U, A, a, b, x, parity, dagger, comm_override,
                                                           profile);
  }
#else
  void ApplyWilsonCloverHasenbuschTwistPCClovInv(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,
                                                 const CloverField &, double, double, const ColorSpinorField &,
                                                 int, bool, const int *, TimeProfile &)
  {
    errorQuda("Clover Hasenbusch Twist dslash has not been built");
  }
#endif

} // namespace quda

#endif
