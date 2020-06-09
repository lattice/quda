#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
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
    TwistedMassPreconditioned(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
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
          Dslash::template instantiate<packShmem, 1, true, xpay_<Arg::asymmetric>()>(tp, stream);
        else
          Dslash::template instantiate<packShmem, 1, true, false>(tp, stream);
      } else {
        if (arg.xpay)
          Dslash::template instantiate<packShmem, 1, not_dagger_<Arg::asymmetric>(), xpay_<Arg::asymmetric>()>(tp, stream);
        else
          Dslash::template instantiate<packShmem, 1, not_dagger_<Arg::asymmetric>(), false>(tp, stream);
      }
    }

    long long flops() const
    {
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        flops += 2 * in.Ncolor() * 4 * 2 * in.Volume(); // complex * Nc * Ns * fma * vol
        break;
      default: break;
      }
      return flops;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct TwistedMassPreconditionedApply {

    inline TwistedMassPreconditionedApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      if (asymmetric) {
        TwistedMassArg<Float, nColor, nDim, recon, true> arg(out, in, U, a, b, xpay, x, parity, dagger, comm_override);
        TwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in);

        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
        policy.apply(0);
      } else {
        TwistedMassArg<Float, nColor, nDim, recon, false> arg(out, in, U, a, b, xpay, x, parity, dagger, comm_override);
        TwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in);

        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
        policy.apply(0);
      }

      checkCudaError();
    }
  };

  /*
    Apply the preconditioned twisted-mass Dslash operator

    out = x + A^{-1} D * in = x + a*(1 + i*b*gamma_5)*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  */
  void ApplyTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
      double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
      const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_TWISTED_MASS_DIRAC
    // with symmetric dagger operator we must use kernel packing
    if (dagger && !asymmetric) pushKernelPackT(true);

    instantiate<TwistedMassPreconditionedApply>(
        out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override, profile);

    if (dagger && !asymmetric) popKernelPackT();
#else
    errorQuda("Twisted-mass dslash has not been built");
#endif // GPU_TWISTED_MASS_DIRAC
  }

} // namespace quda
