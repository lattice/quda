#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_ndeg_twisted_mass_preconditioned.cuh>

/**
   This is the preconditioned twisted-mass operator acting on a non-generate
   quark doublet.
*/

namespace quda
{

  // trait to ensure we don't instantiate asymmetric & xpay
  template <bool symmetric> constexpr bool xpay_() { return true; }
  template <> constexpr bool xpay_<true>() { return false; }

  // trait to ensure we don't instantiate asymmetric & !dagger
  template <bool symmetric> constexpr bool not_dagger_() { return false; }
  template <> constexpr bool not_dagger_<true>() { return true; }

  template <typename Arg> class NdegTwistedMassPreconditioned : public Dslash<nDegTwistedMassPreconditioned, Arg>
  {
    using Dslash = Dslash<nDegTwistedMassPreconditioned, Arg>;
    using Dslash::arg;
    using Dslash::in;

  protected:
    bool shared;
    unsigned int sharedBytesPerThread() const
    {
      return shared ? 2 * in.Ncolor() * 4 * sizeof(typename mapper<typename Arg::Float>::type) : 0;
    }

  public:
    NdegTwistedMassPreconditioned(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash(arg, out, in),
      shared(arg.asymmetric || !arg.dagger)
    {
      TunableVectorYZ::resizeVector(2, arg.nParity);
      if (shared) TunableVectorY::resizeStep(2); // this will force flavor to be contained in the block
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      if (arg.asymmetric && !arg.dagger) errorQuda("asymmetric operator only defined for dagger");
      if (arg.asymmetric && arg.xpay) errorQuda("asymmetric operator not defined for xpay");
      if (arg.nParity != 1) errorQuda("Preconditioned non-degenerate twisted-mass operator not defined nParity=%d", arg.nParity);

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

    void initTuneParam(TuneParam &param) const
    {
      Dslash::initTuneParam(param);
      if (shared) param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
    }

    void defaultTuneParam(TuneParam &param) const
    {
      Dslash::defaultTuneParam(param);
      if (shared) param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
    }

    long long flops() const
    {
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        flops += 2 * in.Ncolor() * 4 * 4 * in.Volume(); // complex * Nc * Ns * fma * vol
        break;
      default: break; // twisted-mass flops are in the interior kernel
      }
      return flops;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct NdegTwistedMassPreconditionedApply {

    inline NdegTwistedMassPreconditionedApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        double a, double b, double c, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      if (asymmetric) {
        NdegTwistedMassArg<Float, nColor, nDim, recon, true> arg(out, in, U, a, b, c, xpay, x, parity, dagger, comm_override);
        NdegTwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in);

        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
          in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
        policy.apply(0);
      } else {
        NdegTwistedMassArg<Float, nColor, nDim, recon, false> arg(out, in, U, a, b, c, xpay, x, parity, dagger, comm_override);
        NdegTwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in);

        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
          in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
        policy.apply(0);
      }

      checkCudaError();
    }
  };

  // Apply the non-degenerate twisted-mass Dslash operator
  // out(x) = M*in = a*(1 + i*b*gamma_5*tau_3 + c*tau_1)*D + x
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyNdegTwistedMassPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
      double a, double b, double c, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
      const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_NDEG_TWISTED_MASS_DIRAC
    // with symmetric dagger operator we must use kernel packing
    if (dagger && !asymmetric) pushKernelPackT(true);

    instantiate<NdegTwistedMassPreconditionedApply>(
        out, in, U, a, b, c, xpay, x, parity, dagger, asymmetric, comm_override, profile);

    if (dagger && !asymmetric) popKernelPackT();
#else
    errorQuda("Non-degenerate twisted-mass dslash has not been built");
#endif // GPU_NDEG_TWISTED_MASS_DIRAC
  }

} // namespace quda
