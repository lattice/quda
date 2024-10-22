#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
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
    using Dslash::halo;
    using Dslash::in;

  protected:
    bool shared;
    unsigned int sharedBytesPerThread() const
    {
      return shared ? 2 * in.Ncolor() * 4 * sizeof(typename mapper<typename Arg::Float>::type) : 0;
    }

  public:
    NdegTwistedMassPreconditioned(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                  const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo), shared(arg.asymmetric || !arg.dagger)
    {
      if (shared) TunableKernel3D::resizeStep(2, 1); // this will force flavor to be contained in the block
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
      case UBER_KERNEL:
      case KERNEL_POLICY:
        flops += 2 * in.Ncolor() * 4 * 4 * halo.Volume(); // complex * Nc * Ns * fma * vol
        break;
      default: break; // twisted-mass flops are in the interior kernel
      }
      return flops;
    }
  };

  template <bool distance_pc> struct DistanceType {
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct NdegTwistedMassPreconditionedApply {
    template <bool distance_pc>
    NdegTwistedMassPreconditionedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                       cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double b,
                                       double c, bool xpay, int parity, bool dagger, bool asymmetric,
                                       const int *comm_override, DistanceType<distance_pc>, TimeProfile &profile)
#ifdef SIGNATURE_ONLY
    ;
#else
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      if (asymmetric) {
        NdegTwistedMassArg<Float, nColor, nDim, DDArg, recon, true> arg(out, in, halo, U, a, b, c, xpay, x, parity,
                                                                        dagger, comm_override);
        NdegTwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in, halo);
        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, in, halo, profile);
      } else {
        NdegTwistedMassArg<Float, nColor, nDim, DDArg, recon, false> arg(out, in, halo, U, a, b, c, xpay, x, parity,
                                                                         dagger, comm_override);
        NdegTwistedMassPreconditioned<decltype(arg)> twisted(arg, out, in, halo);
        dslash::DslashPolicyTune<decltype(twisted)> policy(twisted, in, halo, profile);
      }
    }
#endif
  };

} // namespace quda
