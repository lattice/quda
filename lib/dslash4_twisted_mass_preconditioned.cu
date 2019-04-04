#ifndef USE_LEGACY_DSLASH

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

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct TwistedMassPreconditionedLaunch {
    static constexpr const char *kernel = "quda::twistedMassPreconditionedGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      static_assert(nParity == 1, "preconditioned twisted-mass operator only defined for nParity=1");
      dslash.launch(
          twistedMassPreconditionedGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class TwistedMassPreconditioned : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    TwistedMassPreconditioned(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
        Dslash<Float>(arg, out, in, "kernels/dslash_twisted_mass_preconditioned.cuh"),
        arg(arg),
        in(in)
    {
      if (arg.asymmetric)
        for (int i = 0; i < 8; i++)
          if (i != 4) { strcat(Dslash<Float>::aux[i], ",asym"); }
    }

    virtual ~TwistedMassPreconditioned() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.asymmetric && !arg.dagger) errorQuda("asymmetric operator only defined for dagger");
      if (arg.asymmetric && arg.xpay) errorQuda("asymmetric operator not defined for xpay");

      if (arg.nParity == 1) {
        if (arg.xpay)
          Dslash<Float>::template instantiate<TwistedMassPreconditionedLaunch, nDim, nColor, 1, true>(tp, arg, stream);
        else
          Dslash<Float>::template instantiate<TwistedMassPreconditionedLaunch, nDim, nColor, 1, false>(tp, arg, stream);
      } else {
        errorQuda("Preconditioned twisted-mass operator not defined nParity=%d", arg.nParity);
      }
    }

    long long flops() const
    {
      long long flops = Dslash<Float>::flops();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL: break; // twisted-mass flops are in the interior kernel
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        flops += 2 * nColor * 4 * 2 * in.Volume(); // complex * Nc * Ns * fma * vol
        break;
      }
      return flops;
    }

    TuneKey tuneKey() const
    {
      return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct TwistedMassPreconditionedApply {

    inline TwistedMassPreconditionedApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      TwistedMassArg<Float, nColor, recon> arg(out, in, U, a, b, xpay, x, parity, dagger, asymmetric, comm_override);
      TwistedMassPreconditioned<Float, nDim, nColor, TwistedMassArg<Float, nColor, recon>> twisted(arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
      policy.apply(0);

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
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

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

#endif
