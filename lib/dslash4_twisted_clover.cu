#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson_clover.cuh>

/**
   This is the basic gauged twisted-clover operator
*/

namespace quda
{

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct TwistedCloverLaunch {
    static constexpr const char *kernel = "quda::wilsonCloverGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      static_assert(xpay == true, "Twisted-clover operator only defined for xpay");
      dslash.launch(wilsonCloverGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class TwistedClover : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    TwistedClover(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
        Dslash<Float>(arg, out, in, "kernels/dslash_wilson_clover.cuh"),
        arg(arg),
        in(in)
    {
    }

    virtual ~TwistedClover() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.xpay)
        Dslash<Float>::template instantiate<TwistedCloverLaunch, nDim, nColor, true>(tp, arg, stream);
      else
        errorQuda("Twisted-clover operator only defined for xpay=true");
    }

    long long flops() const
    {
      int clover_flops = 504 + 48;
      long long flops = Dslash<Float>::flops();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL: break; // all clover flops are in the interior kernel
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: flops += clover_flops * in.Volume(); break;
      }
      return flops;
    }

    long long bytes() const
    {
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int clover_bytes = 72 * in.Precision() + (isFixed ? 2 * sizeof(float) : 0);

      long long bytes = Dslash<Float>::bytes();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL: break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: bytes += clover_bytes * in.Volume(); break;
      }

      return bytes;
    }

    TuneKey tuneKey() const
    {
      return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct TwistedCloverApply {

    inline TwistedCloverApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        const CloverField &C, double a, double b, const ColorSpinorField &x, int parity, bool dagger,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      WilsonCloverArg<Float, nColor, recon, true> arg(out, in, U, C, a, b, x, parity, dagger, comm_override);
      TwistedClover<Float, nDim, nColor, WilsonCloverArg<Float, nColor, recon, true>> twisted(arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the twisted-mass Dslash operator
  // out(x) = M*in = (A + i*b*gamma_5)*in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the kappa normalization for the Wilson operator, with a = -kappa.
  void ApplyTwistedClover(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, const CloverField &C,
      double a, double b, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
      TimeProfile &profile)
  {
#ifdef GPU_TWISTED_CLOVER_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U, C);

    // check all locations match
    checkLocation(out, in, U, C);

    instantiate<TwistedCloverApply>(out, in, U, C, a, b, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Twisted-clover dslash has not been built");
#endif // GPU_TWISTED_CLOVEr_DIRAC
  }

} // namespace quda

#endif
