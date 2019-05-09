#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_twisted_clover_preconditioned.cuh>

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
  struct TwistedCloverPreconditionedLaunch {
    static constexpr const char *kernel = "quda::twistedCloverPreconditionedGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      static_assert(nParity == 1, "preconditioned twisted-mass operator only defined for nParity=1");
      if (xpay && dagger) errorQuda("xpay operator only defined for not dagger");
      dslash.launch(twistedCloverPreconditionedGPU < Float, nDim, nColor, nParity, dagger && !xpay, xpay && !dagger,
          kernel_type, Arg >, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class TwistedCloverPreconditioned : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    TwistedCloverPreconditioned(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
        Dslash<Float>(arg, out, in, "kernels/dslash_twisted_clover_preconditioned.cuh"),
        arg(arg),
        in(in)
    {
    }

    virtual ~TwistedCloverPreconditioned() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.nParity == 1) {
        if (arg.xpay)
          Dslash<Float>::template instantiate<TwistedCloverPreconditionedLaunch, nDim, nColor, 1, true>(tp, arg, stream);
        else
          Dslash<Float>::template instantiate<TwistedCloverPreconditionedLaunch, nDim, nColor, 1, false>(tp, arg, stream);
      } else {
        errorQuda("Preconditioned twisted-clover operator not defined nParity=%d", arg.nParity);
      }
    }

    long long flops() const
    {
      int clover_flops = 504 + 48;
      long long flops = Dslash<Float>::flops();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: flops += clover_flops * 2 * in.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL:
        flops += clover_flops * 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        flops += clover_flops * in.Volume();

        if (arg.kernel_type == KERNEL_POLICY) break;
        // now correct for flops done by exterior kernel
        long long ghost_sites = 0;
        for (int d = 0; d < 4; d++)
          if (arg.commDim[d]) ghost_sites += 2 * in.GhostFace()[d];
        flops -= clover_flops * ghost_sites;

        break;
      }
      return flops;
    }

    long long bytes() const
    {
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int clover_bytes = 72 * in.Precision() + (isFixed ? 2 * sizeof(float) : 0);
      if (!arg.dynamic_clover) clover_bytes *= 2;

      long long bytes = Dslash<Float>::bytes();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T: bytes += clover_bytes * 2 * in.GhostFace()[arg.kernel_type]; break;
      case EXTERIOR_KERNEL_ALL:
        bytes += clover_bytes * 2 * (in.GhostFace()[0] + in.GhostFace()[1] + in.GhostFace()[2] + in.GhostFace()[3]);
        break;
      case INTERIOR_KERNEL:
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

    TuneKey tuneKey() const
    {
      return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct TwistedCloverPreconditionedApply {

    inline TwistedCloverPreconditionedApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        const CloverField &C, double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
#ifdef DYNAMIC_CLOVER
      constexpr bool dynamic_clover = true;
#else
      constexpr bool dynamic_clover = false;
#endif
      TwistedCloverArg<Float, nColor, recon, dynamic_clover> arg(
          out, in, U, C, a, b, xpay, x, parity, dagger, comm_override);
      TwistedCloverPreconditioned<Float, nDim, nColor, TwistedCloverArg<Float, nColor, recon, dynamic_clover>> twisted(
          arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
      policy.apply(0);
      checkCudaError();
    }
  };

  /*
    Apply the preconditioned twisted-mass Dslash operator

    out = x + a*A^{-1} D * in = x + a*(C + i*b*gamma_5)^{-1}*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  */
  void ApplyTwistedCloverPreconditioned(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
      const CloverField &C, double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger,
      const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_TWISTED_CLOVER_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U, C);

    // check all locations match
    checkLocation(out, in, U, C);

    instantiate<TwistedCloverPreconditionedApply>(out, in, U, C, a, b, xpay, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Twisted-clover dslash has not been built");
#endif // GPU_TWISTED_CLOVER_DIRAC
  }

} // namespace quda

#endif
