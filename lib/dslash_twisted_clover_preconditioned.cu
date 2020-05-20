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

  template <typename Arg> class TwistedCloverPreconditioned : public Dslash<twistedCloverPreconditioned, Arg>
  {
    using Dslash = Dslash<twistedCloverPreconditioned, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    TwistedCloverPreconditioned(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
      Dslash(arg, out, in)
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
        errorQuda("Preconditioned twisted-clover operator not defined nParity=%d", arg.nParity);
      }
    }

    long long flops() const
    {
      int clover_flops = 504 + 48;
      long long flops = Dslash::flops();
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

      long long bytes = Dslash::bytes();
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
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct TwistedCloverPreconditionedApply {

    inline TwistedCloverPreconditionedApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        const CloverField &C, double a, double b, bool xpay, const ColorSpinorField &x, int parity, bool dagger,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      TwistedCloverArg<Float, nColor, nDim, recon> arg(out, in, U, C, a, b, xpay, x, parity, dagger, comm_override);
      TwistedCloverPreconditioned<decltype(arg)> twisted(arg, out, in);

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
    instantiate<TwistedCloverPreconditionedApply>(out, in, U, C, a, b, xpay, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Twisted-clover dslash has not been built");
#endif // GPU_TWISTED_CLOVER_DIRAC
  }

} // namespace quda
