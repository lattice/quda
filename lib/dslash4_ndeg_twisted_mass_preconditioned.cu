#ifndef USE_LEGACY_DSLASH

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

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct NdegTwistedMassPreconditionedLaunch {
    static constexpr const char *kernel = "quda::ndegTwistedMassPreconditionedGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      static_assert(nParity == 1, "Non-degenerate twisted-mass operator only defined for nParity=1");
      dslash.launch(ndegTwistedMassPreconditionedGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp,
          arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg>
  class NdegTwistedMassPreconditioned : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;
    bool shared;
    unsigned int sharedBytesPerThread() const
    {
      return shared ? 2 * nColor * 4 * sizeof(typename mapper<Float>::type) : 0;
    }

public:
    NdegTwistedMassPreconditioned(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
        Dslash<Float>(arg, out, in, "kernels/dslash_ndeg_twisted_mass_preconditioned.cuh"),
        arg(arg),
        in(in),
        shared(arg.asymmetric || !arg.dagger)
    {
      TunableVectorYZ::resizeVector(2, arg.nParity);
      if (arg.asymmetric)
        for (int i = 0; i < 8; i++)
          if (i != 4) { strcat(Dslash<Float>::aux[i], ",asym"); }
    }

    virtual ~NdegTwistedMassPreconditioned() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      if (arg.asymmetric && !arg.dagger) errorQuda("asymmetric operator only defined for dagger");
      if (arg.asymmetric && arg.xpay) errorQuda("asymmetric operator not defined for xpay");

      if (arg.nParity == 1) {
        if (arg.xpay)
          Dslash<Float>::template instantiate<NdegTwistedMassPreconditionedLaunch, nDim, nColor, 1, true>(
              tp, arg, stream);
        else
          Dslash<Float>::template instantiate<NdegTwistedMassPreconditionedLaunch, nDim, nColor, 1, false>(
              tp, arg, stream);
      } else {
        errorQuda("Preconditioned non-degenerate twisted-mass operator not defined nParity=%d", arg.nParity);
      }
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      if (shared) {
        param.block.y = 2; // flavor must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
      }
    }

    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);
      if (shared) {
        param.block.y = 2; // flavor must be contained in the block
        param.grid.y = 1;
        param.shared_bytes = sharedBytesPerThread() * param.block.x * param.block.y * param.block.z;
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
        flops += 2 * nColor * 4 * 4 * in.Volume(); // complex * Nc * Ns * fma * vol
        break;
      }
      return flops;
    }

    TuneKey tuneKey() const
    {
      return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct NdegTwistedMassPreconditionedApply {

    inline NdegTwistedMassPreconditionedApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U,
        double a, double b, double c, bool xpay, const ColorSpinorField &x, int parity, bool dagger, bool asymmetric,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      NdegTwistedMassArg<Float, nColor, recon> arg(
          out, in, U, a, b, c, xpay, x, parity, dagger, asymmetric, comm_override);
      NdegTwistedMassPreconditioned<Float, nDim, nColor, NdegTwistedMassArg<Float, nColor, recon>> twisted(arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
          in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
      policy.apply(0);

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
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, x, U);

    // check all locations match
    checkLocation(out, in, x, U);

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

#endif
