#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_domain_wall_4d.cuh>

/**
   This is the gauged domain-wall 4-d preconditioned operator.

   Note, for now, this just applies a batched 4-d dslash across the fifth
   dimension.
*/

namespace quda
{

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct DomainWall4DLaunch {
    static constexpr const char *kernel = "quda::domainWall4DGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      dslash.launch(domainWall4DGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class DomainWall4D : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    DomainWall4D(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
        Dslash<Float>(arg, out, in, "kernels/dslash_domain_wall_4d.cuh"),
        arg(arg),
        in(in)
    {
      TunableVectorYZ::resizeVector(in.X(4), arg.nParity);
    }

    virtual ~DomainWall4D() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      typedef typename mapper<Float>::type real;
#ifdef JITIFY
      // we need to break the dslash launch abstraction here to get a handle on the constant memory pointer in the kernel module
      using namespace jitify::reflection;
      const auto kernel = DomainWall4DLaunch<void, 0, 0, 0, false, false, INTERIOR_KERNEL, Arg>::kernel;
      auto instance = Dslash<Float>::program_->kernel(kernel).instantiate(
          Type<Float>(), nDim, nColor, arg.nParity, arg.dagger, arg.xpay, arg.kernel_type, Type<Arg>());
      cuMemcpyHtoDAsync(
          instance.get_constant_ptr("quda::mobius_d"), mobius_h, QUDA_MAX_DWF_LS * sizeof(complex<real>), stream);
      Tunable::jitify_error = instance.configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      cudaMemcpyToSymbolAsync(
          mobius_d, mobius_h, QUDA_MAX_DWF_LS * sizeof(complex<real>), 0, cudaMemcpyHostToDevice, streams[Nstream - 1]);
      Dslash<Float>::template instantiate<DomainWall4DLaunch, nDim, nColor>(tp, arg, stream);
#endif
    }

    TuneKey tuneKey() const
    {
      return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApply {

    inline DomainWall4DApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
        double m_5, const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity, bool dagger,
        const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      DomainWall4DArg<Float, nColor, recon> arg(out, in, U, a, m_5, b_5, c_5, a != 0.0, x, parity, dagger, comm_override);
      DomainWall4D<Float, nDim, nColor, DomainWall4DArg<Float, nColor, recon>> twisted(arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
          in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall4D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
      const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity, bool dagger,
      const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, x, U);

    // check all locations match
    checkLocation(out, in, x, U);

    instantiate<DomainWall4DApply>(out, in, U, a, m_5, b_5, c_5, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Domain-wall dslash has not been built");
#endif // GPU_DOMAIN_WALL_DIRAC
  }

} // namespace quda

#endif
