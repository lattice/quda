#ifndef USE_LEGACY_DSLASH

#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_domain_wall_5d.cuh>

/**
   This is the gauged domain-wall 5-d preconditioned operator.
*/

namespace quda
{

  /**
     @brief This is a helper class that is used to instantiate the
     correct templated kernel for the dslash.
   */
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  struct DomainWall5DLaunch {
    static constexpr const char *kernel = "quda::domainWall5DGPU"; // kernel name for jit compilation
    template <typename Dslash>
    inline static void launch(Dslash &dslash, TuneParam &tp, Arg &arg, const cudaStream_t &stream)
    {
      dslash.launch(domainWall5DGPU<Float, nDim, nColor, nParity, dagger, xpay, kernel_type, Arg>, tp, arg, stream);
    }
  };

  template <typename Float, int nDim, int nColor, typename Arg> class DomainWall5D : public Dslash<Float>
  {

protected:
    Arg &arg;
    const ColorSpinorField &in;

public:
    DomainWall5D(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) :
        Dslash<Float>(arg, out, in, "kernels/dslash_domain_wall_5d.cuh"),
        arg(arg),
        in(in)
    {
      TunableVectorYZ::resizeVector(in.X(4), arg.nParity);
    }

    virtual ~DomainWall5D() {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);
      Dslash<Float>::template instantiate<DomainWall5DLaunch, nDim, nColor>(tp, arg, stream);
    }

    long long flops() const
    {
      long long flops = Dslash<Float>::flops();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL: break; // 5-d flops are in the interior kernel
      case INTERIOR_KERNEL:
      case KERNEL_POLICY:
        int Ls = in.X(4);
        long long bulk = (Ls - 2) * (in.Volume() / Ls);
        long long wall = 2 * (in.Volume() / Ls);
        flops += 96ll * bulk + 120ll * wall;
        break;
      }
      return flops;
    }

    long long bytes() const
    {
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      long long bytes = Dslash<Float>::bytes();
      switch (arg.kernel_type) {
      case EXTERIOR_KERNEL_X:
      case EXTERIOR_KERNEL_Y:
      case EXTERIOR_KERNEL_Z:
      case EXTERIOR_KERNEL_T:
      case EXTERIOR_KERNEL_ALL: break;
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: bytes += 2 * spinor_bytes * in.VolumeCB(); break;
      }
      return bytes;
    }

    TuneKey tuneKey() const
    {
      return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall5DApply {

    inline DomainWall5DApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
        double m_f, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 5;
      DomainWall5DArg<Float, nColor, recon> arg(out, in, U, a, m_f, a != 0.0, x, parity, dagger, comm_override);
      DomainWall5D<Float, nDim, nColor, DomainWall5DArg<Float, nColor, recon>> twisted(arg, out, in);

      dslash::DslashPolicyTune<decltype(twisted)> policy(twisted,
          const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
          in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall5D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_f,
      const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_DOMAIN_WALL_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, x, U);

    // check all locations match
    checkLocation(out, in, x, U);

    // with 5-d checkerboarding we must use kernel packing
    pushKernelPackT(true);

    instantiate<DomainWall5DApply>(out, in, U, a, m_f, x, parity, dagger, comm_override, profile);

    popKernelPackT();
#else
    errorQuda("Domain-wall dslash has not been built");
#endif // GPU_DOMAIN_WALL_DIRAC
  }

} // namespace quda

#endif
