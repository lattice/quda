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

  template <typename Arg> class DomainWall5D : public Dslash<domainWall5D, Arg>
  {
    using Dslash = Dslash<domainWall5D, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    DomainWall5D(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      TunableVectorYZ::resizeVector(in.X(4), arg.nParity);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }

    long long flops() const
    {
      long long flops = Dslash::flops();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: {
        int Ls = in.X(4);
        long long bulk = (Ls - 2) * (in.Volume() / Ls);
        long long wall = 2 * (in.Volume() / Ls);
        flops += 96ll * bulk + 120ll * wall;
      } break;
      default: break; // 5-d flops are in the interior kernel
      }
      return flops;
    }

    long long bytes() const
    {
      bool isFixed = (in.Precision() == sizeof(short) || in.Precision() == sizeof(char)) ? true : false;
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed ? sizeof(float) : 0);
      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case KERNEL_POLICY: bytes += 2 * spinor_bytes * in.VolumeCB(); break;
      default: break;
      }
      return bytes;
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall5DApply {

    inline DomainWall5DApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
        double m_f, const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 5;
      DomainWall5DArg<Float, nColor, nDim, recon> arg(out, in, U, a, m_f, a != 0.0, x, parity, dagger, comm_override);
      DomainWall5D<decltype(arg)> dwf(arg, out, in);

      dslash::DslashPolicyTune<decltype(dwf)> policy(
        dwf, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)),
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
    instantiate<DomainWall5DApply>(out, in, U, a, m_f, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Domain-wall dslash has not been built");
#endif // GPU_DOMAIN_WALL_DIRAC
  }

} // namespace quda
