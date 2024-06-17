#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
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
    DomainWall5D(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                 const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
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
      case UBER_KERNEL:
      case KERNEL_POLICY: {
        int Ls = in.X(4);
        long long bulk = (Ls - 2) * (in.Volume() / Ls);
        long long wall = 2 * (in.Volume() / Ls);
        flops += in.size() * 96ll * bulk + 120ll * wall;
      } break;
      default: break; // 5-d flops are in the interior kernel
      }
      return flops;
    }

    long long bytes() const
    {
      int spinor_bytes = 2 * in.Ncolor() * in.Nspin() * in.Precision() + (isFixed<typename Arg::Float>::value ? sizeof(float) : 0);
      long long bytes = Dslash::bytes();
      switch (arg.kernel_type) {
      case INTERIOR_KERNEL:
      case UBER_KERNEL:
      case KERNEL_POLICY: bytes += in.size() * 2 * spinor_bytes * in.VolumeCB(); break;
      default: break;
      }
      return bytes;
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct DomainWall5DApply {

    DomainWall5DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                      cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double m_f, int parity,
                      bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 5;
      auto halo = ColorSpinorField::create_comms_batch(in);
      DomainWall5DArg<Float, nColor, nDim, DDArg, recon> arg(out, in, halo, U, a, m_f, a != 0.0, x, parity, dagger,
                                                             comm_override);
      DomainWall5D<decltype(arg)> dwf(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(dwf)> policy(dwf, in, halo, profile);
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall5D(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         const GaugeField &U, double a, double m_f, cvector_ref<const ColorSpinorField> &x, int parity,
                         bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_DOMAIN_WALL_DSLASH>()) {
      instantiate<DomainWall5DApply>(out, in, x, U, a, m_f, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Domain-wall operator has not been built");
    }
  }

} // namespace quda
