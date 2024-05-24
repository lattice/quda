#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.hpp>
#include <kernels/dslash_domain_wall_4d.cuh>

/**
   This is the gauged domain-wall 4-d preconditioned operator.

   Note, for now, this just applies a batched 4-d dslash across the fifth
   dimension.
*/

namespace quda
{

  template <typename Arg> class DomainWall4D : public Dslash<domainWall4D, Arg>
  {
    using Dslash = Dslash<domainWall4D, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    DomainWall4D(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
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
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApply {

    DomainWall4DApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, cvector_ref<const ColorSpinorField> &x,
                      const GaugeField &U, double a, double m_5, const Complex *b_5, const Complex *c_5,
                      int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      DomainWall4DArg<Float, nColor, nDim, recon> arg(out, in, halo, U, a, m_5, b_5, c_5, a != 0.0, x, parity, dagger,
                                                      comm_override);
      DomainWall4D<decltype(arg)> dwf(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(dwf)> policy(dwf, in, halo, profile);
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  void ApplyDomainWall4D(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         const GaugeField &U, double a, double m_5, const Complex *b_5, const Complex *c_5,
                         cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                         TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_DOMAIN_WALL_4D_DSLASH>() || is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
      instantiate<DomainWall4DApply>(out, in, x, U, a, m_5, b_5, c_5, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Domain-wall dslash has not been built");
    }
  }

} // namespace quda
