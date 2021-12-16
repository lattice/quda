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

  template <typename Arg> class DomainWall4D : public Dslash<domainWall4D, Arg>
  {
    using Dslash = Dslash<domainWall4D, Arg>;
    using Dslash::arg;
    using Dslash::in;

  public:
    DomainWall4D(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      TunableKernel3D::resizeVector(in.X(4), arg.nParity);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct DomainWall4DApply {

    inline DomainWall4DApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                             double m_5, const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity,
                             bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      DomainWall4DArg<Float, nColor, nDim, recon> arg(out, in, U, a, m_5, b_5, c_5, a != 0.0, x, parity, dagger,
                                                      comm_override);
      DomainWall4D<decltype(arg)> dwf(arg, out, in);

      dslash::DslashPolicyTune<decltype(dwf)> policy(dwf, in, in.getDslashConstant().volume_4d_cb, in.getDslashConstant().ghostFaceCB, profile);
    }
  };

  // Apply the 4-d preconditioned domain-wall Dslash operator
  // out(x) = M*in = in(x) + a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
#if defined(GPU_DOMAIN_WALL_DIRAC) || defined(GPU_NDEG_TWISTED_CLOVER_DIRAC)
  void ApplyDomainWall4D(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a, double m_5,
                         const Complex *b_5, const Complex *c_5, const ColorSpinorField &x, int parity, bool dagger,
                         const int *comm_override, TimeProfile &profile)
  {
    instantiate<DomainWall4DApply>(out, in, U, a, m_5, b_5, c_5, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyDomainWall4D(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double, double,
                         const Complex *, const Complex *, const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Domain-wall dslash has not been built");
  }
#endif // GPU_DOMAIN_WALL_DIRAC

} // namespace quda
