#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_wilson.cuh>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{

  template <typename Arg> class Wilson : public Dslash<wilson, Arg>
  {
    using Dslash = Dslash<wilson, Arg>;

  public:
    Wilson(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in)
    {
      if(in.Ndim() == 5) {
        TunableKernel3D::resizeVector(in.X(4), arg.nParity);
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonApply {

    inline WilsonApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                       const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      WilsonArg<Float, nColor, nDim, recon> arg(out, in, U, a, x, parity, dagger, comm_override);
      Wilson<decltype(arg)> wilson(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, in.VolumeCB(), in.GhostFaceCB(), profile);
    }
  };

  // Apply the Wilson operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the a normalization for the Wilson operator.
#ifdef GPU_WILSON_DIRAC
  void ApplyWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                   const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    instantiate<WilsonApply, WilsonReconstruct>(out, in, U, a, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyWilson(ColorSpinorField &, const ColorSpinorField &, const GaugeField &, double,
                   const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Wilson dslash has not been built");
  }
#endif // GPU_WILSON_DIRAC

} // namespace quda
