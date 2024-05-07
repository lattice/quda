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
    Wilson(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
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

  template <typename Float, int nColor, QudaReconstructType recon> struct WilsonApply {

    WilsonApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                TimeProfile &profile)
    {
      constexpr int nDim = 4;
      auto halo = ColorSpinorField::create_comms_batch(in);
      WilsonArg<Float, nColor, nDim, recon> arg(out, in, halo, U, a, x, parity, dagger, comm_override);
      Wilson<decltype(arg)> wilson(arg, out, in, halo);
      dslash::DslashPolicyTune<decltype(wilson)> policy(wilson, in, halo, profile);
    }
  };

  // Apply the Wilson operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the a normalization for the Wilson operator.
  void ApplyWilson(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                   double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger, const int *comm_override,
                   TimeProfile &profile)
  {
    if (in[0].Ndim() == 5) errorQuda("Unexpected nDim = 5");
    if constexpr (is_enabled<QUDA_WILSON_DSLASH>()) {
      instantiate<WilsonApply, WilsonReconstruct>(out, in, U, a, x, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Wilson operator has not been built");
    }
  }

} // namespace quda
