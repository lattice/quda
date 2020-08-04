#include <gauge_field.h>
#include <color_spinor_field.h>
#include <dslash.h>
#include <worker.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_overlap_wilson.cuh>

/**
   This is the basic gauged Wilson operator
   TODO
   - gauge fix support
*/

namespace quda
{
/*
  template <typename Arg> class OverlapWilson : public Dslash<wilson, Arg>
  {
    using Dslash = Dslash<wilson, Arg>;

  public:
    OverlapWilson(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const hipStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon> struct OverlapWilsonApply {

    inline OverlapWilsonApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                       const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      OverlapWilsonArg<Float, nColor, nDim, recon> arg(out, in, U, a, x, parity, dagger, comm_override);
      OverlapWilson<decltype(arg)> overlap(arg, out, in);

      dslash::DslashPolicyTune<decltype(wilson)> policy(
        overlap, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
        in.GhostFaceCB(), profile);
      policy.apply(0);

      checkCudaError();
    }
  };

  // Apply the OverlapWilson operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu)
  // Uses the a normalization for the OverlapWilson operator.
  void ApplyOverlapWilson(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                   const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_OVERLAP_WILSON_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    instantiate<OverlapWilsonApply, OverlapWilsonReconstruct>(out, in, U, a, x, parity, dagger, comm_override, profile);
#else
    errorQuda("OverlapWilson dslash has not been built");
#endif // GPU_OVERLAP_WILSON_DIRAC
  }
*/
} // namespace quda
