#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <dslash_helper.cuh>
#include <index_helper.cuh>
#include <gauge_field.h>

#include <dslash_policy.cuh>
#include <kernels/dslash_staggered.cuh>

/**
   This is a staggered Dirac operator
*/

namespace quda
{

  template <typename Arg> class Staggered : public Dslash<staggered, Arg>
  {
    using Dslash = Dslash<staggered, Arg>;

  public:
    Staggered(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      Dslash::template instantiate<packStaggeredShmem>(tp, stream);
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon_u> struct StaggeredApply {

    inline StaggeredApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                          const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                          TimeProfile &profile)
    {
      if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
#ifdef BUILD_MILC_INTERFACE
        constexpr int nDim = 4; // MWTODO: this probably should be 5 for mrhs Dslash
        constexpr bool improved = false;

        StaggeredArg<Float, nColor, nDim, recon_u, QUDA_RECONSTRUCT_NO, improved, QUDA_STAGGERED_PHASE_MILC> arg(
          out, in, U, U, a, x, parity, dagger, comm_override);
        Staggered<decltype(arg)> staggered(arg, out, in);

        dslash::DslashPolicyTune<decltype(staggered)> policy(
          staggered, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
        policy.apply(0);
#else
        errorQuda("MILC interface has not been built so MILC phase staggered fermions not enabled");
#endif
      } else if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef BUILD_TIFR_INTERFACE
        constexpr int nDim = 4; // MWTODO: this probably should be 5 for mrhs Dslash
        constexpr bool improved = false;

        StaggeredArg<Float, nColor, nDim, recon_u, QUDA_RECONSTRUCT_NO, improved, QUDA_STAGGERED_PHASE_TIFR> arg(
          out, in, U, U, a, x, parity, dagger, comm_override);
        Staggered<decltype(arg)> staggered(arg, out, in);

        dslash::DslashPolicyTune<decltype(staggered)> policy(
          staggered, const_cast<cudaColorSpinorField *>(static_cast<const cudaColorSpinorField *>(&in)), in.VolumeCB(),
          in.GhostFaceCB(), profile);
        policy.apply(0);
#else
        errorQuda("TIFR interface has not been built so TIFR phase taggered fermions not enabled");
#endif
      } else {
        errorQuda("Unsupported staggered phase type %d", U.StaggeredPhase());
      }

      checkCudaError();
    }
  };

  void ApplyStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                      const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
#ifdef GPU_STAGGERED_DIRAC
    if (in.V() == out.V()) errorQuda("Aliasing pointers");
    if (in.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch in = %d, out = %d", in.FieldOrder(), out.FieldOrder());

    // check all precisions match
    checkPrecision(out, in, U);

    // check all locations match
    checkLocation(out, in, U);

    instantiate<StaggeredApply, StaggeredReconstruct>(out, in, U, a, x, parity, dagger, comm_override, profile);
#else
    errorQuda("Staggered dslash has not been built");
#endif
  }

} // namespace quda
