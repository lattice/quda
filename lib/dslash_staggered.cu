#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
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
    using Dslash::arg;

  public:
    Staggered(Arg &arg, const ColorSpinorField &out, const ColorSpinorField &in) : Dslash(arg, out, in) {}

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      // operator is anti-Hermitian so do not instantiate dagger
      if (arg.nParity == 1) {
        if (arg.xpay)
          Dslash::template instantiate<packStaggeredShmem, 1, false, true>(tp, stream);
        else
          Dslash::template instantiate<packStaggeredShmem, 1, false, false>(tp, stream);
      } else if (arg.nParity == 2) {
        if (arg.xpay)
          Dslash::template instantiate<packStaggeredShmem, 2, false, true>(tp, stream);
        else
          Dslash::template instantiate<packStaggeredShmem, 2, false, false>(tp, stream);
      }
    }
  };

  template <typename Float, int nColor, QudaReconstructType recon_u> struct StaggeredApply {

#if defined(BUILD_MILC_INTERFACE) || defined(BUILD_TIFR_INTERFACE)
    StaggeredApply(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                   const ColorSpinorField &x, int parity, bool dagger, const int *comm_override,
                   TimeProfile &profile)
    {
      if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC || (U.LinkType() == QUDA_GENERAL_LINKS && U.Reconstruct() == QUDA_RECONSTRUCT_NO)) {
#ifdef BUILD_MILC_INTERFACE
        constexpr int nDim = 4;
        constexpr bool improved = false;

        StaggeredArg<Float, nColor, nDim, recon_u, QUDA_RECONSTRUCT_NO, improved, QUDA_STAGGERED_PHASE_MILC> arg(
          out, in, U, U, a, x, parity, dagger, comm_override);
        Staggered<decltype(arg)> staggered(arg, out, in);

        dslash::DslashPolicyTune<decltype(staggered)> policy(staggered, in, in.VolumeCB(), in.GhostFaceCB(), profile);
#else
        errorQuda("MILC interface has not been built so MILC phase staggered fermions not enabled");
#endif
      } else if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef BUILD_TIFR_INTERFACE
        constexpr int nDim = 4;
        constexpr bool improved = false;

        StaggeredArg<Float, nColor, nDim, recon_u, QUDA_RECONSTRUCT_NO, improved, QUDA_STAGGERED_PHASE_TIFR> arg(
          out, in, U, U, a, x, parity, dagger, comm_override);
        Staggered<decltype(arg)> staggered(arg, out, in);

        dslash::DslashPolicyTune<decltype(staggered)> policy(staggered, in, in.VolumeCB(), in.GhostFaceCB(), profile);
#else
        errorQuda("TIFR interface has not been built so TIFR phase taggered fermions not enabled");
#endif
      } else {
        errorQuda("Unsupported combination of staggered phase type %d gauge link type %d and reconstruct %d", U.StaggeredPhase(), U.LinkType(), U.Reconstruct());
      }
    }
#else
    StaggeredApply(ColorSpinorField &, const ColorSpinorField &, const GaugeField &U, double,
                   const ColorSpinorField &, int, bool, const int *, TimeProfile &)
    {
      errorQuda("Unsupported combination of staggered phase type %d gauge link type %d and reconstruct %d", U.StaggeredPhase(), U.LinkType(), U.Reconstruct());
    }
#endif
  };

#ifdef GPU_STAGGERED_DIRAC
  void ApplyStaggered(ColorSpinorField &out, const ColorSpinorField &in, const GaugeField &U, double a,
                      const ColorSpinorField &x, int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    instantiate<StaggeredApply, StaggeredReconstruct>(out, in, U, a, x, parity, dagger, comm_override, profile);
  }
#else
  void ApplyStaggered(ColorSpinorField &, const ColorSpinorField &, const GaugeField &,  double,
                      const ColorSpinorField &, int, bool, const int *, TimeProfile &)
  {
    errorQuda("Staggered dslash has not been built");
  }
#endif

} // namespace quda
