#include <dslash.h>
#include <worker.h>
#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <index_helper.cuh>
#include <gauge_field.h>

#include <dslash_policy.hpp>
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
    Staggered(Arg &arg, cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
              const ColorSpinorField &halo) :
      Dslash(arg, out, in, halo)
    {
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash::setParam(tp);
      // operator is anti-Hermitian so do not instantiate dagger
      if (arg.xpay)
        Dslash::template instantiate<packStaggeredShmem, false, true>(tp, stream);
      else
        Dslash::template instantiate<packStaggeredShmem, false, false>(tp, stream);
    }
  };

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon_u> struct StaggeredApply {
    StaggeredApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, int parity, bool dagger,
                   const int *comm_override, TimeProfile &profile)
    {
      constexpr int nDim = 4;
      constexpr bool improved = false;
      auto halo = ColorSpinorField::create_comms_batch(in);
#ifdef QUDA_DSLASH_DOUBLE_STORE
      GaugeField Uback = shift(U, 1);
#else
      const GaugeField &Uback = shift(U, 1);
#endif

      if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC
          || (U.LinkType() == QUDA_GENERAL_LINKS && U.Reconstruct() == QUDA_RECONSTRUCT_NO)) {
        if constexpr (is_enabled<QUDA_MILC_GAUGE_ORDER>()) {
          StaggeredArg<Float, nColor, nDim, DDArg, recon_u, QUDA_RECONSTRUCT_NO, improved, QUDA_STAGGERED_PHASE_MILC> arg(
            out, in, halo, U, Uback, U, Uback, a, x, parity, dagger, comm_override);
          Staggered<decltype(arg)> staggered(arg, out, in, halo);

          dslash::DslashPolicyTune<decltype(staggered)> policy(staggered, in, halo, profile);
        } else {
          errorQuda("MILC interface has not been built so MILC phase staggered fermions not enabled");
        }
      } else if (U.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
        if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>()) {
          StaggeredArg<Float, nColor, nDim, DDArg, recon_u, QUDA_RECONSTRUCT_NO, improved, QUDA_STAGGERED_PHASE_TIFR> arg(
            out, in, halo, U, Uback, U, Uback, a, x, parity, dagger, comm_override);
          Staggered<decltype(arg)> staggered(arg, out, in, halo);

          dslash::DslashPolicyTune<decltype(staggered)> policy(staggered, in, halo, profile);
        } else {
          errorQuda("TIFR interface has not been built so TIFR phase taggered fermions not enabled");
        }
      }
    }
  };

} // namespace quda
