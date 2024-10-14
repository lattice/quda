#define SIGNATURE_ONLY
#include <dslash_staggered.hpp>
#undef SIGNATURE_ONLY


/**
   This is a staggered Dirac operator
*/

namespace quda
{

  void ApplyStaggered(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                      double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                      const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<StaggeredApply, ReconstructStaggered>(out, in, x, U, a, parity, dagger, comm_override, dummy, profile);
    } else {
      errorQuda("Staggered operator has not been built");
    }
  }

} // namespace quda
