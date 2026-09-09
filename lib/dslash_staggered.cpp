#include <instantiate_dslash.h>

/**
   This is a staggered Dirac operator
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon_u> struct StaggeredApply {
    StaggeredApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                   cvector_ref<const ColorSpinorField> &x, const GaugeField &U, real_t a, int parity, bool dagger,
                   const int *comm_override, TimeProfile &profile);
  };

  void ApplyStaggered(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                      real_t a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                      const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
      instantiate<StaggeredApply, ReconstructStaggered>(out, in, x, U, a, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Staggered operator has not been built");
    }
  }

} // namespace quda
