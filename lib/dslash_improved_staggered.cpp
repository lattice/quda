#include <instantiate_dslash.h>

/**
   This is a staggered Dirac operator
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon_l> struct ImprovedStaggeredApply {
    template <bool distance_pc>
    ImprovedStaggeredApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           cvector_ref<const ColorSpinorField> &x, const GaugeField &L, const GaugeField &U, double a,
                           int parity, bool dagger, const int *comm_override, DistanceType<distance_pc>,
                           TimeProfile &profile);
  };

  void ApplyImprovedStaggered(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                              const GaugeField &U, const GaugeField &L, double a, cvector_ref<const ColorSpinorField> &x,
                              int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_ASQTAD_DSLASH>()) {
      for (int i = 0; i < 4; i++) {
        if (comm_dim_partitioned(i) && (U.X()[i] < 6)) {
          errorQuda("partitioned dimension with local size less than 6 is not supported in improved staggered dslash");
        }
      }
      auto dummy = DistanceType<false>();
      // L must be first gauge field argument since we template on long reconstruct
      instantiate<ImprovedStaggeredApply, ReconstructStaggered>(out, in, x, L, U, a, parity, dagger, comm_override,
                                                                dummy, profile);
    } else {
      errorQuda("Improved staggered operator has not been built");
    }
  }

} // namespace quda
