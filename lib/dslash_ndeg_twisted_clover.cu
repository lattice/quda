#define SIGNATURE_ONLY
#include <dslash_ndeg_twisted_clover.hpp>
#undef SIGNATURE_ONLY

/**
   This is the gauged non-degenerate twisted-clover operator acting on a 
   quark doublet.
*/

namespace quda
{

    void ApplyNdegTwistedClover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                const GaugeField &U, const CloverField &A, double a, double b, double c,
                                cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                const int *comm_override, TimeProfile &profile)
    {
      if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
        auto dummy = DistanceType<false>();
        instantiate<NdegTwistedCloverApply>(out, in, x, U, A, a, b, c, parity, dagger, comm_override, dummy, profile);
      } else {
        errorQuda("Non-degenerate twisted-clover operator has not been built");
      }
    }

} // namespace quda
