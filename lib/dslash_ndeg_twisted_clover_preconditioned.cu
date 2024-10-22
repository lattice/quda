#define SIGNATURE_ONLY
#include <dslash_ndeg_twisted_clover_preconditioned.hpp>
#undef SIGNATURE_ONLY

/**
   This is the gauged preconditioned twisted-clover operator 
   acting on a non-degenerate quark doublet.
*/

namespace quda
{

    void ApplyNdegTwistedCloverPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                              const GaugeField &U, const CloverField &A, double a, double b, double c,
                                              bool xpay, cvector_ref<const ColorSpinorField> &x, int parity,
                                              bool dagger, const int *comm_override, TimeProfile &profile)
    {
      if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
        auto dummy = DistanceType<false>();
        instantiate<NdegTwistedCloverPreconditionedApply>(out, in, x, U, A, a, b, c, xpay, parity, dagger,
                                                          comm_override, dummy, profile);
      } else {
        errorQuda("Non-degenerate preconditioned twisted-clover operator has not been built");
      }
    }

} // namespace quda

