#define SIGNATURE_ONLY
#include <dslash_ndeg_twisted_mass.hpp>
#undef SIGNATURE_ONLY

/**
   This is the gauged twisted-mass operator acting on a non-generate
   quark doublet.
*/

namespace quda
{

  void ApplyNdegTwistedMass(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                            const GaugeField &U, double a, double b, double c, cvector_ref<const ColorSpinorField> &x,
                            int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      auto dummy = DistanceType<false>();
      instantiate<NdegTwistedMassApply>(out, in, x, U, a, b, c, parity, dagger, comm_override,dummy,profile);
    } else {
      errorQuda("Non-degenerate twisted-mass operator has not been built");
    }
  }

} // namespace quda
