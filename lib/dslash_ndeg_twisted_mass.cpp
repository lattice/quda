#include <instantiate_dslash.h>

/**
   This is the gauged twisted-mass operator acting on a non-generate
   quark doublet.
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct NdegTwistedMassApply {
    NdegTwistedMassApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                         cvector_ref<const ColorSpinorField> &x, const GaugeField &U, double a, double b, double c,
                         int parity, bool dagger, const int *comm_override, TimeProfile &profile);
  };

  void ApplyNdegTwistedMass(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                            const GaugeField &U, double a, double b, double c, cvector_ref<const ColorSpinorField> &x,
                            int parity, bool dagger, const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_MASS_DSLASH>()) {
      instantiate<NdegTwistedMassApply>(out, in, x, U, a, b, c, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Non-degenerate twisted-mass operator has not been built");
    }
  }

} // namespace quda
