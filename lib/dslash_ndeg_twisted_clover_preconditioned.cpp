#include <instantiate_dslash.h>

/**
   This is the gauged preconditioned twisted-clover operator
   acting on a non-degenerate quark doublet.
*/

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct NdegTwistedCloverPreconditionedApply {
    NdegTwistedCloverPreconditionedApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                         cvector_ref<const ColorSpinorField> &x, const GaugeField &U,
                                         const CloverField &A, real_t a, real_t b, real_t c, bool xpay, int parity,
                                         bool dagger, const int *comm_override, TimeProfile &profile);
  };

  void ApplyNdegTwistedCloverPreconditioned(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                            const GaugeField &U, const CloverField &A, real_t a, real_t b, real_t c,
                                            bool xpay, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                            const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_TWISTED_CLOVER_DSLASH>()) {
      instantiate<NdegTwistedCloverPreconditionedApply>(out, in, x, U, A, a, b, c, xpay, parity, dagger, comm_override,
                                                        profile);
    } else {
      errorQuda("Non-degenerate preconditioned twisted-clover operator has not been built");
    }
  }

} // namespace quda
