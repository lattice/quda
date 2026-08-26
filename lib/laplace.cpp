#include <instantiate_dslash.h>

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon> struct LaplaceApply {
    LaplaceApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                 cvector_ref<const ColorSpinorField> &x, const GaugeField &U, int dir, real_t a, real_t b, int parity,
                 const int *comm_override, TimeProfile &profile);
  };

  // Apply the Laplace operator
  // out(x) = M*in = - a*\sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu) + b*in(x)
  // Omits direction 'dir' from the operator.
  void ApplyLaplace(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const GaugeField &U,
                    int dir, real_t a, real_t b, cvector_ref<const ColorSpinorField> &x, int parity,
                    const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_LAPLACE_DSLASH>()) {
      instantiate<LaplaceApply>(out, in, x, U, dir, a, b, parity, comm_override, profile);
    } else {
      errorQuda("Laplace operator has not been enabled");
    }
  }
} // namespace quda
