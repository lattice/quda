#include <instantiate_dslash.h>

namespace quda
{

  template <typename Float, int nColor, typename DDArg, QudaReconstructType recon>
  struct WilsonCloverHasenbuschTwistApply {
    WilsonCloverHasenbuschTwistApply(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                     cvector_ref<const ColorSpinorField> &x, const GaugeField &U, const CloverField &A,
                                     real_t a, real_t b, int parity, bool dagger, const int *comm_override,
                                     TimeProfile &profile);
  };

  // Apply the Wilson-clover operator
  // out(x) = M*in = (A(x) + a * \sum_mu U_{-\mu}(x)in(x+mu) + U^\dagger_mu(x-mu)in(x-mu))
  // Uses the kappa normalization for the Wilson operator.
  void ApplyWilsonCloverHasenbuschTwist(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                        const GaugeField &U, const CloverField &A, real_t a, real_t b,
                                        cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
                                        const int *comm_override, TimeProfile &profile)
  {
    if constexpr (is_enabled<QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH>()) {
      instantiate<WilsonCloverHasenbuschTwistApply>(out, in, x, U, A, a, b, parity, dagger, comm_override, profile);
    } else {
      errorQuda("Clover Hasensbuch Twist operator has not been built");
    }
  }

} // namespace quda
