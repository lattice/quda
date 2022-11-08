#include <dslash_coarse.hpp>

namespace quda {

  // dagger = true wrapper
  void ApplyCoarseDagger(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                         cvector_ref<const ColorSpinorField> &inB, const ColorSpinorField &halo,
                         const GaugeField &Y, const GaugeField &X, double kappa, int parity,
                         bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      constexpr bool dagger = true;
      DslashCoarseLaunch<dagger> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash,
                                        clover, commDim, halo_precision);

      DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
      policy.apply(device::get_default_stream());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
