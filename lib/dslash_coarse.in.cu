#include <dslash_coarse.hpp>

namespace quda {

  constexpr bool dagger = @QUDA_MULTIGRID_DAGGER@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC@;

  template<>
  void ApplyCoarse<dagger, coarseColor>(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                                        cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                                        real_t kappa, int parity, bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      // create a halo ndim+1 field for batched comms
      auto halo = ColorSpinorField::create_comms_batch(inA);

      DslashCoarseLaunch<dagger, coarseColor> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash,
                                                     clover, commDim, halo_precision);

      DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
      policy.apply(device::get_default_stream());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
