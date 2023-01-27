#include <dslash_coarse.hpp>
#include <dslash_coarse_mma.hpp>

namespace quda {

  constexpr bool dagger = @QUDA_MULTIGRID_DAGGER@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC@;
  constexpr bool use_mma = true;

  template<>
  void ApplyCoarse<dagger, coarseColor, use_mma>(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                                        cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                                        double kappa, int parity, bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      // create a halo ndim+1 field for batched comms
      auto halo = ColorSpinorField::create_comms_batch(inA);

      DslashCoarseLaunch<dagger, coarseColor, use_mma> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash,
                                                     clover, commDim, halo_precision);

      DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
      policy.apply(device::get_default_stream());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
