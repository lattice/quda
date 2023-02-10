#include <dslash_coarse.hpp>
#include <dslash_coarse_mma.hpp>

namespace quda {

  constexpr bool dagger = @QUDA_MULTIGRID_DAGGER@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC@;
  constexpr bool use_mma = true;
  constexpr int nVec = @QUDA_MULTIGRID_MRHS@;

  template <typename Float, typename yFloat, typename ghostFloat, int Ns, bool dslash, bool clover,
            DslashType type>
  using D = DslashCoarseMma<Float, yFloat, ghostFloat, Ns, coarseColor, dslash, clover, dagger, type, nVec>;

  template<>
  void ApplyCoarseMma<dagger, coarseColor, nVec>(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &inA,
                                        cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y, const GaugeField &X,
                                        double kappa, int parity, bool dslash, bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      if constexpr (coarseColor == 24 || coarseColor == 32 || coarseColor == 64 || coarseColor == 96) {
        // create a halo ndim+1 field for batched comms
        auto halo = ColorSpinorField::create_comms_batch(inA);

        DslashCoarseLaunch<D, dagger, coarseColor, use_mma, nVec> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash,
                                                       clover, commDim, halo_precision);

        DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
        policy.apply(device::get_default_stream());
      } else {
        errorQuda("coarseColor = %d is not supported by MMA.\n", coarseColor);
      }
    } else {
      errorQuda("Multigrid has not been built");
    }
  }

} // namespace quda
