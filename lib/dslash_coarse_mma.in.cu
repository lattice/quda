#include <dslash_coarse.hpp>
#include <dslash_coarse_mma.hpp>
#include <quda_arch.h>

namespace quda
{

  constexpr bool dagger = @QUDA_MULTIGRID_DAGGER@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC@;
  constexpr int nVec = @QUDA_MULTIGRID_MRHS@;

  template <typename Float, typename yFloat, typename ghostFloat, int Ns, bool dslash, bool clover, DslashType type>
  using D = DslashCoarseMma<Float, yFloat, ghostFloat, Ns, coarseColor, dslash, clover, dagger, type, nVec>;

#if defined(QUDA_MMA_AVAILABLE)                                                                                        \
  && ((@QUDA_MULTIGRID_NVEC@ == 24) || (@QUDA_MULTIGRID_NVEC@ == 32) || (@QUDA_MULTIGRID_NVEC@ == 64)               \
      || (@QUDA_MULTIGRID_NVEC@ == 96))

  constexpr bool use_mma = true;

  template <>
  void ApplyCoarseMma<dagger, coarseColor, nVec>(cvector_ref<ColorSpinorField> &out,
                                                 cvector_ref<const ColorSpinorField> &inA,
                                                 cvector_ref<const ColorSpinorField> &inB, const GaugeField &Y,
                                                 const GaugeField &X, double kappa, int parity, bool dslash,
                                                 bool clover, const int *commDim, QudaPrecision halo_precision)
  {
    if constexpr (is_enabled_multigrid()) {
      // create a halo ndim+1 field for batched comms
      auto halo = ColorSpinorField::create_comms_batch(inA);

      DslashCoarseLaunch<D, dagger, coarseColor, use_mma, nVec> Dslash(out, inA, inB, halo, Y, X, kappa, parity, dslash,
                                                                       clover, commDim, halo_precision);

      DslashCoarsePolicyTune<decltype(Dslash)> policy(Dslash);
      policy.apply(device::get_default_stream());
    } else {
      errorQuda("Multigrid has not been built");
    }
  }
#else
  template <>
  void ApplyCoarseMma<dagger, coarseColor, nVec>(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                                 cvector_ref<const ColorSpinorField> &, const GaugeField &,
                                                 const GaugeField &, double, int, bool, bool, const int *, QudaPrecision)
  {
    errorQuda("coarseColor = %d is not supported by MMA.\n", coarseColor);
  }
#endif

} // namespace quda
