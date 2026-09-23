@QUDA_COARSE_FILE_BEGIN@
#include <quda_arch.h>
#include "dslash_coarse_decl.hpp"

#if defined(QUDA_MMA_AVAILABLE)                                                                                        \
  && ((@QUDA_MULTIGRID_NVEC@ == 24) || (@QUDA_MULTIGRID_NVEC@ == 32) || (@QUDA_MULTIGRID_NVEC@ == 64)               \
      || (@QUDA_MULTIGRID_NVEC@ == 96))
#include "dslash_coarse_mma_inst.hpp"
#endif

namespace quda
{

  constexpr bool dagger = @QUDA_MULTIGRID_DAGGER@;
  constexpr int coarseColor = @QUDA_MULTIGRID_NVEC@;
  constexpr int nVec = @QUDA_MULTIGRID_MRHS@;
  using Float = @QUDA_COARSE_FLOAT@;
  using yFloat = @QUDA_COARSE_YFLOAT@;
  using ghostFloat = @QUDA_COARSE_GHOSTFLOAT@;

#if defined(QUDA_MMA_AVAILABLE)                                                                                        \
  && ((@QUDA_MULTIGRID_NVEC@ == 24) || (@QUDA_MULTIGRID_NVEC@ == 32) || (@QUDA_MULTIGRID_NVEC@ == 64)               \
      || (@QUDA_MULTIGRID_NVEC@ == 96))

  INSTANTIATE_DSLASH_COARSE_MMA(dagger, coarseColor, nVec, Float, yFloat, ghostFloat)

#else

  template <>
  void ApplyCoarseMma_t<dagger, coarseColor, nVec, Float, yFloat, ghostFloat>(
    cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
    const GaugeField &, const GaugeField &, real_t, int, bool, bool, DslashType, MemoryLocation *,
    const ColorSpinorField &)
  {
    errorQuda("coarseColor = %d is not supported by MMA.\n", coarseColor);
  }

#endif

} // namespace quda
@QUDA_COARSE_FILE_END@
