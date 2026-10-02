#pragma once

#include "dslash_coarse.hpp"
#include "dslash_coarse_mma.hpp"
#include "dslash_coarse_decl.hpp"

namespace quda
{

#define INSTANTIATE_DSLASH_COARSE_MMA(dagger, coarseColor, nVec, Float, yFloat, ghostFloat)                            \
  template void ApplyCoarseMma_t<dagger, coarseColor, nVec, Float, yFloat, ghostFloat>(                                \
    cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,     \
    const GaugeField &, const GaugeField &, real_t, int, bool, bool, DslashType, MemoryLocation *,                     \
    const ColorSpinorField &);

} // namespace quda
