#pragma once

#include "dslash_coarse.hpp"
#include "dslash_coarse_decl.hpp"

namespace quda
{

#define INSTANTIATE_DSLASH_COARSE(dagger, coarseColor, Float, yFloat, ghostFloat)                                      \
  template void ApplyCoarse_t<dagger, coarseColor, Float, yFloat, ghostFloat>(                                         \
    cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,     \
    const GaugeField &, const GaugeField &, real_t, int, bool, bool, DslashType, MemoryLocation *,                     \
    const ColorSpinorField &);

} // namespace quda
