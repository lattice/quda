#pragma once

#include "block_orthogonalize.hpp"
#include "block_orthogonalize_decl.hpp"

namespace quda
{

#define INSTANTIATE_BLOCK_ORTHOGONALIZE(fineColor, coarseColor, vFloat, bFloat)                                        \
  template void BlockOrthogonalize_t<fineColor, coarseColor, vFloat, bFloat>(                                          \
    ColorSpinorField &, const std::vector<ColorSpinorField> &, const int *, const int *, const int *, int, int, bool);

} // namespace quda
