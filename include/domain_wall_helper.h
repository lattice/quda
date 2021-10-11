#pragma once

namespace quda
{

  enum Dslash5Type {
    DSLASH5_DWF,
    DSLASH5_MOBIUS_PRE,
    DSLASH5_MOBIUS,
    M5_INV_DWF,
    M5_INV_MOBIUS,
    M5_INV_ZMOBIUS,
    M5_EOFA,
    M5INV_EOFA
  };

  /**
    Applying the following five kernels in the order of 4-0-1-2-3 is equivalent to applying
    the full even-odd preconditioned symmetric MdagM operator:
    op = (1 - M5inv * D4 * D5pre * M5inv * D4 * D5pre)^dag
        * (1 - M5inv * D4 * D5pre * M5inv * D4 * D5pre)
  */
  enum class MdwfFusedDslashType {
    D4_D5INV_D5PRE,
    D4_D5INV_D5INVDAG,
    D4DAG_D5PREDAG_D5INVDAG,
    D4DAG_D5PREDAG,
    D5PRE,
  };

} // namespace quda
