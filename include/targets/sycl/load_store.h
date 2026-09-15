#pragma once

namespace quda
{

  /**
     @brief Element type used for coalesced storage.
   */
  template <typename T>
  using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;

} // namespace quda

#include "../generic/load_store.h"
