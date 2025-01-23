#pragma once

namespace quda
{

  /**
    @brief This is a dummy struct that wraps around a list of integers
   */
  template <int... Ints> struct IntList {
  };

} // namespace quda
