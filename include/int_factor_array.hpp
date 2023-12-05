#pragma once

#include <array.h>

namespace quda
{

  inline unsigned int numFactors(unsigned int Int)
  {
    unsigned int i = 0;
    for (unsigned int j = 1u; j <= Int; j++) {
      if (Int % j == 0) { i++; }
    }
    return i;
  }

  /**
   * @brief A struct containing a compile time generated array
   * containing factors of an integer.
   */
  inline auto get_int_factor_array(unsigned int Int)
  {
    std::vector<unsigned int> _out(numFactors(Int));
    unsigned int i = 0;
    for (unsigned int j = 1u; j <= Int; j++) {
      if (Int % j == 0) {
        _out[i] = j;
        i++;
      }
    }
    return _out;
  }

  /**
   * @brief compute number of factors of an integer
   *
   */
  template <unsigned int Int> constexpr unsigned int numFactors() noexcept
  {
    unsigned int i = 0;
    for (unsigned int j = 1u; j <= Int; j++) {
      if (Int % j == 0) { i++; }
    }
    return i;
  }

  /**
   * @brief A struct containing a compile time generated array
   * containing factors of an integer.
   */
  template <unsigned int Int> struct IntFactorArray {

    array<unsigned int, numFactors<Int>()> data_;

    constexpr IntFactorArray() : data_()
    {
      static_assert(Int > 0, "Int has to be > 0");
      for (unsigned int i = 0, j = 1; j <= Int; j++) {
        if (Int % j == 0) {
          data_[i] = j;
          i++;
        }
      }
    }

    /**
     * @brief returns the size of the array
     */
    constexpr unsigned int size() const noexcept { return numFactors<Int>(); }

    /**
     * @brief read only constant index operator[]
     * @param i the index to look up
     */
    constexpr unsigned int operator[](int i) const noexcept { return data_[i]; }

  }; // end struct
} // namespace quda
