#pragma once

#include <array.h>

namespace quda
{

  /**
   * @brief compute number of elements of an array containing powers
   * of 2 starting at a minumum up to and including a maximum
   *
   */
  template <unsigned int Min, unsigned int Max> constexpr unsigned int numElements() noexcept
  {
    unsigned int i = 0;
    for (auto j = Min; j <= Max; j *= 2) i++;
    return i;
  }

  /**
   * @brief A struct containing a compile time generated array
   * containing powers of 2 starting at Min up to and includeing Max
   * with thanks to StackOverflow:
   *   https://stackoverflow.com/questions/19019252/create-n-element-constexpr-array-in-c11
   */
  template <unsigned int Min, unsigned int Max> struct PowerOfTwoArray {

    array<unsigned int, numElements<Min, Max>()> data_;

    constexpr PowerOfTwoArray() : data_()
    {
      static_assert(Min <= Max, "Min has to be <= Max");
      for (unsigned int i = 0, j = Min; j <= Max; j *= 2, i++) data_[i] = j;
    }

    /**
     * @brief returns the size of the array
     */
    constexpr unsigned int size() const noexcept { return numElements<Min, Max>(); }

    /**
     * @brief read only constant index operator[]
     * @param i the index to look up
     */
    constexpr unsigned int operator[](int i) const noexcept { return data_[i]; }

  }; // end struct

} // namespace quda
