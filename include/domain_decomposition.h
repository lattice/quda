#pragma once

#include <fast_intdiv.h>
#include "declare_enum.h"

namespace quda
{

  // using namespace quda;

  DECLARE_ENUM(DD,     // name of the enum class
               in_use, // whether DD is in use

               mode_red_black, // red-black DD, e.g. for SAP
               red_active,     // if red blocks are active
               black_active,   // if black blocks are active
               rb_hopping,     // if hopping between red and black is allowed
               first_black,    // the color of the first block of the local lattice
  );

  // Params for domain decompation
  // we template over Int such that int_fastdiv can be used in kernels
  template <typename Int = int> class DDParam
  {

  public:
    bool flags[(int)DD::size];  // the default value of all flags is 0
    Int blockDim[QUDA_MAX_DIM]; // the size of the block per direction

    // Default constructor
    DDParam() : flags {0}, blockDim {0} { }

    // Default copy
    DDParam(const DDParam &dd) = default;

    // Default copy
    template <typename T> DDParam(const DDParam<T> &dd)
    {
#pragma unroll
      for (auto i = 0u; i < (int)DD::size; i++) flags[i] = dd.flags[i];
#pragma unroll
      for (auto i = 0u; i < QUDA_MAX_DIM; i++) blockDim[i] = dd.blockDim[i];
    }

    // sets the given flag to true
    constexpr inline DDParam &operator&=(const DD &flag)
    {
      flags[(int)flag] = true;
      return *this;
    }

    // returns false if in use
    constexpr inline bool operator!() const { return not flags[(int)DD::in_use]; }

    // returns value of given flag
    constexpr inline bool operator&&(const DD &flag) const { return flags[(int)flag]; }

    constexpr inline bool is(const DD &flag) const { return flags[(int)flag]; }

    // Pretty print the args struct
    void print()
    {
      if (not *this) {
        printfQuda("DDParam not in use\n");
        return;
      }
      printfQuda("Printing DDParam\n");
      for (int i = 0; i < (int)DD::size; i++)
        printfQuda("flags[DD::%s] = %s\n", to_string((DD)i).c_str(), flags[i] ? "true" : "false");
      for (int i = 0; i < QUDA_MAX_DIM; i++) printfQuda("blockDim[%d] = %d\n", i, static_cast<int>(blockDim[i]));
    }

    // template <bool debug = false, typename T> // if true, prints debug information
    inline bool match(const DDParam &dd) const
    {
      bool debug = true;
      // if one of the two is not in use we return true, i.e. one of the two is a full field
      if (not *this or not dd) return true;

      // false if only one is red_black
      if (is(DD::mode_red_black) ^ dd.is(DD::mode_red_black)) {
        if (debug) printfQuda("Only one of the two is red_black\n");
        return false;
      }

      if (is(DD::mode_red_black) and dd.is(DD::mode_red_black))
        for (int i = 0; i < QUDA_MAX_DIM; i++)
          if (blockDim[i] != dd.blockDim[i]) {
            if (debug)
              printfQuda("blockDim[%d] = %d != %d \n", i, static_cast<int>(blockDim[i]),
                         static_cast<int>(dd.blockDim[i]));
            return false;
          }

      return true;
    }

    /* if a field is zero at given coordinates */
    template <typename Coord, typename Arg> constexpr inline bool isZero(const Coord &x, const Arg &arg) const
    {

      // if DD not in use we return immidiatelly
      if (not is(DD::in_use)) return false;

      if (is(DD::mode_red_black)) {

        // Computing block_parity: 0 = red, 1 = black
        int block_parity = is(DD::first_black);
        for (int i = 0; i < x.size(); i++) { block_parity += x[i] / blockDim[i]; }
        block_parity %= 2;

        // Checking if my parity is active
        if (not is(DD::red_active) and block_parity == 0) return true;
        if (not is(DD::black_active) and block_parity == 1) return true;
      }

      return false;
    }

    /* if hopping is allowed */
    template <typename Coord, typename Arg>
    constexpr inline bool doHopping(const Coord &x, const int &mu, const int &dir, const Arg &arg) const
    {

      // if DD not in use we return immidiatelly
      if (not is(DD::in_use)) return true;

      if (is(DD::mode_red_black)) {

        // Checking if we are on the border
        bool on_border = (dir > 0) ? ((x[mu] + 1) % blockDim[mu] == 0) : (x[mu] % blockDim[mu] == 0);

        // If all the follwing are true then it is a full operator
        if (is(DD::red_active) and is(DD::black_active)) {
          if (not on_border || is(DD::rb_hopping)) return true;
          return false;
        }

        // Computing block_parity: 0 = red, 1 = black
        int block_parity = is(DD::first_black) + on_border;
        for (int i = 0; i < x.size(); i++) { block_parity += x[i] / blockDim[i]; }
        block_parity %= 2;

        // Checking if my parity is active
        if (is(DD::red_active) and block_parity == 0) return true;
        if (is(DD::black_active) and block_parity == 1) return true;
      }

      return false;
    }
  };

} // namespace quda
