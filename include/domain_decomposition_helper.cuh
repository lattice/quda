#pragma once

#include <fast_intdiv.h>
#include <color_spinor_field.h>

namespace quda
{

  // No DD (use also as a template for required functions)
  struct DDNo {

    // Initialization of input parameters from ColorSpinorField
    DDNo(const DDParam &dd)
    {
      if (dd.type != QUDA_DD_NO) { errorQuda("Unsupported type %d\n", dd.type); }
    }

    // Only DDNo returns true. All others return false
    constexpr bool operator!() const { return true; }

    // Whether comms are required along given direction
    template <typename DDArg, typename Arg> constexpr bool commDim(const int &, const DDArg &, const Arg &) const
    {
      return true;
    }

    // Whether field at given coord is zero
    template <typename Coord> constexpr bool isZero(const Coord &) const { return false; }

    // Whether do hopping with field at neighboring coord
    template <typename Coord> constexpr bool doHopping(const Coord &, const int &, const int &) const { return true; }
  };

  // Red-black Block DD
  struct DDRedBlack {

    const int_fastdiv blockDim[QUDA_MAX_DIM]; // the size of the block per direction
    const bool red_active;         // if red blocks are active
    const bool black_active;       // if black blocks are active
    const bool block_hopping;      // if hopping between red and black is allowed

    DDRedBlack(const DDParam &dd) :
      blockDim {dd.blockDim[0], dd.blockDim[1], dd.blockDim[2], dd.blockDim[3]},
      red_active(dd.type == QUDA_DD_NO or dd.is(DD::red_active)),
      black_active(dd.type == QUDA_DD_NO or dd.is(DD::black_active)),
      block_hopping(dd.type == QUDA_DD_NO or not dd.is(DD::no_block_hopping))
    {
      if (dd.type != QUDA_DD_NO and dd.type != QUDA_DD_RED_BLACK) { errorQuda("Unsupported type %d\n", dd.type); }
    }

    constexpr bool operator!() const { return false; }

    // Whether comms are required along given direction
    template <typename DDArg, typename Arg> constexpr bool commDim(const int &d, const DDArg &dd, const Arg &arg) const
    {
      if (not red_active and not black_active) return false;
      if (not dd.red_active and not dd.black_active) return false;
      if (arg.dim[d] % blockDim[d] == 0) {
        if (not red_active and not dd.red_active) return false;
        if (not black_active and not dd.black_active) return false;
        if (not block_hopping and not dd.block_hopping) return false;
      }
      return true;
    }

    // Computes block_parity: 0 = red, 1 = black
    template <typename Coord> constexpr bool block_parity(const Coord &x) const
    {
      int block_parity = 0;
      for (int i = 0; i < x.size(); i++) { block_parity += x.gx[i] / blockDim[i]; }
      return block_parity % 2 == 1;
    }

    template <typename Coord> constexpr bool on_border(const Coord &x, const int &mu, const int &dir) const
    {
      return (dir > 0) ? ((x.gx[mu] + 1) % blockDim[mu] == 0) : (x.gx[mu] % blockDim[mu] == 0);
    }

    template <typename Coord> constexpr bool isZero(const Coord &x) const
    {
      if (red_active and black_active) return false;
      if (not red_active and not black_active) return true;

      bool is_black = block_parity(x);

      // Checking if my parity is active
      if (red_active and not is_black) return false;
      if (black_active and is_black) return false;

      return true;
    }

    template <typename Coord> constexpr bool doHopping(const Coord &x, const int &mu, const int &dir) const
    {
      if (red_active and black_active and block_hopping) return true;
      if (not red_active and not black_active) return false;

      bool swap = on_border(x, mu, dir);
      if (swap and not block_hopping) return false;
      if (not swap and red_active and black_active) return true;

      // Neighbor color
      bool is_black = block_parity(x) ^ swap;

      // Checking if neighbor is active
      if (red_active and not is_black) return true;
      if (black_active and is_black) return true;

      return false;
    }
  };

} // namespace quda
