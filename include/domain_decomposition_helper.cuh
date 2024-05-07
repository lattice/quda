#pragma once

#include <fast_intdiv.h>
#include <color_spinor_field.h>

namespace quda
{

  // No DD (use also as a template for required functions)
  struct DDNo {

    // Initialization of input parameters from ColorSpinorField
    DDNo(const ColorSpinorField &in)
    {
      if (in.dd.type != QUDA_DD_NO) { errorQuda("Unsupported type %d\n", in.dd.type); }
    }

    // Whether field at given coord is zero
    template <typename Coord> constexpr inline bool isZero(const Coord &) const { return false; }

    // Whether do hopping with field at neighboring coord
    template <typename Coord> constexpr inline bool doHopping(const Coord &, const int &, const int &) const
    {
      return true;
    }
  };

  // Red-black Block DD
  struct DDRedBlack {

    const int_fastdiv blockDim[4]; // the size of the block per direction
    const bool red_active;         // if red blocks are active
    const bool black_active;       // if black blocks are active
    const bool block_hopping;      // if hopping between red and black is allowed
    const bool first_black;        // if the first block of the local lattice is black instead of red

    DDRedBlack(const ColorSpinorField &in) :
      blockDim {in.dd.blockDim[0], in.dd.blockDim[1], in.dd.blockDim[2], in.dd.blockDim[3]},
      red_active(in.dd.type == QUDA_DD_NO or in.dd.is(DD::red_active)),
      black_active(in.dd.type == QUDA_DD_NO or in.dd.is(DD::black_active)),
      block_hopping(in.dd.type == QUDA_DD_NO or in.dd.is(DD::block_hopping)),
      first_black(false) // TODO
    {
      if (in.dd.type != QUDA_DD_NO and in.dd.type != QUDA_DD_RED_BLACK) {
        errorQuda("Unsupported type %d\n", in.dd.type);
      }
    }

    // Computes block_parity: 0 = red, 1 = black
    template <typename Coord> inline bool block_parity(const Coord &x) const
    {
      int block_parity = first_black;
      for (int i = 0; i < x.size(); i++) { block_parity += x[i] / blockDim[i]; }
      return block_parity % 2 == 0;
    }

    template <typename Coord> inline bool on_border(const Coord &x, const int &mu, const int &dir) const
    {
      return (dir > 0) ? ((x[mu] + 1) % blockDim[mu] == 0) : (x[mu] % blockDim[mu] == 0);
    }

    template <typename Coord> inline bool isZero(const Coord &x) const
    {
      if (red_active and black_active) return false;
      if (not red_active and not black_active) return true;

      bool is_black = block_parity(x);

      // Checking if my parity is active
      if (red_active and not is_black) return false;
      if (black_active and is_black) return false;

      return true;
    }

    template <typename Coord> inline bool doHopping(const Coord &x, const int &mu, const int &dir) const
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
