#pragma once

#include <fast_intdiv.h>
#include <color_spinor_field.h>

namespace quda
{

  // Template structure
  struct DDArg {

    // Initialization of input parameters from ColorSpinorField
    virtual DDArg(const ColorSpinorField &in);

    // Whether field at given coord is zero
    template <typename Coord> virtual inline bool isZero(const Coord &x) const;

    // Whether do hopping with field at neighboring coord
    template <typename Coord> virtual inline bool doHopping(const Coord &x, const int &mu, const int &dir) const;
  };

  struct DDNo : public DDArg {

    DDArg(const ColorSpinorField &in) { assert(in.dd.type == QUDA_DD_NO); }

    constexpr inline bool isZero(const Coord &x) const { return false; }

    constexpr inline bool doHopping(const Coord &x, const int &mu, const int &dir) const { return true; }
  };

  struct DDRedBlack : public DDArg {

    const int_fastdiv blockDim[4]; // the size of the block per direction
    const bool red_active;         // if red blocks are active
    const bool black_active;       // if black blocks are active
    const bool block_hopping;      // if hopping between red and black is allowed
    const bool first_black;        // if the first block of the local lattice is black instead of red

    DDArg(const ColorSpinorField &in) :
      blockDim {in.dd.blockDim(0), in.dd.blockDim(1), in.dd.blockDim(2), in.dd.blockDim(3)},
      red_active(in.dd.type == QUDA_DD_NO or in.dd.is(DD::red_active)),
      black_active(in.dd.type == QUDA_DD_NO or in.dd.is(DD::black_active)),
      block_hopping(in.dd.type == QUDA_DD_NO or in.dd.is(DD::block_hopping)),
      first_black(false) // TODO
    {
      assert(in.dd.type == QUDA_DD_NO or in.dd.type == QUDA_DD_RED_BLACK);
    }

    // Computes block_parity: 0 = red, 1 = black
    inline bool block_parity(const Coord &x) const
    {
      int block_parity = first_black;
      for (int i = 0; i < x.size(); i++) { block_parity += x[i] / blockDim[i]; }
      return block_parity % 2 == 0;
    }

    inline bool on_border(const Coord &x, const int &mu, const int &dir) const
    {
      return (dir > 0) ? ((x[mu] + 1) % blockDim[mu] == 0) : (x[mu] % blockDim[mu] == 0);
    }

    inline bool isZero(const Coord &x) const
    {
      if (red_active and black_active) return false;
      if (not red_active and not black_active) return true;

      bool is_black = block_parity(x);

      // Checking if my parity is active
      if (red_active and not is_black) return false;
      if (black_active and is_black) return false;

      return true;
    }

    inline bool doHopping(const Coord &x, const int &mu, const int &dir) const
    {
      if (red_active and black_active and blockhopping) return true;
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
