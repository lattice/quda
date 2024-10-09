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
    template <typename DDArg, typename Arg> constexpr bool commDim(int, const DDArg &, const Arg &) const
    {
      return true;
    }

    // Whether field at given coord is zero
    template <typename Coord> constexpr bool isZero(const Coord &) const { return false; }

    // Whether do hopping with field at neighboring coord
    template <typename Coord> constexpr bool doHopping(const Coord &, int, int) const { return true; }
  };

  // Red-black Block DD
  struct DDRedBlack {

    const int_fastdiv block_dim[QUDA_MAX_DIM]; // the size of the block per direction
    const bool red_active;         // if red blocks are active
    const bool black_active;       // if black blocks are active
    const bool block_hopping;      // if hopping between red and black is allowed

    DDRedBlack(const DDParam &dd) :
      block_dim {dd.block_dim[0], dd.block_dim[1], dd.block_dim[2], dd.block_dim[3]},
      red_active(dd.type == QUDA_DD_NO or dd.is(DD::red_active)),
      black_active(dd.type == QUDA_DD_NO or dd.is(DD::black_active)),
      block_hopping(dd.type == QUDA_DD_NO or not dd.is(DD::no_block_hopping))
    {
      if (dd.type != QUDA_DD_NO and dd.type != QUDA_DD_RED_BLACK) { errorQuda("Unsupported type %d", dd.type); }
    }

    constexpr bool operator!() const { return false; }

    // Whether comms are required along given direction
    template <typename DDArg, typename Arg> constexpr bool commDim(int d, const DDArg &dd, const Arg &arg) const
    {
      if (not red_active and not black_active) return false;
      if (not dd.red_active and not dd.black_active) return false;
      if (arg.dim[d] % block_dim[d] == 0) {
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
      for (int i = 0; i < x.size(); i++) {
        if (block_dim[i] > 0) block_parity += x.gx[i] / block_dim[i];
      }
      return block_parity % 2 == 1;
    }

    template <typename Coord> constexpr bool on_border(const Coord &x, int mu, int dir) const
    {
      if (block_dim[mu] == 0) return false;
      int x_mu = x.gx[mu] + dir;
      if (x_mu < 0) x_mu += x.gDim[mu];
      if (x_mu >= x.gDim[mu]) x_mu -= x.gDim[mu];
      return x.gx[mu] / block_dim[mu] != x_mu / block_dim[mu];
    }

    template <typename Coord> constexpr bool isZero(const Coord &x) const
    {
      bool is_black = block_parity(x);
      bool is_red = not is_black;

      if (is_red and red_active) return false;
      if (is_black and black_active) return false;
      return true;
    }

    template <typename Coord> constexpr bool doHopping(const Coord &x, int mu, int dir) const
    {
      bool is_black = block_parity(x);
      bool is_red = !is_black;
      bool is_border = on_border(x, mu, dir);

      if (!is_border) { // Within block
        if (is_red and red_active) return true;
        if (is_black and black_active) return true;
      } else if (block_hopping) { // Between blocks
        if (is_red and black_active) return true;
        if (is_black and red_active) return true;
      }
      return false;
    }
  };

} // namespace quda
