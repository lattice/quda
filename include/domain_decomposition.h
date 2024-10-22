#pragma once

#include "declare_enum.h"

namespace quda
{

  // using namespace quda;

  DECLARE_ENUM(DD, // name of the enum class

               reset, // No domain decomposition. It sets all flags to zero.

               red_black_type,   // Flags used by red_black
               red_active,       // if red blocks are active
               black_active,     // if black blocks are active
               no_block_hopping, // if hopping between red and black is allowed
  );

  // Params for domain decompation
  struct DDParam {

    QudaDDType type = QUDA_DD_NO;
    array<bool, static_cast<int>(DD::size)> flags = {}; // the default value of all flags is 0
    array<int, QUDA_MAX_DIM> block_dim = {};            // the size of the block per direction

    // Default constructor
    DDParam() = default;

    // returns false if in use
    constexpr bool operator!() const { return type == QUDA_DD_NO; }

    // returns value of given flag
    constexpr bool is(const DD &flag) const { return flags[(int)flag]; }

    // sets given flag to true
    constexpr void set(const DD &flag)
    {
      flags[(int)flag] = true;

      if ((int)flag == (int)DD::reset) {
#pragma unroll
        for (auto i = 0u; i < (int)DD::size; i++) flags[i] = 0;
        type = QUDA_DD_NO;
      } else if ((int)flag >= (int)DD::red_black_type) {
        type = QUDA_DD_RED_BLACK;
      }
    }

    template <typename... Args> constexpr void set(const DD &flag, const Args &...args)
    {
      set(flag);
      set(args...);
    }

    // Pretty print the args struct
    void print() const
    {
      if (not *this) {
        printfQuda("DD not in use\n");
        return;
      }
      printfQuda("Printing DDParam\n");
      for (int i = 0; i < (int)DD::size; i++)
        printfQuda("flags[DD::%s] = %s\n", to_string((DD)i).c_str(), flags[i] ? "true" : "false");
      for (int i = 0; i < QUDA_MAX_DIM; i++) printfQuda("block_dim[%d] = %d\n", i, static_cast<int>(block_dim[i]));
    }

    // Checks if this matches to given DDParam
    template <typename F> inline bool check(const F &field, bool verbose = false) const
    {
      if (not *this) return true;

      if (type == QUDA_DD_RED_BLACK) {
        for (int i = 0; i < field.Ndim(); i++) {
          if (block_dim[i] < 0) {
            if (verbose) printfQuda("block_dim[%d] = %d is negative\n", i, block_dim[i]);
            return false;
          }
          if (block_dim[i] > 0) {
            int globalDim = comm_dim(i) * field.full_dim(i);
            if (globalDim % block_dim[i] != 0) {
              if (verbose) printfQuda("block_dim[%d] = %d does not divide %d \n", i, block_dim[i], globalDim);
              return false;
            }
            if ((globalDim / block_dim[i]) % 2 != 0) {
              if (verbose)
                printfQuda("block_dim[%d] = %d does not divide %d **evenly** \n", i, block_dim[i], globalDim);
              return false;
            }
          }
        }
        if (block_dim[0] % 2) {
          if (verbose) printfQuda("block_dim[0] = %d must be even \n", block_dim[0]);
          return false;
        }
      }

      return true;
    }

    // Checks if this matches to given DDParam
    inline bool match(const DDParam &dd, bool verbose = false) const
    {
      // if one of the two is not in use we return true, i.e. one of the two is a full field
      if (not *this or not dd) return true;

      // false if type does not match
      if (type != dd.type) {
        if (verbose) printfQuda("DD type do not match (%d != %d)\n", type, dd.type);
        return false;
      }

      if (type == QUDA_DD_RED_BLACK) {
        for (int i = 0; i < QUDA_MAX_DIM; i++)
          if (block_dim[i] != dd.block_dim[i]) {
            if (verbose) printfQuda("block_dim[%d] = %d != %d \n", i, block_dim[i], dd.block_dim[i]);
            return false;
          }
        if (is(DD::no_block_hopping) != dd.is(DD::no_block_hopping)) {
          if (verbose) printfQuda("no_block_hopping do not match.\n");
          return false;
        }
      }

      return true;
    }

    // Checks if this is equal to given DDParam
    inline bool operator==(const DDParam &dd) const
    {
      // if both are not in use we return true
      if (not *this and not dd) return true;

      // false if type does not match
      if (type != dd.type) return false;

      // checking all flags matches (note this should be actually type-wise)
      for (int i = 0; i < (int)DD::size; i++)
        if (flags[i] != dd.flags[i]) return false;

      // checking block_dim matches when needed
      if (type == QUDA_DD_RED_BLACK)
        for (int i = 0; i < QUDA_MAX_DIM; i++)
          if (block_dim[i] != dd.block_dim[i]) return false;

      return true;
    }

    inline bool operator!=(const DDParam &dd) const { return !(*this == dd); }
  };

} // namespace quda