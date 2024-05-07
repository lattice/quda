#pragma once

#include "declare_enum.h"

namespace quda
{

  // using namespace quda;

  DECLARE_ENUM(DD, // name of the enum class

               red_black_type, // Flags used by red_black
               red_active,     // if red blocks are active
               black_active,   // if black blocks are active
               block_hopping,  // if hopping between red and black is allowed
  );

  // Params for domain decompation
  struct DDParam {

    QudaDDType type;
    bool flags[(int)DD::size];  // the default value of all flags is 0
    int blockDim[4];            // the size of the block per direction

    // Default constructor
    DDParam() : type(QUDA_DD_NO), flags {0}, blockDim {0} { }

    // Default copy
    DDParam(const DDParam &dd) = default;

    // returns false if in use
    constexpr inline bool operator!() const { return type == QUDA_DD_NO; }

    // returns value of given flag
    constexpr inline bool is(const DD &flag) const { return flags[(int)flag]; }

    // sets given flag to true
    constexpr inline void set(const DD &flag)
    {
      flags[(int)flag] = true;

      if ((int)flag >= (int)DD::red_black_type) type = QUDA_DD_RED_BLACK;
    }

    template <typename... Args> constexpr inline void set(const DD &flag, Args... args)
    {
      set(flag);
      set(args...);
    }

    constexpr inline void reset()
    {
      type = QUDA_DD_NO;
#pragma unroll
      for (auto i = 0u; i < (int)DD::size; i++) flags[i] = 0;
    }

    template <typename... Args> constexpr inline void reset(Args... args)
    {
      reset();
      set(DD::in_use, args...);
    }

    // Pretty print the args struct
    void print()
    {
      if (not *this) {
        printfQuda("DD not in use\n");
        return;
      }
      printfQuda("Printing DDParam\n");
      for (int i = 0; i < (int)DD::size; i++)
        printfQuda("flags[DD::%s] = %s\n", to_string((DD)i).c_str(), flags[i] ? "true" : "false");
      for (int i = 0; i < QUDA_MAX_DIM; i++) printfQuda("blockDim[%d] = %d\n", i, static_cast<int>(blockDim[i]));
    }

    // Checks if this matches to given DDParam
    // template <bool debug = false, typename T> // if true, prints debug information
    inline bool match(const DDParam &dd) const
    {
      bool debug = true;
      // if one of the two is not in use we return true, i.e. one of the two is a full field
      if (not *this or not dd) return true;

      // false if type does not match
      if (type != dd.type) {
        if (debug) printfQuda("DD type do not match (%d != %d)\n", type, dd.type);
        return false;
      }

      if (type == QUDA_DD_RED_BLACK)
        for (int i = 0; i < QUDA_MAX_DIM; i++)
          if (blockDim[i] != dd.blockDim[i]) {
            if (debug) printfQuda("blockDim[%d] = %d != %d \n", i, blockDim[i], dd.blockDim[i]);
            return false;
          }

      return true;
    }
  };

} // namespace quda
