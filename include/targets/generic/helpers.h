#pragma once

namespace quda
{

  template <typename T>
  using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;

  template <int N> struct SizeStatic {
    static constexpr unsigned int size(dim3 block) {
      return N;
    }
  };

  template <int N> struct SizePerThread {
    static constexpr unsigned int size(dim3 block) {
      return N * block.x * block.y * block.z;
    }
  };

  template <typename D, int N = 1> struct SizeDims {
    static constexpr unsigned int size(dim3 block) {
      dim3 dims = D::dims(block);
      return dims.x * dims.y * dims.z * N;
    }
  };

  struct DimsBlock {
    static constexpr dim3 dims(dim3 block) {
      return block;
    }
  };

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or some wrapper of T
   */
  template <class T, class enable = void> struct get_type {
    using type = T;
  };

}
