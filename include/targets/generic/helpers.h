#pragma once

namespace quda
{

  template <typename T>
  using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;

  template <int N> struct SizeStatic {
    static constexpr unsigned int size(dim3) {
      return N;
    }
  };

  template <int N> struct SizePerThread {
    static constexpr unsigned int size(dim3 block) {
      return N * block.x * block.y * block.z;
    }
  };

  struct SizeBlockDivWarp {
    static constexpr unsigned int size(dim3 b) {
      return (b.x * b.y * b.z + device::warp_size() - 1)/device::warp_size();
    }
  };

  struct SizeZ {
    static constexpr unsigned int size(dim3 block) {
      return block.z;
    }
  };

  template <typename D, int N = 1> struct SizeDims {
    static constexpr unsigned int size(dim3 block) {
      dim3 dims = D::dims(block);
      return dims.x * dims.y * dims.z * N;
    }
  };

  template <typename S> struct SizeSmem {
    template <typename T> static constexpr size_t size(dim3 block) {
      return S::shared_mem_size(block);
    }
  };

  struct DimsBlock {
    static constexpr dim3 dims(dim3 block) {
      return block;
    }
  };

  template <int x, int y, int z>
  struct DimsStatic {
    static constexpr dim3 dims(dim3) {
      return dim3(x,y,z);
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
