#pragma once

namespace quda
{

  /**
     @brief Used to declare an object of fixed size.
   */
  template <int N> struct SizeStatic {
    static constexpr unsigned int size(dim3) { return N; }
  };

  /**
     @brief Used to declare an object of fixed size per thread, N.
   */
  template <int N> struct SizePerThread {
    static constexpr unsigned int size(dim3 block) { return N * block.x * block.y * block.z; }
  };

  /**
     @brief Used to declare an object of size equal to the size of the block Z dimension.
   */
  struct SizeZ {
    static constexpr unsigned int size(dim3 block) {
      return block.z;
    }
  };

  /**
     @brief Used to declare an object of size equal to the block size divided by the warp size.
   */
  struct SizeBlockDivWarp {
    static constexpr unsigned int size(dim3 b) {
      return (b.x * b.y * b.z + device::warp_size() - 1)/device::warp_size();
    }
  };

  /**
     @brief Used to declare an object of fixed size per thread, N, with thread dimensions derermined by D.
   */
  template <typename D, int N = 1> struct SizeDims {
    static constexpr unsigned int size(dim3 block)
    {
      dim3 dims = D::dims(block);
      return dims.x * dims.y * dims.z * N;
    }
  };

  /**
     @brief Used to declare an object with dimensions given by the block size.
   */
  struct DimsBlock {
    static constexpr dim3 dims(dim3 block) { return block; }
  };

  /**
     @brief Used to declare an object with fixed dimensions.
   */
  template <int x, int y, int z> struct DimsStatic {
    static constexpr dim3 dims(dim3) { return dim3(x, y, z); }
  };

  /**
     @brief Uniform helper for exposing type T, whether we are dealing
     with an instance of T or some wrapper of T
   */
  template <class T, class enable = void> struct get_type {
    using type = T;
  };

} // namespace quda
