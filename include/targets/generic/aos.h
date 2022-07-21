#pragma once

namespace quda
{

  /**
     @brief Load n-length block of memory of type T and return in local array
     @tparam T Array element type
     @tparam n Number of elements in the structure
     @param[out] out Output array
     @param[in] in Input memory pointer we are block loading from
   */
  template <typename T, int n> __host__ __device__ void block_load(T out[n], const T *in)
  {
#pragma unroll
    for (int i = 0; i < n; i++) out[i] = in[i];
  }

  /**
     @brief Store n-length array of type T in block of memory
     @tparam T Array element type
     @tparam n Number of elements in the array
     @param[out] out Output memory pointer we are block storing to
     @param[in] in Input array
   */
  template <typename T, int n> __host__ __device__ void block_store(T *out, const T in[n])
  {
#pragma unroll
    for (int i = 0; i < n; i++) out[i] = in[i];
  }

  /**
     @brief Load type T from contiguous memory
     @tparam T Element type
     @param[out] out Output value
     @param[in] in Input memory pointer we are loading from
  */
  template <typename T> __host__ __device__ void block_load(T &out, const T *in) { out = *in; }

  /**
     @brief Store type T in contiguous memory
     @tparam Element type
     @param[out] out Output memory pointer we are storing to
     @param[in] in Input value
  */
  template <typename T> __host__ __device__ void block_store(T *out, const T &in) { *out = in; }

} // namespace quda
