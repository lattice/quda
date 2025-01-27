#pragma once

#include <quda_define.h>

#if (__COMPUTE_CAPABILITY__ >= 900) && (CUDA_VERSION >= 12060)
#define USE_TENSOR_MEMORY_ACCELERATOR
#endif

#ifdef USE_TENSOR_MEMORY_ACCELERATOR
#include <cuda.h>
#include <unordered_map>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

namespace quda
{

  struct tma_descriptor_t {
    CUtensorMap map;
  };

  /**
    @brief The key and the values that defines an TMA destriptor
    @param kRank Number of dimensions for the tensor
    @param tensor_dims The dimensions of the tensor as it resides in the global memory space
    @param box_dims The "box dimensions", or the dimensions of the shape that is to be loaded from
      global memory to shared memory
    @param ptr The global memory pointer
   */
  template <int kRank> struct tma_descriptor_key_t {
    std::array<size_t, kRank> tensor_dims;
    std::array<size_t, kRank> box_dims;
    void *ptr;

    bool operator==(const tma_descriptor_key_t &other) const
    {
      for (size_t i = 0; i < kRank; i++) {
        if (tensor_dims[i] != other.tensor_dims[i] || box_dims[i] != other.box_dims[i]) { return false; }
      }
      if (ptr != other.ptr) { return false; }
      return true;
    }
  };

  template <int kRank> struct tma_descriptor_hash_t {
    std::size_t operator()(const tma_descriptor_key_t<kRank> &key) const
    {
      std::size_t hash = 0;
      for (size_t i = 0; i < kRank; i++) {
        hash = (hash << 1) ^ std::hash<std::size_t> {}(key.tensor_dims[i]);
        hash = (hash << 1) ^ std::hash<std::size_t> {}(key.box_dims[i]);
      }
      hash = (hash << 1) ^ std::hash<void *> {}(key.ptr);
      return hash;
    }
  };

  /**
    @brief Make a TMA descriptor out of the input key: in general the tensor strides can be non-trivial,
      i.e., the tensor in global memory can have non-zero paddings for each dimension, but here we
      always assume the paddings are zero; the box sizes cannot have paddings; for now we do not use
      any non-trivial swizzle/interleave patterns, nor any non-trivial element stride.
    @param key The input key
   */
  template <class T, int kRank> tma_descriptor_t make_tma_descriptor(tma_descriptor_key_t<kRank> key)
  {
    CUtensorMap ret_value;

    cuuint64_t tensor_size[kRank];
    for (int i = 0; i < kRank; i++) { tensor_size[i] = static_cast<cuuint64_t>(key.tensor_dims[i]); }
    cuuint64_t tensor_stride[kRank - 1];
    tensor_stride[0] = tensor_size[0] * sizeof(T);
    for (int i = 1; i < kRank - 1; i++) { tensor_stride[i] = tensor_stride[i - 1] * tensor_size[i]; }
    cuuint32_t box_size[kRank];
    for (int i = 0; i < kRank; i++) { box_size[i] = static_cast<cuuint32_t>(key.box_dims[i]); }
    cuuint32_t elem_str[kRank];
    for (int i = 0; i < kRank; i++) { elem_str[i] = 1; }

    CUtensorMapDataType data_type;
    if constexpr (std::is_same_v<T, float>) {
      data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same_v<T, short>) {
      data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else {
      errorQuda("Unexpected data type for TMA descriptor creation.");
    }

    CUresult error = cuTensorMapEncodeTiled(&ret_value, data_type, kRank, key.ptr, tensor_size, tensor_stride, box_size,
                                            elem_str, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (CUDA_SUCCESS != error) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("TMA descriptor creation returned %s", str);
    }

    return {ret_value};
  }

  /**
    @brief A helper function that implements a naive software cache for the TMA descriptors
    @param key The input key
   */
  template <class T, int kRank> tma_descriptor_t get_tma_descriptor(tma_descriptor_key_t<kRank> key)
  {
    static std::unordered_map<tma_descriptor_key_t<kRank>, tma_descriptor_t, tma_descriptor_hash_t<kRank>> _cache;
    if (_cache.find(key) == _cache.end()) { _cache[key] = make_tma_descriptor<T, kRank>(key); }
    return _cache[key];
  }

  constexpr int keep_divide_by_two_until(int i, int limit)
  {
    if (i <= limit) {
      return i;
    } else {
      return keep_divide_by_two_until(i / 2, limit);
    }
  };

  /**
    @brief TMA box dimensions has a hard limit of 256 for each of the dimensions
   */
  constexpr int tma_box_limit = 256;

  /**
    @brief TMA box sizes have to be less than or equal to 256;
      For now we only consider 2-d box shapes. This is the specialization for 5-d tensors.
    @param box_a Box shape dimension[0]
    @param box_b Box shape dimension[1]
    @param T The element type
    @param tensor_size Tensor dimensions
    @param ptr The tensor pointer in global memory
   */
  template <int box_a, int box_b, class T>
  inline tma_descriptor_t get_tma_descriptor_5d_box_2d(std::array<size_t, 5> tensor_size, complex<T> *ptr)
  {
    static_assert(box_a <= tma_box_limit);
    static_assert(box_b <= tma_box_limit);
    tma_descriptor_key_t<5> key = {tensor_size, std::array<size_t, 5> {box_a, box_b, 1, 1, 1}, ptr};
    return get_tma_descriptor<T, 5>(key);
  }

  /**
    @brief TMA box sizes have to be less than or equal to 256;
      For now we only consider 2-d box shapes. This is the specialization for 4-d tensors.
    @param box_a Box shape dimension[0]
    @param box_b Box shape dimension[1]
    @param T The element type
    @param tensor_size Tensor dimensions
    @param ptr The tensor pointer in global memory
   */
  template <int box_a, int box_b, class T>
  inline tma_descriptor_t get_tma_descriptor_4d_box_2d(std::array<size_t, 4> tensor_size, complex<T> *ptr)
  {
    static_assert(box_a <= tma_box_limit);
    static_assert(box_b <= tma_box_limit);
    tma_descriptor_key_t<4> key = {tensor_size, std::array<size_t, 4> {box_a, box_b, 1, 1}, ptr};
    return get_tma_descriptor<T, 4>(key);
  }

  /**
    @brief Launch TMA load from a 5-d tensor in global memory to a 2-d box in shared memory.
    @param smem_ptr The destination shared memory pointer
    @param map Points to the TMA descriptor
    @param offset_a Offset[0] of the tensor with which this load is to be invoked
    @param offset_b Offset[1] of the tensor with which this load is to be invoked
    @param offset_c Offset[2] of the tensor with which this load is to be invoked
    @param offset_d Offset[3] of the tensor with which this load is to be invoked
    @param offset_e Offset[4] of the tensor with which this load is to be invoked
    @param bar The barrier object that is to be used for the TMA load
   */
  template <int box_a, int box_b, class T>
  __device__ void inline tma_load_gmem_5d_box_2d(complex<T> *smem_ptr, const CUtensorMap *map, int offset_a,
                                                 int offset_b, int offset_c, int offset_d, int offset_e, barrier_t *bar)
  {
    static_assert(box_a <= tma_box_limit);
    static_assert(box_b <= tma_box_limit);
    cde::cp_async_bulk_tensor_5d_global_to_shared(smem_ptr, map, offset_a, offset_b, offset_c, offset_d, offset_e, *bar);
  }

  /**
    @brief Launch TMA load from a 4-d tensor in global memory to a 2-d box in shared memory.
    @param smem_ptr The destination shared memory pointer
    @param map Points to the TMA descriptor
    @param offset_a Offset[0] of the tensor with which this load is to be invoked
    @param offset_b Offset[1] of the tensor with which this load is to be invoked
    @param offset_c Offset[2] of the tensor with which this load is to be invoked
    @param offset_d Offset[3] of the tensor with which this load is to be invoked
    @param bar The barrier object that is to be used for the TMA load
   */
  template <int box_a, int box_b, class T>
  __device__ void inline tma_load_gmem_4d_box_2d(complex<T> *smem_ptr, const CUtensorMap *map, int offset_a,
                                                 int offset_b, int offset_c, int offset_d, barrier_t *bar)
  {
    static_assert(box_a <= tma_box_limit);
    static_assert(box_b <= tma_box_limit);
    cde::cp_async_bulk_tensor_4d_global_to_shared(smem_ptr, map, offset_a, offset_b, offset_c, offset_d, *bar);
  }

} // namespace quda

#endif
