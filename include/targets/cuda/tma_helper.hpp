#include <cuda.h>
#include <unordered_map>

namespace quda {

  struct tma_descriptor_t {
    CUtensorMap map;
  };

  template <int kRank>
  struct tma_descriptor_key_t {
    std::array<size_t, kRank> tensor_dims;
    std::array<size_t, kRank> box_dims;
    void *ptr;

    bool operator==(const tma_descriptor_key_t &other) const
    {
      for (size_t i = 0; i < kRank; i++) {
        if (tensor_dims[i] != other.tensor_dims[i] || box_dims[i] != other.box_dims[i]) {
          return false;
        }
      }
      if (ptr != other.ptr) { return false; }
      return true;
    }
  };

  template <int kRank>
  struct tma_descriptor_hash_t {
    std::size_t operator()(const tma_descriptor_key_t<kRank>& key) const
    {
      std::size_t hash = 0;
      for (size_t i = 0; i < kRank; i++) {
        hash = (hash << 1) ^ std::hash<std::size_t>{}(key.tensor_dims[i]);
        hash = (hash << 1) ^ std::hash<std::size_t>{}(key.box_dims[i]);
      }
      hash = (hash << 1) ^ std::hash<void *>{}(key.ptr);
      return hash;
    }
  };

  template <class T, int kRank>
  tma_descriptor_t make_tma_descriptor(tma_descriptor_key_t<kRank> key)
  {
    CUtensorMap ret_value;

    cuuint64_t tensor_size[kRank];
    for (int i = 0; i < kRank; i++)
    {
      tensor_size[i] = static_cast<cuuint64_t>(key.tensor_dims[i]);
    }
    cuuint64_t tensor_stride[kRank - 1];
    tensor_stride[0] = tensor_size[0] * sizeof(T);
    for (int i = 1; i < kRank - 1; i ++) {
      tensor_stride[i] = tensor_stride[i - 1] * tensor_size[i];
    }
    cuuint32_t box_size[kRank];
    for (int i = 0; i < kRank; i++)
    {
      box_size[i] = static_cast<cuuint32_t>(key.box_dims[i]);
    }
    cuuint32_t elem_str[kRank];
    for (int i = 0; i < kRank; i++)
    {
      elem_str[i] = 1;
    }

    CUtensorMapDataType data_type;
    if constexpr (std::is_same_v<T, float>) {
      data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same_v<T, short>) {
      data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else {
      errorQuda("Unexpected data type for TMA descriptor creation.");
    }

    if (CUDA_SUCCESS != cuTensorMapEncodeTiled(
      &ret_value,
      data_type,
      kRank,
      key.ptr,
      tensor_size,
      tensor_stride,
      box_size,
      elem_str,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) {

      errorQuda("TMA descriptor creation failed.");
    }

    return {ret_value};
  }

  template <class T, int kRank>
  tma_descriptor_t get_tma_descriptor(tma_descriptor_key_t<kRank> key)
  {
    static std::unordered_map<tma_descriptor_key_t<kRank>, tma_descriptor_t, tma_descriptor_hash_t<kRank>> _cache;
    if (_cache.find(key) == _cache.end()) {
      _cache[key] = make_tma_descriptor<T, kRank>(key);
    }
    return _cache[key];
  }

}
