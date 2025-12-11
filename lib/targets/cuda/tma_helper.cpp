#include <cuda.h>
#include <tma_helper.hpp>
#include <map>

#ifdef USE_TENSOR_MEMORY_ACCELERATOR

namespace quda
{

  auto create_descriptor(const GaugeField &u, uint32_t block_size)
  {
    auto precision = u.Precision();
    auto reconstruct = u.Reconstruct();
    auto stride = u.Stride();
    auto geometry = u.Geometry();

    auto get_tensor_data_type = [&](size_t word_size) {
      switch (word_size) {
      case 1: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
      case 2: return CU_TENSOR_MAP_DATA_TYPE_UINT16;
      case 4: return CU_TENSOR_MAP_DATA_TYPE_UINT32;
      case 8: return CU_TENSOR_MAP_DATA_TYPE_UINT64;
      default: errorQuda("Unsupported word size %d", precision);
      }
      return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    };

    auto hasPhase = reconstruct == 9 || reconstruct == 13;
    uint32_t N = gauge::get_vector_order(precision, reconstruct - hasPhase);
    uint32_t M = (reconstruct - hasPhase) / N;
    uint32_t Nrem = reconstruct - hasPhase - M * N;

    CUtensorMapDataType dtype = get_tensor_data_type(precision);
    gauge::tensor_desc_t tensor;

    {
      if (stride % 16 != 0) errorQuda("Volume requirements not met: stride mod 16 = %lu", stride % 16);
      uint64_t global_dim[] = {16llu * N, uint64_t(stride / 16), uint64_t(M), uint64_t(geometry), 2llu};
      uint64_t global_stride[]
        = {precision * 16llu * N, precision * stride * N, precision * stride * (N * M + Nrem), u.Bytes() / 2};
      uint32_t box_dim[] = {16u * N, std::max(1u, block_size / 16), M, 1, 1};
      uint32_t element_stride[] = {1, 1, 1, 1, 1};
      auto data = u.data();
      if (reinterpret_cast<uintptr_t>(data) % 16 != 0) errorQuda("Pointer is not 16-byte aligned");
      auto res = cuTensorMapEncodeTiled(&tensor.N.map, dtype, 5, data, global_dim, global_stride, box_dim,
                                        element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (res != CUDA_SUCCESS) {
        const char *errStr = nullptr;
        cuGetErrorString(res, &errStr);
        errorQuda("cuTensorMapEncodeTiled failed: %s", errStr);
      }
    }

    if (Nrem > 0) {
      if (stride % 16 != 0) errorQuda("Volume requirements not met: stride mod 16 = %lu", stride % 16);
      uint64_t global_dim[]
        = {16llu * Nrem, uint64_t(stride / 16), uint64_t(geometry), 2llu}; // can remove the M dimension?
      uint64_t global_stride[] = {precision * 16llu * Nrem, precision * stride * (N * M + Nrem), u.Bytes() / 2};
      uint32_t box_dim[] = {16u * Nrem, std::max(1u, block_size / 16), 1, 1, 1};
      uint32_t element_stride[] = {1, 1, 1, 1};
      auto data = u.data<char *>() + M * N * stride * precision;
      if (reinterpret_cast<uintptr_t>(data) % 16 != 0) errorQuda("Pointer is not 16-byte aligned");
      auto res = cuTensorMapEncodeTiled(&tensor.Nrem.map, dtype, 4, data, global_dim, global_stride, box_dim,
                                        element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (res != CUDA_SUCCESS) {
        const char *errStr = nullptr;
        cuGetErrorString(res, &errStr);
        errorQuda("cuTensorMapEncodeTiled failed: %s box = {%u, %u, %u, %u}", errStr, box_dim[0], box_dim[1],
                  box_dim[2], box_dim[3]);
      }
    }

    if (hasPhase) {
      if (stride % 16 != 0) errorQuda("Volume requirements not met: stride mod 16 = %lu", stride % 16);
      uint64_t global_dim[] = {16llu, uint64_t(stride / 16), uint64_t(geometry), 2llu};
      uint64_t global_stride[] = {precision * 16llu, precision * stride, u.Bytes() / 2};
      uint32_t box_dim[] = {16u, std::max(1u, block_size / 16u), 1, 1};
      uint32_t element_stride[] = {1, 1, 1, 1};
      auto data = u.data<char *>() + u.PhaseOffset();
      if (reinterpret_cast<uintptr_t>(data) % 16 != 0) errorQuda("Pointer is not 16-byte aligned");
      auto res = cuTensorMapEncodeTiled(&tensor.phase.map, dtype, 4, data, global_dim, global_stride, box_dim,
                                        element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      if (res != CUDA_SUCCESS) {
        const char *errStr = nullptr;
        cuGetErrorString(res, &errStr);
        errorQuda("cuTensorMapEncodeTiled failed: %s box = {%u, %u, %u, %u}", errStr, box_dim[0], box_dim[1],
                  box_dim[2], box_dim[3]);
      }
    }

    return tensor;
  }

  static std::map<int, gauge::tensor_desc_t> tensor_map;

  gauge::tensor_desc_t &get_tensor_descriptor(const GaugeField &u, uint32_t block_size)
  {
    auto tensor = tensor_map.find(block_size);
    if (tensor != tensor_map.end()) {
      return tensor->second;
    } else {
      tensor_map[block_size] = create_descriptor(u, block_size);
    }
    return tensor_map[block_size];
  }

} // namespace quda

#endif
