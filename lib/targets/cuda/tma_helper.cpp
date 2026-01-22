#include <map>
#include <array>
#include <cuda.h>
#include <tma_helper.hpp>

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

  struct tensor_key_t {
    static constexpr std::size_t volume_n = 32;
    static constexpr std::size_t aux_n = 256;

    uint32_t block_size {};
    std::array<char, volume_n> volume {}; // zero-filled
    std::array<char, aux_n> aux {};       // zero-filled
    void *ptr {};

    bool operator<(const tensor_key_t &other) const noexcept
    {
      if (block_size != other.block_size) return block_size < other.block_size;
      int vc = std::memcmp(volume.data(), other.volume.data(), tensor_key_t::volume_n);
      if (vc != 0) return vc < 0;
      int ac = std::memcmp(aux.data(), other.aux.data(), tensor_key_t::aux_n);
      if (ac != 0) return ac < 0;
      // Required for strict weak ordering on arbitrary pointers
      return std::less<void *> {}(ptr, other.ptr);
    }

    friend std::ostream &operator<<(std::ostream &os, const tensor_key_t &key)
    {
      auto print_buf = [&](auto const &buf) {
        auto end = std::find(buf.begin(), buf.end(), '\0');
        os.write(buf.data(), std::distance(buf.begin(), end));
      };

      os << "block_size=" << key.block_size << ", volume=\"";
      print_buf(key.volume);
      os << "\", aux=\"";
      print_buf(key.aux);
      os << "\", ptr=" << key.ptr;
      return os;
    }
  };

  static std::map<tensor_key_t, gauge::tensor_desc_t> tensor_map;

  gauge::tensor_desc_t &get_tensor_descriptor(const GaugeField &u, uint32_t block_size)
  {
    tensor_key_t key {}; // zero-inits arrays + ptr
    key.block_size = block_size;
    key.ptr = u.data();

    const std::size_t vlen = std::min(u.VolString().size(), tensor_key_t::volume_n);
    const std::size_t alen = std::min(u.AuxString().size(), tensor_key_t::aux_n);

    std::memcpy(key.volume.data(), u.VolString().data(), vlen);
    std::memcpy(key.aux.data(), u.AuxString().data(), alen);

    auto it = tensor_map.find(key);
    if (it != tensor_map.end()) return it->second;

    auto [ins_it, inserted] = tensor_map.emplace(key, create_descriptor(u, block_size));
    return ins_it->second;
  }

} // namespace quda

#endif
