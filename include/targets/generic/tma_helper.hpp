#pragma once

namespace quda
{

  enum class PrefetchType { NONE, THREAD, BULK, TENSOR };

  struct tma_descriptor_t {
  };

  namespace gauge
  {
    struct tensor_desc_t {
      tma_descriptor_t N;
      tma_descriptor_t Nrem;
      tma_descriptor_t phase;
    };
  } // namespace gauge

  inline gauge::tensor_desc_t get_tensor_descriptor(const GaugeField &, uint32_t) { return gauge::tensor_desc_t {}; }

} // namespace quda
