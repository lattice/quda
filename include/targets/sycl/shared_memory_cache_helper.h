#pragma once

#include <../generic/shared_memory_cache_helper.h>

namespace quda {
  template <typename T, typename D, typename O> static constexpr bool needsFullBlockImpl<SharedMemoryCache<T,D,O>> = true;
}
