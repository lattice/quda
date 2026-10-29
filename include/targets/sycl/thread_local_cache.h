#pragma once

#include "../generic/thread_local_cache.h"

namespace quda
{
  template <typename T, int N, typename O> static constexpr bool needsFullBlockImpl<ThreadLocalCache<T, N, O>> = false;
  template <typename T, int N, typename O> static constexpr bool needsSharedMemImpl<ThreadLocalCache<T, N, O>> = true;
} // namespace quda
