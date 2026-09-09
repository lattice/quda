#pragma once

#include "../generic/thread_local_cache.h"

namespace quda
{
  template <typename T, int N, typename O> inline constexpr bool needsFullBlockImpl<ThreadLocalCache<T, N, O>> = false;
}
