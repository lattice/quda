#pragma once

#if 1
#include "../generic/thread_local_cache.h"
#else
#include "../generic/thread_local_cache_noshared.h"
#endif

namespace quda {
  template <typename T, int N, typename O> static constexpr bool needsFullBlockImpl<ThreadLocalCache<T,N,O>> = false;
}
