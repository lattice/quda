#pragma once

#if 1

#include "../generic/thread_local_cache.h"

#else

namespace quda
{

  template <typename T, typename O>
  class ThreadLocalCacheBase<T,O,true> : public ThreadLocalCacheDefault<T,O> {
  public:
    using dependentOps = NoSpecialOps;
    using base_type = ThreadLocalCacheDefault<T, O>;
    __device__ __host__ inline ThreadLocalCacheBase() { }
    template <typename Ops> __device__ __host__ inline ThreadLocalCacheBase(const Ops &ops) : base_type(ops) {}
  };

}

#endif
