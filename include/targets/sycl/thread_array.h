#pragma once

//#define SLM

#ifdef SLM

#include <../generic/thread_array.h>

#else

namespace quda
{
  template <typename T, int n> struct thread_array : array<T, n> {
    //constexpr thread_array() : array<T,n>{} {}
    template <typename Ops> constexpr thread_array(Ops &ops) : array<T,n>{} {}
    static constexpr unsigned int shared_mem_size(dim3) { return 0; }
  };
} // namespace quda

#endif

namespace quda {
  template <typename T, int n> static constexpr bool needsFullBlock<thread_array<T,n>> = false;
}
