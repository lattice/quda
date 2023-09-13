#pragma once

#define SLM

#ifdef SLM

#include <../generic/thread_array.h>

namespace quda {
  template <typename T, int n> static constexpr bool needsFullBlock<thread_array<T,n>> = false;
}

#else

namespace quda
{

  template <typename T, int n> struct thread_array : array<T, n> {};

} // namespace quda

#endif
