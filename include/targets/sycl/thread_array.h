#pragma once

#define SLM

#ifdef SLM

#include <../generic/thread_array.h>

#else

namespace quda
{

  template <typename T, int n> struct thread_array : array<T, n> {};

} // namespace quda

#endif
