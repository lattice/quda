#pragma once

#ifdef QUDA_OMPTARGET_THREAD_ARRAY_SIMPLE

namespace quda
{
  template <typename T, int n> struct thread_array {
    using value_type = T;
    static constexpr int N = n;
    T data[n];

    constexpr inline T &operator[](int i) { return data[i]; }
    constexpr inline const T &operator[](int i) const { return data[i]; }
  };
} // namespace quda

#else

#include "../generic/thread_array.h"

#endif
