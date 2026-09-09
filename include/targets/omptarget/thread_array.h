#pragma once

#include <kernel_ops.h>

#ifdef QUDA_OMPTARGET_THREAD_ARRAY_SIMPLE

namespace quda
{
  template <typename T, int n, typename O = void> struct thread_array {
    using value_type = T;
    static constexpr int N = n;
    T data[n] {};

    template <typename... U> constexpr thread_array(const KernelOps<U...> &ops)
    {
      checkKernelOps<thread_array<T, n, O>>(ops);
    }

    template <typename... Arg> static constexpr unsigned int shared_mem_size(dim3, const Arg &...) { return 0; }

    constexpr inline T &operator[](int i) { return data[i]; }
    constexpr inline const T &operator[](int i) const { return data[i]; }
  };
} // namespace quda

#else

#include "../generic/thread_array.h"

#endif
