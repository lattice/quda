#pragma once

#ifndef _NVHPC_CUDA

#include "../generic/thread_array.h"

#else

#include <array.h>

namespace quda
{
  template <typename T, int n> struct thread_array : array<T, n> {
    static constexpr unsigned int shared_mem_size(dim3 block) { return 0; }
  };
} // namespace quda

#endif
