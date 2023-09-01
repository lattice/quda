#pragma once

#ifndef _NVHPC_CUDA

#include "../generic/thread_array.h"

#else

#include <array.h>

namespace quda
{
  template <typename T, int n> struct thread_array : array<T, n> {};
}

#endif
