#pragma once

#include "kernel.h"

namespace quda {

  /**
     @brief Parameter structure for driving init_dslash_atomic
  */
  template <typename T_> struct init_dslash_atomic_arg : kernel_param<> {
    using T = T_;
    T *count;

    init_dslash_atomic_arg(T *count, unsigned int size) :
      kernel_param(dim3(size, 1, 1)),
      count(count) { }
  };

  /**
     @brief Functor that uses placement new constructor to initialize
     the atomic counters
  */
  template <typename Arg> struct init_dslash_atomic {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr init_dslash_atomic(const Arg &arg) : arg(arg) { }
    __device__ void operator()(int i) { new (arg.count + i) typename Arg::T {0}; }
  };

  /**
     @brief Parameter structure for driving init_dslash_arr
  */
  template <typename T_> struct init_arr_arg : kernel_param<> {
    using T = T_;
    T *arr;
    T val;

    init_arr_arg(T *arr, T val, unsigned int size) :
      kernel_param(dim3(size, 1, 1)),
      arr(arr),
      val(val) { }
  };

  /**
     @brief Functor to initialize the arrive signal
  */
  template <typename Arg> struct init_sync_arr {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr init_sync_arr(const Arg &arg) : arg(arg) { }
    __device__ void operator()(int i) { *(arg.arr + i) = arg.val; }
  };

}
