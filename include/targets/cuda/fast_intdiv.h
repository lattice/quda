#pragma once

// declaration of class we wish to specialize
template <bool> struct mul_hi;

template <> struct mul_hi<true> {
  __device__ __forceinline__ int operator()(const int n, const int m)
  {
    int q;
    asm("mul.hi.s32 %0, %1, %2;" : "=r"(q) : "r"(m), "r"(n));
    return q;
  }
};

#include "../generic/fast_intdiv.h"
