#pragma once

namespace quda {

  /**
     struct which acts as a wrapper to a vector of data.
   */
  template <typename scalar, int n>
  struct vector_type {
    scalar data[n];
    __device__ __host__ inline scalar& operator[](int i) { return data[i]; }
    __device__ __host__ inline const scalar& operator[](int i) const { return data[i]; }
    __device__ __host__ inline static constexpr int size() { return n; }
    __device__ __host__ inline void operator+=(const vector_type &a) {
#pragma unroll
      for (int i=0; i<n; i++) data[i] += a[i];
    }
    __device__ __host__ inline void operator=(const vector_type &a) {
#pragma unroll
      for (int i=0; i<n; i++) data[i] = a[i];
    }
    __device__ __host__ vector_type() {
#pragma unroll
      for (int i=0; i<n; i++) zero(data[i]);
    }
  };

  template<typename scalar, int n>
  __device__ __host__ inline void zero(vector_type<scalar,n> &v) {
#pragma unroll
    for (int i=0; i<n; i++) zero(v.data[i]);
  }

  template<typename scalar, int n>
  __device__ __host__ inline vector_type<scalar,n> operator+(const vector_type<scalar,n> &a, const vector_type<scalar,n> &b) {
    vector_type<scalar,n> c;
#pragma unroll
    for (int i=0; i<n; i++) c[i] = a[i] + b[i];
    return c;
  }

} // namespace quda

#ifdef QUDA_TARGET_CUDA
#include <cub_helper.cuh>
#endif
#ifdef QUDA_TARGET_CPU
#include <reduce_helper_cpu.h>
#endif
