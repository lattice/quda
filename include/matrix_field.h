#pragma once

/**
 * @field matrix_accessor.h
 * @brief Simple accessor used for matrix fields, e.g., each lattice
 * site consists of an n x n matrix
 */

// trove requires the warp shuffle instructions introduced with Kepler
#if __COMPUTE_CAPABILITY__ >= 300
#include <trove/ptr.h>
#else
#define DISABLE_TROVE
#endif

#include <quda_matrix.h>

namespace quda
{

  template <typename T, int n> struct matrix_field {
    T *field;
    int volume_cb;

    matrix_field(T *field, int volume_cb) : field(field), volume_cb(volume_cb) {}

    __device__ __host__ inline void load(Matrix<T, n> &A, int x_cb, int parity) const
    {
      int idx = parity * volume_cb + x_cb;
#ifdef __CUDA_ARCH__
      const trove::coalesced_ptr<Matrix<T, n>> field_((Matrix<T, n> *)field);
      A = field_[idx];
#else
#pragma unroll
      for (int i = 0; i < n; i++)
#pragma unroll
        for (int j = 0; j < n; j++) A(i, j) = field[(n * idx + i) * n + j] = A(i, j);
#endif
    }

    __device__ __host__ inline void save(const Matrix<T, n> &A, int x_cb, int parity)
    {
      int idx = parity * volume_cb + x_cb;
#ifdef __CUDA_ARCH__
      trove::coalesced_ptr<Matrix<T, n>> field_((Matrix<T, n> *)field);
      field_[idx] = A;
#else
#pragma unroll
      for (int i = 0; i < n; i++)
#pragma unroll
        for (int j = 0; j < n; j++) field[(n * idx + i) * n + j] = A(i, j);
#endif
    }
  };

} // namespace quda
