#pragma once

/**
 * @field matrix_accessor.h
 * @brief Simple accessor used for matrix fields, e.g., each lattice
 * site consists of an n x n matrix
 */

#include <aos.h>
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
      block_load(A, reinterpret_cast<const Matrix<T, n> *>(field) + idx);
    }

    __device__ __host__ inline void save(const Matrix<T, n> &A, int x_cb, int parity) const
    {
      int idx = parity * volume_cb + x_cb;
      block_store(reinterpret_cast<Matrix<T, n> *>(field) + idx, A);
    }
  };

} // namespace quda
