#pragma once

/**
 * @field matrix_accessor.h
 * @brief Simple accessor used for matrix fields, e.g., each lattice
 * site consists of an n x n matrix
 */

#include <aos.h>
#include <convert.h>
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

    /**
       @brief Store a register matrix in the field's storage type.
       store_cast renormalizes fp32mp2_low before the write. When the
       storage type is the native double (fp32mp2), the site stays in
       that encoding instead of being converted to IEEE.
     */
    template <typename U> __device__ __host__ inline void save(const Matrix<U, n> &A, int x_cb, int parity) const
    {
      using out_real = typename RealType<T>::type;
      Matrix<T, n> out;
#pragma unroll
      for (int i = 0; i < n; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
          out(i, j) = T(store_cast<out_real>(A(i, j).real()), store_cast<out_real>(A(i, j).imag()));
        }
      }
      int idx = parity * volume_cb + x_cb;
      block_store(reinterpret_cast<Matrix<T, n> *>(field) + idx, out);
    }
  };

} // namespace quda
