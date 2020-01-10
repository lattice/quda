#pragma once

#include <quda_matrix.h>
#include <complex_quda.h>

namespace quda {

  /**
     @brief MatrixTile is a fragment of a matrix held in registers.
   */
  template <typename T, int m, int n, bool ghost>
  struct MatrixTile {
    using real = typename RealType<T>::type;
    T tile[m*n];
    inline __device__ __host__ MatrixTile()
    {
#pragma unroll
      for (int i=0; i<m; i++) {
#pragma unroll
        for (int j=0; j<n; j++) {
          tile[i*n+j] = 0.0;
        }
      }
    }

    inline __device__ __host__ const T& operator()(int i, int j) const
    {
      return tile[i*n+j];
    }

    inline __device__ __host__ T& operator()(int i, int j)
    {
      return tile[i*n+j];
    }

    template <int k, bool ghost_a, bool ghost_b>
    inline __device__ __host__ void mma_nn(const MatrixTile<T, m, k, ghost_a> &a, const MatrixTile<T, k, n, ghost_b> &b)
    {
#pragma unroll
      for (int i = 0; i < m; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
#pragma unroll
          for (int l = 0; l < k; l++) {
            tile[i*n+j] = cmac(a.tile[i*k+l], b.tile[l*n+j], tile[i*n+j]);
          }
        }
      }
    }

    template <int k, bool ghost_a, bool ghost_b>
    inline __device__ __host__ void mma_nt(const MatrixTile<T, m, k, ghost_a> &a, const MatrixTile<T, n, k, ghost_b> &b)
    {
#pragma unroll
      for (int i = 0; i < m; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
#pragma unroll
          for (int l = 0; l < k; l++) {
            tile[i*n+j] = cmac(a.tile[i*k+l], conj(b.tile[j*k+l]), tile[i*n+j]);
          }
        }
      }
    }

    template <typename Accessor> inline __device__ __host__ void load(const Accessor &a, int d, int parity, int x_cb, int i0, int j0)
    {
#pragma unroll
      for (int i=0; i<m; i++) {
#pragma unroll
        for (int j=0; j<n; j++) {
          if (ghost)
            tile[i*n+j] = a.Ghost(d, parity, x_cb, i0 + i, j0 + j);
          else
            tile[i*n+j] = a(d, parity, x_cb, i0 + i, j0 + j);
        }
      }
    }

    template <typename Accessor> inline __device__ __host__ void load(const Accessor &a, int d, int parity, int x_cb, int si, int sj, int i0, int j0)
    {
#pragma unroll
      for (int i=0; i<m; i++) {
#pragma unroll
        for (int j=0; j<n; j++) {
          if (ghost)
            tile[i*n+j] = a.Ghost(d, parity, x_cb, si, sj, i0 + i, j0 + j);
          else
            tile[i*n+j] = a(d, parity, x_cb, si, sj, i0 + i, j0 + j);
        }
      }
    }

    template <typename Accessor> inline __device__ __host__ void save(Accessor &a, int d, int parity, int x_cb, int i0, int j0)
    {
#pragma unroll
      for (int i = 0; i < m; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
          if (ghost)
            a.Ghost(d, parity, x_cb, i0 + i, j0 + j) = tile[i*n+j];
          else
            a(d, parity, x_cb, i0 + i, j0 + j) = tile[i*n+j];
        }
      }
    }

    template <typename Accessor> inline __device__ __host__ void save(Accessor &a, int d, int parity, int x_cb, int si, int sj, int i0, int j0)
    {
#pragma unroll
      for (int i = 0; i < m; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
          if (ghost)
            a.Ghost(d, parity, x_cb, si, sj, i0 + i, j0 + j) = tile[i*n+j];
          else
            a(d, parity, x_cb, si, sj, i0 + i, j0 + j) = tile[i*n+j];
        }
      }
    }

    template <typename Accessor> inline __device__ __host__ void loadCS(const Accessor &a, int d, int dir, int parity, int x_cb, int s, int i0, int j0)
    {
#pragma unroll
      for (int i=0; i<m; i++) {
#pragma unroll
        for (int j=0; j<n; j++) {
          if (ghost)
            tile[i*n+j] = a.Ghost(d, dir, parity, x_cb, s, i0 + i, j0 + j);
          else
            tile[i*n+j] = a(parity, x_cb, s, i0 + i, j0 + j);
        }
      }
    }

    template <typename Accessor> inline __device__ __host__ void saveCS(Accessor &a, int d, int dir, int parity, int x_cb, int s, int i0, int j0)
    {
#pragma unroll
      for (int i=0; i<m; i++) {
#pragma unroll
        for (int j=0; j<n; j++) {
          if (ghost)
            a.Ghost(d, dir, parity, x_cb, s, i0 + i, j0 + j) = tile[i*n+j];
          else
            a(parity, x_cb, s, i0 + i, j0 + j) = tile[i*n+j];
        }
      }
    }

    inline __device__ __host__ real abs_max()
    {
      real maxTile[m*n];
#pragma unroll
      for (int i = 0; i < m; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
          maxTile[i*n+j] = fmax(fabs(tile[i*n+j].real()), fabs(tile[i*n+j].imag()));
        }
      }
        
      real max = 0.0;
#pragma unroll
      for (int i = 0; i < m; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
          max = fmax(max, maxTile[i*n+j]);
        }
      }
      return max;
    }
  };

}
