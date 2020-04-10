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

    template <typename T_> inline __device__ __host__ void operator*=(const T_ &a)
    {
#pragma unroll
      for (int i=0; i<m; i++) {
#pragma unroll
        for (int j=0; j<n; j++) {
          tile[i*n+j] *= a;
        }
      }
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

    template <int k, bool ghost_a, bool ghost_b>
    inline __device__ __host__ void mma_tn(const MatrixTile<T, k, m, ghost_a> &a, const MatrixTile<T, k, n, ghost_b> &b)
    {
#pragma unroll
      for (int i = 0; i < m; i++) {
#pragma unroll
        for (int j = 0; j < n; j++) {
#pragma unroll
          for (int l = 0; l < k; l++) {
            tile[i*n+j] = cmac(conj(a.tile[l*m+i]), b.tile[l*n+j], tile[i*n+j]);
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

  template <int m_, int n_, int k_, int M_, int N_, int K_> struct TileSize {
    static constexpr int m = m_;
    static constexpr int n = n_;
    static constexpr int k = k_;
    static constexpr int M = M_;
    static constexpr int N = N_;
    static constexpr int K = K_;
    static constexpr int M_tiles = m / M;
    static constexpr int N_tiles = n / N;
    static constexpr int K_tiles = k / K;

    static_assert(M > m == 0, "tile height must not be larger than matrix height");
    static_assert(N > n == 0, "tile width must not be larger than matrix width");
    static_assert(K > n == 0, "tile depth must not be larger than matrix depth");
    static_assert(m % M == 0, "tile height must be an integer divisor of the matrix height");
    static_assert(n % N == 0, "tile width must be an integer divisor of the matrix width");
    static_assert(k % K == 0, "tile depth must be an integer divisor of the matrix depth");
  };

  template <int m_, int n_, int k_, int M_, int N_, int K_>
  std::ostream& operator<<(std::ostream &out, const TileSize<m_,n_,k_,M_,N_,K_> &tile)
  {
    out << "Matrix size = (" << tile.m << ", " << tile.n << ", " << tile.k << ")" << std::endl;
    out << "Tile size = (" << tile.M << ", " << tile.N << ", " << tile.K << ")" << std::endl;
    out << "Number of tiles = (" << tile.M_tiles << ", " << tile.N_tiles << ", " << tile.K_tiles << ")" << std::endl;
    return out;
  }

  /** @brief Helper for creating an A tile (MxK) */
  template <typename T, bool ghost, typename Tile> __device__ __host__  inline auto make_tile_A(const Tile tile)
  {
    return MatrixTile<T, tile.M, tile.K, ghost>();
  }

  /** @brief Helper for creating an A transpose tile (KxM) */
  template <typename T, bool ghost, typename Tile> __device__ __host__  inline auto make_tile_At(const Tile tile)
  {
    return MatrixTile<T, tile.K, tile.M, ghost>();
  }

  /** @brief Helper for creating a B tile (KxN) */
  template <typename T, bool ghost, typename Tile> __device__ __host__  inline auto make_tile_B(const Tile tile)
  {
    return MatrixTile<T, tile.K, tile.N, ghost>();
  }

  /** @brief Helper for creating a B transpose tile (NxK) */
  template <typename T, bool ghost, typename Tile> __device__ __host__  inline auto make_tile_Bt(const Tile tile)
  {
    return MatrixTile<T, tile.N, tile.K, ghost>();
  }

  /** @brief Helper for creating a C tile (MxN) */
  template <typename T, bool ghost, typename Tile> __device__ __host__  inline auto make_tile_C(const Tile tile)
  {
    return MatrixTile<T, tile.M, tile.N, ghost>();
  }

  /** @brief Helper for creating a C transpose tile (NxM) */
  template <typename T, bool ghost, typename Tile> __device__ __host__  inline auto make_tile_Ct(const Tile tile)
  {
    return MatrixTile<T, tile.N, tile.M, ghost>();
  }

}
