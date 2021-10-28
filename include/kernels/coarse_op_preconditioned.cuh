#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <matrix_tile.cuh>

namespace quda {

  template <typename Float_, typename PreconditionedGauge, typename Gauge, typename GaugeInv, int n_, int M_, int N_>
  struct CalculateYhatArg {
    using Float = Float_;
    using yhatTileType = TileSize<n_, n_, n_, M_, N_, 1>;
    yhatTileType tile;

    static constexpr int M = n_;
    static constexpr int N = n_;
    static constexpr int K = n_;

    static constexpr bool is_mma_compatible = PreconditionedGauge::is_mma_compatible;

    PreconditionedGauge Yhat;
    const Gauge Y;
    const GaugeInv Xinv;
    int dim[QUDA_MAX_DIM];
    int comm_dim[QUDA_MAX_DIM];
    int nFace;

    Float *max_h;  // host scalar that stores the maximum element of Yhat. Pointer b/c pinned.
    Float *max_d; // device scalar that stores the maximum element of Yhat

    CalculateYhatArg(const PreconditionedGauge &Yhat, const Gauge Y, const GaugeInv Xinv, const int *dim,
                     const int *comm_dim, int nFace) :
      Yhat(Yhat), Y(Y), Xinv(Xinv), nFace(nFace), max_h(nullptr), max_d(nullptr)
    {
      for (int i=0; i<4; i++) {
        this->comm_dim[i] = comm_dim[i];
        this->dim[i] = dim[i];
      }
    }
  };

  template <bool compute_max_only, typename Arg>
  inline __device__ __host__ auto computeYhat(Arg &arg, int d, int x_cb, int parity, int i0, int j0)
  {
    using real = typename Arg::Float;
    using complex = complex<real>;
    constexpr int nDim = 4;
    int coord[nDim];
    getCoords(coord, x_cb, arg.dim, parity);

    const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

    real yHatMax = 0.0;

    // first do the backwards links Y^{+\mu} * X^{-\dagger}
    if ( arg.comm_dim[d] && (coord[d] - arg.nFace < 0) ) {

      auto yHat = make_tile_C<complex,true>(arg.tile);

#pragma unroll
      for (int k = 0; k < Arg::yhatTileType::k; k += Arg::yhatTileType::K) {
        auto Y = make_tile_A<complex, true>(arg.tile);
        Y.load(arg.Y, d, 1-parity, ghost_idx, i0, k);

        auto X = make_tile_Bt<complex, false>(arg.tile);
        X.load(arg.Xinv, 0, parity, x_cb, j0, k);

        yHat.mma_nt(Y, X);
      }

      if (compute_max_only) {
        yHatMax = yHat.abs_max();
      } else {
        yHat.save(arg.Yhat, d, 1 - parity, ghost_idx, i0, j0);
      }

    } else {
      const int back_idx = linkIndexM1(coord, arg.dim, d);

      auto yHat = make_tile_C<complex,false>(arg.tile);

#pragma unroll
      for (int k = 0; k < Arg::yhatTileType::k; k += Arg::yhatTileType::K) {
        auto Y = make_tile_A<complex, false>(arg.tile);
        Y.load(arg.Y, d, 1-parity, back_idx, i0, k);

        auto X = make_tile_Bt<complex, false>(arg.tile);
        X.load(arg.Xinv, 0, parity, x_cb, j0, k);

        yHat.mma_nt(Y, X);
      }
      if (compute_max_only) {
        yHatMax = yHat.abs_max();
      } else {
        yHat.save(arg.Yhat, d, 1 - parity, back_idx, i0, j0);
      }
    }

    { // now do the forwards links X^{-1} * Y^{-\mu}
      auto yHat = make_tile_C<complex, false>(arg.tile);

#pragma unroll
      for (int k = 0; k < Arg::yhatTileType::k; k += Arg::yhatTileType::K) {
        auto X = make_tile_A<complex, false>(arg.tile);
        X.load(arg.Xinv, 0, parity, x_cb, i0, k);

        auto Y = make_tile_B<complex, false>(arg.tile);
        Y.load(arg.Y, d + 4, parity, x_cb, k, j0);

        yHat.mma_nn(X, Y);
      }
      if (compute_max_only) {
        yHatMax = fmax(yHatMax, yHat.abs_max());
      } else {
        yHat.save(arg.Yhat, d + 4, parity, x_cb, i0, j0);
      }
    }

    return yHatMax;
  }

  template <bool compute_max_only, typename Arg> void CalculateYhatCPU(Arg &arg)
  {
    typename Arg::Float max = 0.0;
    for (int d=0; d<4; d++) {
      for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
        for (int x_cb = 0; x_cb < arg.Y.VolumeCB(); x_cb++) {
          for (int i = 0; i < Arg::yhatTileType::m; i += Arg::yhatTileType::M)
            for (int j = 0; j < Arg::yhatTileType::n; j += Arg::yhatTileType::N) {
              typename Arg::Float max_x = computeYhat<compute_max_only>(arg, d, x_cb, parity, i, j);
              if (compute_max_only) max = max > max_x ? max : max_x;
            }
        }
      } //parity
    } // dimension
    if (compute_max_only) *arg.max_h = max;
  }

  template <bool compute_max_only, typename Arg> __global__ void CalculateYhatGPU(Arg arg)
  {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.Y.VolumeCB()) return;
    int i_parity = blockDim.y*blockIdx.y + threadIdx.y;
    if (i_parity >= 2 * Arg::yhatTileType::M_tiles) return;
    int j_d = blockDim.z*blockIdx.z + threadIdx.z;
    if (j_d >= 4 * Arg::yhatTileType::N_tiles) return;

    int i = i_parity % Arg::yhatTileType::M_tiles;
    int parity = i_parity / Arg::yhatTileType::M_tiles;
    int j = j_d % Arg::yhatTileType::N_tiles;
    int d = j_d / Arg::yhatTileType::N_tiles;

    typename Arg::Float max
      = computeYhat<compute_max_only>(arg, d, x_cb, parity, i * Arg::yhatTileType::M, j * Arg::yhatTileType::N);
    if (compute_max_only) atomicAbsMax(arg.max_d, max);
  }

} // namespace quda
