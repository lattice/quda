#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <matrix_tile.cuh>
#include <kernel.h>

namespace quda {

  template <typename Float_, typename PreconditionedGauge, typename Gauge, typename GaugeInv, int n_, int M_, int N_, bool compute_max_>
  struct CalculateYhatArg : kernel_param<> {
    using Float = Float_;
    using yhatTileType = TileSize<n_, n_, n_, M_, N_, 1>;
    yhatTileType tile;

    static constexpr int M = n_;
    static constexpr int N = n_;
    static constexpr int K = n_;

    static constexpr bool is_mma_compatible = PreconditionedGauge::is_mma_compatible;
    static constexpr bool compute_max = compute_max_;

    PreconditionedGauge Yhat;
    const Gauge Y;
    const GaugeInv Xinv;
    int dim[QUDA_MAX_DIM];
    int comm_dim[QUDA_MAX_DIM];
    int nFace;

    Float *max_h; // host scalar that stores the maximum element of Yhat. Pointer b/c pinned.
    Float *max_d; // device scalar that stores the maximum element of Yhat
    Float *max;

    CalculateYhatArg(GaugeField &Yhat, const GaugeField &Y, const GaugeField &Xinv) :
      kernel_param(dim3(Y.VolumeCB(), 2 * yhatTileType::M_tiles, 4 * yhatTileType::N_tiles)),
      Yhat(Yhat), Y(Y), Xinv(Xinv), nFace(1), max_h(nullptr), max_d(nullptr)
    {
      for (int i=0; i<4; i++) {
        this->comm_dim[i] = comm_dim_partitioned(i);
        this->dim[i] = Xinv.X()[i];
      }
    }
  };

  template <typename Arg>
  inline __device__ __host__ auto computeYhat(const Arg &arg, int d, int x_cb, int parity, int i0, int j0)
  {
    using real = typename Arg::Float;
    using complex = complex<real>;
    constexpr int nDim = 4;
    int coord[nDim];
    getCoords(coord, x_cb, arg.dim, parity);

    const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

    real yHatMax = 0.0;

    // first do the backwards links Y^{+\mu} * X^{-\dagger}
    if (arg.comm_dim[d] && is_boundary(coord, d, 0, arg)) {

      auto yHat = make_tile_C<complex,true>(arg.tile);

#pragma unroll
      for (int k = 0; k < Arg::yhatTileType::k; k += Arg::yhatTileType::K) {
        auto Y = make_tile_A<complex, true>(arg.tile);
        Y.load(arg.Y, d, 1-parity, ghost_idx, i0, k);

        auto X = make_tile_Bt<complex, false>(arg.tile);
        X.load(arg.Xinv, 0, parity, x_cb, j0, k);

        yHat.mma_nt(Y, X);
      }

      if constexpr (Arg::compute_max) {
        yHatMax = yHat.abs_max();
      } else {
        yHat.save(arg.Yhat, d, 1 - parity, ghost_idx, i0, j0);
      }

    } else {
      const int back_idx = linkIndexHop(coord, arg.dim, d, -arg.nFace);

      auto yHat = make_tile_C<complex,false>(arg.tile);

#pragma unroll
      for (int k = 0; k < Arg::yhatTileType::k; k += Arg::yhatTileType::K) {
        auto Y = make_tile_A<complex, false>(arg.tile);
        Y.load(arg.Y, d, 1-parity, back_idx, i0, k);

        auto X = make_tile_Bt<complex, false>(arg.tile);
        X.load(arg.Xinv, 0, parity, x_cb, j0, k);

        yHat.mma_nt(Y, X);
      }
      if constexpr (Arg::compute_max) {
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
      if constexpr (Arg::compute_max) {
        yHatMax = fmax(yHatMax, yHat.abs_max());
      } else {
        yHat.save(arg.Yhat, d + 4, parity, x_cb, i0, j0);
      }
    }

    return yHatMax;
  }

  template <typename Arg> struct ComputeYhat {
    const Arg &arg;
    constexpr ComputeYhat(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ void operator()(int x_cb, int i_parity, int j_d)
    {
      int i = i_parity % Arg::yhatTileType::M_tiles;
      int parity = i_parity / Arg::yhatTileType::M_tiles;
      int j = j_d % Arg::yhatTileType::N_tiles;
      int d = j_d / Arg::yhatTileType::N_tiles;

      auto max = computeYhat(arg, d, x_cb, parity, i * Arg::yhatTileType::M, j * Arg::yhatTileType::N);
      if (Arg::compute_max) atomic_fetch_abs_max(arg.max, max);
    }
  };

} // namespace quda
