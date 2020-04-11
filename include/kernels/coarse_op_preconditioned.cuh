#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <matrix_tile.cuh>

namespace quda {

  template <typename Float_, typename PreconditionedGauge, typename Gauge, int n_, int M_, int N_> struct CalculateYhatArg {
    using Float = Float_;
    TileSize<n_, n_, n_, M_, N_, 1> tile;

    PreconditionedGauge Yhat;
    const Gauge Y;
    const Gauge Xinv;
    int dim[QUDA_MAX_DIM];
    int comm_dim[QUDA_MAX_DIM];
    int nFace;

    Float *max_h;  // host scalar that stores the maximum element of Yhat. Pointer b/c pinned.
    Float *max_d; // device scalar that stores the maximum element of Yhat

    CalculateYhatArg(const PreconditionedGauge &Yhat, const Gauge Y, const Gauge Xinv, const int *dim,
                     const int *comm_dim, int nFace) :
      Yhat(Yhat),
      Y(Y),
      Xinv(Xinv),
      nFace(nFace),
      max_h(nullptr),
      max_d(nullptr)
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
      for (int k = 0; k<arg.tile.k; k+=arg.tile.K) {
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
      for (int k = 0; k<arg.tile.k; k+=arg.tile.K) {
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
      for (int k = 0; k<arg.tile.k; k+=arg.tile.K) {
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
          for (int i = 0; i < arg.tile.m; i+=arg.tile.M)
            for (int j = 0; j < arg.tile.n; j+=arg.tile.N) {
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
    if (i_parity >= 2*arg.tile.M_tiles) return;
    int j_d = blockDim.z*blockIdx.z + threadIdx.z;
    if (j_d >= 4*arg.tile.N_tiles) return;

    int i = i_parity % arg.tile.M_tiles;
    int parity = i_parity / arg.tile.M_tiles;
    int j = j_d % arg.tile.N_tiles;
    int d = j_d / arg.tile.N_tiles;

    typename Arg::Float max = computeYhat<compute_max_only>(arg, d, x_cb, parity, i * arg.tile.M, j * arg.tile.N);
    if (compute_max_only) atomicAbsMax(arg.max_d, max);
  }

} // namespace quda
