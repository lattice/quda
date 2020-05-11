#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>

#include <mma_tensor_op/gemm.cuh>

#include <type_traits>

namespace quda
{

  namespace mma
  {

    template <typename Float_, typename PreconditionedGauge, typename Gauge, int n> struct CalculateYhatArg {
      using Float = Float_;

      static constexpr int M = n;
      static constexpr int N = n;
      static constexpr int K = n;

      PreconditionedGauge Yhat;
      const Gauge Y;
      const Gauge Xinv;
      int dim[QUDA_MAX_DIM];
      int comm_dim[QUDA_MAX_DIM];
      int nFace;

      Float *max_h; // host scalar that stores the maximum element of Yhat. Pointer b/c pinned.
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
        for (int i = 0; i < 4; i++) {
          this->comm_dim[i] = comm_dim[i];
          this->dim[i] = dim[i];
        }
      }
    };

    template <bool compute_max_only, typename Arg, int bM, int bN, int bK, int block_y, int block_z>
    inline __device__ auto computeYhat(Arg &arg, int d, int x_cb, int parity, half *smem_ptr, int m, int n)
    {
      using real = typename Arg::Float;
      using complex = complex<real>;
      constexpr int nDim = 4;
      int coord[nDim];
      getCoords(coord, x_cb, arg.dim, parity);

      real yHatMax = 0.0;

      // first do the backwards links Y^{+\mu} * X^{-\dagger}
      if (arg.comm_dim[d] && (coord[d] - arg.nFace < 0)) {

        const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

        auto aa = arg.Y.Ghost(d, 1 - parity, ghost_idx);
        auto bb = arg.Xinv(0, parity, x_cb);
        auto cc = arg.Yhat.Ghost(d, 1 - parity, ghost_idx);

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = true;
        yHatMax
          = perform_mma<Arg::N, bM, bN, bK, block_y, block_z, a_dagger, b_dagger, compute_max_only>(aa, bb, cc, m, n);

      } else {

        const int back_idx = linkIndexM1(coord, arg.dim, d);

        auto aa = arg.Y(d, 1 - parity, back_idx);
        auto bb = arg.Xinv(0, parity, x_cb);
        auto cc = arg.Yhat(d, 1 - parity, back_idx);

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = true;
        yHatMax
          = perform_mma<Arg::N, bM, bN, bK, block_y, block_z, a_dagger, b_dagger, compute_max_only>(aa, bb, cc, m, n);
      }

      { // now do the forwards links X^{-1} * Y^{-\mu}

        auto aa = arg.Xinv(0, parity, x_cb);
        auto bb = arg.Y(d + 4, parity, x_cb);
        auto cc = arg.Yhat(d + 4, parity, x_cb);

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;
        real yHatMax_
          = perform_mma<Arg::N, bM, bN, bK, block_y, block_z, a_dagger, b_dagger, compute_max_only>(aa, bb, cc, m, n);
        yHatMax = fmax(yHatMax, yHatMax_);
      }

      return yHatMax;
    }

    template <bool compute_max_only, typename Arg, int N, int bM, int bN, int bK, int block_y, int block_z>
    __global__ void __launch_bounds__(block_y * block_z, 1) CalculateYhatGPU(Arg arg)
    {
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;

      constexpr int t_m = Arg::M / bM;
      constexpr int t_n = Arg::N / bN;
      int x_cb = index_x / (t_m * t_n);
      int mn = index_x % (t_m * t_n);

      int n = (mn % t_n) * bN;
      int m = (mn / t_n) * bM;

      if (x_cb >= arg.Y.VolumeCB()) return;

      int parity = blockIdx.y; // i_parity / arg.tile.M_tiles;

      int d = blockIdx.z; // j_d / arg.tile.N_tiles;

      extern __shared__ half smem_ptr[];

      typename Arg::Float max = 0.0;
      switch (d) {
      case 0:
        max = computeYhat<compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 0, x_cb, parity, smem_ptr, m, n);
        break;
      case 1:
        max = computeYhat<compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 1, x_cb, parity, smem_ptr, m, n);
        break;
      case 2:
        max = computeYhat<compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 2, x_cb, parity, smem_ptr, m, n);
        break;
      case 3:
        max = computeYhat<compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 3, x_cb, parity, smem_ptr, m, n);
        break;
      }
      if (compute_max_only) atomicAbsMax(arg.max_d, max);
    }

  } // namespace mma

} // namespace quda
