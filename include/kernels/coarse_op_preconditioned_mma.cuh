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
        Yhat(Yhat), Y(Y), Xinv(Xinv), nFace(nFace), max_h(nullptr), max_d(nullptr)
      {
        for (int i = 0; i < 4; i++) {
          this->comm_dim[i] = comm_dim[i];
          this->dim[i] = dim[i];
        }
      }
    };

    template <int dir, bool compute_max_only, typename Arg, int bM, int bN, int bK, int block_y, int block_z>
    inline __device__ auto computeYhat(Arg &arg, int d, int x_cb, int parity, half *smem_ptr, int m, int n)
    {
      using real = typename Arg::Float;
      using complex = complex<real>;
      constexpr int nDim = 4;
      int coord[nDim];
      getCoords(coord, x_cb, arg.dim, parity);

      real yHatMax = 0.0;

      if (dir == 0) {
        // first do the backwards links Y^{+\mu} * X^{-\dagger}
        constexpr bool a_dagger = false;
        constexpr bool b_dagger = true;

        using Config = MmaConfig<Arg::M, Arg::N, Arg::K, Arg::M, Arg::N, Arg::K, bM, bN, bK, block_y, block_z>;
        Config config(smem_ptr);

        if (arg.comm_dim[d] && (coord[d] - arg.nFace < 0)) {

          const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

          auto a = arg.Y.wrap_ghost(d, 1 - parity, ghost_idx);
          auto b = arg.Xinv.wrap(0, parity, x_cb);
          auto c = arg.Yhat.wrap_ghost(d, 1 - parity, ghost_idx);

          yHatMax = config.perform_mma<a_dagger, b_dagger, compute_max_only>(a, b, c, m, n);

        } else {

          const int back_idx = linkIndexM1(coord, arg.dim, d);

          auto a = arg.Y.wrap(d, 1 - parity, back_idx);
          auto b = arg.Xinv.wrap(0, parity, x_cb);
          auto c = arg.Yhat.wrap(d, 1 - parity, back_idx);

          yHatMax = config.perform_mma<a_dagger, b_dagger, compute_max_only>(a, b, c, m, n);
        }

        return yHatMax;

      } else { // now do the forwards links X^{-1} * Y^{-\mu}

        auto a = arg.Xinv.wrap(0, parity, x_cb);
        auto b = arg.Y.wrap(d + 4, parity, x_cb);
        auto c = arg.Yhat.wrap(d + 4, parity, x_cb);

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;

        using Config = MmaConfig<Arg::M, Arg::N, Arg::K, Arg::M, Arg::N, Arg::K, bM, bN, bK, block_y, block_z>;
        Config config(smem_ptr);

        yHatMax = config.template perform_mma<a_dagger, b_dagger, compute_max_only>(a, b, c, m, n);
        // real yHatMax_ = config.template perform_mma<a_dagger, b_dagger, compute_max_only>(a, b, c, m, n);
        // yHatMax = fmax(yHatMax, yHatMax_);

        return yHatMax;
      }

      // return yHatMax;
    }

    template <bool compute_max_only, typename Arg, int N, int bM, int bN, int bK, int block_y, int block_z>
    __global__ void __launch_bounds__(block_y *block_z, 1) CalculateYhatGPU(Arg arg)
    {
      int index_x = blockDim.x * blockIdx.x + threadIdx.x;

      constexpr int t_m = Arg::M / bM;
      constexpr int t_n = Arg::N / bN;
      int x_cb = index_x / (t_m * t_n * 2);
      int mn_dir = index_x % (t_m * t_n * 2);

      int dir = mn_dir % 2;
      int mn = mn_dir / 2;

      int n = (mn % t_n) * bN;
      int m = (mn / t_n) * bM;

      if (x_cb >= arg.Y.VolumeCB()) return;

      int parity = blockIdx.y;

      int d = blockIdx.z;

      extern __shared__ half smem_ptr[];

      typename Arg::Float max = 0.0;
      switch (d) {
      case 0:
        max = dir ?
          computeYhat<1, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 0, x_cb, parity, smem_ptr, m, n) :
          computeYhat<0, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 0, x_cb, parity, smem_ptr, m, n);
        break;
      case 1:
        max = dir ?
          computeYhat<1, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 1, x_cb, parity, smem_ptr, m, n) :
          computeYhat<0, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 1, x_cb, parity, smem_ptr, m, n);
        break;
      case 2:
        max = dir ?
          computeYhat<1, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 2, x_cb, parity, smem_ptr, m, n) :
          computeYhat<0, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 2, x_cb, parity, smem_ptr, m, n);
        break;
      case 3:
        max = dir ?
          computeYhat<1, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 3, x_cb, parity, smem_ptr, m, n) :
          computeYhat<0, compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, 3, x_cb, parity, smem_ptr, m, n);
        break;
      }
      if (compute_max_only) atomicAbsMax(arg.max_d, max);
    }

  } // namespace mma

} // namespace quda
