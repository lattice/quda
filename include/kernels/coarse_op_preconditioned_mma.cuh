#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>
#include <mma_tensor_op/gemm.cuh>
#include <block_reduce_helper.h>
#include <kernel.h>
#include <kernels/coarse_op_preconditioned.cuh>

// This is MMA implementation of the computeYhat kernels.

namespace quda
{

  namespace mma
  {

    template <typename Arg, int bM_, int bN_, int bK_, int block_y_, int block_z_, int min_blocks_>
    struct CalculateYhatMMAArg : Arg {
      static constexpr int bM = bM_;
      static constexpr int bN = bN_;
      static constexpr int bK = bK_;
      static constexpr int block_y = block_y_;
      static constexpr int block_z = block_z_;
      static constexpr int block_dim = block_y * block_z;
      static constexpr int min_blocks = min_blocks_;

      CalculateYhatMMAArg(const Arg &arg) : Arg(arg) {}
    };

    template <typename Arg>
    inline __device__ auto computeYhatMMA(Arg &arg, int d, int x_cb, int parity, int m, int n)
    {
      using real = typename Arg::Float;
      constexpr int nDim = 4;
      int coord[nDim];
      getCoords(coord, x_cb, arg.dim, parity);

      real yHatMax = 0.0;

      using mma_t = typename mma::mg_mma_dispatch_t<typename Arg::Float>::type;
      using Config = MmaConfig<mma_t, Arg::M, Arg::N, Arg::K, Arg::M, Arg::N, Arg::K, Arg::bM, Arg::bN, Arg::bK,
                               Arg::block_y, Arg::block_z>;

      {
        // first do the backwards links Y^{+\mu} * X^{-\dagger}
        constexpr bool a_dagger = false;
        constexpr bool b_dagger = true;

        if (arg.comm_dim[d] && (coord[d] - arg.nFace < 0)) {

          const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

          auto a = arg.Y.Ghost(d, 1 - parity, ghost_idx, 0, 0);
          auto b = arg.Xinv(0, parity, x_cb, 0, 0);
          auto c = arg.Yhat.Ghost(d, 1 - parity, ghost_idx, 0, 0);

          yHatMax = Config::template perform_mma<a_dagger, b_dagger, Arg::compute_max>(a, b, c, m, n);

        } else {

          const int back_idx = linkIndexM1(coord, arg.dim, d);

          auto a = arg.Y(d, 1 - parity, back_idx, 0, 0);
          auto b = arg.Xinv(0, parity, x_cb, 0, 0);
          auto c = arg.Yhat(d, 1 - parity, back_idx, 0, 0);

          yHatMax = Config::template perform_mma<a_dagger, b_dagger, Arg::compute_max>(a, b, c, m, n);
        }
      }

      { // now do the forwards links X^{-1} * Y^{-\mu}
        auto a = arg.Xinv(0, parity, x_cb, 0, 0);
        auto b = arg.Y(d + 4, parity, x_cb, 0, 0);
        auto c = arg.Yhat(d + 4, parity, x_cb, 0, 0);

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;

        yHatMax = fmaxf(yHatMax, Config::template perform_mma<a_dagger, b_dagger, Arg::compute_max>(a, b, c, m, n));
      }

      return yHatMax;
    }

    template <typename Arg> struct CalculateYhatMMA {
      Arg &arg;
      constexpr CalculateYhatMMA(Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __forceinline__ void operator()()
      {
        int index_x = blockDim.x * blockIdx.x + threadIdx.x;

        constexpr bool divide_b_no = Arg::bM < Arg::M && Arg::bK == Arg::K && Arg::bN == Arg::N;
        constexpr int t_m = divide_b_no ? 1 : (Arg::M + Arg::bM - 1) / Arg::bM;
        constexpr int t_n = divide_b_no ? 1 : (Arg::N + Arg::bN - 1) / Arg::bN;

        int x_cb = index_x / (t_m * t_n);
        int mn = index_x % (t_m * t_n);

        int n = (mn % t_n) * Arg::bN;
        int m = (mn / t_n) * Arg::bM;

        if (x_cb >= arg.Y.VolumeCB()) return;

        int parity = blockIdx.y;
        int d = blockIdx.z;

        using real = typename Arg::Float;
        real max = 0.0;
        switch (d) {
        case 0: max = computeYhatMMA(arg, 0, x_cb, parity, m, n); break;
        case 1: max = computeYhatMMA(arg, 1, x_cb, parity, m, n); break;
        case 2: max = computeYhatMMA(arg, 2, x_cb, parity, m, n); break;
        case 3: max = computeYhatMMA(arg, 3, x_cb, parity, m, n); break;
        }
        if (Arg::compute_max) {
          constexpr int block_dim = 3;
          unsigned aggregate = BlockReduce<unsigned, block_dim>().Max(__float_as_uint(max));
          if (threadIdx.y == 0 && threadIdx.z == 0) atomic_fetch_abs_max(arg.max_d, __uint_as_float(aggregate));
        }
      }
    };

  } // namespace mma

} // namespace quda
