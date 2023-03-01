#pragma once

#include <mma_tensor_op/simt.cuh>

namespace quda
{

  namespace simt
  {

    template <int inst_m_, int inst_n_, int warp_m_, int warp_n_>
    struct simt_t<mma::half, inst_m_, inst_n_, warp_m_, warp_n_> {

      static constexpr int warp_m = warp_m_; // tiling in m
      static constexpr int warp_n = warp_n_; // tiling in n

      static constexpr int inst_m = inst_m_;
      static constexpr int inst_n = inst_n_;
      static constexpr int inst_k = 1;

      static constexpr int MMA_M = inst_m * warp_m;
      static constexpr int MMA_N = inst_n * warp_n;
      static constexpr int MMA_K = inst_k;

      static constexpr int mma_m = MMA_M;
      static constexpr int mma_n = MMA_N;
      static constexpr int mma_k = MMA_K;

      static constexpr int warp_size = 32;

      static_assert(warp_m % 2 == 0, "warp_m should be a multiple of 2");
      static_assert(warp_n % 2 == 0, "warp_n should be a multiple of 2");
      static_assert(inst_m * inst_n == warp_size, "inst_m * inst_n == warp_size");

      using store_t = mma::half2;

      using compute_t = mma::half;
      using load_t = mma::half2;

      static std::string get_type_name()
      {
        return ",simt_half,m" + std::to_string(MMA_M) + "n" + std::to_string(MMA_N) + "k" + std::to_string(MMA_K);
      }

      static __device__ __host__ constexpr int inline pad_size(int) { return 0; }

      struct WarpRegisterMapping {

        int warp_id;
        int lane_id;
        int idx_m;
        int idx_n;

        __device__ WarpRegisterMapping(int thread_id) :
          warp_id(thread_id / 32), lane_id(thread_id & 31), idx_m(lane_id % inst_m), idx_n(lane_id / inst_m)
        {
        }
      };

      struct OperandA {

        store_t reg[warp_m / 2];

        template <int lda>
        __device__ inline void load(const void *smem_ptr, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
        { // Assuming col major smem layout

          const store_t *A = reinterpret_cast<const store_t *>(smem_ptr);

#pragma unroll
          for (int wm = 0; wm < warp_m; wm += 2) {
            int k = tile_k * mma_k;
            int m = tile_m * mma_m + wrm.idx_m * warp_m + wm;
            reg[wm] = A[(k * lda + m) / 2];
          }
        }
      };

      struct OperandB {

        store_t reg[warp_n / 2];

        template <int ldb>
        __device__ inline void load(const void *smem_ptr, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
        { // Assuming row major smem layout

          const store_t *B = reinterpret_cast<const store_t *>(smem_ptr);

#pragma unroll
          for (int wn = 0; wn < warp_n; wn += 2) {
            int k = tile_k * mma_k;
            int n = tile_n * mma_n + wrm.idx_n * warp_n + wn;
            reg[wn] = B[(k * ldb + n) / 2];
          }
        }
      };

      struct OperandC {

        using reg_type = float;
        reg_type reg[warp_m * warp_n];

        __device__ inline void zero()
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n; i++) { reg[i] = 0; }
        }

        __device__ inline OperandC() { zero(); }

        __device__ inline void ax(float alpha)
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n; i++) { reg[i] *= alpha; }
        }

        template <int ldc> __device__ void store(void *ptr, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
        {
          half2 *C = reinterpret_cast<half2 *>(ptr);
#pragma unroll
          for (int wn = 0; wn < warp_n / 2; wn++) {
#pragma unroll
            for (int wm = 0; wm < warp_m; wm++) {
              int m = warp_row * mma_m + wrm.idx_m * warp_m + wm;
              int n = warp_col * mma_n + wrm.idx_n * warp_n + wn * 2;
              C[(m * ldc + n) / 2] = __floats2half2_rn(reg[(wn * 2 + 0) * warp_m + wm], reg[(wn * 2 + 1) * warp_m + wm]);
            }
          }
        }
      };

      static __device__ void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
#pragma unroll
        for (int wm = 0; wm < warp_m / 2; wm++) {
#pragma unroll
          for (int wn = 0; wn < warp_n / 2; wn++) {
            op_c.reg[(wn * 2 + 0) * warp_m + wm * 2 + 0] += static_cast<float>(__hmul(op_a.reg[wm].x, op_b.reg[wn].x));
            op_c.reg[(wn * 2 + 0) * warp_m + wm * 2 + 1] += static_cast<float>(__hmul(op_a.reg[wm].y, op_b.reg[wn].x));
            op_c.reg[(wn * 2 + 1) * warp_m + wm * 2 + 0] += static_cast<float>(__hmul(op_a.reg[wm].x, op_b.reg[wn].y));
            op_c.reg[(wn * 2 + 1) * warp_m + wm * 2 + 1] += static_cast<float>(__hmul(op_a.reg[wm].y, op_b.reg[wn].y));
          }
        }
      }
    };

  } // namespace simt

} // namespace quda
