#pragma once

namespace quda {
  namespace mma {

    constexpr int t_pad = 8;
    constexpr int n_pad = 4;

    constexpr int WARP_M = 1;
    constexpr int WARP_N = 1;

    constexpr int INST_M = 16;
    constexpr int INST_N = 8;
    constexpr int INST_K = 4;

    constexpr int MMA_M = INST_M * WARP_M;
    constexpr int MMA_N = INST_N * WARP_N;
    constexpr int MMA_K = INST_K;

    struct WarpRegisterMapping {

      int lane_id;
      int group_id;
      int thread_id_in_group;

      __device__ WarpRegisterMapping(int thread_id) :
        lane_id(thread_id & 31),
        group_id(lane_id >> 2),
        thread_id_in_group(lane_id & 3)
      {
      }
    };

    struct MmaOperandA {

      using reg_type = unsigned;
      reg_type reg[WARP_M * 2];

      template <class S> __device__ inline void load(const S &smem, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
      {
        int k = tile_k * MMA_K + wrm.thread_id_in_group;
#pragma unroll
        for (int i = 0; i < 2; i++) {
#pragma unroll
          for (int b = 0; b < WARP_M; b++) {
            int m = tile_m * MMA_M + b * INST_M + i * 8 + wrm.group_id;
            reg[b * 2 + i] = __float_as_uint(smem(m, k));
          }
        }
      }
    };

    struct MmaOperandB {

      using reg_type = unsigned;
      reg_type reg[WARP_N * 1];

      template <class S> __device__ inline void load(const S &smem, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
      {
        int k = tile_k * MMA_K + wrm.thread_id_in_group;

#pragma unroll
        for (int i = 0; i < 1; i++) {
#pragma unroll
          for (int b = 0; b < WARP_N; b++) {
            int n = tile_n * MMA_N + b * INST_N + i * 8 + wrm.group_id;
            reg[b * 1 + i] = __float_as_uint(smem(n, k));
          }
        }
      }
    };

    struct MmaOperandC {

      using reg_type = float;
      reg_type reg[WARP_M * WARP_N * 4];

      __device__ MmaOperandC()
      {
        zero();
      }

      void __device__ zero()
      {
#pragma unroll
        for (int i = 0; i < WARP_M * WARP_N * 4; i++) { reg[i] = 0; }
      }

      template <class S> __device__ void load(S &smem, int m_offset, int n_offset, const WarpRegisterMapping &wrm)
      {
#pragma unroll
        for (int c = 0; c < 4; c++) {
#pragma unroll
          for (int n_b = 0; n_b < WARP_N; n_b++) {
#pragma unroll
            for (int m_b = 0; m_b < WARP_M; m_b++) {
              int gmem_m = m_offset + m_b * INST_M + wrm.group_id + (c / 2) * 8;
              int gmem_n = n_offset + n_b * INST_N + wrm.thread_id_in_group * 2 + c % 2;
              reg[(n_b * WARP_M + m_b) * 4 + c] = smem(gmem_m, gmem_n);
            }
          }
        }
      }

      template <class S> __device__ void store(S &smem, int m_offset, int n_offset, const WarpRegisterMapping &wrm) const
      {
#pragma unroll
        for (int c = 0; c < 4; c++) {
#pragma unroll
          for (int n_b = 0; n_b < WARP_N; n_b++) {
#pragma unroll
            for (int m_b = 0; m_b < WARP_M; m_b++) {
              int gmem_m = m_offset + m_b * INST_M + wrm.group_id + (c / 2) * 8;
              int gmem_n = n_offset + n_b * INST_N + wrm.thread_id_in_group * 2 + c % 2;
              smem(gmem_m, gmem_n) = reg[(n_b * WARP_M + m_b) * 4 + c];
            }
          }
        }
      }
    };

    __device__ void gemm(MmaOperandC &op_c, const MmaOperandA &op_a, const MmaOperandB &op_b)
    {
#pragma unroll
      for (int m_iter = 0; m_iter < WARP_M; m_iter++) {
#pragma unroll
        for (int n_iter = 0; n_iter < WARP_N; n_iter++) {
          int c_iter = (n_iter * WARP_M + m_iter) * 4;
          // asm volatile("mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
          //     : "+f"(op_c.reg[c_iter + 0]), "+f"(op_c.reg[c_iter + 1]), "+f"(op_c.reg[c_iter + 2]), "+f"(op_c.reg[c_iter + 3])
          //     : "r"(op_a.reg[m_iter * 2 + 0]), "r"(op_a.reg[m_iter * 2 + 1]), "r"(op_b.reg[n_iter + 0]));

        }
      }
    }

  }
}
