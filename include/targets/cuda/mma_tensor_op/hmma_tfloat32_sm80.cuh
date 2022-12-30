#pragma once

#include <mma_tensor_op/mma_instruction.cuh>

namespace quda
{

  namespace hmma
  {

    template <int inst_k_, int warp_m_, int warp_n_>
    struct hmma_tfloat32_t {

      static constexpr int warp_m = warp_m_;
      static constexpr int warp_n = warp_n_;

      static constexpr int inst_m = 16;
      static constexpr int inst_n = 8;
      static constexpr int inst_k = inst_k_;

      static constexpr int MMA_M = inst_m * warp_m;
      static constexpr int MMA_N = inst_n * warp_n;
      static constexpr int MMA_K = inst_k;

      static constexpr int mma_m = MMA_M;
      static constexpr int mma_n = MMA_N;
      static constexpr int mma_k = MMA_K;

      static constexpr int warp_size = 32;

      using compute_t = float;
      using load_t = float;

      static __device__ __host__ constexpr int inline pad_size(int m)
      {
        return (m - 8 + 15) / 16 * 16 + 8 - m;
      }

      struct WarpRegisterMapping {

        int warp_id;
        int lane_id;
        int group_id;
        int thread_id_in_group;

        __device__ WarpRegisterMapping(int thread_id) :
          warp_id(thread_id / 32), lane_id(thread_id & 31), group_id(lane_id >> 2), thread_id_in_group(lane_id & 3)
        {
        }
      };

      struct OperandA {

        static constexpr int thread_k = inst_k / 4;
        static constexpr int thread_m = inst_m / 8;
        static constexpr int thread_count = thread_k * thread_m;

        using store_t = unsigned;
        store_t reg[warp_m * thread_count];

        __device__ inline void negate()
        {
#pragma unroll
          for (int i = 0; i < warp_m * thread_count; i++) {
            constexpr unsigned flip_bit_sign = 0x80000000;
            reg[i] ^= flip_bit_sign;
          }
        }

        template <class smem_obj_t>
        __device__ inline void load(const smem_obj_t &smem_obj, int tile_k, int tile_m, const WarpRegisterMapping &wrm)
        { // Assuming col major smem layout

          store_t *A = reinterpret_cast<store_t *>(smem_obj.ptr);

#pragma unroll
          for (int tk = 0; tk < thread_k; tk++) {
#pragma unroll
            for (int tm = 0; tm < thread_m; tm++) {
#pragma unroll
              for (int wm = 0; wm < warp_m; wm++) {
                int k = tile_k * mma_k + tk * 4 + wrm.thread_id_in_group;
                int m = tile_m * mma_m + wm * inst_m + (tm * 8 + wrm.group_id);
                reg[wm * thread_count + (tk * thread_m + tm)] = A[k * smem_obj_t::ldn + m];
              }
            }
          }
        }
      };

      struct OperandB {

        static constexpr int thread_k = inst_k / 4;
        static constexpr int thread_n = inst_n / 8;
        static constexpr int thread_count = thread_k * thread_n;

        using store_t = unsigned;
        store_t reg[warp_n * thread_count];

        template <class smem_obj_t>
        __device__ inline void load(const smem_obj_t &smem_obj, int tile_k, int tile_n, const WarpRegisterMapping &wrm)
        { // Assuming row major smem layout

          store_t *B = reinterpret_cast<store_t *>(smem_obj.ptr);

#pragma unroll
          for (int tk = 0; tk < thread_k; tk++) {
#pragma unroll
            for (int tn = 0; tn < thread_n; tn++) {
#pragma unroll
              for (int wn = 0; wn < warp_n; wn++) {
                int k = tile_k * mma_k + tk * 4 + wrm.thread_id_in_group;
                int n = tile_n * mma_n + wn * inst_n + (tn * 8 + wrm.group_id);
                reg[wn * thread_count + (tk * thread_n + tn)] = B[k * smem_obj_t::ldn + n];
              }
            }
          }
        }
      };

      struct OperandC {

        static constexpr int thread_m = inst_m / 8;
        static constexpr int thread_n = inst_n / 4;
        static constexpr int thread_count = thread_m * thread_n;

        using reg_type = float;
        reg_type reg[warp_m * warp_n * thread_count];

        __device__ inline void zero()
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n * thread_count; i++) { reg[i] = 0; }
        }

        __device__ inline OperandC() { zero(); }

        __device__ inline void ax(float alpha)
        {
#pragma unroll
          for (int i = 0; i < warp_m * warp_n * thread_count; i++) { reg[i] *= alpha; }
        }

        template <int ldc> __device__ void store(void *ptr, int m_offset, int n_offset, const WarpRegisterMapping &wrm)
        {
          reg_type *C = reinterpret_cast<reg_type *>(ptr);
#pragma unroll
          for (int tm = 0; tm < thread_m; tm++) {
#pragma unroll
            for (int tn = 0; tn < thread_n; tn++) {
#pragma unroll
              for (int wn = 0; wn < warp_n; wn++) {
#pragma unroll
                for (int wm = 0; wm < warp_m; wm++) {
                  int m = m_offset + wm * inst_m + (wrm.group_id + tm * 8);
                  int n = n_offset + wn * inst_n + (wrm.thread_id_in_group * 2 + tn);
                  C[m * ldc + n] = reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + tn)];
                }
              }
            }
          }
        }
      };

      static __device__ void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
#pragma unroll
        for (int wm = 0; wm < warp_m; wm++) {
#pragma unroll
          for (int wn = 0; wn < warp_n; wn++) {

            int a_offset = wm * OperandA::thread_count;
            int b_offset = wn * OperandB::thread_count;
            int c_offset = (wn * warp_m + wm) * OperandC::thread_count;

            mma::mma_instruction_t<inst_m, inst_n, inst_k, mma::tfloat32, float> mma_instruction;

            mma_instruction(&op_c.reg[c_offset], &op_a.reg[a_offset], &op_b.reg[b_offset]);
          }
        }
      }

      template <int M, int N, int ldc, class gmem_op_t>
      static inline __device__ void store_complex(int m_offset, int n_offset, const WarpRegisterMapping &wrm,
                                                  gmem_op_t &cc, const OperandC &op_c_real, const OperandC &op_c_imag)
      {
        using store_t = typename gmem_op_t::store_type;
        using complex_t = complex<store_t>;

        auto *C = reinterpret_cast<complex_t *>(cc.data());

        constexpr int thread_m = OperandC::thread_m;
        constexpr int thread_n = OperandC::thread_n;
        constexpr int thread_count = OperandC::thread_count;

        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));
#pragma unroll
        for (int tm = 0; tm < thread_m; tm++) {
#pragma unroll
          for (int tn = 0; tn < thread_n; tn++) {
#pragma unroll
            for (int wn = 0; wn < warp_n; wn++) {
#pragma unroll
              for (int wm = 0; wm < warp_m; wm++) {
                int m = m_offset + wm * inst_m + (wrm.group_id + tm * 8);
                int n = n_offset + wn * inst_n + (wrm.thread_id_in_group * 2 + tn);
                if (!check_bounds || (m < M && n < N)) {
                  if (gmem_op_t::fixed) {
                    auto scale = cc.scale;
                    C[m * ldc + n]
                      = {static_cast<store_t>(
                           scale * op_c_real.reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + tn)]),
                         static_cast<store_t>(
                           scale * op_c_imag.reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + tn)])};
                  } else {
                    C[m * ldc + n] = {op_c_real.reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + tn)],
                                      op_c_imag.reg[(wn * warp_m + wm) * thread_count + (tm * thread_n + tn)]};
                  }
                }
              }
            }
          }
        }
      }
    };

  } // namespace smma

} // namespace quda
