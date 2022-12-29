#pragma once

#include <type_traits>
#include <quda_fp16.cuh>
#include <array.h>
#include <trove/ptr.h>
#include <mma_tensor_op/smma_m16n8_sm80.cuh>

namespace quda
{
  namespace smma
  {

    template <>
    struct smma_t <half, 4, 1, 1> {

      static constexpr bool use_intermediate_accumulator() { return true; };

    static __device__ __host__ constexpr int inline pad_size(int m) { return 0; }

    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 16;
    static constexpr int MMA_K = 4;

    static constexpr int warp_size = 32;

    using compute_t = float;
    using load_t = float;

    struct WarpRegisterMapping {

      int warp_id;
      int row_offset; // quad_row * 8 + quad_hilo * 4
      int col_offset; // quad_col * 8 + quad_hilo * 4
      int quad_col;
      int quad_thread; // 0,1,2,3

      __device__ inline WarpRegisterMapping(int thread_id)
      {
        warp_id = thread_id >> 5;
        int lane_id = thread_id & 31;
        int octl_id = lane_id >> 2;
        int quad_id = octl_id & 3;
        int quad_row = quad_id & 1;
        int quad_hilo = (octl_id >> 2) & 1;
        quad_col = quad_id >> 1;
        quad_thread = lane_id & 3;
        row_offset = quad_row * 8 + quad_hilo * 4;
        col_offset = quad_col * 8 + quad_hilo * 4;
      }
    };

    struct OperandA {

      unsigned big[2];
      unsigned small[2];

      template <class SmemObj>
      __device__ inline void load(const SmemObj &smem_obj, int k, int warp_row, const WarpRegisterMapping &wrm)
      {
        const float *A = reinterpret_cast<const float *>(smem_obj.ptr);
        int idx_strided = k * MMA_K + wrm.quad_thread;
        int idx_contiguous = warp_row * MMA_M + wrm.row_offset;
        const int thread_offset_a = idx_strided * SmemObj::ldn + idx_contiguous;

#pragma unroll
        for (int v = 0; v < 2; v++) {
          float f[2];
          f[0] = A[thread_offset_a + 2 * v + 0];
          f[1] = A[thread_offset_a + 2 * v + 1];
          Shuffle<half, 2> s;
          s(big[v], small[v], f);
        }
      }

      __device__ inline void negate()
      {
        asm volatile("neg.f16x2 %0, %0;" : "+r"(big[0]));
        asm volatile("neg.f16x2 %0, %0;" : "+r"(big[1]));
        asm volatile("neg.f16x2 %0, %0;" : "+r"(small[0]));
        asm volatile("neg.f16x2 %0, %0;" : "+r"(small[1]));
      }
    };

    struct OperandB {

      unsigned big[2];
      unsigned small[2];

      template <class SmemObj>
      __device__ inline void load(const SmemObj &smem_obj, int k, int warp_col, const WarpRegisterMapping &wrm)
      {
        const float *B = reinterpret_cast<const float *>(smem_obj.ptr);
        int idx_strided = k * MMA_K + wrm.quad_thread;
        int idx_contiguous = warp_col * MMA_N + wrm.col_offset;
        const int thread_offset_b = idx_strided * SmemObj::ldn + idx_contiguous;

#pragma unroll
        for (int v = 0; v < 2; v++) {
          float f[2];
          f[0] = B[thread_offset_b + 2 * v + 0];
          f[1] = B[thread_offset_b + 2 * v + 1];
          Shuffle<half, 2> s;
          s(big[v], small[v], f);
        }
      }
    };

    struct OperandC {

      using reg_type = float;
      reg_type reg[8];

      __device__ inline OperandC() { zero(); }

      __device__ inline void zero()
      {
#pragma unroll
        for (int i = 0; i < 8; i++) { reg[i] = 0; }
      }

      __device__ inline void ax(float alpha)
      {
#pragma unroll
        for (int i = 0; i < 8; i++) { reg[i] *= alpha; }
      }

      template <int ldc>
      __device__ inline void store(void *smem, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
      {
        float *C = reinterpret_cast<float *>(smem);

        const int idx_strided = warp_row * 16 + wrm.row_offset + (wrm.quad_thread % 2);
        const int idx_contiguous = warp_col * 16 + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

        int thread_offset_c = idx_strided * ldc + idx_contiguous;
        C[thread_offset_c + 0] = reg[0];
        C[thread_offset_c + 1] = reg[1];

        thread_offset_c = (idx_strided + 2) * ldc + idx_contiguous;
        C[thread_offset_c + 0] = reg[2];
        C[thread_offset_c + 1] = reg[3];

        thread_offset_c = idx_strided * ldc + (idx_contiguous + 2);
        C[thread_offset_c + 0] = reg[4];
        C[thread_offset_c + 1] = reg[5];

        thread_offset_c = (idx_strided + 2) * ldc + (idx_contiguous + 2);
        C[thread_offset_c + 0] = reg[6];
        C[thread_offset_c + 1] = reg[7];
      }

      template <class F> __device__ inline void abs_max(F &max)
      {
#pragma unroll
        for (int i = 0; i < 8; i++) { max = fmax(max, fabsf(reg[i])); }
      }
    };

    static __device__ void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
    {
      mma_instruction_t<MMA_M, MMA_N, MMA_K, half, float> mma_instruction;

      if (use_intermediate_accumulator()) {
        float acc[8];
#pragma unroll
        for (int c = 0; c < 8; c++) { acc[c] = 0; }

        mma_instruction(acc, op_a.big, op_b.big);
        mma_instruction(acc, op_a.big, op_b.small);
        mma_instruction(acc, op_a.small, op_b.big);

#pragma unroll
        for (int c = 0; c < 8; c++) { op_c.reg[c] += acc[c]; }
      } else {
        mma_instruction(op_c.reg, op_a.big, op_b.big);
        mma_instruction(op_c.reg, op_a.big, op_b.small);
        mma_instruction(op_c.reg, op_a.small, op_b.big);
      }
    }

    template <int M, int N, int ldc, class GmemOperandC>
    static inline __device__ void
    store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc, const OperandC &op_c_real,
                  const OperandC &op_c_imag)
    {
      using store_t = typename GmemOperandC::store_type;
      using complex_t = complex<store_t>;

      auto *C = reinterpret_cast<complex_t *>(cc.data());

      const int row = warp_row + wrm.row_offset + (wrm.quad_thread % 2);
      const int col = warp_col + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

#pragma unroll
      for (int i = 0; i < 8; i++) {
        int m = row + (i % 4) / 2 * 2;
        int n = col + (i / 4) * 4 + i % 2;
        if (GmemOperandC::fixed) {
          auto scale = cc.scale;
          C[m * ldc + n] = {static_cast<store_t>(scale * op_c_real.reg[i]), static_cast<store_t>(scale * op_c_imag.reg[i])};
        } else {
          C[m * ldc + n] = {op_c_real.reg[i], op_c_imag.reg[i]};
        }
      }
    }
  };

  } // namespace smma
} // namespace quda
