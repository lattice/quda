#pragma once

#include <type_traits>
#include <quda_fp16.cuh>
#include <array.h>
#include <trove/ptr.h>
#include <mma_tensor_op/hmma_m16n8k8_sm70.cuh>

namespace quda
{
  namespace smma
  {

    template <class shuffle_t, int inst_k_, int warp_m_, int warp_n_> struct smma_x_t {
    };

    template <> struct smma_x_t<mma::half, 8, 1, 1> {

      static constexpr bool use_intermediate_accumulator() { return true; };

      static __device__ __host__ constexpr int inline pad_size(int) { return 0; }

      static constexpr bool do_rescale()
      {
        return true; // false because we use FP16
      }

      static constexpr int MMA_M = 16;
      static constexpr int MMA_N = 8;
      static constexpr int MMA_K = 8;

      static constexpr int warp_size = 32;

      using compute_t = float;
      using load_t = float;

      using base_t = hmma::hmma_x_t<16, 8, 8, half, half2>;

      using WarpRegisterMapping = typename base_t::WarpRegisterMapping;

      static std::string get_type_name()
      {
        char s[TuneKey::aux_n] = ",3xfp16,m";
        i32toa(s + strlen(s), MMA_M);
        strcat(s, "n");
        i32toa(s + strlen(s), MMA_N);
        strcat(s, "k");
        i32toa(s + strlen(s), MMA_K);
        return s;
      }

      struct OperandA {

        unsigned big[2];
        unsigned small[2];

        template <class SmemObj>
        __device__ inline void load(const SmemObj &smem_obj, int k, int warp_row, const WarpRegisterMapping &wrm)
        {
          const float *A = reinterpret_cast<const float *>(smem_obj.ptr);
          int idx_strided = k * MMA_K + wrm.quad_thread + wrm.quad_col * 4;
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
#pragma unroll
          for (int v = 0; v < 2; v++) {
            asm volatile("neg.f16x2 %0, %0;" : "+r"(big[v]));
            asm volatile("neg.f16x2 %0, %0;" : "+r"(small[v]));
          }
        }
      };

      struct OperandB {

        unsigned big[2];
        unsigned small[2];

        template <class SmemObj>
        __device__ inline void load(const SmemObj &smem_obj, int k, int warp_col, const WarpRegisterMapping &wrm)
        {
          const float *B = reinterpret_cast<const float *>(smem_obj.ptr);
          int idx_strided = k * MMA_K + wrm.quad_thread + wrm.quad_col * 4;
          int idx_contiguous = warp_col * MMA_N + wrm.quad_hilo * 4;
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

      using OperandC = typename base_t::OperandC;

      static __device__ void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
        mma::mma_instruction_t<MMA_M, 16, 4, mma::half, float> mma_instruction;
        float acc[8];
#pragma unroll
        for (int c = 0; c < 8; c++) { acc[c] = 0; }

        mma_instruction(acc, op_a.big, op_b.big);
        mma_instruction(acc, op_a.big, op_b.small);
        mma_instruction(acc, op_a.small, op_b.big);
        float other_acc[8];
#pragma unroll
        for (int x = 0; x < 8; x++) {
          other_acc[x] = __shfl_xor_sync(0xffffffff, acc[x], 0x8);
        }
        int lane_id = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) % 32;
        if (lane_id / 8 % 2 == 0) {
#pragma unroll
          for (int x = 0; x < 4; x++) {
            op_c.reg[x] += acc[x] + other_acc[x];
          }
        } else {
#pragma unroll
          for (int x = 0; x < 4; x++) {
            op_c.reg[x] += acc[x + 4] + other_acc[x + 4];
          }
        }
      }

      template <int M, int N, int ldc, bool dagger, class GmemOperandC, class op_t>
      static inline __device__ void store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm,
                                                  GmemOperandC &cc, const OperandC &op_c_real,
                                                  const OperandC &op_c_imag, op_t op)
      {
        base_t::template store_complex<M, N, ldc, dagger>(warp_row, warp_col, wrm, cc, op_c_real, op_c_imag, op);
      }
    };

  } // namespace smma
} // namespace quda
