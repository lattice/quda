#pragma once

#include <type_traits>
#include <quda_fp16.cuh>
#include <array.h>
#include <mma_tensor_op/smma_m16n8_sm80.cuh>

// Here we implement the architecture dependent part of MMA for Turing/Ampere (sm75/sm80, the mma.sync.m16n8k8 instruction).

namespace quda
{
  namespace hmma
  {

    template <> struct hmma_t<16, 8, 8, half, half2> {

      static __device__ __host__ constexpr int inline pad_size(int m) { return m == 192 ? 0 : 8; }

      static constexpr int MMA_M = 16;
      static constexpr int MMA_N = 8;
      static constexpr int MMA_K = 8;

      static constexpr int warp_size = 32;

      using compute_t = half;
      using load_t = half2;

      using base_t = smma::smma_t<half, 8, 1, 1>;

      using WarpRegisterMapping = typename base_t::WarpRegisterMapping;

      static std::string get_type_name()
      {
        char s[TuneKey::aux_n] = ",1xfp16,m";
        i32toa(s + strlen(s), MMA_M);
        strcat(s, "n");
        i32toa(s + strlen(s), MMA_N);
        strcat(s, "k");
        i32toa(s + strlen(s), MMA_K);
        return s;
      }

      struct OperandA {

        unsigned reg[2];

        template <int lda>
        __device__ inline void load(const void *smem, int k, int warp_row, const WarpRegisterMapping &wrm)
        {
          const unsigned *A = reinterpret_cast<const unsigned *>(smem);
          int idx_strided = k * MMA_K + (wrm.lane_id & 7);
          int idx_contiguous = warp_row * (MMA_M / 2) + ((wrm.lane_id >> 3) & 1) * 4;
          const unsigned *addr = &A[idx_strided * (lda / 2) + idx_contiguous];

          asm volatile("ldmatrix.sync.aligned.m8n8.trans.x2.b16 {%0,%1}, [%2];"
                       : "=r"(reg[0]), "=r"(reg[1])
                       : "l"(addr));
        }

        template <class SmemObj>
        __device__ inline void load(const SmemObj &smem_obj, int k, int warp_row, const WarpRegisterMapping &wrm)
        {
          const unsigned *A = reinterpret_cast<const unsigned *>(smem_obj.ptr);
          int idx_strided = k * MMA_K + (wrm.lane_id & 7);
          int idx_contiguous = warp_row * (MMA_M / 2) + ((wrm.lane_id >> 3) & 1) * 4;
          const unsigned *addr = &A[idx_strided * (SmemObj::ldn / 2) + idx_contiguous];

          asm volatile("ldmatrix.sync.aligned.m8n8.trans.x2.b16 {%0,%1}, [%2];"
                       : "=r"(reg[0]), "=r"(reg[1])
                       : "l"(addr));
        }

        __device__ inline void negate()
        {
          asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[0]));
          asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[1]));
        }
      };

      struct OperandB {

        unsigned reg[1];

        template <int ldb>
        __device__ inline void load(const void *smem, int k, int warp_col, const WarpRegisterMapping &wrm)
        {
          const unsigned *B = reinterpret_cast<const unsigned *>(smem);
          int idx_strided = k * MMA_K + (wrm.lane_id & 7);
          int idx_contiguous = warp_col * (MMA_N / 2);
          const unsigned *addr = &B[idx_strided * (ldb / 2) + idx_contiguous];
          asm volatile("ldmatrix.sync.aligned.m8n8.trans.x1.b16 {%0}, [%1];" : "=r"(reg[0]) : "l"(addr));
        }

        template <class SmemObj>
        __device__ inline void load(const SmemObj &smem_obj, int k, int warp_col, const WarpRegisterMapping &wrm)
        {
          const unsigned *B = reinterpret_cast<const unsigned *>(smem_obj.ptr);
          int idx_strided = k * MMA_K + (wrm.lane_id & 7);
          int idx_contiguous = warp_col * (MMA_N / 2);
          const unsigned *addr = &B[idx_strided * (SmemObj::ldn / 2) + idx_contiguous];
          asm volatile("ldmatrix.sync.aligned.m8n8.trans.x1.b16 {%0}, [%1];" : "=r"(reg[0]) : "l"(addr));
        }
      };

      using OperandC = typename base_t::OperandC;

      static __device__ inline void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
        mma::mma_instruction_t<MMA_M, MMA_N, MMA_K, mma::half, float> mma_instruction;
        mma_instruction(op_c.reg, op_a.reg, op_b.reg);
      }

      template <int M, int N, int ldc, bool dagger, class GmemOperandC, class op_t>
      static inline __device__ void store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm,
                                                  GmemOperandC &cc, const OperandC &op_c_real,
                                                  const OperandC &op_c_imag, op_t op)
      {
        base_t::template store_complex<M, N, ldc, dagger>(warp_row, warp_col, wrm, cc, op_c_real, op_c_imag, op);
      }
    };

  } // namespace hmma
} // namespace quda
