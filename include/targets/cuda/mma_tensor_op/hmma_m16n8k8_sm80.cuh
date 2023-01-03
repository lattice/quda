#pragma once

#include <type_traits>
#include <quda_fp16.cuh>
#include <array.h>

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

      struct WarpRegisterMapping {

        int warp_id;
        int lane_id;
        int group_id;
        int thread_id_in_group;

        __device__ WarpRegisterMapping(int thread_id) :
          warp_id(thread_id / warp_size), lane_id(thread_id & 31), group_id(lane_id >> 2), thread_id_in_group(lane_id & 3)
        {
        }
      };

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

      template <typename real, int length> struct Structure {
        real v[length];
        __device__ inline const real &operator[](int i) const { return v[i]; }
        __device__ inline real &operator[](int i) { return v[i]; }
      };

      struct OperandC {

        using reg_type = float;
        reg_type reg[4];

        __device__ inline OperandC() { zero(); }

        __device__ inline void zero()
        {
#pragma unroll
          for (int i = 0; i < 4; i++) { reg[i] = 0; }
        }

        __device__ inline void ax(float alpha)
        {
#pragma unroll
          for (int i = 0; i < 4; i++) { reg[i] *= alpha; }
        }

        template <int ldc>
        __device__ inline void store(void *smem, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
        {
          half2 *C = reinterpret_cast<half2 *>(smem);

          int idx_strided = warp_row * MMA_M + wrm.group_id;
          int idx_contiguous = warp_col * (MMA_N / 2) + wrm.thread_id_in_group;
          int thread_offset_c = idx_strided * (ldc / 2) + idx_contiguous;

          C[thread_offset_c] = __floats2half2_rn(reg[0], reg[1]);
          C[thread_offset_c + 8 * (ldc / 2)] = __floats2half2_rn(reg[2], reg[3]);
        }

        template <class F> __device__ inline void abs_max(F &max)
        {
#pragma unroll
          for (int i = 0; i < 4; i++) { max = fmax(max, fabsf(reg[i])); }
        }
      };

      static __device__ inline void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+f"(op_c.reg[0]), "+f"(op_c.reg[1]), "+f"(op_c.reg[2]), "+f"(op_c.reg[3])
                     : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]));
      }

      template <int M, int N, int ldc, bool dagger, class GmemOperandC, class op_t>
      static inline __device__ void store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm,
                                                  GmemOperandC &cc, const OperandC &op_c_real, const OperandC &op_c_imag, op_t op)
      {
        using store_t = typename GmemOperandC::store_type;

        int row = warp_row + wrm.group_id;
        int col = warp_col + wrm.thread_id_in_group * 2;

        constexpr bool fixed = GmemOperandC::fixed;
        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
        for (int i = 0; i < 2; i++) {
          int row_index = row + i * 8;
          int col_index = col;
          if (dagger) {
            using complex_t = complex<store_t>;
            auto ptr = reinterpret_cast<complex_t *>(cc.data());
            complex_t s;
            if constexpr (fixed) {
              auto scale = cc.get_scale();
              s = {static_cast<store_t>(op_c_real.reg[i * 2 + 0] * scale), -static_cast<store_t>(op_c_imag.reg[i * 2 + 0] * scale)};
              op(&ptr[(col_index + 0) * ldc + row_index], s);
              s = {static_cast<store_t>(op_c_real.reg[i * 2 + 1] * scale), -static_cast<store_t>(op_c_imag.reg[i * 2 + 1] * scale)};
              op(&ptr[(col_index + 1) * ldc + row_index], s);
            } else {
              s = {op_c_real.reg[i * 2 + 0], -op_c_imag.reg[i * 2 + 0]};
              op(&ptr[(col_index + 0) * ldc + row_index], s);
              s = {op_c_real.reg[i * 2 + 1], -op_c_imag.reg[i * 2 + 1]};
              op(&ptr[(col_index + 1) * ldc + row_index], s);
            }
          } else {
            using array_t = array<store_t, 4>;
            auto ptr = reinterpret_cast<array_t *>(cc.data());
            array_t s;
            if constexpr (fixed) {
              auto scale = cc.get_scale();
              s[0] = static_cast<store_t>(op_c_real.reg[i * 2 + 0] * scale);
              s[1] = static_cast<store_t>(op_c_imag.reg[i * 2 + 0] * scale);
              s[2] = static_cast<store_t>(op_c_real.reg[i * 2 + 1] * scale);
              s[3] = static_cast<store_t>(op_c_imag.reg[i * 2 + 1] * scale);
            } else {
              s[0] = op_c_real.reg[i * 2 + 0];
              s[1] = op_c_imag.reg[i * 2 + 0];
              s[2] = op_c_real.reg[i * 2 + 1];
              s[3] = op_c_imag.reg[i * 2 + 1];
            }
            if (!check_bounds || (row_index < M && col_index < N)) { op(&ptr[(row_index * ldc + col_index) / 2], s); }
          }
        }
      }

    };

  } // namespace hmma
} // namespace quda
