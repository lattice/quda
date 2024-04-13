#pragma once

#include <type_traits>
#include <quda_fp16.cuh>
#include <array.h>
#include <trove/ptr.h>
#include <mma_tensor_op/mma_instruction.cuh>
#include <tune_key.h>
#include <uint_to_char.h>

// Here we implement the architecture dependent part of MMA for Volta (sm70, the mma.sync.m8n8k4 instruction).

namespace quda
{
  namespace hmma
  {

    using half = mma::half;
    using half2 = mma::half2;

    template <> struct hmma_t<16, 16, 4, half, half2> {

      static __device__ __host__ constexpr int inline pad_size(int m) { return m == 48 ? 2 : 10; }

      static constexpr int MMA_M = 16;
      static constexpr int MMA_N = 16;
      static constexpr int MMA_K = 4;

      static constexpr int warp_size = 32;

      using compute_t = half;
      using load_t = half2;

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
          int idx_strided = k * MMA_K + wrm.quad_thread;
          int idx_contiguous = (warp_row * MMA_M + wrm.row_offset) / 2;
          int thread_offset_a = idx_strided * (lda / 2) + idx_contiguous;
          reg[0] = A[thread_offset_a + 0];
          reg[1] = A[thread_offset_a + 1];
        }

        template <class SmemObj>
        __device__ inline void load(const SmemObj &smem_obj, int k, int warp_row, const WarpRegisterMapping &wrm)
        {
          const unsigned *A = reinterpret_cast<const unsigned *>(smem_obj.ptr);
          int idx_strided = k * MMA_K + wrm.quad_thread;
          int idx_contiguous = (warp_row * MMA_M + wrm.row_offset) / 2;
          const int thread_offset_a = idx_strided * (SmemObj::ldn / 2) + idx_contiguous;
          reg[0] = A[thread_offset_a];
          reg[1] = A[thread_offset_a + 1];
        }

        __device__ inline void negate()
        {
          asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[0]));
          asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[1]));
        }
      };

      struct OperandB {

        unsigned reg[2];

        template <int ldb>
        __device__ inline void load(const void *smem, int k, int warp_col, const WarpRegisterMapping &wrm)
        {
          const unsigned *B = reinterpret_cast<const unsigned *>(smem);
          int idx_strided = k * MMA_K + wrm.quad_thread;
          int idx_contiguous = (warp_col * MMA_N + wrm.col_offset) / 2;
          int thread_offset_b = idx_strided * (ldb / 2) + idx_contiguous;
          reg[0] = B[thread_offset_b + 0];
          reg[1] = B[thread_offset_b + 1];
        }

        template <class SmemObj>
        __device__ inline void load(const SmemObj &smem_obj, int k, int warp_col, const WarpRegisterMapping &wrm)
        {
          const unsigned *B = reinterpret_cast<const unsigned *>(smem_obj.ptr);
          int idx_strided = k * MMA_K + wrm.quad_thread;
          int idx_contiguous = (warp_col * MMA_N + wrm.col_offset) / 2;
          const int thread_offset_b = idx_strided * (SmemObj::ldn / 2) + idx_contiguous;
          reg[0] = B[thread_offset_b];
          reg[1] = B[thread_offset_b + 1];
        }
      };

      template <typename real, int length> struct Structure {
        real v[length];
        __device__ inline const real &operator[](int i) const { return v[i]; }
        __device__ inline real &operator[](int i) { return v[i]; }
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
          half2 *C = reinterpret_cast<half2 *>(smem);

          const int idx_strided = warp_row * 16 + wrm.row_offset + (wrm.quad_thread % 2);
          const int idx_contiguous = warp_col * 8 + wrm.quad_col * 4 + (wrm.quad_thread / 2);

          int thread_offset_c = idx_strided * (ldc / 2) + idx_contiguous;
          C[thread_offset_c] = __floats2half2_rn(reg[0], reg[1]);

          thread_offset_c = (idx_strided + 2) * (ldc / 2) + idx_contiguous;
          C[thread_offset_c] = __floats2half2_rn(reg[2], reg[3]);

          thread_offset_c = idx_strided * (ldc / 2) + (idx_contiguous + 2);
          C[thread_offset_c] = __floats2half2_rn(reg[4], reg[5]);

          thread_offset_c = (idx_strided + 2) * (ldc / 2) + (idx_contiguous + 2);
          C[thread_offset_c] = __floats2half2_rn(reg[6], reg[7]);
        }

        template <class F> __device__ inline void abs_max(F &max)
        {
#pragma unroll
          for (int i = 0; i < 8; i++) { max = fmax(max, fabsf(reg[i])); }
        }
      };

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
        using store_t = typename GmemOperandC::store_type;

        const int row = warp_row + wrm.row_offset + (wrm.quad_thread % 2);
        const int col = warp_col + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

        constexpr bool fixed = GmemOperandC::fixed;
        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int m_index = row + (i % 2) * 2;
          int n_index = col + (i / 2) * 4;

          if constexpr (dagger) {
            using complex_t = complex<store_t>;
            auto ptr = reinterpret_cast<complex_t *>(cc.data());
            complex_t s;
            if constexpr (fixed) {
              auto scale = cc.get_scale();
              s = {f2i_round<store_t>(op_c_real.reg[i * 2 + 0] * scale),
                   f2i_round<store_t>(-op_c_imag.reg[i * 2 + 0] * scale)};
              op(&ptr[(n_index + 0) * ldc + m_index], s);
              s = {f2i_round<store_t>(op_c_real.reg[i * 2 + 1] * scale),
                   f2i_round<store_t>(-op_c_imag.reg[i * 2 + 1] * scale)};
              op(&ptr[(n_index + 1) * ldc + m_index], s);
            } else {
              s = {op_c_real.reg[i * 2 + 0], -op_c_imag.reg[i * 2 + 0]};
              op(&ptr[(n_index + 0) * ldc + m_index], s);
              s = {op_c_real.reg[i * 2 + 1], -op_c_imag.reg[i * 2 + 1]};
              op(&ptr[(n_index + 1) * ldc + m_index], s);
            }
          } else {
            using array_t = typename VectorType<store_t, 4>::type; // array<store_t, 4>;
            array_t *ptr = reinterpret_cast<array_t *>(cc.data());
            array_t s;
            if constexpr (fixed) {
              auto scale = cc.get_scale();
              s.x = f2i_round<store_t>(op_c_real.reg[i * 2 + 0] * scale);
              s.y = f2i_round<store_t>(op_c_imag.reg[i * 2 + 0] * scale);
              s.z = f2i_round<store_t>(op_c_real.reg[i * 2 + 1] * scale);
              s.w = f2i_round<store_t>(op_c_imag.reg[i * 2 + 1] * scale);
            } else {
              s.x = op_c_real.reg[i * 2 + 0];
              s.y = op_c_imag.reg[i * 2 + 0];
              s.z = op_c_real.reg[i * 2 + 1];
              s.w = op_c_imag.reg[i * 2 + 1];
            }
            if (!check_bounds || (m_index < M && n_index < N)) { op(&ptr[(m_index * ldc + n_index) / 2], s); }
          }
        }
      }
    };

  } // namespace hmma
} // namespace quda
