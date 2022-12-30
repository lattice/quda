#pragma once

#include <type_traits>
#include <quda_fp16.cuh>
#include <array.h>
#include <trove/ptr.h>

// Here we implement the architecture dependent part of MMA for Volta (sm70, the mma.sync.m8n8k4 instruction).

namespace quda
{
  namespace hmma
  {

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

#ifdef USE_FP16_HMMA_ACCUMULATE

      struct OperandC {

        using reg_type = unsigned;
        reg_type reg[4];

        __device__ inline OperandC() { zero(); }

        __device__ inline void zero()
        {
#pragma unroll
          for (int i = 0; i < 4; i++) { reg[i] = 0; }
        }

        __device__ inline void ax(float alpha)
        {
          half2 alpha_h2 = __float2half2_rn(alpha);
#pragma unroll
          for (int i = 0; i < 4; i++) {
            half2 &h2 = *(reinterpret_cast<half2 *>(&(reg[i])));
            h2 = __hmul2(alpha_h2, h2);
          }
        }

        template <int ldc>
        __device__ inline void store(void *smem, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
        {
          reg_type *C = reinterpret_cast<reg_type *>(smem);

          const int idx_strided = warp_row * 16 + wrm.row_offset + wrm.quad_thread;
          const int idx_contiguous = warp_col * 8 + wrm.quad_col * 4;
          const int thread_offset_c = idx_strided * (ldc / 2) + idx_contiguous;
#pragma unroll
          for (int i = 0; i < 4; i++) { C[thread_offset_c + i] = reg[i]; }
        }

        template <class F> __device__ inline void abs_max(F &max)
        {
#pragma unroll
          for (int i = 0; i < 4; i++) {
            const half2 h2 = habs2(*(reinterpret_cast<const half2 *>(&(reg[i]))));
            max = fmax(max, h2.x);
            max = fmax(max, h2.y);
          }
        }
      };

      static __device__ inline void mma(const OperandA &op_a, const OperandB &op_b, OperandC &op_c)
      {
        asm("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
            : "+r"(op_c.reg[0]), "+r"(op_c.reg[1]), "+r"(op_c.reg[2]), "+r"(op_c.reg[3])
            : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
      }

      template <int M, int N, int ldc, class GmemOperandC>
      static inline __device__ void store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm,
                                                  GmemOperandC &cc, const OperandC &op_c_real, const OperandC &op_c_imag)
      {
        using store_type = typename GmemOperandC::store_type;

        const int row = warp_row + wrm.row_offset + wrm.quad_thread;
        const int col = warp_col + wrm.quad_col * 8;

        constexpr bool fixed = GmemOperandC::fixed;
        using structure = Structure<store_type, 16>;
        trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(cc.data()));
        structure s;

        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
        for (int i = 0; i < 4; i++) {
          const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real.reg[i])));
          const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag.reg[i])));
          if (fixed) {
            auto scale = cc.scale;
            s[i * 4 + 0] = __half2short_rn(__half2float(r2.x) * scale);
            s[i * 4 + 1] = __half2short_rn(__half2float(i2.x) * scale);
            s[i * 4 + 2] = __half2short_rn(__half2float(r2.y) * scale);
            s[i * 4 + 3] = __half2short_rn(__half2float(i2.y) * scale);
          } else {
            s[i * 4 + 0] = __half2float(r2.x);
            s[i * 4 + 1] = __half2float(i2.x);
            s[i * 4 + 2] = __half2float(r2.y);
            s[i * 4 + 3] = __half2float(i2.y);
          }
        }
        if (!check_bounds || (row < M && col < N)) { ptr_[(row * ldc + col) / 8] = s; }
      }

      template <int M, int N, int ldc, bool dagger, class GmemOperandC>
      static inline __device__ void store_complex_atomic(int warp_row, int warp_col, const WarpRegisterMapping &wrm,
                                                         GmemOperandC &cc, const OperandC &op_c_real,
                                                         const OperandC &op_c_imag)
      {
        using store_type = typename GmemOperandC::store_type;

        const int row = warp_row + wrm.row_offset + wrm.quad_thread;
        const int col = warp_col + wrm.quad_col * 8;

        constexpr bool fixed = GmemOperandC::fixed;

        using array = array<store_type, 2>;
        auto ptr = reinterpret_cast<array *>(cc.data());

        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
        for (int i = 0; i < 4; i++) {
          const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real.reg[i])));
          const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag.reg[i])));
          auto scale = fixed ? cc.scale : 1.0f;

          int m_index = row;
          int n_index = col + 2 * i;

          array value {0};
          if (dagger) {
            if (!check_bounds || (n_index < M && m_index < N)) {
              value[0] = +static_cast<store_type>(__half2float(r2.x) * scale);
              value[1] = -static_cast<store_type>(__half2float(i2.x) * scale);
              atomic_fetch_add(&ptr[(n_index + 0) * ldc + m_index], value);

              value[0] = +static_cast<store_type>(__half2float(r2.y) * scale);
              value[1] = -static_cast<store_type>(__half2float(i2.y) * scale);
              atomic_fetch_add(&ptr[(n_index + 1) * ldc + m_index], value);
            }
          } else {
            if (!check_bounds || (m_index < M && n_index < N)) {
              value[0] = +static_cast<store_type>(__half2float(r2.x) * scale);
              value[1] = +static_cast<store_type>(__half2float(i2.x) * scale);
              atomic_fetch_add(&ptr[m_index * ldc + (n_index + 0)], value);

              value[0] = +static_cast<store_type>(__half2float(r2.y) * scale);
              value[1] = +static_cast<store_type>(__half2float(i2.y) * scale);
              atomic_fetch_add(&ptr[m_index * ldc + (n_index + 1)], value);
            }
          }
        }
      }

#else

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
        asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
            "{%0,%1,%2,%3,%4,%5,%6,%7};"
            : "+f"(op_c.reg[0]), "+f"(op_c.reg[1]), "+f"(op_c.reg[2]), "+f"(op_c.reg[3]), "+f"(op_c.reg[4]),
              "+f"(op_c.reg[5]), "+f"(op_c.reg[6]), "+f"(op_c.reg[7])
            : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
      }

      template <int M, int N, int ldc, class GmemOperandC>
      static inline __device__ void store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm,
                                                  GmemOperandC &cc, const OperandC &op_c_real, const OperandC &op_c_imag)
      {
        using store_type = typename GmemOperandC::store_type;

        const int row = warp_row + wrm.row_offset + (wrm.quad_thread % 2);
        const int col = warp_col + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

        constexpr bool fixed = GmemOperandC::fixed;
        using structure = Structure<store_type, 4>;
        trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(cc.data()));
        structure s;

        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int m_index = row + (i % 2) * 2;
          int n_index = col + (i / 2) * 4;

          if (fixed) {
            auto scale = cc.scale;
            s[0] = static_cast<store_type>(op_c_real.reg[i * 2 + 0] * scale);
            s[1] = static_cast<store_type>(op_c_imag.reg[i * 2 + 0] * scale);
            s[2] = static_cast<store_type>(op_c_real.reg[i * 2 + 1] * scale);
            s[3] = static_cast<store_type>(op_c_imag.reg[i * 2 + 1] * scale);
          } else {
            s[0] = op_c_real.reg[i * 2 + 0];
            s[1] = op_c_imag.reg[i * 2 + 0];
            s[2] = op_c_real.reg[i * 2 + 1];
            s[3] = op_c_imag.reg[i * 2 + 1];
          }
          if (!check_bounds || (m_index < M && n_index < N)) { ptr_[(m_index * ldc + n_index) / 2] = s; }
        }
      }

      template <int M, int N, int ldc, bool dagger, class GmemOperandC>
      static inline __device__ void store_complex_atomic(int warp_row, int warp_col, const WarpRegisterMapping &wrm,
                                                         GmemOperandC &cc, const OperandC &op_c_real,
                                                         const OperandC &op_c_imag)
      {
        using store_type = typename GmemOperandC::store_type;

        const int row = warp_row + wrm.row_offset + (wrm.quad_thread % 2);
        const int col = warp_col + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

        constexpr bool fixed = GmemOperandC::fixed;

        using array = array<store_type, 2>;
        auto ptr = reinterpret_cast<array *>(cc.data());

        constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
        for (int i = 0; i < 4; i++) {
          int m_index = row + (i % 2) * 2;
          int n_index = col + (i / 2) * 4;

          array value {0};

          auto scale = fixed ? cc.scale : 1.0f;
          if (dagger) {
            if (!check_bounds || (n_index < M && m_index < N)) {
              value[0] = +static_cast<store_type>(round(op_c_real.reg[i * 2 + 0] * scale));
              value[1] = -static_cast<store_type>(round(op_c_imag.reg[i * 2 + 0] * scale));
              atomic_fetch_add(&ptr[(n_index + 0) * ldc + m_index], value);

              value[0] = +static_cast<store_type>(round(op_c_real.reg[i * 2 + 1] * scale));
              value[1] = -static_cast<store_type>(round(op_c_imag.reg[i * 2 + 1] * scale));
              atomic_fetch_add(&ptr[(n_index + 1) * ldc + m_index], value);
            }
          } else {
            if (!check_bounds || (m_index < M && n_index < N)) {
              value[0] = +static_cast<store_type>(round(op_c_real.reg[i * 2 + 0] * scale));
              value[1] = +static_cast<store_type>(round(op_c_imag.reg[i * 2 + 0] * scale));
              atomic_fetch_add(&ptr[m_index * ldc + (n_index + 0)], value);

              value[0] = +static_cast<store_type>(round(op_c_real.reg[i * 2 + 1] * scale));
              value[1] = +static_cast<store_type>(round(op_c_imag.reg[i * 2 + 1] * scale));
              atomic_fetch_add(&ptr[m_index * ldc + (n_index + 1)], value);
            }
          }
        }
      }

#endif
    };

  } // namespace hmma
} // namespace quda
