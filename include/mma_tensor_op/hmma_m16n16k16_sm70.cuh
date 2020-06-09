#pragma once

#include <type_traits>

// #define USE_FP16_HMMA_ACCUMULATE

constexpr QudaPrecision accumulate_precision()
{
#ifdef USE_FP16_HMMA_ACCUMULATE
  return QUDA_HALF_PRECISION;
#else
  return QUDA_SINGLE_PRECISION;
#endif
}

namespace quda
{
  namespace mma
  {
    __device__ __host__ constexpr int inline pad_size(int m) { return m == 48 ? 2 : 10; }

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 4;

    constexpr int warp_size = 32;

    struct WarpRegisterMapping {

      int warp_id;
      // int quad_row;
      int row_offset; // quad_row * 8 + quad_hilo * 4
      int col_offset; // quad_col * 8 + quad_hilo * 4
      int quad_col;
      // int quad_hilo;   // quad higher or lower.
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

    struct MmaOperandA {

      unsigned reg[2];

      template <int lda> __device__ inline void load(const void *smem, int k, int warp_row, const WarpRegisterMapping &wrm)
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

        // XXX the following code is more concise, but CUDA spills the registers if we use it
        // smem_obj.vector_store(idx_contiguous + 0, idx_strided, reg[0]);
        // smem_obj.vector_store(idx_contiguous + 2, idx_strided, reg[1]);
      }

      __device__ inline void negate()
      {
        asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[0]));
        asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[1]));
      }

    };

    struct MmaOperandB {

      unsigned reg[2];

      template <int ldb> __device__ inline void load(const void *smem, int k, int warp_col, const WarpRegisterMapping &wrm)
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

        // XXX the following code is more concise, but CUDA spills the registers if we use it
        // smem_obj.vector_store(idx_contiguous + 0, idx_strided, reg[0]);
        // smem_obj.vector_store(idx_contiguous + 2, idx_strided, reg[1]);
      }
    };

    template <class store_type> struct MmaOperandC {
    };

    template <> struct MmaOperandC<half> {

      using reg_type = unsigned;
      reg_type reg[4];

      __device__ inline MmaOperandC() { zero(); }

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
          const half2 h2 = __habs2(*(reinterpret_cast<const half2 *>(&(reg[i]))));
          max = fmax(max, h2.x);
          max = fmax(max, h2.y);
        }
      }
    };

    template <> struct MmaOperandC<float> {

      using reg_type = float;
      reg_type reg[8];

      __device__ inline MmaOperandC() { zero(); }

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

    template <class TA, class TB, class TC>
    __device__ inline typename std::enable_if<std::is_same<typename TC::reg_type, unsigned>::value, void>::type
    gemm(const TA &op_a, const TB &op_b, TC &op_c)
    {
      asm("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
          : "+r"(op_c.reg[0]), "+r"(op_c.reg[1]), "+r"(op_c.reg[2]), "+r"(op_c.reg[3])
          : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
    }

    template <class TA, class TB, class TC>
    __device__ inline typename std::enable_if<std::is_same<typename TC::reg_type, float>::value, void>::type
    gemm(const TA &op_a, const TB &op_b, TC &op_c)
    {
      asm("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
          "{%0,%1,%2,%3,%4,%5,%6,%7};"
          : "+f"(op_c.reg[0]), "+f"(op_c.reg[1]), "+f"(op_c.reg[2]), "+f"(op_c.reg[3]), "+f"(op_c.reg[4]),
            "+f"(op_c.reg[5]), "+f"(op_c.reg[6]), "+f"(op_c.reg[7])
          : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
    }

    template <typename real, int length> struct Structure {
      real v[length];
      __device__ inline const real &operator[](int i) const { return v[i]; }
      __device__ inline real &operator[](int i) { return v[i]; }
    };

    template <int ldc, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, unsigned>::value, void>::type
    store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc, const TC &op_c_real,
                  const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      const int row = warp_row + wrm.row_offset + wrm.quad_thread;
      const int col = warp_col + wrm.quad_col * 8;

      constexpr bool fixed = GmemOperandC::fixed;
      using structure = Structure<store_type, 16>;
      trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(cc.data()));
      structure s;

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

      ptr_[(row * ldc + col) / 8] = s;
    }

    template <int ldc, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, float>::value, void>::type
    store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc, const TC &op_c_real,
                  const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      const int row = warp_row + wrm.row_offset + (wrm.quad_thread % 2);
      const int col = warp_col + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

      constexpr bool fixed = GmemOperandC::fixed;
      using structure = Structure<store_type, 4>;
      trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(cc.data()));
      structure s;

#pragma unroll
      for (int i = 0; i < 4; i++) {
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
        ptr_[((row + (i % 2) * 2) * ldc + (col + (i / 2) * 4)) / 2] = s;
      }
    }

    template <int ldc, bool dagger, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, unsigned>::value, void>::type
    store_complex_atomic(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc,
                         const TC &op_c_real, const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      const int row = warp_row + wrm.row_offset + wrm.quad_thread;
      const int col = warp_col + wrm.quad_col * 8;

      constexpr bool fixed = GmemOperandC::fixed;

      using vector_type = typename vector<store_type, 2>::type;
      auto ptr = reinterpret_cast<vector_type *>(cc.data());

#pragma unroll
      for (int i = 0; i < 4; i++) {
        const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real.reg[i])));
        const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag.reg[i])));
        if (fixed) {
          auto scale = cc.scale;

          int m_index = row;
          int n_index = col + 2 * i;

          vector_type value;
          if (dagger) {
            value.x = +static_cast<store_type>(__half2float(r2.x) * scale);
            value.y = -static_cast<store_type>(__half2float(i2.x) * scale);
            atomicAdd(&ptr[(n_index + 0) * ldc + m_index], value);

            value.x = +static_cast<store_type>(__half2float(r2.y) * scale);
            value.y = -static_cast<store_type>(__half2float(i2.y) * scale);
            atomicAdd(&ptr[(n_index + 1) * ldc + m_index], value);
          } else {
            value.x = +static_cast<store_type>(__half2float(r2.x) * scale);
            value.y = +static_cast<store_type>(__half2float(i2.x) * scale);
            atomicAdd(&ptr[m_index * ldc + (n_index + 0)], value);

            value.x = +static_cast<store_type>(__half2float(r2.y) * scale);
            value.y = +static_cast<store_type>(__half2float(i2.y) * scale);
            atomicAdd(&ptr[m_index * ldc + (n_index + 1)], value);
          }

        } else {
          // TODO: Need to be added.
#if 0
          s[i * 4 + 0] = __half2float(r2.x);
          s[i * 4 + 1] = __half2float(i2.x);
          s[i * 4 + 2] = __half2float(r2.y);
          s[i * 4 + 3] = __half2float(i2.y);
#endif
        }
      }
    }

    template <int ldc, bool dagger, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, float>::value, void>::type
    store_complex_atomic(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc,
                         const TC &op_c_real, const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      const int row = warp_row + wrm.row_offset + (wrm.quad_thread % 2);
      const int col = warp_col + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

      constexpr bool fixed = GmemOperandC::fixed;

      using vector_type = typename vector<store_type, 2>::type;
      auto ptr = reinterpret_cast<vector_type *>(cc.data());

#pragma unroll
      for (int i = 0; i < 4; i++) {
        if (fixed) {
          auto scale = cc.scale;

          int m_index = row + (i % 2) * 2;
          int n_index = col + (i / 2) * 4;

          vector_type value;
          if (dagger) {
            value.x = +static_cast<store_type>(op_c_real.reg[i * 2 + 0] * scale);
            value.y = -static_cast<store_type>(op_c_imag.reg[i * 2 + 0] * scale);
            atomicAdd(&ptr[(n_index + 0) * ldc + m_index], value);

            value.x = +static_cast<store_type>(op_c_real.reg[i * 2 + 1] * scale);
            value.y = -static_cast<store_type>(op_c_imag.reg[i * 2 + 1] * scale);
            atomicAdd(&ptr[(n_index + 1) * ldc + m_index], value);
          } else {
            value.x = +static_cast<store_type>(op_c_real.reg[i * 2 + 0] * scale);
            value.y = +static_cast<store_type>(op_c_imag.reg[i * 2 + 0] * scale);
            atomicAdd(&ptr[m_index * ldc + (n_index + 0)], value);

            value.x = +static_cast<store_type>(op_c_real.reg[i * 2 + 1] * scale);
            value.y = +static_cast<store_type>(op_c_imag.reg[i * 2 + 1] * scale);
            atomicAdd(&ptr[m_index * ldc + (n_index + 1)], value);
          }

        } else {
          // TODO: Need to be added.
#if 0
          s[0] = op_c_real.reg[i * 2 + 0];
          s[1] = op_c_imag.reg[i * 2 + 0];
          s[2] = op_c_real.reg[i * 2 + 1];
          s[3] = op_c_imag.reg[i * 2 + 1];
#endif
        }
      }
    }

  } // namespace mma
} // namespace quda
