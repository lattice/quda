#pragma once

#include <type_traits>
#include <quda_fp16.cuh>
#include <array.h>
#include <convert.h>

// This macro determines whether or not we are using the fp16 accumulation of the MMA instruction.
// #define USE_FP16_HMMA_ACCUMULATE

constexpr QudaPrecision accumulate_precision()
{
#ifdef USE_FP16_HMMA_ACCUMULATE
  return QUDA_HALF_PRECISION;
#else
  return QUDA_SINGLE_PRECISION;
#endif
}

// Here we implement the architecture dependent part of MMA for Turing/Ampere (sm75/sm80, the mma.sync.m16n8k8 instruction).

namespace quda
{

  namespace mma
  {
    __device__ __host__ constexpr int inline pad_size(int m) { return m == 192 ? 0 : 8; }

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 8;

    constexpr int warp_size = 32;

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

    struct MmaOperandA {

      unsigned reg[2];

      template <int lda>
      __device__ inline void load(const void *smem, int k, int warp_row, const WarpRegisterMapping &wrm)
      {
        const unsigned *A = reinterpret_cast<const unsigned *>(smem);
        int idx_strided = k * MMA_K + (wrm.lane_id & 7);
        int idx_contiguous = warp_row * (MMA_M / 2) + ((wrm.lane_id >> 3) & 1) * 4;
        const unsigned *addr = &A[idx_strided * (lda / 2) + idx_contiguous];

        asm volatile("ldmatrix.sync.aligned.m8n8.trans.x2.b16 {%0,%1}, [%2];" : "=r"(reg[0]), "=r"(reg[1]) : "l"(addr));
      }

      template <class SmemObj>
      __device__ inline void load(const SmemObj &smem_obj, int k, int warp_row, const WarpRegisterMapping &wrm)
      {
        const unsigned *A = reinterpret_cast<const unsigned *>(smem_obj.ptr);
        int idx_strided = k * MMA_K + (wrm.lane_id & 7);
        int idx_contiguous = warp_row * (MMA_M / 2) + ((wrm.lane_id >> 3) & 1) * 4;
        const unsigned *addr = &A[idx_strided * (SmemObj::ldn / 2) + idx_contiguous];

        asm volatile("ldmatrix.sync.aligned.m8n8.trans.x2.b16 {%0,%1}, [%2];" : "=r"(reg[0]), "=r"(reg[1]) : "l"(addr));
      }

      __device__ inline void negate()
      {
        asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[0]));
        asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[1]));
      }
    };

    struct MmaOperandB {

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

    template <class store_type> struct MmaOperandC {
    };

    template <> struct MmaOperandC<half> {

      using reg_type = unsigned;
      reg_type reg[2];

      __device__ inline MmaOperandC() { zero(); }

      __device__ inline void zero()
      {
#pragma unroll
        for (int i = 0; i < 2; i++) { reg[i] = 0; }
      }

      __device__ inline void ax(float alpha)
      {
        half2 alpha_h2 = __float2half2_rn(alpha);
#pragma unroll
        for (int i = 0; i < 2; i++) {
          half2 &h2 = *(reinterpret_cast<half2 *>(&(reg[i])));
          h2 = __hmul2(alpha_h2, h2);
        }
      }

      template <int ldc>
      __device__ inline void store(void *smem, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
      {
        reg_type *C = reinterpret_cast<reg_type *>(smem);

        int idx_strided = warp_row * MMA_M + wrm.group_id;
        int idx_contiguous = warp_col * (MMA_N / 2) + wrm.thread_id_in_group;
        int thread_offset_c = idx_strided * (ldc / 2) + idx_contiguous;

        C[thread_offset_c] = reg[0];
        C[thread_offset_c + 8 * (ldc / 2)] = reg[1];
      }

      template <class F> __device__ inline void abs_max(F &max)
      {
#pragma unroll
        for (int i = 0; i < 2; i++) {
          const half2 h2 = habs2(*(reinterpret_cast<const half2 *>(&(reg[i]))));
          max = fmax(max, h2.x);
          max = fmax(max, h2.y);
        }
      }
    };

    template <> struct MmaOperandC<float> {

      using reg_type = float;
      reg_type reg[4];

      __device__ inline MmaOperandC() { zero(); }

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

    template <class TA, class TB, class TC>
    __device__ inline typename std::enable_if<std::is_same<typename TC::reg_type, unsigned>::value, void>::type
    gemm(const TA &op_a, const TB &op_b, TC &op_c)
    {
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};"
                   : "+r"(op_c.reg[0]), "+r"(op_c.reg[1])
                   : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]));
    }

    template <class TA, class TB, class TC>
    __device__ inline typename std::enable_if<std::is_same<typename TC::reg_type, float>::value, void>::type
    gemm(const TA &op_a, const TB &op_b, TC &op_c)
    {
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                   : "+f"(op_c.reg[0]), "+f"(op_c.reg[1]), "+f"(op_c.reg[2]), "+f"(op_c.reg[3])
                   : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]));
    }

    template <typename real, int length> struct Structure {
      real v[length];
      __device__ inline const real &operator[](int i) const { return v[i]; }
      __device__ inline real &operator[](int i) { return v[i]; }
    };

    template <int M, int N, int ldc, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, unsigned>::value, void>::type
    store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc, const TC &op_c_real,
                  const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      int row = warp_row + wrm.group_id;
      int col = warp_col + wrm.thread_id_in_group * 2;

      constexpr bool fixed = GmemOperandC::fixed;
      using structure = typename VectorType<store_type, 4>::type;
      auto ptr = reinterpret_cast<structure *>(cc.data());
      structure s;

      constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
      for (int i = 0; i < 2; i++) {
        const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real.reg[i])));
        const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag.reg[i])));
        if (fixed) {
          auto scale = cc.scale;
          s.x = f2i_round<store_type>(__half2float(r2.x) * scale);
          s.y = f2i_round<store_type>(__half2float(i2.x) * scale);
          s.z = f2i_round<store_type>(__half2float(r2.y) * scale);
          s.w = f2i_round<store_type>(__half2float(i2.y) * scale);
        } else {
          s.x = __half2float(r2.x);
          s.y = __half2float(i2.x);
          s.z = __half2float(r2.y);
          s.w = __half2float(i2.y);
        }
        if (!check_bounds || (row + i * 8 < M && col < N)) { ptr[((row + i * 8) * ldc + col) / 2] = s; }
      }
    }

    template <int M, int N, int ldc, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, float>::value, void>::type
    store_complex(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc, const TC &op_c_real,
                  const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      int row = warp_row + wrm.group_id;
      int col = warp_col + wrm.thread_id_in_group * 2;

      constexpr bool fixed = GmemOperandC::fixed;
      using structure = typename VectorType<store_type, 4>::type;
      auto ptr = reinterpret_cast<structure *>(cc.data());
      structure s;

      constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
      for (int i = 0; i < 2; i++) {
        if (fixed) {
          auto scale = cc.scale;
          s.x = f2i_round<store_type>(op_c_real.reg[i * 2 + 0] * scale);
          s.y = f2i_round<store_type>(op_c_imag.reg[i * 2 + 0] * scale);
          s.z = f2i_round<store_type>(op_c_real.reg[i * 2 + 1] * scale);
          s.w = f2i_round<store_type>(op_c_imag.reg[i * 2 + 1] * scale);
        } else {
          s.x = op_c_real.reg[i * 2 + 0];
          s.y = op_c_imag.reg[i * 2 + 0];
          s.z = op_c_real.reg[i * 2 + 1];
          s.w = op_c_imag.reg[i * 2 + 1];
        }
        if (!check_bounds || (row + i * 8 < M && col < N)) { ptr[((row + i * 8) * ldc + col) / 2] = s; }
      }
    }

    template <int M, int N, int ldc, bool dagger, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, unsigned>::value, void>::type
    store_complex_atomic(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc,
                         const TC &op_c_real, const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      int row = warp_row + wrm.group_id;
      int col = warp_col + wrm.thread_id_in_group * 2;

      constexpr bool fixed = GmemOperandC::fixed;
      static_assert(fixed, "This method should only be called with a fixed type");

      using array = array<store_type, 2>;
      auto ptr = reinterpret_cast<array *>(cc.data());

      constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
      for (int i = 0; i < 2; i++) {
        const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real.reg[i])));
        const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag.reg[i])));
        auto scale = fixed ? cc.scale : 1.0f;

        int m_index = row + 8 * i;
        int n_index = col;

        array value{0};
        if (dagger) {
          if (!check_bounds || (n_index < M && m_index < N)) {
            value[0] = f2i_round<store_type>(__half2float(r2.x) * scale);
            value[1] = f2i_round<store_type>(-__half2float(i2.x) * scale);
            atomic_fetch_add(&ptr[(n_index + 0) * ldc + m_index], value);

            value[0] = f2i_round<store_type>(__half2float(r2.y) * scale);
            value[1] = f2i_round<store_type>(-__half2float(i2.y) * scale);
            atomic_fetch_add(&ptr[(n_index + 1) * ldc + m_index], value);
          }
        } else {
          if (!check_bounds || (m_index < M && n_index < N)) {
            value[0] = f2i_round<store_type>(__half2float(r2.x) * scale);
            value[1] = f2i_round<store_type>(__half2float(i2.x) * scale);
            atomic_fetch_add(&ptr[m_index * ldc + (n_index + 0)], value);

            value[0] = f2i_round<store_type>(__half2float(r2.y) * scale);
            value[1] = f2i_round<store_type>(__half2float(i2.y) * scale);
            atomic_fetch_add(&ptr[m_index * ldc + (n_index + 1)], value);
          }
        }
      }
    }

    template <int M, int N, int ldc, bool dagger, class TC, class GmemOperandC>
    inline __device__ typename std::enable_if<std::is_same<typename TC::reg_type, float>::value, void>::type
    store_complex_atomic(int warp_row, int warp_col, const WarpRegisterMapping &wrm, GmemOperandC &cc,
                         const TC &op_c_real, const TC &op_c_imag)
    {
      using store_type = typename GmemOperandC::store_type;

      int row = warp_row + wrm.group_id;
      int col = warp_col + wrm.thread_id_in_group * 2;

      constexpr bool fixed = GmemOperandC::fixed;
      static_assert(fixed, "This method should only be called with a fixed type");

      using array = array<store_type, 2>;
      auto ptr = reinterpret_cast<array *>(cc.data());

      constexpr bool check_bounds = !((M % MMA_M == 0) && (N % MMA_N == 0));

#pragma unroll
      for (int i = 0; i < 2; i++) {
        auto scale = fixed ? cc.scale : 1.0f;

        int m_index = row + i * 8;
        int n_index = col;

        array value{0};
        if (dagger) {
          if (!check_bounds || (n_index < M && m_index < N)) {
            value[0] = f2i_round<store_type>(op_c_real.reg[i * 2 + 0] * scale);
            value[1] = f2i_round<store_type>(-op_c_imag.reg[i * 2 + 0] * scale);
            atomic_fetch_add(&ptr[(n_index + 0) * ldc + m_index], value);

            value[0] = f2i_round<store_type>(op_c_real.reg[i * 2 + 1] * scale);
            value[1] = f2i_round<store_type>(-op_c_imag.reg[i * 2 + 1] * scale);
            atomic_fetch_add(&ptr[(n_index + 1) * ldc + m_index], value);
          }
        } else {
          if (!check_bounds || (m_index < M && n_index < N)) {
            value[0] = f2i_round<store_type>(op_c_real.reg[i * 2 + 0] * scale);
            value[1] = f2i_round<store_type>(op_c_imag.reg[i * 2 + 0] * scale);
            atomic_fetch_add(&ptr[m_index * ldc + (n_index + 0)], value);

            value[0] = f2i_round<store_type>(op_c_real.reg[i * 2 + 1] * scale);
            value[1] = f2i_round<store_type>(op_c_imag.reg[i * 2 + 1] * scale);
            atomic_fetch_add(&ptr[m_index * ldc + (n_index + 1)], value);
          }
        }
      }
    }

  } // namespace mma
} // namespace quda
