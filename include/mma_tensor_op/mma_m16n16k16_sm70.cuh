#pragma once

#include <mma.h>

#define USE_FP16_MMA_ACCUMULATE

namespace quda
{
  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 4;

  constexpr int warp_size = 32;

  struct WarpRegisterMapping {

    int quad_id;
    int quad_row;
    int quad_col;
    int quad_hilo;   // quad higher or lower.
    int quad_thread; // 0,1,2,3

    __device__ WarpRegisterMapping(int thread_id)
    {
      const int lane_id = thread_id & 31;
      const int octl_id = lane_id >> 2;
      quad_id = octl_id & 3;
      quad_row = quad_id & 1;
      quad_col = quad_id >> 1;
      quad_hilo = (octl_id >> 2) & 1;
      quad_thread = lane_id & 3;
    }
  };

  template <class T, int M, int N, int ldm, int ldn> struct SharedMemoryObject {

    T *ptr;

    __device__ inline T &operator()(int i, int j) { return ptr[i * ldm + j * ldn]; }

    __device__ inline const T &operator()(int i, int j) const { return ptr[i * ldm + j * ldn]; }

    template <class VecType> __device__ inline void vector_load(int i, int j, VecType vec)
    {
      VecType *ptr_ = reinterpret_cast<VecType *>(ptr);
      constexpr int vector_length = sizeof(VecType) / sizeof(T);
      ptr_[(i * ldm + j * ldn) / vector_length] = vec;
    }
  };

  template <int M, int N, int ldm, int ldn, class T> __device__ auto make_smem_obj(T *ptr_)
  {
    return SharedMemoryObject<T, M, N, ldm, ldn>{ptr_};
  }

  template <int stride> struct MmaOperandA {

    unsigned reg[2];

    __device__ inline void load(void *smem, int k, int warp_row, const WarpRegisterMapping &wrm)
    {
      unsigned *A = reinterpret_cast<unsigned *>(smem);
      const int idx_strided = k * 4 + wrm.quad_thread;
      const int idx_contiguous = warp_row * 8 + wrm.quad_row * 4 + wrm.quad_hilo * 2;
      const int thread_offset_a = idx_strided * stride + idx_contiguous;
      reg[0] = A[thread_offset_a + 0];
      reg[1] = A[thread_offset_a + 1];
    }

    __device__ inline void negate()
    {
      asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[0]));
      asm volatile("neg.f16x2 %0, %0;" : "+r"(reg[1]));
    }
  };

  template <int stride> struct MmaOperandB {

    unsigned reg[2];

    __device__ inline void load(void *smem, int k, int warp_col, const WarpRegisterMapping &wrm)
    {
      unsigned *B = reinterpret_cast<unsigned *>(smem);
      const int idx_strided = k * 4 + wrm.quad_thread;
      const int idx_contiguous = warp_col * 8 + wrm.quad_col * 4 + wrm.quad_hilo * 2;
      const int thread_offset_b = idx_strided * stride + idx_contiguous;
      reg[0] = B[thread_offset_b + 0];
      reg[1] = B[thread_offset_b + 1];
    }
  };

  template <int stride, class store_type> struct MmaOperandC {
  };

  template <int stride> struct MmaOperandC<stride, half> {

    using reg_type = unsigned;
    reg_type reg[4];

    __device__ MmaOperandC()
    {
#pragma unroll
      for (int i = 0; i < 4; i++) { reg[i] = 0; }
    }

    __device__ void store(void *smem, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
    {
      reg_type *C = reinterpret_cast<reg_type *>(smem);

      const int idx_strided = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + wrm.quad_thread;
      const int idx_contiguous = warp_col * 8 + wrm.quad_col * 4;
      const int thread_offset_c = idx_strided * stride + idx_contiguous;
#pragma unroll
      for (int i = 0; i < 4; i++) { C[thread_offset_c + i] = reg[i]; }
    }

    template <class F> __device__ void abs_max(F &max)
    {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const half2 h2 = __habs2(*(reinterpret_cast<const half2 *>(&(reg[i]))));
        max = fmax(max, h2.x);
        max = fmax(max, h2.y);
      }
    }
  };

  template <int stride> struct MmaOperandC<stride, float> {

    using reg_type = float;
    reg_type reg[8];

    __device__ MmaOperandC()
    {
#pragma unroll
      for (int i = 0; i < 8; i++) { reg[i] = 0; }
    }

    __device__ void store(void *smem, int warp_row, int warp_col, const WarpRegisterMapping &wrm)
    {
      half2 *C = reinterpret_cast<half2 *>(smem);

      const int idx_strided = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + (wrm.quad_thread % 2);
      const int idx_contiguous = warp_col * 8 + wrm.quad_col * 4 + (wrm.quad_thread / 2);

      int thread_offset_c = idx_strided * stride + idx_contiguous;
      C[thread_offset_c] = __floats2half2_rn(reg[0], reg[1]);

      thread_offset_c = (idx_strided + 2) * stride + idx_contiguous;
      C[thread_offset_c] = __floats2half2_rn(reg[2], reg[3]);

      thread_offset_c = idx_strided * stride + (idx_contiguous + 2);
      C[thread_offset_c] = __floats2half2_rn(reg[4], reg[5]);

      thread_offset_c = (idx_strided + 2) * stride + (idx_contiguous + 2);
      C[thread_offset_c] = __floats2half2_rn(reg[6], reg[7]);
    }
  };

  template <class TA, class TB, class TC> __device__ inline void gemm(const TA &op_a, const TB &op_b, TC &op_c)
  {
#ifdef USE_FP16_MMA_ACCUMULATE
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
                 : "+r"(op_c.reg[0]), "+r"(op_c.reg[1]), "+r"(op_c.reg[2]), "+r"(op_c.reg[3])
                 : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#else
    asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(op_c.reg[0]), "+f"(op_c.reg[1]), "+f"(op_c.reg[2]), "+f"(op_c.reg[3]), "+f"(op_c.reg[4]),
                   "+f"(op_c.reg[5]), "+f"(op_c.reg[6]), "+f"(op_c.reg[7])
                 : "r"(op_a.reg[0]), "r"(op_a.reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#endif
  }

  template <typename real, int length> struct Structure {
    real v[length];
    __host__ __device__ const real &operator[](int i) const { return v[i]; }
    __host__ __device__ real &operator[](int i) { return v[i]; }
  };

  template <int total_warp, int M, int N, int K, int M_PAD, int N_PAD, bool compute_max, class Accessor>
  __device__ inline float zmma_sync_gemm(half *sm_a_real, half *sm_a_imag, half *sm_b_real, half *sm_b_imag,
                                         Accessor accessor)
  {
#ifdef USE_FP16_MMA_ACCUMULATE
    using accumuate_reg_type = half;
#else
    using accumuate_reg_type = float;
#endif

    constexpr int tile_row_dim = M / WMMA_M; // number of tiles in the column dimension
    constexpr int tile_col_dim = N / WMMA_N; // number of tiles in the row dimension
    constexpr int tile_acc_dim = K / WMMA_K; // number of tiles in the row dimension

    static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
                  "Total number of tiles should be divisible by the number of warps.");

    constexpr int total_tile = tile_row_dim * tile_col_dim;
    constexpr int warp_cycle = total_tile / total_warp;

    const int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / warp_size;
    const WarpRegisterMapping wrm(thread_id);

    float max = 0.0f; // XXX: Accessor::Float

#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {

      // The logical warp assigned to each part of the matrix.
      const int logical_warp_index = warp_id * warp_cycle + c;
      const int warp_row = logical_warp_index / tile_col_dim;
      const int warp_col = logical_warp_index - warp_row * tile_col_dim;

      MmaOperandC<N_PAD / 2, accumuate_reg_type> op_c_real;
      MmaOperandC<N_PAD / 2, accumuate_reg_type> op_c_imag;

#pragma unroll
      for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

        const int k_idx = tile_k;

        MmaOperandA<M_PAD / 2> op_a_real;
        op_a_real.load(sm_a_real, k_idx, warp_row, wrm);
        MmaOperandA<M_PAD / 2> op_a_imag;
        op_a_imag.load(sm_a_imag, k_idx, warp_row, wrm);

        MmaOperandB<N_PAD / 2> op_b_real;
        op_b_real.load(sm_b_real, k_idx, warp_col, wrm);
        MmaOperandB<N_PAD / 2> op_b_imag;
        op_b_imag.load(sm_b_imag, k_idx, warp_col, wrm);

        gemm(op_a_real, op_b_real, op_c_real);
        gemm(op_a_imag, op_b_real, op_c_imag);
        gemm(op_a_real, op_b_imag, op_c_imag);
        // revert op_imag
        op_a_imag.negate();
        gemm(op_a_imag, op_b_imag, op_c_real);
      }
#ifdef USE_FP16_MMA_ACCUMULATE

      if (compute_max) {

        op_c_real.abs_max(max);
        op_c_imag.abs_max(max);

      } else {

        const int row = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + wrm.quad_thread;
        // const int col = warp_col * 16 + wrm.quad_col * 8;
        const int col = warp_col * 2 + wrm.quad_col;

        constexpr bool fixed = Accessor::fixed;
        using structure = typename std::conditional<fixed, Structure<short, 16>, Structure<float, 16>>::type;
        trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(&accessor.v[accessor.idx]));
        structure s;

#pragma unroll
        for (int i = 0; i < 4; i++) {
          const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real.reg[i])));
          const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag.reg[i])));
          if (fixed) {
            const float scale = accessor.scale;
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

        ptr_[row * (N / 8) + col] = s;
      }
#else
      static_assert(false, "fp32 accumulate hasn't been implemented.");
#endif
    }

    return max;
  }

} // namespace quda
