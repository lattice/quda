#pragma once

#include <mma.h>

//#define USE_FP16_HMMA_ACCUMULATE

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

    template <class F> __device__ void abs_max(F &max)
    {
#pragma unroll
      for (int i = 0; i < 8; i++) { max = fmax(max, fabsf(reg[i])); }
    }
  };

  template <class TA, class TB, class TC> __device__ inline void gemm(const TA &op_a, const TB &op_b, TC &op_c)
  {
#ifdef USE_FP16_HMMA_ACCUMULATE
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
#ifdef USE_FP16_HMMA_ACCUMULATE
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

      if (compute_max) {

        op_c_real.abs_max(max);
        op_c_imag.abs_max(max);

      } else {
#ifdef USE_FP16_HMMA_ACCUMULATE
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
#else // USE_FP16_HMMA_ACCUMULATE
        const int row = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + (wrm.quad_thread % 2);
        const int col = warp_col * 16 + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

        constexpr bool fixed = Accessor::fixed;
        using structure = typename std::conditional<fixed, Structure<short, 4>, Structure<float, 4>>::type;
        trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(&accessor.v[accessor.idx]));
        structure s;

#pragma unroll
        for (int i = 0; i < 4; i++) {
          const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real.reg[i])));
          const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag.reg[i])));
          if (fixed) {
            const float scale = accessor.scale;
            s[0] = short(op_c_real.reg[i * 2 + 0] * scale);
            s[1] = short(op_c_imag.reg[i * 2 + 0] * scale);
            s[2] = short(op_c_real.reg[i * 2 + 1] * scale);
            s[3] = short(op_c_imag.reg[i * 2 + 1] * scale);
          } else {
            s[0] = op_c_real.reg[i * 2 + 0];
            s[1] = op_c_imag.reg[i * 2 + 0];
            s[2] = op_c_real.reg[i * 2 + 1];
            s[3] = op_c_imag.reg[i * 2 + 1];
          }
          ptr_[((row + (i % 2) * 2) * N + (col + (i / 2) * 4)) / 2] = s;
        }
#endif
      }
    }

    return max;
  }

  __device__ __host__ constexpr int inline pad_size(int m) { return m == 48 ? 2 : 10; }

  template <int M, int N, int row_stride, int col_stride, bool dagger, class AccessorTo, class AccessorFrom>
  __device__ inline void load_cache(AccessorTo to_real, AccessorTo to_imag, AccessorFrom from)
  {
    for (int col = threadIdx.y; col < N; col += col_stride) {
      for (int row = threadIdx.z * 2; row < M; row += row_stride * 2) {
        if (!dagger) {
          auto x = from(row + 0, col);
          auto y = from(row + 1, col);
          to_real.vector_load(row, col, __floats2half2_rn(+x.real(), +y.real()));
          to_imag.vector_load(row, col, __floats2half2_rn(+x.imag(), +y.imag()));
        } else {
          auto x = from(col, row + 0);
          auto y = from(col, row + 1);
          to_real.vector_load(row, col, __floats2half2_rn(+x.real(), +y.real()));
          to_imag.vector_load(row, col, __floats2half2_rn(-x.imag(), -y.imag()));
        }
      }
    }
  }

  template <int M, int N, int row_stride, int col_stride, bool dagger, class SmemAccessor> struct GlobalMemoryLoader {

    static constexpr int m_dim = (M + row_stride * 2 - 1) / (row_stride * 2);
    static constexpr int n_dim = (N + col_stride - 1) / col_stride;

    SmemAccessor smem_real;
    SmemAccessor smem_imag;

    half2 reg_real[m_dim][n_dim];
    half2 reg_imag[m_dim][n_dim];

    __device__ GlobalMemoryLoader(SmemAccessor real_, SmemAccessor imag_) : smem_real(real_), smem_imag(imag_) {}

    template <class GmemAccessor> __device__ inline void g2r(GmemAccessor gmem)
    {
      for (int col = threadIdx.y; col < N; col += col_stride) {
        for (int row = threadIdx.z * 2; row < M; row += row_stride * 2) {
          if (!dagger) {
            auto x = gmem(row + 0, col);
            auto y = gmem(row + 1, col);
            reg_real[row / (row_stride * 2)][col / col_stride] = __floats2half2_rn(+x.real(), +y.real());
            reg_imag[row / (row_stride * 2)][col / col_stride] = __floats2half2_rn(+x.imag(), +y.imag());
          } else {
            auto x = gmem(col, row + 0);
            auto y = gmem(col, row + 1);
            reg_real[row / (row_stride * 2)][col / col_stride] = __floats2half2_rn(+x.real(), +y.real());
            reg_imag[row / (row_stride * 2)][col / col_stride] = __floats2half2_rn(-x.imag(), -y.imag());
          }
        }
      }
    }

    __device__ inline void r2s()
    {
      for (int col = threadIdx.y; col < N; col += col_stride) {
        for (int row = threadIdx.z * 2; row < M; row += row_stride * 2) {
          smem_real.vector_load(row, col, reg_real[row / (row_stride * 2)][col / col_stride]);
          smem_imag.vector_load(row, col, reg_imag[row / (row_stride * 2)][col / col_stride]);
        }
      }
    }
  };

  template <int N, int bM, int bN, int bK, int block_y, int block_z, bool a_dag, bool b_dag, bool compute_max_only,
            class A, class B, class C>
  __device__ inline float perform_mma(A aa, B bb, C cc)
  {
    constexpr int lda = bM + pad_size(bM);
    constexpr int ldb = bN + pad_size(bN);

    constexpr int n_row = block_z;
    constexpr int n_col = block_y;

    extern __shared__ half smem_ptr[];

    half *smem_a_real = smem_ptr;
    half *smem_a_imag = smem_a_real + lda * bK;
    half *smem_b_real = smem_a_imag + lda * bK;
    half *smem_b_imag = smem_b_real + ldb * bK;

    auto smem_obj_a_real = make_smem_obj<bM, bK, 1, lda>(smem_a_real);
    auto smem_obj_a_imag = make_smem_obj<bM, bK, 1, lda>(smem_a_imag);
    auto smem_obj_b_real = make_smem_obj<bN, bK, 1, ldb>(smem_b_real);
    auto smem_obj_b_imag = make_smem_obj<bN, bK, 1, ldb>(smem_b_imag);

    constexpr int total_warp = n_row * n_col / warp_size;

#ifdef USE_FP16_HMMA_ACCUMULATE
    using accumuate_reg_type = half;
#else
    using accumuate_reg_type = float;
#endif
    static_assert(bM % WMMA_M == 0, "bM must be divisible by WMMA_M.");
    static_assert(bN % WMMA_N == 0, "bM must be divisible by WMMA_N.");
    static_assert(bK % WMMA_K == 0, "bM must be divisible by WMMA_K.");

    constexpr int tile_row_dim = bM / WMMA_M; // number of tiles in the column dimension
    constexpr int tile_col_dim = bN / WMMA_N; // number of tiles in the row dimension
    constexpr int tile_acc_dim = bK / WMMA_K; // number of tiles in the accumulate dimension

    static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
                  "Total number of tiles should be divisible by the number of warps.");

    constexpr int total_tile = tile_row_dim * tile_col_dim;
    constexpr int warp_cycle = total_tile / total_warp;

    const int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / warp_size;
    const WarpRegisterMapping wrm(thread_id);

    float max = 0.0f; // XXX: Accessor::Float

    MmaOperandC<ldb / 2, accumuate_reg_type> op_c_real[warp_cycle];
    MmaOperandC<ldb / 2, accumuate_reg_type> op_c_imag[warp_cycle];

#pragma unroll
    for (int bk = 0; bk < N; bk += bK) {

      __syncthreads();

      auto aa_offset = [&](int i, int j) { return aa(i, j + bk); };

      auto bb_offset = [&](int i, int j) { return b_dag ? bb(i + bk, j) : bb(i, j + bk); };

      // load_cache<bM, bK, n_row, n_col, a_dag>(smem_obj_a_real, smem_obj_a_imag, aa_offset);
      // load_cache<bN, bK, n_row, n_col, b_dag>(smem_obj_b_real, smem_obj_b_imag, bb_offset);

      GlobalMemoryLoader<bM, bK, n_row, n_col, a_dag, decltype(smem_obj_a_real)> aa_loader(smem_obj_a_real,
                                                                                           smem_obj_a_imag);
      aa_loader.g2r(aa_offset);
      aa_loader.r2s();

      GlobalMemoryLoader<bN, bK, n_row, n_col, b_dag, decltype(smem_obj_b_real)> bb_loader(smem_obj_b_real,
                                                                                           smem_obj_b_imag);
      bb_loader.g2r(bb_offset);
      bb_loader.r2s();

      __syncthreads();

#pragma unroll
      for (int c = 0; c < warp_cycle; c++) {

        // The logical warp assigned to each part of the matrix.
        const int logical_warp_index = warp_id * warp_cycle + c;
        const int warp_row = logical_warp_index / tile_col_dim;
        const int warp_col = logical_warp_index - warp_row * tile_col_dim;

#pragma unroll
        for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

          const int k_idx = tile_k;

          MmaOperandA<lda / 2> op_a_real;
          op_a_real.load(smem_a_real, k_idx, warp_row, wrm);
          MmaOperandA<lda / 2> op_a_imag;
          op_a_imag.load(smem_a_imag, k_idx, warp_row, wrm);

          MmaOperandB<ldb / 2> op_b_real;
          op_b_real.load(smem_b_real, k_idx, warp_col, wrm);
          MmaOperandB<ldb / 2> op_b_imag;
          op_b_imag.load(smem_b_imag, k_idx, warp_col, wrm);

          gemm(op_a_real, op_b_real, op_c_real[c]);
          gemm(op_a_imag, op_b_real, op_c_imag[c]);
          gemm(op_a_real, op_b_imag, op_c_imag[c]);
          // revert op_imag
          op_a_imag.negate();
          gemm(op_a_imag, op_b_imag, op_c_real[c]);
        }
      }
    }

    // wrap up!
#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {

      if (compute_max_only) {

        op_c_real[c].abs_max(max);
        op_c_imag[c].abs_max(max);

      } else {

        // The logical warp assigned to each part of the matrix.
        const int logical_warp_index = warp_id * warp_cycle + c;
        const int warp_row = logical_warp_index / tile_col_dim;
        const int warp_col = logical_warp_index - warp_row * tile_col_dim;

#ifdef USE_FP16_HMMA_ACCUMULATE
        const int row = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + wrm.quad_thread;
        // const int col = warp_col * 16 + wrm.quad_col * 8;
        const int col = warp_col * 2 + wrm.quad_col;

        constexpr bool fixed = Accessor::fixed;
        using structure = typename std::conditional<fixed, Structure<short, 16>, Structure<float, 16>>::type;
        trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(&cc.v[cc.idx]));
        structure s;

#pragma unroll
        for (int i = 0; i < 4; i++) {
          const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real[c].reg[i])));
          const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag[c].reg[i])));
          if (fixed) {
            const float scale = cc.scale;
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
#else // USE_FP16_HMMA_ACCUMULATE
        const int row = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + (wrm.quad_thread % 2);
        const int col = warp_col * 16 + wrm.quad_col * 8 + (wrm.quad_thread / 2) * 2;

        constexpr bool fixed = C::fixed;
        using structure = typename std::conditional<fixed, Structure<short, 4>, Structure<float, 4>>::type;
        trove::coalesced_ptr<structure> ptr_(reinterpret_cast<structure *>(&cc.v[cc.idx]));
        structure s;

#pragma unroll
        for (int i = 0; i < 4; i++) {
          const half2 r2 = *(reinterpret_cast<const half2 *>(&(op_c_real[c].reg[i])));
          const half2 i2 = *(reinterpret_cast<const half2 *>(&(op_c_imag[c].reg[i])));
          if (fixed) {
            const float scale = cc.scale;
            s[0] = short(op_c_real[c].reg[i * 2 + 0] * scale);
            s[1] = short(op_c_imag[c].reg[i * 2 + 0] * scale);
            s[2] = short(op_c_real[c].reg[i * 2 + 1] * scale);
            s[3] = short(op_c_imag[c].reg[i * 2 + 1] * scale);
          } else {
            s[0] = op_c_real[c].reg[i * 2 + 0];
            s[1] = op_c_imag[c].reg[i * 2 + 0];
            s[2] = op_c_real[c].reg[i * 2 + 1];
            s[3] = op_c_imag[c].reg[i * 2 + 1];
          }
          ptr_[((row + (i % 2) * 2) * N + (col + (i / 2) * 4)) / 2] = s;
        }
#endif
      }
    }

    return max;
  }

} // namespace quda
