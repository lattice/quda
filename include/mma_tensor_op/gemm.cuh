#pragma once

#include <algorithm>

#if (__COMPUTE_CAPABILITY__ == 700)

#include <mma_tensor_op/hmma_m16n16k16_sm70.cuh>

#else

#include <mma_tensor_op/hmma_m16n8k8_sm80.cuh>

#endif

namespace quda
{
  namespace mma
  {
    // return the size of the shared memory needed for MMA with block shape bM, bN, bK.
    constexpr int shared_memory_bytes(int bM, int bN, int bK)
    {
      return (bM + pad_size(bM) + bN + pad_size(bN)) * bK * 2 * sizeof(half);
    }

    // A shared memory object that bakes with it a 2-d index access method.
    template <class T, int M_, int N_, int ldm_, int ldn_> struct SharedMemoryObject {

      static constexpr int M = M_;
      static constexpr int N = N_;
      static constexpr int ldm = ldm_;
      static constexpr int ldn = ldn_;

      T *ptr;

      __device__ inline SharedMemoryObject(T *ptr_) : ptr(ptr_) { }

      __device__ inline T &operator()(int i, int j) { return ptr[i * ldm + j * ldn]; }

      __device__ inline const T &operator()(int i, int j) const { return ptr[i * ldm + j * ldn]; }

      template <class VecType> __device__ inline void vector_load(int i, int j, VecType vec)
      {
        VecType *ptr_ = reinterpret_cast<VecType *>(ptr);
        constexpr int vector_length = sizeof(VecType) / sizeof(T);
        ptr_[(i * ldm + j * ldn) / vector_length] = vec;
      }
    };

    template <int M, int N, int ldm, int ldn, class T> __device__ inline auto make_smem_obj(T *ptr_)
    {
      return SharedMemoryObject<T, M, N, ldm, ldn> {ptr_};
    }

    /**
     * A loader object that loads data from global memory to registers (g2r), and then to shared memory (r2s)
     * M, N: the global memory matrix size, for bound check only
     * bM, bN: the shared memory matrix size
     * block_y, block_z: CTA dimension in the y and z directions
     * transpose: the global memory always assumes a column-major order, transpose = true if the destination
          shared memory is row-major.
     */
    template <int M, int N, int bM, int bN, int block_y, int block_z, bool transpose> struct GlobalMemoryLoader {

      static constexpr int m_stride_n = block_y * 2;
      static constexpr int n_stride_n = block_z * 1;
      static constexpr int m_stride_t = block_z * 2;
      static constexpr int n_stride_t = block_y * 1;

      static constexpr int register_count
        = std::max(((bN + n_stride_n - 1) / n_stride_n) * ((bM + m_stride_n - 1) / m_stride_n),
                   ((bN + n_stride_t - 1) / n_stride_t) * ((bM + m_stride_t - 1) / m_stride_t));

      half2 reg_real[register_count];
      half2 reg_imag[register_count];

      /**
       * ld: leading dimension of global memory
       * dagger: if we need to store daggered (tranpose and hermision conjugate)
       */
      template <int ld, bool dagger, class GmemAccessor>
      __device__ inline void g2r(const GmemAccessor &gmem, int m_offset, int n_offset)
      {
        auto p = gmem.data();
        auto scale_inv = gmem.scale_inv;
        constexpr bool fixed = GmemAccessor::fixed;

        static constexpr int n_stride = transpose == dagger ? block_y * 1 : block_z * 1;
        static constexpr int m_stride = transpose == dagger ? block_z * 2 : block_y * 2;
        int n_thread_offset = transpose == dagger ? threadIdx.y * 1 : threadIdx.z * 1;
        int m_thread_offset = transpose == dagger ? threadIdx.z * 2 : threadIdx.y * 2;

        static constexpr int n_dim = (bN + n_stride - 1) / n_stride;
        static constexpr int m_dim = (bM + m_stride - 1) / m_stride;

        static constexpr bool check_global_bound = !(M % bM == 0 && N % bN == 0);
        static constexpr bool check_shared_bound = !(bM % m_stride == 0 && bN % n_stride == 0);

#pragma unroll
        for (int n = 0; n < n_dim; n++) {

#pragma unroll
          for (int m = 0; m < m_dim; m++) {

            int n_idx_blk = n * n_stride + n_thread_offset;
            int m_idx_blk = m * m_stride + m_thread_offset;

            if (!check_shared_bound || (m_idx_blk < bM && n_idx_blk < bN)) {

              int n_idx = n_idx_blk + n_offset;
              int m_idx = m_idx_blk + m_offset;

              if (!check_global_bound || (n_idx < N && m_idx < M)) {

                if (transpose == dagger) {

                  auto xx = p[(m_idx + 0) * ld + n_idx];
                  auto yy = p[(m_idx + 1) * ld + n_idx];

                  if (fixed) {
                    reg_real[m * n_dim + n] = __floats2half2_rn(scale_inv * xx.real(), scale_inv * yy.real());
                    auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
                    reg_imag[m * n_dim + n] = __floats2half2_rn(scale_inv_conj * xx.imag(), scale_inv_conj * yy.imag());
                  } else {
                    reg_real[m * n_dim + n] = __floats2half2_rn(+xx.real(), +yy.real());
                    reg_imag[m * n_dim + n]
                      = __floats2half2_rn(dagger ? -xx.imag() : +xx.imag(), dagger ? -yy.imag() : +yy.imag());
                  }

                } else {

                  using store_type = typename GmemAccessor::store_type;
                  using store_vector_type = typename VectorType<store_type, 4>::type;
                  store_vector_type v = *reinterpret_cast<store_vector_type *>(&p[n_idx * ld + m_idx]);

                  if (fixed) {
                    reg_real[m * n_dim + n] = __floats2half2_rn(scale_inv * v.x, scale_inv * v.z);
                    auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
                    reg_imag[m * n_dim + n] = __floats2half2_rn(scale_inv_conj * v.y, scale_inv_conj * v.w);
                  } else {
                    reg_real[m * n_dim + n] = __floats2half2_rn(+v.x, +v.z);
                    reg_imag[m * n_dim + n] = __floats2half2_rn(dagger ? -v.y : +v.y, dagger ? -v.w : +v.w);
                  }
                }

              } else {

                reg_real[m * n_dim + n] = __half2half2(0);
                reg_imag[m * n_dim + n] = __half2half2(0);
              }
            }
          }
        }
      }

      template <bool dagger, class SmemObj> __device__ inline void r2s(SmemObj &smem_real, SmemObj &smem_imag)
      {
        static constexpr int n_stride = transpose == dagger ? block_y * 1 : block_z * 1;
        static constexpr int m_stride = transpose == dagger ? block_z * 2 : block_y * 2;
        int n_thread_offset = transpose == dagger ? threadIdx.y * 1 : threadIdx.z * 1;
        int m_thread_offset = transpose == dagger ? threadIdx.z * 2 : threadIdx.y * 2;

        static constexpr int n_dim = (bN + n_stride - 1) / n_stride;
        static constexpr int m_dim = (bM + m_stride - 1) / m_stride;

#pragma unroll
        for (int n = 0; n < n_dim; n++) {
#pragma unroll
          for (int m = 0; m < m_dim; m++) {
            const int n_idx = n * n_stride + n_thread_offset;
            const int m_idx = m * m_stride + m_thread_offset;
            if (m_idx < bM && n_idx < bN) {
              smem_real.vector_load(m_idx, n_idx, reg_real[m * n_dim + n]);
              smem_imag.vector_load(m_idx, n_idx, reg_imag[m * n_dim + n]);
            }
          }
        }
      }
    };

    /**
     * Perform the complex GEMM
     * @param m, n, k the corresponding offset in the M, N, and K direction
     */
    template <class A, class B, class C>
    __device__ inline void zgemm(const A &smem_obj_a_real, const A &smem_obj_a_imag, const B &smem_obj_b_real,
                                 const B &smem_obj_b_imag, C &op_c_real, C &op_c_imag, int m, int n, int k,
                                 const WarpRegisterMapping &wrm)
    {

      MmaOperandA op_a_real;
      op_a_real.load(smem_obj_a_real, k, m, wrm);
      MmaOperandA op_a_imag;
      op_a_imag.load(smem_obj_a_imag, k, m, wrm);

      MmaOperandB op_b_real;
      op_b_real.load(smem_obj_b_real, k, n, wrm);
      MmaOperandB op_b_imag;
      op_b_imag.load(smem_obj_b_imag, k, n, wrm);

      gemm(op_a_real, op_b_real, op_c_real);
      gemm(op_a_imag, op_b_real, op_c_imag);
      gemm(op_a_real, op_b_imag, op_c_imag);
      // negate op_imag
      op_a_imag.negate();
      gemm(op_a_imag, op_b_imag, op_c_real);
    }

    // A wrapper that wraps the OperandC objects, together with the various methods to loop over it
    template <class OperandC, int warp_cycle, int tile_col_dim> struct MmaAccumulator {

      static constexpr int size = warp_cycle;

      OperandC op_c_real[warp_cycle];
      OperandC op_c_imag[warp_cycle];

      WarpRegisterMapping wrm;

      __device__ inline MmaAccumulator(int thread_id) : wrm(thread_id) { }

      __device__ inline void zero()
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          op_c_real[c].zero();
          op_c_imag[c].zero();
        }
      }

      __device__ inline void ax(float alpha)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          op_c_real[c].ax(alpha);
          op_c_imag[c].ax(alpha);
        }
      }

      template <int tile_acc_dim, class SmemObjA, class SmemObjB>
      __device__ inline void mma(const SmemObjA &smem_obj_a_real, const SmemObjA &smem_obj_a_imag,
                                 const SmemObjB &smem_obj_b_real, const SmemObjB &smem_obj_b_imag)
      {

#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {

          // The logical warp assigned to each part of the matrix.
          const int logical_warp_index = wrm.warp_id * warp_cycle + c;
          const int warp_row = logical_warp_index / tile_col_dim;
          const int warp_col = logical_warp_index - warp_row * tile_col_dim;

#pragma unroll
          for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
            zgemm(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real[c], op_c_imag[c],
                  warp_row, warp_col, tile_k, wrm);
          }
        }
      }

      template <int M, int N, int ldc, class C> __device__ inline void store(C &c_accessor, int m_offset, int n_offset)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {

          const int logical_warp_index = wrm.warp_id * warp_cycle + c;
          const int warp_row = logical_warp_index / tile_col_dim;
          const int warp_col = logical_warp_index - warp_row * tile_col_dim;

          const int warp_m_offset = warp_row * MMA_M + m_offset;
          const int warp_n_offset = warp_col * MMA_N + n_offset;

          store_complex<M, N, ldc>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real[c], op_c_imag[c]);
        }
      }

      template <int M, int N, int ldc, bool dagger, class C>
      __device__ inline void store_atomic(C &c_accessor, int m_offset, int n_offset)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {

          const int logical_warp_index = wrm.warp_id * warp_cycle + c;
          const int warp_row = logical_warp_index / tile_col_dim;
          const int warp_col = logical_warp_index - warp_row * tile_col_dim;

          const int warp_m_offset = warp_row * MMA_M + m_offset;
          const int warp_n_offset = warp_col * MMA_N + n_offset;

          store_complex_atomic<M, N, ldc, dagger>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real[c],
                                                  op_c_imag[c]);
        }
      }

      __device__ inline void abs_max(float &max)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          op_c_real[c].abs_max(max);
          op_c_imag[c].abs_max(max);
        }
      }
    };

    // A conceptual class that stores all the static MMA sizes.
    template <int M_, int N_, int K_, int lda_, int ldb_, int ldc_, int bM_, int bN_, int bK_, int block_y, int block_z>
    struct MmaConfig {

      static constexpr int M = M_; // the global matrix sizes
      static constexpr int N = N_;
      static constexpr int K = K_;
      static constexpr int lda = lda_; // the leading dimensions of A, B, and C in global memory
      static constexpr int ldb = ldb_;
      static constexpr int ldc = ldc_;
      static constexpr int bM = bM_; // The tile size for a CTA
      static constexpr int bN = bN_;
      static constexpr int bK = bK_;

      static constexpr int tile_row_dim = bM / MMA_M; // number of tiles in the column dimension
      static constexpr int tile_col_dim = bN / MMA_N; // number of tiles in the row dimension
      static constexpr int tile_acc_dim = bK / MMA_K; // number of tiles in the accumulate dimension

      static constexpr int smem_lda = bM + pad_size(bM); // shared memory leading dimensions
      static constexpr int smem_ldb = bN + pad_size(bN);

      static constexpr int n_row = block_y;
      static constexpr int n_col = block_z;

      static constexpr int total_warp = n_row * n_col / warp_size; // Total number of warps in the CTA

      static constexpr int total_tile = tile_row_dim * tile_col_dim; // Total number of tiles dividing operand C
      static constexpr int warp_cycle = total_tile / total_warp;     // Number of tiles each warp needs to calculate

      static constexpr bool a_transpose
        = false; // In our setup, specifically in the arch-dependent code, A is always column-major, while B is always row-major
      static constexpr bool b_transpose = true;

      // What accumulation type we are using for the MMA, fp16 (half) or fp32 (float)
#ifdef USE_FP16_HMMA_ACCUMULATE
      using accumuate_reg_type = half;
#else
      using accumuate_reg_type = float;
#endif
      static_assert(bM % MMA_M == 0, "bM must be divisible by MMA_M.");
      static_assert(bN % MMA_N == 0, "bN must be divisible by MMA_N.");
      static_assert(bK % MMA_K == 0, "bK must be divisible by MMA_K.");

      static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
                    "Total number of tiles should be divisible by the number of warps.");

      using SmemObjA = SharedMemoryObject<half, bM, bK, 1, smem_lda>;
      using SmemObjB = SharedMemoryObject<half, bN, bK, 1, smem_ldb>;

      using OperandC = MmaOperandC<accumuate_reg_type>;

      using Accumulator = MmaAccumulator<OperandC, warp_cycle, tile_col_dim>;

      using ALoader = GlobalMemoryLoader<M, K, bM, bK, n_row, n_col, a_transpose>;
      using BLoader = GlobalMemoryLoader<N, K, bN, bK, n_row, n_col, b_transpose>;

      // This is the most general MMA code: bM < M, bN < N, bK < K.
      // We divide M and N, and we stream over K, which means we need to store the accumulate register for ALL tiles.
      template <bool a_dagger, bool b_dagger, bool compute_max_only, class A, class B, class C>
      static __device__ inline float perform_mma_divide_k_yes(const A &a, const B &b, C &c, int m_offset, int n_offset)
      {
        float max = 0;

        extern __shared__ half smem_ptr[];

        SmemObjA smem_obj_a_real(smem_ptr);
        SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + smem_ldb * bK);

        Accumulator accumulator((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

        ALoader a_loader;
        BLoader b_loader;

        __syncthreads();
        a_loader.template g2r<lda, a_dagger>(a, m_offset, 0); // bk = 0
        a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);

        b_loader.template g2r<ldb, b_dagger>(b, n_offset, 0); // bk = 0
        b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
        __syncthreads();

#pragma unroll 1
        for (int bk = 0; bk < K; bk += bK) {

          if (bk + bK < K) { // Pipelining: retrieve data for the next K-stage and storage in the registers.
            a_loader.template g2r<lda, a_dagger>(a, m_offset, bk + bK);
            b_loader.template g2r<ldb, b_dagger>(b, n_offset, bk + bK);
          }

          accumulator.template mma<tile_acc_dim>(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);

          if (bk + bK < K) { // We have used all data in smem for this K-stage: move the data for the next K-stage
                             // to smem.
            __syncthreads();

            a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);
            b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);

            __syncthreads();
          }
        }

        // wrap up!
        if (compute_max_only) {
          accumulator.abs_max(max);
        } else {
          accumulator.template store<M, N, ldc>(c, m_offset, n_offset);
        }

        return max;
      }

      // This is version of the MMA code: bM < M, bN < N, bK >= K.
      // We divide M and N, but we have all K-stages, which means we do NOT need to store the accumulate register
      // for ALL tiles. This saves register usage.
      template <bool a_dagger, bool b_dagger, bool compute_max_only, class A, class B, class C>
      static __device__ inline float perform_mma_divide_k_no(const A &a, const B &b, C &c_accessor, int m_offset,
                                                             int n_offset)
      {
        float max = 0;

        extern __shared__ half smem_ptr[];

        SmemObjA smem_obj_a_real(smem_ptr);
        SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + smem_ldb * bK);

        OperandC op_c_real;
        OperandC op_c_imag;

        WarpRegisterMapping wrm((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

        ALoader a_loader;
        BLoader b_loader;

        __syncthreads();
        a_loader.template g2r<lda, a_dagger>(a, m_offset, 0);
        a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);

        b_loader.template g2r<ldb, b_dagger>(b, n_offset, 0);
        b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
        __syncthreads();

#pragma unroll 1
        for (int c = 0; c < warp_cycle; c++) {

          // The logical warp assigned to each part of the matrix.
          int logical_warp_index = wrm.warp_id * warp_cycle + c;
          int warp_row = logical_warp_index / tile_col_dim;
          int warp_col = logical_warp_index - warp_row * tile_col_dim;

          op_c_real.zero();
          op_c_imag.zero();

#pragma unroll 1
          for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
            zgemm(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real, op_c_imag, warp_row,
                  warp_col, tile_k, wrm);
          }

          if (compute_max_only) {

            op_c_real.abs_max(max);
            op_c_imag.abs_max(max);

          } else {

            int warp_m_offset = warp_row * MMA_M + m_offset;
            int warp_n_offset = warp_col * MMA_N + n_offset;

            store_complex<M, N, ldc>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real, op_c_imag);
          }
        }

        return max;
      }

      // This is version of the MMA code: bM < M, bN >= N, bK >= K.
      // We divide M, have the whole of N, but we have all K-stages, i.e. we have all of operand B in smem.
      // This means we do NOT need to store the accumulate register for ALL tiles. This saves register usage.
      // We loop over different sub-paritions of operand A.
      template <bool a_dagger, bool b_dagger, bool compute_max_only, class A, class B, class C>
      static __device__ inline float perform_mma_divide_b_no(const A &a, const B &b, C &c_accessor)
      {
        float max = 0;

        extern __shared__ half smem_ptr[];

        SmemObjA smem_obj_a_real(smem_ptr);
        SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + smem_ldb * bK);

        OperandC op_c_real;
        OperandC op_c_imag;

        WarpRegisterMapping wrm((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

        ALoader a_loader;
        BLoader b_loader;

        __syncthreads();
        a_loader.template g2r<lda, a_dagger>(a, 0, 0);
        a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);

        b_loader.template g2r<ldb, b_dagger>(b, 0, 0);
        b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
        __syncthreads();

#pragma unroll 1
        for (int a_m = 0; a_m < M; a_m += bM) {

          if (a_m + bM < M) { a_loader.template g2r<lda, a_dagger>(a, a_m + bM, 0); }

#pragma unroll 1
          for (int c = 0; c < warp_cycle; c++) {

            // The logical warp assigned to each part of the matrix.
            int logical_warp_index = wrm.warp_id * warp_cycle + c;
            int warp_row = logical_warp_index / tile_col_dim;
            int warp_col = logical_warp_index - warp_row * tile_col_dim;

            op_c_real.zero();
            op_c_imag.zero();

#pragma unroll 1
            for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
              zgemm(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real, op_c_imag, warp_row,
                    warp_col, tile_k, wrm);
            }

            if (compute_max_only) {

              op_c_real.abs_max(max);
              op_c_imag.abs_max(max);

            } else {

              int warp_m_offset = warp_row * MMA_M + a_m;
              int warp_n_offset = warp_col * MMA_N;

              store_complex<M, N, ldc>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real, op_c_imag);
            }
          }

          if (a_m + bM < M) {
            __syncthreads();
            a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);
            __syncthreads();
          }

        }

        return max;
      }
      /**
       * The general interface for performance MMA:
       * @param a_dagger is A daggered
       * @param b_dagger is B daggered
       * @param compute_max_only do we actually store to global memory or we just want to find the maximum
       * @param a wrapper for operand A: the object needs to have the following methods:
       *        - .data() that returns the (global memory) address to which we are loading/storing
       *        - ::type the type for the computing type
       *        - ::store_type the type for the storage type
       *        - ::fixed a bool indicates if the object ueses fix point format
       *        - .scale/scale_inv the scales for the fixed point format objects
       * @param b similar to a
       * @param c similar to a
       */
      template <bool a_dagger, bool b_dagger, bool compute_max_only, class A, class B, class C>
      static __device__ inline float perform_mma(const A &a, const B &b, C &c, int m_offset, int n_offset)
      {

        // The typical streaming K MMA type: needs registers to hold all accumulates.
        if (bK < K) {
          return perform_mma_divide_k_yes<a_dagger, b_dagger, compute_max_only>(a, b, c, m_offset, n_offset);
        } else {
          // Shared memory can hold the whole of operand B: we don't need to store all accumulates: save registers.
          if (bM < M && bN == N) {
            return perform_mma_divide_b_no<a_dagger, b_dagger, compute_max_only>(a, b, c);
          } else {
            // Shared memory can hold everything in the K direction: we don't need to store all accumulates: save registers.
            return perform_mma_divide_k_no<a_dagger, b_dagger, compute_max_only>(a, b, c, m_offset, n_offset);
          }
        }
      }
    };

  } // namespace mma

} // namespace quda
