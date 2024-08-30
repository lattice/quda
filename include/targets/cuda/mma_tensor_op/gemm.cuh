#pragma once

#include <algorithm>
#include <array.h>
#include <mma_tensor_op/mma_dispatch.cuh>
#include <mma_tensor_op/gmem_loader.cuh>
#include <register_traits.h>

namespace quda
{
  namespace mma
  {
    // return the size of the shared memory needed for MMA with block shape bM, bN, bK.
    template <class mma_t> constexpr int shared_memory_bytes(int bM, int bN, int bK)
    {
      return (bM + mma_t::pad_size(bM) + bN + mma_t::pad_size(bN)) * bK * 2 * sizeof(typename mma_t::compute_t);
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
     * Perform the complex GEMM
     * @param m, n, k the corresponding offset in the M, N, and K direction
     */
    template <class mma_t, class A, class B, class C>
    __device__ inline void complex_mma(const A &smem_obj_a_real, const A &smem_obj_a_imag, const B &smem_obj_b_real,
                                       const B &smem_obj_b_imag, C &op_c_real, C &op_c_imag, int m, int n, int k,
                                       const typename mma_t::WarpRegisterMapping &wrm)
    {

      typename mma_t::OperandA op_a_real;
      op_a_real.load(smem_obj_a_real, k, m, wrm);
      typename mma_t::OperandA op_a_imag;
      op_a_imag.load(smem_obj_a_imag, k, m, wrm);

      typename mma_t::OperandB op_b_real;
      op_b_real.load(smem_obj_b_real, k, n, wrm);
      typename mma_t::OperandB op_b_imag;
      op_b_imag.load(smem_obj_b_imag, k, n, wrm);

      mma_t::mma(op_a_real, op_b_real, op_c_real);
      mma_t::mma(op_a_imag, op_b_real, op_c_imag);
      mma_t::mma(op_a_real, op_b_imag, op_c_imag);
      // negate op_imag
      op_a_imag.negate();
      mma_t::mma(op_a_imag, op_b_imag, op_c_real);
    }

    // A wrapper that wraps the OperandC objects, together with the various methods to loop over it
    template <class mma_t, int warp_cycle, int tile_row_dim, int tile_col_dim, int tile_acc_dim> struct MmaAccumulator {

      static constexpr int size = warp_cycle;

      typename mma_t::OperandC op_c_real[warp_cycle];
      typename mma_t::OperandC op_c_imag[warp_cycle];

      typename mma_t::WarpRegisterMapping wrm;

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

      /** @brief Apply MMA, but doing a rescaling before accumulate into the final accumulator */
      template <class SmemObjA, class SmemObjB>
      __device__ inline void mma_rescale(const SmemObjA &smem_obj_a_real, const SmemObjA &smem_obj_a_imag,
                                         const SmemObjB &smem_obj_b_real, const SmemObjB &smem_obj_b_imag, float rescale)
      {

#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          typename mma_t::OperandC op_c_real_tmp;
          op_c_real_tmp.zero();
          typename mma_t::OperandC op_c_imag_tmp;
          op_c_imag_tmp.zero();

#pragma unroll 1
          for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

            // The logical warp assigned to each part of the matrix.
            const int logical_warp_index = wrm.warp_id * warp_cycle + c;
            if (logical_warp_index < tile_row_dim * tile_col_dim) {
              const int warp_row = logical_warp_index / tile_col_dim;
              const int warp_col = logical_warp_index - warp_row * tile_col_dim;

              complex_mma<mma_t>(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real_tmp,
                                 op_c_imag_tmp, warp_row, warp_col, tile_k, wrm);
            }
          }
          op_c_real[c].axpy(rescale, op_c_real_tmp);
          op_c_imag[c].axpy(rescale, op_c_imag_tmp);
        }
      }

      /** @brief Apply MMA */
      template <class SmemObjA, class SmemObjB>
      __device__ inline void mma(const SmemObjA &smem_obj_a_real, const SmemObjA &smem_obj_a_imag,
                                 const SmemObjB &smem_obj_b_real, const SmemObjB &smem_obj_b_imag)
      {

#pragma unroll 1
        for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
#pragma unroll
          for (int c = 0; c < warp_cycle; c++) {

            // The logical warp assigned to each part of the matrix.
            const int logical_warp_index = wrm.warp_id * warp_cycle + c;
            if (logical_warp_index < tile_row_dim * tile_col_dim) {
              const int warp_row = logical_warp_index / tile_col_dim;
              const int warp_col = logical_warp_index - warp_row * tile_col_dim;

              complex_mma<mma_t>(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real[c],
                                 op_c_imag[c], warp_row, warp_col, tile_k, wrm);
            }
          }
        }
      }

      template <int M, int N, int ldc, bool dagger, class C, class op_t>
      __device__ inline void store(C &c_accessor, int m_offset, int n_offset, op_t op)
      {
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {

          const int logical_warp_index = wrm.warp_id * warp_cycle + c;
          if (logical_warp_index < tile_row_dim * tile_col_dim) {
            const int warp_row = logical_warp_index / tile_col_dim;
            const int warp_col = logical_warp_index - warp_row * tile_col_dim;

            const int warp_m_offset = warp_row * mma_t::MMA_M + m_offset;
            const int warp_n_offset = warp_col * mma_t::MMA_N + n_offset;

            mma_t::template store_complex<M, N, ldc, dagger>(warp_m_offset, warp_n_offset, wrm, c_accessor,
                                                             op_c_real[c], op_c_imag[c], op);
          }
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
    template <class mma_t, int M_, int N_, int K_, int lda_, int ldb_, int ldc_, int bM_, int bN_, int bK_, int block_y,
              int block_z>
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

      static constexpr int tile_row_dim = bM / mma_t::MMA_M; // number of tiles in the column dimension
      static constexpr int tile_col_dim = bN / mma_t::MMA_N; // number of tiles in the row dimension
      static constexpr int tile_acc_dim = bK / mma_t::MMA_K; // number of tiles in the accumulate dimension

      static constexpr int smem_lda = bM + mma_t::pad_size(bM); // shared memory leading dimensions
      static constexpr int smem_ldb = bN + mma_t::pad_size(bN);

      static constexpr int n_row = block_y;
      static constexpr int n_col = block_z;

      static constexpr int total_warp = n_row * n_col / mma_t::warp_size; // Total number of warps in the CTA

      static constexpr int total_tile = tile_row_dim * tile_col_dim; // Total number of tiles dividing operand C
      static constexpr int warp_cycle
        = (total_tile + total_warp - 1) / total_warp; // Number of tiles each warp needs to calculate

      static constexpr bool a_transpose
        = false; // In our setup, specifically in the arch-dependent code, A is always column-major, while B is always row-major
      static constexpr bool b_transpose = true;

      static_assert(bM % mma_t::MMA_M == 0, "bM must be divisible by MMA_M.");
      static_assert(bN % mma_t::MMA_N == 0, "bN must be divisible by MMA_N.");
      static_assert(bK % mma_t::MMA_K == 0, "bK must be divisible by MMA_K.");

      // static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
      //               "Total number of tiles should be divisible by the number of warps.");

      using SmemObjA = SharedMemoryObject<typename mma_t::compute_t, bM, bK, 1, smem_lda>;
      using SmemObjB = SharedMemoryObject<typename mma_t::compute_t, bN, bK, 1, smem_ldb>;

      using Accumulator = MmaAccumulator<mma_t, warp_cycle, tile_row_dim, tile_col_dim, tile_acc_dim>;

      using ALoader = GlobalMemoryLoader<typename mma_t::load_t, M, K, bM, bK, n_row, n_col, a_transpose>;
      using BLoader = GlobalMemoryLoader<typename mma_t::load_t, N, K, bN, bK, n_row, n_col, b_transpose>;

      // This is the most general MMA code: bM < M, bN < N, bK < K.
      // We divide M and N, and we stream over K, which means we need to store the accumulate register for ALL tiles.
      template <bool a_dagger, bool b_dagger, bool compute_max_only, class A, class B, class C>
      static __device__ inline float perform_mma_divide_k_yes(const A &a, const B &b, C &c, int m_offset, int n_offset)
      {
        float max = 0;

        extern __shared__ typename mma_t::compute_t smem_ptr[];

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

          accumulator.mma(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag);

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
          accumulator.template store<M, N, ldc, false>(c, m_offset, n_offset, assign_t());
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

        extern __shared__ typename mma_t::compute_t smem_ptr[];

        SmemObjA smem_obj_a_real(smem_ptr);
        SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + smem_ldb * bK);

        typename mma_t::OperandC op_c_real;
        typename mma_t::OperandC op_c_imag;

        typename mma_t::WarpRegisterMapping wrm((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

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

          // TODO: Check if some of the warps are idle
          // The logical warp assigned to each part of the matrix.
          int logical_warp_index = wrm.warp_id * warp_cycle + c;
          int warp_row = logical_warp_index / tile_col_dim;
          int warp_col = logical_warp_index - warp_row * tile_col_dim;

          op_c_real.zero();
          op_c_imag.zero();

#pragma unroll 1
          for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
            complex_mma<mma_t>(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real, op_c_imag,
                               warp_row, warp_col, tile_k, wrm);
          }

          if (compute_max_only) {

            op_c_real.abs_max(max);
            op_c_imag.abs_max(max);

          } else {

            int warp_m_offset = warp_row * mma_t::MMA_M + m_offset;
            int warp_n_offset = warp_col * mma_t::MMA_N + n_offset;

            mma_t::template store_complex<M, N, ldc, false>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real,
                                                            op_c_imag, assign_t());
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

        extern __shared__ typename mma_t::compute_t smem_ptr[];

        SmemObjA smem_obj_a_real(smem_ptr);
        SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + smem_lda * bK);
        SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + smem_ldb * bK);

        typename mma_t::OperandC op_c_real;
        typename mma_t::OperandC op_c_imag;

        typename mma_t::WarpRegisterMapping wrm((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

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
            // TODO: Check if some of the warps are idle
            // The logical warp assigned to each part of the matrix.
            int logical_warp_index = wrm.warp_id * warp_cycle + c;
            int warp_row = logical_warp_index / tile_col_dim;
            int warp_col = logical_warp_index - warp_row * tile_col_dim;

            op_c_real.zero();
            op_c_imag.zero();

#pragma unroll 1
            for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {
              complex_mma<mma_t>(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real,
                                 op_c_imag, warp_row, warp_col, tile_k, wrm);
            }

            if (compute_max_only) {

              op_c_real.abs_max(max);
              op_c_imag.abs_max(max);

            } else {

              int warp_m_offset = warp_row * mma_t::MMA_M + a_m;
              int warp_n_offset = warp_col * mma_t::MMA_N;

              mma_t::template store_complex<M, N, ldc, false>(warp_m_offset, warp_n_offset, wrm, c_accessor, op_c_real,
                                                              op_c_imag, assign_t());
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
