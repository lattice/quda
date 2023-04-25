#pragma once

#include <algorithm>
#include <array.h>
#include <mma_tensor_op/mma_dispatch.cuh>
#include <pipeline.cuh>
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

    /**
      @brief Defining how many elements/atoms are there in type T ...
     */
    template <class T> struct batch_multiple {
    };

    /**
      @brief ... e.g. there are 2 half's in a half2
     */
    template <> struct batch_multiple<half2> {
      static constexpr int value = 2;
    };

    template <> struct batch_multiple<float> {
      static constexpr int value = 1;
    };

    inline __device__ void zero(half2 &reg_real, half2 &reg_imag)
    {
      reg_real = __half2half2(0);
      reg_imag = __half2half2(0);
    }

    inline __device__ void zero(float &reg_real, float &reg_imag)
    {
      reg_real = 0;
      reg_imag = 0;
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, class T>
    inline __device__ void convert_x(half2 &reg_real, half2 &reg_imag, complex<T> *p, int m_idx, int n_idx,
                                     float scale_inv)
    {
      if (x) {
        auto xx = p[(m_idx + 0) * ld + n_idx];
        auto yy = p[(m_idx + 1) * ld + n_idx];

        if (fixed) {
          reg_real = __floats2half2_rn(scale_inv * xx.real(), scale_inv * yy.real());
          auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
          reg_imag = __floats2half2_rn(scale_inv_conj * xx.imag(), scale_inv_conj * yy.imag());
        } else {
          reg_real = __floats2half2_rn(+xx.real(), +yy.real());
          reg_imag = __floats2half2_rn(dagger ? -xx.imag() : +xx.imag(), dagger ? -yy.imag() : +yy.imag());
        }
      } else {
        using store_type = T;
        using store_array = typename VectorType<store_type, 4>::type;
        store_array v = *reinterpret_cast<store_array *>(&p[n_idx * ld + m_idx]);

        if (fixed) {
          reg_real = __floats2half2_rn(scale_inv * v.x, scale_inv * v.z);
          auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
          reg_imag = __floats2half2_rn(scale_inv_conj * v.y, scale_inv_conj * v.w);
        } else {
          reg_real = __floats2half2_rn(+v.x, +v.z);
          reg_imag = __floats2half2_rn(dagger ? -v.y : +v.y, dagger ? -v.w : +v.w);
        }
      }
    }

    /**
      @brief Load from global memory and store data in registers.
     */
    template <bool x, bool fixed, bool dagger, int ld, class T>
    inline __device__ void convert_x(float &reg_real, float &reg_imag, complex<T> *p, int m_idx, int n_idx,
                                     float scale_inv)
    {
      if (x) {
        auto xx = p[m_idx * ld + n_idx];

        if (fixed) {
          reg_real = scale_inv * xx.real();
          auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
          reg_imag = scale_inv_conj * xx.imag();
        } else {
          reg_real = +xx.real();
          reg_imag = dagger ? -xx.imag() : +xx.imag();
        }
      } else {
        auto xx = p[n_idx * ld + m_idx];
        using store_type = T;
        using store_array = typename VectorType<store_type, 2>::type;
        store_array v = *reinterpret_cast<store_array *>(&p[n_idx * ld + m_idx]);

        if (fixed) {
          reg_real = scale_inv * xx.real();
          auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
          reg_imag = scale_inv_conj * xx.imag();
        } else {
          reg_real = xx.real();
          reg_imag = dagger ? -xx.imag() : xx.imag();
        }
      }
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
    template <class load_t, int M, int N, int bM, int bN, int block_y, int block_z, bool transpose>
    struct GlobalMemoryLoader {

      static constexpr int batch = batch_multiple<load_t>::value;

      static constexpr int m_stride_n = block_y * batch;
      static constexpr int n_stride_n = block_z * 1;
      static constexpr int m_stride_t = block_z * batch;
      static constexpr int n_stride_t = block_y * 1;

      static constexpr int register_count
        = std::max(((bN + n_stride_n - 1) / n_stride_n) * ((bM + m_stride_n - 1) / m_stride_n),
                   ((bN + n_stride_t - 1) / n_stride_t) * ((bM + m_stride_t - 1) / m_stride_t));

      load_t reg_real[register_count];
      load_t reg_imag[register_count];

      template <int ld, bool dagger, class T, class gmem_accessor_t>
      __device__ inline float g2tmp(const gmem_accessor_t &gmem, int m_offset, int n_offset, complex<T> *smem_ptr,
                                    pipeline_t &pipe)
      {
        auto p = gmem.data();

        int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        constexpr int element_per_thread = 16 / (sizeof(T) * 2);
        while (thread_id * element_per_thread < bM * bN) {
          if (transpose != dagger) {
            int m = element_per_thread * (thread_id % (bM / element_per_thread));
            int n = thread_id / (bM / element_per_thread);
            auto dst_ptr = reinterpret_cast<float4 *>(&smem_ptr[n * (bM + 4) + m]);
            auto src_ptr = reinterpret_cast<float4 *>(&p[(n + n_offset) * ld + m + m_offset]);
            memcpy_async(dst_ptr, src_ptr, sizeof(float4), pipe);
          } else {
            int m = thread_id / (bN / element_per_thread);
            int n = element_per_thread * (thread_id % (bN / element_per_thread));
            auto dst_ptr = reinterpret_cast<float4 *>(&smem_ptr[m * (bN + 4) + n]);
            auto src_ptr = reinterpret_cast<float4 *>(&p[(m + m_offset) * ld + n + n_offset]);
            memcpy_async(dst_ptr, src_ptr, sizeof(float4), pipe);
          }
          thread_id += blockDim.x * blockDim.y * blockDim.z;
        }
        return gmem.get_scale_inv();
      }

      template <int ld, bool dagger, bool fixed, class T, class smem_accessor_t>
      __device__ inline void tmp2s(complex<T> *smem_ptr, float scale_inv, smem_accessor_t &smem_real,
                                   smem_accessor_t &smem_imag)
      {
        // for each iteration, each warp loads a tile
        int thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        int warp_id = thread_id / 32;
        int lane_id = thread_id % 32;
        int thread_in_group = lane_id % 4;
        int group_id = lane_id / 4;
        constexpr int w_m = 8 * batch;
        constexpr int w_k = 4;
        static_assert(bM % w_m == 0, "bM %% w_m");
        static_assert(bN % w_k == 0, "bN %% w_k");

        constexpr int tile_dim_m = bM / w_m;
        constexpr int tile_dim_k = bN / w_k;

        constexpr int total_tiles = tile_dim_k * tile_dim_m;
        constexpr int n_warp = block_y * block_z / 32;
        constexpr int warp_cycle = (total_tiles + n_warp - 1) / n_warp;
#pragma unroll
        for (int c = 0; c < warp_cycle; c++) {
          int logical_warp_index = c * n_warp + warp_id;
          if (logical_warp_index < total_tiles) {
            int warp_m = (c * n_warp + warp_id) % tile_dim_m;
            int warp_k = (c * n_warp + warp_id) / tile_dim_m;

            int smem_m_offset = warp_m * w_m + group_id * batch;
            int smem_k_offset = warp_k * w_k + thread_in_group;

            int gmem_m_offset = smem_m_offset;
            int gmem_k_offset = smem_k_offset;

            load_t real;
            load_t imag;

            constexpr bool x = (transpose == dagger);
            convert_x<x, fixed, dagger, x ? bN + 4 : bM + 4>(real, imag, smem_ptr, gmem_m_offset, gmem_k_offset,
                                                             scale_inv);
            smem_real.vector_load(smem_m_offset, smem_k_offset, real);
            smem_imag.vector_load(smem_m_offset, smem_k_offset, imag);
          }
        }
      }

      /**
       * ld: leading dimension of global memory
       * dagger: if we need to store daggered (tranpose and hermision conjugate)
       */
      template <int ld, bool dagger, class GmemAccessor>
      __device__ inline void g2r(const GmemAccessor &gmem, int m_offset, int n_offset)
      {
        auto p = gmem.data();
        auto scale_inv = gmem.get_scale_inv();
        constexpr bool fixed = GmemAccessor::fixed;

        constexpr bool x = (transpose == dagger);

        constexpr int n_stride = x ? block_y * 1 : block_z * 1;
        constexpr int m_stride = x ? block_z * batch : block_y * batch;
        int n_thread_offset = x ? threadIdx.y * 1 : threadIdx.z * 1;
        int m_thread_offset = x ? threadIdx.z * batch : threadIdx.y * batch;

        constexpr int n_dim = (bN + n_stride - 1) / n_stride;
        constexpr int m_dim = (bM + m_stride - 1) / m_stride;

        constexpr bool check_global_bound = !(M % bM == 0 && N % bN == 0);
        constexpr bool check_shared_bound = !(bM % m_stride == 0 && bN % n_stride == 0);

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
                convert_x<x, fixed, dagger, ld>(reg_real[m * n_dim + n], reg_imag[m * n_dim + n], p, m_idx, n_idx,
                                                scale_inv);
              } else {
                zero(reg_real[m * n_dim + n], reg_imag[m * n_dim + n]);
              }
            }
          }
        }
      }

      template <bool dagger, class SmemObj> __device__ inline void r2s(SmemObj &smem_real, SmemObj &smem_imag)
      {
        constexpr int n_stride = transpose == dagger ? block_y * 1 : block_z * 1;
        constexpr int m_stride = transpose == dagger ? block_z * batch : block_y * batch;
        int n_thread_offset = transpose == dagger ? threadIdx.y * 1 : threadIdx.z * 1;
        int m_thread_offset = transpose == dagger ? threadIdx.z * batch : threadIdx.y * batch;

        constexpr int n_dim = (bN + n_stride - 1) / n_stride;
        constexpr int m_dim = (bM + m_stride - 1) / m_stride;

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
