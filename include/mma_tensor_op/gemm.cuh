#pragma once

// XXX: Load different header for different archs.
#include <mma_tensor_op/hmma_m16n16k16_sm70.cuh>

namespace quda
{
  namespace mma
  {
    constexpr int shared_memory_bytes(int bM, int bN, int bK)
    {
      return (bM + pad_size(bM) + bN + pad_size(bN)) * bK * 2 * sizeof(half);
    }

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

    template <int M, int N, int ldm, int ldn, class T> __device__ inline auto make_smem_obj(T *ptr_)
    {
      return SharedMemoryObject<T, M, N, ldm, ldn> {ptr_};
    }

    template <int M, int N, int m_stride, int n_stride, bool dagger, class SmemAccessor> struct GlobalMemoryLoader {

      static constexpr int m_stride_pack = m_stride * 2;
      static constexpr int m_dim = (M + m_stride_pack - 1) / m_stride_pack;
      static constexpr int n_dim = (N + n_stride - 1) / n_stride;

      static_assert(M % m_stride_pack == 0, "M needs to be divisible by (m_stride * 2).");
      static_assert(N % n_stride == 0, "N needs to be divisible by n_stride.");

      SmemAccessor smem_real;
      SmemAccessor smem_imag;

      half2 reg_real[m_dim * n_dim];
      half2 reg_imag[m_dim * n_dim];

      const int y;
      const int z;

      __device__ inline GlobalMemoryLoader(SmemAccessor real_, SmemAccessor imag_) :
        smem_real(real_),
        smem_imag(imag_),
        y(threadIdx.y),
        z(threadIdx.z * 2)
      {
      }

      template <int matrix_n, bool transpose, class GmemAccessor>
      __device__ inline void g2r(const GmemAccessor &gmem, int m_offset, int n_offset)
      {
        auto p = gmem.data();
        auto scale_inv = gmem.scale_inv;
        constexpr bool fixed = GmemAccessor::fixed;
#pragma unroll
        for (int n = 0; n < n_dim; n++) {
#pragma unroll
          for (int m = 0; m < m_dim; m++) {
            const int n_idx = n * n_stride + y + n_offset;
            const int m_idx = m * m_stride_pack + z + m_offset;
            if (transpose == dagger) {
              auto xx = p[(m_idx + 0) * matrix_n + n_idx];
              auto yy = p[(m_idx + 1) * matrix_n + n_idx];

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
              store_vector_type v = *reinterpret_cast<store_vector_type *>(&p[n_idx * matrix_n + m_idx]);

              if (fixed) {
                reg_real[m * n_dim + n] = __floats2half2_rn(scale_inv * v.x, scale_inv * v.z);
                auto scale_inv_conj = dagger ? -scale_inv : scale_inv;
                reg_imag[m * n_dim + n] = __floats2half2_rn(scale_inv_conj * v.y, scale_inv_conj * v.w);
              } else {
                reg_real[m * n_dim + n] = __floats2half2_rn(+v.x, +v.z);
                reg_imag[m * n_dim + n] = __floats2half2_rn(dagger ? -v.y : +v.y, dagger ? -v.w : +v.w);
              }
            }
          }
        }
      }

      __device__ inline void r2s()
      {
#pragma unroll
        for (int n = 0; n < n_dim; n++) {
#pragma unroll
          for (int m = 0; m < m_dim; m++) {
            const int n_idx = n * n_stride + y;
            const int m_idx = m * m_stride_pack + z;
            smem_real.vector_load(m_idx, n_idx, reg_real[m * n_dim + n]);
            smem_imag.vector_load(m_idx, n_idx, reg_imag[m * n_dim + n]);
          }
        }
      }
    };

    template <int N, int bM, int bN, int bK, int block_y, int block_z, bool a_dag, bool b_dag, bool compute_max_only,
              class A, class B, class C>
    __device__ inline float perform_mma(const A &aa, const B &bb, C &cc, int m_offset, int n_offset)
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

      GlobalMemoryLoader<bM, bK, n_row, n_col, a_dag, decltype(smem_obj_a_real)> aa_loader(smem_obj_a_real,
                                                                                           smem_obj_a_imag);
      GlobalMemoryLoader<bN, bK, n_row, n_col, b_dag, decltype(smem_obj_b_real)> bb_loader(smem_obj_b_real,
                                                                                           smem_obj_b_imag);

      constexpr bool a_transpose = false;
      constexpr bool b_transpose = true;

      aa_loader.g2r<N, a_transpose>(aa, m_offset, 0);
      aa_loader.r2s();

      bb_loader.g2r<N, b_transpose>(bb, n_offset, 0);
      bb_loader.r2s();

      __syncthreads();

#pragma unroll 1
      for (int bk = 0; bk < N; bk += bK) {

        if (bk + bK < N) {
          aa_loader.g2r<N, a_transpose>(aa, m_offset, bk + bK);
          bb_loader.g2r<N, b_transpose>(bb, n_offset, bk + bK);
        }

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

        if (bk + bK < N) {
          __syncthreads();

          aa_loader.r2s();
          bb_loader.r2s();

          __syncthreads();
        }
      }

      // wrap up!
#pragma unroll
      for (int c = 0; c < warp_cycle; c++) {

        if (compute_max_only) {

          op_c_real[c].abs_max(max);
          op_c_imag[c].abs_max(max);

        } else {

          const int logical_warp_index = warp_id * warp_cycle + c;
          const int warp_row = logical_warp_index / tile_col_dim;
          const int warp_col = logical_warp_index - warp_row * tile_col_dim;

          store_complex<N>(warp_row * WMMA_M + m_offset, warp_col * WMMA_N + n_offset, wrm, cc, op_c_real[c],
                           op_c_imag[c]);
        }
      }

      return max;
    }

  } // namespace mma
} // namespace quda
