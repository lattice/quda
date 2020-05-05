#pragma once

// XXX: Load different header for different archs.
#include <mma_tensor_op/hmma_m16n16k16_sm70.cuh>

namespace quda
{
  namespace mma
  {
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

      static constexpr int row_stride_pack = row_stride * 2;
      static constexpr int m_dim = (M + row_stride_pack - 1) / row_stride_pack;
      static constexpr int n_dim = (N + col_stride - 1) / col_stride;

      static_assert(M % row_stride_pack == 0, "M needs to be divisible by (row_stride * 2).");
      static_assert(N % col_stride == 0, "N needs to be divisible by col_stride.");

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

      template <class GmemAccessor> __device__ inline void g2r(GmemAccessor gmem)
      {
#pragma unroll
        for (int col = 0; col < n_dim; col++) {
#pragma unroll
          for (int row = 0; row < m_dim; row++) {
            const int col_idx = col * col_stride + y;
            const int row_idx = row * row_stride_pack + z;
            if (row_idx < M && col_idx < N) {
              if (!dagger) {
                auto x = gmem(row_idx + 0, col_idx);
                auto y = gmem(row_idx + 1, col_idx);
                reg_real[row * n_dim + col] = __floats2half2_rn(+x.real(), +y.real());
                reg_imag[row * n_dim + col] = __floats2half2_rn(+x.imag(), +y.imag());
              } else {
                auto x = gmem(col_idx, row_idx + 0);
                auto y = gmem(col_idx, row_idx + 1);
                reg_real[row * n_dim + col] = __floats2half2_rn(+x.real(), +y.real());
                reg_imag[row * n_dim + col] = __floats2half2_rn(-x.imag(), -y.imag());
              }
            }
          }
        }
      }

      __device__ inline void r2s()
      {
#pragma unroll
        for (int col = 0; col < n_dim; col++) {
#pragma unroll
          for (int row = 0; row < m_dim; row++) {
            const int col_idx = col * col_stride + y;
            const int row_idx = row * row_stride_pack + z;
            if (row_idx < M && col_idx < N) {
              smem_real.vector_load(row_idx, col_idx, reg_real[row * n_dim + col]);
              smem_imag.vector_load(row_idx, col_idx, reg_imag[row * n_dim + col]);
            }
          }
        }
      }
    };

    template <int N, int bM, int bN, int bK, int block_y, int block_z, bool a_dag, bool b_dag, bool compute_max_only,
              class A, class B, class C>
    __device__ inline float perform_mma(A aa, B bb, C cc, int m, int n)
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

      {
        auto aa_offset = [&](int i, int j) { return aa(i + m, j); };
        auto bb_offset = [&](int i, int j) { return b_dag ? bb(i, j + n) : bb(i + n, j); };

        aa_loader.g2r(aa_offset);
        aa_loader.r2s();

        bb_loader.g2r(bb_offset);
        bb_loader.r2s();
      }

      __syncthreads();

#pragma unroll 1
      for (int bk = 0; bk < N; bk += bK) {

        if (bk + bK < N) {
          auto aa_offset = [&](int i, int j) { return aa(i + m, j + bk + bK); };
          auto bb_offset = [&](int i, int j) { return b_dag ? bb(i + bk + bK, j + n) : bb(i + n, j + bk + bK); };

          aa_loader.g2r(aa_offset);
          bb_loader.g2r(bb_offset);
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

          store_complex<N>(warp_row * WMMA_M + m, warp_col * WMMA_N + n, wrm, cc, op_c_real[c], op_c_imag[c]);
        }
      }

      return max;
    }

  } // namespace mma
} // namespace quda
