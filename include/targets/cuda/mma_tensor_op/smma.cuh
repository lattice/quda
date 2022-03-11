#pragma once

#include <mma_tensor_op/smma_m16n8_sm80.cuh>

namespace quda
{

  template <class Mma, int block_dim_x, int Ls, int m, int n, int smem_ld_a, int smem_ld_b, int smem_ld_c, bool reload, bool reuse_c_for_b, class T, class real, class S>
  __device__ inline void mma_sync_gemm(T &op_a, real *smem_a, real *smem_b, real *smem_c, const S &wrm)
  {

    constexpr int tile_row_dim = m / Mma::mma_m; // number of tiles in the column dimension
    constexpr int tile_col_dim = n / Mma::mma_n; // number of tiles in the row dimension
    constexpr int tile_acc_dim = m / Mma::mma_k; // number of tiles in the row dimension

    constexpr int total_warp = block_dim_x * Ls / 32;

    static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
                  "Total number of tiles should be divisible by the number of warps.");
    static_assert(tile_col_dim % (tile_row_dim * tile_col_dim / total_warp) == 0,
                  "Each warp should only be responsible a single tile row.");

    constexpr int total_tile = tile_row_dim * tile_col_dim;
    constexpr int warp_cycle = total_tile / total_warp;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = thread_id / 32;
    const int warp_row = warp_id * warp_cycle / tile_col_dim;

#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {

      typename Mma::OperandC op_c;

      // The logical warp assigned to each part of the matrix.
      const int logical_warp_index = warp_id * warp_cycle + c;
      const int warp_col = logical_warp_index - warp_row * tile_col_dim;
      // e.g. for 12 warps:
      // 000|111|222|333
      // 444|555|666|777
      // 888|999|000|111

#pragma unroll
      for (int tile_k = 0; tile_k < tile_acc_dim; tile_k++) {

        if (reload) { // the data in registers can be resued.
          op_a[0].template load<smem_ld_a>(smem_a, tile_k, warp_row, wrm);
        }

        typename Mma::OperandB op_b;
        op_b.template load<smem_ld_b>(smem_b, tile_k, warp_col, wrm);

        if (reload) {
          Mma::mma(op_c, op_a[0], op_b);
        } else {
          Mma::mma(op_c, op_a[tile_k], op_b);
        }
      }

      if (reuse_c_for_b) __syncthreads();

      op_c.template store<smem_ld_c>(smem_c, warp_row * Mma::mma_m, warp_col * Mma::mma_n, wrm);
    }
  }

  template <typename T> struct mma_mapper { };

  template <> struct mma_mapper <double> { using type = Smma<tfloat32, 8, 1, 1>; };
  template <> struct mma_mapper <float> { using type = Smma<tfloat32, 8, 1, 1>; };
  template <> struct mma_mapper <short> { using type = Smma<bfloat16, 16, 1, 1>; };
  template <> struct mma_mapper <int8_t> { using type = Smma<bfloat16, 16, 1, 1>; };

} // namespace quda

