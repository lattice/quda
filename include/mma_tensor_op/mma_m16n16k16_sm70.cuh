#if (__COMPUTE_CAPABILITY__ == 700)
#include <mma.h>
#endif

namespace quda
{

#if (__COMPUTE_CAPABILITY__ == 700)

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

  // For "reload" version(reload == true) of wmma gemm, matrix a is loaded when
  // needed.
  // It is a waste of time but has less register usage.
  // For "preload" version(reload == false) of wmma gemm, matrix a is preloaded
  // before hand.
  // It saves time but uses more registers.
  template <int BlockDimX, int Ls, int M, int N, int M_sm, int N_sm, bool reload, class T>
  __device__ inline void wmma_gemm(T *a_frag, half *sm_a, half *sm_b, half *sm_c)
  {
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    constexpr int tm_dim = M / WMMA_M;
    constexpr int tn_dim = N / WMMA_N;

    constexpr int total_warp = BlockDimX * Ls / 32;

    static_assert((tm_dim * tn_dim) % total_warp == 0, "(tm_dim*tn_dim)%%total_warp==0\n");
    static_assert(tn_dim % (tm_dim * tn_dim / total_warp) == 0, "tn_dim%%(tm_dim*tn_dim/total_warp)==0\n");

    const int this_warp = (threadIdx.y * blockDim.x + threadIdx.x) >> 5;

    constexpr int total_tile = tm_dim * tn_dim;

    constexpr int warp_cycle = total_tile / total_warp;
    const int warp_m = this_warp * warp_cycle / tn_dim;
#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {
      // Set up the wmma stuff
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

      // The logical warp assigned to each part of the matrix.
      const int phys_warp_index = this_warp * warp_cycle + c;
      const int warp_n = phys_warp_index - warp_m * tn_dim;
      // eg. for 12 warps:
      // 000|111|222|333
      // 444|555|666|777
      // 888|999|000|111

      // Zero the initial acc.
      nvcuda::wmma::fill_fragment(c_frag, static_cast<half>(0.0f));

#pragma unroll
      for (int k = 0; k < tm_dim; k++) {
        const int a_row = warp_m * WMMA_M;
        const int a_col = k * WMMA_K;
        const int b_row = k * WMMA_K;
        const int b_col = warp_n * WMMA_N;

        // Load Matrix
        if (reload) { nvcuda::wmma::load_matrix_sync(a_frag[0], sm_a + a_row + a_col * M_sm, M_sm); }
        nvcuda::wmma::load_matrix_sync(b_frag, sm_c + b_col + b_row * N_sm, N_sm);
        // Perform the matrix multiplication
        if (reload) {
          nvcuda::wmma::mma_sync(c_frag, a_frag[0], b_frag, c_frag);
        } else {
          nvcuda::wmma::mma_sync(c_frag, a_frag[k], b_frag, c_frag);
        }
      }

      __syncthreads();

      int c_row = warp_m * WMMA_M;
      int c_col = warp_n * WMMA_N;

      nvcuda::wmma::store_matrix_sync(sm_c + c_col + c_row * N_sm, c_frag, N_sm, nvcuda::wmma::mem_row_major);
    }
  }

  template <class data_type, class smem_access_type>
  __device__ inline void mma_load_a_frag(smem_access_type ra[], const data_type *smem_a, const int reg_offset,
                                         const int smem_offset)
  {
    const smem_access_type *A = reinterpret_cast<const smem_access_type *>(smem_a);
    ra[reg_offset + 0] = A[smem_offset + 0];
    ra[reg_offset + 1] = A[smem_offset + 1];
  }

  template <int stride> struct MmaOperandA {

    unsigned reg[2];

    __device__ void load(unsigned *smem, int k, int warp_row, const WarpRegisterMapping &wrm)
    {
      const int idx_strided = k * 4 + wrm.quad_thread;
      const int idx_contiguous = warp_row * 8 + wrm.quad_row * 4 + wrm.quad_hilo * 2;
      const int thread_offset_a = idx_strided * stride + idx_contiguous;
      reg[0] = smem[thread_offset_a + 0];
      reg[1] = smem[thread_offset_a + 1];
    }
  };

  template <int stride> struct MmaOperandB {

    unsigned reg[2];

    __device__ void load(unsigned *smem, int k, int warp_col, const WarpRegisterMapping &wrm)
    {
      const int idx_strided = k * 4 + wrm.quad_thread;
      const int idx_contiguous = warp_col * 8 + wrm.quad_col * 4 + wrm.quad_hilo * 2;
      const int thread_offset_b = idx_strided * stride + idx_contiguous;
      reg[0] = smem[thread_offset_b + 0];
      reg[1] = smem[thread_offset_b + 1];
    }
  };

#define USE_FP16_MMA_ACCUMULATE

  template <int BlockDimX, int Ls, int M, int N, int M_PAD, int N_PAD>
  __device__ inline void mma_sync_gemm(half *sm_a, half *sm_b, half *sm_c)
  {
    using data_type = half;
    using smem_access_type = unsigned;
    constexpr int data_pack_factor = sizeof(unsigned) / sizeof(data_type);

    constexpr int WMMA_M = 16; // WMMA_M == WMMA_K
    constexpr int WMMA_N = 16;
#ifdef USE_FP16_MMA_ACCUMULATE
    constexpr bool fp16_accumulate = true;
    using accumuate_reg_type = unsigned;
#else
    constexpr bool fp16_accumulate = false;
    using accumuate_reg_type = float;
#endif
    constexpr int accumuate_regs = fp16_accumulate ? 4 : 8;

    constexpr int tile_row_dim = M / WMMA_M; // number of tiles in the column dimension
    constexpr int tile_col_dim = N / WMMA_N; // number of tiles in the row dimension

    constexpr int total_warp = BlockDimX * Ls / 32;

    static_assert((tile_row_dim * tile_col_dim) % total_warp == 0,
                  "Total number of tiles should be divisible by the number of warps.");
    static_assert(tile_col_dim % (tile_row_dim * tile_col_dim / total_warp) == 0,
                  "Each warp should only be responsible a single tile row.");

    constexpr int total_tile = tile_row_dim * tile_col_dim;

    constexpr int warp_cycle = total_tile / total_warp;

    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = thread_id >> 5;
    const int warp_row = warp_id * warp_cycle / tile_col_dim;

    const WarpRegisterMapping wrm(thread_id);
    MmaOperandA<M_PAD / data_pack_factor> op_a[tile_row_dim * 4];

#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {

      accumuate_reg_type rc[accumuate_regs];
#pragma unroll
      for (int r = 0; r < accumuate_regs; r++) { rc[r] = 0; }

      // The logical warp assigned to each part of the matrix.
      const int logical_warp_index = warp_id * warp_cycle + c;
      const int warp_col = logical_warp_index - warp_row * tile_col_dim;
      // e.g. for 12 warps:
      // 000|111|222|333
      // 444|555|666|777
      // 888|999|000|111

#pragma unroll
      for (int tile_k = 0; tile_k < tile_row_dim; tile_k++) {
#pragma unroll
        for (int warp_k = 0; warp_k < 4; warp_k++) {

          const int k_idx = tile_k * 4 + warp_k;

          if (c == 0) { // the data in registers can be resued.
            smem_access_type *A = reinterpret_cast<smem_access_type *>(sm_a);
            op_a[k_idx].load(A, k_idx, warp_row, wrm);
          }

          MmaOperandB<N_PAD / data_pack_factor> op_b;
          smem_access_type *B = reinterpret_cast<smem_access_type *>(sm_b);
          op_b.load(B, k_idx, warp_col, wrm);

#ifdef USE_FP16_MMA_ACCUMULATE
          asm volatile("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
                       : "+r"(rc[0]), "+r"(rc[1]), "+r"(rc[2]), "+r"(rc[3])
                       : "r"(op_a[k_idx].reg[0]), "r"(op_a[k_idx].reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#else
          asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                       "{%0,%1,%2,%3,%4,%5,%6,%7};"
                       : "+f"(rc[0]), "+f"(rc[1]), "+f"(rc[2]), "+f"(rc[3]), "+f"(rc[4]), "+f"(rc[5]), "+f"(rc[6]),
                         "+f"(rc[7])
                       : "r"(op_a[k_idx].reg[0]), "r"(op_a[k_idx].reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#endif
        }
      }

      __syncthreads();
#ifdef USE_FP16_MMA_ACCUMULATE
      smem_access_type *C = reinterpret_cast<smem_access_type *>(sm_c);
      const int row = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + wrm.quad_thread;
      const int col = warp_col * 8 + wrm.quad_col * 4;
      const int thread_offset_c = row * (N_PAD / data_pack_factor) + col;
#pragma unroll
      for (int i = 0; i < 4; i++) { C[thread_offset_c + i] = rc[i]; }
#else
      half2 *C = reinterpret_cast<half2 *>(sm_c);

      const int row = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + (wrm.quad_thread % 2);
      const int col = warp_col * 8 + wrm.quad_col * 4 + (wrm.quad_thread / 2);

      int thread_offset_c = row * (N_PAD / data_pack_factor) + col;
      C[thread_offset_c] = __floats2half2_rn(rc[0], rc[1]);

      thread_offset_c = (row + 2) * (N_PAD / data_pack_factor) + col;
      C[thread_offset_c] = __floats2half2_rn(rc[2], rc[3]);

      thread_offset_c = row * (N_PAD / data_pack_factor) + (col + 2);
      C[thread_offset_c] = __floats2half2_rn(rc[4], rc[5]);

      thread_offset_c = (row + 2) * (N_PAD / data_pack_factor) + (col + 2);
      C[thread_offset_c] = __floats2half2_rn(rc[6], rc[7]);
#endif
    }
  }

#endif // defined (__COMPUTE_CAPABILITY__ == 700)

} // namespace quda
