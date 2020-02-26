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
      unsigned *C = reinterpret_cast<unsigned *>(smem);

      const int idx_strided = warp_row * 16 + wrm.quad_row * 8 + wrm.quad_hilo * 4 + wrm.quad_thread;
      const int idx_contiguous = warp_col * 8 + wrm.quad_col * 4;
      const int thread_offset_c = idx_strided * stride + idx_contiguous;
#pragma unroll
      for (int i = 0; i < 4; i++) { C[thread_offset_c + i] = reg[i]; }
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

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1)

#define USE_FP16_MMA_ACCUMULATE

  template <int BlockDimX, int Ls, int M, int N, int M_PAD, int N_PAD, bool reload, class T>
  __device__ inline void mma_sync_gemm(T op_a[], half *sm_a, half *sm_b, half *sm_c, const WarpRegisterMapping &wrm)
  {
    constexpr int WMMA_M = 16; // WMMA_M == WMMA_K
    constexpr int WMMA_N = 16;

#ifdef USE_FP16_MMA_ACCUMULATE
    using accumuate_reg_type = half;
#else
    using accumuate_reg_type = float;
#endif

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

#pragma unroll
    for (int c = 0; c < warp_cycle; c++) {

      MmaOperandC<N_PAD / 2, accumuate_reg_type> op_c;

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

          if (reload) { // the data in registers can be resued.
            op_a[0].load(sm_a, k_idx, warp_row, wrm);
          }

          MmaOperandB<N_PAD / 2> op_b;
          op_b.load(sm_b, k_idx, warp_col, wrm);

          if (reload) {
#ifdef USE_FP16_MMA_ACCUMULATE
            asm volatile(
              "mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
              : "+r"(op_c.reg[0]), "+r"(op_c.reg[1]), "+r"(op_c.reg[2]), "+r"(op_c.reg[3])
              : "r"(op_a[0].reg[0]), "r"(op_a[0].reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#else
            asm volatile(
              "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
              "{%0,%1,%2,%3,%4,%5,%6,%7};"
              : "+f"(op_c.reg[0]), "+f"(op_c.reg[1]), "+f"(op_c.reg[2]), "+f"(op_c.reg[3]), "+f"(op_c.reg[4]),
                "+f"(op_c.reg[5]), "+f"(op_c.reg[6]), "+f"(op_c.reg[7])
              : "r"(op_a[0].reg[0]), "r"(op_a[0].reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#endif
          } else {
#ifdef USE_FP16_MMA_ACCUMULATE
            asm volatile(
              "mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
              : "+r"(op_c.reg[0]), "+r"(op_c.reg[1]), "+r"(op_c.reg[2]), "+r"(op_c.reg[3])
              : "r"(op_a[k_idx].reg[0]), "r"(op_a[k_idx].reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#else
            asm volatile(
              "mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
              "{%0,%1,%2,%3,%4,%5,%6,%7};"
              : "+f"(op_c.reg[0]), "+f"(op_c.reg[1]), "+f"(op_c.reg[2]), "+f"(op_c.reg[3]), "+f"(op_c.reg[4]),
                "+f"(op_c.reg[5]), "+f"(op_c.reg[6]), "+f"(op_c.reg[7])
              : "r"(op_a[k_idx].reg[0]), "r"(op_a[k_idx].reg[1]), "r"(op_b.reg[0]), "r"(op_b.reg[1]));
#endif
          }
        }
      }

      __syncthreads();

      op_c.store(sm_c, warp_row, warp_col, wrm);
    }
  }

#endif // #if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1)

#endif // defined (__COMPUTE_CAPABILITY__ == 700)

} // namespace quda
