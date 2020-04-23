#pragma once

#include <gauge_field_order.h>
#include <index_helper.cuh>

// #include <matrix_tile.cuh>

#include <mma_tensor_op/mma_m16n16k16_sm70.cuh>

namespace quda
{

  namespace mma
  {

    template <typename Float_, typename PreconditionedGauge, typename Gauge, int n> struct CalculateYhatArg {
      using Float = Float_;

      static constexpr int M = n;
      static constexpr int N = n;
      static constexpr int K = n;

      PreconditionedGauge Yhat;
      const Gauge Y;
      const Gauge Xinv;
      int dim[QUDA_MAX_DIM];
      int comm_dim[QUDA_MAX_DIM];
      int nFace;

      Float *max_h; // host scalar that stores the maximum element of Yhat. Pointer b/c pinned.
      Float *max_d; // device scalar that stores the maximum element of Yhat

      CalculateYhatArg(const PreconditionedGauge &Yhat, const Gauge Y, const Gauge Xinv, const int *dim,
                       const int *comm_dim, int nFace) :
        Yhat(Yhat),
        Y(Y),
        Xinv(Xinv),
        nFace(nFace),
        max_h(nullptr),
        max_d(nullptr)
      {
        for (int i = 0; i < 4; i++) {
          this->comm_dim[i] = comm_dim[i];
          this->dim[i] = dim[i];
        }
      }
    };

    template <int M, int N, int row_stride, int col_stride, bool dagger, class AccessorTo, class AccessorFrom>
    __device__ inline void load_cache(AccessorTo to_real, AccessorTo to_imag, AccessorFrom from)
    {
#if 1
      for (int col = threadIdx.y; col < N; col += col_stride) {
        for (int row = threadIdx.z; row < M; row += row_stride) {
          if (!dagger) {
            // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
            //   printf("y = %02d, z = %02d, row = %02d, col = %02d, y.real = %8.4e\n", threadIdx.y, threadIdx.z, row,
            //   col, double(from(col, row).real()));
            auto y = from(row, col);
            to_real(row, col) = __float2half(+y.real());
            to_imag(row, col) = __float2half(+y.imag());
          } else {
            auto y = from(col, row);
            to_real(row, col) = __float2half(+y.real());
            to_imag(row, col) = __float2half(-y.imag());
          }
        }
      }
#else
      for (int row = threadIdx.y * row_dim; row < (threadIdx.y + 1) * row_dim; row++) {
        for (int col = threadIdx.z * col_dim; col < (threadIdx.z + 1) * col_dim; col++) {
          if (!dagger) {
            // if(blockIdx.x == 2 && threadIdx.y == 3 && threadIdx.z == 0)
            //   printf("y = %02d, z = %02d, row = %02d, col = %02d, y.real = %8.4e\n", threadIdx.y, threadIdx.z, row,
            //   col, double(from(col, row).real()));
            auto y = from(row, col);
            to_real(row, col) = __float2half(+y.real());
            to_imag(row, col) = __float2half(+y.imag());
          } else {
            auto y = from(col, row);
            to_real(row, col) = __float2half(+y.real());
            to_imag(row, col) = __float2half(-y.imag());
          }
        }
      }
#endif
    }

    template <int M, int N, int row_stride, int col_stride, class AccessorTo, class AccessorFrom>
    __device__ inline void store_cache(AccessorTo to, AccessorFrom from_real, AccessorFrom from_imag)
    {
#if 1
      for (int col = threadIdx.y; col < N; col += col_stride) {
        for (int row = threadIdx.z; row < M; row += row_stride) {
          // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("y = %02d, z = %02d, row = %02d, col =
          // %02d\n", threadIdx.y, threadIdx.z, row, col);
          to(row, col) = complex<float>(__half2float(from_real(row, col)), __half2float(from_imag(row, col)));
        }
      }
#else
      for (int row = threadIdx.y * row_dim; row < (threadIdx.y + 1) * row_dim; row++) {
        for (int col = threadIdx.z * col_dim; col < (threadIdx.z + 1) * col_dim; col++) {
          // if(blockIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("y = %02d, z = %02d, row = %02d, col =
          // %02d\n", threadIdx.y, threadIdx.z, row, col);
          to(row, col) = complex<float>(__half2float(from_real(row, col)), __half2float(from_imag(row, col)));
        }
      }
#endif
    }

    template <bool compute_max_only, typename Arg, int bM, int bN, int bK, int block_y, int block_z>
    inline __device__ auto computeYhat(Arg &arg, int d, int x_cb, int parity, half *smem_ptr)
    {
      using real = typename Arg::Float;
      using complex = complex<real>;
      constexpr int nDim = 4;
      int coord[nDim];
      getCoords(coord, x_cb, arg.dim, parity);

      const int ghost_idx = ghostFaceIndex<0, nDim>(coord, arg.dim, d, arg.nFace);

      real yHatMax = 0.0;

      constexpr int M = bM; // block sizes, currently the same as matrix size.
      constexpr int N = bN;
      constexpr int K = bK;

      constexpr int lda = M + (M == 48 ? 2 : 10);
      constexpr int ldb = N + (M == 48 ? 2 : 10);

      constexpr int n_row = block_z;
      constexpr int n_col = block_y;

      constexpr int total_warp = n_row * n_col / warp_size;

      half *smem_a_real = smem_ptr;
      half *smem_a_imag = smem_a_real + lda * K;
      half *smem_b_real = smem_a_imag + lda * K;
      half *smem_b_imag = smem_b_real + ldb * K;

      auto smem_obj_a_real = make_smem_obj<M, K, 1, lda>(smem_a_real);
      auto smem_obj_a_imag = make_smem_obj<M, K, 1, lda>(smem_a_imag);
      auto smem_obj_b_real = make_smem_obj<N, K, 1, ldb>(smem_b_real);
      auto smem_obj_b_imag = make_smem_obj<N, K, 1, ldb>(smem_b_imag);

      // first do the backwards links Y^{+\mu} * X^{-\dagger}
      if (arg.comm_dim[d] && (coord[d] - arg.nFace < 0)) {

        auto aa = [&](int i, int j) { return arg.Y.Ghost(d, 1 - parity, ghost_idx, i, j); };
        load_cache<M, K, n_row, n_col, false>(smem_obj_a_real, smem_obj_a_imag, aa);

        auto bb = [&](int i, int j) { return arg.Xinv(0, parity, x_cb, j, i); };
        load_cache<K, N, n_row, n_col, true>(smem_obj_b_real, smem_obj_b_imag, bb);

        __syncthreads();

        zmma_sync_gemm<total_warp, M, N, K, lda, ldb>(smem_a_real, smem_a_imag, smem_b_real, smem_b_imag,
                                                      arg.Yhat(d, 1 - parity, ghost_idx, 0, 0));

        __syncthreads();

        /**
             auto yHat = make_tile_C<complex,true>(arg.tile);

       #pragma unroll
             for (int k = 0; k<arg.tile.k; k+=arg.tile.K) {
               auto Y = make_tile_A<complex, true>(arg.tile);
               Y.load(arg.Y, d, 1-parity, ghost_idx, i0, k);

               auto X = make_tile_Bt<complex, false>(arg.tile);
               X.load(arg.Xinv, 0, parity, x_cb, j0, k);

               yHat.mma_nt(Y, X);
             }
       */

        if (compute_max_only) {
          // yHatMax = yHat.abs_max();
        } else {
          // yHat.save(arg.Yhat, d, 1 - parity, ghost_idx, i0, j0);
        }

      } else {
        const int back_idx = linkIndexM1(coord, arg.dim, d);

        auto aa = [&](int i, int j) { return arg.Y(d, 1 - parity, back_idx, i, j); };
        load_cache<M, K, n_row, n_col, false>(smem_obj_a_real, smem_obj_a_imag, aa);

        auto bb = [&](int i, int j) { return arg.Xinv(0, parity, x_cb, j, i); };
        load_cache<K, N, n_row, n_col, true>(smem_obj_b_real, smem_obj_b_imag, bb);

        __syncthreads();

        zmma_sync_gemm<total_warp, M, N, K, lda, ldb>(smem_a_real, smem_a_imag, smem_b_real, smem_b_imag,
                                                      arg.Yhat(d, 1 - parity, back_idx, 0, 0));

        __syncthreads();

        /**
              auto yHat = make_tile_C<complex,false>(arg.tile);

        #pragma unroll
              for (int k = 0; k<arg.tile.k; k+=arg.tile.K) {
                auto Y = make_tile_A<complex, false>(arg.tile);
                Y.load(arg.Y, d, 1-parity, back_idx, i0, k);

                auto X = make_tile_Bt<complex, false>(arg.tile);
                X.load(arg.Xinv, 0, parity, x_cb, j0, k);

                yHat.mma_nt(Y, X);
              }
        */

        if (compute_max_only) {
          // yHatMax = yHat.abs_max();
        } else {
          // yHat.save(arg.Yhat, d, 1 - parity, back_idx, i0, j0);
        }
      }

      { // now do the forwards links X^{-1} * Y^{-\mu}

        auto aa = [&](int i, int j) { return arg.Xinv(0, parity, x_cb, i, j); };
        load_cache<M, K, n_row, n_col, false>(smem_obj_a_real, smem_obj_a_imag, aa);
        auto bb = [&](int i, int j) { return arg.Y(d + 4, parity, x_cb, j, i); };
        load_cache<K, N, n_row, n_col, false>(smem_obj_b_real, smem_obj_b_imag, bb);

        __syncthreads();

        zmma_sync_gemm<total_warp, M, N, K, lda, ldb>(smem_a_real, smem_a_imag, smem_b_real, smem_b_imag,
                                                      arg.Yhat(d + 4, parity, x_cb, 0, 0));

        __syncthreads();

        /**
              auto yHat = make_tile_C<complex, false>(arg.tile);

        #pragma unroll
              for (int k = 0; k<arg.tile.k; k+=arg.tile.K) {
                auto X = make_tile_A<complex, false>(arg.tile);
                X.load(arg.Xinv, 0, parity, x_cb, i0, k);

                auto Y = make_tile_B<complex, false>(arg.tile);
                Y.load(arg.Y, d + 4, parity, x_cb, k, j0);

                yHat.mma_nn(X, Y);
              }
         */
        if (compute_max_only) {
          // yHatMax = fmax(yHatMax, yHat.abs_max());
        } else {
          // yHat.save(arg.Yhat, d + 4, parity, x_cb, i0, j0);
        }
      }

      return yHatMax;
    }

    template <bool compute_max_only, typename Arg, int bM, int bN, int bK, int block_y, int block_z>
    __global__ void CalculateYhatGPU(Arg arg)
    {
      int x_cb = blockDim.x * blockIdx.x + threadIdx.x;
      if (x_cb >= arg.Y.VolumeCB()) return;

      // blockDim.y == arg.tile.M_tiles
      // int i_parity = blockDim.y * blockIdx.y + threadIdx.y;
      // if (i_parity >= 2 * arg.tile.M_tiles) return;
      // int i = threadIdx.y;     // i_parity % arg.tile.M_tiles;
      int parity = blockIdx.y; // i_parity / arg.tile.M_tiles;

      // blockDim.z == arg.tile.N_tiles
      // int j_d = blockDim.z*blockIdx.z + threadIdx.z;
      // if (j_d >= 4*arg.tile.N_tiles) return;
      // int j = threadIdx.z; // j_d % arg.tile.N_tiles;
      int d = blockIdx.z; // j_d / arg.tile.N_tiles;

      extern __shared__ half smem_ptr[];

      typename Arg::Float max
        = computeYhat<compute_max_only, Arg, bM, bN, bK, block_y, block_z>(arg, d, x_cb, parity, smem_ptr);
      if (compute_max_only) atomicAbsMax(arg.max_d, max);
    }

  } // namespace mma

} // namespace quda
