#pragma once

#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <linalg.cuh>
#include <matrix_tile.cuh>
#include <mma_tensor_op/gemm.cuh>
#include <block_reduce_helper.h>
#include <kernel.h>
#include <kernels/coarse_op_kernel.cuh>

namespace quda
{

  namespace mma
  {

    template <typename Arg, int dim_, QudaDirection dir_, int bM_, int bN_, int bK_, int block_y_, int block_z_,
              int min_blocks_ = 1>
    struct mmaArg : Arg {
      static constexpr int dim = dim_;
      static constexpr QudaDirection dir = dir_;
      static constexpr int bM = bM_;
      static constexpr int bN = bN_;
      static constexpr int bK = bK_;
      static constexpr int block_y = block_y_;
      static constexpr int block_z = block_z_;
      static constexpr int block_dim = block_y * block_z;
      static constexpr int min_blocks = min_blocks_;

      mmaArg(const Arg &arg) : Arg(arg)
      {
        static_assert(Arg::from_coarse, "The MMA implementation is only for from_coarse == true.");
      }
    };

    // This is the MMA implementation of the computeUV and computeVUV kernels for from_coarse == true.

    namespace impl
    {

      /**
        Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
        Where: mu = dir, s = fine spin, c' = coarse color, c = fine color
       */
      template <typename Wtype, typename Arg>
      __device__ __host__ inline auto computeUV(Arg &arg, const Wtype &Wacc, int parity, int x_cb, int m_offset,
                                                int n_offset)
      {
        using real = typename Arg::Float;
        constexpr int fineSpin = Arg::fineSpin;

        int coord[4];
        getCoords(coord, x_cb, arg.x_size, parity);

        real uvMax = 0.0;

        constexpr int nFace = 1;

        using TileType = typename Arg::uvTileType;

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;

        // Here instead of fineColor x coarseColor x fineColor,
        // we do (fineColor * fineSpin) x coarseColor x fineColor

        constexpr int M = TileType::m * fineSpin;
        constexpr int N = TileType::n;
        constexpr int K = TileType::k;

        constexpr int lda = K * fineSpin;
        constexpr int ldb = N;
        constexpr int ldc = N;

        using mma_t = typename mma::mg_mma_dispatch_t<typename Arg::Float>::type;
        using Config = MmaConfig<mma_t, M, N, K, lda, ldb, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

        if (Arg::dir == QUDA_IN_PLACE) {

          for (int s_col = 0; s_col < fineSpin; s_col++) {

            auto a = arg.C(0, parity, x_cb, 0, s_col, 0, 0);
            auto b = Wacc(parity, x_cb, s_col, 0, 0);
            auto c = arg.UV(parity, x_cb, s_col * fineSpin, 0, 0);

            uvMax = fmax(
              uvMax, Config::template perform_mma<a_dagger, b_dagger, Arg::compute_max>(a, b, c, m_offset, n_offset));
          }

        } else if (arg.comm_dim[Arg::dim] && (coord[Arg::dim] + nFace >= arg.x_size[Arg::dim])) {

          int ghost_idx = ghostFaceIndex<1>(coord, arg.x_size, Arg::dim, nFace);

          for (int s_col = 0; s_col < fineSpin; s_col++) {

            auto a = arg.U(Arg::dim + (Arg::dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, 0, s_col, 0, 0);
            auto b = Wacc.Ghost(Arg::dim, 1, (parity + 1) & 1, ghost_idx, s_col, 0, 0);
            auto c = arg.UV(parity, x_cb, s_col * fineSpin, 0, 0);

            uvMax = fmax(
              uvMax, Config::template perform_mma<a_dagger, b_dagger, Arg::compute_max>(a, b, c, m_offset, n_offset));
          }

        } else {

          int y_cb = linkIndexP1(coord, arg.x_size, Arg::dim);

          for (int s_col = 0; s_col < fineSpin; s_col++) {

            auto a = arg.U(Arg::dim + (Arg::dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, 0, s_col, 0, 0);
            auto b = Wacc((parity + 1) & 1, y_cb, s_col, 0, 0);
            auto c = arg.UV(parity, x_cb, s_col * fineSpin, 0, 0);

            uvMax = fmax(
              uvMax, Config::template perform_mma<a_dagger, b_dagger, Arg::compute_max>(a, b, c, m_offset, n_offset));
          }
        }

        return uvMax;
      } // computeUV

    } // namespace impl

    template <typename Arg> struct ComputeUVMMA {
      Arg &arg;
      constexpr ComputeUVMMA(Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __forceinline__ void operator()()
      {
        int index_x = blockDim.x * blockIdx.x + threadIdx.x;
        int parity = blockIdx.y;

        constexpr int M = Arg::uvTileType::m * Arg::fineSpin;
        constexpr int N = Arg::uvTileType::n;
        constexpr int K = Arg::uvTileType::k;

        constexpr bool divide_b_no = Arg::bM < M && Arg::bK >= K && Arg::bN == N;

        constexpr int t_m = divide_b_no ? 1 : (Arg::uvTileType::m * Arg::fineSpin + Arg::bM - 1) / Arg::bM;
        constexpr int t_n = divide_b_no ? 1 : (Arg::uvTileType::n + Arg::bN - 1) / Arg::bN;

        int x_cb = index_x / (t_m * t_n);
        if (x_cb >= arg.fineVolumeCB) return;

        int mn = index_x % (t_m * t_n);

        int n_offset = (mn % t_n) * Arg::bN;
        int m_offset = (mn / t_n) * Arg::bM;

        using real = typename Arg::Float;
        real max = 0.0;
        if (Arg::dir == QUDA_FORWARDS || Arg::dir == QUDA_IN_PLACE) // only for preconditioned clover is V != AV
          max = impl::computeUV(arg, arg.V, parity, x_cb, m_offset, n_offset);
        else
          max = impl::computeUV(arg, arg.AV, parity, x_cb, m_offset, n_offset);

        if (Arg::compute_max) {
          constexpr int block_dim = 3;
          unsigned aggregate = BlockReduce<unsigned, block_dim>().Max(__float_as_uint(max));
          if (threadIdx.y == 0 && threadIdx.z == 0) atomic_fetch_abs_max(arg.max_d, __uint_as_float(aggregate));
        }
      }
    };

    namespace impl
    {

      template <typename Arg> __device__ void computeVUV(Arg &arg, int parity, int x_cb, int m_offset, int n_offset)
      {
        constexpr int fineSpin = Arg::fineSpin;
        constexpr int coarseSpin = Arg::coarseSpin;

        constexpr int nDim = 4;
        constexpr int nFace = 1;
        int coord[QUDA_MAX_DIM];
        int coord_coarse[QUDA_MAX_DIM];

        getCoords(coord, x_cb, arg.x_size, parity);
        for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d] / arg.geo_bs[d];

        constexpr bool isFromCoarseClover = Arg::dir == QUDA_IN_PLACE;

        // Check to see if we are on the edge of a block.  If adjacent site
        // is in same block, M = X, else M = Y
        const bool isDiagonal = isFromCoarseClover || isCoarseDiagonal(coord, coord_coarse, Arg::dim, nFace, arg);

        int coarse_parity = 0;

        for (int d = 0; d < nDim; d++) coarse_parity += coord_coarse[d];
        coarse_parity &= 1;
        coord_coarse[0] /= 2;

        int coarse_x_cb = ((coord_coarse[3] * arg.xc_size[2] + coord_coarse[2]) * arg.xc_size[1] + coord_coarse[1])
            * (arg.xc_size[0] / 2)
          + coord_coarse[0];

        using TileType = typename Arg::vuvTileType;
        // We do coarseColor x coarseColor x fineColor

        constexpr bool a_dagger = true;
        constexpr bool b_dagger = false;

        constexpr int M = TileType::m;
        constexpr int N = TileType::n;
        constexpr int K = TileType::k;

        constexpr int lda = N; // Since a_dagger == true here it's N instead of K.
        constexpr int ldb = N;
        constexpr int ldc = N * coarseSpin;

        using mma_t = typename mma::mg_mma_dispatch_t<typename Arg::Float>::type;

        extern __shared__ typename mma_t::compute_t smem_ptr[];

        using Config = MmaConfig<mma_t, M, N, K, lda, ldb, ldc, Arg::bM, Arg::bN, Arg::bK, Arg::block_y, Arg::block_z>;

        static_assert(K <= Arg::bK, "Dividing K has NOT been implemented yet.\n");

        typename Config::SmemObjA smem_obj_a_real(smem_ptr);
        typename Config::SmemObjA smem_obj_a_imag(smem_obj_a_real.ptr + Config::smem_lda * Arg::bK);
        typename Config::SmemObjB smem_obj_b_real(smem_obj_a_imag.ptr + Config::smem_lda * Arg::bK);
        typename Config::SmemObjB smem_obj_b_imag(smem_obj_b_real.ptr + Config::smem_ldb * Arg::bK);

        typename mma_t::WarpRegisterMapping wrm((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);

        using op_c_type = typename mma_t::OperandC;

        typename Config::ALoader a_loader;
        typename Config::BLoader b_loader;

        /**
          Here we directly put the implementation of the MMA kernel here because computeVUV uses more
          atomic for storing output data, and the shared memory loaded for operand A can be reused for
          various spin compoents
        */

        // Not unrolling to lift regiter pressure
        for (int s = 0; s < fineSpin; s++) {

          auto a = arg.AV(parity, x_cb, s, 0, 0);

          __syncthreads();
          a_loader.template g2r<Config::lda, a_dagger>(a, m_offset, 0);
          a_loader.template r2s<a_dagger>(smem_obj_a_real, smem_obj_a_imag);
          __syncthreads();

          for (int s_col = 0; s_col < fineSpin; s_col++) { // which chiral block

            auto b = arg.UV(parity, x_cb, s_col * fineSpin + s, 0, 0);

            __syncthreads();
            b_loader.template g2r<Config::ldb, b_dagger>(b, n_offset, 0);
            b_loader.template r2s<b_dagger>(smem_obj_b_real, smem_obj_b_imag);
            __syncthreads();

#pragma unroll 1
            for (int c = 0; c < Config::warp_cycle; c++) {

              // The logical warp assigned to each part of the matrix.
              int logical_warp_index = wrm.warp_id * Config::warp_cycle + c;
              int warp_row = logical_warp_index / Config::tile_col_dim;
              int warp_col = logical_warp_index - warp_row * Config::tile_col_dim;

              op_c_type op_c_real;
              op_c_type op_c_imag;

#pragma unroll 1
              for (int tile_k = 0; tile_k < Config::tile_acc_dim; tile_k++) {
                complex_mma<mma_t>(smem_obj_a_real, smem_obj_a_imag, smem_obj_b_real, smem_obj_b_imag, op_c_real,
                                   op_c_imag, warp_row, warp_col, tile_k, wrm);
              }

              int warp_m_offset = warp_row * mma_t::MMA_M + m_offset;
              int warp_n_offset = warp_col * mma_t::MMA_N + n_offset;

              if (Arg::dir == QUDA_IN_PLACE) {

                auto cc = arg.X_atomic(0, coarse_parity, coarse_x_cb, s, s_col, 0, 0);
                constexpr bool atomic_dagger = false;
                mma_t::template store_complex<M, N, N * fineSpin, atomic_dagger>(
                  warp_m_offset, warp_n_offset, wrm, cc, op_c_real, op_c_imag, fetch_add_atomic_t());

              } else if (!isDiagonal) {

                int dim_index = arg.dim_index % arg.Y_atomic.geometry;
                auto cc = arg.Y_atomic(dim_index, coarse_parity, coarse_x_cb, s, s_col, 0, 0);
                constexpr bool atomic_dagger = false;
                mma_t::template store_complex<M, N, N * fineSpin, atomic_dagger>(
                  warp_m_offset, warp_n_offset, wrm, cc, op_c_real, op_c_imag, fetch_add_atomic_t());

              } else {

                if (!isFromCoarseClover) {
                  op_c_real.ax(-arg.kappa);
                  op_c_imag.ax(-arg.kappa);
                }

                if (Arg::dir == QUDA_BACKWARDS) {
                  auto cc = arg.X_atomic(0, coarse_parity, coarse_x_cb, s_col, s, 0, 0);
                  constexpr bool atomic_dagger = true;
                  mma_t::template store_complex<M, N, N * fineSpin, atomic_dagger>(
                    warp_m_offset, warp_n_offset, wrm, cc, op_c_real, op_c_imag, fetch_add_atomic_t());
                } else {
                  auto cc = arg.X_atomic(0, coarse_parity, coarse_x_cb, s, s_col, 0, 0);
                  constexpr bool atomic_dagger = false;
                  mma_t::template store_complex<M, N, N * fineSpin, atomic_dagger>(
                    warp_m_offset, warp_n_offset, wrm, cc, op_c_real, op_c_imag, fetch_add_atomic_t());
                }

                if (!arg.bidirectional) {
                  if (s != s_col) {
                    op_c_real.ax(static_cast<float>(-1.0));
                    op_c_imag.ax(static_cast<float>(-1.0));
                  }
                  constexpr bool atomic_dagger = false;
                  auto cc = arg.X_atomic(0, coarse_parity, coarse_x_cb, s, s_col, 0, 0);
                  mma_t::template store_complex<M, N, N * fineSpin, atomic_dagger>(
                    warp_m_offset, warp_n_offset, wrm, cc, op_c_real, op_c_imag, fetch_add_atomic_t());
                }
              }
            }
          } // Fine color
        }   // Fine spin
      }

    } // namespace impl

    template <typename Arg> struct ComputeVUVMMA {
      Arg &arg;
      constexpr ComputeVUVMMA(Arg &arg) : arg(arg) {}
      static constexpr const char *filename() { return KERNEL_FILE; }

      __device__ __forceinline__ void operator()()
      {
        int index_x = blockDim.x * blockIdx.x + threadIdx.x;
        int parity = blockIdx.y;

        constexpr int t_m = (Arg::vuvTileType::m + Arg::bM - 1) / Arg::bM;
        constexpr int t_n = (Arg::vuvTileType::n + Arg::bN - 1) / Arg::bN;

        int x_cb = index_x / (t_m * t_n);
        if (x_cb >= arg.fineVolumeCB) return;

        int mn = index_x % (t_m * t_n);

        int n_offset = (mn % t_n) * Arg::bN;
        int m_offset = (mn / t_n) * Arg::bM;

        impl::computeVUV(arg, parity, x_cb, m_offset, n_offset);
      }
    };

  } // namespace mma

} // namespace quda
