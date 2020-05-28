#pragma once

#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <linalg.cuh>
#include <matrix_tile.cuh>

#include <mma_tensor_op/gemm.cuh>

namespace quda
{

  namespace mma
  {

    // this is the storage type used when computing the coarse link variables
    // by using integers we have deterministic atomics
    typedef int storeType;

    template <typename Float_, int fineSpin, int coarseSpin, int fineColor, int coarseColor, typename coarseGauge,
              typename coarseGaugeAtomic, typename fineGauge, typename fineSpinor, typename fineSpinorTmp,
              typename fineSpinorV, typename fineClover>
    struct CalculateYArg {
      using Float = Float_;

      coarseGauge Y; /** Computed coarse link field */
      coarseGauge X; /** Computed coarse clover field */

      coarseGaugeAtomic Y_atomic; /** Y atomic accessor used for computation before conversion to final format */
      coarseGaugeAtomic X_atomic; /** X atomic accessor used for computation before conversion to final format */

      fineSpinorTmp UV; /** Temporary that stores the fine-link * spinor field product */
      fineSpinor AV;    /** Temporary that stores the clover * spinor field product */

      const fineGauge U;     /** Fine grid link field */
      const fineSpinorV V;   /** Fine grid spinor field */
      const fineClover C;    /** Fine grid clover field */
      const fineClover Cinv; /** Fine grid clover field */

      int_fastdiv x_size[QUDA_MAX_DIM]; /** Dimensions of fine grid */
      int xc_size[QUDA_MAX_DIM];        /** Dimensions of coarse grid */

      int_fastdiv geo_bs[QUDA_MAX_DIM];                 /** Geometric block dimensions */
      const int spin_bs;                                /** Spin block size */
      const spin_mapper<fineSpin, coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

      int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

      Float kappa;     /** kappa value */
      Float mu;        /** mu value */
      Float mu_factor; /** multiplicative factor for mu applied when mu is added to the operator */
      Float rescale;   /** rescaling factor used when rescaling the Y links if the maximum increases */

      const int fineVolumeCB;   /** Fine grid volume */
      const int coarseVolumeCB; /** Coarse grid volume */

      const int *fine_to_coarse;
      const int *coarse_to_fine;

      const bool bidirectional;

      // To increase L2 locality we can schedule the geometry to grid.y and
      // the coarse colors to grid.x.  This will increase the potential for
      // L2 reuse since a given wave of thread blocks will be for different
      // coarse color but the same coarse grid point which will have common
      // loads.
      static constexpr bool coarse_color_wave = false;

      // Enable this for shared-memory atomics instead of global atomics.
      // Doing so means that all (modulo the parity) of the coarsening for a
      // coarse degree of freedom is handled by a single thread block.
      // For computeVUV only at present
      bool shared_atomic;

      // With parity_flip enabled we make parity the slowest running
      // dimension in the y-thread axis, and coarse color runs faster.  This
      // improves read locality at the expense of write locality
      bool parity_flip;

      int_fastdiv aggregates_per_block; // number of aggregates per thread block
      int_fastdiv grid_z; // this is the coarseColor grid that is wrapped into the x grid when coarse_color_wave is enabled
      int_fastdiv coarse_color_grid_z; // constant we ned to divide by

      Float max_h;  // scalar that stores the maximum element of the dynamic clover inverse
      Float *max_d; // array that stores the maximum element per lattice site of the dynamic clover inverse

      int dim;           // which dimension are we working on
      QudaDirection dir; // which direction are working on
      int dim_index;     // which direction / dimension we are working on

      // tile used for computeUV
      static constexpr int tile_height_uv = fineColor % 4 == 0 ? 4 : fineColor % 3 == 0 ? 3 : fineColor % 2 ? 2 : 1;
      static constexpr int tile_width_uv = coarseColor % 2 == 0 ? 2 : 1;
      TileSize<fineColor, coarseColor, fineColor, tile_height_uv, tile_width_uv, 1> uvTile;

      // tile used for computeVUV - for fine grids best to use 4, else use max of 3
      static constexpr int tile_height_vuv = (coarseColor % 4 == 0 && fineSpin == 4) ? 4 : coarseColor % 3 == 0 ? 3 : 2;
      static constexpr int tile_width_vuv = coarseColor % 2 == 0 ? 2 : 1;
      TileSize<coarseColor, coarseColor, fineColor, tile_height_vuv, tile_width_vuv, 1> vuvTile;

      // max colors per block is 8, rounded up to whole multiples of tile size
      static constexpr int max_color_height_per_block
        = coarseColor < 8 ? coarseColor : ((8 + tile_height_vuv - 1) / tile_height_vuv) * tile_height_vuv;
      static constexpr int max_color_width_per_block
        = coarseColor < 8 ? coarseColor : ((8 + tile_width_vuv - 1) / tile_width_vuv) * tile_width_vuv;
      static constexpr int max_height_tiles_per_block = max_color_height_per_block / tile_height_vuv;
      static constexpr int max_width_tiles_per_block = max_color_width_per_block / tile_width_vuv;
      static_assert(max_color_height_per_block % tile_height_vuv == 0,
                    "max_color_height_per_block must be divisible by tile height");
      static_assert(max_color_width_per_block % tile_width_vuv == 0,
                    "max_color_width_per_block must be divisible by tile width");

      CalculateYArg(coarseGauge &Y, coarseGauge &X, coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
                    fineSpinorTmp &UV, fineSpinor &AV, const fineGauge &U, const fineSpinorV &V, const fineClover &C,
                    const fineClover &Cinv, double kappa, double mu, double mu_factor, const int *x_size_,
                    const int *xc_size_, int *geo_bs_, int spin_bs_, const int *fine_to_coarse,
                    const int *coarse_to_fine, bool bidirectional) :
        Y(Y),
        X(X),
        Y_atomic(Y_atomic),
        X_atomic(X_atomic),
        UV(UV),
        AV(AV),
        U(U),
        V(V),
        C(C),
        Cinv(Cinv),
        spin_bs(spin_bs_),
        spin_map(),
        kappa(static_cast<Float>(kappa)),
        mu(static_cast<Float>(mu)),
        mu_factor(static_cast<Float>(mu_factor)),
        fineVolumeCB(V.VolumeCB()),
        coarseVolumeCB(X.VolumeCB()),
        fine_to_coarse(fine_to_coarse),
        coarse_to_fine(coarse_to_fine),
        bidirectional(bidirectional),
        shared_atomic(false),
        parity_flip(shared_atomic ? true : false),
        aggregates_per_block(1),
        max_d(nullptr)
      {
        if (V.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) errorQuda("Gamma basis %d not supported", V.GammaBasis());

        for (int i = 0; i < QUDA_MAX_DIM; i++) {
          x_size[i] = x_size_[i];
          xc_size[i] = xc_size_[i];
          geo_bs[i] = geo_bs_[i];
          comm_dim[i] = comm_dim_partitioned(i);
        }
      }
    };

    namespace impl
    {

      /**
        Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
  Where: mu = dir, s = fine spin, c' = coarse color, c = fine color
       */
      template <bool from_coarse, int dim, QudaDirection dir, int fineSpin, int coarseSpin, int bM, int bN, int bK,
                int block_y, int block_z, typename Wtype, typename Arg>
      __device__ __host__ inline void computeUV(Arg &arg, const Wtype &Wacc, int parity, int x_cb)
      {
        int coord[4];
        getCoords(coord, x_cb, arg.x_size, parity);

        // constexpr int uvSpin = fineSpin * (from_coarse ? 2 : 1);
        constexpr int nFace = 1;

        auto &tile = arg.uvTile;

        constexpr bool a_dagger = false;
        constexpr bool b_dagger = false;
        constexpr bool compute_max_only = false;

        if (arg.comm_dim[dim] && (coord[dim] + nFace >= arg.x_size[dim])) {
          int ghost_idx = ghostFaceIndex<1>(coord, arg.x_size, dim, nFace);

          if (!from_coarse) {
#if 1

#else
            for (int k = 0; k < tile.k; k += tile.K) { // Fine Color columns of gauge field
              auto U = make_tile_A<complex, false>(tile);
              U.load(arg.U, dim, parity, x_cb, i0, k);
              for (int s = 0; s < fineSpin; s++) { // Fine Spin
                auto W = make_tile_B<complex, true>(tile);
                W.loadCS(Wacc, dim, 1, (parity + 1) & 1, ghost_idx, s, k, j0);
                UV[s].mma_nn(U, W);
              } // Fine color columns
            }   // Fine spin (tensor)
#endif
          } else {
#if 1
            for (int s_col = 0; s_col < fineSpin; s_col++) {
              // here instead of fineColor x coarseColor x fineColor,
              // we do (fineColor * fineSpin) x coarseColor x fineColor
              auto aa = arg.U.wrap(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, 0, s_col);
              auto bb = Wacc.wrap_ghost(dim, 1, (parity + 1) & 1, ghost_idx, s_col);
              auto cc = arg.UV.wrap(parity, x_cb, s_col * fineSpin);

              constexpr int M = tile.m * fineSpin;
              constexpr int N = tile.n;
              constexpr int K = tile.k;

              constexpr int lda = K * fineSpin;
              constexpr int ldb = N;
              constexpr int ldc = N;

              using Config = MmaConfig<M, N, K, lda, ldb, ldc, bM, bN, bK>;

              perform_mma<Config, block_y, block_z, a_dagger, b_dagger, compute_max_only>(aa, bb, cc, 0, 0);
            }
#else
            for (int k = 0; k < tile.k; k += tile.K) { // Fine Color columns of gauge field
              for (int s_col = 0; s_col < fineSpin; s_col++) {
                auto W = make_tile_B<complex, true>(tile);
                W.loadCS(Wacc, dim, 1, (parity + 1) & 1, ghost_idx, s_col, k, j0);
                for (int s = 0; s < fineSpin; s++) { // Fine Spin
                  // on coarse lattice, if forwards then use forwards links
                  auto U = make_tile_A<complex, false>(tile);
                  U.load(arg.U, dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, i0, k);

                  UV[s_col * fineSpin + s].mma_nn(U, W);
                } // which chiral block
              }   // Fine color columns
            }     // Fine Spin
#endif
          } // from_coarse
        } else {
          int y_cb = linkIndexP1(coord, arg.x_size, dim);

          if (!from_coarse) {
#if 1

#else
            for (int k = 0; k < tile.k; k += tile.K) { // Fine Color columns of gauge field
              auto U = make_tile_A<complex, false>(tile);
              U.load(arg.U, dim, parity, x_cb, i0, k);
              for (int s = 0; s < fineSpin; s++) { // Fine Spin
                auto W = make_tile_B<complex, false>(tile);
                W.loadCS(Wacc, 0, 0, (parity + 1) & 1, y_cb, s, k, j0);
                UV[s].mma_nn(U, W);
              } // Fine color columns
            }   // Fine Spin
#endif
          } else {
#if 1
            for (int s_col = 0; s_col < fineSpin; s_col++) {
              // here instead of fineColor x coarseColor x fineColor,
              // we do (fineColor * fineSpin) x coarseColor x fineColor
              // TODO: Need to have accessor methods in the corresponding color spinor fields.
              auto aa = arg.U.wrap(dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, 0, s_col);
              auto bb = Wacc.wrap((parity + 1) & 1, y_cb, s_col);
              auto cc = arg.UV.wrap(parity, x_cb, s_col * fineSpin);

              constexpr int M = tile.m * fineSpin;
              constexpr int N = tile.n;
              constexpr int K = tile.k;

              constexpr int lda = K * fineSpin;
              constexpr int ldb = N;
              constexpr int ldc = N;

              using Config = MmaConfig<M, N, K, lda, ldb, ldc, bM, bN, bK>;

              perform_mma<Config, block_y, block_z, a_dagger, b_dagger, compute_max_only>(aa, bb, cc, 0, 0);
            }
#else
            for (int k = 0; k < tile.k; k += tile.K) { // Fine Color columns of gauge field
              for (int s_col = 0; s_col < fineSpin; s_col++) {
                auto W = make_tile_B<complex, false>(tile);
                W.loadCS(Wacc, 0, 0, (parity + 1) & 1, y_cb, s_col, k, j0);
                for (int s = 0; s < fineSpin; s++) { // Fine Spin
                  // on coarse lattice, if forwards then use forwards links
                  auto U = make_tile_A<complex, false>(tile);
                  U.load(arg.U, dim + (dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, i0, k);

                  UV[s_col * fineSpin + s].mma_nn(U, W);
                } // which chiral block
              }   // Fine Spin
            }     // Fine color columns
#endif
          }
        }
#if 0
      for (int s = 0; s < uvSpin; s++) UV[s].saveCS(arg.UV, 0, 0, parity, x_cb, s, i0, j0);
#endif
      } // computeUV

    } // namespace impl

    template <bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int coarseSpin, int bM,
              int bN, int bK, int block_y, int block_z, typename Arg>
    __global__ void ComputeUVMMA(Arg arg)
    {
      int x_cb = blockDim.x * blockIdx.x + threadIdx.x;
      if (x_cb >= arg.fineVolumeCB) return;

      int parity = blockIdx.y;

      if (dir == QUDA_FORWARDS) // only for preconditioned clover is V != AV
        impl::computeUV<from_coarse, dim, dir, fineSpin, coarseSpin, bM, bN, bK, block_y, block_z>(arg, arg.V, parity,
                                                                                                   x_cb);
      else
        impl::computeUV<from_coarse, dim, dir, fineSpin, coarseSpin, bM, bN, bK, block_y, block_z>(arg, arg.AV, parity,
                                                                                                   x_cb);
    }

    namespace impl
    {
      /**
         @brief Do a single (AV)^\dagger * UV product, where for preconditioned
         clover, AV correspond to the clover inverse multiplied by the
         packed null space vectors, else AV is simply the packed null
         space vectors.

         @param[out] vuv Result array
         @param[in,out] arg Arg storing the fields and parameters
         @param[in] Fine grid parity we're working on
         @param[in] x_cb Checkboarded x dimension
       */
      template <bool from_coarse, typename Float, int dim, QudaDirection dir, int fineSpin, int coarseSpin,
                typename Arg, typename Gamma, typename Out>
      __device__ __host__ inline void multiplyVUV(Out &vuv, const Arg &arg, const Gamma &gamma, int parity, int x_cb,
                                                  int i0, int j0)
      {
        using complex = complex<Float>;
        auto &tile = arg.vuvTile;

        if (!from_coarse) { // fine grid is top level
#if 1

#else
#pragma unroll
          for (int s = 0; s < fineSpin; s++) { // Loop over fine spin

            // Spin part of the color matrix.  Will always consist
            // of two terms - diagonal and off-diagonal part of
            // P_mu = (1+/-\gamma_mu)

            const int s_c_row = arg.spin_map(s, parity); // Coarse spin row index

            // Use Gamma to calculate off-diagonal coupling and
            // column index.  Diagonal coupling is always 1.
            // If computing the backwards (forwards) direction link then
            // we desire the positive (negative) projector

            const int s_col = gamma.getcol(s);
            const int s_c_col = arg.spin_map(s_col, parity); // Coarse spin col index

#pragma unroll
            for (int k = 0; k < tile.k; k += tile.K) { // Sum over fine color
              if (dir == QUDA_BACKWARDS) {
                auto V = make_tile_At<complex, false>(tile);
                V.loadCS(arg.V, 0, 0, parity, x_cb, s, k, i0);

                // here UV is really UAV
                // Diagonal Spin
                auto UV = make_tile_B<complex, false>(tile);
                UV.loadCS(arg.UV, 0, 0, parity, x_cb, s, k, j0);
                vuv[s_c_row * coarseSpin + s_c_row].mma_tn(V, UV);

                // Off-diagonal Spin (backward link / positive projector applied)
                auto gammaV = make_tile_A<complex, false>(tile);
                for (int i = 0; i < tile.K; i++)
                  for (int j = 0; j < tile.M; j++) { gammaV(j, i) = gamma.apply(s, conj(V(i, j))); }
                UV.loadCS(arg.UV, 0, 0, parity, x_cb, s_col, k, j0);
                vuv[s_c_row * coarseSpin + s_c_col].mma_nn(gammaV, UV);
              } else {
                auto AV = make_tile_At<complex, false>(tile);
                AV.loadCS(arg.AV, 0, 0, parity, x_cb, s, k, i0);

                auto UV = make_tile_B<complex, false>(tile);
                UV.loadCS(arg.UV, 0, 0, parity, x_cb, s, k, j0);

                // Diagonal Spin
                vuv[s_c_row * coarseSpin + s_c_row].mma_tn(AV, UV);

                // Off-diagonal Spin (forward link / negative projector applied)
                auto gammaAV = make_tile_A<complex, false>(tile);

                for (int i = 0; i < tile.K; i++)
                  for (int j = 0; j < tile.M; j++) { gammaAV(j, i) = -gamma.apply(s, conj(AV(i, j))); }
                UV.loadCS(arg.UV, 0, 0, parity, x_cb, s_col, k, j0);
                vuv[s_c_row * coarseSpin + s_c_col].mma_nn(gammaAV, UV);
              }
            } // Fine color
          }
#endif
        } else { // fine grid operator is a coarse operator
#if 1
          // We do coarseColor x coarseColor x fineColor

          constexpr bool a_dagger = true;
          constexpr bool b_dagger = false;

          constexpr int M = tile.m;
          constexpr int N = tile.n;
          constexpr int K = tile.k;

          constexpr int lda = K;
          constexpr int ldb = N;
          constexpr int ldc = N;

          constexpr int bM = M;
          constexpr int bN = N;
          constexpr int bK = K;

          using Config = MmaConfig<M, N, K, lda, ldb, ldc, bM, bN, bK>;

          // Not unrolling to lift regiter pressure
          for (int s = 0; s < fineSpin; s++) {

            for (int s_col = 0; s_col < fineSpin; s_col++) { // which chiral block

              auto aa = arg.AV.wrap(parity, x_cb, s);
              auto bb = arg.UV.wrap(parity, x_cb, s_col * fineSpin + s);

              // XXX: Need to factor the atomic store in here to the wrapper cc.
              // auto cc = arg.Y_atomic.wrap( ... )

            } // Fine color
          }   // Fine spin

#else
#pragma unroll
          for (int k = 0; k < tile.k; k += tile.K) { // Sum over fine color
#pragma unroll
            for (int s = 0; s < fineSpin; s++) {
              auto AV = make_tile_At<complex, false>(tile);
              AV.loadCS(arg.AV, 0, 0, parity, x_cb, s, k, i0);
#pragma unroll
              for (int s_col = 0; s_col < fineSpin; s_col++) { // which chiral block
                auto UV = make_tile_B<complex, false>(tile);
                UV.loadCS(arg.UV, 0, 0, parity, x_cb, s_col * fineSpin + s, k, j0);
                vuv[s * coarseSpin + s_col].mma_tn(AV, UV);
              } // Fine color
            }   // Fine spin
          }
#endif
        } // from_coarse
      }

      template <bool parity_flip, typename Float, QudaDirection dir, int coarseSpin, typename VUV, typename Arg>
      inline __device__ __host__ void storeCoarseGlobalAtomic(VUV &vuv, bool isDiagonal, int coarse_x_cb,
                                                              int coarse_parity, int i0, int j0, Arg &arg)
      {
        const int dim_index = arg.dim_index % arg.Y_atomic.geometry;
        auto &tile = arg.vuvTile;

        if (!isDiagonal) {
#pragma unroll
          for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
            for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
#pragma unroll
              for (int i = 0; i < tile.M; i++)
#pragma unroll
                for (int j = 0; j < tile.N; j++)
                  arg.Y_atomic.atomicAdd(dim_index, coarse_parity, coarse_x_cb, s_row, s_col, i0 + i, j0 + j,
                                         vuv[s_row * coarseSpin + s_col](i, j));
            }
          }
        } else {

          if (dir == QUDA_BACKWARDS) {
#pragma unroll
            for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
              for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
#pragma unroll
                for (int i = 0; i < tile.M; i++)
#pragma unroll
                  for (int j = 0; j < tile.N; j++)
                    arg.X_atomic.atomicAdd(0, coarse_parity, coarse_x_cb, s_col, s_row, j0 + j, i0 + i,
                                           conj(vuv[s_row * coarseSpin + s_col](i, j)));
              }
            }
          } else {
#pragma unroll
            for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
              for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
#pragma unroll
                for (int i = 0; i < tile.M; i++)
#pragma unroll
                  for (int j = 0; j < tile.N; j++)
                    arg.X_atomic.atomicAdd(0, coarse_parity, coarse_x_cb, s_row, s_col, i0 + i, j0 + j,
                                           vuv[s_row * coarseSpin + s_col](i, j));
              }
            }
          }

          if (!arg.bidirectional) {
#pragma unroll
            for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
              for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
                if (s_row != s_col) vuv[s_row * coarseSpin + s_col] *= static_cast<Float>(-1.0);
#pragma unroll
                for (int i = 0; i < tile.M; i++)
#pragma unroll
                  for (int j = 0; j < tile.N; j++)
                    arg.X_atomic.atomicAdd(0, coarse_parity, coarse_x_cb, s_row, s_col, i0 + i, j0 + j,
                                           vuv[s_row * coarseSpin + s_col](i, j));
              }
            }
          }
        }
      }

      template <bool shared_atomic, bool parity_flip, bool from_coarse, typename Float, int dim, QudaDirection dir,
                int fineSpin, int coarseSpin, typename Arg, typename Gamma>
      __device__ __host__ void computeVUV(Arg &arg, const Gamma &gamma, int parity, int x_cb, int i0, int j0,
                                          int parity_coarse_, int coarse_x_cb_)
      {
        constexpr int nDim = 4;
        int coord[QUDA_MAX_DIM];
        int coord_coarse[QUDA_MAX_DIM];

        getCoords(coord, x_cb, arg.x_size, parity);
        for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d] / arg.geo_bs[d];

        // Check to see if we are on the edge of a block.  If adjacent site
        // is in same block, M = X, else M = Y
        const bool isDiagonal
          = ((coord[dim] + 1) % arg.x_size[dim]) / arg.geo_bs[dim] == coord_coarse[dim] ? true : false;

        int coarse_parity = shared_atomic ? parity_coarse_ : 0;
        if (!shared_atomic) {
          for (int d = 0; d < nDim; d++) coarse_parity += coord_coarse[d];
          coarse_parity &= 1;
          coord_coarse[0] /= 2;
        }
        int coarse_x_cb = shared_atomic ?
          coarse_x_cb_ :
          ((coord_coarse[3] * arg.xc_size[2] + coord_coarse[2]) * arg.xc_size[1] + coord_coarse[1]) * (arg.xc_size[0] / 2)
            + coord_coarse[0];

        using Ctype = decltype(make_tile_C<complex<Float>, false>(arg.vuvTile));
        Ctype vuv[coarseSpin * coarseSpin];
        multiplyVUV<from_coarse, Float, dim, dir, fineSpin, coarseSpin, Arg>(vuv, arg, gamma, parity, x_cb, i0, j0);

        if (isDiagonal) {
#pragma unroll
          for (int s2 = 0; s2 < coarseSpin * coarseSpin; s2++) vuv[s2] *= -arg.kappa;
        }

        if (shared_atomic) {
          // blank
        } else {
          storeCoarseGlobalAtomic<parity_flip, Float, dir, coarseSpin>(vuv, isDiagonal, coarse_x_cb, coarse_parity, i0,
                                                                       j0, arg);
        }
      }

      // compute indices for global-atomic kernel
      template <bool shared_atomic, bool parity_flip, typename Arg>
      __device__ inline void getIndices(const Arg &arg, int &parity, int &x_cb, int &parity_coarse, int &x_coarse_cb,
                                        int &c_row, int &c_col)
      {
        if (arg.coarse_color_wave) {
          // blank
        } else {
          int parity_c_row = blockDim.y * blockIdx.y + threadIdx.y;
          c_row = parity_flip ? (parity_c_row % arg.vuvTile.M_tiles) : (parity_c_row / 2);  // coarse color row index
          parity = parity_flip ? (parity_c_row / arg.vuvTile.M_tiles) : (parity_c_row % 2); // coarse color row index
          c_col = blockDim.z * blockIdx.z + threadIdx.z;                                    // coarse color col index
        }

        if (!shared_atomic) {
          x_cb = blockDim.x * (arg.coarse_color_wave ? blockIdx.y : blockIdx.x) + threadIdx.x;
          x_coarse_cb = 0;
          parity_coarse = 0;
        } else {
          // blank
        }
      }

    } // namespace impl

    template <bool shared_atomic, bool parity_flip, bool from_coarse, typename Float, int dim, QudaDirection dir,
              int fineSpin, int coarseSpin, typename Arg>
    __global__ void ComputeVUVMMA(Arg arg)
    {
      static_assert(shared_atomic == false, "shared_atomic == true NOT implemented");
      static_assert(parity_flip == false, "parity_flip == true NOT implemented");
      Gamma<Float, QUDA_DEGRAND_ROSSI_GAMMA_BASIS, dim> gamma;
      int parity, x_cb, parity_coarse, x_coarse_cb, c_col, c_row;
      getIndices<shared_atomic, parity_flip>(arg, parity, x_cb, parity_coarse, x_coarse_cb, c_row, c_col);

      if (parity > 1) return;
      if (c_row >= arg.vuvTile.M_tiles) return;
      if (c_col >= arg.vuvTile.N_tiles) return;
      if (!shared_atomic && x_cb >= arg.fineVolumeCB) return;

      impl::computeVUV<shared_atomic, parity_flip, from_coarse, Float, dim, dir, fineSpin, coarseSpin>(
        arg, gamma, parity, x_cb, c_row * arg.vuvTile.M, c_col * arg.vuvTile.N, parity_coarse, x_coarse_cb);
    }

  } // namespace mma

} // namespace quda
