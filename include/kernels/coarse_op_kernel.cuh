#pragma once

#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <clover_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <gamma.cuh>
#include <linalg.cuh>
#include <matrix_tile.cuh>
#include <target_device.h>
#include <kernel.h>

namespace quda {

  // this is the storage type used when computing the coarse link variables
  // by using integers we have deterministic atomics
  typedef int storeType;

  template <bool from_coarse_, typename Float_, int fineSpin_, int coarseSpin_, int fineColor_, int coarseColor_, typename coarseGauge,
            typename coarseGaugeAtomic, typename fineGauge, typename fineSpinor, typename fineSpinorTmp,
            typename fineSpinorV, typename fineClover>
  struct CalculateYArg : kernel_param<> {
    using Float = Float_;

    static constexpr int fineSpin = fineSpin_;
    static constexpr int coarseSpin = coarseSpin_;

    static constexpr int fineColor = fineColor_;
    static constexpr int coarseColor = coarseColor_;

    static constexpr int fineDof = fineSpin * fineColor;
    static constexpr int coarseDof = coarseSpin * coarseColor;

    static constexpr bool from_coarse = from_coarse_;
    static constexpr bool is_mma_compatible = coarseGauge::is_mma_compatible;

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */

    coarseGaugeAtomic Y_atomic;    /** Y atomic accessor used for computation before conversion to final format */
    coarseGaugeAtomic X_atomic;    /** X atomic accessor used for computation before conversion to final format */

    fineSpinorTmp UV;        /** Temporary that stores the fine-link * spinor field product */
    fineSpinor AV;           /** Temporary that stores the clover * spinor field product */

    const fineGauge U;       /** Fine grid link field */
    const fineSpinorV V;     /** Fine grid spinor field */
    const fineClover C;      /** Fine grid clover field, or Xinv for coarsening the optimized KD op */
    const fineClover Cinv;   /** Fine grid clover field, or Xinv for coarsening the optimize KD op */

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int_fastdiv geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */
    const int spin_bs;          /** Spin block size */
    const spin_mapper<fineSpin,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

    Float kappa;                /** kappa value */
    Float mass;                 /** mass value */
    Float mu;                   /** mu value */
    Float mu_factor;            /** multiplicative factor for mu applied when mu is added to the operator */
    Float rescale;              /** rescaling factor used when rescaling the Y links if the maximum increases */

    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    const int *fine_to_coarse;
    const int *coarse_to_fine;

    const bool bidirectional;

    // To increase L2 locality we can schedule the geometry to grid.y and
    // the coarse colors to grid.x.  This will increase the potential for
    // L2 reuse since a given wave of thread blocks will be for different
    // coarse color but the same coarse grid point which will have common
    // loads.
    bool coarse_color_wave = false;

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

    static constexpr bool compute_max = false;
    Float *max_h; // scalar that stores the maximum element on the host
    Float *max_d; // scalar that stores the maximum elenent on the device
    Float *max;   // points to either max_h or max_d, for host or device, respectively

    int dim;           // which dimension are we working on
    QudaDirection dir; // which direction are working on
    int dim_index;     // which direction / dimension we are working on

    bool twist; // whether we are doing twisted or non-twisted

    // tile used for computeUV
    static constexpr int tile_height_uv = fineColor % 4 == 0 ? 4 : fineColor % 3 == 0 ? 3 : fineColor % 2 ? 2 : 1;
    static constexpr int tile_width_uv = coarseColor % 2 == 0 ? 2 : 1;

    using uvTileType = TileSize<fineColor, coarseColor, fineColor, tile_height_uv, tile_width_uv, 1>;
    uvTileType uvTile;

    // tile used for computeVUV - for fine grids best to use 4, else use max of 3
    static constexpr int tile_height_vuv = (coarseColor % 4 == 0 && fineSpin == 4) ? 4 : coarseColor % 3 == 0 ? 3 : 2;
    static constexpr int tile_width_vuv = coarseColor % 2 == 0 ? 2 : 1;
    using vuvTileType = TileSize<coarseColor, coarseColor, fineColor, tile_height_vuv, tile_width_vuv, 1>;
    vuvTileType vuvTile;

    // max colors per block is 8, rounded up to whole multiples of tile size
    static constexpr int max_color_height_per_block = coarseColor < 8 ? coarseColor : ((8 + tile_height_vuv - 1) / tile_height_vuv) * tile_height_vuv;
    static constexpr int max_color_width_per_block = coarseColor < 8 ? coarseColor : ((8 + tile_width_vuv - 1) / tile_width_vuv) * tile_width_vuv;
    static constexpr int max_height_tiles_per_block = max_color_height_per_block / tile_height_vuv;
    static constexpr int max_width_tiles_per_block = max_color_width_per_block / tile_width_vuv;
    static_assert(max_color_height_per_block % tile_height_vuv == 0, "max_color_height_per_block must be divisible by tile height");
    static_assert(max_color_width_per_block % tile_width_vuv == 0, "max_color_width_per_block must be divisible by tile width");

    CalculateYArg(coarseGauge &Y, coarseGauge &X,
      coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
      fineSpinorTmp &UV, fineSpinor &AV, const fineGauge &U, const fineSpinorV &V,
      const fineClover &C, const fineClover &Cinv, double kappa, double mass, double mu, double mu_factor,
      const int *x_size_, const int *xc_size_, int *geo_bs_, int spin_bs_,
      const int *fine_to_coarse, const int *coarse_to_fine, bool bidirectional)
      : Y(Y), X(X), Y_atomic(Y_atomic), X_atomic(X_atomic),
      UV(UV), AV(AV), U(U), V(V), C(C), Cinv(Cinv), spin_bs(spin_bs_), spin_map(),
      kappa(static_cast<Float>(kappa)), mass(static_cast<Float>(mass)), mu(static_cast<Float>(mu)), mu_factor(static_cast<Float>(mu_factor)),
      fineVolumeCB(V.VolumeCB()), coarseVolumeCB(X.VolumeCB()),
      fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
      bidirectional(bidirectional), shared_atomic(false), parity_flip(false),
        aggregates_per_block(1), max_h(nullptr), max_d(nullptr), max(nullptr)
    {
      if (V.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
        errorQuda("Gamma basis %d not supported", V.GammaBasis());

      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
        geo_bs[i] = geo_bs_[i];
        comm_dim[i] = comm_dim_partitioned(i);
      }
    }
  };

  template <typename T>
  struct ArgMax : public T {
    static constexpr bool compute_max = true;
    ArgMax(const T& t) : T(t) { }
  };

  template <typename Arg> constexpr bool isHalo(const int coord[], int dim, int nFace, const Arg &arg)
  {
    switch (dim) {
    case 0: return arg.comm_dim[0] && coord[0] + nFace >= arg.x_size[0];
    case 1: return arg.comm_dim[1] && coord[1] + nFace >= arg.x_size[1];
    case 2: return arg.comm_dim[2] && coord[2] + nFace >= arg.x_size[2];
    case 3: return arg.comm_dim[3] && coord[3] + nFace >= arg.x_size[3];
    }
    return false;
  }

  template <typename I, typename Coord>
  __device__ __host__ inline auto linkIndexHop(const Coord &x, const I X[4], const int mu, int nFace)
  {
    int y[4];
#pragma unroll
    for ( int i = 0; i < 4; i++ ) y[i] = x[i];
    switch (mu) {
    case 0: y[0] = (y[0] + nFace + X[0]) % X[0]; break;
    case 1: y[1] = (y[1] + nFace + X[1]) % X[1]; break;
    case 2: y[2] = (y[2] + nFace + X[2]) % X[2]; break;
    case 3: y[3] = (y[3] + nFace + X[3]) % X[3]; break;
    }
    return (((y[3] * X[2] + y[2]) * X[1] + y[1]) * X[0] + y[0]) >> 1;
  }

  template <typename Arg> constexpr bool isCoarseDiagonal(const int coord[], const int coord_coarse[], int dim, const Arg &arg)
  {
    switch (dim) {
    case 0: return ((coord[0] + 1) % arg.x_size[0]) / arg.geo_bs[0] == coord_coarse[0];
    case 1: return ((coord[1] + 1) % arg.x_size[1]) / arg.geo_bs[1] == coord_coarse[1];
    case 2: return ((coord[2] + 1) % arg.x_size[2]) / arg.geo_bs[2] == coord_coarse[2];
    case 3: return ((coord[3] + 1) % arg.x_size[3]) / arg.geo_bs[3] == coord_coarse[3];
    }
    return false;
  }

  /**
     Calculates the matrix UV^{s,c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{s,c}_mu(x+mu)
     Where: mu = dir, s = fine spin, c' = coarse color, c = fine color
     or, if dir == QUDA_IN_PLACE, UV^{s,c'}(x) = \sum_c C^{c}_mu(x) * V^{s,c}_mu(x+mu)
  */
  template <typename Wtype, typename Arg>
  __device__ __host__ inline auto computeUV(const Arg &arg, const Wtype &Wacc, int parity, int x_cb, int i0, int j0)
  {
    constexpr int uvSpin = Arg::fineSpin * (Arg::from_coarse ? 2 : 1);
    constexpr int nFace = 1; // to do: nFace == 3 version for long links

    using real = typename Arg::Float;
    using complex = complex<real>;
    using TileType = typename Arg::uvTileType;
    auto &tile = arg.uvTile;
    auto dim = arg.dim;
    using Ctype = decltype(make_tile_C<complex, false>(tile));
    Ctype UV[uvSpin];

    int coord[4];
    getCoords(coord, x_cb, arg.x_size, parity);

    if (arg.dir == QUDA_IN_PLACE) {

#pragma unroll
      for (int k = 0; k < TileType::k; k += TileType::K) { // Fine Color columns of coarse clover field
#pragma unroll
        for (int s_col = 0; s_col < Arg::fineSpin; s_col++) {
          auto W = make_tile_B<complex, false>(tile);
          W.loadCS(Wacc, 0, 0, parity, x_cb, s_col, k, j0);
#pragma unroll
          for (int s = 0; s < Arg::fineSpin; s++) {  //Fine Spin
            auto C = make_tile_A<complex, false>(tile);
            C.load(arg.C, 0, parity, x_cb, s, s_col, i0, k);
            UV[s_col * Arg::fineSpin + s].mma_nn(C, W);
          } // which chiral block
        }  //Fine Spin
      }    // Fine color columns

    } else if ( isHalo(coord, dim, nFace, arg) ) {

      int ghost_idx = ghostFaceIndex<1>(coord, arg.x_size, dim, nFace);
      if (!Arg::from_coarse) {

#pragma unroll
        for (int k = 0; k < TileType::k; k += TileType::K) { // Fine Color columns of gauge field
          auto U = make_tile_A<complex, false>(tile);
          U.load(arg.U, dim, parity, x_cb, i0, k);
#pragma unroll
          for (int s = 0; s < Arg::fineSpin; s++) {  //Fine Spin
            auto W = make_tile_B<complex, true>(tile);
            W.loadCS(Wacc, dim, 1, (parity+1)&1, ghost_idx, s, k, j0);
            UV[s].mma_nn(U, W);
          } // Fine color columns
        }   // Fine spin (tensor)

      } else {

#pragma unroll
        for (int k = 0; k < TileType::k; k += TileType::K) { // Fine Color columns of gauge field
#pragma unroll
          for (int s_col=0; s_col<Arg::fineSpin; s_col++) {
            auto W = make_tile_B<complex, true>(tile);
            W.loadCS(Wacc, dim, 1, (parity+1)&1, ghost_idx, s_col, k, j0);
#pragma unroll
            for (int s = 0; s < Arg::fineSpin; s++) {  //Fine Spin
              // on coarse lattice, if forwards then use forwards links
              auto U = make_tile_A<complex, false>(tile);
              U.load(arg.U, dim + (arg.dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, i0, k);

              UV[s_col * Arg::fineSpin + s].mma_nn(U, W);
            } // which chiral block
          }  //Fine color columns
        }    // Fine Spin

      } // Arg::from_coarse

    } else {

      int y_cb = linkIndexHop(coord, arg.x_size, dim, nFace);
      if (!Arg::from_coarse) {

#pragma unroll
        for (int k = 0; k < TileType::k; k += TileType::K) { // Fine Color columns of gauge field
          auto U = make_tile_A<complex, false>(tile);
          U.load(arg.U, dim, parity, x_cb, i0, k);
#pragma unroll
          for (int s = 0; s < Arg::fineSpin; s++) {  //Fine Spin
            auto W = make_tile_B<complex, false>(tile);
            W.loadCS(Wacc, 0, 0, (parity+1)&1, y_cb, s, k, j0);
            UV[s].mma_nn(U, W);
          }  //Fine color columns
        }    // Fine Spin

      } else {

#pragma unroll
        for (int k = 0; k < TileType::k; k += TileType::K) { // Fine Color columns of gauge field
#pragma unroll
          for (int s_col = 0; s_col < Arg::fineSpin; s_col++) {
            auto W = make_tile_B<complex, false>(tile);
            W.loadCS(Wacc, 0, 0, (parity+1)&1, y_cb, s_col, k, j0);
#pragma unroll
            for (int s = 0; s < Arg::fineSpin; s++) {  //Fine Spin
              // on coarse lattice, if forwards then use forwards links
              auto U = make_tile_A<complex, false>(tile);
              U.load(arg.U, dim + (arg.dir == QUDA_FORWARDS ? 4 : 0), parity, x_cb, s, s_col, i0, k);

              UV[s_col * Arg::fineSpin + s].mma_nn(U, W);
            } // which chiral block
          }  //Fine Spin
        }    // Fine color columns
      }
    }

    real uv_max = static_cast<real>(0.0);
#pragma unroll
    for (int s = 0; s < uvSpin; s++) {
      if (Arg::compute_max) {
        uv_max = fmax(UV[s].abs_max(), uv_max);
      } else {
        UV[s].saveCS(arg.UV, 0, 0, parity, x_cb, s, i0, j0);
      }
    }

    return uv_max;
  } // computeUV

  template <typename Arg> struct compute_uv {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr compute_uv(const Arg &arg) : arg(arg) { }

    __device__ __host__ void operator()(int x_cb, int ic_parity, int jc)
    {
      int ic = ic_parity % arg.uvTile.M_tiles;
      int parity = ic_parity / arg.uvTile.M_tiles;

      typename Arg::Float max;
      if (arg.dir == QUDA_FORWARDS || arg.dir == QUDA_IN_PLACE) // only for preconditioned clover is V != AV, will need extra logic for staggered KD
        max = computeUV(arg, arg.V, parity, x_cb, ic * arg.uvTile.M, jc * arg.uvTile.N);
      else
        max = computeUV(arg, arg.AV, parity, x_cb, ic * arg.uvTile.M, jc * arg.uvTile.N);

      if (Arg::compute_max) atomic_fetch_abs_max(arg.max, max);
    }
  };

  /**
     Calculates the matrix A V^{s,c'}(x) = \sum_c A^{c}(x) * V^{s,c}(x)
     Where: s = fine spin, c' = coarse color, c = fine color
  */
  template <typename Arg> struct compute_av {
    using Float = typename Arg::Float;
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr compute_av(const Arg &arg) : arg(arg) { }

    __device__ __host__ inline void operator()(int x_cb, int ch_parity, int ic_c)
    {
      int ch = ch_parity % 2;
      int parity = ch_parity / 2;

      constexpr int N = Arg::fineSpin * Arg::fineColor / 2;
      HMatrix<Float, N> A;

#pragma unroll
      for (int i = 0; i < N; i++) {
        int s_i = 2 * ch + i / Arg::fineColor;
        int c_i = i % Arg::fineColor;
#pragma unroll
        for (int j = 0; j <= i; j++) {
          int s_j = 2 * ch + j / Arg::fineColor;
          int c_j = j % Arg::fineColor;
#ifndef DYNAMIC_CLOVER
          A(i, j) = arg.Cinv(0, parity, x_cb, s_i, s_j, c_i, c_j);
#else
          A(i, j) = arg.C(0, parity, x_cb, s_i, s_j, c_i, c_j);
#endif
        }
      }

      ColorSpinor<Float, Arg::fineColor, Arg::fineSpin / 2> V;
      for (int s = 0; s < Arg::fineSpin / 2; s++) {
        for (int c = 0; c < Arg::fineColor; c++) { V(s, c) = arg.V(parity, x_cb, 2 * ch + s, c, ic_c); }
      }

#ifndef DYNAMIC_CLOVER
      auto AV = A * V;
#else
      // solve for the matrix
      linalg::Cholesky<HMatrix, Float, N> cholesky(A);
      auto AV = cholesky.backward(cholesky.forward(V));
#endif

      if (!Arg::compute_max) {
#pragma unroll
        for (int s = 0; s < Arg::fineSpin / 2; s++) {
#pragma unroll
          for (int ic = 0; ic < Arg::fineColor; ic++) { arg.AV(parity, x_cb, 2 * ch + s, ic, ic_c) = AV(s, ic); }
        }
      } else {
        Float max = static_cast<Float>(0.0);
#pragma unroll
        for (int s = 0; s < Arg::fineSpin / 2; s++) {
#pragma unroll
          for (int ic = 0; ic < Arg::fineColor; ic++) {
            auto abs_max = fmax(abs(AV(s, ic).real()), abs(AV(s, ic).imag()));
            max = fmax(abs_max, max);
          }
        }
        atomic_fetch_abs_max(arg.max, max);
      }
    }
  };

  /**
     Calculates the matrix A V^{s,c'}(x) = \sum_c A^{c}(x) * V^{s,c}(x) for twisted-mass fermions
     Where: s = fine spin, c' = coarse color, c = fine color
  */
  template <typename Arg> struct compute_tmav {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr compute_tmav(const Arg &arg) : arg(arg) { }

    __device__ __host__ inline void operator()(int x_cb, int parity, int v)
    {
      complex<typename Arg::Float> fp(1./(1.+arg.mu*arg.mu),-arg.mu/(1.+arg.mu*arg.mu));
      complex<typename Arg::Float> fm(1./(1.+arg.mu*arg.mu),+arg.mu/(1.+arg.mu*arg.mu));

#pragma unroll
      for (int s = 0; s < Arg::fineSpin/2; s++) {
#pragma unroll
        for (int c = 0; c < Arg::fineColor; c++) {
          arg.AV(parity,x_cb,s,c,v) = arg.V(parity,x_cb,s,c,v) * fp;
        }
      }

#pragma unroll
      for (int s = Arg::fineSpin/2; s < Arg::fineSpin; s++) {
#pragma unroll
        for (int c = 0; c < Arg::fineColor; c++) {
          arg.AV(parity,x_cb,s,c,v) = arg.V(parity,x_cb,s,c,v) * fm;
        }
      }

    }
  };

  /**
     Calculates the matrix A V^{s,c'}(x) = \sum_c A^{c}(x) * V^{s,c}(x) for twisted-clover fermions
     Where: s = fine spin, c' = coarse color, c = fine color
  */
  template <typename Arg> struct compute_tmcav {
    using Float = typename Arg::Float;
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr compute_tmcav(const Arg &arg) : arg(arg) { }

    __device__ __host__ inline void operator()(int x_cb, int ch_parity, int ic_c)
    {
      int ch = ch_parity % 2;
      int parity = ch_parity / 2;
      constexpr int N = Arg::fineSpin * Arg::fineColor / 2;
      HMatrix<Float, N> A;

#pragma unroll
      for (int i = 0; i < N; i++) {
        int s_i = 2 * ch + i / Arg::fineColor;
        int c_i = i % Arg::fineColor;
#pragma unroll
        for (int j = 0; j <= i; j++) {
          int s_j = 2 * ch + j / Arg::fineColor;
          int c_j = j % Arg::fineColor;
          A(i, j) = arg.C(0, parity, x_cb, s_i, s_j, c_i, c_j);
        }
      }

      complex<Float> mu(0., arg.mu);
      if (ch == 0) mu *= static_cast<Float>(-1.0);

      ColorSpinor<Float, Arg::fineColor, Arg::fineSpin / 2> V;

#pragma unroll
      for (int s = 0; s < Arg::fineSpin / 2; s++) {
#pragma unroll
        for (int c = 0; c < Arg::fineColor; c++) {
          V(s, c) = arg.V(parity, x_cb, 2 * ch + s, c, ic_c);
        }
      }

      // first apply the clover matrix directly, including mu
      auto UV = A * V;
      UV += mu * V;

      // Then we calculate AV = Cinv UV, so  [AV = (C^2 + mu^2)^{-1} (Clover -/+ i mu)·Vector]
      // for in twisted-clover fermions, Cinv keeps (C^2 + mu^2)^{-1}

      if (!dynamic_clover_inverse()) {
        // load in the clover inverse matrix
        HMatrix<Float, N> Ainv;
#pragma unroll
        for (int i = 0; i < N; i++) {
          int s_i = 2 * ch + i / Arg::fineColor;
          int c_i = i % Arg::fineColor;
#pragma unroll
          for (int j = 0; j <= i; j++) {
            int s_j = 2 * ch + j / Arg::fineColor;
            int c_j = j % Arg::fineColor;
            Ainv(i, j) = arg.Cinv(0, parity, x_cb, s_i, s_j, c_i, c_j);
          }
        }
        auto AV = Ainv * UV;

        if (!Arg::compute_max) {
#pragma unroll
          for (int s = 0; s < Arg::fineSpin / 2; s++)
#pragma unroll
            for (int c = 0; c < Arg::fineColor; c++)
              arg.AV(parity, x_cb, 2 * ch + s, c, ic_c) = AV(s, c);
        } else {
          Float max = static_cast<Float>(0.0);
#pragma unroll
          for (int s = 0; s < Arg::fineSpin / 2; s++) {
#pragma unroll
            for (int c = 0; c < Arg::fineColor; c++) {
              auto abs_max = fmax(abs(AV(s, c).real()), abs(AV(s, c).imag()));
              max = fmax(abs_max, max);
            }
          }
          atomic_fetch_abs_max(arg.max, max);
        }
      } else {
        // compute the clover inverse matrix with the already loaded clover matrix
        A = A.square();
        A += arg.mu * arg.mu;

        linalg::Cholesky<HMatrix, Float, N> cholesky(A);
        const auto AV = cholesky.backward(cholesky.forward(UV));

        if (!Arg::compute_max) {
#pragma unroll
          for (int s = 0; s < Arg::fineSpin / 2; s++)
#pragma unroll
            for (int c = 0; c < Arg::fineColor; c++)
              arg.AV(parity, x_cb, 2 * ch + s, c, ic_c) = AV(s, c);
        } else {
          Float max = static_cast<Float>(0.0);
#pragma unroll
          for (int s = 0; s < Arg::fineSpin / 2; s++) {
#pragma unroll
            for (int c = 0; c < Arg::fineColor; c++) {
              auto abs_max = fmax(abs(AV(s, c).real()), abs(AV(s, c).imag()));
              max = fmax(abs_max, max);
            }
          }
          atomic_fetch_abs_max(arg.max, max);
        }
      }
    }
  };

  template <typename Arg>
  __device__ __host__ inline int virtualThreadIdx(const Arg &arg) {
    QUDA_RT_CONSTS;
    int warp_id = threadIdx.x / device::warp_size();
    int warp_lane = threadIdx.x % device::warp_size();
    int tx = warp_id * (device::warp_size() / arg.aggregates_per_block) + warp_lane / arg.aggregates_per_block;
    return tx;
  }

  template <typename Arg>
  __device__ __host__ inline int virtualBlockDim(const Arg &arg) {
    QUDA_RT_CONSTS;
    int block_dim_x = blockDim.x / arg.aggregates_per_block;
    return block_dim_x;
  }

  template <typename Arg>
  __device__ __host__ inline int coarseIndex(const Arg &arg) {
    QUDA_RT_CONSTS;
    int warp_lane = threadIdx.x % device::warp_size();
    int x_coarse = (arg.coarse_color_wave ? blockIdx.y : blockIdx.x) * arg.aggregates_per_block + warp_lane % arg.aggregates_per_block;
    return x_coarse;
  }

  template <int dim, typename Out, typename Arg>
  __device__ __host__ inline void multiplyVUV(Out &vuv, const Arg &arg, int parity, int x_cb, int i0, int j0)
  {
    using complex = complex<typename Arg::Float>;
    using TileType = typename Arg::vuvTileType;
    auto &tile = arg.vuvTile;

#pragma unroll
    for (int s = 0; s < Arg::fineSpin; s++) { // Loop over fine spin

      //Spin part of the color matrix.  Will always consist
      //of two terms - diagonal and off-diagonal part of
      //P_mu = (1+/-\gamma_mu)

      const int s_c_row = arg.spin_map(s, parity); // Coarse spin row index

      // Use Gamma to calculate off-diagonal coupling and
      // column index.  Diagonal coupling is always 1.
      // If computing the backwards (forwards) direction link then
      // we desire the positive (negative) projector
      Gamma<typename Arg::Float, QUDA_DEGRAND_ROSSI_GAMMA_BASIS, dim> gamma;

      const int s_col = gamma.getcol(s);
      const int s_c_col = 1 - s_c_row; // always off-diagonal relative to row coord

#pragma unroll
      for (int k = 0; k < TileType::k; k += TileType::K) { // Sum over fine color

        if (arg.dir == QUDA_BACKWARDS) {
          auto V = make_tile_At<complex, false>(tile);
          V.loadCS(arg.V, 0, 0, parity, x_cb, s, k, i0);

          // here UV is really UAV
          //Diagonal Spin
          auto UV = make_tile_B<complex, false>(tile);
          UV.loadCS(arg.UV, 0, 0, parity, x_cb, s, k, j0);
          vuv[s_c_row*Arg::coarseSpin+s_c_row].mma_tn(V, UV);

          //Off-diagonal Spin (backward link / positive projector applied)
          auto gammaV = make_tile_A<complex, false>(tile);
#pragma unroll
          for (int i = 0; i < TileType::K; i++)
#pragma unroll
            for (int j = 0; j < TileType::M; j++)
              gammaV(j, i) = gamma.apply(s, conj(V(i, j)));

          UV.loadCS(arg.UV, 0, 0, parity, x_cb, s_col, k, j0);
          vuv[s_c_row*Arg::coarseSpin+s_c_col].mma_nn(gammaV, UV);
        } else {
          auto AV = make_tile_At<complex, false>(tile);
          AV.loadCS(arg.AV, 0, 0, parity, x_cb, s, k, i0);

          auto UV = make_tile_B<complex, false>(tile);
          UV.loadCS(arg.UV, 0, 0, parity, x_cb, s, k, j0);

          //Diagonal Spin
          vuv[s_c_row*Arg::coarseSpin+s_c_row].mma_tn(AV, UV);

          //Off-diagonal Spin (forward link / negative projector applied)
          auto gammaAV = make_tile_A<complex, false>(tile);

#pragma unroll
          for (int i = 0; i < TileType::K; i++)
#pragma unroll
            for (int j = 0; j < TileType::M; j++)
              gammaAV(j, i) = -gamma.apply(s, conj(AV(i, j)));

          UV.loadCS(arg.UV, 0, 0, parity, x_cb, s_col, k, j0);
          vuv[s_c_row*Arg::coarseSpin+s_c_col].mma_nn(gammaAV, UV);
        }
      }
    }
  }

  /**
     @brief Do a single (AV)^\dagger * UV product, where for
     preconditioned clover, AV correspond to the clover inverse
     multiplied by the packed null space vectors, else AV is simply
     the packed null space vectors. This is the specialized form for
     fine-grid Wilson-type fermions, where we template on the
     dimension to ensure the spin projector structure is known to the
     compiler.

     @param[out] vuv Result array
     @param[in,out] arg Arg storing the fields and parameters
     @param[in] Fine grid parity we're working on
     @param[in] x_cb Checkboarded x dimension
   */
  template <typename Arg, typename Out>
  __device__ __host__ inline std::enable_if_t<!Arg::from_coarse && Arg::fineSpin == 4, void>
  multiplyVUV(Out &vuv, const Arg &arg, int parity, int x_cb, int i0, int j0)
  {
    switch (arg.dim) {
    case 0: multiplyVUV<0>(vuv, arg, parity, x_cb, i0, j0); break;
    case 1: multiplyVUV<1>(vuv, arg, parity, x_cb, i0, j0); break;
    case 2: multiplyVUV<2>(vuv, arg, parity, x_cb, i0, j0); break;
    case 3: multiplyVUV<3>(vuv, arg, parity, x_cb, i0, j0); break;
    }
  }

  /**
     @brief Do a single (AV)^\dagger * UV product, where for
     preconditioned clover, AV correspond to the clover inverse
     multiplied by the packed null space vectors, else AV is simply
     the packed null space vectors.  This is the specialization form
     for fine-grid staggered/asqtad fermions.

     @param[out] vuv Result array
     @param[in,out] arg Arg storing the fields and parameters
     @param[in] Fine grid parity we're working on
     @param[in] x_cb Checkboarded x dimension
   */
  template <typename Arg, typename Out>
  __device__ __host__ inline std::enable_if_t<!Arg::from_coarse && Arg::fineSpin == 1, void>
  multiplyVUV(Out &vuv, const Arg &arg, int parity, int x_cb, int i0, int j0)
  {
    using complex = complex<typename Arg::Float>;
    using TileType = typename Arg::vuvTileType;
    auto &tile = arg.vuvTile;

    // the KD op will even to even, odd to odd terms
    const int s_c_row = arg.spin_map(0, parity);

    // the column is the opposite contribution
    const int s_c_col = arg.spin_map(0, 1-parity);

    for (int k = 0; k < TileType::k; k += TileType::K) { // Sum over fine color

      if (arg.dir == QUDA_BACKWARDS) {
        auto V = make_tile_At<complex, false>(tile);
        V.loadCS(arg.V, 0, 0, parity, x_cb, 0, k, i0);

        // here UV is really UAV
        auto UV = make_tile_B<complex, false>(tile);
        UV.loadCS(arg.UV, 0, 0, parity, x_cb, 0, k, j0);

        vuv[s_c_row*Arg::coarseSpin+s_c_col].mma_tn(V, UV);
      } else {
        auto AV = make_tile_At<complex, false>(tile);
        AV.loadCS(arg.AV, 0, 0, parity, x_cb, 0, k, i0);
        AV *= static_cast<typename Arg::Float>(-1.);

        auto UV = make_tile_B<complex, false>(tile);
        UV.loadCS(arg.UV, 0, 0, parity, x_cb, 0, k, j0);

        vuv[s_c_row*Arg::coarseSpin+s_c_col].mma_tn(AV, UV);
      }
    }
  }

  /**
     @brief Do a single (AV)^\dagger * UV product, where for
     preconditioned clover, AV correspond to the clover inverse
     multiplied by the packed null space vectors, else AV is simply
     the packed null space vectors.  This is the specialization form
     for when the fine-grid operator is itself a coarse-grid operator.

     @param[out] vuv Result array
     @param[in,out] arg Arg storing the fields and parameters
     @param[in] Fine grid parity we're working on
     @param[in] x_cb Checkboarded x dimension
   */
  template <typename Arg, typename Out>
  __device__ __host__ inline std::enable_if_t<Arg::from_coarse, void>
  multiplyVUV(Out &vuv, const Arg &arg, int parity, int x_cb, int i0, int j0)
  {
    using complex = complex<typename Arg::Float>;
    using TileType = typename Arg::vuvTileType;
    auto &tile = arg.vuvTile;

#pragma unroll
    for (int k = 0; k < TileType::k; k += TileType::K) { // Sum over fine color
#pragma unroll
      for (int s = 0; s < Arg::fineSpin; s++) {
        auto AV = make_tile_At<complex, false>(tile);
        AV.loadCS(arg.AV, 0, 0, parity, x_cb, s, k, i0);
#pragma unroll
        for (int s_col = 0; s_col < Arg::fineSpin; s_col++) { // which chiral block
          auto UV = make_tile_B<complex, false>(tile);
          UV.loadCS(arg.UV, 0, 0, parity, x_cb, s_col*Arg::fineSpin+s, k, j0);
          vuv[s*Arg::coarseSpin+s_col].mma_tn(AV, UV);
        } //Fine color
      } //Fine spin
    }
  }

  template <typename Float, typename storeType, typename Accessor>
  inline __host__ __device__ void atomic_helper(complex<storeType> *Y, const Accessor &A, const complex<Float> &vuv)
  {
    if (gauge::fixed_point<Float,storeType>()) {
      Float scale = A.accessor.scale;
      complex<storeType> a(round(scale * vuv.real()), round(scale * vuv.imag()));
      atomic_fetch_add(Y, a);
    } else {
      atomic_fetch_add(Y, reinterpret_cast<const complex<storeType>&>(vuv));
    }
  }

  template <QudaDirection dir_>
  struct Pack {
    static constexpr QudaDirection dir = dir_;
  };

  template <bool is_device> struct storeCoarseSharedAtomic_impl {
    template <typename ...Args> void operator()(Args...)
    {
      errorQuda("Shared-memory atomic aggregation not supported on host");
    }
  };

  template <> struct storeCoarseSharedAtomic_impl<true> {
    template <typename VUV, typename Pack, typename Arg>
    inline __device__ void operator()(VUV &vuv, bool isDiagonal, int coarse_x_cb, int coarse_parity, int i0, int j0, int parity, const Pack &pack, const Arg &arg)
    {
      QUDA_RT_CONSTS;
      using Float = typename Arg::Float;
      using TileType = typename Arg::vuvTileType;
      const int dim_index = arg.dim_index % arg.Y_atomic.geometry;
      __shared__ complex<storeType> X[Arg::max_color_height_per_block][Arg::max_color_width_per_block][4][Arg::coarseSpin][Arg::coarseSpin];
      __shared__ complex<storeType> Y[Arg::max_color_height_per_block][Arg::max_color_width_per_block][4][Arg::coarseSpin][Arg::coarseSpin];

      int x_ = coarse_x_cb % arg.aggregates_per_block;
      int tx = virtualThreadIdx(arg);
      int s_col = tx / Arg::coarseSpin;
      int s_row = tx % Arg::coarseSpin;

      // this relies on the indexing as used in getIndices
      int i_block0 = (threadIdx.y / (arg.parity_flip ? 1 : 2)) * TileType::M;
      int j_block0 = threadIdx.z * TileType::N;

#pragma unroll
      for (int i = 0; i < TileType::M; i++) {
#pragma unroll
        for (int j = 0; j < TileType::N; j++) {
          if (tx < Arg::coarseSpin*Arg::coarseSpin) {
            if (pack.dir != QUDA_IN_PLACE) Y[i_block0+i][j_block0+j][x_][s_row][s_col] = 0;
            X[i_block0+i][j_block0+j][x_][s_row][s_col] = 0;
          }
        }
      }

      __syncthreads();

#pragma unroll
      for (int i = 0; i < TileType::M; i++) {
#pragma unroll
        for (int j = 0; j < TileType::N; j++) {

          if (pack.dir == QUDA_IN_PLACE || isDiagonal) {
#pragma unroll
            for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { // Chiral row block
#pragma unroll
              for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { // Chiral column block
                atomic_helper<Float, storeType>(&X[i_block0+i][j_block0+j][x_][s_row][s_col],
                                                arg.X_atomic, vuv[s_row*Arg::coarseSpin+s_col](i,j));
              }
            }
          } else {
#pragma unroll
            for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { // Chiral row block
#pragma unroll
              for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { // Chiral column block
                atomic_helper<Float, storeType>(&Y[i_block0+i][j_block0+j][x_][s_row][s_col],
                                                arg.Y_atomic, vuv[s_row*Arg::coarseSpin+s_col](i,j));
              }
            }
          }
        }
      }

      __syncthreads();

      if (tx < Arg::coarseSpin*Arg::coarseSpin && (parity == 0 || arg.parity_flip == 1) ) {

#pragma unroll
        for (int i = 0; i < TileType::M; i++) {
#pragma unroll
          for (int j = 0; j < TileType::N; j++) {
            if (pack.dir == QUDA_IN_PLACE) {
              // same as dir == QUDA_FORWARDS
              arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,
                                     X[i_block0+i][j_block0+j][x_][s_row][s_col]);
            } else {
              arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,
                                     Y[i_block0+i][j_block0+j][x_][s_row][s_col]);

              if (pack.dir == QUDA_BACKWARDS) {
                arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_col,s_row,j0+j,i0+i,
                                       conj(X[i_block0+i][j_block0+j][x_][s_row][s_col]));
              } else {
                arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,
                                       X[i_block0+i][j_block0+j][x_][s_row][s_col]);
              }

              if (!arg.bidirectional) {
                if (Arg::fineSpin != 1 && s_row == s_col) arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,
                                                                                 X[i_block0+i][j_block0+j][x_][s_row][s_col]);
                else arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,
                                            -X[i_block0+i][j_block0+j][x_][s_row][s_col]);
              }
            } // dir == QUDA_IN_PLACE
          }
        }
      }
    }
  };

  template <typename VUV, typename Arg>
  __device__ __host__ void storeCoarseSharedAtomic(VUV &vuv, bool isDiagonal, int coarse_x_cb, int coarse_parity, int i0, int j0, int parity, const Arg &arg)
  {
    switch (arg.dir) {
    case QUDA_BACKWARDS:
      target::dispatch<storeCoarseSharedAtomic_impl>(vuv, isDiagonal, coarse_x_cb, coarse_parity, i0, j0, parity, Pack<QUDA_BACKWARDS>(), arg); break;
    case QUDA_FORWARDS:
      target::dispatch<storeCoarseSharedAtomic_impl>(vuv, isDiagonal, coarse_x_cb, coarse_parity, i0, j0, parity, Pack<QUDA_FORWARDS>(), arg); break;
    case QUDA_IN_PLACE:
      target::dispatch<storeCoarseSharedAtomic_impl>(vuv, isDiagonal, coarse_x_cb, coarse_parity, i0, j0, parity, Pack<QUDA_IN_PLACE>(), arg); break;
    default:
      break;// do nothing
    }
  }

  template <typename VUV, typename Arg>
  inline __device__ __host__ void storeCoarseGlobalAtomic(VUV &vuv, bool isDiagonal, int coarse_x_cb, int coarse_parity, int i0, int j0, const Arg &arg)
  {
    using Float = typename Arg::Float;
    const int dim_index = arg.dim_index % arg.Y_atomic.geometry;
    using TileType = typename Arg::vuvTileType;

    if (arg.dir == QUDA_IN_PLACE) {
      // same as dir == QUDA_FORWARDS
#pragma unroll
      for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { // Chiral row block
#pragma unroll
        for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { // Chiral column block
#pragma unroll
          for (int i = 0; i < TileType::M; i++)
#pragma unroll
            for (int j = 0; j < TileType::N; j++)
              arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,vuv[s_row*Arg::coarseSpin+s_col](i,j));
        }
      }
    } else if (!isDiagonal) {
#pragma unroll
      for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { // Chiral row block
#pragma unroll
        for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { // Chiral column block
#pragma unroll
          for (int i = 0; i < TileType::M; i++)
#pragma unroll
            for (int j = 0; j < TileType::N; j++)
              arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,vuv[s_row*Arg::coarseSpin+s_col](i,j));
        }
      }
    } else {

      if (arg.dir == QUDA_BACKWARDS) {
#pragma unroll
        for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { // Chiral row block
#pragma unroll
          for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { // Chiral column block
#pragma unroll
            for (int i = 0; i < TileType::M; i++)
#pragma unroll
              for (int j = 0; j < TileType::N; j++)
                arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_col,s_row,j0+j,i0+i,conj(vuv[s_row*Arg::coarseSpin+s_col](i,j)));
          }
        }
      } else {
#pragma unroll
        for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { // Chiral row block
#pragma unroll
          for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { // Chiral column block
#pragma unroll
            for (int i = 0; i < TileType::M; i++)
#pragma unroll
              for (int j = 0; j < TileType::N; j++)
                arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,vuv[s_row*Arg::coarseSpin+s_col](i,j));
          }
        }
      }

      if (!arg.bidirectional) {
#pragma unroll
        for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { // Chiral row block
#pragma unroll
          for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { // Chiral column block
            if (s_row != s_col) vuv[s_row * Arg::coarseSpin + s_col] *= static_cast<Float>(-1.0);
            if (Arg::fineSpin != 1 || s_row != s_col) {
#pragma unroll
              for (int i = 0; i < TileType::M; i++)
#pragma unroll
                for (int j = 0; j < TileType::N; j++)
                  arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,i0+i,j0+j,vuv[s_row*Arg::coarseSpin+s_col](i,j));
            }
          }
        }
      }
    }

  }

  template <typename Arg>
  __device__ __host__ void computeVUV(const Arg &arg, int parity, int x_cb, int i0, int j0, int parity_coarse_, int coarse_x_cb_)
  {
    using Float = typename Arg::Float;
    constexpr int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    getCoords(coord, x_cb, arg.x_size, parity);
    for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    //Check to see if we are on the edge of a block.  If adjacent site
    //is in same block, M = X, else M = Y
    const bool isFromCoarseClover = Arg::fineSpin == 2 && arg.dir == QUDA_IN_PLACE;
    const bool isDiagonal = (isFromCoarseClover || isCoarseDiagonal(coord, coord_coarse, arg.dim, arg)) ? true : false;

    int coarse_parity = arg.shared_atomic ? parity_coarse_ : 0;
    if (!arg.shared_atomic) {
      for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
      coarse_parity &= 1;
      coord_coarse[0] /= 2;
    }
    int coarse_x_cb = arg.shared_atomic ? coarse_x_cb_ : ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];

    using Ctype = decltype(make_tile_C<complex<Float>, false>(arg.vuvTile));
    Ctype vuv[Arg::coarseSpin * Arg::coarseSpin];
    multiplyVUV(vuv, arg, parity, x_cb, i0, j0);

    if (isDiagonal && !isFromCoarseClover) {
#pragma unroll
      for (int s2=0; s2<Arg::coarseSpin*Arg::coarseSpin; s2++) vuv[s2] *= -arg.kappa;
    }

    if (arg.shared_atomic)
      storeCoarseSharedAtomic(vuv, isDiagonal, coarse_x_cb, coarse_parity, i0, j0, parity, arg);
    else
      storeCoarseGlobalAtomic(vuv, isDiagonal, coarse_x_cb, coarse_parity, i0, j0, arg);
  }

  template <bool is_device> struct getIndices {
    template <typename Arg> inline void operator()(int &parity_coarse, int &x_coarse_cb, int &parity, int &,
                                                   int &parity_c_row, int &c_row, int &, const Arg &arg)
    {
      c_row  = arg.parity_flip ? (parity_c_row % arg.vuvTile.M_tiles) : (parity_c_row / 2); // coarse color row index
      parity = arg.parity_flip ? (parity_c_row / arg.vuvTile.M_tiles) : (parity_c_row % 2); // coarse color row index
      parity_coarse = 0;
      x_coarse_cb = 0;
    }
  };

  template <> struct getIndices<true> {
    template <typename Arg> __device__ inline void operator()(int &parity_coarse, int &x_coarse_cb, int &parity, int &x_cb,
                                                              int &parity_c_row, int &c_row, int &c_col, const Arg &arg)
    {
      QUDA_RT_CONSTS;
      if (arg.coarse_color_wave) {
        int parity_c_row_block_idx_z = blockDim.y*blockIdx.x + threadIdx.y;
        int c_row_block_idx_z = arg.parity_flip ? (parity_c_row_block_idx_z % arg.coarse_color_grid_z ) : (parity_c_row_block_idx_z / 2); // coarse color row index
        parity = arg.parity_flip ? (parity_c_row_block_idx_z / arg.coarse_color_grid_z ) : (parity_c_row_block_idx_z % 2);
        c_row = c_row_block_idx_z % arg.vuvTile.M_tiles;
        int block_idx_z = c_row_block_idx_z / arg.vuvTile.M_tiles;
        c_col = blockDim.z*block_idx_z + threadIdx.z; // coarse color col index
      } else {
        c_row  = arg.parity_flip ? (parity_c_row % arg.vuvTile.M_tiles) : (parity_c_row / 2); // coarse color row index
        parity = arg.parity_flip ? (parity_c_row / arg.vuvTile.M_tiles) : (parity_c_row % 2); // coarse color row index
        c_col = blockDim.z*blockIdx.z + threadIdx.z; // coarse color col index
      }

      // if (parity > 1 && shared_atomic), you can go outside the bounds of arg.coarse_to_fine below
      if (parity > 1) return;

      if (!arg.shared_atomic) {
        x_cb = blockDim.x*(arg.coarse_color_wave ? blockIdx.y : blockIdx.x) + threadIdx.x;
        x_coarse_cb = 0;
        parity_coarse = 0;
      } else {
        int block_dim_x = virtualBlockDim(arg);
        int thread_idx_x = virtualThreadIdx(arg);
        int x_coarse = coarseIndex(arg);

        parity_coarse = x_coarse >= arg.coarseVolumeCB ? 1 : 0;
        x_coarse_cb = x_coarse - parity_coarse*arg.coarseVolumeCB;

        // obtain fine index from this look-up table
        // since both parities map to the same block, each thread block must do both parities

        // threadIdx.x - fine checkerboard offset
        // threadIdx.y - fine parity offset
        // blockIdx.x  - which coarse block are we working on
        // assume that coarse_to_fine look up map is ordered as (coarse-block-id + fine-point-id)
        // and that fine-point-id is parity ordered

        int x_fine = arg.coarse_to_fine[ (x_coarse*2 + parity) * block_dim_x + thread_idx_x];
        x_cb = x_fine - parity*arg.fineVolumeCB;
      }
    }
  };

    template <typename Arg> struct compute_vuv {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr compute_vuv(const Arg &arg) : arg(arg) { }

    __device__ __host__ inline void operator()(int x_cb, int parity_c_row, int c_col)
    {
      int parity, parity_coarse, x_coarse_cb, c_row;
      target::dispatch<getIndices>(parity_coarse, x_coarse_cb, parity, x_cb, parity_c_row, c_row, c_col, arg);

      if (parity > 1) return;
      if (c_row >= arg.vuvTile.M_tiles) return;
      if (c_col >= arg.vuvTile.N_tiles) return;
      if (!arg.shared_atomic && x_cb >= arg.fineVolumeCB) return;

      computeVUV(arg, parity, x_cb, c_row * arg.vuvTile.M, c_col * arg.vuvTile.N, parity_coarse, x_coarse_cb);
    }
  };

  template <typename Arg> struct compute_coarse_clover {
    static_assert(!Arg::from_coarse, "computeCoarseClover is only defined on the fine grid");
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr compute_coarse_clover(const Arg &arg) : arg(arg) { }

    __device__ __host__ inline void operator()(int x_cb, int parity_c_col, int c_row)
    {
      int c_col = parity_c_col % Arg::coarseColor; // coarse color col index
      int parity = parity_c_col / Arg::coarseColor;
      constexpr int nDim = 4;

      int coord[QUDA_MAX_DIM];
      int coord_coarse[QUDA_MAX_DIM];

      getCoords(coord, x_cb, arg.x_size, parity);
      for (int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

      int coarse_parity = 0;
      for (int d = 0; d < nDim; d++) coarse_parity += coord_coarse[d];
      coarse_parity &= 1;
      coord_coarse[0] /= 2;
      int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];

      coord[0] /= 2;

      complex<typename Arg::Float> X[Arg::coarseSpin*Arg::coarseSpin];
      for (int i = 0; i < Arg::coarseSpin * Arg::coarseSpin; i++) X[i] = 0.0;

      // If Nspin = 4, then the clover term has structure C_{\mu\nu} = \gamma_{\mu\nu}C^{\mu\nu}
#pragma unroll
      for (int s = 0; s < Arg::fineSpin; s++) { // Loop over fine spin row
        const int s_c = arg.spin_map(s,parity);
        // On the fine lattice, the clover field is chirally blocked, so loop over rows/columns
        // in the same chiral block.
#pragma unroll
        for (int s_col = s_c * arg.spin_bs; s_col < (s_c+1) * arg.spin_bs; s_col++) { // Loop over fine spin column
#pragma unroll
          for (int ic = 0; ic < Arg::fineColor; ic++) { // Sum over fine color row
            complex<typename Arg::Float> CV = 0.0;
#pragma unroll
            for (int jc = 0; jc < Arg::fineColor; jc++) {  // Sum over fine color column
              CV = cmac(arg.C(0, parity, x_cb, s, s_col, ic, jc), arg.V(parity, x_cb, s_col, jc, c_col), CV);
            } // Fine color column
            X[s_c*Arg::coarseSpin + s_c] = cmac(conj(arg.V(parity, x_cb, s, ic, c_row)), CV, X[s_c*Arg::coarseSpin + s_c]);
          }  // Fine color row
        }  // Fine spin column
      } // Fine spin

#pragma unroll
      for (int si = 0; si < Arg::coarseSpin; si++) {
#pragma unroll
        for (int sj = 0; sj < Arg::coarseSpin; sj++) {
          arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,si,sj, c_row, c_col, X[si*Arg::coarseSpin+sj]);
        }
      }
    }
  };

  /**
   * Wilson-type: Compute the forward links from backwards links by flipping the
   * sign of the spin projector
   * Staggered-type: there's no spin-diagonal term, only flip off-spin term
   */
  template <typename Arg> struct reverse {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr reverse(const Arg &arg) : arg(arg) { }

    __device__ __host__ void operator()(int x_cb, int parity_c_col, int c_row)
    {
      int parity = parity_c_col / Arg::coarseColor;
      int c_col = parity_c_col % Arg::coarseColor;

#pragma unroll
      for (int d=0; d<4; d++) {
#pragma unroll
        for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { //Spin row
#pragma unroll
          for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { //Spin column
            if (s_row == s_col && Arg::coarseSpin != 1)
              arg.Y(d+4, parity, x_cb, s_row, s_col, c_row, c_col) = arg.Y(d, parity, x_cb, s_row, s_col, c_row, c_col);
            else
              arg.Y(d+4, parity, x_cb, s_row, s_col, c_row, c_col) = -arg.Y(d, parity, x_cb, s_row, s_col, c_row, c_col);
          } //Spin column
        } //Spin row

      } // dimension
    }
  };

  /**
   * Adds the identity matrix to the coarse local term.
   */
  template <typename Arg> struct add_coarse_diagonal {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr add_coarse_diagonal(const Arg &arg) : arg(arg) { }

    __device__ __host__ void operator()(int x_cb, int parity, int c)
    {
      for (int s = 0; s < Arg::coarseSpin; s++) { //Spin
        arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<typename Arg::Float>(1.0, 0.0);
      } //Spin
    }
  };

  /**
   * Adds the staggered mass to the coarse local term.
   */
  template <typename Arg> struct add_coarse_staggered_mass {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr add_coarse_staggered_mass(const Arg &arg) : arg(arg) { }

    __device__ __host__ void operator()(int x_cb, int parity, int)
    {
      for (int s = 0; s < Arg::coarseSpin; s++) { //Spin
        for (int c = 0; c < Arg::coarseColor; c++) { //Color
          arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<typename Arg::Float>(static_cast<typename Arg::Float>(2) * arg.mass, 0.0);
        } //Color
      } //Spin
    }
  };

  /**
   * Adds the twisted-mass term to the coarse local term.
   */
  template <typename Arg> struct add_coarse_tm {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr add_coarse_tm(const Arg &arg) : arg(arg) { }

    __device__ __host__ void operator()(int x_cb, int parity, int c)
    {
      const complex<typename Arg::Float> mu(0., arg.mu*arg.mu_factor);

      for (int s = 0; s < Arg::coarseSpin/2; s++) { //Spin
        arg.X_atomic(0, parity, x_cb, s, s, c, c) += mu;
      } //Spin
      for (int s = Arg::coarseSpin/2; s < Arg::coarseSpin; s++) { //Spin
        arg.X_atomic(0, parity, x_cb, s, s, c, c) -= mu;
      } //Spin
    }
  };

  /**
   * Convert the field from the atomic format to the required computation format, e.g. fixed point to floating point
   */
  template <typename Arg> struct convert {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr convert(const Arg &arg) : arg(arg) { }

    __device__ __host__ void operator()(int x_cb, int parity_c_col, int c_row)
    {
      int c_col = parity_c_col % Arg::coarseColor; // color col index
      int parity = parity_c_col / Arg::coarseColor;

      if (arg.dim_index < 8) {
        const auto &in = arg.Y_atomic;
        int d_in = arg.dim_index % in.geometry;
        int d_out = arg.dim_index % arg.Y.geometry;

#pragma unroll
        for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { //Spin row
#pragma unroll
          for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { //Spin column
            complex<typename Arg::Float> M = in(d_in,parity,x_cb,s_row,s_col,c_row,c_col);
            arg.Y(d_out,parity,x_cb,s_row,s_col,c_row,c_col) = M;
          } //Spin column
        } //Spin row
      } else {
        const auto &in = arg.X_atomic;
        int d_in = arg.dim_index % in.geometry;
        int d_out = arg.dim_index % arg.X.geometry;

#pragma unroll
        for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { //Spin row
#pragma unroll
          for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { //Spin column
            complex<typename Arg::Float> M = in(d_in,parity,x_cb,s_row,s_col,c_row,c_col);
            arg.X(d_out,parity,x_cb,s_row,s_col,c_row,c_col) = M;
          } //Spin column
        } //Spin row
      }
    }
  };

  /**
   * Rescale the matrix elements by arg.rescale
   */
  template <typename Arg> struct rescale {
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr rescale(const Arg &arg) : arg(arg) { }

    __device__ __host__ void operator()(int x_cb, int parity_c_col, int c_row)
    {
      int c_col = parity_c_col % Arg::coarseColor; // color col index
      int parity = parity_c_col / Arg::coarseColor;

#pragma unroll
      for (int s_row = 0; s_row < Arg::coarseSpin; s_row++) { //Spin row
#pragma unroll
        for (int s_col = 0; s_col < Arg::coarseSpin; s_col++) { //Spin column
          complex<typename Arg::Float> M = arg.Y(arg.dim_index,parity,x_cb,s_row,s_col,c_row,c_col);
          arg.Y(arg.dim_index,parity,x_cb,s_row,s_col,c_row,c_col) = arg.rescale*M;
        } //Spin column
      } //Spin row
    }
  };

} // namespace quda
