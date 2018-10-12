#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <multigrid_helper.cuh>
#include <index_helper.cuh>
#include <gamma.cuh>

#define max_color_per_block 8

namespace quda {

  // this is the storage type used when computing the coarse link variables
  // by using integers we have deterministic atomics
  typedef int storeType;

  template <typename Float, int coarseSpin, int fineColor, int coarseColor,
      typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge,
      typename fineSpinorTmp, typename fineSpinorV>
  struct CalculateStaggeredYArg {

    coarseGauge Y;           /** Computed coarse link field */
    coarseGauge X;           /** Computed coarse clover field */

    coarseGaugeAtomic Y_atomic;    /** Y atomic accessor used for computation before conversion to final format */
    coarseGaugeAtomic X_atomic;    /** X atomic accessor used for computation before conversion to final format */

    fineSpinorTmp UV;        /** Temporary that stores the fine-link * spinor field product */

    const fineGauge U;       /** Fine grid (fat-)link field */
    // May have a long-link variable in the future.
    const fineSpinorV V;     /** Fine grid spinor field */
    // Staggered doesn't have a clover term.

    int_fastdiv x_size[QUDA_MAX_DIM];   /** Dimensions of fine grid */
    int xc_size[QUDA_MAX_DIM];  /** Dimensions of coarse grid */

    int_fastdiv geo_bs[QUDA_MAX_DIM];   /** Geometric block dimensions */
    const int spin_bs;          /** Spin block size */
    const spin_mapper<1,coarseSpin> spin_map; /** Helper that maps fine spin to coarse spin */

    int comm_dim[QUDA_MAX_DIM]; /** Node parition array */

    Float mass;                 /** staggered mass value */
    Float rescale;              /** rescaling factor used when rescaling the Y links if the maximum increases */

    const int fineVolumeCB;     /** Fine grid volume */
    const int coarseVolumeCB;   /** Coarse grid volume */

    const int *fine_to_coarse;
    const int *coarse_to_fine;

    const bool bidirectional;

    static constexpr int coarse_color = coarseColor;
    // To increase L2 locality we can schedule the geometry to grid.y and
    // the coarse colors to grid.x.  This will increase the potential for
    // L2 reuse since a given wave of thread blocks will be for different
    // coarse color but the same coarse grid point which will have common
    // loads.
    static constexpr bool coarse_color_wave = true;
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

    int dim_index; // which direction / dimension we are working on

    CalculateStaggeredYArg(coarseGauge &Y, coarseGauge &X,
      coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
      fineSpinorTmp &UV, const fineGauge &U, const fineSpinorV &V,
      double mass, const int *x_size_, const int *xc_size_, int *geo_bs_, int spin_bs_,
      const int *fine_to_coarse, const int *coarse_to_fine, bool bidirectional)
      : Y(Y), X(X), Y_atomic(Y_atomic), X_atomic(X_atomic),
        UV(UV), U(U), V(V), spin_bs(spin_bs_), spin_map(),
        mass(static_cast<Float>(mass)), 
        fineVolumeCB(V.VolumeCB()), coarseVolumeCB(X.VolumeCB()),
        fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
        bidirectional(bidirectional), shared_atomic(false), parity_flip(shared_atomic ? true : false),
        aggregates_per_block(1)
    {
      if (V.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS) // Maybe we can comment this out?
        errorQuda("Gamma basis %d not supported", V.GammaBasis());

      for (int i=0; i<QUDA_MAX_DIM; i++) {
        x_size[i] = x_size_[i];
        xc_size[i] = xc_size_[i];
        geo_bs[i] = geo_bs_[i];
        comm_dim[i] = comm_dim_partitioned(i);
      }
    }

    ~CalculateStaggeredYArg() { }
  };

  // complex multiply-add with optimal use of fma
  template<typename Float>
  inline __device__ __host__ void caxpy(const complex<Float> &a, const complex<Float> &x, complex<Float> &y) {
    y.x += a.x*x.x;
    y.x -= a.y*x.y;
    y.y += a.y*x.x;
    y.y += a.x*x.y;
  }


  /**
     Calculates the matrix UV^{c'}_mu(x) = \sum_c U^{c}_mu(x) * V^{c}_mu(x+mu)
     Where: mu = dir, c' = coarse color, c = fine color
     Staggered fermions don't carry spin.
  */
  template<typename Float, int dim, QudaDirection dir, int fineColor,
     int coarseSpin, int coarseColor, typename Wtype, typename Arg>
  __device__ __host__ inline void computeStaggeredUV(Arg &arg, const Wtype &W, int parity, int x_cb, int ic_c) {

    int coord[5];
    coord[4] = 0;
    getCoords(coord, x_cb, arg.x_size, parity);

    // UV doesn't have any spin.
    complex<Float> UV[fineColor];

    for(int c = 0; c < fineColor; c++) {
      UV[c] = static_cast<Float>(0.0);
    }

    if ( arg.comm_dim[dim] && (coord[dim] + 1 >= arg.x_size[dim]) ) {
      int nFace = 1;
      int ghost_idx = ghostFaceIndex<1>(coord, arg.x_size, dim, nFace);

      for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
        for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
          caxpy(arg.U(dim, parity, x_cb, ic, jc), W.Ghost(dim, 1, (parity+1)&1, ghost_idx, 0, jc, ic_c), UV[ic]);
        }  //Fine color columns
      }  //Fine color rows

    } else {
      int y_cb = linkIndexP1(coord, arg.x_size, dim);

      for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
        for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
          caxpy(arg.U(dim, parity, x_cb, ic, jc), W((parity+1)&1, y_cb, 0, jc, ic_c), UV[ic]);
        }  //Fine color columns
      }  //Fine color rows

    }

    for(int c = 0; c < fineColor; c++) {
      arg.UV(parity,x_cb,0,c,ic_c) = UV[c];
    }

  } // computeStaggeredUV

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeStaggeredUVCPU(Arg &arg) {

    for (int parity=0; parity<2; parity++) {
      #pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) {
        for (int ic_c=0; ic_c < coarseColor; ic_c++) { // coarse color
          computeStaggeredUV<Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, arg.V, parity, x_cb, ic_c);
        } // coarse color
      } // c/b volume
    }   // parity
  }

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeStaggeredUVGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // coarse color
    if (ic_c >= coarseColor) return;
    computeStaggeredUV<Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, arg.V, parity, x_cb, ic_c);
  }

  /**
     @brief Do a single (V)^\dagger * UV product, where V is simply
     the packed null space vectors.

     @param[out] vuv Result 
     @param[in,out] arg Arg storing the fields and parameters
     @param[in] Fine grid parity we're working on
     @param[in] x_cb Checkboarded x dimension
   */
  template <typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ inline void multiplyStaggeredVUV(complex<Float> vuv[], const Arg &arg, int parity, int x_cb, int ic_c, int jc_c) {

#pragma unroll
    for (int i=0; i<coarseSpin*coarseSpin; i++) vuv[i] = 0.0;

    const int s = 0; // fine spin is always 0, since it's staggered.
    const int s_c_row = arg.spin_map(s,parity); // Coarse spin row index

    // If computing the backwards (forwards) direction link then
    // we desire the positive (negative) hopping term.

    int s_col = 0;
    const int s_c_col = arg.spin_map(s_col,1-parity); // Coarse spin col index

#pragma unroll
    for (int ic = 0; ic < fineColor; ic++) { //Sum over fine color

      complex<Float> V = arg.V(parity, x_cb, 0, ic, ic_c);

      if (dir == QUDA_BACKWARDS) {

        // off diagonal contribution
        caxpy(conj(V), arg.UV(parity, x_cb, s_col, ic, jc_c), vuv[s_c_row*coarseSpin+s_c_col]);

      } else {

        // off diagonal contribution
        caxpy(-conj(V), arg.UV(parity, x_cb, s_col, ic, jc_c), vuv[s_c_row*coarseSpin+s_c_col]);
      }
    } //Fine color

  }

  template<typename Arg>
  __device__ __host__ inline int virtualThreadIdx(const Arg &arg) {
    constexpr int warp_size = 32;
    int warp_id = threadIdx.x / warp_size;
    int warp_lane = threadIdx.x % warp_size;
    int tx = warp_id * (warp_size / arg.aggregates_per_block) + warp_lane / arg.aggregates_per_block;
    return tx;
  }

  template<typename Arg>
  __device__ __host__ inline int virtualBlockDim(const Arg &arg) {
    int block_dim_x = blockDim.x / arg.aggregates_per_block;
    return block_dim_x;
  }

  template<typename Arg>
  __device__ __host__ inline int coarseIndex(const Arg &arg) {
    constexpr int warp_size = 32;
    int warp_lane = threadIdx.x % warp_size;
    int x_coarse = (arg.coarse_color_wave ? blockIdx.y : blockIdx.x)*arg.aggregates_per_block + warp_lane % arg.aggregates_per_block;
    return x_coarse;
  }

  template<bool shared_atomic, bool parity_flip, typename Float, int dim, QudaDirection dir,
           int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ void ComputeStaggeredVUV(Arg &arg, int parity, int x_cb, int c_row, int c_col, int parity_coarse_, int coarse_x_cb_) {

    constexpr int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    getCoords(coord, x_cb, arg.x_size, parity);
    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    // ESW note: debug this later. We need to update
    // multiplyStaggeredVUV to only grab the relevant component.

    // Can use the fine coordinates to find out s_row and s_col.
    //const int s_row = (coord[0]+coord[1]+coord[2]+coord[3])%2;
    //const int s_col = 1-s_row;

    //Check to see if we are on the edge of a block.  If adjacent site
    //is in same block, M = X, else M = Y
    const bool isDiagonal = ((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim] ? true : false;

    int coarse_parity = shared_atomic ? parity_coarse_ : 0;
    if (!shared_atomic) {
      for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
      coarse_parity &= 1;
      coord_coarse[0] /= 2;
    }
    int coarse_x_cb = shared_atomic ? coarse_x_cb_ : ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];


    complex<Float> vuv[coarseSpin*coarseSpin];
    multiplyStaggeredVUV<Float,dim,dir,fineColor,coarseSpin,coarseColor,Arg>(vuv, arg, parity, x_cb, c_row, c_col);

    const int dim_index = arg.dim_index % arg.Y_atomic.geometry;


    if (shared_atomic) {

#ifdef __CUDA_ARCH__
      __shared__ complex<storeType> X[max_color_per_block][max_color_per_block][4][coarseSpin][coarseSpin];
      __shared__ complex<storeType> Y[max_color_per_block][max_color_per_block][4][coarseSpin][coarseSpin];
      int x_ = coarse_x_cb%arg.aggregates_per_block;

      int tx = virtualThreadIdx(arg);
      int s_col = tx / coarseSpin;
      int s_row = tx % coarseSpin;

      int c_col_block = c_col % max_color_per_block;
      int c_row_block = c_row % max_color_per_block;

      if (tx < coarseSpin*coarseSpin) {
        Y[c_row_block][c_col_block][x_][s_row][s_col] = 0;
        X[c_row_block][c_col_block][x_][s_row][s_col] = 0;
      }

      __syncthreads();

      if (!isDiagonal) {
#pragma unroll
        for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
          for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
            if (gauge::fixed_point<Float,storeType>()) {
              Float scale = arg.Y_atomic.accessor.scale;
              complex<storeType> a(round(scale * vuv[s_row*coarseSpin+s_col].real()),
                                   round(scale * vuv[s_row*coarseSpin+s_col].imag()));
              atomicAdd(&Y[c_row_block][c_col_block][x_][s_row][s_col],a);
            } else {
              atomicAdd(&Y[c_row_block][c_col_block][x_][s_row][s_col],reinterpret_cast<complex<storeType>*>(vuv)[s_row*coarseSpin+s_col]);
            }
          } // chiral column blocks
        } // chiral row blocks
      } else { // is diagonal
#pragma unroll
        for (int s_row = 0; s_row < coarseSpin; s_row++) { // Chiral row block
#pragma unroll
          for (int s_col = 0; s_col < coarseSpin; s_col++) { // Chiral column block
            if (gauge::fixed_point<Float,storeType>()) {
              Float scale = arg.X_atomic.accessor.scale;
              complex<storeType> a(round(scale * vuv[s_row*coarseSpin+s_col].real()),
                                   round(scale * vuv[s_row*coarseSpin+s_col].imag()));
              atomicAdd(&X[c_row_block][c_col_block][x_][s_row][s_col],a);
            } else {
              atomicAdd(&X[c_row_block][c_col_block][x_][s_row][s_col],reinterpret_cast<complex<storeType>*>(vuv)[s_row*coarseSpin+s_col]);
            }
          } // chiral column blocks
        } // chiral row blocks
      } // end is diagonal

      __syncthreads();

      if (tx < coarseSpin*coarseSpin && (parity == 0 || parity_flip == 1)) {
        arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,Y[c_row_block][c_col_block][x_][s_row][s_col]);

        if (dir == QUDA_BACKWARDS) {
          arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_col,s_row,c_col,c_row,conj(X[c_row_block][c_col_block][x_][s_row][s_col]));
        } else {
          arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,X[c_row_block][c_col_block][x_][s_row][s_col]);
        }

        if (!arg.bidirectional) {
          // In reality, the s_row == s_col contribution is zero b/c staggered parity is even/odd.
          // That's why, in contrast to Wilson, we can get away with just doing the "-" contribution, since
          // the pieces that are "wrong" are zero anyway.
          arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,-X[c_row_block][c_col_block][x_][s_row][s_col]);
        }
      }

#else // #ifndef __CUDA_ARCH__
      errorQuda("Shared-memory atomic aggregation not supported on CPU");
#endif

    } else { // (!shared_atomic)

      if (!isDiagonal) {
#pragma unroll
        for (int s_row = 0; s_row < coarseSpin; s_row++) { // chiral row block
#pragma unroll
          for (int s_col = 0; s_col < coarseSpin; s_col++) { // chiral column block
            arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,vuv[s_row*coarseSpin+s_col]);
          } // chiral col
        } // chiral row
      } else { // (isDiagonal)

        if (dir == QUDA_BACKWARDS) {
#pragma unroll
          for (int s_row = 0; s_row < coarseSpin; s_row++) { // chiral row block
#pragma unroll
            for (int s_col = 0; s_col < coarseSpin; s_col++) { // chiral column block
              arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_col,s_row,c_col,c_row,conj(vuv[s_row*coarseSpin+s_col]));
            } // chiral col
          } // chiral row
        } else { // (dir == QUDA_FORWARDS)
#pragma unroll
          for (int s_row = 0; s_row < coarseSpin; s_row++) { // chiral row block
#pragma unroll
            for (int s_col = 0; s_col < coarseSpin; s_col++) { // chiral column block
              arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,vuv[s_row*coarseSpin+s_col]);
            } // chiral col
          } // chiral row
        }

        if (!arg.bidirectional) {
  #pragma unroll
          for (int s_row = 0; s_row < coarseSpin; s_row++) { // chiral row block
  #pragma unroll
            for (int s_col = 0; s_col < coarseSpin; s_col++) { // chiral column block
              arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,-vuv[s_row*coarseSpin+s_col]);
            } // chiral col
          } // chiral row
        } // end (!arg.bidirectional)

      } // end (isDiagonal)
    } // end (!shared_atomic)


  }

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeStaggeredVUVCPU(Arg arg) {

    constexpr bool shared_atomic = false; // no supported on CPU
    constexpr bool parity_flip = true;

    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
        for (int c_row=0; c_row<coarseColor; c_row++) {
          for (int c_col=0; c_col<coarseColor; c_col++) {
            ComputeStaggeredVUV<shared_atomic,parity_flip,Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, c_row, c_col, 0, 0);
          } // coarse color columns
        } // coarse color rows
      } // c/b volume
    } // parity
  }

  // compute indices for shared-atomic kernel
  template <bool parity_flip, typename Arg>
  __device__ inline void getIndicesShared(const Arg &arg, int &parity, int &x_cb, int &parity_coarse, int &x_coarse_cb, int &c_col, int &c_row) {

    if (arg.coarse_color_wave) {
      int parity_c_col_block_z = blockDim.y*blockIdx.x + threadIdx.y;
      int c_col_block_z = parity_flip ? (parity_c_col_block_z % arg.coarse_color_grid_z ) : (parity_c_col_block_z / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col_block_z / arg.coarse_color_grid_z ) : (parity_c_col_block_z % 2);
      c_col = c_col_block_z % arg.coarse_color;
      c_row = blockDim.z*(c_col_block_z/arg.coarse_color) + threadIdx.z; // coarse color row index
    } else {
      int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
      c_col = parity_flip ? (parity_c_col % arg.coarse_color) : (parity_c_col / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col / arg.coarse_color) : (parity_c_col % 2);
      c_row = blockDim.z*blockIdx.z + threadIdx.z; // coarse color row index
    }

    int block_dim_x = virtualBlockDim(arg);
    int thread_idx_x = virtualThreadIdx(arg);
    int x_coarse = coarseIndex(arg);

    parity_coarse = x_coarse >= arg.coarseVolumeCB ? 1 : 0;
    x_coarse_cb = x_coarse - parity_coarse*arg.coarseVolumeCB;

    // obtain fine index from this look-up table
    // since both parities map to the same block, each thread block must do both parities

    // threadIdx.x - fine checkboard offset
    // threadIdx.y - fine parity offset
    // blockIdx.x  - which coarse block are we working on
    // assume that coarse_to_fine look up map is ordered as (coarse-block-id + fine-point-id)
    // and that fine-point-id is parity ordered

    int x_fine = arg.coarse_to_fine[ (x_coarse*2 + parity) * block_dim_x + thread_idx_x];
    x_cb = x_fine - parity*arg.fineVolumeCB;
  }

  // compute indices for global-atomic kernel
  template <bool parity_flip, typename Arg>
  __device__ inline void getIndicesGlobal(const Arg &arg, int &parity, int &x_cb, int &parity_coarse, int &x_coarse_cb, int &c_col, int &c_row) {

    x_cb = blockDim.x*(arg.coarse_color_wave ? blockIdx.y : blockIdx.x) + threadIdx.x;

    if (arg.coarse_color_wave) {
      int parity_c_col_block_z = blockDim.y*blockIdx.x + threadIdx.y;
      int c_col_block_z = parity_flip ? (parity_c_col_block_z % arg.coarse_color_grid_z ) : (parity_c_col_block_z / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col_block_z / arg.coarse_color_grid_z ) : (parity_c_col_block_z % 2);
      c_col = c_col_block_z % arg.coarse_color;
      c_row = blockDim.z*(c_col_block_z/arg.coarse_color) + threadIdx.z; // coarse color row index
    } else {
      int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
      c_col  = parity_flip ? (parity_c_col % arg.coarse_color) : (parity_c_col / 2); // coarse color col index
      parity = parity_flip ? (parity_c_col / arg.coarse_color) : (parity_c_col % 2); // coarse color col index
      c_row = blockDim.z*blockIdx.z + threadIdx.z; // coarse color row index
    }

    x_coarse_cb = 0;
    parity_coarse = 0;
  }


  template<bool shared_atomic, bool parity_flip, typename Float, int dim, QudaDirection dir,
           int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeStaggeredVUVGPU(Arg arg) {

    int parity, x_cb, parity_coarse, x_coarse_cb, c_col, c_row;
    if (shared_atomic) getIndicesShared<parity_flip>(arg, parity, x_cb, parity_coarse, x_coarse_cb, c_col, c_row);
    else getIndicesGlobal<parity_flip>(arg, parity, x_cb, parity_coarse, x_coarse_cb, c_col, c_row);
    if (parity > 1) return;
    if (c_col >= arg.coarse_color) return;
    if (c_row >= arg.coarse_color) return;
    if (!shared_atomic && x_cb >= arg.fineVolumeCB) return;
    
    ComputeStaggeredVUV<shared_atomic,parity_flip,Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, c_row, c_col, parity_coarse, x_coarse_cb);
  }

  /**
   * Compute the forward links from backwards links by flipping the
   * sign
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeStaggeredYreverse(Arg &arg, int parity, int x_cb, int ic_c, int jc_c) {
    auto &Y = arg.Y;

#pragma unroll
    for (int d=0; d<4; d++) {
#pragma unroll
      for (int s_row = 0; s_row < nSpin; s_row++) { // chiral row block
#pragma unroll
        for (int s_col = 0; s_col < nSpin; s_col++) { // chiral col block
          Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = -Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
        } // chiral col
      } // chiral row
    } // dimension

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeStaggeredYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
          for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color col
            computeStaggeredYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c, jc_c);
          }
        }
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeStaggeredYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity_jc_c = blockDim.y*blockIdx.y + threadIdx.y; // parity and color col
    if (parity_jc_c >= 2*nColor) return;
    int parity = parity_jc_c / nColor;
    int jc_c = parity_jc_c % nColor;


    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // color row
    if (ic_c >= nColor) return;

    computeStaggeredYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c, jc_c);
  }

  //Adds the mass to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  void AddCoarseStaggeredMassCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int s = 0; s < nSpin; s++) { //Spin
          for(int c = 0; c < nColor; c++) { //Color
            arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(2.0*arg.mass,0.0); // staggered conventions
          } //Color
        } //Spin
      } // x_cb
    } //parity
  }


  // Adds the mass to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void AddCoarseStaggeredMassGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int parity = blockDim.y*blockIdx.y + threadIdx.y;

    for(int s = 0; s < nSpin; s++) { //Spin
      for(int c = 0; c < nColor; c++) { //Color
        arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(2.0*arg.mass,0.0); // staggered conventions
      } //Color
    } //Spin
   }

  /**
   * Convert the field from the atomic format to the required computation format, e.g. fixed point to floating point
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void convertStaggered(Arg &arg, int parity, int x_cb, int c_row, int c_col) {

    if (arg.dim_index < 8) {

      const auto &in = arg.Y_atomic;
      int d_in = arg.dim_index % in.geometry;
      int d_out = arg.dim_index % arg.Y.geometry;

#pragma unroll
      for (int s_row = 0; s_row < nSpin; s_row++) { //Spin row
#pragma unroll
        for (int s_col = 0; s_col < nSpin; s_col++) { //Spin column
          complex<Float> M = in(d_in,parity,x_cb,s_row,s_col,c_row,c_col);
          arg.Y(d_out,parity,x_cb,s_row,s_col,c_row,c_col) = M;
        } //Spin column
      } //Spin row
    } else {
      const auto &in = arg.X_atomic;
      int d_in = arg.dim_index % in.geometry;
      int d_out = arg.dim_index % arg.X.geometry;
#pragma unroll
      for (int s_row = 0; s_row < nSpin; s_row++) { //Spin row
#pragma unroll
        for (int s_col = 0; s_col < nSpin; s_col++) { //Spin column
          complex<Float> M = in(d_in,parity,x_cb,s_row,s_col,c_row,c_col);
          arg.X(d_out,parity,x_cb,s_row,s_col,c_row,c_col) = M;
        } //Spin column
      } // spin row
    } // arg.dim_index >= 8
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ConvertStaggeredCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int c_row = 0; c_row < nColor; c_row++) { //Color row
          for(int c_col = 0; c_col < nColor; c_col++) { //Color column
            convertStaggered<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
          }
        }
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ConvertStaggeredGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*nColor) return;

    int c_col = parity_c_col % nColor; // color col index
    int parity = parity_c_col / nColor;

    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // color row index
    if (c_row >= nColor) return;

    convertStaggered<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
  }

  /**
   * Rescale the matrix elements by arg.rescale
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void rescaleStaggeredY(Arg &arg, int parity, int x_cb, int c_row, int c_col) {
#pragma unroll
    for (int s_row = 0; s_row < nSpin; s_row++) { //Spin row
#pragma unroll
      for (int s_col = 0; s_col < nSpin; s_col++) { //Spin column
        complex<Float> M = arg.Y(arg.dim_index,parity,x_cb,s_row,s_col,c_row,c_col);
        arg.Y(arg.dim_index,parity,x_cb,s_row,s_col,c_row,c_col) = arg.rescale*M;
      } //Spin column
    } //Spin row
  }
  template<typename Float, int nSpin, int nColor, typename Arg>
  void RescaleStaggeredYCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
  for(int c_row = 0; c_row < nColor; c_row++) { //Color row
    for(int c_col = 0; c_col < nColor; c_col++) { //Color column
      rescaleStaggeredY<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
    }
  }
      } // c/b volume
    } // parity
  }
  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void RescaleStaggeredYGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;
    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*nColor) return;
    int c_col = parity_c_col % nColor; // color col index
    int parity = parity_c_col / nColor;
    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // color row index
    if (c_row >= nColor) return;
    rescaleStaggeredY<Float,nSpin,nColor,Arg>(arg, parity, x_cb, c_row, c_col);
  }

} // namespace quda
