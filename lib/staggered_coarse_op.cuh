#include <multigrid_helper.cuh>

#define max_color_per_block 8

namespace quda {

  // All staggered operators are un-preconditioned, so we use uni-directional
  // coarsening. For debugging, though, we can force bi-directional coarsening.
  static bool bidirectional_debug = false;

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

    constexpr int dim_index = (dir == QUDA_BACKWARDS) ? dim : dim + 4;


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
    auto &Y = arg.Y_atomic;

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

    const auto &Yin = arg.Y_atomic;
    const auto &Xin = arg.X_atomic;

    for (int d=0; d<8; d++) {
      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
        for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
          complex<Float> Y = Yin(d,parity,x_cb,s_row,s_col,c_row,c_col);
          arg.Y(d,parity,x_cb,s_row,s_col,c_row,c_col) = Y;
        } //Spin column
      } //Spin row
    } // dimension

    for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
      for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column
        arg.X(0,parity,x_cb,s_row,s_col,c_row,c_col) = Xin(0,parity,x_cb,s_row,s_col,c_row,c_col);
      }
    }

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

  enum ComputeType {
    COMPUTE_UV,
    COMPUTE_VUV,
    COMPUTE_REVERSE_Y,
    COMPUTE_MASS,
    COMPUTE_CONVERT,
    COMPUTE_INVALID
  };

  template <typename Float,
	    int fineColor, int coarseSpin, int coarseColor, typename Arg>
  class CalculateStaggeredY : public TunableVectorYZ {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;
    GaugeField &Y;
    GaugeField &X;

    int dim;
    QudaDirection dir;
    ComputeType type;

    // NEED TO UPDATE
    long long flops() const
    {
      long long flops_ = 0;
      switch (type) {
      case COMPUTE_UV:
        // when fine operator is coarse take into account that the link matrix has spin dependence
        flops_ = 2l * arg.fineVolumeCB * 8 * coarseColor * fineColor * fineColor;
        break;
      case COMPUTE_VUV:
        // when the fine operator is truly fine the VUV multiplication is block sparse which halves the number of operations
        flops_ = 2l * arg.fineVolumeCB * 8 * coarseColor * coarseColor * fineColor / coarseSpin;
        break;
      case COMPUTE_REVERSE_Y:
        // no floating point operations
        flops_ = 0;
        break;
      case COMPUTE_MASS:
        flops_ = 2l * arg.coarseVolumeCB*coarseSpin*coarseColor;
        break;
      case COMPUTE_CONVERT: // no floating point operations
        flops_ = 0;
        break;
      default:
        errorQuda("Undefined compute type %d", type);
      }
      // 2 from parity, 8 from complex
      return flops_;
    }

    long long bytes() const
    {
      long long bytes_ = 0;
      switch (type) {
      case COMPUTE_UV:
        bytes_ = arg.UV.Bytes() + arg.V.Bytes() + 2*arg.U.Bytes()*coarseColor;
        break;
      case COMPUTE_VUV:
        {
          // formula for shared-atomic variant assuming parity_flip = true
          int writes = 4;
          // we use a (coarseColor * coarseColor) matrix of threads so each load is input element is loaded coarseColor times
          // we ignore the multiple loads of spin since these are per thread (and should be cached?)
          bytes_ = 2*writes*arg.Y.Bytes() + (arg.bidirectional ? 1 : 2) * 2*writes*arg.X.Bytes() + coarseColor*(arg.UV.Bytes() + arg.V.Bytes());
          break;
        }
      case COMPUTE_REVERSE_Y:
        bytes_ = 4*2*2*arg.Y.Bytes(); // 4 from direction, 2 from i/o, 2 from parity
      case COMPUTE_MASS:
        bytes_ = 2*2*arg.X.Bytes(); // 2 from i/o, 2 from parity
        break;
      case COMPUTE_CONVERT:
        bytes_ = 2*(arg.X.Bytes() + arg.X_atomic.Bytes() + 8*(arg.Y.Bytes() + arg.Y_atomic.Bytes()));
        break;
      default:
        errorQuda("Undefined compute type %d", type);
      }
      return bytes_;
    }

    unsigned int minThreads() const {
      unsigned int threads = 0;
      switch (type) {
      case COMPUTE_UV:
      case COMPUTE_VUV:
        threads = arg.fineVolumeCB;
        break;
      case COMPUTE_REVERSE_Y:
      case COMPUTE_MASS:
      case COMPUTE_CONVERT:
        threads = arg.coarseVolumeCB;
        break;
      default:
        errorQuda("Undefined compute type %d", type);
      }
      return threads;
    }

    bool tuneGridDim() const { return false; } // don't tune the grid dimension
    bool tuneAuxDim() const { return type != COMPUTE_VUV ? false : true; }

    unsigned int sharedBytesPerBlock(const TuneParam &param) const {
      if (arg.shared_atomic && type == COMPUTE_VUV) return 4*sizeof(storeType)*max_color_per_block*max_color_per_block*4*coarseSpin*coarseSpin;
      return TunableVectorYZ::sharedBytesPerBlock(param);
    }

  public:
    CalculateStaggeredY(Arg &arg, const ColorSpinorField &meta, GaugeField &Y, GaugeField &X)
      : TunableVectorYZ(2,1), arg(arg), type(COMPUTE_INVALID),
        meta(meta), Y(Y), X(X), dim(0), dir(QUDA_BACKWARDS)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux,comm_dim_partitioned_string());
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) strcat(aux, getOmpThreadStr());
    }
    virtual ~CalculateStaggeredY() { }

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE);

      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {

      	if (type == COMPUTE_UV) {

      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredUVCPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredUVCPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredUVCPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredUVCPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredUVCPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredUVCPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredUVCPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredUVCPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else {
      	    errorQuda("Undefined direction %d", dir);
      	  }


      	} else if (type == COMPUTE_VUV) {

      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVCPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredVUVCPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredVUVCPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredVUVCPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVCPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==1) ComputeStaggeredVUVCPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==2) ComputeStaggeredVUVCPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	    else if (dim==3) ComputeStaggeredVUVCPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor>(arg);
      	  } else {
      	    errorQuda("Undefined direction %d", dir);
      	  }

      	} else if (type == COMPUTE_REVERSE_Y) {

      	  ComputeStaggeredYReverseCPU<Float,coarseSpin,coarseColor>(arg);

      	} else if (type == COMPUTE_MASS) {

      	  AddCoarseStaggeredMassCPU<Float,coarseSpin,coarseColor>(arg);

      	} else if (type == COMPUTE_CONVERT) {

      	  ConvertStaggeredCPU<Float,coarseSpin,coarseColor>(arg);

      	} else {
      	  errorQuda("Undefined compute type %d", type);
      	}
      } else {

      	if (type == COMPUTE_UV) {

      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredUVGPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==1) ComputeStaggeredUVGPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==2) ComputeStaggeredUVGPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==3) ComputeStaggeredUVGPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredUVGPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==1) ComputeStaggeredUVGPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==2) ComputeStaggeredUVGPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==3) ComputeStaggeredUVGPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	  } else {
      	    errorQuda("Undefined direction %d", dir);
      	  }

      	} else if (type == COMPUTE_VUV) {

          // need to resize the grid since we don't tune over the entire coarseColor dimension
          // factor of two comes from parity onto different blocks (e.g. in the grid)
          tp.grid.y = (2*coarseColor + tp.block.y - 1) / tp.block.y;
          tp.grid.z = (coarseColor + tp.block.z - 1) / tp.block.z;

          arg.shared_atomic = tp.aux.y;
          arg.parity_flip = tp.aux.z;

          if (arg.shared_atomic) {
            // check we have a valid problem size for shared atomics
            // constrint is due to how shared memory initialization and global store are done
            int block_size = arg.fineVolumeCB/arg.coarseVolumeCB;
            if (block_size/2 < coarseSpin*coarseSpin)
              errorQuda("Block size %d not supported in shared-memory atomic coarsening", block_size);

            arg.aggregates_per_block = tp.aux.x;
            tp.block.x *= tp.aux.x;
            tp.grid.x /= tp.aux.x;
          }

          if (arg.coarse_color_wave) {
            // swap x and y grids
            std::swap(tp.grid.y,tp.grid.x);
            // augment x grid with coarseColor row grid (z grid)
            arg.grid_z = tp.grid.z;
            arg.coarse_color_grid_z = coarseColor*tp.grid.z;
            tp.grid.x *= tp.grid.z;
            tp.grid.z = 1;
          }

          tp.shared_bytes -= sharedBytesPerBlock(tp); // shared memory is static so don't include it in launch

          if (arg.shared_atomic) {
            if (arg.parity_flip != true) errorQuda("parity_flip = %d not instantiated", arg.parity_flip);
            constexpr bool parity_flip = true;
            if (dir == QUDA_BACKWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<true,parity_flip,Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<true,parity_flip,Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<true,parity_flip,Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<true,parity_flip,Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else if (dir == QUDA_FORWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<true,parity_flip,Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<true,parity_flip,Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<true,parity_flip,Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<true,parity_flip,Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else {
              errorQuda("Undefined direction %d", dir);
            }
          } else {
            if (arg.parity_flip != false) errorQuda("parity_flip = %d not instantiated", arg.parity_flip);
            constexpr bool parity_flip = false;
            if (dir == QUDA_BACKWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<false,parity_flip,Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<false,parity_flip,Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<false,parity_flip,Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<false,parity_flip,Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else if (dir == QUDA_FORWARDS) {
              if      (dim==0) ComputeStaggeredVUVGPU<false,parity_flip,Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==1) ComputeStaggeredVUVGPU<false,parity_flip,Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==2) ComputeStaggeredVUVGPU<false,parity_flip,Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
              else if (dim==3) ComputeStaggeredVUVGPU<false,parity_flip,Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
            } else {
              errorQuda("Undefined direction %d", dir);
            }
          }

          tp.shared_bytes += sharedBytesPerBlock(tp); // restore shared memory

          if (arg.coarse_color_wave) {
            // revert the grids
            tp.grid.z = arg.grid_z;
            tp.grid.x /= tp.grid.z;
            std::swap(tp.grid.x,tp.grid.y);
          }
          if (arg.shared_atomic) {
            tp.block.x /= tp.aux.x;
            tp.grid.x *= tp.aux.x;
          }

      	} else if (type == COMPUTE_REVERSE_Y) {

      	  ComputeStaggeredYReverseGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

      	} else if (type == COMPUTE_MASS) {

      	  AddCoarseStaggeredMassGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

      	} else if (type == COMPUTE_CONVERT) {

      	  tp.grid.y = 2*coarseColor;
      	  ConvertStaggeredGPU<Float,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);

      	} else {
      	  errorQuda("Undefined compute type %d", type);
        }
      }
    }

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDimension(int dim_) { dim = dim_; }

    /**
       Set which dimension we are working on (where applicable)
    */
    void setDirection(QudaDirection dir_) { dir = dir_; }

    /**
       Set which computation we are doing
     */
    void setComputeType(ComputeType type_) {
      type = type_;
      switch(type) {
      case COMPUTE_VUV:
        arg.shared_atomic = false;
        arg.parity_flip = false;
        if (arg.shared_atomic) {
          // if not parity flip then we need to force parity within the block (hence factor of 2)
          resizeVector( (arg.parity_flip ? 1 : 2) * max_color_per_block,max_color_per_block);
        } else {
          resizeVector(2*max_color_per_block,max_color_per_block);
        }
      break;
      case COMPUTE_REVERSE_Y:
        resizeVector(2*coarseColor,coarseColor);
        break;
      case COMPUTE_CONVERT:
        resizeVector(1,coarseColor);
        break;
      case COMPUTE_UV:
        resizeVector(2,coarseColor);
        break;
      default:
      	resizeVector(2,1);
      	break;
      }

      resizeStep(1,1);
      if (arg.shared_atomic && type == COMPUTE_VUV && !arg.parity_flip) resizeStep(2,1);

      // do not tune spatial block size for VUV
      tune_block_x = (type == COMPUTE_VUV) ? false : true;
    }

    bool advanceAux(TuneParam &param) const
    {
      if (type != COMPUTE_VUV) return false;

      // exhausted the global-atomic search space so switch to
      // shared-atomic space
      if (param.aux.y == 0) {
        // pre-Maxwell does not support shared-memory atomics natively so no point in trying
        if ( __COMPUTE_CAPABILITY__ < 500 ) return false;
        // before advancing, check we can use shared-memory atomics
        int block_size = arg.fineVolumeCB/arg.coarseVolumeCB;
        if (block_size/2 < coarseSpin*coarseSpin) return false;

        arg.shared_atomic = true;
        arg.parity_flip = true; // this is usually optimal for shared atomics

        resizeVector( (arg.parity_flip ? 1 : 2) * max_color_per_block,max_color_per_block);
        if (!arg.parity_flip) resizeStep(2,1);

        // need to reset since we're switching to shared-memory atomics
        initTuneParam(param);

        return true;
      } else {
        // already doing shared-memory atomics but can tune number of
        // coarse grid points per block
        if (param.aux.x < 4) {
          param.aux.x *= 2;
          return true;
        } else {
          param.aux.x = 1;
          // completed all shared-memory tuning so reset to global atomics
          arg.shared_atomic = false;
          arg.parity_flip = false; // this is usually optimal for global atomics
          initTuneParam(param);
          return false;
        }
      }
    }

    bool advanceSharedBytes(TuneParam &param) const {
      return (!arg.shared_atomic && type == COMPUTE_VUV) ? false : Tunable::advanceSharedBytes(param);
    }

    bool advanceTuneParam(TuneParam &param) const {
      // only do autotuning if we have device fields
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_DEVICE) return Tunable::advanceTuneParam(param);
       
      else return false;
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      param.aux.x = 1; // aggregates per block
      param.aux.y = arg.shared_atomic;
      param.aux.z = arg.parity_flip; // not actually tuned over at present

      // with shared-atomic VUV, each block.x matches exactly to a c/b aggregate
      if (arg.shared_atomic && type == COMPUTE_VUV) {
      	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
      	param.grid.x = 2*arg.coarseVolumeCB;
      }
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);

      param.aux.x = 1; // aggregates per block
      param.aux.y = arg.shared_atomic;
      param.aux.z = arg.parity_flip; // not actually tuned over at present

      // with shared-atomic VUV, each block.x matches exactly to a c/b aggregate
      if (arg.shared_atomic && type == COMPUTE_VUV) {
        param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
        param.grid.x = 2*arg.coarseVolumeCB;
      }
    }

    TuneKey tuneKey() const {
      char Aux[TuneKey::aux_n];
      strcpy(Aux,aux);

      if      (type == COMPUTE_UV)                 strcat(Aux,",computeStaggeredUV");
      else if (type == COMPUTE_VUV)                strcat(Aux,",computeStaggeredVUV");
      else if (type == COMPUTE_REVERSE_Y)          strcat(Aux,",computeStaggeredYreverse");
      else if (type == COMPUTE_MASS)               strcat(Aux,",computeStaggeredMass");
      else if (type == COMPUTE_CONVERT)            strcat(Aux,",computeStaggeredConvert");
      else errorQuda("Unknown type=%d\n", type);

      if (type == COMPUTE_UV || type == COMPUTE_VUV) {
      	if      (dim == 0) strcat(Aux,",dim=0");
      	else if (dim == 1) strcat(Aux,",dim=1");
      	else if (dim == 2) strcat(Aux,",dim=2");
      	else if (dim == 3) strcat(Aux,",dim=3");

      	if (dir == QUDA_BACKWARDS) strcat(Aux,",dir=back");
      	else if (dir == QUDA_FORWARDS) strcat(Aux,",dir=fwd");

        if (arg.bidirectional && type == COMPUTE_VUV) strcat(Aux,",bidirectional");
      }

      const char *vol_str = (type == COMPUTE_REVERSE_Y || type == COMPUTE_MASS ||
			     type == COMPUTE_CONVERT) ? X.VolString () : meta.VolString();

      if (type == COMPUTE_VUV) {
      	strcat(Aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped," :
                meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device," : ",CPU,");
      	strcat(Aux,"coarse_vol=");
      	strcat(Aux,X.VolString());
      } else {
        strcat(Aux, (meta.Location()==QUDA_CUDA_FIELD_LOCATION && Y.MemType() == QUDA_MEMORY_MAPPED) ? ",GPU-mapped" :
                meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU-device" : ",CPU");
      }

      return TuneKey(vol_str, typeid(*this).name(), Aux);
    }

    void preTune() {
      switch (type) {
      case COMPUTE_VUV:
      case COMPUTE_CONVERT:
        Y.backup();
      case COMPUTE_MASS:
        X.backup();
      case COMPUTE_UV:
      case COMPUTE_REVERSE_Y:
        break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }

    void postTune() {
      switch (type) {
      case COMPUTE_VUV:
      case COMPUTE_CONVERT:
        Y.restore();
      case COMPUTE_MASS:
        X.restore();
      case COMPUTE_UV:
      case COMPUTE_REVERSE_Y:
	break;
      default:
	errorQuda("Undefined compute type %d", type);
      }
    }
  };



  /**
     @brief Calculate the coarse-link field, including the coarse clover field.

     @param Y[out] Coarse (fat-)link field accessor
     @param X[out] Coarse clover field accessor
     @param UV[out] Temporary accessor used to store fine link field * null space vectors
     @param V[in] Packed null-space vector accessor
     @param G[in] Fine grid link / gauge field accessor
     @param Y_[out] Coarse link field
     @param X_[out] Coarse clover field
     @param X_[out] Coarse clover inverese field (used as temporary here)
     @param v[in] Packed null-space vectors
     @param mass[in] Kappa parameter
     @param matpc[in] The type of preconditioning of the source fine-grid operator
   */
  template<typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	   typename Ftmp, typename Vt, typename coarseGauge, typename coarseGaugeAtomic, typename fineGauge>
  void calculateStaggeredY(coarseGauge &Y, coarseGauge &X,
		  coarseGaugeAtomic &Y_atomic, coarseGaugeAtomic &X_atomic,
		  Ftmp &UV, Vt &V, fineGauge &G,
		  GaugeField &Y_, GaugeField &X_, ColorSpinorField &uv,
		  const ColorSpinorField &v,
		  double mass, QudaDiracType dirac, QudaMatPCType matpc,
		  const int *fine_to_coarse, const int *coarse_to_fine) {

    // sanity checks
    if (matpc == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC || matpc == QUDA_MATPC_ODD_ODD_ASYMMETRIC)
      errorQuda("Unsupported coarsening of matpc = %d", matpc);

    // This is the last time we use fineSpin, since this file only coarsens
    // staggered-type ops, not wilson-type AND coarse-type.
    if (fineSpin != 1)
      errorQuda("Input Dirac operator %d should have nSpin=1, not nSpin=%d\n", dirac, fineSpin);
    if (fineColor != 3)
      errorQuda("Input Dirac operator %d should have nColor=3, not nColor=%d\n", dirac, fineColor);

    if (G.Ndim() != 4) errorQuda("Number of dimensions not supported");
    const int nDim = 4;

    int x_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) x_size[i] = v.X(i);
    x_size[4] = 1;

    int xc_size[QUDA_MAX_DIM] = { };
    for (int i=0; i<4; i++) xc_size[i] = X_.X()[i];
    xc_size[4] = 1;

    int geo_bs[QUDA_MAX_DIM] = { };
    for(int d = 0; d < nDim; d++) geo_bs[d] = x_size[d]/xc_size[d];
    int spin_bs = V.Nspin()/Y.NspinCoarse(); // may need to check this. I believe 0 was meant for spin-less types.

    // If doing a preconditioned operator with a clover term then we
    // have bi-directional links, though we can do the bidirectional setup for all operators for debugging
    bool bidirectional_links = bidirectional_debug;
    if (bidirectional_links) printfQuda("Doing bi-directional link coarsening\n");
    else printfQuda("Doing uni-directional link coarsening\n");

    //Calculate UV and then VUV for each dimension, accumulating directly into the coarse gauge field Y

    typedef CalculateStaggeredYArg<Float,coarseSpin,fineColor,coarseColor,coarseGauge,coarseGaugeAtomic,fineGauge,Ftmp,Vt> Arg;
    Arg arg(Y, X, Y_atomic, X_atomic, UV, G, V, mass, x_size, xc_size, geo_bs, spin_bs, fine_to_coarse, coarse_to_fine, bidirectional_links);
    CalculateStaggeredY<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, v, Y_, X_);

    QudaFieldLocation location = checkLocation(Y_, X_, v);
    printfQuda("Running link coarsening on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // do exchange of null-space vectors
    const int nFace = 1;
    v.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
    arg.V.resetGhost(v, v.Ghost());  // point the accessor to the correct ghost buffer
    LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange

    printfQuda("V2 = %e\n", arg.V.norm2());

    // work out what to set the scales to
    // ESW hack
    if (coarseGaugeAtomic::fixedPoint()) {
      double max = 100.0; // FIXME - more accurate computation needed?
      arg.Y_atomic.resetScale(max);
      arg.X_atomic.resetScale(max);
    }

    // First compute the coarse forward links if needed
    if (bidirectional_links) {
      for (int d = 0; d < nDim; d++) {
      	y.setDimension(d);
      	y.setDirection(QUDA_FORWARDS);
      	printfQuda("Computing forward %d UV and VUV\n", d);

      	if (uv.Precision() == QUDA_HALF_PRECISION) {
      	  double U_max = 3.0*arg.U.abs_max(d);
      	  double uv_max = U_max * v.Scale();
      	  uv.Scale(uv_max);
      	  arg.UV.resetScale(uv_max);

      	  if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
      	}

      	y.setComputeType(COMPUTE_UV);  // compute U*V product
      	y.apply(0);
      	printfQuda("UV2[%d] = %e\n", d, arg.UV.norm2());

      	y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      	y.apply(0);
      	printfQuda("Y2[%d] = %e\n", d, arg.Y_atomic.norm2(4+d));
      }
    }

    // Now compute the backward links
    for (int d = 0; d < nDim; d++) {
      y.setDimension(d);
      y.setDirection(QUDA_BACKWARDS);
      printfQuda("Computing backward %d UV and VUV\n", d);

      if (uv.Precision() == QUDA_HALF_PRECISION) {
      	double U_max = 3.0*arg.U.abs_max(d);
      	double uv_max = U_max * v.Scale();
      	uv.Scale(uv_max);
      	arg.UV.resetScale(uv_max);

      	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("%d U_max = %e v_max = %e uv_max = %e\n", d, U_max, v.Scale(), uv_max);
      }

      y.setComputeType(COMPUTE_UV);  // compute U*A*V product
      y.apply(0);
      printfQuda("UV2[%d] = %e\n", d+4, arg.UV.norm2());

      y.setComputeType(COMPUTE_VUV); // compute Y += VUV
      y.apply(0);
      printfQuda("Y2[%d] = %e\n", d+4, arg.Y_atomic.norm2(d));

    }
    printfQuda("X2 = %e\n", arg.X_atomic.norm2(0));

    // if not doing a preconditioned operator then we can trivially
    // construct the forward links from the backward links
    if ( !bidirectional_links ) {
      printfQuda("Reversing links\n");
      y.setComputeType(COMPUTE_REVERSE_Y);  // reverse the links for the forwards direction
      y.apply(0);
    }

    // Add the mass.
    printfQuda("Adding diagonal mass contribution to coarse clover\n");
    y.setComputeType(COMPUTE_MASS);
    y.apply(0);

    printfQuda("X2 = %e\n", arg.X_atomic.norm2(0));

    // now convert from atomic to application computation format if necesaary
    if (coarseGaugeAtomic::fixedPoint()) {
      y.setComputeType(COMPUTE_CONVERT);
      y.apply(0);
    }

  }


} // namespace quda
