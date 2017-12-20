#include <multigrid_helper.cuh>

// enable this for shared-memory atomics instead of global atomics.
// Doing so means that all of the coarsening for a coarse degree of
// freedom is handled by a single thread block.  This is presently
// slower than using global atomics (due to increased latency from
// having to run larger thread blocks)
//#define SHARED_ATOMIC

#ifdef SHARED_ATOMIC
// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
// if disabled then we pack multiple aggregates into a single block to improve coalescing
#ifdef SWIZZLE
#undef SWIZZLE
#endif
#endif

namespace quda {

  // All staggered operators are un-preconditioned, so we use uni-directional
  // coarsening. For debugging, though, we can force bi-directional coarsening.
  static bool bidirectional_debug = false;

  template <typename Float, int coarseSpin,
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

    int_fastdiv aggregates_per_block; // number of aggregates per thread block
    int_fastdiv swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate

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
        bidirectional(bidirectional), aggregates_per_block(1), swizzle(1)
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
          // need to check if I'm properly accessing elements of spin-less objs
          UV[ic] += arg.U(dim, parity, x_cb, ic, jc) * W.Ghost(dim, 1, (parity+1)&1, ghost_idx, 0 /*b/c no spin? */, jc, ic_c);
        }  //Fine color columns
      }  //Fine color rows

    } else {
      int y_cb = linkIndexP1(coord, arg.x_size, dim);

      for(int ic = 0; ic < fineColor; ic++) { //Fine Color rows of gauge field
        for(int jc = 0; jc < fineColor; jc++) {  //Fine Color columns of gauge field
          // need to check if I'm properly accessing elements of spin-less objs
          UV[ic] += arg.U(dim, parity, x_cb, ic, jc) * W((parity+1)&1, y_cb, 0 /* b/c no spin? */, jc, ic_c);
        }  //Fine color columns
      }  //Fine color rows

    }

    for(int c = 0; c < fineColor; c++) {
      arg.UV(parity,x_cb,0 /*?*/,c,ic_c) = UV[c];
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
  __device__ __host__ inline void multiplyStaggeredVUV(complex<Float>& vuv, const Arg &arg, int parity, int x_cb, int ic_c, int jc_c) {

    // We can reduce this to computing just one value. All gauge links
    // couple even-to-odd, and chirality is also even-to-odd,
    // so once we know the parity we're working with we exactly
    // know the spins.

    vuv = 0.0;

    const int s = 0; // fine spin is always 0, since it's staggered.
    const int s_c_row = arg.spin_map(s,parity); // Coarse spin row index

    // If computing the backwards (forwards) direction link then
    // we desire the positive (negative) hopping term.

    int s_col = 0;
    const int s_c_col = arg.spin_map(s_col,parity); // Coarse spin col index

#pragma unroll
    for (int ic = 0; ic < fineColor; ic++) { //Sum over fine color
      if (dir == QUDA_BACKWARDS) {

        // off diagonal contribution
        vuv += conj(arg.V(parity, x_cb, 0, ic, ic_c)) * arg.UV(parity, x_cb, 0, ic, jc_c);

      } else {

        // off diagonal contribution
        vuv -= conj(arg.V(parity, x_cb, 0, ic, ic_c)) * arg.UV(parity, x_cb, 0, ic, jc_c);
        
      }
    } //Fine color

  }

#ifndef SWIZZLE
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
    int x_coarse = blockIdx.x*arg.aggregates_per_block + warp_lane % arg.aggregates_per_block;
    return x_coarse;
  }

#else
  template<typename Arg>
  __device__ __host__ inline int virtualThreadIdx(const Arg &arg) { return threadIdx.x; }

  template<typename Arg>
  __device__ __host__ inline int virtualBlockDim(const Arg &arg) { return blockDim.x; }

  template<typename Arg>
  __device__ __host__ inline int coarseIndex(const Arg &arg) {
    // the portion of the grid that is exactly divisible by the number of SMs
    const int gridp = gridDim.x - gridDim.x % arg.swizzle;

    int x_coarse = blockIdx.x;
    if (blockIdx.x < gridp) {
      // this is the portion of the block that we are going to transpose
      const int i = blockIdx.x % arg.swizzle;
      const int j = blockIdx.x / arg.swizzle;

      // tranpose the coordinates
      x_coarse = i * (gridp / arg.swizzle) + j;
    }
    return x_coarse;
  }
#endif

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __device__ __host__ void ComputeStaggeredVUV(Arg &arg, int parity, int x_cb, int c_row, int c_col, int parity_coarse_, int coarse_x_cb_) {

    constexpr int nDim = 4;
    int coord[QUDA_MAX_DIM];
    int coord_coarse[QUDA_MAX_DIM];

    // Right off the bat: the fine lattice parity tells me s_row and s_col.
    const int s_row = parity;
    const int s_col = 1-parity;

    getCoords(coord, x_cb, arg.x_size, parity);
    for(int d = 0; d < nDim; d++) coord_coarse[d] = coord[d]/arg.geo_bs[d];

    //Check to see if we are on the edge of a block.  If adjacent site
    //is in same block, M = X, else M = Y
    const bool isDiagonal = ((coord[dim]+1)%arg.x_size[dim])/arg.geo_bs[dim] == coord_coarse[dim] ? true : false;

#if defined(SHARED_ATOMIC) && __CUDA_ARCH__
    int coarse_parity = parity_coarse_;
    int coarse_x_cb = coarse_x_cb_;
#else
    int coarse_parity = 0;
    for (int d=0; d<nDim; d++) coarse_parity += coord_coarse[d];
    coarse_parity &= 1;
    coord_coarse[0] /= 2;
    int coarse_x_cb = ((coord_coarse[3]*arg.xc_size[2]+coord_coarse[2])*arg.xc_size[1]+coord_coarse[1])*(arg.xc_size[0]/2) + coord_coarse[0];
#endif

    complex<Float> vuv = 0.0;
    multiplyStaggeredVUV<Float,dim,dir,fineColor,coarseSpin,coarseColor,Arg>(vuv, arg, parity, x_cb, c_row, c_col);

    constexpr int dim_index = (dir == QUDA_BACKWARDS) ? dim : dim + 4;
#if defined(SHARED_ATOMIC) && __CUDA_ARCH__
    __shared__ complex<storeType> X[4];
    __shared__ complex<storeType> Y[4];
    int x_ = coarse_x_cb%arg.aggregates_per_block;

    if (virtualThreadIdx(arg) == 0 && threadIdx.y == 0) {
      for (int s_row = 0; s_row<coarseSpin; s_row++) { 
        for (int s_col = 0; s_col<coarseSpin; s_col++) {
          Y[x_]= 0; X[x_] = 0;
        }
      }
    }

    __syncthreads();

    if (!isDiagonal) {
      if (gauge::fixed_point<Float,storeType()) {
        Float scale = arg.Y_atomic.accessor.scale;
        complex<storeType> a(round(scale * vuv.real()),
          round(scale * vuv.imag()));
        atomicAdd(&Y[x_],a);
      } else {
        atomicAdd(&Y[x_],(*reinterpret_cast<complex<storeType> >(&vuv)));
      }
    } else {
      if (gauge::fixed_point<Float,storeType()) {
        Float scale = arg.X_atomic.accessor.scale;
        complex<storeType> a(round(scale * vuv.real()),
          round(scale * vuv.imag()));
        atomicAdd(&X[x_],a);
      } else {
        atomicAdd(&X[x_],(*reinterpret_cast<complex<storeType> >(&vuv)));
      }
    }

    __syncthreads();

    if (virtualThreadIdx(arg)==0 && threadIdx.y==0) {

      arg.Y_atomic(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) = Y[x_];

      if (dir == QUDA_BACKWARDS) {
        arg.X_atomic(0,coarse_parity,coarse_x_cb,s_col,s_row,c_col,c_row) += conj(X[x_]);
      } else {
        arg.X_atomic(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) += X[x_];
      }

      if (!arg.bidirectional) {
        arg.X_atomic(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col) -= X[x_];
      }

    }

#else

    if (!isDiagonal) {
      arg.Y_atomic.atomicAdd(dim_index,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,vuv);
    } else {

      if (dir == QUDA_BACKWARDS) {
        arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_col,s_row,c_col,c_row,conj(vuv));
      } else {
        arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,vuv);
      }

      if (!arg.bidirectional) {
        arg.X_atomic.atomicAdd(0,coarse_parity,coarse_x_cb,s_row,s_col,c_row,c_col,-vuv);
      }

    }
#endif

  }

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  void ComputeStaggeredVUVCPU(Arg arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.fineVolumeCB; x_cb++) { // Loop over fine volume
        for (int c_row=0; c_row<coarseColor; c_row++) {
          for (int c_col=0; c_col<coarseColor; c_col++) {
            ComputeStaggeredVUV<Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, c_row, c_col, 0, 0);
          } // coarse color columns
        } // coarse color rows
      } // c/b volume
    } // parity
  }

  template<typename Float, int dim, QudaDirection dir, int fineColor, int coarseSpin, int coarseColor, typename Arg>
  __global__ void ComputeStaggeredVUVGPU(Arg arg) {

    int parity_c_col = blockDim.y*blockIdx.y + threadIdx.y;
    if (parity_c_col >= 2*coarseColor) return;

#ifdef SHARED_ATOMIC
    int c_col = parity_c_col / 2; // coarse color col index
    int parity = parity_c_col % 2;

    int block_dim_x = virtualBlockDim(arg);
    int thread_idx_x = virtualThreadIdx(arg);
    int x_coarse = coarseIndex(arg);

    int parity_coarse = x_coarse >= arg.coarseVolumeCB ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*arg.coarseVolumeCB;

    // obtain fine index from this look up table
    // since both parities map to the same block, each thread block must do both parities

    // threadIdx.x - fine checkboard offset
    // threadIdx.y - fine parity offset
    // blockIdx.x  - which coarse block are we working on (optionally swizzled to improve cache efficiency)
    // assume that coarse_to_fine look up map is ordered as (coarse-block-id + fine-point-id)
    // and that fine-point-id is parity ordered

    int x_fine = arg.coarse_to_fine[ (x_coarse*2 + parity) * block_dim_x + thread_idx_x];
    int x_cb = x_fine - parity*arg.fineVolumeCB;
#else
    int x_coarse_cb = 0;
    int parity_coarse = 0;

    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.fineVolumeCB) return;

    int c_col = parity_c_col % coarseColor; // coarse color col index
    int parity = parity_c_col / coarseColor;
#endif

    int c_row = blockDim.z*blockIdx.z + threadIdx.z; // coarse color row index
    if (c_row >= coarseColor) return;

    ComputeStaggeredVUV<Float,dim,dir,fineColor,coarseSpin,coarseColor>(arg, parity, x_cb, c_row, c_col, parity_coarse, x_coarse_cb);
  }

  /**
   * Compute the forward links from backwards links by flipping the
   * sign of the spin projector
   */
  template<typename Float, int nSpin, int nColor, typename Arg>
  __device__ __host__ void computeStaggeredYreverse(Arg &arg, int parity, int x_cb, int ic_c) {
    auto &Y = arg.Y_atomic;

    for (int d=0; d<4; d++) {
      for(int s_row = 0; s_row < nSpin; s_row++) { //Spin row
        for(int s_col = 0; s_col < nSpin; s_col++) { //Spin column

          const Float sign = (s_row == s_col) ? static_cast<Float>(1.0) : static_cast<Float>(-1.0);

          for(int jc_c = 0; jc_c < nColor; jc_c++) { //Color column
            Y(d+4,parity,x_cb,s_row,s_col,ic_c,jc_c) = sign*Y(d,parity,x_cb,s_row,s_col,ic_c,jc_c);
          } //Color column

        } //Spin column
      } //Spin row
    } // dimension

  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  void ComputeStaggeredYReverseCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int ic_c = 0; ic_c < nColor; ic_c++) { //Color row
          computeStaggeredYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c);
        }
      } // c/b volume
    } // parity
  }

  template<typename Float, int nSpin, int nColor, typename Arg>
  __global__ void ComputeStaggeredYReverseGPU(Arg arg) {
    int x_cb = blockDim.x*blockIdx.x + threadIdx.x;
    if (x_cb >= arg.coarseVolumeCB) return;

    int ic_c = blockDim.z*blockIdx.z + threadIdx.z; // color row
    if (ic_c >= nColor) return;

    int parity = blockDim.y*blockIdx.y + threadIdx.y;
    computeStaggeredYreverse<Float,nSpin,nColor,Arg>(arg, parity, x_cb, ic_c);
  }

  //Adds the mass to the coarse local term.
  template<typename Float, int nSpin, int nColor, typename Arg>
  void AddCoarseStaggeredMassCPU(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
#pragma omp parallel for
      for (int x_cb=0; x_cb<arg.coarseVolumeCB; x_cb++) {
        for(int s = 0; s < nSpin; s++) { //Spin
          for(int c = 0; c < nColor; c++) { //Color
            arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(arg.mass,0.0);
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
        arg.X_atomic(0,parity,x_cb,s,s,c,c) += complex<Float>(arg.mass,0.0);
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
	bytes_ = 2*arg.Y.Bytes() + (arg.bidirectional ? 1 : 2) * 2*arg.X.Bytes() + arg.UV.Bytes() + arg.V.Bytes();
	break;
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
#ifndef SHARED_ATOMIC
      	  //tp.grid.y = 2*coarseColor;
#else
      	  tp.block.y = 2;
      	  tp.grid.y = coarseColor;
      	  tp.grid.z = coarseColor;
      	  arg.swizzle = tp.aux.x;

      	  arg.aggregates_per_block = tp.aux.y;
      	  tp.block.x *= tp.aux.y;
      	  tp.grid.x /= tp.aux.y;
#endif
      	  if (dir == QUDA_BACKWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVGPU<Float,0,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==1) ComputeStaggeredVUVGPU<Float,1,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==2) ComputeStaggeredVUVGPU<Float,2,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==3) ComputeStaggeredVUVGPU<Float,3,QUDA_BACKWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	  } else if (dir == QUDA_FORWARDS) {
      	    if      (dim==0) ComputeStaggeredVUVGPU<Float,0,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==1) ComputeStaggeredVUVGPU<Float,1,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==2) ComputeStaggeredVUVGPU<Float,2,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	    else if (dim==3) ComputeStaggeredVUVGPU<Float,3,QUDA_FORWARDS,fineColor,coarseSpin,coarseColor><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
      	  } else {
      	    errorQuda("Undefined direction %d", dir);
      	  }

#ifdef SHARED_ATOMIC
      	  tp.block.x /= tp.aux.y;
      	  tp.grid.x *= tp.aux.y;
#endif

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
#ifdef SHARED_ATOMIC
        resizeVector(1,1);
#else
        resizeVector(2*coarseColor,coarseColor);
#endif
      break;
      case COMPUTE_CONVERT:
        resizeVector(1,coarseColor);
        break;
      case COMPUTE_UV:
      case COMPUTE_REVERSE_Y:
        resizeVector(2,coarseColor);
        break;
      default:
      	resizeVector(2,1);
      	break;
      }
      // do not tune spatial block size for VUV
      tune_block_x = type == COMPUTE_VUV ? false : true;
    }

    bool advanceAux(TuneParam &param) const
    {
      if (type != COMPUTE_VUV) return false;
#ifdef SHARED_ATOMIC
#ifdef SWIZZLE
      constexpr int max_swizzle = 4;
      if (param.aux.x < max_swizzle) {
        param.aux.x++;
	return true;
      } else {
        param.aux.x = 1;
	return false;
      }
#else
      if (param.aux.y < 4) {
        param.aux.y *= 2;
	return true;
      } else {
        param.aux.y = 1;
	return false;
      }
#endif
#else
      return false;
#endif
    }

    bool advanceSharedBytes(TuneParam &param) const {
      return type == COMPUTE_VUV ? false : Tunable::advanceSharedBytes(param);
    }

    bool advanceTuneParam(TuneParam &param) const {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) return Tunable::advanceTuneParam(param);
      else return false;
    }

    void initTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::initTuneParam(param);
      if (type == COMPUTE_VUV) {
#ifdef SHARED_ATOMIC
      	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
      	param.grid.x = 2*arg.coarseVolumeCB;
      	param.aux.x = 1; // swizzle factor
      	param.aux.y = 1; // aggregates per block
#endif
      }
    }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const
    {
      TunableVectorYZ::defaultTuneParam(param);
      if (type == COMPUTE_VUV) {
#ifdef SHARED_ATOMIC
      	param.block.x = arg.fineVolumeCB/(2*arg.coarseVolumeCB); // checker-boarded block size
      	param.grid.x = 2*arg.coarseVolumeCB;
      	param.aux.x = 1; // swizzle factor
      	param.aux.y = 4; // aggregates per block
#endif
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
      }

      const char *vol_str = (type == COMPUTE_REVERSE_Y || type == COMPUTE_MASS ||
			     type == COMPUTE_CONVERT) ? X.VolString () : meta.VolString();

      if (type == COMPUTE_VUV) {
      	strcat(Aux,meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU," : ",CPU,");
      	strcat(Aux,"coarse_vol=");
      	strcat(Aux,X.VolString());
      } else {
        strcat(Aux,meta.Location()==QUDA_CUDA_FIELD_LOCATION ? ",GPU" : ",CPU");
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

    typedef CalculateStaggeredYArg<Float,coarseSpin,coarseGauge,coarseGaugeAtomic,fineGauge,Ftmp,Vt> Arg;
    Arg arg(Y, X, Y_atomic, X_atomic, UV, G, V, mass, x_size, xc_size, geo_bs, spin_bs, fine_to_coarse, coarse_to_fine, bidirectional_links);
    CalculateStaggeredY<Float, fineColor, coarseSpin, coarseColor, Arg> y(arg, v, Y_, X_);

    QudaFieldLocation location = checkLocation(Y_, X_, v);
    printfQuda("Running link coarsening on the %s\n", location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU");

    // do exchange of null-space vectors
    const int nFace = 1;
    v.exchangeGhost(QUDA_INVALID_PARITY, nFace, 0);
    arg.V.resetGhost(v.Ghost());  // point the accessor to the correct ghost buffer
    LatticeField::bufferIndex = (1 - LatticeField::bufferIndex); // update ghost bufferIndex for next exchange

    printfQuda("V2 = %e\n", arg.V.norm2());

    // work out what to set the scales to
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
