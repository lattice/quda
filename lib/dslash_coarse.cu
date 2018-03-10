#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#if __COMPUTE_CAPABILITY__ >= 300
#include <generics/shfl.h>
#endif

namespace quda {

#ifdef GPU_MULTIGRID

  enum DslashType {
    DSLASH_INTERIOR,
    DSLASH_EXTERIOR,
    DSLASH_FULL
  };

  template <typename Float, int coarseSpin, int coarseColor, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  struct DslashCoarseArg {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;

    F out;
    const F inA;
    const F inB;
    const G Y;
    const G X;
    const Float kappa;
    const int parity; // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int nFace;  // hard code to 1 for now
    const int_fastdiv X0h; // X[0]/2
    const int_fastdiv dim[5];   // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;

    inline DslashCoarseArg(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			   const GaugeField &Y, const GaugeField &X, Float kappa, int parity)
      : out(const_cast<ColorSpinorField&>(out)), inA(const_cast<ColorSpinorField&>(inA)),
	inB(const_cast<ColorSpinorField&>(inB)), Y(const_cast<GaugeField&>(Y)),
	X(const_cast<GaugeField&>(X)), kappa(kappa), parity(parity),
	nParity(out.SiteSubset()), nFace(1), X0h( ((3-nParity) * out.X(0)) /2),
	dim{ (3-nParity) * out.X(0), out.X(1), out.X(2), out.X(3), out.Ndim() == 5 ? out.X(4) : 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(out.VolumeCB()/dim[4])
    {  }
  };

  /**
     @brief Helper function to determine if should halo computation
  */
  template <DslashType type>
  static __host__ __device__ bool doHalo() {
    switch(type) {
    case DSLASH_EXTERIOR:
    case DSLASH_FULL:
      return true;
    default:
      return false;
    }
  }

  /**
     @brief Helper function to determine if should interior computation
  */
  template <DslashType type>
  static __host__ __device__ bool doBulk() {
    switch(type) {
    case DSLASH_INTERIOR:
    case DSLASH_FULL:
      return true;
    default:
      return false;
    }
  }

  /**
     Compute the 4-d spatial index from the checkerboarded 1-d index at parity parity

     @param x Computed spatial index
     @param cb_index 1-d checkerboarded index
     @param X Full lattice dimensions
     @param parity Site parity
   */
  template <typename I>
  static __device__ __host__ inline void getCoordsCB(int x[], int cb_index, const I X[], const I X0h, int parity) {
    //x[3] = cb_index/(X[2]*X[1]*X[0]/2);
    //x[2] = (cb_index/(X[1]*X[0]/2)) % X[2];
    //x[1] = (cb_index/(X[0]/2)) % X[1];
    //x[0] = 2*(cb_index%(X[0]/2)) + ((x[3]+x[2]+x[1]+parity)&1);

    int za = (cb_index / X0h);
    int zb =  (za / X[1]);
    x[1] = (za - zb * X[1]);
    x[3] = (zb / X[2]);
    x[2] = (zb - x[3] * X[2]);
    int x1odd = (x[1] + x[2] + x[3] + parity) & 1;
    x[0] = (2 * cb_index + x1odd  - za * X[0]);
    return;
  }

  /**
     Applies the coarse dslash on a given parity and checkerboard site index

     @param out The result - kappa * Dslash in
     @param Y The coarse gauge field
     @param kappa Kappa value
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  extern __shared__ float s[];
  template <typename Float, int nDim, int Ns, int Nc, int Mc, int color_stride, int dim_stride, int thread_dir, int thread_dim, bool dagger, DslashType type, typename Arg>
  __device__ __host__ inline void applyDslash(complex<Float> out[], Arg &arg, int x_cb, int src_idx, int parity, int s_row, int color_block, int color_offset) {
    const int their_spinor_parity = (arg.nParity == 2) ? 1-parity : 0;

    int coord[5];
    getCoordsCB(coord, x_cb, arg.dim, arg.X0h, parity);
    coord[4] = src_idx;

#ifdef __CUDA_ARCH__
    complex<Float> *shared_sum = (complex<Float>*)s;
    if (!thread_dir) {
#endif

      //Forward gather - compute fwd offset for spinor fetch
#pragma unroll
      for(int d = thread_dim; d < nDim; d+=dim_stride) // loop over dimension
      {
	const int fwd_idx = linkIndexP1(coord, arg.dim, d);

	if ( arg.commDim[d] && (coord[d] + arg.nFace >= arg.dim[d]) ) {
	  if (doHalo<type>()) {
	    int ghost_idx = ghostFaceIndex<1>(coord, arg.dim, d, arg.nFace);

#pragma unroll
	    for(int color_local = 0; color_local < Mc; color_local++) { //Color row
	      int c_row = color_block + color_local; // global color index
	      int row = s_row*Nc + c_row;
#pragma unroll
	      for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
#pragma unroll
		for(int c_col = 0; c_col < Nc; c_col+=color_stride) { //Color column
		  int col = s_col*Nc + c_col + color_offset;
		  if (!dagger)
		    out[color_local] += arg.Y(d+4, parity, x_cb, row, col)
		      * arg.inA.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
		  else
		    out[color_local] += arg.Y(d, parity, x_cb, row, col)
		      * arg.inA.Ghost(d, 1, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
		}
	      }
	    }
	  }
	} else if (doBulk<type>()) {
#pragma unroll
	  for(int color_local = 0; color_local < Mc; color_local++) { //Color row
	    int c_row = color_block + color_local; // global color index
	    int row = s_row*Nc + c_row;
#pragma unroll
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
#pragma unroll
	      for(int c_col = 0; c_col < Nc; c_col+=color_stride) { //Color column
		int col = s_col*Nc + c_col + color_offset;
		if (!dagger)
		  out[color_local] += arg.Y(d+4, parity, x_cb, row, col)
		    * arg.inA(their_spinor_parity, fwd_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
		else
		  out[color_local] += arg.Y(d, parity, x_cb, row, col)
		    * arg.inA(their_spinor_parity, fwd_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
	      }
	    }
	  }
	}

      } // nDim

#if defined(__CUDA_ARCH__)
      if (thread_dim > 0) { // only need to write to shared memory if not master thread
#pragma unroll
	for (int color_local=0; color_local < Mc; color_local++) {
	  shared_sum[((color_local * blockDim.z + threadIdx.z )*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x] = out[color_local];
	}
      }
#endif

#ifdef __CUDA_ARCH__
    } else {
#endif

      //Backward gather - compute back offset for spinor and gauge fetch
#pragma unroll
      for(int d = thread_dim; d < nDim; d+=dim_stride)
	{
	const int back_idx = linkIndexM1(coord, arg.dim, d);
	const int gauge_idx = back_idx;
	if ( arg.commDim[d] && (coord[d] - arg.nFace < 0) ) {
	  if (doHalo<type>()) {
	    const int ghost_idx = ghostFaceIndex<0>(coord, arg.dim, d, arg.nFace);
#pragma unroll
	    for (int color_local=0; color_local<Mc; color_local++) {
	      int c_row = color_block + color_local;
	      int row = s_row*Nc + c_row;
#pragma unroll
	      for (int s_col=0; s_col<Ns; s_col++)
#pragma unroll
		for (int c_col=0; c_col<Nc; c_col+=color_stride) {
		  int col = s_col*Nc + c_col + color_offset;
		  if (!dagger)
		    out[color_local] += conj(arg.Y.Ghost(d, 1-parity, ghost_idx, col, row))
		      * arg.inA.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
		  else
		    out[color_local] += conj(arg.Y.Ghost(d+4, 1-parity, ghost_idx, col, row))
		      * arg.inA.Ghost(d, 0, their_spinor_parity, ghost_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
		}
	    }
	  }
	} else if (doBulk<type>()) {
#pragma unroll
	  for(int color_local = 0; color_local < Mc; color_local++) {
	    int c_row = color_block + color_local;
	    int row = s_row*Nc + c_row;
#pragma unroll
	    for(int s_col = 0; s_col < Ns; s_col++)
#pragma unroll
	      for(int c_col = 0; c_col < Nc; c_col+=color_stride) {
		int col = s_col*Nc + c_col + color_offset;
		if (!dagger)
		  out[color_local] += conj(arg.Y(d, 1-parity, gauge_idx, col, row))
		    * arg.inA(their_spinor_parity, back_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
		else
		  out[color_local] += conj(arg.Y(d+4, 1-parity, gauge_idx, col, row))
		    * arg.inA(their_spinor_parity, back_idx + src_idx*arg.volumeCB, s_col, c_col+color_offset);
	      }
	  }
	}

      } //nDim

#if defined(__CUDA_ARCH__)

#pragma unroll
      for (int color_local=0; color_local < Mc; color_local++) {
	shared_sum[ ((color_local * blockDim.z + threadIdx.z )*blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x] = out[color_local];
      }

    } // forwards / backwards thread split
#endif

#ifdef __CUDA_ARCH__ // CUDA path has to recombine the foward and backward results
    __syncthreads();

    // (colorspin * dim_stride + dim * 2 + dir)
    if (thread_dim == 0 && thread_dir == 0) {

      // full split over dimension and direction
#pragma unroll
      for (int d=1; d<dim_stride; d++) { // get remaining forward fathers (if any)
	// 4-way 1,2,3  (stride = 4)
	// 2-way 1      (stride = 2)
#pragma unroll
	for (int color_local=0; color_local < Mc; color_local++) {
	  out[color_local] +=
	    shared_sum[(((color_local*blockDim.z/(2*dim_stride) + threadIdx.z/(2*dim_stride)) * 2 * dim_stride + d * 2 + 0)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x];
	}
      }

#pragma unroll
      for (int d=0; d<dim_stride; d++) { // get all backward gathers
#pragma unroll
	for (int color_local=0; color_local < Mc; color_local++) {
	  out[color_local] +=
	    shared_sum[(((color_local*blockDim.z/(2*dim_stride) + threadIdx.z/(2*dim_stride)) * 2 * dim_stride + d * 2 + 1)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x];
	}
      }

      // apply kappa
#pragma unroll
      for (int color_local=0; color_local<Mc; color_local++) out[color_local] *= -arg.kappa;

    }

#else // !__CUDA_ARCH__
    for (int color_local=0; color_local<Mc; color_local++) out[color_local] *= -arg.kappa;
#endif

    }

  /**
     Applies the coarse clover matrix on a given parity and
     checkerboard site index

     @param out The result out += X * in
     @param X The coarse clover field
     @param in The input field
     @param parity The site parity
     @param x_cb The checkerboarded site index
   */
  template <typename Float, int Ns, int Nc, int Mc, int color_stride, bool dagger, typename Arg>
  __device__ __host__ inline void applyClover(complex<Float> out[], Arg &arg, int x_cb, int src_idx, int parity, int s, int color_block, int color_offset) {
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;

    // M is number of colors per thread
#pragma unroll
    for(int color_local = 0; color_local < Mc; color_local++) {//Color out
      int c = color_block + color_local; // global color index
      int row = s*Nc + c;
#pragma unroll
      for (int s_col = 0; s_col < Ns; s_col++) //Spin in
#pragma unroll
	for (int c_col = 0; c_col < Nc; c_col+=color_stride) { //Color in
	  //Factor of kappa and diagonal addition now incorporated in X
	  int col = s_col*Nc + c_col + color_offset;
	  if (!dagger) {
	    out[color_local] += arg.X(0, parity, x_cb, row, col)
	      * arg.inB(spinor_parity, x_cb+src_idx*arg.volumeCB, s_col, c_col+color_offset);
	  } else {
	    out[color_local] += conj(arg.X(0, parity, x_cb, col, row))
	      * arg.inB(spinor_parity, x_cb+src_idx*arg.volumeCB, s_col, c_col+color_offset);
	  }
	}
    }

  }

  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, int nDim, int Ns, int Nc, int Mc, int color_stride, int dim_thread_split,
	    bool dslash, bool clover, bool dagger, DslashType type, int dir, int dim, typename Arg>
  __device__ __host__ inline void coarseDslash(Arg &arg, int x_cb, int src_idx, int parity, int s, int color_block, int color_offset)
  {
    complex <Float> out[Mc];
#pragma unroll
    for (int c=0; c<Mc; c++) out[c] = 0.0;
    if (dslash) applyDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dir,dim,dagger,type>(out, arg, x_cb, src_idx, parity, s, color_block, color_offset);
    if (doBulk<type>() && clover && dir==0 && dim==0) applyClover<Float,Ns,Nc,Mc,color_stride,dagger>(out, arg, x_cb, src_idx, parity, s, color_block, color_offset);

    if (dir==0 && dim==0) {
      const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;
#pragma unroll
      for (int color_local=0; color_local<Mc; color_local++) {
#if __CUDA_ARCH__ >= 300
	// reduce down to the first group of column-split threads
	constexpr int warp_size = 32; // FIXME - this is buggy when x-dim * color_stride < 32
#pragma unroll
	for (int offset = warp_size/2; offset >= warp_size/color_stride; offset /= 2)
#if (__CUDACC_VER_MAJOR__ >= 9)
	  out[color_local] += __shfl_down_sync(WARP_CONVERGED, out[color_local], offset);
#else
	  out[color_local] += __shfl_down(out[color_local], offset);
#endif

#endif
	int c = color_block + color_local; // global color index
	if (color_offset == 0) {
	  // if not halo we just store, else we accumulate
	  if (doBulk<type>()) arg.out(my_spinor_parity, x_cb+src_idx*arg.volumeCB, s, c) = out[color_local];
	  else arg.out(my_spinor_parity, x_cb+src_idx*arg.volumeCB, s, c) += out[color_local];
	}
      }
    }
  }

  // CPU kernel for applying the coarse Dslash to a vector
  template <typename Float, int nDim, int Ns, int Nc, int Mc, bool dslash, bool clover, bool dagger, DslashType type, typename Arg>
  void coarseDslash(Arg arg)
  {
    // the fine-grain parameters mean nothing for CPU variant
    const int color_stride = 1;
    const int color_offset = 0;
    const int dim_thread_split = 1;
    const int dir = 0;
    const int dim = 0;

    for (int parity= 0; parity < arg.nParity; parity++) {
      // for full fields then set parity from loop else use arg setting
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int src_idx = 0; src_idx < arg.dim[4]; src_idx++) {
	//#pragma omp parallel for
	for(int x_cb = 0; x_cb < arg.volumeCB; x_cb++) { // 4-d volume
	  for (int s=0; s<2; s++) {
	    for (int color_block=0; color_block<Nc; color_block+=Mc) { // Mc=Nc means all colors in a thread
	      coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,dir,dim>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
	    }
	  }
	} // 4-d volumeCB
      } // src index
    } // parity

  }

  // GPU Kernel for applying the coarse Dslash to a vector
  template <typename Float, int nDim, int Ns, int Nc, int Mc, int color_stride, int dim_thread_split, bool dslash, bool clover, bool dagger, DslashType type, typename Arg>
  __global__ void coarseDslashKernel(Arg arg)
  {
    constexpr int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int vector_site_width = warp_size / color_stride;

    int x_cb = blockIdx.x*(blockDim.x/color_stride) + warp_id*(warp_size/color_stride) + lane_id % vector_site_width;

    const int color_offset = lane_id / vector_site_width;

    // for full fields set parity from y thread index else use arg setting
#if 0  // disable multi-src since this has a measurable impact on single src performance
    int paritySrc = blockDim.y*blockIdx.y + threadIdx.y;
    if (paritySrc >= arg.nParity * arg.dim[4]) return;
    const int src_idx = (arg.nParity == 2) ? paritySrc / 2 : paritySrc; // maybe want to swap order or source and parity for improved locality of same parity
    const int parity = (arg.nParity == 2) ? paritySrc % 2 : arg.parity;
#else
    const int src_idx = 0;
    const int parity = (arg.nParity == 2) ? blockDim.y*blockIdx.y + threadIdx.y : arg.parity;
#endif

    // z thread dimension is (( s*(Nc/Mc) + color_block )*dim_thread_split + dim)*2 + dir
    int sMd = blockDim.z*blockIdx.z + threadIdx.z;
    int dir = sMd & 1;
    int sMdim = sMd >> 1;
    int dim = sMdim % dim_thread_split;
    int sM = sMdim / dim_thread_split;
    int s = sM / (Nc/Mc);
    int color_block = (sM % (Nc/Mc)) * Mc;

    if (x_cb >= arg.volumeCB) return;

    if (dir == 0) {
      if (dim == 0)      coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,0,0>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 1) coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,0,1>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 2) coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,0,2>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 3) coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,0,3>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
    } else if (dir == 1) {
      if (dim == 0)      coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,1,0>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 1) coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,1,1>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 2) coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,1,2>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 3) coarseDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dagger,type,1,3>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
    }
  }

  template <typename Float, int nDim, int Ns, int Nc, int Mc, bool dslash, bool clover, bool dagger, DslashType type>
  class DslashCoarse : public Tunable {

  protected:
    ColorSpinorField &out;
    const ColorSpinorField &inA;
    const ColorSpinorField &inB;
    const GaugeField &Y;
    const GaugeField &X;
    const double kappa;
    const int parity;
    const int nParity;
    const int nSrc;

#ifdef EIGHT_WAY_WARP_SPLIT
    const int max_color_col_stride = 8;
#else
    const int max_color_col_stride = 4;
#endif
    mutable int color_col_stride;
    mutable int dim_threads;
    char *saveOut;

    long long flops() const
    {
      return ((dslash*2*nDim+clover*1)*(8*Ns*Nc*Ns*Nc)-2*Ns*Nc)*nParity*(long long)out.VolumeCB();
    }
    long long bytes() const
    {
     return (dslash||clover) * out.Bytes() + dslash*8*inA.Bytes() + clover*inB.Bytes() +
       nSrc*nParity*(dslash*Y.Bytes()*Y.VolumeCB()/(2*Y.Stride()) + clover*X.Bytes()/2);
    }
    unsigned int sharedBytesPerThread() const { return (sizeof(complex<Float>) * Mc); }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions
    bool tuneAuxDim() const { return true; } // Do tune the aux dimensions
    unsigned int minThreads() const { return color_col_stride * X.VolumeCB(); } // 4-d volume since this x threads only
    unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock / (dim_threads * 2 * nParity); }

    bool advanceBlockDim(TuneParam &param) const
    {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool ret = Tunable::advanceBlockDim(param);
      param.block.y = block.y; param.block.z = block.z;
      param.grid.y = grid.y; param.grid.z = grid.z;

      if (ret) { // we advanced the block.x so we're done
	return true;
      } else { // block.x (spacetime) was reset

	if (param.block.y < (unsigned int)(nParity * nSrc)) { // advance parity / 5th dimension
	  param.block.y++;
	  param.grid.y = (nParity * nSrc + param.block.y - 1) / param.block.y;
	  return true;
	} else {
	  // reset parity / 5th dimension
	  param.block.y = 1;
	  param.grid.y = nParity * nSrc;

	  // let's try to advance spin/block-color
	  while(param.block.z <= (unsigned int)(dim_threads * 2 * 2 * (Nc/Mc))) {
	    param.block.z+=dim_threads * 2;
	    if ( (dim_threads*2*2*(Nc/Mc)) % param.block.z == 0) {
	      param.grid.z = (dim_threads * 2 * 2 * (Nc/Mc)) / param.block.z;
	      break;
	    }
	  }

	  // we can advance spin/block-color since this is valid
	  if (param.block.z <= (unsigned int)(dim_threads * 2 * 2 * (Nc/Mc)) &&
	      param.block.z <= (unsigned int)deviceProp.maxThreadsDim[2] ) { //
	    return true;
	  } else { // we have run off the end so let's reset
	    param.block.z = dim_threads * 2;
	    param.grid.z = 2 * (Nc/Mc);
	    return false;
	  }
        }
      }
    }

    // FIXME: understand why this leads to slower perf and variable correctness
    //int blockStep() const { return deviceProp.warpSize/4; }
    //int blockMin() const { return deviceProp.warpSize/4; }

    // Experimental autotuning of the color column stride
    bool advanceAux(TuneParam &param) const
    {

#if __COMPUTE_CAPABILITY__ >= 300
      // we can only split the dot product on Kepler and later since we need the __shfl instruction
      if (2*param.aux.x <= max_color_col_stride && Nc % (2*param.aux.x) == 0 &&
	  param.block.x % deviceProp.warpSize == 0) {
	// An x-dimension block size that is not a multiple of the
	// warp size is incompatible with splitting the dot product
	// across the warp so we must skip this

	param.aux.x *= 2; // safe to advance
	color_col_stride = param.aux.x;

	// recompute grid size since minThreads() has now been updated
	param.grid.x = (minThreads()+param.block.x-1)/param.block.x;

	// check this grid size is valid before returning
	if (param.grid.x < (unsigned int)deviceProp.maxGridSize[0]) return true;
      }
#endif

      // reset color column stride if too large or not divisible
      param.aux.x = 1;
      color_col_stride = param.aux.x;

      // recompute grid size since minThreads() has now been updated
      param.grid.x = (minThreads()+param.block.x-1)/param.block.x;

      if (2*param.aux.y <= nDim) {
	param.aux.y *= 2;
	dim_threads = param.aux.y;

	// need to reset z-block/grid size/shared_bytes since dim_threads has changed
	param.block.z = dim_threads * 2;
	param.grid.z = 2* (Nc / Mc);

	param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z > sharedBytesPerBlock(param) ?
	  sharedBytesPerThread()*param.block.x*param.block.y*param.block.z : sharedBytesPerBlock(param);

	return true;
      } else {
	param.aux.y = 1;
	dim_threads = param.aux.y;

	// need to reset z-block/grid size/shared_bytes since
	// dim_threads has changed.  Strictly speaking this isn't needed
	// since this is the outer dimension to tune, but would be
	// needed if we added an aux.z tuning dimension
	param.block.z = dim_threads * 2;
	param.grid.z = 2* (Nc / Mc);

	param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z > sharedBytesPerBlock(param) ?
	  sharedBytesPerThread()*param.block.x*param.block.y*param.block.z : sharedBytesPerBlock(param);

	return false;
      }
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      param.aux = make_int4(1,1,1,1);
      color_col_stride = param.aux.x;
      dim_threads = param.aux.y;

      Tunable::initTuneParam(param);
      param.block.y = 1;
      param.grid.y = nParity * nSrc;
      param.block.z = dim_threads * 2;
      param.grid.z = 2*(Nc/Mc);
      param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z > sharedBytesPerBlock(param) ?
	sharedBytesPerThread()*param.block.x*param.block.y*param.block.z : sharedBytesPerBlock(param);
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      param.aux = make_int4(1,1,1,1);
      color_col_stride = param.aux.x;
      dim_threads = param.aux.y;

      Tunable::defaultTuneParam(param);
      // ensure that the default x block size is divisible by the warpSize
      param.block.x = deviceProp.warpSize;
      param.grid.x = (minThreads()+param.block.x-1)/param.block.x;
      param.block.y = 1;
      param.grid.y = nParity * nSrc;
      param.block.z = dim_threads * 2;
      param.grid.z = 2*(Nc/Mc);
      param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z > sharedBytesPerBlock(param) ?
	sharedBytesPerThread()*param.block.x*param.block.y*param.block.z : sharedBytesPerBlock(param);
    }

  public:
    inline DslashCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			const GaugeField &Y, const GaugeField &X, double kappa, int parity, MemoryLocation *halo_location)
      : out(out), inA(inA), inB(inB), Y(Y), X(X), kappa(kappa), parity(parity),
      nParity(out.SiteSubset()), nSrc(out.Ndim()==5 ? out.X(4) : 1)
    {
      strcpy(aux, out.AuxString());
      strcat(aux, comm_dim_partitioned_string());

      // record the location of where each pack buffer is in [2*dim+dir] ordering
      // 0 - no packing
      // 1 - pack to local GPU memory
      // 2 - pack to local mapped CPU memory
      // 3 - pack to remote mapped GPU memory
      switch(type) {
      case DSLASH_INTERIOR: strcat(aux,",interior"); break;
      case DSLASH_EXTERIOR: strcat(aux,",exterior"); break;
      case DSLASH_FULL:     strcat(aux,",full"); break;
      }

      if (doHalo<type>()) {
	char label[15] = ",halo=";
	for (int dim=0; dim<4; dim++) {
	  for (int dir=0; dir<2; dir++) {
	    label[2*dim+dir+6] = !comm_dim_partitioned(dim) ? '0' : halo_location[2*dim+dir] == Device ? '1' : halo_location[2*dim+dir] == Host ? '2' : '3';
	  }
	}
	label[14] = '\0';
	strcat(aux,label);
      }
    }
    virtual ~DslashCoarse() { }

    inline void apply(const cudaStream_t &stream) {

      if (out.Location() == QUDA_CPU_FIELD_LOCATION) {

	if (out.FieldOrder() != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER || Y.FieldOrder() != QUDA_QDP_GAUGE_ORDER)
	  errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", inA.FieldOrder(), Y.FieldOrder());

	DslashCoarseArg<Float,Ns,Nc,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,QUDA_QDP_GAUGE_ORDER> arg(out, inA, inB, Y, X, (Float)kappa, parity);
	coarseDslash<Float,nDim,Ns,Nc,Mc,dslash,clover,dagger,type>(arg);
      } else {

        const TuneParam &tp = tuneLaunch(*this, getTuning(), QUDA_VERBOSE /*getVerbosity()*/);

	if (out.FieldOrder() != QUDA_FLOAT2_FIELD_ORDER || Y.FieldOrder() != QUDA_FLOAT2_GAUGE_ORDER)
	  errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", inA.FieldOrder(), Y.FieldOrder());

	DslashCoarseArg<Float,Ns,Nc,QUDA_FLOAT2_FIELD_ORDER,QUDA_FLOAT2_GAUGE_ORDER> arg(out, inA, inB, Y, X, (Float)kappa, parity);

	switch (tp.aux.y) { // dimension gather parallelisation
	case 1:
	  switch (tp.aux.x) { // this is color_col_stride
	  case 1:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,1,1,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 2:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,2,1,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 4:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,4,1,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
#ifdef EIGHT_WAY_WARP_SPLIT
	  case 8:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,8,1,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
#endif
	  default:
	    errorQuda("Color column stride %d not valid", tp.aux.x);
	  }
	  break;
	case 2:
	  switch (tp.aux.x) { // this is color_col_stride
	  case 1:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,1,2,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 2:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,2,2,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 4:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,4,2,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
#ifdef EIGHT_WAY_WARP_SPLIT
	  case 8:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,8,2,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
#endif
	  default:
	    errorQuda("Color column stride %d not valid", tp.aux.x);
	  }
	  break;
	case 4:
	  switch (tp.aux.x) { // this is color_col_stride
	  case 1:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,1,4,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 2:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,2,4,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 4:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,4,4,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
#ifdef EIGHT_WAY_WARP_SPLIT
	  case 8:
	    coarseDslashKernel<Float,nDim,Ns,Nc,Mc,8,4,dslash,clover,dagger,type> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
#endif
	  default:
	    errorQuda("Color column stride %d not valid", tp.aux.x);
	  }
	  break;
	default:
	  errorQuda("Invalid dimension thread splitting %d", tp.aux.y);
	}
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(out.VolString(), typeid(*this).name(), aux);
    }

    void preTune() {
      saveOut = new char[out.Bytes()];
      cudaMemcpy(saveOut, out.V(), out.Bytes(), cudaMemcpyDeviceToHost);
    }

    void postTune()
    {
      cudaMemcpy(out.V(), saveOut, out.Bytes(), cudaMemcpyHostToDevice);
      delete[] saveOut;
    }

  };


  template <typename Float, int coarseColor, int coarseSpin>
  inline void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			  const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash,
			  bool clover, bool dagger, DslashType type, MemoryLocation *halo_location) {

    const int colors_per_thread = 1;
    const int nDim = 4;

    if (dagger) {
      if (dslash) {
	if (clover) {
	  if (type == DSLASH_FULL) {
	    DslashCoarse<Float,nDim,coarseSpin,coarseColor,colors_per_thread,true,true,true,DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity, halo_location);
	    dslash.apply(0);
	  } else { errorQuda("Dslash type %d not instantiated", type); }
	} else {
	  if (type == DSLASH_FULL) {
	    DslashCoarse<Float,nDim,coarseSpin,coarseColor,colors_per_thread,true,false,true,DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity, halo_location);
	    dslash.apply(0);
	  } else { errorQuda("Dslash type %d not instantiated", type); }
	}
      } else {
	if (type == DSLASH_EXTERIOR) errorQuda("Cannot call halo on pure clover kernel");
	if (clover) {
	  DslashCoarse<Float,nDim,coarseSpin,coarseColor,colors_per_thread,false,true,true,DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity, halo_location);
	  dslash.apply(0);
	} else {
	  errorQuda("Unsupported dslash=false clover=false");
	}
      }
    } else {
      if (dslash) {
	if (clover) {
	  if (type == DSLASH_FULL) {
	    DslashCoarse<Float,nDim,coarseSpin,coarseColor,colors_per_thread,true,true,false,DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity, halo_location);
	    dslash.apply(0);
	  } else { errorQuda("Dslash type %d not instantiated", type); }
	} else {
	  if (type == DSLASH_FULL) {
	    DslashCoarse<Float,nDim,coarseSpin,coarseColor,colors_per_thread,true,false,false,DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity, halo_location);
	    dslash.apply(0);
	  } else { errorQuda("Dslash type %d not instantiated", type); }
	}
      } else {
	if (type == DSLASH_EXTERIOR) errorQuda("Cannot call halo on pure clover kernel");
	if (clover) {
	  DslashCoarse<Float,nDim,coarseSpin,coarseColor,colors_per_thread,false,true,false,DSLASH_FULL> dslash(out, inA, inB, Y, X, kappa, parity, halo_location);
	  dslash.apply(0);
	} else {
	  errorQuda("Unsupported dslash=false clover=false");
	}
      }
    }
  }

  // template on the number of coarse colors
  template <typename Float>
  inline void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			  const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash,
			  bool clover, bool dagger, DslashType type, MemoryLocation *halo_location) {

    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch inA = %d, out = %d", inA.FieldOrder(), out.FieldOrder());

    if (inA.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",inA.Nspin());

    if (inA.Ncolor() == 2) {
      ApplyCoarse<Float,2,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
#if 0
    } else if (inA.Ncolor() == 4) {
      ApplyCoarse<Float,4,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 8) {
      ApplyCoarse<Float,8,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 12) {
      ApplyCoarse<Float,12,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 16) {
      ApplyCoarse<Float,16,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 20) {
      ApplyCoarse<Float,20,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
#endif
    } else if (inA.Ncolor() == 24) {
      ApplyCoarse<Float,24,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
#if 0
    } else if (inA.Ncolor() == 28) {
      ApplyCoarse<Float,28,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
#endif
    } else if (inA.Ncolor() == 32) {
      ApplyCoarse<Float,32,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  // this is the Worker pointer that may have issue additional work
  // while we're waiting on communication to finish
  namespace dslash {
    extern Worker* aux_worker;
  }

#endif // GPU_MULTIGRID

  enum DslashCoarsePolicy {
    DSLASH_COARSE_BASIC,          // stage both sends and recvs in host memory using memcpys
    DSLASH_COARSE_ZERO_COPY_PACK, // zero copy write pack buffers
    DSLASH_COARSE_ZERO_COPY_READ, // zero copy read halos in dslash kernel
    DSLASH_COARSE_ZERO_COPY,      // full zero copy
    DSLASH_COARSE_GDR_SEND,       // GDR send
    DSLASH_COARSE_GDR_RECV,       // GDR recv
    DSLASH_COARSE_GDR,             // full GDR
    DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV, // zero copy write and GDR recv
    DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ // GDR send and zero copy read
  };

  struct DslashCoarseLaunch {

    ColorSpinorField &out;
    const ColorSpinorField &inA;
    const ColorSpinorField &inB;
    const GaugeField &Y;
    const GaugeField &X;
    double kappa;
    int parity;
    bool dslash;
    bool clover;
    bool dagger;

    inline DslashCoarseLaunch(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
			      const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover, bool dagger)
      : out(out), inA(inA), inB(inB), Y(Y), X(X), kappa(kappa), parity(parity), dslash(dslash), clover(clover), dagger(dagger) { }

    /**
       @brief Execute the coarse dslash using the given policy
     */
    inline void operator()(DslashCoarsePolicy policy) {
#ifdef GPU_MULTIGRID
      if (inA.V() == out.V()) errorQuda("Aliasing pointers");

      // check all precisions match
      QudaPrecision precision = checkPrecision(out, inA, inB, Y, X);

      // check all locations match
      checkLocation(out, inA, inB, Y, X);

      MemoryLocation pack_destination[2*QUDA_MAX_DIM]; // where we will pack the ghost buffer to
      MemoryLocation halo_location[2*QUDA_MAX_DIM]; // where we load the halo from
      for (int i=0; i<2*QUDA_MAX_DIM; i++) {
	pack_destination[i] = (policy == DSLASH_COARSE_ZERO_COPY_PACK || policy == DSLASH_COARSE_ZERO_COPY ||
			       policy == DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV) ? Host : Device;
	halo_location[i] = (policy == DSLASH_COARSE_ZERO_COPY_READ || policy == DSLASH_COARSE_ZERO_COPY ||
			    policy == DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) ? Host : Device;
      }
      bool gdr_send = (policy == DSLASH_COARSE_GDR_SEND || policy == DSLASH_COARSE_GDR ||
		       policy == DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) ? true : false;
      bool gdr_recv = (policy == DSLASH_COARSE_GDR_RECV || policy == DSLASH_COARSE_GDR ||
		       policy == DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV) ? true : false;

      if (dslash && comm_partitioned()) {
	const int nFace = 1;
	inA.exchangeGhost((QudaParity)(1-parity), nFace, dagger, pack_destination, halo_location, gdr_send, gdr_recv);
      }

      if (dslash::aux_worker) dslash::aux_worker->apply(0);

      if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
	ApplyCoarse<double>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, DSLASH_FULL, halo_location);
	//if (dslash && comm_partitioned()) ApplyCoarse<double>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, true, halo_location);
#else
	errorQuda("Double precision multigrid has not been enabled");
#endif
      } else if (precision == QUDA_SINGLE_PRECISION) {
	ApplyCoarse<float>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, DSLASH_FULL, halo_location);
	//if (dslash && comm_partitioned()) ApplyCoarse<float>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, true, halo_location);
      } else {
	errorQuda("Unsupported precision %d\n", Y.Precision());
      }

      if (dslash && comm_partitioned()) inA.bufferIndex = (1 - inA.bufferIndex);
#else
      errorQuda("Multigrid has not been built");
#endif
    }

  };

  // hooks into tune.cpp variables for policy tuning
  typedef std::map<TuneKey, TuneParam> map;
  const map& getTuneCache();

  void disableProfileCount();
  void enableProfileCount();
  void setPolicyTuning(bool);

  static bool dslash_init = false;
  static std::vector<DslashCoarsePolicy> policy;
  static int config = 0; // 2-bit number used to record the machine config (p2p / gdr) and if this changes we will force a retune

 class DslashCoarsePolicyTune : public Tunable {

   DslashCoarseLaunch &dslash;

   bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
   bool tuneAuxDim() const { return true; } // Do tune the aux dimensions.
   unsigned int sharedBytesPerThread() const { return 0; }
   unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

 public:
   inline DslashCoarsePolicyTune(DslashCoarseLaunch &dslash) : dslash(dslash)
   {
      strcpy(aux,"policy,");
      if (dslash.dslash) strcat(aux,"dslash");
      strcat(aux, dslash.clover ? "clover," : ",");
      strcat(aux,dslash.inA.AuxString());
      strcat(aux,comm_dim_partitioned_string());

      if (!dslash_init) {
	policy.reserve(9);
	static char *dslash_policy_env = getenv("QUDA_ENABLE_DSLASH_COARSE_POLICY");

	if (dslash_policy_env) { // set the policies to tune for explicitly
	  std::stringstream policy_list(dslash_policy_env);

	  int policy_;
	  while (policy_list >> policy_) {
	    DslashCoarsePolicy dslash_policy = static_cast<DslashCoarsePolicy>(policy_);

	    // check this is a valid policy choice
	    if ( (dslash_policy == DSLASH_COARSE_GDR_SEND || dslash_policy == DSLASH_COARSE_GDR_RECV ||
		  dslash_policy == DSLASH_COARSE_GDR || dslash_policy == DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV ||
		  dslash_policy == DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ) && !comm_gdr_enabled() ) {
	      errorQuda("Cannot select a GDR policy %d unless QUDA_ENABLE_GDR is set", dslash_policy);
	    }

	    policy.push_back(static_cast<DslashCoarsePolicy>(policy_));
	    if (policy_list.peek() == ',') policy_list.ignore();
	  }
	} else {
	  policy.push_back(DSLASH_COARSE_BASIC);
	  policy.push_back(DSLASH_COARSE_ZERO_COPY_PACK);
	  policy.push_back(DSLASH_COARSE_ZERO_COPY_READ);
	  policy.push_back(DSLASH_COARSE_ZERO_COPY);
	  if (comm_gdr_enabled()) {
	    policy.push_back(DSLASH_COARSE_GDR_SEND);
	    policy.push_back(DSLASH_COARSE_GDR_RECV);
	    policy.push_back(DSLASH_COARSE_GDR);
	    policy.push_back(DSLASH_COARSE_ZERO_COPY_PACK_GDR_RECV);
	    policy.push_back(DSLASH_COARSE_GDR_SEND_ZERO_COPY_READ);
	  }
	}

	config += comm_peer2peer_enabled_global();
	config += comm_gdr_enabled() * 2;
	dslash_init = true;
      }

      // before we do policy tuning we must ensure the kernel
      // constituents have been tuned since we can't do nested tuning
      if (getTuning() && getTuneCache().find(tuneKey()) == getTuneCache().end()) {
	disableProfileCount();
	for (auto &i : policy) dslash(i);
	enableProfileCount();
	setPolicyTuning(true);
      }
    }

   virtual ~DslashCoarsePolicyTune() { setPolicyTuning(false); }

   inline void apply(const cudaStream_t &stream) {
     TuneParam tp = tuneLaunch(*this, getTuning(), QUDA_DEBUG_VERBOSE /*getVerbosity()*/);

     if (config != tp.aux.y) {
       errorQuda("Machine configuration (P2P/GDR=%d) changed since tunecache was created (P2P/GDR=%d).  Please delete "
		 "this file or set the QUDA_RESOURCE_PATH environment variable to point to a new path.",
		 config, tp.aux.y);
     }

     if (tp.aux.x >= (int)policy.size()) errorQuda("Requested policy that is outside of range");
     dslash(policy[tp.aux.x]);
   }

   int tuningIter() const { return 10; }

   bool advanceAux(TuneParam &param) const
   {
     if ((unsigned)param.aux.x < policy.size()-1) {
       param.aux.x++;
       return true;
     } else {
       param.aux.x = 0;
       return false;
     }
   }

   bool advanceTuneParam(TuneParam &param) const { return advanceAux(param); }

   void initTuneParam(TuneParam &param) const  {
     Tunable::initTuneParam(param);
     param.aux.x = 0; param.aux.y = config; param.aux.z = 0; param.aux.w = 0;
   }

   void defaultTuneParam(TuneParam &param) const  {
     Tunable::defaultTuneParam(param);
     param.aux.x = 0; param.aux.y = config; param.aux.z = 0; param.aux.w = 0;
   }

   TuneKey tuneKey() const {
     return TuneKey(dslash.inA.VolString(), typeid(*this).name(), aux);
   }

   long long flops() const {
     int nDim = 4;
     int Ns = dslash.inA.Nspin();
     int Nc = dslash.inA.Ncolor();
     int nParity = dslash.inA.SiteSubset();
     int volumeCB = dslash.inA.VolumeCB();
     return ((dslash.dslash*2*nDim+dslash.clover*1)*(8*Ns*Nc*Ns*Nc)-2*Ns*Nc)*nParity*volumeCB;
   }

   long long bytes() const {
     int nParity = dslash.inA.SiteSubset();
     return (dslash.dslash||dslash.clover) * dslash.out.Bytes() +
       dslash.dslash*8*dslash.inA.Bytes() + dslash.clover*dslash.inB.Bytes() +
       nParity*(dslash.dslash*dslash.Y.Bytes()*dslash.Y.VolumeCB()/(2*dslash.Y.Stride())
		+ dslash.clover*dslash.X.Bytes()/2);
     // multiply Y by volume / stride to correct for pad
   }
  };


  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //  or
  //out(x) = M^dagger*in = X^dagger*in - kappa*\sum_mu Y^\dagger_{-\mu}(x)in(x+mu) + Y_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
	           const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover, bool dagger) {

    DslashCoarseLaunch Dslash(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger);

    DslashCoarsePolicyTune policy(Dslash);
    policy.apply(0);

  }//ApplyCoarse


} // namespace quda
