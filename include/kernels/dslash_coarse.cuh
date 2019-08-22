#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#include <cub_helper.cuh> // for vector_type
#if (__COMPUTE_CAPABILITY__ >= 300 || __CUDA_ARCH__ >= 300)
#include <generics/shfl.h>
#endif

// splitting the dot-product between threads is buggy with CUDA 7.0
#if __COMPUTE_CAPABILITY__ >= 300 && CUDA_VERSION >= 7050
#define DOT_PRODUCT_SPLIT
#endif

namespace quda {

  enum DslashType {
    DSLASH_INTERIOR,
    DSLASH_EXTERIOR,
    DSLASH_FULL
  };

  template <typename Float, typename yFloat, typename ghostFloat, int coarseSpin, int coarseColor, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  struct DslashCoarseArg {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder,Float,ghostFloat> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,yFloat> G;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder,true,yFloat> GY;

    F out;
    const F inA;
    const F inB;
    const GY Y;
    const GY X;
    const Float kappa;
    const int parity; // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int nFace;  // hard code to 1 for now
    const int_fastdiv X0h; // X[0]/2
    const int_fastdiv dim[5];   // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;

    inline DslashCoarseArg(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
                           const GaugeField &Y, const GaugeField &X, Float kappa, int parity) :
      out(const_cast<ColorSpinorField &>(out)),
      inA(const_cast<ColorSpinorField &>(inA)),
      inB(const_cast<ColorSpinorField &>(inB)),
      Y(const_cast<GaugeField &>(Y)),
      X(const_cast<GaugeField &>(X)),
      kappa(kappa),
      parity(parity),
      nParity(out.SiteSubset()),
      nFace(1),
      X0h(((3 - nParity) * out.X(0)) / 2),
      dim {(3 - nParity) * out.X(0), out.X(1), out.X(2), out.X(3), out.Ndim() == 5 ? out.X(4) : 1},
      commDim {comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB((unsigned int)out.VolumeCB() / dim[4])
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
            int ghost_idx = ghostFaceIndex<1, 5>(coord, arg.dim, d, arg.nFace);

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
            const int ghost_idx = ghostFaceIndex<0, 5>(coord, arg.dim, d, arg.nFace);
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
    vector_type<complex <Float>, Mc> out;
    if (dslash) applyDslash<Float,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dir,dim,dagger,type>(out.data, arg, x_cb, src_idx, parity, s, color_block, color_offset);
    if (doBulk<type>() && clover && dir==0 && dim==0) applyClover<Float,Ns,Nc,Mc,color_stride,dagger>(out.data, arg, x_cb, src_idx, parity, s, color_block, color_offset);

    if (dir==0 && dim==0) {
      const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;
#if __CUDA_ARCH__ >= 300 // only have warp shuffle on Kepler and above

#pragma unroll
      for (int color_local=0; color_local<Mc; color_local++) {
	// reduce down to the first group of column-split threads
	constexpr int warp_size = 32; // FIXME - this is buggy when x-dim * color_stride < 32
#pragma unroll
	for (int offset = warp_size/2; offset >= warp_size/color_stride; offset /= 2)
#define WARP_CONVERGED 0xffffffff // we know warp should be converged here
	  out[color_local] += __shfl_down_sync(WARP_CONVERGED, out[color_local], offset);
      }

#endif // __CUDA_ARCH__ >= 300

#pragma unroll
      for (int color_local=0; color_local<Mc; color_local++) {
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

} // namespace quda
