#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>
#include <index_helper.cuh>
#if __COMPUTE_CAPABILITY__ >= 300
#include <generics/shfl.h>
#endif

namespace quda {

#ifdef GPU_MULTIGRID

  template <typename Float, typename F, typename G>
  struct CoarseDslashArg {
    F out;
    const F inA;
    const F inB;
    const G Y;
    const G X;
    const Float kappa;
    const int parity; // only use this for single parity fields
    const int nParity; // number of parities we're working on
    const int nFace;  // hard code to 1 for now
    const int dim[5];   // full lattice dimensions
    const int commDim[4]; // whether a given dimension is partitioned or not
    const int volumeCB;

    CoarseDslashArg(F &out, const F &inA, const F &inB, const G &Y, const G &X,
		    Float kappa, int parity, const ColorSpinorField &meta)
      : out(out), inA(inA), inB(inB), Y(Y), X(X), kappa(kappa), parity(parity),
	nParity(meta.SiteSubset()), nFace(1),
	dim{ (3-nParity) * meta.X(0), meta.X(1), meta.X(2), meta.X(3), meta.Ndim() == 5 ? meta.X(4) : 1 },
      commDim{comm_dim_partitioned(0), comm_dim_partitioned(1), comm_dim_partitioned(2), comm_dim_partitioned(3)},
      volumeCB(meta.VolumeCB()/dim[4])
    {  }
  };

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
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc, int color_stride, int dim_stride, int thread_dir, int thread_dim>
  __device__ __host__ inline void applyDslash(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int src_idx, int parity, int s_row, int color_block, int color_offset) {
    const int their_spinor_parity = (arg.nParity == 2) ? (parity+1)&1 : 0;

    int coord[5];
    getCoords(coord, x_cb, arg.dim, parity);
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
		out[color_local] += arg.Y(d+4, parity, x_cb, row, col) * arg.inA.Ghost(d, 1, their_spinor_parity, ghost_idx, s_col, c_col+color_offset);
	      }
	    }
	  }
	} else {
#pragma unroll
	  for(int color_local = 0; color_local < Mc; color_local++) { //Color row
	    int c_row = color_block + color_local; // global color index
	    int row = s_row*Nc + c_row;
#pragma unroll
	    for(int s_col = 0; s_col < Ns; s_col++) { //Spin column
#pragma unroll
	      for(int c_col = 0; c_col < Nc; c_col+=color_stride) { //Color column
		int col = s_col*Nc + c_col + color_offset;
		out[color_local] += arg.Y(d+4, parity, x_cb, row, col) * arg.inA(their_spinor_parity, fwd_idx, s_col, c_col+color_offset);
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
		out[color_local] += conj(arg.Y.Ghost(d, (parity+1)&1, ghost_idx, col, row)) * arg.inA.Ghost(d, 0, their_spinor_parity, ghost_idx, s_col, c_col+color_offset);
	      }
	  }
	} else {
#pragma unroll
	  for(int color_local = 0; color_local < Mc; color_local++) {
	    int c_row = color_block + color_local;
	    int row = s_row*Nc + c_row;
#pragma unroll
	    for(int s_col = 0; s_col < Ns; s_col++)
#pragma unroll
	      for(int c_col = 0; c_col < Nc; c_col+=color_stride) {
		int col = s_col*Nc + c_col + color_offset;
		out[color_local] += conj(arg.Y(d, (parity+1)&1, gauge_idx, col, row)) * arg.inA(their_spinor_parity, back_idx, s_col, c_col+color_offset);
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
  template <typename Float, typename F, typename G, int Ns, int Nc, int Mc, int color_stride>
  __device__ __host__ inline void applyClover(complex<Float> out[], CoarseDslashArg<Float,F,G> &arg, int x_cb, int src_idx, int parity, int s, int color_block, int color_offset) {
    const int spinor_parity = (arg.nParity == 2) ? parity : 0;

    // M is number of colors per thread
#pragma unroll
    for(int color_local = 0; color_local < Mc; color_local++) {//Color out
      int c = color_block + color_local; // global color index
      int row = s*Nc + c;
#pragma unroll
      for(int s_col = 0; s_col < Ns; s_col++) //Spin in
#pragma unroll
	for(int c_col = 0; c_col < Nc; c_col+=color_stride) { //Color in
	  //Factor of kappa and diagonal addition now incorporated in X
	  int col = s_col*Nc + c_col + color_offset;
	  out[color_local] += arg.X(0, parity, x_cb, row, col) * arg.inB(spinor_parity, x_cb+src_idx*arg.volumeCB, s_col, c_col+color_offset);
	}
    }

  }

  //out(x) = M*in = \sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc, int color_stride,
	    int dim_thread_split, bool dslash, bool clover, int dir, int dim>
  __device__ __host__ inline void coarseDslash(CoarseDslashArg<Float,F,G> &arg, int x_cb, int src_idx, int parity, int s, int color_block, int color_offset)
  {
    complex <Float> out[Mc];
#pragma unroll
    for (int c=0; c<Mc; c++) out[c] = 0.0;
    if (dslash) applyDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dir,dim>(out, arg, x_cb, src_idx, parity, s, color_block, color_offset);
    if (clover && dir==0 && dim==0) applyClover<Float,F,G,Ns,Nc,Mc,color_stride>(out, arg, x_cb, src_idx, parity, s, color_block, color_offset);

    if (dir==0 && dim==0) {
      const int my_spinor_parity = (arg.nParity == 2) ? parity : 0;
#pragma unroll
      for (int color_local=0; color_local<Mc; color_local++) {
#if __CUDA_ARCH__ >= 300
	// reduce down to the first group of column-split threads
	const int warp_size = 32; // FIXME - this is buggy when x-dim * color_stride < 32
#pragma unroll
	for (int offset = warp_size/2; offset >= warp_size/color_stride; offset /= 2) out[color_local] += __shfl_down(out[color_local], offset);
#endif
	int c = color_block + color_local; // global color index
	if (color_offset == 0) arg.out(my_spinor_parity, x_cb+src_idx*arg.volumeCB, s, c) = out[color_local];
      }
    }
  }

  // CPU kernel for applying the coarse Dslash to a vector
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc, bool dslash, bool clover>
  void coarseDslash(CoarseDslashArg<Float,F,G> arg)
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
	      coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,dir,dim>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
	    }
	  }
	} // 4-d volumeCB
      } // src index
    } // parity
    
  }

  // GPU Kernel for applying the coarse Dslash to a vector
  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc, int color_stride, int dim_thread_split, bool dslash, bool clover>
  __global__ void coarseDslashKernel(CoarseDslashArg<Float,F,G> arg)
  {
    constexpr int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int vector_site_width = warp_size / color_stride;

    int x_cb = blockIdx.x*(blockDim.x/color_stride) + warp_id*(warp_size/color_stride) + lane_id % vector_site_width;

    const int color_offset = lane_id / vector_site_width;

    // for full fields set parity from y thread index else use arg setting
    int paritySrc = blockDim.y*blockIdx.y + threadIdx.y;
    int src_idx = (arg.nParity == 2) ? paritySrc / 2 : paritySrc; // maybe want to swap order or source and parity for improved locality of same parity
    int parity = (arg.nParity == 2) ? paritySrc % 2 : arg.parity;

    // z thread dimension is (( s*(Nc/Mc) + color_block )*dim_thread_split + dim)*2 + dir
    int sMd = blockDim.z*blockIdx.z + threadIdx.z;
    int dir = sMd & 1;
    int sMdim = sMd >> 1;
    int dim = sMdim % dim_thread_split;
    int sM = sMdim / dim_thread_split;
    int s = sM / (Nc/Mc);
    int color_block = (sM % (Nc/Mc)) * Mc;

    if (x_cb >= arg.volumeCB) return;
    if (paritySrc >= arg.nParity * arg.dim[4]) return;

    if (dir == 0) {
      if (dim == 0)      coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,0,0>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 1) coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,0,1>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 2) coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,0,2>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 3) coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,0,3>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
    } else if (dir == 1) {
      if (dim == 0)      coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,1,0>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 1) coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,1,1>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 2) coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,1,2>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
      else if (dim == 3) coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,color_stride,dim_thread_split,dslash,clover,1,3>(arg, x_cb, src_idx, parity, s, color_block, color_offset);
    }
  }

  template <typename Float, typename F, typename G, int nDim, int Ns, int Nc, int Mc, bool dslash, bool clover>
  class CoarseDslash : public Tunable {

  protected:
    CoarseDslashArg<Float,F,G> &arg;
    const ColorSpinorField &meta;

    const int max_color_col_stride = 4;
    mutable int color_col_stride;
    mutable int dim_threads;

    long long flops() const
    {
      return ((dslash*2*nDim+clover*1)*(8*Ns*Nc*Ns*Nc)-2*Ns*Nc)*arg.nParity*(long long)meta.VolumeCB();
    }
    long long bytes() const
    {
      return (dslash||clover) * arg.out.Bytes() + dslash*8*arg.inA.Bytes() + clover*arg.inB.Bytes() +
	arg.dim[4]*arg.nParity*(dslash*8*arg.Y.Bytes() + clover*arg.X.Bytes());
    }
    unsigned int sharedBytesPerThread() const { return (sizeof(complex<Float>) * Mc); }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions
    unsigned int minThreads() const { return color_col_stride * arg.volumeCB; } // 4-d volume since this x threads only
    unsigned int maxBlockSize() const { return deviceProp.maxThreadsPerBlock / (dim_threads * 2 * arg.nParity); }

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

	if (param.block.y < arg.nParity * arg.dim[4]) { // advance parity / 5th dimension
	  param.block.y++;
	  param.grid.y = (arg.nParity * arg.dim[4] + param.block.y - 1) / param.block.y;
	  return true;
	} else {
	  // reset parity / 5th dimension
	  param.block.y = 1;
	  param.grid.y = arg.nParity * arg.dim[4];

	  // let's try to advance spin/block-color
	  while(param.block.z <= dim_threads * 2 * 2 * (Nc/Mc)) {
	    param.block.z+=dim_threads * 2;
	    if ( (dim_threads*2*2*(Nc/Mc)) % param.block.z == 0) {
	      param.grid.z = (dim_threads * 2 * 2 * (Nc/Mc)) / param.block.z;
	      break;
	    }
	  }

	  // we can advance spin/block-color since this is valid
	  if (param.block.z <= dim_threads * 2 * 2 * (Nc/Mc) && param.block.z <= deviceProp.maxThreadsDim[2] ) { //
	    return true;
	  } else { // we have run off the end so let's reset
	    param.block.z = dim_threads * 2;
	    param.grid.z = 2 * (Nc/Mc);
	    return false;
	  }

	}
      }
    }

    int blockStep() const { return deviceProp.warpSize/4; }
    int blockMin() const { return deviceProp.warpSize/4; }

    // Experimental autotuning of the color column stride
    bool advanceAux(TuneParam &param) const
    {
#if __COMPUTE_CAPABILITY__ >= 300
      // we can only split the dot product on Kepler and later since we need the __shfl instruction
      if (2*param.aux.x <= max_color_col_stride && Nc % (2*param.aux.x) == 0) {
	param.aux.x *= 2; // safe to advance
	color_col_stride = param.aux.x;

	// recompute grid size since minThreads() has now been updated
	param.grid.x = (minThreads()+param.block.x-1)/param.block.x;

	// check this grid size is valid before returning
	if (param.grid.x < deviceProp.maxGridSize[0]) return true;
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
      param.grid.y = arg.nParity * arg.dim[4];
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
      param.block.y = 1;
      param.grid.y = arg.nParity * arg.dim[4];
      param.block.z = dim_threads * 2;
      param.grid.z = 2*(Nc/Mc);
      param.shared_bytes = sharedBytesPerThread()*param.block.x*param.block.y*param.block.z > sharedBytesPerBlock(param) ?
	sharedBytesPerThread()*param.block.x*param.block.y*param.block.z : sharedBytesPerBlock(param);
    }

  public:
    CoarseDslash(CoarseDslashArg<Float,F,G> &arg, const ColorSpinorField &meta)
      : arg(arg), meta(meta) {
      strcpy(aux, meta.AuxString());
#ifdef MULTI_GPU
      char comm[5];
      comm[0] = (arg.commDim[0] ? '1' : '0');
      comm[1] = (arg.commDim[1] ? '1' : '0');
      comm[2] = (arg.commDim[2] ? '1' : '0');
      comm[3] = (arg.commDim[3] ? '1' : '0');
      comm[4] = '\0';
      strcat(aux,",comm=");
      strcat(aux,comm);
#endif
    }
    virtual ~CoarseDslash() { }

    void apply(const cudaStream_t &stream) {
      if (meta.Location() == QUDA_CPU_FIELD_LOCATION) {
	coarseDslash<Float,F,G,nDim,Ns,Nc,Mc,dslash,clover>(arg);
      } else {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

	switch (tp.aux.y) { // dimension gather parallelisation
	case 1:
	  switch (tp.aux.x) { // this is color_col_stride
	  case 1:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,1,1,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 2:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,2,1,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 4:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,4,1,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  default:
	    errorQuda("Color column stride %d not valid", tp.aux.x);
	  }
	  break;
	case 2:
	  switch (tp.aux.x) { // this is color_col_stride
	  case 1:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,1,2,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 2:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,2,2,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 4:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,4,2,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  default:
	    errorQuda("Color column stride %d not valid", tp.aux.x);
	  }
	  break;
	case 4:
	  switch (tp.aux.x) { // this is color_col_stride
	  case 1:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,1,4,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 2:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,2,4,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
	  case 4:
	    coarseDslashKernel<Float,F,G,nDim,Ns,Nc,Mc,4,4,dslash,clover> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	    break;
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
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

  };


  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor,
	    int coarseSpin, QudaFieldLocation location>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover) {
    typedef typename colorspinor::FieldOrderCB<Float,coarseSpin,coarseColor,1,csOrder> F;
    typedef typename gauge::FieldOrder<Float,coarseColor*coarseSpin,coarseSpin,gOrder> G;

    F outAccessor(const_cast<ColorSpinorField&>(out));
    F inAccessorA(const_cast<ColorSpinorField&>(inA));
    F inAccessorB(const_cast<ColorSpinorField&>(inB));
    G yAccessor(const_cast<GaugeField&>(Y));
    G xAccessor(const_cast<GaugeField&>(X));
    CoarseDslashArg<Float,F,G> arg(outAccessor, inAccessorA, inAccessorB, yAccessor, xAccessor, (Float)kappa, parity, inA);

    const int colors_per_thread = 1;
    if (dslash) {
      if (clover) {
	CoarseDslash<Float,F,G,4,coarseSpin,coarseColor,colors_per_thread,true,true> dslash(arg, inA);
	dslash.apply(0);
      } else {
	CoarseDslash<Float,F,G,4,coarseSpin,coarseColor,colors_per_thread,true,false> dslash(arg, inA);
	dslash.apply(0);
      }
    } else {
      if (clover) {
	CoarseDslash<Float,F,G,4,coarseSpin,coarseColor,colors_per_thread,false,true> dslash(arg, inA);
	dslash.apply(0);
      } else {
	errorQuda("Unsupported dslash=false clover=false");
      }
    }
  }

  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder, int coarseColor, int coarseSpin>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover) {
    if (inA.Location() == QUDA_CUDA_FIELD_LOCATION) {
      ApplyCoarse<Float,csOrder,gOrder,coarseColor,coarseSpin,QUDA_CUDA_FIELD_LOCATION>
	(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else {
      ApplyCoarse<Float,csOrder,gOrder,coarseColor,coarseSpin,QUDA_CPU_FIELD_LOCATION>
	(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    }
  }

  // template on the number of coarse colors
  template <typename Float, QudaFieldOrder csOrder, QudaGaugeFieldOrder gOrder>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover) {
    if (inA.Nspin() != 2)
      errorQuda("Unsupported number of coarse spins %d\n",inA.Nspin());

    if (inA.Ncolor() == 2) {
      ApplyCoarse<Float,csOrder,gOrder,2,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
#if 0
    } else if (inA.Ncolor() == 4) {
      ApplyCoarse<Float,csOrder,gOrder,4,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else if (inA.Ncolor() == 8) {
      ApplyCoarse<Float,csOrder,gOrder,8,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else if (inA.Ncolor() == 12) {
      ApplyCoarse<Float,csOrder,gOrder,12,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else if (inA.Ncolor() == 16) {
      ApplyCoarse<Float,csOrder,gOrder,16,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else if (inA.Ncolor() == 20) {
      ApplyCoarse<Float,csOrder,gOrder,20,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
#endif
    } else if (inA.Ncolor() == 24) {
      ApplyCoarse<Float,csOrder,gOrder,24,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
#if 0
    } else if (inA.Ncolor() == 28) {
      ApplyCoarse<Float,csOrder,gOrder,28,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
#endif
    } else if (inA.Ncolor() == 32) {
      ApplyCoarse<Float,csOrder,gOrder,32,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else {
      errorQuda("Unsupported number of coarse dof %d\n", Y.Ncolor());
    }
  }

  template <typename Float>
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover) {

    if (Y.FieldOrder() != X.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() != out.FieldOrder())
      errorQuda("Field order mismatch Y = %d, X = %d", Y.FieldOrder(), X.FieldOrder());

    if (inA.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER && Y.FieldOrder() == QUDA_FLOAT2_GAUGE_ORDER) {
      ApplyCoarse<Float,QUDA_FLOAT2_FIELD_ORDER, QUDA_FLOAT2_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else if (inA.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER && Y.FieldOrder() == QUDA_QDP_GAUGE_ORDER) {
      ApplyCoarse<Float,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER,QUDA_QDP_GAUGE_ORDER>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else {
      errorQuda("Unsupported field order colorspinor=%d gauge=%d combination\n", inA.FieldOrder(), Y.FieldOrder());
    }
  }

#endif // GPU_MULTIGRID

  //Apply the coarse Dirac matrix to a coarse grid vector
  //out(x) = M*in = X*in - kappa*\sum_mu Y_{-\mu}(x)in(x+mu) + Y^\dagger_mu(x-mu)in(x-mu)
  //Uses the kappa normalization for the Wilson operator.
  void ApplyCoarse(ColorSpinorField &out, const ColorSpinorField &inA, const ColorSpinorField &inB,
		   const GaugeField &Y, const GaugeField &X, double kappa, int parity, bool dslash, bool clover) {
#ifdef GPU_MULTIGRID
    if (inA.V() == out.V()) errorQuda("Aliasing pointers");

    if (out.Precision() != inA.Precision() ||
	Y.Precision() != inA.Precision() ||
	X.Precision() != inA.Precision())
      errorQuda("Precision mismatch out=%d inA=%d inB=%d Y=%d X=%d",
		out.Precision(), inA.Precision(), inB.Precision(), Y.Precision(), X.Precision());

    // check all locations match
    Location(out, inA, inB, Y, X);

    int dummy = 0; // ignored
    inA.exchangeGhost((QudaParity)(1-parity), dummy);

    if (Y.Precision() == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      ApplyCoarse<double>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (Y.Precision() == QUDA_SINGLE_PRECISION) {
      ApplyCoarse<float>(out, inA, inB, Y, X, kappa, parity, dslash, clover);
    } else {
      errorQuda("Unsupported precision %d\n", Y.Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif
  }//ApplyCoarse

} // namespace quda
