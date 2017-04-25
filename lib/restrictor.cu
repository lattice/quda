#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <tune_quda.h>
#include <cub/cub.cuh>
#include <typeinfo>
#include <multigrid_helper.cuh>

// enabling CTA swizzling improves spatial locality of MG blocks reducing cache line wastage
#define SWIZZLE

namespace quda {

#ifdef GPU_MULTIGRID

  using namespace quda::colorspinor;

  /** 
      Kernel argument struct
  */
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, QudaFieldOrder order>
  struct RestrictArg {

    FieldOrderCB<Float,coarseSpin,coarseColor,1,order> out;
    const FieldOrderCB<Float,fineSpin,fineColor,1,order> in;
    const FieldOrderCB<Float,fineSpin,fineColor,coarseColor,order> V;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const spin_mapper<fineSpin,coarseSpin> spin_map;
    const int parity; // the parity of the input field (if single parity)
    const int nParity; // number of parities of input fine field
    int swizzle; // swizzle factor for transposing blockIdx.x mapping to coarse grid coordinate

    RestrictArg(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &V,
		const int *fine_to_coarse, const int *coarse_to_fine, int parity)
      : out(out), in(in), V(V), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
	spin_map(), parity(parity), nParity(in.SiteSubset()), swizzle(1)
    { }

    RestrictArg(const RestrictArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,order> &arg) :
      out(arg.out), in(arg.in), V(arg.V), 
      fine_to_coarse(arg.fine_to_coarse), coarse_to_fine(arg.coarse_to_fine), spin_map(),
      parity(arg.parity), nParity(arg.nParity), swizzle(arg.swizzle)
    { }
  };

  /**
     Rotates from the fine-color basis into the coarse-color basis.
  */
  template <typename Float, int fineSpin, int fineColor, int coarseColor, int coarse_colors_per_thread,
	    class FineColor, class Rotator>
  __device__ __host__ inline void rotateCoarseColor(complex<Float> out[fineSpin*coarse_colors_per_thread],
						    const FineColor &in, const Rotator &V,
						    int parity, int nParity, int x_cb, int coarse_color_block) {
    const int spinor_parity = (nParity == 2) ? parity : 0;
    const int v_parity = (V.Nparity() == 2) ? parity : 0;

#pragma unroll
    for (int s=0; s<fineSpin; s++)
#pragma unroll
      for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
	out[s*coarse_colors_per_thread+coarse_color_local] = 0.0;
      }

#pragma unroll
    for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
      int i = coarse_color_block + coarse_color_local;
#pragma unroll
      for (int s=0; s<fineSpin; s++) {

	constexpr int color_unroll = fineColor == 3 ? 3 : 2;

	complex<Float> partial[color_unroll];
#pragma unroll
	for (int k=0; k<color_unroll; k++) partial[k] = 0.0;

#pragma unroll
	for (int j=0; j<fineColor; j+=color_unroll) {
#pragma unroll
	  for (int k=0; k<color_unroll; k++)
	    partial[k] += conj(V(v_parity, x_cb, s, j+k, i)) * in(spinor_parity, x_cb, s, j+k);
	}

#pragma unroll
	for (int k=0; k<color_unroll; k++) out[s*coarse_colors_per_thread + coarse_color_local] += partial[k];
      }
    }

  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int coarse_colors_per_thread, typename Arg>
  void Restrict(Arg arg) {
    for (int parity_coarse=0; parity_coarse<2; parity_coarse++) 
      for (int x_coarse_cb=0; x_coarse_cb<arg.out.VolumeCB(); x_coarse_cb++)
	for (int s=0; s<coarseSpin; s++) 
	  for (int c=0; c<coarseColor; c++)
	    arg.out(parity_coarse, x_coarse_cb, s, c) = 0.0;

    // loop over fine degrees of freedom
    for (int parity=0; parity<arg.nParity; parity++) {
      parity = (arg.nParity == 2) ? parity : arg.parity;

      for (int x_cb=0; x_cb<arg.in.VolumeCB(); x_cb++) {

	int x = parity*arg.in.VolumeCB() + x_cb;
	int x_coarse = arg.fine_to_coarse[x];
	int parity_coarse = (x_coarse >= arg.out.VolumeCB()) ? 1 : 0;
	int x_coarse_cb = x_coarse - parity_coarse*arg.out.VolumeCB();
	
	for (int coarse_color_block=0; coarse_color_block<coarseColor; coarse_color_block+=coarse_colors_per_thread) {
	  complex<Float> tmp[fineSpin*coarse_colors_per_thread];
	  rotateCoarseColor<Float,fineSpin,fineColor,coarseColor,coarse_colors_per_thread>
	    (tmp, arg.in, arg.V, parity, arg.nParity, x_cb, coarse_color_block);

	  for (int s=0; s<fineSpin; s++) {
	    for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
	      int c = coarse_color_block + coarse_color_local;
	      arg.out(parity_coarse,x_coarse_cb,arg.spin_map(s),c) += tmp[s*coarse_colors_per_thread+coarse_color_local];
	    }
	  }

	}
      }
    }

  }

  /**
     struct which acts as a wrapper to a vector of data.
   */
  template <typename scalar, int n>
  struct vector_type {
    scalar data[n];
    __device__ __host__ inline scalar& operator[](int i) { return data[i]; }
    __device__ __host__ inline const scalar& operator[](int i) const { return data[i]; }
    __device__ __host__ inline static constexpr int size() { return n; }
    __device__ __host__ vector_type() { for (int i=0; i<n; i++) data[i] = 0.0; }
  };

  /**
     functor that defines how to do a multi-vector reduction
   */
  template <typename T>
  struct reduce {
    __device__ __host__ inline T operator()(const T &a, const T &b) {
      T sum;
      for (int i=0; i<sum.size(); i++) sum[i] = a[i] + b[i];
      return sum;
    }
  };

  /**
     Here, we ensure that each thread block maps exactly to a
     geometric block.  Each thread block corresponds to one geometric
     block, with number of threads equal to the number of fine grid
     points per aggregate, so each thread represents a fine-grid
     point.  The look up table coarse_to_fine is the mapping to
     each fine grid point.
  */
  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor, int coarse_colors_per_thread,
	    typename Arg, int block_size>
  __global__ void RestrictKernel(Arg arg) {

#ifdef SWIZZLE
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
#else
    int x_coarse = blockIdx.x;
#endif

    int parity_coarse = x_coarse >= arg.out.VolumeCB() ? 1 : 0;
    int x_coarse_cb = x_coarse - parity_coarse*arg.out.VolumeCB();

    // obtain fine index from this look up table
    // since both parities map to the same block, each thread block must do both parities

    // threadIdx.x - fine checkboard offset
    // threadIdx.y - fine parity offset
    // blockIdx.x  - which coarse block are we working on (swizzled to improve cache efficiency)
    // assume that coarse_to_fine look up map is ordered as (coarse-block-id + fine-point-id)
    // and that fine-point-id is parity ordered
    int parity = arg.nParity == 2 ? threadIdx.y : arg.parity;
    int x_fine = arg.coarse_to_fine[ (x_coarse*2 + parity) * blockDim.x + threadIdx.x];
    int x_fine_cb = x_fine - parity*arg.in.VolumeCB();

    int coarse_color_block = (blockDim.z*blockIdx.z + threadIdx.z) * coarse_colors_per_thread;
    if (coarse_color_block >= coarseColor) return;

    complex<Float> tmp[fineSpin*coarse_colors_per_thread];
    rotateCoarseColor<Float,fineSpin,fineColor,coarseColor,coarse_colors_per_thread>
      (tmp, arg.in, arg.V, parity, arg.nParity, x_fine_cb, coarse_color_block);

    typedef vector_type<complex<Float>, coarseSpin*coarse_colors_per_thread> vector;
    vector reduced;

    // first lets coarsen spin locally
    for (int s=0; s<fineSpin; s++) {
      for (int v=0; v<coarse_colors_per_thread; v++) {
	reduced[arg.spin_map(s)*coarse_colors_per_thread+v] += tmp[s*coarse_colors_per_thread+v];
      }
    }

    // now lets coarsen geometry across threads
    if (arg.nParity == 2) {
      typedef cub::BlockReduce<vector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 2> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      reduce<vector> reducer; // reduce functor

      // note this is not safe for blockDim.z > 1
      reduced = BlockReduce(temp_storage).Reduce(reduced, reducer);
    } else {
      typedef cub::BlockReduce<vector, block_size, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      reduce<vector> reducer; // reduce functor

      // note this is not safe for blockDim.z > 1
      reduced = BlockReduce(temp_storage).Reduce(reduced, reducer);
    }

    if (threadIdx.x==0 && threadIdx.y == 0) {
      for (int s=0; s<coarseSpin; s++) {
	for (int coarse_color_local=0; coarse_color_local<coarse_colors_per_thread; coarse_color_local++) {
	  int v = coarse_color_block + coarse_color_local;
	  arg.out(parity_coarse, x_coarse_cb, s, v) = reduced[s*coarse_colors_per_thread+coarse_color_local];
	}
      }
    }
  }

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor,
	    int coarse_colors_per_thread>
  class RestrictLaunch : public Tunable {

  protected:
    ColorSpinorField &out;
    const ColorSpinorField &in;
    const ColorSpinorField &v;
    const int *fine_to_coarse;
    const int *coarse_to_fine;
    const int parity;
    const QudaFieldLocation location;
    const int block_size;
    char vol[TuneKey::volume_n];

    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return in.VolumeCB(); } // fine parity is the block y dimension

  public:
    RestrictLaunch(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		   const int *fine_to_coarse, const int *coarse_to_fine, int parity)
      : out(out), in(in), v(v), fine_to_coarse(fine_to_coarse), coarse_to_fine(coarse_to_fine),
	parity(parity), location(Location(out,in,v)), block_size(in.VolumeCB()/(2*out.VolumeCB()))
    {
      strcpy(vol, out.VolString());
      strcat(vol, ",");
      strcat(vol, in.VolString());

      strcpy(aux, out.AuxString());
      strcat(aux, ",");
      strcat(aux, in.AuxString());
    } // block size is checkerboard fine length / full coarse length
    virtual ~RestrictLaunch() { }

    void apply(const cudaStream_t &stream) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
	if (out.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {
	  RestrictArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_SPACE_SPIN_COLOR_FIELD_ORDER>
	    arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
	  Restrict<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread>(arg);
	} else {
	  errorQuda("Unsupported field order %d", out.FieldOrder());
	}
      } else {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

	if (out.FieldOrder() == QUDA_FLOAT2_FIELD_ORDER) {
	  typedef RestrictArg<Float,fineSpin,fineColor,coarseSpin,coarseColor,QUDA_FLOAT2_FIELD_ORDER> Arg;
	  Arg arg(out, in, v, fine_to_coarse, coarse_to_fine, parity);
	  arg.swizzle = tp.aux.x;

	  if (block_size == 4) {          // for 2x2x2x1 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,4>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 8) {   // for 2x2x2x2 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,8>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 16) {  // for 4x2x2x2 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,16>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 27) {  // for 3x3x3x2 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,27>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 36) {  // for 3x3x2x4 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,36>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 54) {  // for 3x3x3x4 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,54>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 64) {  // for 2x4x4x4 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,64>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 96) {  // for 4x4x3x4 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,96>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 100) {  // for 5x5x2x4 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,100>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 108) {  // for 3x3x3x8 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,108>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 128) { // for 4x4x4x4 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,128>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 144) {  // for 4x4x3x6 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,144>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 192) {  // for 4x4x3x8 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,192>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 200) { // for 5x5x2x8  aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,200>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 256) { // for 4x4x4x8  aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,256>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
#if __COMPUTE_CAPABILITY__ >= 300
	  } else if (block_size == 432) { // for 6x6x6x4 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,432>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	  } else if (block_size == 500) { // 5x5x5x8 aggregates
	    RestrictKernel<Float,fineSpin,fineColor,coarseSpin,coarseColor,coarse_colors_per_thread,Arg,500>
	      <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
#endif
	  } else {
	    errorQuda("Block size %d not instantiated", block_size);
	  }
	} else {
	  errorQuda("Unsupported field order %d", out.FieldOrder());
	}
      }
    }

    // This block tuning tunes for the optimal amount of color
    // splitting between blockDim.z and gridDim.z.  However, enabling
    // blockDim.z > 1 gives incorrect results due to cub reductions
    // being unable to do independent sliced reductions along
    // blockDim.z.  So for now we only split between colors per thread
    // and grid.z.
    bool advanceBlockDim(TuneParam &param) const
    {
      // let's try to advance spin/block-color
      while(param.block.z <= coarseColor/coarse_colors_per_thread) {
	param.block.z++;
	if ( (coarseColor/coarse_colors_per_thread) % param.block.z == 0) {
	  param.grid.z = (coarseColor/coarse_colors_per_thread) / param.block.z;
	  break;
	}
      }

      // we can advance spin/block-color since this is valid
      if (param.block.z <= (coarseColor/coarse_colors_per_thread) ) { //
	return true;
      } else { // we have run off the end so let's reset
	param.block.z = 1;
	param.grid.z = coarseColor/coarse_colors_per_thread;
	return false;
      }
    }

    int tuningIter() const { return 3; }

    bool advanceAux(TuneParam &param) const
    {
#ifdef SWIZZLE
      if (param.aux.x < 2*deviceProp.multiProcessorCount) {
        param.aux.x++;
	return true;
      } else {
        param.aux.x = 1;
	return false;
      }
#else
      return false;
#endif
    }

    // only tune shared memory per thread (disable tuning for block.z for now)
    bool advanceTuneParam(TuneParam &param) const { return advanceSharedBytes(param) || advanceAux(param); }

    TuneKey tuneKey() const { return TuneKey(vol, typeid(*this).name(), aux); }

    void initTuneParam(TuneParam &param) const { defaultTuneParam(param); }

    /** sets default values for when tuning is disabled */
    void defaultTuneParam(TuneParam &param) const {
      param.block = dim3(block_size, in.SiteSubset(), 1);
      param.grid = dim3( (minThreads()+param.block.x-1) / param.block.x, 1, 1);
      param.shared_bytes = 0;

      param.block.z = 1;
      param.grid.z = coarseColor / coarse_colors_per_thread;
      param.aux.x = 1; // swizzle factor
    }

    long long flops() const { return 8 * fineSpin * fineColor * coarseColor * in.SiteSubset()*in.VolumeCB(); }

    long long bytes() const {
      size_t v_bytes = v.Bytes() / (v.SiteSubset() == in.SiteSubset() ? 1 : 2);
      return in.Bytes() + out.Bytes() + v_bytes + in.SiteSubset()*in.VolumeCB()*sizeof(int);
    }

  };

  template <typename Float, int fineSpin, int fineColor, int coarseSpin, int coarseColor>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		const int *fine_to_coarse, const int *coarse_to_fine, int parity) {

    // for fine grids (Nc=3) have more parallelism so can use more coarse strategy
    constexpr int coarse_colors_per_thread = fineColor != 3 ? 2 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;
    //coarseColor >= 8 && coarseColor % 8 == 0 ? 8 : coarseColor >= 4 && coarseColor % 4 == 0 ? 4 : 2;

    RestrictLaunch<Float, fineSpin, fineColor, coarseSpin, coarseColor, coarse_colors_per_thread>
      restrictor(out, in, v, fine_to_coarse, coarse_to_fine, parity);
    restrictor.apply(0);

    if (Location(out, in, v) == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  template <typename Float, int fineSpin>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		int nVec, const int *fine_to_coarse, const int *coarse_to_fine, const int *spin_map, int parity) {

    if (out.Nspin() != 2) errorQuda("Unsupported nSpin %d", out.Nspin());
    const int coarseSpin = 2;

    // first check that the spin_map matches the spin_mapper
    spin_mapper<fineSpin,coarseSpin> mapper;
    for (int s=0; s<fineSpin; s++) 
      if (mapper(s) != spin_map[s]) errorQuda("Spin map does not match spin_mapper");


    // Template over fine color
    if (in.Ncolor() == 3) { // standard QCD
      const int fineColor = 3;
      if (nVec == 2) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,2>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else if (nVec == 4) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,4>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else if (nVec == 24) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else if (nVec == 32) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
	errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (in.Ncolor() == 2) {
      const int fineColor = 2;
      if (nVec == 2) { // these are probably only for debugging only
	Restrict<Float,fineSpin,fineColor,coarseSpin,2>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else if (nVec == 4) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,4>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
	errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (in.Ncolor() == 24) { // to keep compilation under control coarse grids have same or more colors
      const int fineColor = 24;
      if (nVec == 24) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,24>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else if (nVec == 32) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
	errorQuda("Unsupported nVec %d", nVec);
      }
    } else if (in.Ncolor() == 32) {
      const int fineColor = 32;
      if (nVec == 32) {
	Restrict<Float,fineSpin,fineColor,coarseSpin,32>(out, in, v, fine_to_coarse, coarse_to_fine, parity);
      } else {
	errorQuda("Unsupported nVec %d", nVec);
      }
    } else {
      errorQuda("Unsupported nColor %d", in.Ncolor());
    }
  }

  template <typename Float>
  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int *spin_map, int parity) {

    if (in.Nspin() == 4) {
      Restrict<Float,4>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
    } else if (in.Nspin() == 2) {
      Restrict<Float,2>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
#if GPU_STAGGERED_DIRAC
    } else if (in.Nspin() == 1) {
      Restrict<Float,1>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
#endif
    } else {
      errorQuda("Unsupported nSpin %d", in.Nspin());
    }
  }

#endif // GPU_MULTIGRID

  void Restrict(ColorSpinorField &out, const ColorSpinorField &in, const ColorSpinorField &v,
		int Nvec, const int *fine_to_coarse, const int *coarse_to_fine, const int *spin_map, int parity) {

#ifdef GPU_MULTIGRID
    if (out.FieldOrder() != in.FieldOrder() ||	out.FieldOrder() != v.FieldOrder())
      errorQuda("Field orders do not match (out=%d, in=%d, v=%d)",
		out.FieldOrder(), in.FieldOrder(), v.FieldOrder());

    QudaPrecision precision = Precision(out, in, v);

    if (precision == QUDA_DOUBLE_PRECISION) {
#ifdef GPU_MULTIGRID_DOUBLE
      Restrict<double>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
#else
      errorQuda("Double precision multigrid has not been enabled");
#endif
    } else if (precision == QUDA_SINGLE_PRECISION) {
      Restrict<float>(out, in, v, Nvec, fine_to_coarse, coarse_to_fine, spin_map, parity);
    } else {
      errorQuda("Unsupported precision %d", out.Precision());
    }
#else
    errorQuda("Multigrid has not been built");
#endif
  }

} // namespace quda
