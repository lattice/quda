#include <transfer.h>
#include <gauge_field_order.h>
#include <color_spinor_field_order.h>

#ifdef JITIFY
// display debugging info
//#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE        1
#define JITIFY_PRINT_LOG           1
//#define JITIFY_PRINT_PTX           1
//#define JITIFY_PRINT_LAUNCH        1
#include <jitify.hpp>
#endif // JITIFY

#include <dslash_coarse.cuh>

namespace quda {

#ifdef GPU_MULTIGRID

#ifdef JITIFY
  using namespace jitify;
  using namespace jitify::reflection;
  static JitCache kernel_cache;
  static Program program = kernel_cache.program("/home/kate/github/quda/lib/dslash_coarse.cuh", 0, {"-std=c++11"});
#endif

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

    const int max_color_col_stride = 4;
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

#ifdef JITIFY
      strcat(aux,",jitify");
#else
      strcat(aux,",offline");
#endif
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

#ifdef JITIFY
	jitify_error = (tp.block.x*tp.block.y*tp.block.z > (unsigned)deviceProp.maxThreadsPerBlock) ?
	  CUDA_ERROR_LAUNCH_FAILED  : program.kernel("quda::coarseDslashKernel")
	  .instantiate(Type<Float>(),nDim,Ns,Nc,Mc,tp.aux.x,tp.aux.y,dslash,clover,dagger,type,type_of(arg))
	  .configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
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
	  default:
	    errorQuda("Color column stride %d not valid", tp.aux.x);
	  }
	  break;
	default:
	  errorQuda("Invalid dimension thread splitting %d", tp.aux.y);
	}
#endif // JITIFY
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
#if 1
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
#if 1
    } else if (inA.Ncolor() == 28) {
      ApplyCoarse<Float,28,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
#endif
    } else if (inA.Ncolor() == 32) {
      ApplyCoarse<Float,32,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 36) {
      ApplyCoarse<Float,36,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 40) {
      ApplyCoarse<Float,40,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 44) {
      ApplyCoarse<Float,44,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 48) {
      ApplyCoarse<Float,48,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 52) {
      ApplyCoarse<Float,52,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 56) {
      ApplyCoarse<Float,56,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 60) {
      ApplyCoarse<Float,60,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
    } else if (inA.Ncolor() == 64) {
      ApplyCoarse<Float,64,2>(out, inA, inB, Y, X, kappa, parity, dslash, clover, dagger, type, halo_location);
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
      QudaPrecision precision = Precision(out, inA, inB, Y, X);

      // check all locations match
      Location(out, inA, inB, Y, X);

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

      if (dslash && comm_partitioned())
	inA.exchangeGhost((QudaParity)(1-parity), dagger, pack_destination, halo_location, gdr_send, gdr_recv);

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
