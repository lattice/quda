#include <gauge_field_order.h>

namespace quda {

  using namespace gauge;
  
  /**
     Kernel argument struct
   */
  template <typename OutOrder, typename InOrder>
  struct CopyGaugeArg {
    OutOrder out;
    const InOrder in;
    int volume;
    int faceVolumeCB[QUDA_MAX_DIM];
    int nDim;
    int geometry;
    CopyGaugeArg(const OutOrder &out, const InOrder &in, int volume, 
		 const int *faceVolumeCB, int nDim, int geometry) 
      : out(out), in(in), volume(volume), nDim(nDim), geometry(geometry) {
      for (int d=0; d<nDim; d++) this->faceVolumeCB[d] = faceVolumeCB[d];
    }
  };

  __host__ __device__ inline constexpr int ct_sqrt(int n, int i = 1){
    return n == i ? n : (i * i < n ? ct_sqrt(n, i + 1) : i);
  }

  __host__ __device__ inline constexpr int Ncolor(int length) { return ct_sqrt(length/2); }

  /**
     Generic CPU gauge reordering and packing 
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  void copyGauge(CopyGaugeArg<OutOrder,InOrder> arg) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.geometry; d++) {
	for (int x=0; x<arg.volume/2; x++) {
#ifdef FINE_GRAINED_ACCESS
	  for (int i=0; i<Ncolor(length); i++)
	    for (int j=0; j<Ncolor(length); j++) {
	      arg.out(d, parity, x, i, j) = arg.in(d, parity, x, i, j);
	    }
#else
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  arg.in.load(in, x, d, parity);
	  for (int i=0; i<length; i++) out[i] = in[i];
	  arg.out.save(out, x, d, parity);
#endif
	}
      }

    }
  }

  /**
     Check whether the field contains Nans
  */
  template <typename Float, int length, typename Arg>
  void checkNan(Arg arg) {  
    typedef typename mapper<Float>::type RegType;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.geometry; d++) {
	for (int x=0; x<arg.volume/2; x++) {
#ifdef FINE_GRAINED_ACCESS
	  for (int i=0; i<Ncolor(length); i++)
	    for (int j=0; j<Ncolor(length); j++) {
              complex<Float> u = arg.in(d, parity, x, i, j);
	      if (isnan(u.real()))
	        errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, 2*(i*Ncolor(length)+j));
	      if (isnan(u.imag()))
		errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, 2*(i*Ncolor(length)+j+1));
	}
#else
	  RegType u[length];
	  arg.in.load(u, x, d, parity);
	  for (int i=0; i<length; i++) 
	    if (isnan(u[i])) 
	      errorQuda("Nan detected at parity=%d, dir=%d, x=%d, i=%d", parity, d, x, i);
#endif
	}
      }

    }
  }

  /** 
      Generic CUDA gauge reordering and packing.  Adopts a similar form as
      the CPU version, using the same inlined functions.
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  __global__ void copyGaugeKernel(CopyGaugeArg<OutOrder,InOrder> arg) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.geometry; d++) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= arg.volume/2) return;

#ifdef FINE_GRAINED_ACCESS
	for (int i=0; i<Ncolor(length); i++)
	  for (int j=0; j<Ncolor(length); j++)
	    arg.out(d, parity, x, i, j) = arg.in(d, parity, x, i, j);
#else
	RegTypeIn in[length];
	RegTypeOut out[length];
	arg.in.load(in, x, d, parity);
	for (int i=0; i<length; i++) out[i] = in[i];
	arg.out.save(out, x, d, parity);
#endif
      }
    }
  }

  /**
     Generic CPU gauge ghost reordering and packing 
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void copyGhost(CopyGaugeArg<OutOrder,InOrder> arg) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {

      for (int d=0; d<arg.nDim; d++) {
	for (int x=0; x<arg.faceVolumeCB[d]; x++) {
#ifdef FINE_GRAINED_ACCESS
	  for (int i=0; i<Ncolor(length); i++)
	    for (int j=0; j<Ncolor(length); j++)
	      arg.out.Ghost(d, parity, x, i, j) = arg.in.Ghost(d, parity, x, i, j);
#else
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  arg.in.loadGhost(in, x, d, parity); // assumes we are loading 
	  for (int i=0; i<length; i++) out[i] = in[i];
	  arg.out.saveGhost(out, x, d, parity);
#endif
	}
      }

    }
  }

  /**
     Generic CUDA kernel for copying the ghost zone.  Adopts a similar form as
     the CPU version, using the same inlined functions.
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  __global__ void copyGhostKernel(CopyGaugeArg<OutOrder,InOrder> arg) {  
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    for (int parity=0; parity<2; parity++) {
      for (int d=0; d<arg.nDim; d++) {
	if (x < arg.faceVolumeCB[d]) {
#ifdef FINE_GRAINED_ACCESS
	  for (int i=0; i<Ncolor(length); i++)
	    for (int j=0; j<Ncolor(length); j++)
	      arg.out.Ghost(d, parity, x, i, j) = arg.in.Ghost(d, parity, x, i, j);
#else
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  arg.in.loadGhost(in, x, d, parity); // assumes we are loading 
	  for (int i=0; i<length; i++) out[i] = in[i];
	  arg.out.saveGhost(out, x, d, parity);
#endif
	}
      }

    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder, bool isGhost>
  class CopyGauge : Tunable {
    CopyGaugeArg<OutOrder,InOrder> arg;
    int size;
    const GaugeField &meta;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return size; }

  public:
    CopyGauge(CopyGaugeArg<OutOrder,InOrder> &arg, const GaugeField &meta) : arg(arg), meta(meta) { 
      int faceMax = 0;
      for (int d=0; d<arg.nDim; d++) {
	faceMax = (arg.faceVolumeCB[d] > faceMax ) ? arg.faceVolumeCB[d] : faceMax;
      }
      size = isGhost ? faceMax : arg.volume/2;
#ifndef FINE_GRAINED_ACCESS
      writeAuxString("out_stride=%d,in_stride=%d,geometry=%d", arg.out.stride, arg.in.stride, arg.in.geometry);
#endif
    }

    virtual ~CopyGauge() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#if (__COMPUTE_CAPABILITY__ >= 200)
      if (!isGhost) {
	copyGaugeKernel<FloatOut, FloatIn, length, OutOrder, InOrder> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      } else {
	copyGhostKernel<FloatOut, FloatIn, length, OutOrder, InOrder> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
#else
      errorQuda("Gauge copy not supported on pre-Fermi architecture");
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; } 
    long long bytes() const { 
      int sites = 4*arg.volume/2;
      if (isGhost) {
	sites = 0;
	for (int d=0; d<4; d++) sites += arg.faceVolumeCB[d];
      }
#if __COMPUTE_CAPABILITY__ >= 200 && !defined(FINE_GRAINED_ACCESS)
      return 2 * sites * (  arg.in.Bytes() + arg.in.hasPhase*sizeof(FloatIn) 
			    + arg.out.Bytes() + arg.out.hasPhase*sizeof(FloatOut) ); 
#else      
      return 2 * sites * (  arg.in.Bytes() + arg.out.Bytes() );
#endif
    } 
  };


  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void copyGauge(OutOrder &&outOrder, const InOrder &inOrder, int volume, const int *faceVolumeCB,
		   int nDim, int geometry, const GaugeField &out, QudaFieldLocation location, int type) {

    CopyGaugeArg<OutOrder,InOrder> arg(outOrder, inOrder, volume, faceVolumeCB, nDim, geometry);

    if (location == QUDA_CPU_FIELD_LOCATION) {
#ifdef HOST_DEBUG
      checkNan<FloatIn, length>(arg);
#endif

      if (type == 0 || type == 2) {
	copyGauge<FloatOut, FloatIn, length>(arg);
      }
#ifdef MULTI_GPU // only copy the ghost zone if doing multi-gpu
      if (type == 0 || type == 1) {
	if (geometry == QUDA_VECTOR_GEOMETRY || geometry == QUDA_COARSE_GEOMETRY) copyGhost<FloatOut, FloatIn, length>(arg);
	//else warningQuda("Cannot copy for %d geometry gauge field", geometry);
      }
#endif
    } else if (location == QUDA_CUDA_FIELD_LOCATION) {
      // first copy body
      if (type == 0 || type == 2) {
	CopyGauge<FloatOut, FloatIn, length, OutOrder, InOrder, 0> gaugeCopier(arg, out);
	gaugeCopier.apply(0);
      }
#ifdef MULTI_GPU
      if (type == 0 || type == 1) {
	if (geometry == QUDA_VECTOR_GEOMETRY || geometry == QUDA_COARSE_GEOMETRY) {
	  // now copy ghost
	  CopyGauge<FloatOut, FloatIn, length, OutOrder, InOrder, 1> ghostCopier(arg, out);
	  ghostCopier.apply(0);
	} else {
	  //warningQuda("Cannot copy for %d geometry gauge field", geometry);
	}
      }
#endif
    } else {
      errorQuda("Undefined field location %d for copyGauge", location);
    }

  }

} // namespace quda
