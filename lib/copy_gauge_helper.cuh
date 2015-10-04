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
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  arg.in.load(in, x, d, parity);
	  for (int i=0; i<length; i++) out[i] = in[i];
	  arg.out.save(out, x, d, parity);
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

	RegTypeIn in[length];
	RegTypeOut out[length];
	arg.in.load(in, x, d, parity);
	for (int i=0; i<length; i++) out[i] = in[i];
	arg.out.save(out, x, d, parity);
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
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  arg.in.loadGhost(in, x, d, parity); // assumes we are loading 
	  for (int i=0; i<length; i++) out[i] = in[i];
	  arg.out.saveGhost(out, x, d, parity);
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
	  RegTypeIn in[length];
	  RegTypeOut out[length];
	  arg.in.loadGhost(in, x, d, parity); // assumes we are loading 
	  for (int i=0; i<length; i++) out[i] = in[i];
	  arg.out.saveGhost(out, x, d, parity);
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
      writeAuxString("out_stride=%d,in_stride=%d", arg.out.stride, arg.in.stride);
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

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 0; } 
    long long bytes() const { 
      int sites = 4*arg.volume/2;
      if (isGhost) {
	sites = 0;
	for (int d=0; d<4; d++) sites += arg.faceVolumeCB[d];
      }
#if __COMPUTE_CAPABILITY__ >= 200
      return 2 * sites * (  arg.in.Bytes() + arg.in.hasPhase*sizeof(FloatIn) 
			    + arg.out.Bytes() + arg.out.hasPhase*sizeof(FloatOut) ); 
#else      
      return 2 * sites * (  arg.in.Bytes() + arg.out.Bytes() );
#endif
    } 
  };


  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void copyGauge(OutOrder outOrder, const InOrder inOrder, int volume, const int *faceVolumeCB, 
		   int nDim, int geometry, const GaugeField &out, QudaFieldLocation location, int type) {

    CopyGaugeArg<OutOrder,InOrder> arg(outOrder, inOrder, volume, faceVolumeCB, nDim, geometry);

    if (location == QUDA_CPU_FIELD_LOCATION) {
      if (type == 0 || type == 2) {
	copyGauge<FloatOut, FloatIn, length>(arg);
      }
#ifdef MULTI_GPU // only copy the ghost zone if doing multi-gpu
      if (type == 0 || type == 1) {
	if (geometry == QUDA_VECTOR_GEOMETRY) copyGhost<FloatOut, FloatIn, length>(arg);
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
	if (geometry == QUDA_VECTOR_GEOMETRY) {
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
