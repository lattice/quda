#include <gauge_field_order.h>

namespace quda {

  /**
     Kernel argument struct
   */
  template <typename OutOrder, typename InOrder>
  struct CopyGaugeExArg {
    OutOrder out;
    const InOrder in;
    int E[QUDA_MAX_DIM]; // geometry of extended gauge field
    int X[QUDA_MAX_DIM]; // geometry of the normal gauge field
    int volume;
    int volumeEx;
    int nDim;
    int geometry;
    int faceVolumeCB[QUDA_MAX_DIM];
    CopyGaugeExArg(const OutOrder &out, const InOrder &in, const int *E, const int *X, 
		   const int *faceVolumeCB, int nDim, int geometry) 
      : out(out), in(in), volume(2*in.volumeCB), volumeEx(2*out.volumeCB), 
	nDim(nDim), geometry(geometry) {
      for (int d=0; d<nDim; d++) {
	this->E[d] = E[d];
	this->X[d] = X[d];
	this->faceVolumeCB[d] = faceVolumeCB[d];
      }
    }
  };

  /**
     Copy a regular gauge field into an extended gauge field
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  __device__ __host__ void copyGaugeEx(CopyGaugeExArg<OutOrder,InOrder> &arg, int X, int parity) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x[4];
    int R[4];
    for (int d=0; d<4; d++) R[d] = (arg.E[d] - arg.X[d]) >> 1;
    
    int za = X/(arg.X[0]/2);
    int x0h = X - za*(arg.X[0]/2);
    int zb = za/arg.X[1];
    x[1] = za - zb*arg.X[1];
    x[3] = zb / arg.X[2];
    x[2] = zb - x[3]*arg.X[2];
    x[0] = 2*x0h + ((x[1] + x[2] + x[3] + parity) & 1);
    
    // Y is the cb spatial index into the extended gauge field
    int Y = ((((x[3]+R[3])*arg.E[2] + (x[2]+R[2]))*arg.E[1] + (x[1]+R[1]))*arg.E[0]+(x[0]+R[0])) >> 1;
    
    for(int d=0; d<arg.geometry; d++){
      RegTypeIn in[length];
      RegTypeOut out[length];
      arg.in.load(in, X, d, parity);
      for (int i=0; i<length; i++) out[i] = in[i];
      arg.out.save(out, Y, d, parity);
    }//dir
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  void copyGaugeEx(CopyGaugeExArg<OutOrder,InOrder> arg) {
    for (int parity=0; parity<2; parity++) {
      for(int X=0; X<arg.volume/2; X++){
	copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder>(arg, X, parity);
      }
    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  __global__ void copyGaugeExKernel(CopyGaugeExArg<OutOrder,InOrder> arg) {
    for (int parity=0; parity<2; parity++) {
      int X = blockIdx.x * blockDim.x + threadIdx.x;
      copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder>(arg, X, parity);
    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    class CopyGaugeEx : Tunable {
    CopyGaugeExArg<OutOrder,InOrder> arg;
    QudaFieldLocation location;
    int size;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return size; }

  public:
    CopyGaugeEx(CopyGaugeExArg<OutOrder,InOrder> &arg, QudaFieldLocation location) 
      : arg(arg), location(location) { 
      size = arg.volume/2;
    }
    virtual ~CopyGaugeEx() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (location == QUDA_CPU_FIELD_LOCATION) {
	copyGaugeEx<FloatOut, FloatIn, length>(arg);
      } else if (location == QUDA_CUDA_FIELD_LOCATION) {
	copyGaugeExKernel<FloatOut, FloatIn, length, OutOrder, InOrder> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.X[0] << "x";
      vol << arg.X[1] << "x";
      vol << arg.X[2] << "x";
      vol << arg.X[3];    
      aux << "out_stride=" << arg.out.stride << ",in_stride=" << arg.in.stride;
      aux << "geometry=" << arg.geometry;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    long long flops() const { return 0; } 
    long long bytes() const { 
      int sites = 4*arg.volume/2;
      return 2 * sites * (  arg.in.Bytes() + arg.in.hasPhase*sizeof(FloatIn) 
                          + arg.out.Bytes() + arg.out.hasPhase*sizeof(FloatOut) ); 
    } 
  };


  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  void copyGaugeEx(OutOrder outOrder, const InOrder inOrder, 
		   const int *E, const int *X, const int *faceVolumeCB, int nDim, int geometry, 
		   QudaFieldLocation location) {

    CopyGaugeExArg<OutOrder,InOrder> 
      arg(outOrder, inOrder, E, X, faceVolumeCB, nDim, geometry);
    CopyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder> copier(arg, location);
    copier.apply(0);
  }
  
  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGaugeEx(const InOrder &inOrder, const int *X, GaugeField &out, 
		   QudaFieldLocation location, FloatOut *Out) {
    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface(); 

    if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>
	(MILCOrder<FloatOut,length>(out, Out), inOrder,
	 out.X(), X, faceVolumeCB, out.Ndim(), out.Geometry(), location);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>
	(TIFROrder<FloatOut,length>(out, Out), inOrder,
	 out.X(), X, faceVolumeCB, out.Ndim(), out.Geometry(), location);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location, 
		   FloatOut *Out, FloatIn *In) {

    if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>(MILCOrder<FloatIn,length>(in, In), 
					   in.X(), out, location, Out);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>(TIFROrder<FloatIn,length>(in, In), 
					   in.X(), out, location, Out);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }

  }

  template <typename FloatOut, typename FloatIn>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location, 
		   FloatOut *Out, FloatIn *In) {
    
    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }
    
    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    if (in.LinkType() != QUDA_ASQTAD_MOM_LINKS && out.LinkType() != QUDA_ASQTAD_MOM_LINKS) {
      // we are doing gauge field packing
      copyGaugeEx<FloatOut,FloatIn,18>(out, in, location, Out, In);
    } else {
      errorQuda("Not supported");
    }
  }

  void copyExtendedGauge(GaugeField &out, const GaugeField &in,
			 QudaFieldLocation location, void *Out, void *In) {

    for (int d=0; d<in.Ndim(); d++) {
      if ( (out.X()[d] - in.X()[d]) % 2 != 0)
	errorQuda("Cannot copy into an asymmetrically extended gauge field");
    }

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGaugeEx(out, in, location, (double*)Out, (double*)In);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeEx(out, in, location, (double*)Out, (float*)In);
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGaugeEx(out, in, location, (float*)Out, (double*)In);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeEx(out, in, location, (float*)Out, (float*)In);
      }
    }
  }

} // namespace quda
