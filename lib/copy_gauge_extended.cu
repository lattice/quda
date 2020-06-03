#include <tune_quda.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>

namespace quda {

  using namespace gauge;

  /**
     Kernel argument struct
   */
  template <typename OutOrder, typename InOrder>
  struct CopyGaugeExArg {
    OutOrder out;
    const InOrder in;
    int Xin[QUDA_MAX_DIM];
    int Xout[QUDA_MAX_DIM];
    int volume;
    int volumeEx;
    int nDim;
    int geometry;
    int faceVolumeCB[QUDA_MAX_DIM];
    bool regularToextended;
    CopyGaugeExArg(const OutOrder &out, const InOrder &in, const int *Xout, const int *Xin,
       const int *faceVolumeCB, int nDim, int geometry)
      : out(out), in(in), nDim(nDim), geometry(geometry) {
      for (int d=0; d<nDim; d++) {
	this->Xout[d] = Xout[d];
	this->Xin[d] = Xin[d];
	this->faceVolumeCB[d] = faceVolumeCB[d];
      }

      if (out.volumeCB > in.volumeCB) {
        this->volume = 2*in.volumeCB;
        this->volumeEx = 2*out.volumeCB;
        this->regularToextended = true;
      } else {
        this->volume = 2*out.volumeCB;
        this->volumeEx = 2*in.volumeCB;
        this->regularToextended = false;
      }
    }

  };

  /**
     Copy a regular/extended gauge field into an extended/regular gauge field
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder, bool regularToextended>
  __device__ __host__ void copyGaugeEx(CopyGaugeExArg<OutOrder,InOrder> &arg, int X, int parity) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;
    constexpr int nColor = Ncolor(length);

    int x[4];
    int R[4];
    int xin, xout;
    if(regularToextended){
      //regular to extended
      for (int d=0; d<4; d++) R[d] = (arg.Xout[d] - arg.Xin[d]) >> 1;
      int za = X/(arg.Xin[0]/2);
      int x0h = X - za*(arg.Xin[0]/2);
      int zb = za/arg.Xin[1];
      x[1] = za - zb*arg.Xin[1];
      x[3] = zb / arg.Xin[2];
      x[2] = zb - x[3]*arg.Xin[2];
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + parity) & 1);
      // Y is the cb spatial index into the extended gauge field
      xout = ((((x[3]+R[3])*arg.Xout[2] + (x[2]+R[2]))*arg.Xout[1] + (x[1]+R[1]))*arg.Xout[0]+(x[0]+R[0])) >> 1;
      xin = X;
    } else{
      //extended to regular gauge
      for (int d=0; d<4; d++) R[d] = (arg.Xin[d] - arg.Xout[d]) >> 1;
      int za = X/(arg.Xout[0]/2);
      int x0h = X - za*(arg.Xout[0]/2);
      int zb = za/arg.Xout[1];
      x[1] = za - zb*arg.Xout[1];
      x[3] = zb / arg.Xout[2];
      x[2] = zb - x[3]*arg.Xout[2];
      x[0] = 2*x0h + ((x[1] + x[2] + x[3] + parity) & 1);
      // Y is the cb spatial index into the extended gauge field
      xin = ((((x[3]+R[3])*arg.Xin[2] + (x[2]+R[2]))*arg.Xin[1] + (x[1]+R[1]))*arg.Xin[0]+(x[0]+R[0])) >> 1;
      xout = X;
    }
    for (int d=0; d<arg.geometry; d++) {
      const Matrix<complex<RegTypeIn>,nColor> in = arg.in(d, xin, parity);
      Matrix<complex<RegTypeOut>,nColor> out = in;
      arg.out(d, xout, parity) = out;
    }//dir
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder, bool regularToextended>
  void copyGaugeEx(CopyGaugeExArg<OutOrder,InOrder> arg) {
    for (int parity=0; parity<2; parity++) {
      for(int X=0; X<arg.volume/2; X++){
        copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder, regularToextended>(arg, X, parity);
      }
    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder, bool regularToextended>
  __global__ void copyGaugeExKernel(CopyGaugeExArg<OutOrder,InOrder> arg) {
    for (int parity=0; parity<2; parity++) {
      int X = blockIdx.x * blockDim.x + threadIdx.x;
      if (X >= arg.volume/2) return;
      copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder, regularToextended>(arg, X, parity);
    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    class CopyGaugeEx : Tunable {
    CopyGaugeExArg<OutOrder,InOrder> arg;
    const GaugeField &meta; // use for metadata
    QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.volume/2; }

  public:
    CopyGaugeEx(CopyGaugeExArg<OutOrder,InOrder> &arg, const GaugeField &meta, QudaFieldLocation location)
      : arg(arg), meta(meta), location(location) {
      writeAuxString("out_stride=%d,in_stride=%d,geometry=%d",arg.out.stride,arg.in.stride,arg.geometry);
    }
    virtual ~CopyGaugeEx() { ; }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      if (location == QUDA_CPU_FIELD_LOCATION) {
	if(arg.regularToextended) copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder, true>(arg);
	else copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder, false>(arg);
      } else if (location == QUDA_CUDA_FIELD_LOCATION) {
	if(arg.regularToextended) copyGaugeExKernel<FloatOut, FloatIn, length, OutOrder, InOrder, true>
				    <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
	else copyGaugeExKernel<FloatOut, FloatIn, length, OutOrder, InOrder, false>
	       <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    long long flops() const { return 0; }
    long long bytes() const {
      int sites = 4*arg.volume/2;
      return 2 * sites * (  arg.in.Bytes() + arg.in.hasPhase*sizeof(FloatIn)
                          + arg.out.Bytes() + arg.out.hasPhase*sizeof(FloatOut) );
    }
  };


  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  void copyGaugeEx(OutOrder outOrder, const InOrder inOrder, const int *E,
		   const int *X, const int *faceVolumeCB, const GaugeField &meta, QudaFieldLocation location) {

    CopyGaugeExArg<OutOrder,InOrder>
      arg(outOrder, inOrder, E, X, faceVolumeCB, meta.Ndim(), meta.Geometry());
    CopyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder> copier(arg, meta, location);
    copier.apply(0);
    if (location == QUDA_CUDA_FIELD_LOCATION) checkCudaError();
  }

  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGaugeEx(const InOrder &inOrder, const int *X, GaugeField &out,
		   QudaFieldLocation location, FloatOut *Out) {

    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface();

    if (out.isNative()) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_NO>::type G;
        copyGaugeEx<FloatOut, FloatIn, length>(G(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_12>::type G;
	copyGaugeEx<FloatOut,FloatIn,length>
	  (G(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_8>::type G;
	copyGaugeEx<FloatOut,FloatIn,length>
	  (G(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_13>::type G;
        copyGaugeEx<FloatOut,FloatIn,length>
	  (G(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_9>::type G;
        copyGaugeEx<FloatOut,FloatIn,length>
	  (G(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
	errorQuda("Reconstruction %d and order %d not supported", out.Reconstruct(), out.Order());
      }
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>
	(QDPOrder<FloatOut,length>(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>
	(MILCOrder<FloatOut,length>(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>
	(TIFROrder<FloatOut,length>(out, Out), inOrder, out.X(), X, faceVolumeCB, out, location);
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

    if (in.isNative()) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_NO>::type G;
        copyGaugeEx<FloatOut, FloatIn, length>(G(in, In), in.X(), out, location, Out);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_12>::type G;
	copyGaugeEx<FloatOut,FloatIn,length> (G(in, In), in.X(), out, location, Out);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_8>::type G;
	copyGaugeEx<FloatOut,FloatIn,length> (G(in, In), in.X(), out, location, Out);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_13>::type G;
	copyGaugeEx<FloatOut,FloatIn,length> (G(in, In), in.X(), out, location, Out);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_9>::type G;
	copyGaugeEx<FloatOut,FloatIn,length> (G(in, In), in.X(), out, location, Out);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
	errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>(QDPOrder<FloatIn,length>(in, In),
					   in.X(), out, location, Out);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

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
#if QUDA_PRECISION & 4
        copyGaugeEx(out, in, location, (double*)Out, (float*)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Precision %d not instantiated", in.Precision());
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
        copyGaugeEx(out, in, location, (float *)Out, (double *)In);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
#if QUDA_PRECISION & 4
        copyGaugeEx(out, in, location, (float *)Out, (float *)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Precision %d not instantiated", in.Precision());
      }
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        copyGaugeEx(out, in, location, (short *)Out, (short *)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Precision %d not instantiated", in.Precision());
      }
    } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
      if (in.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        copyGaugeEx(out, in, location, (char *)Out, (char *)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable single precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Precision %d not instantiated", in.Precision());
      }
    } else {
      errorQuda("Precision %d not instantiated", out.Precision());
    }
  }

} // namespace quda
