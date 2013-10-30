#include <gauge_field_order.h>

namespace quda {

  /**
     Kernel argument struct
   */
  template <typename OutOrder, typename InOrder>
  struct CopyGaugeExArg {
    OutOrder out;
    const InOrder in;
    int R; // the radius of the extended region
    int volume;
    int volumeEx;
    int nDim;
    int geometry;
    int X[QUDA_MAX_DIM]; // geometry of the normal gauge field
    int faceVolumeCB[QUDA_MAX_DIM];
    CopyGaugeExArg(const OutOrder &out, const InOrder &in, int R,
		 const int *X, const int *faceVolumeCB, int nDim, int geometry) 
      : out(out), in(in), R(R), volume(2*in.volumeCB), volumeEx(2*out.volumeCB), 
	nDim(nDim), geometry(geometry) {
      for (int d=0; d<nDim; d++) {
	this->X[d] = X[d];
	this->faceVolumeCB[d] = faceVolumeCB[d];
      }
    }
  };

  /**
     Copy a regular gauge field into an extended gauge field
  */
  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  __device__ __host__ void copyGaugeEx(CopyGaugeExArg<OutOrder,InOrder> arg, int X, int parity) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x[4];
    int E[4];
    for (int d=0; d<4; d++) E[d] = arg.X[d] + 2*arg.R;
    
    int za = X/(arg.X[0]/2);
    int x0h = X - za*(arg.X[0]/2);
    int zb = za/arg.X[1];
    x[1] = za - zb*arg.X[1];
    x[3] = zb / arg.X[2];
    x[2] = zb - x[3]*arg.X[2];
    x[0] = 2*x0h + ((x[1] + x[2] + x[3] + parity) & 1);
    
    // Y is the cb spatial index into the extended gauge field
    int Y = ((((x[3]+arg.R)*E[2] + (x[2]+arg.R))*E[1] + (x[1]+arg.R))*E[0]+(x[0]+arg.R)) >> 1;
    
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
	copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder> (arg, X, parity);
      }
    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  __global__ void copyGaugeExKernel(CopyGaugeExArg<OutOrder,InOrder> arg) {
    for (int parity=0; parity<2; parity++) {
      int X = blockIdx.x * blockDim.x + threadIdx.x;
      copyGaugeEx<FloatOut, FloatIn, length, OutOrder, InOrder> (arg, X, parity);
    }
  }

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  void copyGaugeEx(OutOrder outOrder, const InOrder inOrder, 
		   const int *X, const int *faceVolumeCB, int nDim, int geometry, int R,
		   QudaFieldLocation location) {

    CopyGaugeExArg<OutOrder,InOrder> 
      arg(outOrder, inOrder, R, X, faceVolumeCB, nDim, geometry);

    if (location == QUDA_CPU_FIELD_LOCATION) {
      copyGaugeEx<FloatOut, FloatIn, length>(arg);
    } /*else if (location == QUDA_CUDA_FIELD_LOCATION) {
      CopyGauge<FloatOut, FloatIn, length, OutOrder, InOrder, 0> gaugeCopier(arg);
      gaugeCopier.apply(0);
      } */else {
      errorQuda("Undefined field location %d for copyGauge", location);
    }

  }
  
  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGaugeEx(const InOrder &inOrder, const int *X, GaugeField &out, int R,
		   QudaFieldLocation location, FloatOut *Out) {
    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface(); 

    if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>
	(MILCOrder<FloatOut,length>(out, Out), inOrder,
	 X, faceVolumeCB, out.Ndim(), out.Geometry(), R, location);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>
	(TIFROrder<FloatOut,length>(out, Out), inOrder,
	 X, faceVolumeCB, out.Ndim(), out.Geometry(), R, location);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, int R, QudaFieldLocation location, 
		   FloatOut *Out, FloatIn *In) {

    if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>(MILCOrder<FloatIn,length>(in, In), 
					   in.X(), out, R, location, Out);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGaugeEx<FloatOut,FloatIn,length>(TIFROrder<FloatIn,length>(in, In), 
					   in.X(), out, R, location, Out);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }

  }

  template <typename FloatOut, typename FloatIn>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, int R, QudaFieldLocation location, 
		 FloatOut *Out, FloatIn *In) {
    
    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }
    
    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    if (in.LinkType() != QUDA_ASQTAD_MOM_LINKS && out.LinkType() != QUDA_ASQTAD_MOM_LINKS) {
      // we are doing gauge field packing
      copyGaugeEx<FloatOut,FloatIn,18>(out, in, R, location, Out, In);
    } else {
      errorQuda("Not supported");
    }
  }

  void copyExtendedGauge(GaugeField &out, const GaugeField &in, int R, QudaFieldLocation location,
			 void *Out, void *In) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGaugeEx(out, in, R, location, (double*)Out, (double*)In);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeEx(out, in, R, location, (double*)Out, (float*)In);
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGaugeEx(out, in, R, location, (float*)Out, (double*)In);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGaugeEx(out, in, R, location, (float*)Out, (float*)In);
      }
    }
  }

} // namespace quda
