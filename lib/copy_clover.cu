#include <clover_field_order.h>
#include <tune_quda.h>

namespace quda {

  using namespace clover;

#ifdef GPU_CLOVER_DIRAC

  /** 
      Kernel argument struct
  */
  template <typename Out, typename In>
  struct CopyCloverArg {
    Out out;
    const In in;
    int volumeCB;
    CopyCloverArg (const Out &out, const In in, int volume) : out(out), in(in), volumeCB(in.volumeCB) { }
  };

  /** 
      Generic CPU clover reordering and packing
  */
  template <typename FloatOut, typename FloatIn, int length, typename Out, typename In>
  void copyClover(CopyCloverArg<Out,In> arg) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    for (int parity=0; parity<2; parity++) {
      for (int x=0; x<arg.volumeCB; x++) {
	RegTypeIn in[length];
	RegTypeOut out[length];
	arg.in.load(in, x, parity);
	for (int i=0; i<length; i++) out[i] = in[i];
	arg.out.save(out, x, parity);
      }
    }

  }

  /** 
      Generic CUDA clover reordering and packing
  */
  template <typename FloatOut, typename FloatIn, int length, typename Out, typename In>
  __global__ void copyCloverKernel(CopyCloverArg<Out,In> arg) {
    typedef typename mapper<FloatIn>::type RegTypeIn;
    typedef typename mapper<FloatOut>::type RegTypeOut;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= arg.volumeCB) return;
    int parity = blockIdx.y * blockDim.y + threadIdx.y;

    RegTypeIn in[length];
    RegTypeOut out[length];
    arg.in.load(in, x, parity);
#pragma unroll
    for (int i=0; i<length; i++) out[i] = in[i];
    arg.out.save(out, x, parity);

  }  

  template <typename FloatOut, typename FloatIn, int length, typename Out, typename In>
    class CopyClover : TunableVectorY {
    CopyCloverArg<Out,In> arg;
    const CloverField &meta;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.volumeCB; }

  public:
    CopyClover(CopyCloverArg<Out,In> &arg, const CloverField &meta)
      : TunableVectorY(2), arg(arg), meta(meta) {
      writeAuxString("out_stride=%d,in_stride=%d", arg.out.stride, arg.in.stride);
    }
    virtual ~CopyClover() { ; }
  
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      copyCloverKernel<FloatOut, FloatIn, length, Out, In> 
	<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; } 
    long long bytes() const { return 2*arg.volumeCB*(arg.in.Bytes() + arg.out.Bytes()); } 
  };

 template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
 void copyClover(OutOrder outOrder, const InOrder inOrder, const CloverField &out, QudaFieldLocation location) {

   CopyCloverArg<OutOrder,InOrder> arg(outOrder, inOrder, out.Volume());
   
   if (location == QUDA_CPU_FIELD_LOCATION) {
     copyClover<FloatOut, FloatIn, length, OutOrder, InOrder>(arg);
   } else if (location == QUDA_CUDA_FIELD_LOCATION) {
     CopyClover<FloatOut, FloatIn, length, OutOrder, InOrder> cloverCopier(arg, out);
     cloverCopier.apply(0);
   } else {
     errorQuda("Undefined field location %d for copyClover", location);
   }

 }

 template <typename FloatOut, typename FloatIn, int length, typename InOrder>
 void copyClover(const InOrder &inOrder, CloverField &out, bool inverse, QudaFieldLocation location, FloatOut *Out, float *outNorm) {

    if (out.isNative()) {
      const bool override = true;
      typedef typename clover_mapper<FloatOut>::type C;
      copyClover<FloatOut,FloatIn,length>(C(out, inverse, Out, outNorm, override), inOrder, out, location);
    } else if (out.Order() == QUDA_PACKED_CLOVER_ORDER) {
      copyClover<FloatOut,FloatIn,length>
	(QDPOrder<FloatOut,length>(out, inverse, Out), inOrder, out, location);
    } else if (out.Order() == QUDA_QDPJIT_CLOVER_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      copyClover<FloatOut,FloatIn,length>
	(QDPJITOrder<FloatOut,length>(out, inverse, Out), inOrder, out, location);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_BQCD_CLOVER_ORDER) {
      errorQuda("BQCD output not supported");
    } else {
      errorQuda("Clover field %d order not supported", out.Order());
    }

  }

 template <typename FloatOut, typename FloatIn, int length>
 void copyClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location, 
		 FloatOut *Out, FloatIn *In, float *outNorm, float *inNorm) {

    // reconstruction only supported on FloatN fields currently
   if (in.isNative()) {
      const bool override = true;
      typedef typename clover_mapper<FloatIn>::type C;
      copyClover<FloatOut,FloatIn,length>(C(in, inverse, In, inNorm, override), out, inverse, location, Out, outNorm);
    } else if (in.Order() == QUDA_PACKED_CLOVER_ORDER) {
      copyClover<FloatOut,FloatIn,length>
	(QDPOrder<FloatIn,length>(in, inverse, In), out, inverse, location, Out, outNorm);
    } else if (in.Order() == QUDA_QDPJIT_CLOVER_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      copyClover<FloatOut,FloatIn,length>
	(QDPJITOrder<FloatIn,length>(in, inverse, In), out, inverse, location, Out, outNorm);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_BQCD_CLOVER_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      copyClover<FloatOut,FloatIn,length>
	(BQCDOrder<FloatIn,length>(in, inverse, In), out, inverse, location, Out, outNorm);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else {
      errorQuda("Clover field %d order not supported", in.Order());
    }

  }

#endif

  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericClover(CloverField &out, const CloverField &in, bool inverse, QudaFieldLocation location,
			void *Out, void *In, void *outNorm, void *inNorm) {
#ifdef GPU_CLOVER_DIRAC
    if (out.Precision() == QUDA_HALF_PRECISION && out.Order() > 4) 
      errorQuda("Half precision not supported for order %d", out.Order());
    if (in.Precision() == QUDA_HALF_PRECISION && in.Order() > 4) 
      errorQuda("Half precision not supported for order %d", in.Order());

    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyClover<double,double,72>(out, in, inverse, location, (double*)Out, (double*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyClover<double,float,72>(out, in, inverse, location, (double*)Out, (float*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyClover<double,short,72>(out, in, inverse, location, (double*)Out, (short*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
        copyClover<double, char, 72>(
            out, in, inverse, location, (double *)Out, (char *)In, (float *)outNorm, (float *)inNorm);
      } else {
        errorQuda("Unknown precision %d", in.Precision());
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyClover<float,double,72>(out, in, inverse, location, (float*)Out, (double*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyClover<float,float,72>(out, in, inverse, location, (float*)Out, (float*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyClover<float,short,72>(out, in, inverse, location, (float*)Out, (short*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
        copyClover<float, char, 72>(
            out, in, inverse, location, (float *)Out, (char *)In, (float *)outNorm, (float *)inNorm);
      } else {
        errorQuda("Unknown precision %d", in.Precision());
      }
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION){
	copyClover<short,double,72>(out, in, inverse, location, (short*)Out, (double*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyClover<short,float,72>(out, in, inverse, location, (short*)Out, (float*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyClover<short,short,72>(out, in, inverse, location, (short*)Out, (short*)In, (float*)outNorm, (float*)inNorm);
      } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
        copyClover<short, char, 72>(
            out, in, inverse, location, (short *)Out, (char *)In, (float *)outNorm, (float *)inNorm);
      } else {
        errorQuda("Unknown precision %d", in.Precision());
      }
    } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
        copyClover<char, double, 72>(
            out, in, inverse, location, (char *)Out, (double *)In, (float *)outNorm, (float *)inNorm);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
        copyClover<char, float, 72>(
            out, in, inverse, location, (char *)Out, (float *)In, (float *)outNorm, (float *)inNorm);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
        copyClover<char, short, 72>(
            out, in, inverse, location, (char *)Out, (short *)In, (float *)outNorm, (float *)inNorm);
      } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
        copyClover<char, char, 72>(
            out, in, inverse, location, (char *)Out, (char *)In, (float *)outNorm, (float *)inNorm);
      } else {
        errorQuda("Unknown precision %d", in.Precision());
      }
    } else {
      errorQuda("Unknown precision %d", out.Precision());
    }
#else
    errorQuda("Clover has not been built");
#endif
  }


} // namespace quda
