#include <gauge_field_order.h>

namespace quda {

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
    CopyGaugeArg(const OutOrder &out, const InOrder &in, int volume, const int *faceVolumeCB, int nDim) 
      : out(out), in(in), volume(volume), nDim(nDim) {
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

      for (int d=0; d<arg.nDim; d++) {
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

      for (int d=0; d<arg.nDim; d++) {
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

  private:
    int sharedBytesPerThread() const { return 0; }
    int sharedBytesPerBlock(const TuneParam &param) const { return 0 ;}

    bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
    bool advanceBlockDim(TuneParam &param) const {
      bool advance = Tunable::advanceBlockDim(param);
      if (advance) param.grid = dim3( (size+param.block.x-1) / param.block.x, 1, 1);
      return advance;
    }

  public:
    CopyGauge(CopyGaugeArg<OutOrder,InOrder> &arg) : arg(arg) { 
      int faceMax = 0;
      for (int d=0; d<arg.nDim; d++) {
	faceMax = (arg.faceVolumeCB[d] > faceMax ) ? arg.faceVolumeCB[d] : faceMax;
      }
      size = isGhost ? faceMax : arg.volume/2;
    }
    virtual ~CopyGauge() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, QUDA_TUNE_YES, QUDA_VERBOSE);
      if (!isGhost) {
	copyGaugeKernel<FloatOut, FloatIn, length, OutOrder, InOrder> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      } else {
	copyGhostKernel<FloatOut, FloatIn, length, OutOrder, InOrder> 
	  <<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << arg.in.volumeCB; 
      aux << "out_stride=" << arg.out.stride << ",in_stride=" << arg.in.stride;
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }

    std::string paramString(const TuneParam &param) const { // Don't bother printing the grid dim.
      std::stringstream ps;
      ps << "block=(" << param.block.x << "," << param.block.y << "," << param.block.z << "), ";
      ps << "shared=" << param.shared_bytes;
      return ps.str();
    }

    virtual void initTuneParam(TuneParam &param) const {
      Tunable::initTuneParam(param);
      param.grid = dim3( (size+param.block.x-1) / param.block.x, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const {
      Tunable::defaultTuneParam(param);
      param.grid = dim3( (size+param.block.x-1) / param.block.x, 1, 1);
    }

    long long flops() const { return 0; } 
    long long bytes() const { 
      int sites = 4*arg.volume/2;
      if (isGhost) {
	sites = 0;
	for (int d=0; d<4; d++) sites += arg.faceVolumeCB[d];
      }
      return 2 * sites * (arg.in.Bytes() + arg.out.Bytes()); 
    } 
  };

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
    void copyGauge(OutOrder outOrder, const InOrder inOrder, int volume, 
		   const int *faceVolumeCB, int nDim, QudaFieldLocation location, int type) {

    CopyGaugeArg<OutOrder,InOrder> arg(outOrder, inOrder, volume, faceVolumeCB, nDim);

    if (location == QUDA_CPU_FIELD_LOCATION) {
      if (type == 0) copyGauge<FloatOut, FloatIn, length>(arg);
#ifdef MULTI_GPU // only copy the ghost zone if doing multi-gpu
      copyGhost<FloatOut, FloatIn, length>(arg);
#endif
    } else if (location == QUDA_CUDA_FIELD_LOCATION) {
      // first copy body
      if (type == 0) {
	CopyGauge<FloatOut, FloatIn, length, OutOrder, InOrder, 0> gaugeCopier(arg);
	gaugeCopier.apply(0);
      }
#ifdef MULTI_GPU
      // now copy ghost
      CopyGauge<FloatOut, FloatIn, length, OutOrder, InOrder, 1> ghostCopier(arg);
      ghostCopier.apply(0);
#endif
    } else {
      errorQuda("Undefined field location %d for copyGauge", location);
    }

  }
  
  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGauge(const InOrder &inOrder, GaugeField &out, QudaFieldLocation location, 
		 FloatOut *Out, FloatOut **outGhost, int type) {
    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface(); 
    if (out.Order() == QUDA_FLOAT_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	copyGauge<FloatOut,FloatIn,length>(FloatNOrder<FloatOut,length,1,18>(out, Out, outGhost), 
					   inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
      } else {
	errorQuda("Reconstruction %d and order %d not supported", out.Reconstruct(), out.Order());
      }
    } else if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  copyGauge<FloatOut,FloatIn,length>
	    (FloatNOrder<FloatOut,length,2,19>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
	} else {
	  copyGauge<FloatOut,FloatIn,length>
	    (FloatNOrder<FloatOut,length,2,18>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
	}
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	copyGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,2,12>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);	   
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
	copyGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,2,8>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);	   
      }
    } else if (out.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	copyGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,4,12>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
	copyGauge<FloatOut,FloatIn,length> 
	  (FloatNOrder<FloatOut,length,4,8>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
      } else {
	errorQuda("Reconstruction %d and order %d not supported", out.Reconstruct(), out.Order());
      }
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>
	(QDPOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
    } else if (out.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>
	(CPSOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>
	(MILCOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
    } else if (out.Order() == QUDA_BQCD_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>
	(BQCDOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(), faceVolumeCB, out.Ndim(), location, type);
    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
    void copyGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, 
		   FloatOut *Out, FloatIn *In, FloatOut **outGhost, FloatIn **inGhost, int type) {

    // reconstruction only supported on FloatN fields currently
    if (in.Order() == QUDA_FLOAT_GAUGE_ORDER) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	copyGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,1,18>(in, In, inGhost), out, location, Out, outGhost, type);
      } else {
	errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatIn)==typeid(short) && in.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  copyGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,19>(in, In, inGhost), 
					      out, location, Out, outGhost, type);
	} else {
	  copyGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,18>(in, In, inGhost),
					      out, location, Out, outGhost, type);
	}
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
	copyGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,12>(in, In, inGhost),
					    out, location, Out, outGhost, type);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
	copyGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,2,8>(in, In, inGhost), 
					    out, location, Out, outGhost, type);
      }
    } else if (in.Order() == QUDA_FLOAT4_GAUGE_ORDER) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
	copyGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,4,12>(in, In, inGhost), 
					    out, location, Out, outGhost, type);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
	copyGauge<FloatOut,FloatIn,length> (FloatNOrder<FloatIn,length,4,8>(in, In, inGhost), 
					    out, location, Out, outGhost, type);
      } else {
	errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>(QDPOrder<FloatIn,length>(in, In, inGhost), 
					 out, location, Out, outGhost, type);
    } else if (in.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>(CPSOrder<FloatIn,length>(in, In, inGhost), 
					 out, location, Out, outGhost, type);
    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>(MILCOrder<FloatIn,length>(in, In, inGhost), 
					 out, location, Out, outGhost, type);
    } else if (in.Order() == QUDA_BQCD_GAUGE_ORDER) {
      copyGauge<FloatOut,FloatIn,length>(BQCDOrder<FloatIn,length>(in, In, inGhost), 
					 out, location, Out, outGhost, type);
    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }

  }

  template <typename FloatOut, typename FloatIn>
  void copyGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, FloatOut *Out, 
		 FloatIn *In, FloatOut **outGhost, FloatIn **inGhost, int type) {

    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }
    
    if (in.LinkType() != QUDA_ASQTAD_MOM_LINKS && out.LinkType() != QUDA_ASQTAD_MOM_LINKS) {
      // we are doing gauge field packing
      copyGauge<FloatOut,FloatIn,18>(out, in, location, Out, In, outGhost, inGhost, type);
    } else {
      if (location != QUDA_CPU_FIELD_LOCATION) errorQuda("Location %d not supported", location);

      // we are doing momentum field packing
      if (in.Reconstruct() != QUDA_RECONSTRUCT_10 || out.Reconstruct() != QUDA_RECONSTRUCT_10) {
	errorQuda("Unsupported reconstruction types out=%d in=%d for momentum field", 
		  out.Reconstruct(), in.Reconstruct());
      }
    
      int faceVolumeCB[QUDA_MAX_DIM];
      for (int d=0; d<in.Ndim(); d++) faceVolumeCB[d] = in.SurfaceCB(d) * in.Nface();

      // momentum only currently supported on MILC and Float2 fields currently
	if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	  CopyGaugeArg<FloatNOrder<FloatOut,10,2,10>, FloatNOrder<FloatIn,10,2,10> >
	    arg(FloatNOrder<FloatOut,10,2,10>(out, Out), 
		FloatNOrder<FloatIn,10,2,10>(in, In), in.Volume(), faceVolumeCB, in.Ndim());
	  copyGauge<FloatOut,FloatIn,10>(arg);
	} else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
	  CopyGaugeArg<FloatNOrder<FloatOut,10,2,10>, MILCOrder<FloatIn,10> >
	    arg(FloatNOrder<FloatOut,10,2,10>(out, Out), MILCOrder<FloatIn,10>(in, In), 
		in.Volume(), faceVolumeCB, in.Ndim());
	  copyGauge<FloatOut,FloatIn,10>(arg);
	} else {
	  errorQuda("Gauge field orders %d not supported", in.Order());
	}
      } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
	if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	  CopyGaugeArg<MILCOrder<FloatOut,10>, FloatNOrder<FloatIn,10,2,10> >
	    arg(MILCOrder<FloatOut,10>(out, Out), FloatNOrder<FloatIn,10,2,10>(in, In),
		in.Volume(), faceVolumeCB, in.Ndim());
	  copyGauge<FloatOut,FloatIn,10>(arg);
	} else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
	  CopyGaugeArg<MILCOrder<FloatOut,10>, MILCOrder<FloatIn,10> >
	    arg(MILCOrder<FloatOut,10>(out, Out), MILCOrder<FloatIn,10>(in, In),
		in.Volume(), faceVolumeCB, in.Ndim());
	  copyGauge<FloatOut,FloatIn,10>(arg);
	} else {
	  errorQuda("Gauge field orders %d not supported", in.Order());
	}
      } else {
	errorQuda("Gauge field orders %d not supported", out.Order());
      }
    }
  }

  // this is the function that is actually called, from here on down we instantiate all required templates
  void copyGenericGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
			void *Out, void *In, void **ghostOut, void **ghostIn, int type) {
    if (out.Precision() == QUDA_DOUBLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGauge(out, in, location, (double*)Out, (double*)In, (double**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGauge(out, in, location, (double*)Out, (float*)In, (double**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyGauge(out, in, location, (double*)Out, (short*)In, (double**)ghostOut, (short**)ghostIn, type);
      }
    } else if (out.Precision() == QUDA_SINGLE_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION) {
	copyGauge(out, in, location, (float*)Out, (double*)In, (float**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGauge(out, in, location, (float*)Out, (float*)In, (float**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyGauge(out, in, location, (float*)Out, (short*)In, (float**)ghostOut, (short**)ghostIn, type);
      }
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_DOUBLE_PRECISION){
	copyGauge(out, in, location, (short*)Out, (double*)In, (short**)ghostOut, (double**)ghostIn, type);
      } else if (in.Precision() == QUDA_SINGLE_PRECISION) {
	copyGauge(out, in, location, (short*)Out, (float*)In, (short**)ghostOut, (float**)ghostIn, type);
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
	copyGauge(out, in, location, (short*)Out, (short*)In, (short**)ghostOut, (short**)ghostIn, type);
      }
    } 
  }

} // namespace quda
