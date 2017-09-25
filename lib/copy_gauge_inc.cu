#include <gauge_field_order.h>
#include <copy_gauge_helper.cuh>

namespace quda {

  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGauge(const InOrder &inOrder, GaugeField &out, QudaFieldLocation location, 
		 FloatOut *Out, FloatOut **outGhost, int type) {
    int faceVolumeCB[QUDA_MAX_DIM];
    for (int i=0; i<4; i++) faceVolumeCB[i] = out.SurfaceCB(i) * out.Nface(); 
    if (out.isNative()) {
      // this overrides the check that the texture maps to the gauge
      // pointer - this is safe here since it only occurs when running
      // the copier on the host when we will not be using texture
      // reads
      const bool override = true;
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatOut)==typeid(short) && out.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  copyGauge<short,FloatIn,length>
	    (FloatNOrder<short,length,2,19>(out, (short*)Out, (short**)outGhost, override), inOrder,
	     out.Volume(), faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
	} else {
	  typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_NO>::type G;
	  copyGauge<FloatOut,FloatIn,length>
	    (G(out, Out, outGhost, override), inOrder, out.Volume(), faceVolumeCB,
	     out.Ndim(), out.Geometry(), out, location, type);
	}
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_12>::type G;
	copyGauge<FloatOut,FloatIn,length>
	  (G(out, Out, outGhost, override), inOrder, out.Volume(), faceVolumeCB,
	   out.Ndim(), out.Geometry(), out, location, type);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_8>::type G;
	copyGauge<FloatOut,FloatIn,length> 
	  (G(out, Out, outGhost, override), inOrder, out.Volume(), faceVolumeCB,
	   out.Ndim(), out.Geometry(), out, location, type);
#ifdef GPU_STAGGERED_DIRAC
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_13) {
	typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_13>::type G;
        copyGauge<FloatOut,FloatIn,length>
	  (G(out, Out, outGhost, override), inOrder, out.Volume(), faceVolumeCB,
	   out.Ndim(),  out.Geometry(), out, location, type);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_9) {
	typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_9>::type G;
        copyGauge<FloatOut,FloatIn,length>
	  (G(out, Out, outGhost, override), inOrder, out.Volume(), faceVolumeCB,
	   out.Ndim(), out.Geometry(), out, location, type);
#endif
      } else {
	errorQuda("Reconstruction %d and order %d not supported", out.Reconstruct(), out.Order());
      }
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(QDPOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(), 
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(QDPJITOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(),
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(CPSOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(),
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("CPS interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(MILCOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(),
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(BQCDOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(),
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(TIFROrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(),
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(TIFRPaddedOrder<FloatOut,length>(out, Out, outGhost), inOrder, out.Volume(),
	 faceVolumeCB, out.Ndim(), out.Geometry(), out, location, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
    void copyGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, 
		   FloatOut *Out, FloatIn *In, FloatOut **outGhost, FloatIn **inGhost, int type) {

    // reconstruction only supported on FloatN fields currently
    if (in.isNative()) {      
      // this overrides the check that the texture maps to the gauge
      // pointer - this is safe here since it only occurs when running
      // the copier on the host when we will not be using texture
      // reads
      const bool override = true;
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	if (typeid(FloatIn)==typeid(short) && in.LinkType() == QUDA_ASQTAD_FAT_LINKS) {
	  copyGauge<FloatOut,short,length> (FloatNOrder<short,length,2,19>
					    (in,(short*)In,(short**)inGhost,override),
					    out, location, Out, outGhost, type);
	} else {
	  typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_NO>::type G;
	  copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost,override), out, location, Out, outGhost, type);
	}
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
	typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_12>::type G;
	copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost,override), out, location, Out, outGhost, type);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
	typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_8>::type G;
	copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost,override), out, location, Out, outGhost, type);
#ifdef GPU_STAGGERED_DIRAC
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
	typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_13>::type G;
	copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost,override), out, location, Out, outGhost, type);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
	typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_9>::type G;
	copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost,override), out, location, Out, outGhost, type);
#endif
      } else {
	errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(QDPOrder<FloatIn,length>(in, In, inGhost),
					 out, location, Out, outGhost, type);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(QDPJITOrder<FloatIn,length>(in, In, inGhost),
					 out, location, Out, outGhost, type);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(CPSOrder<FloatIn,length>(in, In, inGhost),
					 out, location, Out, outGhost, type);
#else
      errorQuda("CPS interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(MILCOrder<FloatIn,length>(in, In, inGhost),
					 out, location, Out, outGhost, type);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(BQCDOrder<FloatIn,length>(in, In, inGhost),
					 out, location, Out, outGhost, type);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(TIFROrder<FloatIn,length>(in, In, inGhost),
					 out, location, Out, outGhost, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(TIFRPaddedOrder<FloatIn,length>(in, In, inGhost),
					 out, location, Out, outGhost, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field order %d not supported", in.Order());
    }

  }

  void checkMomOrder(const GaugeField &u);

  template <typename FloatOut, typename FloatIn, int length, typename Out, typename In, typename Arg>
  void copyMom(Arg &arg, const GaugeField &meta, QudaFieldLocation location) {

    if (location == QUDA_CPU_FIELD_LOCATION) {
      copyGauge<FloatOut,FloatIn,length>(arg);
    } else if (location == QUDA_CUDA_FIELD_LOCATION) {
      CopyGauge<FloatOut,FloatIn,length, Out, In, 0> momCopier(arg, meta);
      momCopier.apply(0);
    } else {
      errorQuda("Undefined field location %d for copyMom", location);
    }

  }

  template <typename FloatOut, typename FloatIn>
  void copyGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, FloatOut *Out, 
		 FloatIn *In, FloatOut **outGhost, FloatIn **inGhost, int type) {

    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }
    
    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    if (in.LinkType() != QUDA_ASQTAD_MOM_LINKS && out.LinkType() != QUDA_ASQTAD_MOM_LINKS) {
      // we are doing gauge field packing
      copyGauge<FloatOut,FloatIn,18>(out, in, location, Out, In, outGhost, inGhost, type);
    } else {
      if (out.Geometry() != QUDA_VECTOR_GEOMETRY) errorQuda("Unsupported geometry %d", out.Geometry());

      checkMomOrder(in);
      checkMomOrder(out);
    
      int faceVolumeCB[QUDA_MAX_DIM];
      for (int d=0; d<in.Ndim(); d++) faceVolumeCB[d] = in.SurfaceCB(d) * in.Nface();

      // this overrides the check that the texture maps to the gauge
      // pointer - this is safe here since it only occurs when running
      // the copier on the host when we will not be using texture
      // reads
      const bool override = true;

      // momentum only currently supported on MILC (10), TIFR (18) and Float2 (10) fields currently
	if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    typedef FloatNOrder<FloatOut,10,2,10> momOut;
	    typedef FloatNOrder<FloatIn,10,2,10> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out, 0, override), momIn(in, In, 0, override), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,10,momOut,momIn>(arg,in,location);
	  } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
#ifdef BUILD_MILC_INTERFACE
	    typedef FloatNOrder<FloatOut,10,2,10> momOut;
	    typedef MILCOrder<FloatIn,10> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out, 0, override), momIn(in, In), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,10,momOut,momIn>(arg,in,location);
#else
	    errorQuda("MILC interface has not been built\n");
#endif
	  } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {
#ifdef BUILD_TIFR_INTERFACE
	    typedef FloatNOrder<FloatOut,18,2,11> momOut;
	    typedef TIFROrder<FloatIn,18> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out, 0, override), momIn(in, In), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,18,momOut,momIn>(arg,in,location);
#else
	    errorQuda("TIFR interface has not been built\n");
#endif
	  } else if (in.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
#ifdef BUILD_TIFR_INTERFACE
	    typedef FloatNOrder<FloatOut,18,2,11> momOut;
	    typedef TIFRPaddedOrder<FloatIn,18> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out, 0, override), momIn(in, In), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,18,momOut,momIn>(arg,in,location);
#else
	    errorQuda("TIFR interface has not been built\n");
#endif
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
	} else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
	  typedef MILCOrder<FloatOut,10> momOut;
#ifdef BUILD_MILC_INTERFACE
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    typedef FloatNOrder<FloatIn,10,2,10> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out), momIn(in, In, 0, override), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,10,momOut,momIn>(arg,in,location);
	  } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
	    typedef MILCOrder<FloatIn,10> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out), momIn(in, In), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,10,momOut,momIn>(arg,in,location);
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
#else
	  errorQuda("MILC interface has not been built\n");
#endif
	} else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {
	  typedef TIFROrder<FloatOut,18> momOut;
#ifdef BUILD_TIFR_INTERFACE
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    // FIX ME - 11 is a misnomer to avoid confusion in template instantiation
	    typedef FloatNOrder<FloatIn,18,2,11> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out), momIn(in, In, 0, override), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,18,momOut,momIn>(arg,in,location);
	  } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {
	    typedef TIFROrder<FloatIn,18> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out), momIn(in, In), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,18,momOut,momIn>(arg,in,location);
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
#else
	  errorQuda("TIFR interface has not been built\n");
#endif
	} else if (out.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	  typedef TIFRPaddedOrder<FloatOut,18> momOut;
#ifdef BUILD_TIFR_INTERFACE
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    // FIX ME - 11 is a misnomer to avoid confusion in template instantiation
	    typedef FloatNOrder<FloatIn,18,2,11> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out), momIn(in, In, 0, override), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,18,momOut,momIn>(arg,in,location);
	  } else if (in.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	    typedef TIFRPaddedOrder<FloatIn,18> momIn;
	    CopyGaugeArg<momOut,momIn> arg(momOut(out, Out), momIn(in, In), in.Volume(),
				     faceVolumeCB, in.Ndim(), in.Geometry());
	    copyMom<FloatOut,FloatIn,18,momOut,momIn>(arg,in,location);
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
#else
	  errorQuda("TIFR interface has not been built\n");
#endif
	} else {
	  errorQuda("Gauge field orders %d not supported", out.Order());
	}
    }
  }


} // namespace quda
