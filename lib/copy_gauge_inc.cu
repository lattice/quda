#include <gauge_field_order.h>
#include <copy_gauge_helper.cuh>
#include <instantiate.h>

namespace quda {

  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGauge(const InOrder &inOrder, GaugeField &out, const GaugeField &in,
		 QudaFieldLocation location, FloatOut *Out, FloatOut **outGhost, int type) {
    if (out.isNative()) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_NO>::type G;
        copyGauge<FloatOut, FloatIn, length>(G(out, Out, outGhost), inOrder, out, in, location, type);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_12>::type G;
        copyGauge<FloatOut, FloatIn, length>(G(out, Out, outGhost), inOrder, out, in, location, type);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_8>::type G;
        copyGauge<FloatOut, FloatIn, length>(G(out, Out, outGhost), inOrder, out, in, location, type);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_13>::type G;
        copyGauge<FloatOut, FloatIn, length>(G(out, Out, outGhost), inOrder, out, in, location, type);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        if (out.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
          typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_9, 2*N_COLORS*N_COLORS, QUDA_STAGGERED_PHASE_MILC>::type G;
          copyGauge<FloatOut, FloatIn, length>(G(out, Out, outGhost), inOrder, out, in, location, type);
        } else if (out.StaggeredPhase() == QUDA_STAGGERED_PHASE_TIFR) {
#ifdef BUILD_TIFR_INTERFACE
          typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_9, 2*N_COLORS*N_COLORS, QUDA_STAGGERED_PHASE_TIFR>::type G;
          copyGauge<FloatOut, FloatIn, length>(G(out, Out, outGhost), inOrder, out, in, location, type);
#else
          errorQuda("TIFR interface has not been built so TIFR phase type not enabled\n");
#endif
        } else if (out.StaggeredPhase() == QUDA_STAGGERED_PHASE_NO) {
          typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_9>::type G;
          copyGauge<FloatOut, FloatIn, length>(G(out, Out, outGhost), inOrder, out, in, location, type);
        } else {
          errorQuda("Staggered phase type %d not supported", out.StaggeredPhase());
        }
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
        errorQuda("Reconstruction %d and order %d not supported", out.Reconstruct(), out.Order());
      }
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(QDPOrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(QDPJITOrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(CPSOrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("CPS interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(MILCOrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(MILCSiteOrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(BQCDOrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(TIFROrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>
	(TIFRPaddedOrder<FloatOut,length>(out, Out, outGhost), inOrder, out, in, location, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
  void copyGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, FloatOut *Out, FloatIn *In,
      FloatOut **outGhost, FloatIn **inGhost, int type)
  {

    // reconstruction only supported on FloatN fields currently
    if (in.isNative()) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_NO>::type G;
        copyGauge<FloatOut, FloatIn, length>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_12>::type G;
	copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost), out, in, location, Out, outGhost, type);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_8>::type G;
	copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost), out, in, location, Out, outGhost, type);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_13>::type G;
	copyGauge<FloatOut,FloatIn,length> (G(in,In,inGhost), out, in, location, Out, outGhost, type);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        if (in.StaggeredPhase() == QUDA_STAGGERED_PHASE_MILC) {
          typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_9, 2*N_COLORS*N_COLORS, QUDA_STAGGERED_PHASE_MILC>::type G;
          copyGauge<FloatOut, FloatIn, length>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
        } else if (in.StaggeredPhase() == QUDA_STAGGERED_PHASE_NO) {
          typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_9>::type G;
          copyGauge<FloatOut, FloatIn, length>(G(in, In, inGhost), out, in, location, Out, outGhost, type);
        } else {
          errorQuda("Staggered phase type %d not supported", in.StaggeredPhase());
        }
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
	errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(QDPOrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_QDPJIT_GAUGE_ORDER) {

#ifdef BUILD_QDPJIT_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(QDPJITOrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("QDPJIT interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_CPS_WILSON_GAUGE_ORDER) {

#ifdef BUILD_CPS_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(CPSOrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("CPS interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(MILCOrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(MILCSiteOrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_BQCD_GAUGE_ORDER) {

#ifdef BUILD_BQCD_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(BQCDOrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("BQCD interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(TIFROrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      copyGauge<FloatOut,FloatIn,length>(TIFRPaddedOrder<FloatIn,length>(in, In, inGhost),
					 out, in, location, Out, outGhost, type);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field order %d not supported", in.Order());
    }
  }

  void checkMomOrder(const GaugeField &u);

  template <typename Arg>
  void copyMom(Arg &arg, GaugeField &out, const GaugeField &in, QudaFieldLocation location) {
    CopyGauge<Arg> momCopier(arg, out, in, location);
    momCopier.apply(device::get_default_stream());
  }

  template <typename FloatOut, typename FloatIn> struct GaugeCopy {
    GaugeCopy(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out_, void *In_,
              void **outGhost_, void **inGhost_, int type)
    {
      FloatOut *Out = reinterpret_cast<FloatOut*>(Out_);
      FloatIn *In = reinterpret_cast<FloatIn*>(In_);
      FloatOut **outGhost = reinterpret_cast<FloatOut**>(outGhost_);
      FloatIn **inGhost = reinterpret_cast<FloatIn**>(inGhost_);

      if (in.Ncolor() != N_COLORS && out.Ncolor() != N_COLORS) {
        errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
      }

      if (out.Geometry() != in.Geometry()) {
        errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
      }

      if (in.LinkType() != QUDA_ASQTAD_MOM_LINKS && out.LinkType() != QUDA_ASQTAD_MOM_LINKS) {
        // we are doing gauge field packing
        copyGauge<FloatOut,FloatIn,2*N_COLORS*N_COLORS>(out, in, location, Out, In, outGhost, inGhost, type);
      } else {
        if (out.Geometry() != QUDA_VECTOR_GEOMETRY) errorQuda("Unsupported geometry %d", out.Geometry());

        checkMomOrder(in);
        checkMomOrder(out);

	constexpr int MILC_mom_size = N_COLORS*N_COLORS+1;
	
        // momentum only currently supported on MILC (10), TIFR (18) and Float2 (10) fields currently
	if (out.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    typedef FloatNOrder<FloatOut,MILC_mom_size,2,MILC_mom_size> momOut;
	    typedef FloatNOrder<FloatIn,MILC_mom_size,2,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out, 0), momIn(in, In, 0), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {
#ifdef BUILD_QDP_INTERFACE
	    typedef FloatNOrder<FloatOut,MILC_mom_size,2,MILC_mom_size> momOut;
	    typedef QDPOrder<FloatIn,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out, 0), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
#else
	    errorQuda("QDP interface has not been built\n");
#endif
	  } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
#ifdef BUILD_MILC_INTERFACE
	    typedef FloatNOrder<FloatOut,MILC_mom_size,2,MILC_mom_size> momOut;
	    typedef MILCOrder<FloatIn,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out, 0), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
#else
	    errorQuda("MILC interface has not been built\n");
#endif
	  } else if (in.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {
#ifdef BUILD_MILC_INTERFACE
	    typedef FloatNOrder<FloatOut,MILC_mom_size,2,MILC_mom_size> momOut;
	    typedef MILCSiteOrder<FloatIn,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out, 0), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
#else
	    errorQuda("MILC interface has not been built\n");
#endif
	  } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {
#ifdef BUILD_TIFR_INTERFACE
	    typedef FloatNOrder<FloatOut,2*N_COLORS*N_COLORS,2,11> momOut;
	    typedef TIFROrder<FloatIn,2*N_COLORS*N_COLORS> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, 2*N_COLORS*N_COLORS, momOut, momIn> arg(momOut(out, Out, 0), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
#else
	    errorQuda("TIFR interface has not been built\n");
#endif
	  } else if (in.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
#ifdef BUILD_TIFR_INTERFACE
	    typedef FloatNOrder<FloatOut,2*N_COLORS*N_COLORS,2,11> momOut;
	    typedef TIFRPaddedOrder<FloatIn,2*N_COLORS*N_COLORS> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, 2*N_COLORS*N_COLORS, momOut, momIn> arg(momOut(out, Out, 0), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
#else
	    errorQuda("TIFR interface has not been built\n");
#endif
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
	} else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {
#ifdef BUILD_QDP_INTERFACE
	  typedef QDPOrder<FloatOut,MILC_mom_size> momOut;
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    typedef FloatNOrder<FloatIn,MILC_mom_size,2,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out), momIn(in, In, 0), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
	    typedef MILCOrder<FloatIn,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
#else
	  errorQuda("QDP interface has not been built\n");
#endif
	} else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {
#ifdef BUILD_MILC_INTERFACE
	  typedef MILCOrder<FloatOut,MILC_mom_size> momOut;
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    typedef FloatNOrder<FloatIn,MILC_mom_size,2,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out), momIn(in, In, 0), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {
	    typedef MILCOrder<FloatIn,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
          } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {
	    typedef QDPOrder<FloatIn,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
#else
	  errorQuda("MILC interface has not been built\n");
#endif
	} else if (out.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {
#ifdef BUILD_MILC_INTERFACE
	  typedef MILCSiteOrder<FloatOut,MILC_mom_size> momOut;
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    typedef FloatNOrder<FloatIn,MILC_mom_size,2,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out), momIn(in, In, 0), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else if (in.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {
	    typedef MILCSiteOrder<FloatIn,MILC_mom_size> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, MILC_mom_size, momOut, momIn> arg(momOut(out, Out), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
#else
	  errorQuda("MILC interface has not been built\n");
#endif
	} else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {
#ifdef BUILD_TIFR_INTERFACE
	  typedef TIFROrder<FloatOut,2*N_COLORS*N_COLORS> momOut;
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    // FIX ME - 11 is a misnomer to avoid confusion in template instantiation
	    typedef FloatNOrder<FloatIn,2*N_COLORS*N_COLORS,2,11> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, 2*N_COLORS*N_COLORS, momOut, momIn> arg(momOut(out, Out), momIn(in, In, 0), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {
	    typedef TIFROrder<FloatIn,2*N_COLORS*N_COLORS> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, 2*N_COLORS*N_COLORS, momOut, momIn> arg(momOut(out, Out), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else {
	    errorQuda("Gauge field orders %d not supported", in.Order());
	  }
#else
	  errorQuda("TIFR interface has not been built\n");
#endif
	} else if (out.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
#ifdef BUILD_TIFR_INTERFACE
	  typedef TIFRPaddedOrder<FloatOut,2*N_COLORS*N_COLORS> momOut;
	  if (in.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
	    // FIX ME - 11 is a misnomer to avoid confusion in template instantiation
	    typedef FloatNOrder<FloatIn,2*N_COLORS*N_COLORS,2,11> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, 2*N_COLORS*N_COLORS, momOut, momIn> arg(momOut(out, Out), momIn(in, In, 0), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
	  } else if (in.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	    typedef TIFRPaddedOrder<FloatIn,2*N_COLORS*N_COLORS> momIn;
	    CopyGaugeArg<FloatOut, FloatIn, 2*N_COLORS*N_COLORS, momOut, momIn> arg(momOut(out, Out), momIn(in, In), in);
	    copyMom<decltype(arg)>(arg,out,in,location);
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
  };

  template <typename FloatIn>
  void copyGenericGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out, void *In,
                        void **ghostOut, void **ghostIn, int type)
  {
    instantiatePrecision2<GaugeCopy, FloatIn>(out, in, location, Out, In, ghostOut, ghostIn, type);
  }

} // namespace quda
