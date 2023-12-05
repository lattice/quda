#include <tunable_nd.h>
#include <kernels/copy_gauge_extended.cuh>

namespace quda {

  template <typename FloatOut, typename FloatIn, int length, typename OutOrder, typename InOrder>
  class CopyGaugeEx : TunableKernel2D {
    template <bool expand> using Arg = CopyGaugeExArg<FloatOut, FloatIn, length, OutOrder, InOrder, expand>;
    GaugeField &out;
    const GaugeField &in;
    QudaFieldLocation location;
    FloatOut *Out;
    FloatIn *In;

    unsigned int minThreads() const { return in.VolumeCB() == out.VolumeCB() ? in.VolumeCB() : in.LocalVolumeCB(); }

  public:
    CopyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
                FloatOut *Out, FloatIn *In) :
      TunableKernel2D(in, 2, location),
      out(out),
      in(in),
      location(location),
      Out(Out),
      In(In)
    {
      strcat(aux, out.AuxString().c_str());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      constexpr bool enable_host = true;
      if (out.Volume() > in.Volume()) launch<CopyGaugeEx_, enable_host>(tp, stream, Arg<true>(out, in, Out, In));
      else                            launch<CopyGaugeEx_, enable_host>(tp, stream, Arg<false>(out, in, Out, In));
    }

    long long flops() const { return 0; }
    long long bytes() const
    { // only count interior sites
      return (out.LocalVolume() * out.Bytes()) / out.Volume() +  (in.LocalVolume() * in.Bytes()) / in.Volume();
    }
  };

  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location, FloatOut *Out, FloatIn *In)
  {
    if (out.isNative()) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_NO>::type G;
        CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, Out, In);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_12>::type G;
	CopyGaugeEx<FloatOut,FloatIn,length, G, InOrder>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_8>::type G;
	CopyGaugeEx<FloatOut,FloatIn,length, G, InOrder>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_13>::type G;
        CopyGaugeEx<FloatOut,FloatIn,length, G, InOrder>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatOut,QUDA_RECONSTRUCT_9>::type G;
        CopyGaugeEx<FloatOut,FloatIn,length, G, InOrder>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
	errorQuda("Reconstruction %d and order %d not supported", out.Reconstruct(), out.Order());
      }
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      using G = QDPOrder<FloatOut,length>;
      CopyGaugeEx<FloatOut,FloatIn,length, G, InOrder>(out, in, location, Out, In);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      using G = MILCOrder<FloatOut, length>;
      CopyGaugeEx<FloatOut,FloatIn,length, G, InOrder>(out, in, location, Out, In);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      using G = TIFROrder<FloatOut,length>;
      CopyGaugeEx<FloatOut,FloatIn,length, G, InOrder>(out, in, location, Out, In);
#else
      errorQuda("TIFR interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }

  }

  template <typename FloatOut, typename FloatIn, int length>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location,
		   FloatOut *Out, FloatIn *In)
  {
    if (in.isNative()) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_NO>::type G;
        copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, Out, In);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_12>::type G;
	copyGaugeEx<FloatOut,FloatIn,length, G>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_8>::type G;
	copyGaugeEx<FloatOut,FloatIn,length, G>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_13>::type G;
	copyGaugeEx<FloatOut,FloatIn,length, G>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        typedef typename gauge_mapper<FloatIn,QUDA_RECONSTRUCT_9>::type G;
	copyGaugeEx<FloatOut,FloatIn,length, G>(out, in, location, Out, In);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
	errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

#ifdef BUILD_QDP_INTERFACE
      using G = QDPOrder<FloatIn, length>;
      copyGaugeEx<FloatOut,FloatIn,length, G>(out, in, location, Out, In);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      using G = MILCOrder<FloatIn, length>;
      copyGaugeEx<FloatOut,FloatIn,length, G>(out, in, location, Out, In);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {

#ifdef BUILD_TIFR_INTERFACE
      using G = TIFROrder<FloatIn,length>;
      copyGaugeEx<FloatOut,FloatIn,length, G>(out, in, location, Out, In);
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
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        copyGaugeEx(out, in, location, (double*)Out, (short*)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        copyGaugeEx(out, in, location, (double*)Out, (int8_t*)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
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
      } else if (in.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        copyGaugeEx(out, in, location, (float *)Out, (short *)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else if (in.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        copyGaugeEx(out, in, location, (float *)Out, (int8_t *)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Precision %d not instantiated", in.Precision());
      }
    } else if (out.Precision() == QUDA_HALF_PRECISION) {
      if (in.Precision() == QUDA_HALF_PRECISION) {
#if QUDA_PRECISION & 2
        copyGaugeEx(out, in, location, (short *)Out, (short *)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable half precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Precision %d not instantiated", in.Precision());
      }
    } else if (out.Precision() == QUDA_QUARTER_PRECISION) {
      if (in.Precision() == QUDA_QUARTER_PRECISION) {
#if QUDA_PRECISION & 1
        copyGaugeEx(out, in, location, (int8_t *)Out, (int8_t *)In);
#else
        errorQuda("QUDA_PRECISION=%d does not enable quarter precision", QUDA_PRECISION);
#endif
      } else {
        errorQuda("Precision %d not instantiated", in.Precision());
      }
    } else {
      errorQuda("Precision %d not instantiated", out.Precision());
    }
  }

} // namespace quda
