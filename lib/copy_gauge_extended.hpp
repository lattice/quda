#include <tunable_nd.h>
#include <instantiate.h>
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
    double scale;

    bool tuneSharedBytes() const override { return false; }
    unsigned int minThreads() const override { return in.VolumeCB() == out.VolumeCB() ? in.VolumeCB() : in.LocalVolumeCB(); }

  public:
    CopyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale_, FloatOut *Out,
                FloatIn *In) :
      TunableKernel2D(in, 2, location), out(out), in(in), location(location), Out(Out), In(In), scale(scale_)
    {
      strcat(aux, out.AuxString().c_str());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      constexpr bool enable_host = true;
      if (out.Volume() > in.Volume())
        launch<CopyGaugeEx_, enable_host>(tp, stream, Arg<true>(out, in, Out, In, scale));
      else
        launch<CopyGaugeEx_, enable_host>(tp, stream, Arg<false>(out, in, Out, In, scale));
    }

    long long bytes() const override
    { // only count interior sites
      return (out.LocalVolume() * out.Bytes()) / out.Volume() +  (in.LocalVolume() * in.Bytes()) / in.Volume();
    }
  };

  template <typename FloatOut, typename FloatIn, int length, typename InOrder>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, FloatOut *Out,
                   FloatIn *In)
  {
    if (out.isNative()) {
      if (out.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_NO>::type G;
        CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_12) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_12>()) {
          typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_12>::type G;
          CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
        }
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_8) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_8>()) {
          typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_8>::type G;
          CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
        }
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_13) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_13>()) {
          typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_13>::type G;
          CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
        }
      } else if (out.Reconstruct() == QUDA_RECONSTRUCT_9) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_9>()) {
          typedef typename gauge_mapper<FloatOut, QUDA_RECONSTRUCT_9>::type G;
          CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
        }
      } else {
	errorQuda("Reconstruction %d and order %d not supported", out.Reconstruct(), out.Order());
      }
    } else if (out.Order() == QUDA_QDP_GAUGE_ORDER) {

      if constexpr (is_enabled<QUDA_QDP_GAUGE_ORDER>()) {
        using G = QDPOrder<FloatOut, length>;
        CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
      } else {
        errorQuda("QDP interface has not been built\n");
      }

    } else if (out.Order() == QUDA_MILC_GAUGE_ORDER) {

      if constexpr (is_enabled<QUDA_MILC_GAUGE_ORDER>()) {
        using G = MILCOrder<FloatOut, length>;
        CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
      } else {
        errorQuda("MILC interface has not been built\n");
      }

    } else if (out.Order() == QUDA_TIFR_GAUGE_ORDER) {

      if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>()) {
        using G = TIFROrder<FloatOut, length>;
        CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
      } else {
        errorQuda("TIFR interface has not been built\n");
      }

    } else if (out.Order() == QUDA_OPENQCD_GAUGE_ORDER) {

      if constexpr (is_enabled<QUDA_OPENQCD_GAUGE_ORDER>()) {
        using G = OpenQCDOrder<FloatOut, length>;
        CopyGaugeEx<FloatOut, FloatIn, length, G, InOrder>(out, in, location, scale, Out, In);
      } else {
        errorQuda("OPENQCD interface has not been built");
      }

    } else {
      errorQuda("Gauge field %d order not supported", out.Order());
    }
  }

  template <typename FloatOut, typename FloatIn, int length>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, FloatOut *Out,
                   FloatIn *In)
  {
    if (in.isNative()) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_NO>::type G;
        copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_12>()) {
          typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_12>::type G;
          copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
        }
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_8>()) {
          typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_8>::type G;
          copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
        }
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_13>()) {
          typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_13>::type G;
          copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
        }
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
        if constexpr (is_enabled<QUDA_RECONSTRUCT_9>()) {
          typedef typename gauge_mapper<FloatIn, QUDA_RECONSTRUCT_9>::type G;
          copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
        } else {
          errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
        }
      } else {
	errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) {

      if constexpr (is_enabled<QUDA_QDP_GAUGE_ORDER>()) {
        using G = QDPOrder<FloatIn, length>;
        copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
      } else {
        errorQuda("QDP interface has not been built\n");
      }

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

      if constexpr (is_enabled<QUDA_MILC_GAUGE_ORDER>()) {
        using G = MILCOrder<FloatIn, length>;
        copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
      } else {
        errorQuda("MILC interface has not been built\n");
      }

    } else if (in.Order() == QUDA_TIFR_GAUGE_ORDER) {

      if constexpr (is_enabled<QUDA_TIFR_GAUGE_ORDER>()) {
        using G = TIFROrder<FloatIn, length>;
        copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
      } else {
        errorQuda("TIFR interface has not been built\n");
      }

    } else if (in.Order() == QUDA_OPENQCD_GAUGE_ORDER) {
      if constexpr (is_enabled<QUDA_OPENQCD_GAUGE_ORDER>()) {
        using G = OpenQCDOrder<FloatIn, length>;
        copyGaugeEx<FloatOut, FloatIn, length, G>(out, in, location, scale, Out, In);
      } else {
        errorQuda("OpenQCD interface has not been built\n");
      }

    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }
  }

  template <typename store_out_t, typename store_in_t>
  void copyGaugeEx(GaugeField &out, const GaugeField &in, QudaFieldLocation location, double scale, store_out_t *Out,
                   store_in_t *In)
  {
    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }

    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    if (in.LinkType() != QUDA_ASQTAD_MOM_LINKS && out.LinkType() != QUDA_ASQTAD_MOM_LINKS) {
      // we are doing gauge field packing
      copyGaugeEx<store_out_t, store_in_t, 18>(out, in, location, scale, Out, In);
    } else {
      errorQuda("Not supported");
    }
  }

} // namespace quda
