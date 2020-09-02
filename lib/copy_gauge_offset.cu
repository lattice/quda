#include <tune_quda.h>
#include <gauge_field_order.h>
#include <quda_matrix.h>

#include <instantiate.h>

namespace quda
{

  using namespace gauge;

  /**
     Kernel argument struct
   */
  template <class Float, int nColor, class Order> struct CopyGaugeOffsetArg {

    static constexpr int nDim = 4;

    using real = typename mapper<Float>::type;
    using Matrix = Matrix<complex<real>, nColor>;

    Order out;
    const Order in;

    int Xin[QUDA_MAX_DIM];
    int Xout[QUDA_MAX_DIM];

    int offset[QUDA_MAX_DIM];

    int volume_cb;

    bool collect_disperse; // Whether we are collecting (true) or dispersing (false)

    CopyGaugeOffsetArg(GaugeField &out, const GaugeField &in, const int *offset) : out(out), in(in)
    {
      for (int d = 0; d < nDim; d++) {
        this->Xout[d] = out.X()[d];
        this->Xin[d] = in.X()[d];
        this->offset[d] = offset[d];
      }

      if (out.VolumeCB() > in.VolumeCB()) {
        volume_cb = in.VolumeCB();
        collect_disperse = true;
      } else {
        volume_cb = out.VolumeCB();
        collect_disperse = false;
      }
    }
  };

  /**
     Copy a regular/extended gauge field into an extended/regular gauge field
  */
  template <bool collect_disperse, class Arg> __device__ __host__ void copyGaugeOffset(Arg &arg, int x_cb, int parity)
  {

    using Matrix = typename Arg::Matrix;

    int coordinate[4];
    int x_in;
    int x_out;
    if (collect_disperse) {
      // we are collecting so x_cb is the index for the input.
      x_in = x_cb;
      getCoordsExtended(coordinate, x_cb, arg.Xin, parity, arg.offset);
      x_out = linkIndex(coordinate, arg.Xout);
    } else {
      // we are dispersing so x_cb is the index for the output.
      x_out = x_cb;
      getCoordsExtended(coordinate, x_cb, arg.Xout, parity, arg.offset);
      x_in = linkIndex(coordinate, arg.Xin);
    }

#pragma unroll
    for (int d = 0; d < 4; d++) {
      Matrix mat = arg.in(d, x_in, parity);
      arg.out(d, x_out, parity) = mat;
    } // dir
  }

  template <bool collect_disperse, class Arg> void copyGaugeOffsetCpu(Arg arg)
  {
    for (int x_cb = 0; x_cb < arg.volume_cb; x_cb++) {
      for (int parity = 0; parity < 2; parity++) { copyGaugeOffset<collect_disperse>(arg, x_cb, parity); }
    }
  }

  template <bool collect_disperse, class Arg> __global__ void copyGaugeOffsetKernel(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.volume_cb) return;
#pragma unroll
    for (int parity = 0; parity < 2; parity++) { copyGaugeOffset<collect_disperse>(arg, x_cb, parity); }
  }

  template <class Float, int nColor, class Arg> class CopyGaugeOffset : Tunable
  {
    Arg arg;

    const GaugeField &meta; // use for metadata
    QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.volume_cb; }

  public:
    CopyGaugeOffset(GaugeField &out, const GaugeField &in, const Arg &arg) : arg(arg), meta(in), location(in.Location())
    {
      writeAuxString("volumn_out=%d,volume_in=%d,%s,offset=%d%d%d%d,location=%s", out.VolumeCB(), in.VolumeCB(),
                     arg.collect_disperse ? "collect" : "disperse", arg.offset[0], arg.offset[1], arg.offset[2],
                     arg.offset[3], location == QUDA_CPU_FIELD_LOCATION ? "cpu" : "gpu");
      apply(0);
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (arg.collect_disperse) {
          copyGaugeOffsetCpu<true>(arg);
        } else {
          copyGaugeOffsetCpu<false>(arg);
        }
      } else if (location == QUDA_CUDA_FIELD_LOCATION) {
        if (arg.collect_disperse) {
          copyGaugeOffsetKernel<true><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        } else {
          copyGaugeOffsetKernel<false><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        }
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; }
    long long bytes() const { return arg.in.Bytes() + arg.out.Bytes(); }
  };

  template <typename Float, int nColor> void copyGaugeOffset(GaugeField &out, const GaugeField &in, const int offset[4])
  {

    if (in.isNative()) {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_NO>::type;
        using Arg = CopyGaugeOffsetArg<Float, nColor, G>;
        Arg arg(out, in, offset);
        CopyGaugeOffset<Float, nColor, Arg> copier(out, in, arg);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
#if QUDA_RECONSTRUCT & 2
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_12>::type;
        using Arg = CopyGaugeOffsetArg<Float, nColor, G>;
        Arg arg(out, in, offset);
        CopyGaugeOffset<Float, nColor, Arg> copier(out, in, arg);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-12", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
#if QUDA_RECONSTRUCT & 1
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_8>::type;
        using Arg = CopyGaugeOffsetArg<Float, nColor, G>;
        Arg arg(out, in, offset);
        CopyGaugeOffset<Float, nColor, Arg> copier(out, in, arg);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-8", QUDA_RECONSTRUCT);
#endif
#ifdef GPU_STAGGERED_DIRAC
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
#if QUDA_RECONSTRUCT & 2
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_13>::type;
        using Arg = CopyGaugeOffsetArg<Float, nColor, G>;
        Arg arg(out, in, offset);
        CopyGaugeOffset<Float, nColor, Arg> copier(out, in, arg);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-13", QUDA_RECONSTRUCT);
#endif
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
#if QUDA_RECONSTRUCT & 1
        using G = typename gauge_mapper<Float, QUDA_RECONSTRUCT_9>::type;
        using Arg = CopyGaugeOffsetArg<Float, nColor, G>;
        Arg arg(out, in, offset);
        CopyGaugeOffset<Float, nColor, Arg> copier(out, in, arg);
#else
        errorQuda("QUDA_RECONSTRUCT=%d does not enable reconstruct-9", QUDA_RECONSTRUCT);
#endif
#endif // GPU_STAGGERED_DIRAC
      } else {
        errorQuda("Reconstruction %d and order %d not supported", in.Reconstruct(), in.Order());
      }
    } else if (in.Order() == QUDA_QDP_GAUGE_ORDER) { // TODO: Add other gauge field orders.
#ifdef BUILD_QDP_INTERFACE
      using G = typename gauge_order_mapper<Float, QUDA_QDP_GAUGE_ORDER, nColor>::type;
      using Arg = CopyGaugeOffsetArg<Float, nColor, G>;
      Arg arg(out, in, offset);
      CopyGaugeOffset<Float, nColor, Arg> copier(out, in, arg);
#else
      errorQuda("QDP interface has not been built\n");
#endif

    } else if (in.Order() == QUDA_MILC_GAUGE_ORDER) {

#ifdef BUILD_MILC_INTERFACE
      using G = typename gauge_order_mapper<Float, QUDA_MILC_GAUGE_ORDER, nColor>::type;
      using Arg = CopyGaugeOffsetArg<Float, nColor, G>;
      Arg arg(out, in, offset);
      CopyGaugeOffset<Float, nColor, Arg> copier(out, in, arg);
#else
      errorQuda("MILC interface has not been built\n");
#endif

    } else {
      errorQuda("Gauge field %d order not supported", in.Order());
    }
  }

  template <typename Float> void copyGaugeOffset(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    if (in.Ncolor() != 3 && out.Ncolor() != 3) {
      errorQuda("Unsupported number of colors; out.Nc=%d, in.Nc=%d", out.Ncolor(), in.Ncolor());
    }

    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    copyGaugeOffset<Float, 3>(out, in, offset);
  }

  void copyOffsetGauge(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);

    switch (in.Precision()) {
    case QUDA_DOUBLE_PRECISION: copyGaugeOffset<double>(out, in, offset); break;
    case QUDA_SINGLE_PRECISION: copyGaugeOffset<float>(out, in, offset); break;
    case QUDA_HALF_PRECISION: copyGaugeOffset<short>(out, in, offset); break;
    case QUDA_QUARTER_PRECISION: copyGaugeOffset<int8_t>(out, in, offset); break;
    default: errorQuda("unknown precision.");
    }
  }

} // namespace quda
