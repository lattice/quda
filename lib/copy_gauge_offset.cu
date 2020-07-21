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
  template <class Order> struct CopyGaugeOffsetArg {

    static constexpr int nDim = 4;

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
  template <class Float, int nColor, bool collect_disperse, class Arg>
  __device__ __host__ void copyGaugeOffset(Arg &arg, int x_cb, int parity)
  {
    using real = typename mapper<Float>::type;
    using Matrix = Matrix<complex<real>, nColor>;

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

  template <class Float, int nColor, bool collect_disperse, class Arg> void copyGaugeOffsetCpu(Arg arg)
  {
    for (int x_cb = 0; x_cb < arg.volume_cb; x_cb++) {
      for (int parity = 0; parity < 2; parity++) {
        copyGaugeOffset<Float, nColor, collect_disperse>(arg, x_cb, parity);
      }
    }
  }

  template <class Float, int nColor, bool collect_disperse, class Arg> __global__ void copyGaugeOffsetKernel(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    if (x_cb >= arg.volume_cb) return;
#pragma unroll
    for (int parity = 0; parity < 2; parity++) { copyGaugeOffset<Float, nColor, collect_disperse>(arg, x_cb, parity); }
  }

  template <class Float, int nColor, QudaReconstructType recon> class CopyGaugeOffset : Tunable
  {
    using Order = typename gauge_mapper<Float, recon>::type;

    using Arg = CopyGaugeOffsetArg<Order>;
    Arg arg;

    const GaugeField &meta; // use for metadata
    QudaFieldLocation location;

  private:
    unsigned int sharedBytesPerThread() const { return 0; }
    unsigned int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.volume_cb; }

  public:
    CopyGaugeOffset(GaugeField &out, const GaugeField &in, const int offset[4]) :
      arg(out, in, offset), meta(in), location(out.Location())
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
          copyGaugeOffsetCpu<Float, nColor, true>(arg);
        } else {
          copyGaugeOffsetCpu<Float, nColor, false>(arg);
        }
      } else if (location == QUDA_CUDA_FIELD_LOCATION) {
        if (arg.collect_disperse) {
          copyGaugeOffsetKernel<Float, nColor, true><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        } else {
          copyGaugeOffsetKernel<Float, nColor, false><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
        }
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    long long flops() const { return 0; }
    long long bytes() const { return arg.in.Bytes() + arg.out.Bytes(); }
  };

  void copyOffsetGauge(GaugeField &out, const GaugeField &in, const int offset[4])
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    // if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    // if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());

    instantiate<CopyGaugeOffset, WilsonReconstruct>(out, in, offset);
  }

} // namespace quda
