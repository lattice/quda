#include <lattice_field.h>
#include <index_helper.cuh>

namespace quda
{

  template <class Field_, class Element_, class Accessor_> struct CopyFieldOffsetArg {

    static constexpr int nDim = 4; // No matter what the underlying field is, the dimension is 4

    using Field = Field_;
    using Element = Element_;
    using Accessor = Accessor_;

    Accessor out;      // output field
    const Accessor in; // input field

    int Xin[nDim];
    int Xout[nDim];

    int offset[nDim];

    int nParity;

    int volume_cb_in;
    int volume_4d_cb_in;

    int volume_cb_out;
    int volume_4d_cb_out;

    int volume_cb;
    int volume_4d_cb;

    int Ls; // The fifth dimension size

    QudaOffsetCopyMode mode;

    CopyFieldOffsetArg(Field &out, const Field &in, const int offset[4]) :
      out(out), in(in), nParity(in.SiteSubset())
    {
      const int *X_in = in.X();
      const int *X_out = out.X();

      Ls = in.Ndim() == 4 ? 1 : X_in[4];

      if (Ls > 1 && X_out[4] != Ls) { errorQuda("Ls mismatch: in: %d, out: %d", X_out[4], Ls); }

      for (int d = 0; d < nDim; d++) {
        this->Xout[d] = X_out[d];
        this->Xin[d] = X_in[d];
        this->offset[d] = offset[d];
      }

      volume_cb_in = in.VolumeCB();
      volume_cb_out = out.VolumeCB();

      if (volume_cb_out > volume_cb_in) {
        volume_cb = volume_cb_in;
        mode = QudaOffsetCopyMode::COLLECT;
      } else {
        volume_cb = volume_cb_out;
        mode = QudaOffsetCopyMode::DISPERSE;
      }

      volume_4d_cb_in = volume_cb_in / Ls;
      volume_4d_cb_out = volume_cb_out / Ls;
      volume_4d_cb = volume_cb / Ls;
    }

  };

  template <class Arg>
  __device__ __host__ inline typename std::enable_if<std::is_same<typename Arg::Field, ColorSpinorField>::value, void>::type copy_field(int out, int in, int parity, Arg &arg)
  {
    using Element = typename Arg::Element;

    Element element = arg.in(in, parity);
    arg.out(out, parity) = element;
  }

  template <class Arg>
  __device__ __host__ inline typename std::enable_if<std::is_same<typename Arg::Field, GaugeField>::value, void>::type copy_field(int out, int in, int parity, Arg &arg)
  {
    using Element = typename Arg::Element;
#pragma unroll
    for (int d = 0; d < 4; d++) {
      Element element = arg.in(d, in, parity);
      arg.out(d, out, parity) = element;
    }
  }

  template <QudaOffsetCopyMode mode, class Arg>
  __device__ __host__ void copy_field_offset(int x_cb, int s, int parity, Arg &arg)
  {
    int coordinate[4];
    int x_in;
    int x_out;
    if (mode == QudaOffsetCopyMode::COLLECT) {
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

    copy_field(s * arg.volume_4d_cb_out + x_out, s * arg.volume_4d_cb_in + x_in, parity, arg);
  }

  template <QudaOffsetCopyMode mode, class Arg> __global__ void copy_field_offset_kernel(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int parity = blockIdx.z * blockDim.z + threadIdx.z;

    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    copy_field_offset<mode>(x_cb, s, parity, arg);
  }

  template <QudaOffsetCopyMode mode, class Arg> __host__ void copy_field_offset_cpu(Arg arg)
  {
    for (int parity = 0; parity < arg.nParity; parity++) {
      for (int s = 0; s < arg.Ls; s++) {
        for (int x_cb = 0; x_cb < arg.volume_4d_cb; x_cb++) {
          copy_field_offset<mode>(x_cb, s, parity, arg);
        }
      }
    }
  }

  template <class Arg> class CopyFieldOffset : public TunableVectorYZ
  {

    using Field = typename Arg::Field;
  protected:
    Arg &arg;
    const Field &meta;
    QudaFieldLocation location;

    long long flops() const { return 0; }
    long long bytes() const { return 2ll * (arg.mode == QudaOffsetCopyMode::COLLECT ? arg.in.Bytes() : arg.out.Bytes()); }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volume_4d_cb; }
    int blockStep() const { return 4; }
    int blockMin() const { return 4; }
    unsigned int sharedBytesPerThread() const { return 0; }

  public:
    CopyFieldOffset(Arg &arg, const Field &meta) :
      TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta), location(meta.Location())
    {
      writeAuxString("(%d,%d,%d,%d)->(%d,%d,%d,%d),Ls=%d,nParity=%d,%s,offset=%d%d%d%d,location=%s", arg.Xin[0], arg.Xin[1],
                     arg.Xin[2], arg.Xin[3], arg.Xout[0], arg.Xout[1], arg.Xout[2], arg.Xout[3], arg.Ls, arg.nParity,
                     arg.mode == QudaOffsetCopyMode::COLLECT ? "COLLECT" : "DISPERSE", arg.offset[0], arg.offset[1], arg.offset[2],
                     arg.offset[3], location == QUDA_CPU_FIELD_LOCATION ? "CPU" : "GPU");
      apply(0);
    }

    virtual ~CopyFieldOffset() { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (arg.mode == QudaOffsetCopyMode::COLLECT) {
          copy_field_offset_cpu<QudaOffsetCopyMode::COLLECT>(arg);
        } else {
          copy_field_offset_cpu<QudaOffsetCopyMode::DISPERSE>(arg);
        }
      } else {
        auto kernel = arg.mode == QudaOffsetCopyMode::COLLECT ? copy_field_offset_kernel<QudaOffsetCopyMode::COLLECT, Arg> :
                                 copy_field_offset_kernel<QudaOffsetCopyMode::DISPERSE, Arg>;
        kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    virtual bool advanceTuneParam(TuneParam &param) const
    {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        return false;
      } else {
        return TunableVectorYZ::advanceTuneParam(param);
      }
    }
  };

} // namespace quda
