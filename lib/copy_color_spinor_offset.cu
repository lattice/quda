#include <color_spinor_field.h>
#include <color_spinor_field_order.h>
#include <dslash_quda.h>
#include <index_helper.cuh>

#include <kernels/dslash_domain_wall_m5.cuh>

namespace quda
{

  template <typename Float, int nColor, class F> struct CopyColorSpinorOffsetArg {

    static constexpr int nDim = 4;

    using real = typename mapper<Float>::type;
    using Vector = ColorSpinor<real, nColor, 4>;

    F out;      // output vector field
    const F in; // input vector field

    int Xin[QUDA_MAX_DIM];
    int Xout[QUDA_MAX_DIM];

    int offset[QUDA_MAX_DIM];

    int nParity;

    int volume_cb_in;
    int volume_4d_cb_in;

    int volume_cb_out;
    int volume_4d_cb_out;

    int volume_cb;
    int volume_4d_cb;

    int Ls;

    bool collect_disperse;

    CopyColorSpinorOffsetArg(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4]) :
      out(out), in(in), nParity(in.SiteSubset())
    {
      if (in.Nspin() != 4) { errorQuda("nSpin = %d not support", in.Nspin()); }
      // if (!in.isNative() || !out.isNative()) {
      //   errorQuda("Unsupported field order out=%d in=%d\n", out.FieldOrder(), in.FieldOrder());
      // }

      Ls = (in.Ndim() == 4 ? 1 : in.X(4));

      if (Ls > 1 && out.X(4) != Ls) { errorQuda("Ls mismatch: in: %d, out: %d", out.X(4), Ls); }

      for (int d = 0; d < nDim; d++) {
        this->Xout[d] = out.X(d);
        this->Xin[d] = in.X(d);
        this->offset[d] = offset[d];
      }

      if (out.VolumeCB() > in.VolumeCB()) {
        volume_cb_in = in.VolumeCB();
        volume_cb_out = out.VolumeCB();
        volume_cb = volume_cb_in;
        collect_disperse = true;
      } else {
        volume_cb_in = in.VolumeCB();
        volume_cb_out = out.VolumeCB();
        volume_cb = volume_cb_out;
        collect_disperse = false;
      }
      volume_4d_cb_in = volume_cb_in / Ls;
      volume_4d_cb_out = volume_cb_out / Ls;
      volume_4d_cb = volume_cb / Ls;
    }
  };

  template <bool collect_disperse, class Arg>
  __device__ __host__ void copy_color_spinor_offset(int x_cb, int s, int parity, Arg &arg)
  {
    using Vector = typename Arg::Vector;

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

    Vector v = arg.in(s * arg.volume_4d_cb_in + x_in, parity);
    arg.out(s * arg.volume_4d_cb_out + x_out, parity) = v;
  }

  template <bool collect_disperse, class Arg> __global__ void copy_color_spinor_offset_kernel(Arg arg)
  {
    int x_cb = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y * blockDim.y + threadIdx.y;
    int parity = blockIdx.z * blockDim.z + threadIdx.z;

    if (x_cb >= arg.volume_4d_cb) return;
    if (s >= arg.Ls) return;
    if (parity >= arg.nParity) return;

    copy_color_spinor_offset<collect_disperse>(x_cb, s, parity, arg);
  }

  template <bool collect_disperse, class Arg> __host__ void copy_color_spinor_offset_cpu(Arg arg)
  {
    for (int parity = 0; parity < arg.nParity; parity++) {
      for (int s = 0; s < arg.Ls; s++) {
        for (int x_cb = 0; x_cb < arg.volume_4d_cb; x_cb++) {
          copy_color_spinor_offset<collect_disperse>(x_cb, s, parity, arg);
        }
      }
    }
  }

  template <class Float, int nColor, class Arg> class CopyColorSpinorOffset : public TunableVectorYZ
  {

  protected:
    Arg &arg;
    const ColorSpinorField &meta;
    QudaFieldLocation location;

    long long flops() const { return 0; }
    long long bytes() const { return 2ll * (arg.collect_disperse ? arg.in.Bytes() : arg.out.Bytes()); }

    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.volume_4d_cb; }
    int blockStep() const { return 4; }
    int blockMin() const { return 4; }
    unsigned int sharedBytesPerThread() const { return 0; }

  public:
    CopyColorSpinorOffset(Arg &arg, const ColorSpinorField &meta) :
      TunableVectorYZ(arg.Ls, arg.nParity), arg(arg), meta(meta), location(meta.Location())
    {
      writeAuxString("(%d,%d,%d,%d)->(%d,%d,%d,%d),Ls=%d,%s,offset=%d%d%d%d,location=%s", arg.Xin[0], arg.Xin[1],
                     arg.Xin[2], arg.Xin[3], arg.Xout[0], arg.Xout[1], arg.Xout[2], arg.Xout[3], arg.Ls,
                     arg.collect_disperse ? "collect" : "disperse", arg.offset[0], arg.offset[1], arg.offset[2],
                     arg.offset[3], location == QUDA_CPU_FIELD_LOCATION ? "CPU" : "GPU");
      apply(0);
    }
    virtual ~CopyColorSpinorOffset() { }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (location == QUDA_CPU_FIELD_LOCATION) {
        if (arg.collect_disperse) {
          copy_color_spinor_offset_cpu<true>(arg);
        } else {
          copy_color_spinor_offset_cpu<false>(arg);
        }
      } else {
        auto kernel = arg.collect_disperse ? copy_color_spinor_offset_kernel<true, Arg> :
                                             copy_color_spinor_offset_kernel<false, Arg>;
        kernel<<<tp.grid, tp.block, tp.shared_bytes, stream>>>(arg);
      }
    }

    void initTuneParam(TuneParam &param) const { TunableVectorYZ::initTuneParam(param); }

    void defaultTuneParam(TuneParam &param) const { TunableVectorYZ::defaultTuneParam(param); }

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

  template <typename Float, int nColor>
  void copy_color_spinor_offset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    if (in.Location() == QUDA_CPU_FIELD_LOCATION) {
      if (in.FieldOrder() == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) {

        using F = typename colorspinor_order_mapper<Float, QUDA_SPACE_COLOR_SPIN_FIELD_ORDER, 4, nColor>::type;

        CopyColorSpinorOffsetArg<Float, nColor, F> arg(out, in, offset);
        CopyColorSpinorOffset<Float, nColor, decltype(arg)> dummy(arg, in);

      } else if (in.FieldOrder() == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) {

        using F = typename colorspinor_order_mapper<Float, QUDA_SPACE_SPIN_COLOR_FIELD_ORDER, 4, nColor>::type;

        CopyColorSpinorOffsetArg<Float, nColor, F> arg(out, in, offset);
        CopyColorSpinorOffset<Float, nColor, decltype(arg)> dummy(arg, in);

      } else {

        errorQuda("Unsupported field order = %d.", in.FieldOrder());
      }

    } else {

      if (!in.isNative() || !out.isNative()) { errorQuda("CUDA field has be in native order."); }

      using F = typename colorspinor_mapper<Float, 4, nColor>::type;
      CopyColorSpinorOffsetArg<Float, nColor, F> arg(out, in, offset);
      CopyColorSpinorOffset<Float, nColor, decltype(arg)> dummy(arg, in);
    }
  }

  // template on the number of colors
  template <typename Float>
  void copy_color_spinor_offset(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    switch (in.Ncolor()) {
    case 3: copy_color_spinor_offset<Float, 3>(out, in, offset); break;
    default: errorQuda("Unsupported number of colors %d\n", in.Ncolor());
    }
  }

  // Apply the 5th dimension dslash operator to a colorspinor field
  // out = Dslash5*in
  void copyOffsetColorSpinor(ColorSpinorField &out, const ColorSpinorField &in, const int offset[4])
  {
    checkLocation(out, in); // check all locations match

    switch (checkPrecision(out, in)) {
    case QUDA_DOUBLE_PRECISION: copy_color_spinor_offset<double>(out, in, offset); break;
    case QUDA_SINGLE_PRECISION: copy_color_spinor_offset<float>(out, in, offset); break;
    case QUDA_HALF_PRECISION: copy_color_spinor_offset<short>(out, in, offset); break;
    case QUDA_QUARTER_PRECISION: copy_color_spinor_offset<char>(out, in, offset); break;
    default: errorQuda("Unsupported precision %d\n", in.Precision());
    }
  }

} // namespace quda
