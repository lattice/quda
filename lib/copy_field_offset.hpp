#include <kernels/copy_field_offset.cuh>
#include <tunable_nd.h>

namespace quda
{

  template <class Arg> class CopyFieldOffset : public TunableKernel3D
  {
    using Field = typename Arg::Field;

  protected:
    Arg &arg;
    const Field &meta;

    long long bytes() const
    {
      return 2ll * (arg.mode == QudaOffsetCopyMode::COLLECT ? arg.in.Bytes() : arg.out.Bytes());
    }

    unsigned int minThreads() const { return arg.volume_4d_cb; }
    int blockStep() const { return 4; }
    int blockMin() const { return 4; }

  public:
    CopyFieldOffset(Arg &arg, const Field &meta) : TunableKernel3D(meta, arg.Ls, arg.nParity), arg(arg), meta(meta)
    {
      char tmp[TuneKey::aux_n];
      sprintf(tmp, ",(%d,%d,%d,%d)->(%d,%d,%d,%d),Ls=%d,nParity=%d,%s,offset=%d%d%d%d", static_cast<int>(arg.dim_in[0]),
              static_cast<int>(arg.dim_in[1]), static_cast<int>(arg.dim_in[2]), static_cast<int>(arg.dim_in[3]),
              static_cast<int>(arg.dim_out[0]), static_cast<int>(arg.dim_out[1]), static_cast<int>(arg.dim_out[2]),
              static_cast<int>(arg.dim_out[3]), arg.Ls, arg.nParity,
              arg.mode == QudaOffsetCopyMode::COLLECT ? "COLLECT" : "DISPERSE", arg.offset[0], arg.offset[1],
              arg.offset[2], arg.offset[3]);
      strcat(aux, tmp);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<copy_field_offset, true>(tp, stream, arg);
    }
  };

} // namespace quda
