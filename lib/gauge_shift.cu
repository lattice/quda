#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/gauge_shift.cuh>

namespace quda
{

  template <typename Float, int nColor> class GaugeShifter : public TunableKernel3D
  {
    GaugeField &out;
    const GaugeField &in;
    int shift;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    GaugeShifter(GaugeField &out, const GaugeField &in, int shift) :
      TunableKernel3D(in, 2, 4), out(out), in(in), shift(shift)
    {
      assert(shift == 1 || shift == 3);
      strcat(aux, ",shift=");
      char shift_str[16];
      u32toa(shift_str, shift);
      strcat(aux, shift_str);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_NO> arg(out, in, shift);
        launch<GaugeShift>(tp, stream, arg);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
        GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_13> arg(out, in, shift);
        launch<GaugeShift>(tp, stream, arg);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
        GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_12> arg(out, in, shift);
        launch<GaugeShift>(tp, stream, arg);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
        GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_9> arg(out, in, shift);
        launch<GaugeShift>(tp, stream, arg);
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
        GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_8> arg(out, in, shift);
        launch<GaugeShift>(tp, stream, arg);
      }
    }

    long long bytes() const { return out.Bytes() + in.Bytes(); }
  };

  GaugeField shift(const GaugeField &in, int shift)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    if (in.GhostExchange() == QUDA_GHOST_EXCHANGE_EXTENDED)
      errorQuda("Extended ghost exchange not supported");
    if (in.GhostExchange() == QUDA_GHOST_EXCHANGE_NO && comm_partitioned())
      errorQuda("comm_dim_partition() == true requires we have GhostExchange = QUDA_GHOST_EXCHANGE_PAD");
    GaugeFieldParam param(in);
    param.create = QUDA_NULL_FIELD_CREATE;
    GaugeField out(param);
    const_cast<double&>(out.LinkMax()) = in.LinkMax();
    instantiate<GaugeShifter>(out, in, shift);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return out;
  }

} // namespace quda
