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
    bool verify;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    GaugeShifter(GaugeField &out, const GaugeField &in, int shift, bool verify) :
      TunableKernel3D(in, 2, 4), out(out), in(in), shift(shift), verify(verify)
    {
      assert(shift == 1 || shift == 3);
      strcat(aux, ",shift=");
      char shift_str[16];
      u32toa(shift_str, shift);
      strcat(aux, shift_str);
      strcat(aux, verify ? ",verify" : "");
      apply(device::get_default_stream());
    }

    template <bool verify> void instantiate(TuneParam &tp, const qudaStream_t &stream)
    {
      if (in.Reconstruct() == QUDA_RECONSTRUCT_NO) {
        launch<GaugeShift>(tp, stream, GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_NO, verify>(out, in, shift));
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_13) {
        launch<GaugeShift>(tp, stream, GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_13, verify>(out, in, shift));
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_12) {
        launch<GaugeShift>(tp, stream, GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_12, verify>(out, in, shift));
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_9) {
        launch<GaugeShift>(tp, stream, GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_9, verify>(out, in, shift));
      } else if (in.Reconstruct() == QUDA_RECONSTRUCT_8) {
        launch<GaugeShift>(tp, stream, GaugeShiftArg<Float, nColor, QUDA_RECONSTRUCT_8, verify>(out, in, shift));
      }
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (verify)
        instantiate<true>(tp, stream);
      else
        instantiate<false>(tp, stream);
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
    out.is_shifted = true;

    instantiate<GaugeShifter>(out, in, shift, false);
    constexpr bool verify = false;
    if constexpr (verify) instantiate<GaugeShifter>(out, in, shift, true);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return out;
  }

} // namespace quda
