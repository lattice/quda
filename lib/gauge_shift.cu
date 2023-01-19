#include <tunable_nd.h>
#include <instantiate.h>
#include <gauge_field.h>
#include <kernels/gauge_shift.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon_u> class ShiftGauge : public TunableKernel3D
  {
    GaugeField &out;
    const GaugeField &in;
    const int *dx;
    unsigned int minThreads() const { return in.VolumeCB(); }

  public:
    ShiftGauge(GaugeField &out, const GaugeField &in, const int *dx) :
      TunableKernel3D(in, 2, in.Geometry()), out(out), in(in), dx(dx)
    {
      strcat(aux, ",shift=");
      for (int i = 0; i < in.Ndim(); i++) { strcat(aux, std::to_string(dx[i]).c_str()); }
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<GaugeShift>(tp, stream, GaugeShiftArg<Float, nColor, recon_u>(out, in, dx));
    }

    void preTune() { }
    void postTune() { }

    long long flops() const { return in.Volume() * 4; }
    long long bytes() const { return in.Bytes(); }
  };

  void gaugeShift(GaugeField &out, const GaugeField &in, const int *dx)
  {
    checkPrecision(in, out);
    checkLocation(in, out);
    checkReconstruct(in, out);

    if (out.Geometry() != in.Geometry()) {
      errorQuda("Field geometries %d %d do not match", out.Geometry(), in.Geometry());
    }

    // gauge field must be passed as first argument so we peel off its reconstruct type
    instantiate<ShiftGauge>(out, in, dx);
  }

} // namespace quda
