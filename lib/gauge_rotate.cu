#include <quda_internal.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/gauge_rotate.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeRotate : TunableKernel3D
  {
    GaugeField &out;
    const GaugeField &in;
    const GaugeField &rot;

    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    GaugeRotate(GaugeField &out, const GaugeField &in, const GaugeField &rot) :
      TunableKernel3D(in, 2, 4), out(out), in(in), rot(rot)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<RotateGauge>(tp, stream, RotateGaugeArg<Float, nColor, recon>(out, in, rot));
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return 2 * mat_flops * 4 * in.LocalVolume();
    }
    long long bytes() const // 2 rot, 1 in, 1 out, per dim.
    {
      return (2 * rot.Reconstruct() * rot.Precision() + in.Reconstruct() * in.Precision()
              + out.Reconstruct() * out.Precision())
        * 4 * in.LocalVolume();
    }

  }; // RotateGauge

  void gaugeRotate(GaugeField &out, const GaugeField &in, const GaugeField &rot)
  {
    checkPrecision(out, in, rot);
    checkReconstruct(out, in, rot);
    checkNative(out, in, rot);

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeRotate>(out, in, rot);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
