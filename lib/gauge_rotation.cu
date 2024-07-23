#include <quda_internal.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/gauge_rotation.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeRotation : TunableKernel3D
  {
    GaugeField &out;
    const GaugeField &in;
    const GaugeField &rot;

    unsigned int minThreads() const { return in.LocalVolumeCB(); }
    unsigned int sharedBytesPerThread() const { return 4 * sizeof(int); } // for thread_array

  public:
    GaugeRotation(GaugeField &out, const GaugeField &in, const GaugeField &rot) :
      TunableKernel3D(in, 2, 4), out(out), in(in), rot(rot)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<GaugeRotate>(tp, stream, GaugeRotateArg<Float, nColor, recon>(out, in, rot));
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const { return 0; }

    long long bytes() const // 2 rot, 1 in, 1 out, per dim.
    {
      return (2 * rot.Reconstruct() * rot.Precision() + in.Reconstruct() * in.Precision()
              + out.Reconstruct() * out.Precision())
        * 4 * in.LocalVolume();
    }

  }; // GaugeRotate

  void gaugeRotation(GaugeField &out, GaugeField &in, GaugeField &rot)
  {
    checkPrecision(out, in, rot);
    checkReconstruct(out, in, rot);
    checkNative(out, in, rot);

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeRotation>(out, in, rot);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    out.exchangeExtendedGhost(out.R(), false);
  }

} // namespace quda
