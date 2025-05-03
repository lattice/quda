#include <quda_internal.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/spinor_rotate.cuh>

namespace quda
{
  template <typename Float, int nColor, QudaReconstructType recon> class SpinorRotate : TunableKernel2D
  {
    const GaugeField &rot;
    ColorSpinorField &src;

    unsigned int minThreads() const { return src.LocalVolumeCB(); }

  public:
    SpinorRotate(const GaugeField &rot, ColorSpinorField &src) :
      TunableKernel2D(src, 2), rot(rot), src(src)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (src.Nspin() == 4) {
        if constexpr (is_enabled_spin(4)) launch<RotateSpinor>(tp, stream, RotateSpinorArg<Float, 4, nColor, recon>(src, rot));
      } else if (src.Nspin() == 1) {
        if constexpr (is_enabled_spin(1)) launch<RotateSpinor>(tp, stream, RotateSpinorArg<Float, 1, nColor, recon>(src, rot));
      } else {
        errorQuda("Nspin = %d not implemented", src.Nspin());
      }
    }

    void preTune() { src.backup(); } // defensive measure in case they alias
    void postTune() { src.restore(); }

    long long flops() const { return 0; }

    long long bytes() const // 2 rot, 1 in, 1 out, per dim.
    {
      return rot.Reconstruct() * rot.Precision() * rot.LocalVolume()
        + src.Nspin() * src.Ncolor() * 2 * src.Precision() * src.LocalVolume();
    }
  }; // SpinorRotate

  void spinorRotate(ColorSpinorField &src, const GaugeField &rot)
  {
    checkPrecision(src, rot);
    checkNative(src, rot);

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<SpinorRotate>(rot, src);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
