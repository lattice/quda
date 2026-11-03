#include <quda_internal.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/gauge_fix_ovr2.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeFix : TunableKernel1D
  {
    GaugeField &rot;
    const GaugeField &u;
    const Float omega;
    const int dir_ignore;
    const int fixDim;
    const int parity;
    unsigned int minThreads() const { return u.LocalVolumeCB(); }

  public:
    GaugeFix(GaugeField &rot, const GaugeField &u, double omega, int dir_ignore, int parity) :
      TunableKernel1D(u),
      rot(rot),
      u(u),
      omega(static_cast<Float>(omega)),
      dir_ignore(dir_ignore),
      fixDim((dir_ignore == 4) ? 4 : 3),
      parity(parity)
    {
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, ",parity=");
      i32toa(aux + strlen(aux), parity);
      if (omega != 1.0) { strcat(aux, ",over_relaxation"); }
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (omega == 1.0) {
        if (parity == 0) {
          FixGaugeArg<Float, nColor, recon, 0, false> arg(rot, u, omega, dir_ignore);
          launch<FixGauge>(tp, stream, arg);
        } else if (parity == 1) {
          FixGaugeArg<Float, nColor, recon, 1, false> arg(rot, u, omega, dir_ignore);
          launch<FixGauge>(tp, stream, arg);
        }
      } else {
        if (parity == 0) {
          FixGaugeArg<Float, nColor, recon, 0, true> arg(rot, u, omega, dir_ignore);
          launch<FixGauge>(tp, stream, arg);
        } else if (parity == 1) {
          FixGaugeArg<Float, nColor, recon, 1, true> arg(rot, u, omega, dir_ignore);
          launch<FixGauge>(tp, stream, arg);
        }
      }
    }

    void preTune() { rot.backup(); } // defensive measure in case they alias
    void postTune() { rot.restore(); }

    long long flops() const
    {
      auto mat_flops = u.Ncolor() * u.Ncolor() * (8ll * u.Ncolor() - 2ll);
      return (fixDim * 2 + 2 * 3) * mat_flops * u.LocalVolumeCB();
    }

    long long bytes() const // 2 links per dim, 2 rot in per dim, 1 rot in, 1 rot out.
    {
      return ((fixDim * 2) * u.Reconstruct() * u.Precision() + (1 + fixDim * 2 + 1) * rot.Reconstruct() * rot.Precision())
        * u.LocalVolumeCB();
    }

  }; // GaugeFix

  void gaugeFixOVRStep(GaugeField &rot, const GaugeField &u, double omega, int dir_ignore)
  {
    checkPrecision(rot, u);
    checkReconstruct(rot, u);
    checkNative(rot, u);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    // loop over parity
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeFix>(rot, u, omega, dir_ignore, 0);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    rot.exchangeExtendedGhost(rot.R(), getProfile(), false);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeFix>(rot, u, omega, dir_ignore, 1);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    rot.exchangeExtendedGhost(rot.R(), getProfile(), false);
  }

} // namespace quda
