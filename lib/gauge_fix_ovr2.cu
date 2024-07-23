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

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeFixingOVR : TunableKernel1D
  {
    const GaugeField &u;
    GaugeField &rot;
    const Float relax_boost;
    const int dir_ignore;
    const int fixDim;
    const int parity;
    unsigned int minThreads() const { return u.LocalVolumeCB(); }
    unsigned int sharedBytesPerThread() const { return 4 * sizeof(int); } // for thread_array

  public:
    GaugeFixingOVR(GaugeField &u, GaugeField &rot, double relax_boost, int dir_ignore, int parity) :
      TunableKernel1D(u),
      u(u),
      rot(rot),
      relax_boost(static_cast<Float>(relax_boost)),
      dir_ignore(dir_ignore),
      fixDim((dir_ignore == 4) ? 4 : 3),
      parity(parity)
    {
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (relax_boost == 1.0) {
        if (parity == 0) {
          GaugeFixArg<Float, nColor, recon, 0, false> arg(u, rot, relax_boost, dir_ignore);
          launch<GaugeFix>(tp, stream, arg);
        } else if (parity == 1) {
          GaugeFixArg<Float, nColor, recon, 1, false> arg(u, rot, relax_boost, dir_ignore);
          launch<GaugeFix>(tp, stream, arg);
        }
      } else {
        if (parity == 0) {
          GaugeFixArg<Float, nColor, recon, 0, true> arg(u, rot, relax_boost, dir_ignore);
          launch<GaugeFix>(tp, stream, arg);
        } else if (parity == 1) {
          GaugeFixArg<Float, nColor, recon, 1, true> arg(u, rot, relax_boost, dir_ignore);
          launch<GaugeFix>(tp, stream, arg);
        }
      }
    }

    void preTune() { rot.backup(); } // defensive measure in case they alias
    void postTune() { rot.restore(); }

    long long flops() const
    {
      auto mat_flops = u.Ncolor() * u.Ncolor() * (8ll * u.Ncolor() - 2ll);
      return (2 + (fixDim - 1) * 4) * mat_flops * fixDim * u.LocalVolume();
    }

    long long bytes() const // 2 links per dim, 1 rot in, 1 rot out.
    {
      return ((fixDim * 2) * u.Reconstruct() * u.Precision() + 2 * rot.Reconstruct() * rot.Precision()) * u.LocalVolume();
    }

  }; // GaugeFixingOVR

  void gaugeFixingOVR2(GaugeField &out, GaugeField &in, GaugeField &rot, double relax_boost, int dir_ignore)
  {
    checkPrecision(out, in, rot);
    checkReconstruct(out, in, rot);
    checkNative(out, in, rot);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeFixingOVR>(in, rot, relax_boost, dir_ignore, 0);
    rot.exchangeExtendedGhost(rot.R(), false);
    instantiate<GaugeFixingOVR>(in, rot, relax_boost, dir_ignore, 1);
    rot.exchangeExtendedGhost(rot.R(), false);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
