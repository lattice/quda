#include <quda_internal.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include <unitarization_links.h>
#include <comm_quda.h>
#include <instantiate.h>
#include <tunable_reduction.h>
#include <kernels/gauge_fix_quality.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeFixingQuality : TunableReduction2D
  {
    const GaugeField &u;
    double *quality;
    const int dir_ignore;
    const int fixDim;
    const bool compute_theta;
    unsigned int minThreads() const { return u.LocalVolumeCB(); }

  public:
    GaugeFixingQuality(const GaugeField &u, double quality[2], int dir_ignore, bool compute_theta) :
      TunableReduction2D(u, 2),
      u(u),
      quality(quality),
      dir_ignore(dir_ignore),
      fixDim((dir_ignore == 4) ? 4 : 3),
      compute_theta(compute_theta)
    {
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      array<double, 2> value {};
      if (compute_theta) {
        GaugeFixQualityArg<Float, nColor, recon, true> arg(u, dir_ignore);
        launch<GaugeFixQuality>(value, tp, stream, arg);
      } else {
        GaugeFixQualityArg<Float, nColor, recon, false> arg(u, dir_ignore);
        launch<GaugeFixQuality>(value, tp, stream, arg);
      }
      quality[0] = value[0] / static_cast<double>(fixDim * u.Ncolor() * u.Volume());
      quality[1] = value[1] / static_cast<double>(u.Ncolor() * u.Volume());
    }

    long long flops() const { return u.Ncolor() * u.LocalVolume(); }

    long long bytes() const { return fixDim * u.Reconstruct() * u.Precision() * u.LocalVolume(); }

  }; // GaugeFixingQuality

  void gaugeFixingQuality(double quality[2], const GaugeField &u, int dir_ignore, bool compute_theta)
  {
    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeFixingQuality>(u, quality, dir_ignore, compute_theta);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
