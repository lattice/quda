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
    array<double, 2> &quality;
    const int dir_ignore;
    const int fixDim;
    const bool compute_theta;
    unsigned int minThreads() const { return u.LocalVolumeCB(); }
    unsigned int sharedBytesPerThread() const { return 4 * sizeof(int); } // for thread_array

  public:
    GaugeFixingQuality(const GaugeField &u, array<double, 2> &quality, int dir_ignore, bool compute_theta) :
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
      if (compute_theta) {
        GaugeFixQualityArg<Float, nColor, recon, true> arg(u, dir_ignore);
        launch<GaugeFixQuality>(quality, tp, stream, arg);
      } else {
        GaugeFixQualityArg<Float, nColor, recon, false> arg(u, dir_ignore);
        launch<GaugeFixQuality>(quality, tp, stream, arg);
      }
      quality[0] /= static_cast<double>(fixDim * u.Ncolor() * u.Volume());
      quality[1] /= static_cast<double>(u.Ncolor() * u.Volume());
    }

    long long flops() const { return u.Ncolor() * u.LocalVolume(); }

    long long bytes() const { return fixDim * u.Reconstruct() * u.Precision() * u.LocalVolume(); }

  }; // GaugeFixingQuality

  double2 gaugeFixingQuality(const GaugeField &u, int dir_ignore, bool compute_theta)
  {
    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    array<double, 2> quality_ {};
    instantiate<GaugeFixingQuality>(u, quality_, dir_ignore, compute_theta);
    double2 quality = make_double2(quality_[0], quality_[1]);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    return quality;
  }

} // namespace quda
