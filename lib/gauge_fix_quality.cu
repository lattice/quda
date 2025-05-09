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
    const GaugeField &rot;
    double *quality;
    const int dir_ignore;
    const int fixDim;
    const bool compute_theta;
    unsigned int minThreads() const { return u.LocalVolumeCB(); }

  public:
    GaugeFixingQuality(const GaugeField &u, const GaugeField &rot, double quality[2], int dir_ignore, bool compute_theta) :
      TunableReduction2D(u, 2),
      u(u),
      rot(rot),
      quality(quality),
      dir_ignore(dir_ignore),
      fixDim((dir_ignore == 4) ? 4 : 3),
      compute_theta(compute_theta)
    {
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, ",compute_theta=");
      i32toa(aux + strlen(aux), compute_theta);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      array<double, 2> value {};
      if (compute_theta) {
        GaugeFixQualityArg<Float, nColor, recon, true> arg(u, rot, dir_ignore);
        launch<GaugeFixQuality>(value, tp, stream, arg);
      } else {
        GaugeFixQualityArg<Float, nColor, recon, false> arg(u, rot, dir_ignore);
        launch<GaugeFixQuality>(value, tp, stream, arg);
      }
      quality[0] = value[0] / static_cast<double>(fixDim * u.Ncolor() * u.LocalVolume() * comm_size());
      quality[1] = value[1] / static_cast<double>(u.Ncolor() * u.LocalVolume() * comm_size());
    }

    long long flops() const { return 0; }
    long long bytes() const
    {
      return ((compute_theta ? 2 : 1) * fixDim * u.Reconstruct() * u.Precision()
              + (1 + (compute_theta ? 2 : 1) * fixDim) * rot.Reconstruct() * rot.Precision())
        * u.LocalVolume();
    }

  }; // GaugeFixingQuality

  void gaugeFixQuality(double quality[2], const GaugeField &rot, const GaugeField &u, int dir_ignore, bool compute_theta)
  {
    checkPrecision(rot, u);
    checkReconstruct(rot, u);
    checkNative(rot, u);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeFixingQuality>(u, rot, quality, dir_ignore, compute_theta);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
