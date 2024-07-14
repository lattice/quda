#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_ape.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeAPE : TunableKernel3D
  {
    GaugeField &out;
    const GaugeField &in;
    const Float alpha;
    const int dir_ignore;
    const int apeDim;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }
    unsigned int sharedBytesPerThread() const { return 4 * sizeof(int); } // for thread_array

  public:
    // (2,3/4): 2 for parity in the y thread dim, 3 or 4 corresponds to mapping direction to the z thread dim
    GaugeAPE(GaugeField &out, const GaugeField &in, double alpha, int dir_ignore) :
      TunableKernel3D(in, 2, (dir_ignore == 4) ? 4 : 3),
      out(out),
      in(in),
      alpha(static_cast<Float>(alpha)),
      dir_ignore(dir_ignore),
      apeDim((dir_ignore == 4) ? 4 : 3)
    {
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (apeDim == 3) {
        launch<APE>(tp, stream, GaugeAPEArg<Float, nColor, recon, 3>(out, in, alpha, dir_ignore));
      } else if (apeDim == 4) {
        launch<APE>(tp, stream, GaugeAPEArg<Float, nColor, recon, 4>(out, in, alpha, dir_ignore));
      }
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return (2 + (apeDim - 1) * 4) * mat_flops * apeDim * in.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (apeDim - 1) * 6) * in.Reconstruct() * in.Precision() +
              out.Reconstruct() * out.Precision()) * apeDim * in.LocalVolume();
    }

  }; // GaugeAPE

  void APEStep(GaugeField &out, GaugeField &in, double alpha, int dir_ignore)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeAPE>(out, in, alpha, dir_ignore);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
