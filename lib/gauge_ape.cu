#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_ape.cuh>

#include <cerrno>
#include <cstdlib>

namespace quda {

  namespace {

    int ape_prefetch_distance()
    {
      static const int distance = []() {
        const char *env = getenv("QUDA_APE_PREFETCH_DISTANCE");
        if (!env) return 0;

        char *end = nullptr;
        errno = 0;
        const long value = strtol(env, &end, 10);
        if (errno == ERANGE || end == env || *end != '\0' || value < 0 || value > 4) {
          errorQuda("QUDA_APE_PREFETCH_DISTANCE=%s is invalid; expected an integer in [0,4]", env);
        }
        return static_cast<int>(value);
      }();

      return distance;
    }

  } // namespace

  template <typename Float, int nColor, QudaReconstructType recon, int prefetch_distance_>
  class GaugeAPE : TunableKernel3D
  {
    static_assert(prefetch_distance_ >= 0 && prefetch_distance_ <= 4, "Invalid APE prefetch distance");
    static constexpr int prefetch_distance = prefetch_distance_;

    GaugeField &out;
    const GaugeField &in;
    const real_t alpha;
    const int dir_ignore;
    const real_t anisotropy;
    const int apeDim;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3/4): 2 for parity in the y thread dim, 3 or 4 corresponds to mapping direction to the z thread dim
    GaugeAPE(GaugeField &out, const GaugeField &in, real_t alpha, int dir_ignore, real_t anisotropy) :
      TunableKernel3D(in, 2, (dir_ignore == 4) ? 4 : 3),
      out(out),
      in(in),
      alpha(alpha),
      dir_ignore(dir_ignore),
      anisotropy(anisotropy),
      apeDim((dir_ignore == 4) ? 4 : 3)
    {
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, ",prefetch_distance=");
      i32toa(aux + strlen(aux), prefetch_distance);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (apeDim == 3) {
        launch<APE>(tp, stream,
                    GaugeAPEArg<Float, nColor, recon, 3, prefetch_distance>(out, in, alpha, dir_ignore, anisotropy));
      } else if (apeDim == 4) {
        launch<APE>(tp, stream,
                    GaugeAPEArg<Float, nColor, recon, 4, prefetch_distance>(out, in, alpha, dir_ignore, anisotropy));
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
      const long long in_bytes
        = static_cast<long long>(static_cast<int>(in.Reconstruct()) * static_cast<int>(in.Precision()));
      const long long out_bytes
        = static_cast<long long>(static_cast<int>(out.Reconstruct()) * static_cast<int>(out.Precision()));
      return ((1 + (apeDim - 1) * 6) * in_bytes + out_bytes) * apeDim * in.LocalVolume();
    }

  }; // GaugeAPE

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeAPEDispatch
  {
  public:
    GaugeAPEDispatch(GaugeField &out, const GaugeField &in, real_t alpha, int dir_ignore, real_t anisotropy)
    {
      switch (ape_prefetch_distance()) {
      case 0: GaugeAPE<Float, nColor, recon, 0>(out, in, alpha, dir_ignore, anisotropy); break;
      case 1: GaugeAPE<Float, nColor, recon, 1>(out, in, alpha, dir_ignore, anisotropy); break;
      case 2: GaugeAPE<Float, nColor, recon, 2>(out, in, alpha, dir_ignore, anisotropy); break;
      case 3: GaugeAPE<Float, nColor, recon, 3>(out, in, alpha, dir_ignore, anisotropy); break;
      case 4: GaugeAPE<Float, nColor, recon, 4>(out, in, alpha, dir_ignore, anisotropy); break;
      default: errorQuda("Unexpected APE prefetch distance %d", ape_prefetch_distance());
      }
    }
  };

  void APEStep(GaugeField &out, GaugeField &in, real_t alpha, int dir_ignore, real_t smear_anisotropy)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeAPEDispatch>(out, in, alpha, dir_ignore, smear_anisotropy);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
