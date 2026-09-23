#pragma once

#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_hyp.cuh>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeHYP : TunableKernel3D
  {
    GaugeField &out;
    GaugeField *tmp[4];
    const GaugeField &in;
    const real_t alpha;
    const int level;
    const int dir_ignore;
    const int hypDim;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3/4): 2 for parity in the y thread dim, 3 or 4 corresponds to mapping direction to the z thread dim
    GaugeHYP(GaugeField &out, GaugeField *tmp[4], const GaugeField &in, real_t alpha, int level, int dir_ignore) :
      TunableKernel3D(in, 2, (dir_ignore == 4) ? 4 : 3),
      out(out),
      tmp {tmp[0], tmp[1], tmp[2], tmp[3]},
      in(in),
      alpha(alpha),
      level(level),
      dir_ignore(dir_ignore),
      hypDim((dir_ignore == 4) ? 4 : 3)
    {
      strcat(aux, ",level=");
      i32toa(aux + strlen(aux), level);
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (hypDim == 4) {
        if (level == 1) {
          launch<HYP>(tp, stream, GaugeHYPArg<Float, nColor, recon, 1, 4>(out, tmp, in, alpha, dir_ignore));
        } else if (level == 2) {
          launch<HYP>(tp, stream, GaugeHYPArg<Float, nColor, recon, 2, 4>(out, tmp, in, alpha, dir_ignore));
        } else if (level == 3) {
          launch<HYP>(tp, stream, GaugeHYPArg<Float, nColor, recon, 3, 4>(out, tmp, in, alpha, dir_ignore));
        }
      } else if (hypDim == 3) {
        if (level == 1) {
          launch<HYP3D>(tp, stream, GaugeHYPArg<Float, nColor, recon, 1, 3>(out, tmp, in, alpha, dir_ignore));
        } else if (level == 2) {
          launch<HYP3D>(tp, stream, GaugeHYPArg<Float, nColor, recon, 2, 3>(out, tmp, in, alpha, dir_ignore));
        }
      }
    }

    void preTune() { out.backup(); } // defensive measure in case they alias
    void postTune() { out.restore(); }

    long long flops() const
    {
      long long flops = 0;
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      if ((hypDim == 4 && level == 1) || (hypDim == 3 && level == 1)) {
        flops += ((hypDim - 1) * 2 + (hypDim - 1) * 4) * mat_flops * hypDim * in.LocalVolume();
      } else if (hypDim == 4 && level == 2) {
        flops += ((hypDim - 1) * 2 + (hypDim - 1) * (hypDim - 2) * 4) * mat_flops * hypDim * in.LocalVolume();
      } else if ((hypDim == 4 && level == 3) || (hypDim == 3 && level == 2)) {
        flops += (2 + (hypDim - 1) * 4) * mat_flops * hypDim * in.LocalVolume();
      }
      return flops;
    }

    long long bytes() const
    {
      const auto link_bytes = [](const GaugeField &g) {
        return static_cast<long long>(static_cast<int>(g.Reconstruct()) * static_cast<int>(g.Precision()));
      };
      const long long in_lp = link_bytes(in);
      const long long tmp0_lp = link_bytes(*tmp[0]);
      const long long out_lp = link_bytes(out);

      long long bytes = 0;
      if ((hypDim == 4 && level == 1) || (hypDim == 3 && level == 1)) { // 6 links per dim, 1 in, hypDim-1 tmp
        bytes += (in_lp + (hypDim - 1) * 6 * in_lp + (hypDim - 1) * tmp0_lp) * hypDim * in.LocalVolume();
      } else if (hypDim == 4 && level == 2) { // 6 links per dim, 1 in, hypDim-1 tmp
        bytes += (in_lp + (hypDim - 1) * (hypDim - 2) * 6 * tmp0_lp + (hypDim - 1) * tmp0_lp) * hypDim * in.LocalVolume();
      } else if ((hypDim == 4 && level == 3) || (hypDim == 3 && level == 2)) { // 6 links per dim, 1 in, 1 out
        bytes += (in_lp + (hypDim - 1) * 6 * tmp0_lp + out_lp) * hypDim * in.LocalVolume();
      }
      return bytes;
    }

  }; // GaugeHYP

  template <typename Float, QudaReconstructType recon>
  void applyGaugeHYP(GaugeField &out, GaugeField *tmp[4], const GaugeField &in, real_t alpha, int level, int dir_ignore)
  {
    GaugeHYP<Float, 3, recon>(out, tmp, in, alpha, level, dir_ignore);
  }

} // namespace quda
