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
    GaugeField* tmp[4];
    const GaugeField &in;
    const Float alpha;
    const int level;
    const int dir_ignore;
    const int hypDim;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }
    unsigned int sharedBytesPerThread() const { return 4 * sizeof(int); } // for thread_array

  public:
    // (2,3): 2 for parity in the y thread dim, 3 or 4 corresponds to mapping direction to the z thread dim
    GaugeHYP(GaugeField &out, GaugeField* tmp[4], const GaugeField &in, double alpha, int level, int dir_ignore) :
      TunableKernel3D(in, 2, (dir_ignore == 4) ? 4 : 3),
      out(out),
      tmp {tmp[0], tmp[1], tmp[2], tmp[3]},
      in(in),
      alpha(static_cast<Float>(alpha)),
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
      long long bytes = 0;
      if ((hypDim == 4 && level == 1) || (hypDim == 3 && level == 1)) { // 6 links per dim, 1 in, hypDim-1 tmp
        bytes += (in.Reconstruct() * in.Precision() + (hypDim - 1) * 6 * in.Reconstruct() * in.Precision()
                  + (hypDim - 1) * tmp[0]->Reconstruct() * tmp[0]->Precision())
          * hypDim * in.LocalVolume();
      } else if (hypDim == 4 && level == 2) { // 6 links per dim, 1 in, hypDim-1 tmp
        bytes += (in.Reconstruct() * in.Precision()
                  + (hypDim - 1) * (hypDim - 2) * 6 * tmp[0]->Reconstruct() * tmp[0]->Precision()
                  + (hypDim - 1) * tmp[0]->Reconstruct() * tmp[0]->Precision())
          * hypDim * in.LocalVolume();
      } else if ((hypDim == 4 && level == 3) || (hypDim == 3 && level == 2)) { // 6 links per dim, 1 in, 1 out
        bytes += (in.Reconstruct() * in.Precision() + (hypDim - 1) * 6 * tmp[0]->Reconstruct() * tmp[0]->Precision()
                  + out.Reconstruct() * out.Precision())
          * hypDim * in.LocalVolume();
      }
      return bytes;
    }

  }; // GaugeAPE

  void HYPStep(GaugeField &out, GaugeField &in, double alpha1, double alpha2, double alpha3,
               int dir_ignore)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    GaugeFieldParam gParam(out);
    gParam.location = QUDA_CUDA_FIELD_LOCATION;
    const int smearDim = (dir_ignore >= 0 && dir_ignore <= 3) ? 3 : 4;
    //GaugeField tmp[4];
    gParam.geometry = QUDA_TENSOR_GEOMETRY;
    GaugeFieldParam gParam2(gParam);
    gParam2.create = QUDA_REFERENCE_FIELD_CREATE;

    GaugeField* tmp[4];
    if (smearDim == 3) {
      tmp[0] = new GaugeField(gParam);
      // aux[1], aux[2] and aux[3] will not be used for smearDim == 3
      gParam.create = QUDA_REFERENCE_FIELD_CREATE;
      for (int i = 1; i < 4; ++i) { tmp[i] = new GaugeField(gParam); }
    } else {
      for (int i = 0; i < 4; ++i) { tmp[i] = new GaugeField(gParam); }
    }

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    if (dir_ignore == 4) {
      copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
      in.exchangeExtendedGhost(in.R(), false);
      instantiate<GaugeHYP>(out, tmp, in, alpha3, 1, dir_ignore);
      tmp[0]->exchangeExtendedGhost(tmp[0]->R(), false);
      tmp[1]->exchangeExtendedGhost(tmp[1]->R(), false);
      instantiate<GaugeHYP>(out, tmp, in, alpha2, 2, dir_ignore);
      tmp[2]->exchangeExtendedGhost(tmp[2]->R(), false);
      tmp[3]->exchangeExtendedGhost(tmp[3]->R(), false);
      instantiate<GaugeHYP>(out, tmp, in, alpha1, 3, dir_ignore);
      out.exchangeExtendedGhost(out.R(), false);
    } else {
      copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
      in.exchangeExtendedGhost(in.R(), false);
      instantiate<GaugeHYP>(out, tmp, in, alpha3, 1, dir_ignore);
      tmp[0]->exchangeExtendedGhost(tmp[0]->R(), false);
      instantiate<GaugeHYP>(out, tmp, in, alpha2, 2, dir_ignore);
      out.exchangeExtendedGhost(out.R(), false);
    }
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    for (int i = 0; i < 4; i++) delete tmp[i];

  }

} // namespace quda