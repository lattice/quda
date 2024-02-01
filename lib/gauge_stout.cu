#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_stout.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeSTOUT : TunableKernel3D
  {
    GaugeField &out;
    const GaugeField &in;
    const bool improved;
    const Float rho;
    const Float epsilon;
    const int dir_ignore;
    const int stoutDim;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

    unsigned int maxSharedBytesPerBlock() const { return maxDynamicSharedBytesPerBlock(); }
    unsigned int sharedBytesPerThread() const
    {
      // use ThreadLocalCache if using over improvement for two link fields
      return (improved ? 2 * in.Ncolor() * in.Ncolor() * 2 * sizeof(typename mapper<Float>::type) : 0)
        + 4 * sizeof(int); // for thread_array
    }

  public:
    // (2,3): 2 for parity in the y thread dim, 3 or 4 corresponds to mapping direction to the z thread dim
    GaugeSTOUT(GaugeField &out, const GaugeField &in, bool improved, double rho, double epsilon, int dir_ignore) :
      TunableKernel3D(in, 2, (dir_ignore == 4) ? 4 : 3),
      out(out),
      in(in),
      improved(improved),
      rho(static_cast<Float>(rho)),
      epsilon(static_cast<Float>(epsilon)),
      dir_ignore(dir_ignore),
      stoutDim((dir_ignore == 4) ? 4 : 3)
    {
      if (improved) strcat(aux, ",improved");
      strcat(aux, ",dir_ignore=");
      i32toa(aux + strlen(aux), dir_ignore);
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (!improved) {
        if (stoutDim == 3) {
          launch<STOUT>(tp, stream, STOUTArg<Float, nColor, recon, 3>(out, in, rho, 0.0, dir_ignore));
        } else if (stoutDim == 4) {
          launch<STOUT>(tp, stream, STOUTArg<Float, nColor, recon, 4>(out, in, rho, 0.0, dir_ignore));
        }
      } else if (improved) {
        tp.set_max_shared_bytes = true;
        if (stoutDim == 3) {
          launch<OvrImpSTOUT>(tp, stream, STOUTArg<Float, nColor, recon, 3>(out, in, rho, epsilon, dir_ignore));
        } else if (stoutDim == 4) {
          launch<OvrImpSTOUT>(tp, stream, STOUTArg<Float, nColor, recon, 4>(out, in, rho, epsilon, dir_ignore));
        }
      }
    }

    void preTune()
    {
      if (out.data() == in.data()) out.backup();
    }
    void postTune()
    {
      if (out.data() == in.data()) out.restore();
    }

    long long flops() const // just counts matrix multiplication
    {
      auto mat_flops = in.Ncolor() * in.Ncolor() * (8ll * in.Ncolor() - 2ll);
      return (2 + (stoutDim - 1) * (improved ? 28 : 4)) * mat_flops * stoutDim * in.LocalVolume();
    }

    long long bytes() const // 6 links per dim, 1 in, 1 out.
    {
      return ((1 + (stoutDim - 1) * (improved ? 24 : 6)) * in.Reconstruct() * in.Precision() +
              out.Reconstruct() * out.Precision()) * stoutDim * in.LocalVolume();    }
  };

  void STOUTStep(GaugeField &out, GaugeField &in, double rho, int dir_ignore)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeSTOUT>(out, in, false, rho, 0.0, dir_ignore);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    out.exchangeExtendedGhost(out.R(), false);
  }

  void OvrImpSTOUTStep(GaugeField &out, GaugeField &in, double rho, double epsilon, int dir_ignore)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiate<GaugeSTOUT>(out, in, true, rho, epsilon, dir_ignore);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
