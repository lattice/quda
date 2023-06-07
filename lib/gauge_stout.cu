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
    const int stoutDim;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

    unsigned int maxSharedBytesPerBlock() const { return maxDynamicSharedBytesPerBlock(); }
    unsigned int sharedBytesPerThread() const
    {
      // use SharedMemoryCache if using over improvement for two link fields
      return improved ? 2 * in.Ncolor() * in.Ncolor() * 2 * sizeof(typename mapper<Float>::type) : 0;
    }

  public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeSTOUT(GaugeField &out, const GaugeField &in, bool improved, double rho, double epsilon = 0.0) :
      TunableKernel3D(in, 2, improved ? 4 : 3),
      out(out),
      in(in),
      improved(improved),
      rho(static_cast<Float>(rho)),
      epsilon(static_cast<Float>(epsilon)),
      stoutDim(improved ? 4 : 3)
    {
      if (improved) strcat(aux, ",improved");
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (!improved) {
        launch<STOUT>(tp, stream, STOUTArg<Float, nColor, recon, 3>(out, in, rho));
      } else if (improved) {
        tp.set_max_shared_bytes = true;
        launch<OvrImpSTOUT>(tp, stream, STOUTArg<Float, nColor, recon, 4>(out, in, rho, epsilon));
      }
    }

    void preTune() { if (out.Gauge_p() == in.Gauge_p()) out.backup(); }
    void postTune() { if (out.Gauge_p() == in.Gauge_p()) out.restore(); }

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

  void STOUTStep(GaugeField &out, GaugeField &in, double rho)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    instantiate<GaugeSTOUT>(out, in, false, rho);
    out.exchangeExtendedGhost(out.R(), false);
  }

  void OvrImpSTOUTStep(GaugeField &out, GaugeField& in, double rho, double epsilon)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    instantiate<GaugeSTOUT>(out, in, true, rho, epsilon);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
