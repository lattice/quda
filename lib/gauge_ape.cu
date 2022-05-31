#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/gauge_ape.cuh>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeAPE : TunableKernel3D
  {
    static constexpr int apeDim = 3; // apply APE in space only
    GaugeField &out;
    const GaugeField &in;
    const Float alpha;
    unsigned int minThreads() const { return in.LocalVolumeCB(); }

  public:
    // (2,3): 2 for parity in the y thread dim, 3 corresponds to mapping direction to the z thread dim
    GaugeAPE(GaugeField &out, const GaugeField &in, double alpha) :
      TunableKernel3D(in, 2, apeDim),
      out(out),
      in(in),
      alpha(static_cast<Float>(alpha))
    {
      strcat(aux, comm_dim_partitioned_string());
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<APE>(tp, stream, GaugeAPEArg<Float,nColor,recon, apeDim>(out, in, alpha));
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

  void APEStep(GaugeField &out, GaugeField& in, double alpha)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    instantiate<GaugeAPE>(out, in, alpha);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
