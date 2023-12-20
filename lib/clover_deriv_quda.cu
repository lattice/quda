#include <tunable_nd.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <kernels/clover_deriv.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class DerivativeClover : TunableKernel3D
  {
    GaugeField &force;
    const GaugeField &gauge;
    const GaugeField &oprod;
    double coeff;
    unsigned int minThreads() const override { return gauge.LocalVolumeCB(); }

  public:
    DerivativeClover(const GaugeField &gauge, GaugeField &force, const GaugeField &oprod, double coeff) :
      TunableKernel3D(gauge, 2, 4), force(force), gauge(gauge), oprod(oprod), coeff(coeff)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CloverDerivative>(tp, stream, CloverDerivArg<Float, nColor, recon>(force, gauge, oprod, coeff));
    }

    // The force field is updated so we must preserve its initial state
    void preTune() override { force.backup(); }
    void postTune() override { force.restore(); }

    long long flops() const override
    {
      auto gemm_flops = 8 * nColor * nColor * nColor - 2 * nColor * nColor;
      return 32 * gemm_flops * 12 * gauge.LocalVolume();
    }
    long long bytes() const override
    {
      return (16 * gauge.Reconstruct() + 8 * oprod.Reconstruct() + 2 * force.Reconstruct()) * 12 * gauge.Precision()
        * gauge.LocalVolume();
    }
  };

  void cloverDerivative(GaugeField &force, const GaugeField &gauge, const GaugeField &oprod, double coeff)
  {
    if constexpr (is_enabled_clover()) {
      checkPrecision(force, gauge, oprod);
      assert(oprod.Geometry() == QUDA_TENSOR_GEOMETRY);
      assert(force.Geometry() == QUDA_VECTOR_GEOMETRY);
      if (oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");
      if (force.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");

      GaugeField *oprodEx = createExtendedGauge(oprod, gauge.R(), getProfile());

      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      instantiate<DerivativeClover, ReconstructNo12>(gauge, force, *oprodEx, coeff);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

      delete oprodEx;
    } else {
      errorQuda("Clover has not been built");
    }
  }

} // namespace quda
