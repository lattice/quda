#include <clover_field.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/clover_trace.cuh>

namespace quda {

  template <typename Float, int nColor>
  class CloverSigmaTrace : TunableKernel1D {
    GaugeField &output;
    const CloverField &clover;
    const bool twisted;
    Float coeff;
    const int parity;
    unsigned int minThreads() const override { return clover.VolumeCB(); }

  public:
    CloverSigmaTrace(GaugeField &output, const CloverField &clover, double coeff, int parity) :
      TunableKernel1D(output),
      output(output),
      clover(clover),
      twisted(clover.TwistFlavor() == QUDA_TWIST_SINGLET || clover.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET),
      coeff(static_cast<Float>(coeff)),
      parity(parity)
    {
      if (twisted) strcat(aux, ",twisted");
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream) override
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (twisted) {
        launch<CloverSigmaTr>(tp, stream, CloverTraceArg<Float, nColor, true>(output, clover, coeff, parity));
      } else {
        launch<CloverSigmaTr>(tp, stream, CloverTraceArg<Float, nColor, false>(output, clover, coeff, parity));
      }
    }

    void preTune() override { output.backup(); }
    void postTune() override { output.restore(); }

    long long flops() const override { return 0; } // Fix this
    long long bytes() const override { return clover.Bytes() + output.Bytes(); }
  };

  void computeCloverSigmaTrace(GaugeField& output, const CloverField& clover, double coeff, int parity)
  {
    if constexpr (clover::is_enabled()) {
      checkNative(output, clover);
      instantiate<CloverSigmaTrace>(output, clover, coeff, parity);
    } else {
      errorQuda("Clover has not been built");
    }
  }

} // namespace quda
