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
    Float coeff;
    const int parity;
    unsigned int minThreads() const { return clover.VolumeCB(); }

  public:
    CloverSigmaTrace(GaugeField& output, const CloverField& clover, double coeff, int parity) :
      TunableKernel1D(output),
      output(output),
      clover(clover),
      coeff(static_cast<Float>(coeff)),
      parity(parity)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (clover.TwistFlavor() == QUDA_TWIST_SINGLET || clover.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET) {
        launch<CloverSigmaTr>(tp, stream, CloverTraceArg<Float, nColor, true>(output, clover, coeff, parity));
      } else {
        launch<CloverSigmaTr>(tp, stream, CloverTraceArg<Float, nColor, false>(output, clover, coeff, parity));
      }
    }

    void preTune() { output.backup(); }
    void postTune() { output.restore(); }

    long long flops() const { return 0; } // Fix this
    long long bytes() const { return clover.Bytes() + output.Bytes(); }
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
