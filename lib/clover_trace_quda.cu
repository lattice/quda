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
      if (clover.TwistFlavor() == QUDA_TWIST_SINGLET ||
          clover.TwistFlavor() == QUDA_TWIST_NONDEG_DOUBLET) {
        CloverTraceArg<Float, nColor, true> arg(output, clover, coeff, parity);
        launch<CloverSigmaTr>(tp, stream, arg);
      } else {
        CloverTraceArg<Float, nColor, false> arg(output, clover, coeff, parity);
        launch<CloverSigmaTr>(tp, stream, arg);
      }
    }

    void preTune() { output.backup(); }
    void postTune() { output.restore(); }

    long long flops() const { return 0; } // Fix this
    long long bytes() const { return clover.Bytes() + output.Bytes(); }
  };

#ifdef GPU_CLOVER_DIRAC
  void computeCloverSigmaTrace(GaugeField& output, const CloverField& clover, double coeff, int parity)
  {
    checkNative(output, clover);
    instantiate<CloverSigmaTrace>(output, clover, coeff, parity);
  }
#else
  void computeCloverSigmaTrace(GaugeField&, const CloverField&, double, int)
  {
    errorQuda("Clover has not been built");
  }
#endif

} // namespace quda
