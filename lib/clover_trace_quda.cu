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
    unsigned int minThreads() const { return clover.VolumeCB(); }

  public:
    CloverSigmaTrace(GaugeField& output, const CloverField& clover, double coeff) :
      TunableKernel1D(output),
      output(output),
      clover(clover),
      coeff(static_cast<Float>(coeff))
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CloverSigmaTr>(tp, stream, CloverTraceArg<Float, nColor>(output, clover, coeff));
    }

    long long flops() const { return 0; } // Fix this
    long long bytes() const { return clover.Bytes() + output.Bytes(); }
  };

#ifdef GPU_CLOVER_DIRAC
  void computeCloverSigmaTrace(GaugeField& output, const CloverField& clover, double coeff)
  {
    checkNative(output, clover);
    instantiate<CloverSigmaTrace>(output, clover, coeff);
  }
#else
  void computeCloverSigmaTrace(GaugeField&, const CloverField&, double)
  {
    errorQuda("Clover has not been built");
  }
#endif

} // namespace quda
