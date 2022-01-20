#include <tune_quda.h>
#include <clover_field.h>
#include <gauge_field.h>
#include <instantiate.h>
#include <tunable_nd.h>
#include <kernels/clover_compute.cuh>

namespace quda {

  template <typename store_t>
  class ComputeClover : TunableKernel2D {
    CloverArg<store_t> arg;
    const GaugeField &meta;
    bool tuneSharedBytes() const { return false; } // Don't tune the shared memory.
    unsigned int minThreads() const { return arg.threads.x; }

  public:
    ComputeClover(CloverField &clover, const GaugeField& f, double coeff) :
      TunableKernel2D(clover, 2),
      arg(clover, f, coeff),
      meta(f)
    {
      checkNative(clover, f);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CloverCompute>(tp, stream, arg);
    }

    long long flops() const { return 2*arg.threads.x*480ll; }
    long long bytes() const { return 2*arg.threads.x*(6*arg.f.Bytes() + arg.clover.Bytes()); }
  };

#ifdef GPU_CLOVER_DIRAC
  void computeClover(CloverField &clover, const GaugeField& f, double coeff)
  {
    if (clover.Precision() < QUDA_SINGLE_PRECISION) errorQuda("Cannot use fixed-point precision here");
    clover.Diagonal(0.5); // 0.5 comes from scaling used on native fields
    instantiate<ComputeClover>(clover, f, coeff);
  }
#else
  void computeClover(CloverField &, const GaugeField &, double)
  {
    errorQuda("Clover has not been built");
  }
#endif

} // namespace quda

