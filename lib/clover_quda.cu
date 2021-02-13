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
    ComputeClover(CloverField &clover, const GaugeField& f, const double kappa, const double c_sw) :
      TunableKernel2D(clover, 2),
      arg(clover, f, kappa, c_sw),
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

    long long flops() const {
      // Clover construction
      long long flops = 2*arg.threads.x*480ll;
      if(!exponentiated_clover()) {
	// Clover exponentiation
	// 19th order exponentiation (19 mat vec, 19 mat scale, 19 mat addition),
	// done on 2 chiral blocks, with a scaling and an addition at the end
	long long dim = (arg.nColor * arg.nSpin / 2);
	long long expo = 19ll * 2ll * ((6ll * dim + 2ll * (dim - 1)) * dim*dim);
	long long scale_add = 4ll * dim*dim;
	flops += (expo + scale_add) * arg.threads.x;
      }
      return flops;
    }
    long long bytes() const { return 2*arg.threads.x*(6*arg.f.Bytes() + arg.clover.Bytes()); }
  };

#ifdef GPU_CLOVER_DIRAC
  void computeClover(CloverField &clover, const GaugeField& f, const double kappa, const double c_sw)
  {
    instantiate<ComputeClover>(clover, f, kappa, c_sw);
  }
#else
  void computeClover(CloverField &, const GaugeField &, const double, const double)
  {
    errorQuda("Clover has not been built");
  }
#endif

} // namespace quda

