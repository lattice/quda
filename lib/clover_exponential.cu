#include <tune_quda.h>
#include <clover_field.h>
#include <launch_kernel.cuh>
#include <instantiate.h>

#include <jitify_helper.cuh>
#include <kernels/clover_exponential.cuh>

namespace quda {

  using namespace clover;

  template <typename store_t>
  class CloverExponential : TunableLocalParity {
    CloverExponentialArg<store_t> arg;
    const CloverField &meta; // used for meta data only
    bool inverse;
    bool tuneGridDim() const { return true; }

  public:
    CloverExponential(CloverField &clover, int order, double mass, bool inverse) :
      arg(clover, order, mass),
      meta(clover),
      inverse(inverse)
    {
      writeAuxString("stride=%d,prec=%lu,order=%d,mass=%f,inverse=%s", arg.clover.stride, sizeof(store_t),
		     order, mass, inverse ? "true" : "false");
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/clover_exponential.cuh");
#endif
        arg.c = static_cast<double *>(pool_device_malloc((arg.order+1) * sizeof(double)));
        apply(0);
        pool_device_free(arg.c);
        checkCudaError();
      }
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      using Arg = decltype(arg);
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::cloverExponentialKernel")
                           .instantiate((int)tp.block.x, Type<Arg>(), inverse)
                           .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                           .launch(arg);
#else
        if (inverse) {
          cloverExponentialKernel<Arg,true> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
        } else {
          cloverExponentialKernel<Arg,false> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
        }
#endif
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 0; } 
    long long bytes() const { return 2*arg.clover.volumeCB*(arg.clover.Bytes()); } 
    void preTune() { arg.clover.save(); }
    void postTune() { arg.clover.load(); }
  };

  // this is the function that is actually called, from here on down we instantiate all required templates
  void cloverExponential(CloverField &clover, int order, double mass, bool inverse)
  {
#ifdef GPU_CLOVER_DIRAC
    instantiate<CloverExponential>(clover, order, mass, inverse);
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
