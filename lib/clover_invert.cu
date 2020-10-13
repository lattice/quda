#include <tune_quda.h>
#include <clover_field.h>
#include <launch_kernel.cuh>
#include <instantiate.h>

#include <jitify_helper.cuh>
#include <kernels/clover_invert.cuh>

namespace quda {

  using namespace clover;

  template <typename store_t>
  class CloverInvert : TunableLocalParityReduction {
    CloverInvertArg<store_t> arg;
    const CloverField &meta; // used for meta data only

  public:
    CloverInvert(CloverField &clover, bool compute_tr_log) :
      arg(clover, compute_tr_log),
      meta(clover)
    {
      writeAuxString("stride=%d,prec=%lu,trlog=%s,twist=%s", arg.clover.stride, sizeof(store_t),
		     compute_tr_log ? "true" : "false", arg.twist ? "true" : "false");
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/clover_invert.cuh");
#endif
      }

      apply(0);
      if (compute_tr_log) {
        arg.complete(*clover.TrLog());
        comm_allreduce_array(clover.TrLog(), 2);
      }
    }

    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      using Arg = decltype(arg);
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::cloverInvertKernel")
                           .instantiate((int)tp.block.x, Type<Arg>(), arg.compute_tr_log, arg.twist)
                           .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                           .launch(arg);
#else
        if (arg.compute_tr_log) {
          if (arg.twist) {
	    errorQuda("Not instantiated");
	  } else {
	    LAUNCH_KERNEL_LOCAL_PARITY(cloverInvertKernel, (*this), tp, stream, arg, Arg, true, false);
	  }
        } else {
          if (arg.twist) {
            qudaLaunchKernel(cloverInvertKernel<1,Arg,false,true>, tp, stream, arg);
          } else {
            qudaLaunchKernel(cloverInvertKernel<1,Arg,false,false>, tp, stream, arg);
          }
        }
#endif
      } else {
        errorQuda("Not implemented");
      }
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
    long long flops() const { return 0; } 
    long long bytes() const { return 2*arg.clover.volumeCB*(arg.inverse.Bytes() + arg.clover.Bytes()); } 
    void preTune() { if (arg.clover.clover == arg.inverse.clover) arg.inverse.save(); }
    void postTune() { if (arg.clover.clover == arg.inverse.clover) arg.inverse.load(); }
  };

  // this is the function that is actually called, from here on down we instantiate all required templates
  void cloverInvert(CloverField &clover, bool computeTraceLog)
  {
#ifdef GPU_CLOVER_DIRAC
    instantiate<CloverInvert>(clover, computeTraceLog);
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
