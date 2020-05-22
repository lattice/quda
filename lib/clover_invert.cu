#include <tune_quda.h>
#include <clover_field.h>
#include <launch_kernel.cuh>

#include <jitify_helper.cuh>
#include <kernels/clover_invert.cuh>

namespace quda {

  using namespace clover;

#ifdef GPU_CLOVER_DIRAC

  template <typename Float, typename Arg>
  class CloverInvert : TunableLocalParity {
    Arg arg;
    const CloverField &meta; // used for meta data only

  private:
    bool tuneGridDim() const { return true; }

  public:
    CloverInvert(Arg &arg, const CloverField &meta) : arg(arg), meta(meta) {
      writeAuxString("stride=%d,prec=%lu,trlog=%s,twist=%s", arg.clover.stride, sizeof(Float),
		     arg.computeTraceLog ? "true" : "false", arg.twist ? "true" : "false");
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        create_jitify_program("kernels/clover_invert.cuh");
#endif
      }
    }

    virtual ~CloverInvert() { ; }
  
    void apply(const qudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      arg.result_h[0] = make_double2(0.,0.);
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::cloverInvertKernel")
                           .instantiate((int)tp.block.x, Type<Float>(), Type<Arg>(), arg.computeTraceLog, arg.twist)
                           .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                           .launch(arg);
#else
        if (arg.computeTraceLog) {
          if (arg.twist) {
	    errorQuda("Not instantiated");
	  } else {
	    LAUNCH_KERNEL_LOCAL_PARITY(cloverInvertKernel, (*this), tp, stream, arg, Float, Arg, true, false);
	  }
        } else {
          if (arg.twist) {
            cloverInvertKernel<1,Float,Arg,false,true> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          } else {
            cloverInvertKernel<1,Float,Arg,false,false> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
          }
        }
#endif
      } else {
	if (arg.computeTraceLog) {
	  if (arg.twist) {
	    cloverInvert<Float, Arg, true, true>(arg);
	  } else {
	    cloverInvert<Float, Arg, true, false>(arg);
	  }
	} else {
	  if (arg.twist) {
	    cloverInvert<Float, Arg, false, true>(arg);
	  } else {
	    cloverInvert<Float, Arg, false, false>(arg);
	  }
	}
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    long long flops() const { return 0; } 
    long long bytes() const { return 2*arg.clover.volumeCB*(arg.inverse.Bytes() + arg.clover.Bytes()); } 

    void preTune() { if (arg.clover.clover == arg.inverse.clover) arg.inverse.save(); }
    void postTune() { if (arg.clover.clover == arg.inverse.clover) arg.inverse.load(); }

  };

  template <typename Float>
  void cloverInvert(CloverField &clover, bool computeTraceLog) {
    CloverInvertArg<Float> arg(clover, computeTraceLog);
    CloverInvert<Float,CloverInvertArg<Float>> invert(arg, clover);
    invert.apply(0);

    if (arg.computeTraceLog) {
      qudaDeviceSynchronize();
      comm_allreduce_array((double*)arg.result_h, 2);
      clover.TrLog()[0] = arg.result_h[0].x;
      clover.TrLog()[1] = arg.result_h[0].y;
    }
  }

#endif

  // this is the function that is actually called, from here on down we instantiate all required templates
  void cloverInvert(CloverField &clover, bool computeTraceLog) {

#ifdef GPU_CLOVER_DIRAC
    if (clover.Precision() == QUDA_HALF_PRECISION && clover.Order() > 4) 
      errorQuda("Half precision not supported for order %d", clover.Order());

    if (clover.Precision() == QUDA_DOUBLE_PRECISION) {
      cloverInvert<double>(clover, computeTraceLog);
    } else if (clover.Precision() == QUDA_SINGLE_PRECISION) {
      cloverInvert<float>(clover, computeTraceLog);
    } else {
      errorQuda("Precision %d not supported", clover.Precision());
    }
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
