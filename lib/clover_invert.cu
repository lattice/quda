#include <tune_quda.h>
#include <clover_field_order.h>
#include <complex_quda.h>
#include <launch_kernel.cuh>
#include <atomic.cuh>
#include <cub_helper.cuh>
#include <quda_matrix.h>
#include <linalg.cuh>

namespace quda {

  using namespace clover;

#ifdef GPU_CLOVER_DIRAC

  template <typename Float, typename Clover>
  struct CloverInvertArg : public ReduceArg<double2> {
    const Clover clover;
    Clover inverse;
    bool computeTraceLog;
    bool twist;
    Float mu2;
    CloverInvertArg(Clover &inverse, const Clover &clover, bool computeTraceLog=0) :
      ReduceArg<double2>(), inverse(inverse), clover(clover), computeTraceLog(computeTraceLog),
      twist(clover.Twisted()), mu2(clover.Mu2()) { }
  };

  /**
     Use a Cholesky decomposition and invert the clover matrix
   */
  template <typename Float, typename Arg, bool computeTrLog, bool twist>
  __device__ __host__ inline double cloverInvertCompute(Arg &arg, int x_cb, int parity) {

    constexpr int nColor = 3;
    constexpr int nSpin = 4;
    constexpr int N = nColor*nSpin/2;
    typedef HMatrix<Float,N> Mat;
    double trlogA = 0.0;

    for (int ch=0; ch<2; ch++) {
      Mat A = arg.clover(x_cb, parity, ch);
      A *= static_cast<Float>(2.0); // factor of two is inherent to QUDA clover storage

      if (twist) { // Compute (T^2 + mu2) first, then invert
	A = A.square();
	A += arg.mu2;
      }

      // compute the Colesky decomposition
      linalg::Cholesky<HMatrix,Float,N> cholesky(A);

      // Accumulate trlogA
      if (computeTrLog) for (int j=0; j<N; j++) trlogA += 2.0*log(cholesky.D(j));

      Mat Ainv = static_cast<Float>(0.5)*cholesky.invert(); // return full inverse
      arg.inverse(x_cb, parity, ch) = Ainv;
    }

    return trlogA;
  }

  template <typename Float, typename Arg, bool computeTrLog, bool twist>
  void cloverInvert(Arg &arg) {
    for (int parity=0; parity<2; parity++) {
      for (int x=0; x<arg.clover.volumeCB; x++) {
	// should make this thread safe if we ever apply threads to cpu code
	double trlogA = cloverInvertCompute<Float,Arg,computeTrLog,twist>(arg, x, parity);
	if (computeTrLog) {
	  if (parity) arg.result_h[0].y += trlogA;
	  else arg.result_h[0].x += trlogA;
	}
      }
    }
  }

  template <int blockSize, typename Float, typename Arg, bool computeTrLog, bool twist>
  __global__ void cloverInvertKernel(Arg arg) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = threadIdx.y;
    double trlogA_parity = 0.0;
    if (idx < arg.clover.volumeCB)
      trlogA_parity = cloverInvertCompute<Float,Arg,computeTrLog,twist>(arg, idx, parity);
    double2 trlogA = parity ? make_double2(0.0,trlogA_parity) : make_double2(trlogA_parity, 0.0);

    if (computeTrLog) reduce2d<blockSize,2>(arg, trlogA);
  }

  template <typename Float, typename Arg>
  class CloverInvert : TunableLocalParity {
    Arg arg;
    const CloverField &meta; // used for meta data only
    const QudaFieldLocation location;

  private:
    bool tuneSharedBytes() const { return false; } // Don't tune the shared memory
    unsigned int minThreads() const { return arg.clover.volumeCB; }

  public:
    CloverInvert(Arg &arg, const CloverField &meta, QudaFieldLocation location)
      : arg(arg), meta(meta), location(location) { 
      writeAuxString("stride=%d,prec=%lu,trlog=%s,twist=%s", arg.clover.stride, sizeof(Float),
		     arg.computeTraceLog ? "true" : "false", arg.twist ? "true" : "false");
    }
    virtual ~CloverInvert() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      arg.result_h[0] = make_double2(0.,0.);
      if (location == QUDA_CUDA_FIELD_LOCATION) {
	if (arg.computeTraceLog) {
	  if (arg.twist) {
	    errorQuda("Not instantiated");
	  } else {
	    LAUNCH_KERNEL_LOCAL_PARITY(cloverInvertKernel, tp, stream, arg, Float, Arg, true, false);
	  }
	} else {
	  if (arg.twist) {
	    cloverInvertKernel<1,Float,Arg,false,true> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  } else {
	    cloverInvertKernel<1,Float,Arg,false,false> <<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
	  }
	}
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

  template <typename Float, typename Clover>
  void cloverInvert(Clover inverse, const Clover clover, bool computeTraceLog, 
		    double* const trlog, const CloverField &meta, QudaFieldLocation location) {
    CloverInvertArg<Float,Clover> arg(inverse, clover, computeTraceLog);
    CloverInvert<Float,CloverInvertArg<Float,Clover>> invert(arg, meta, location);
    invert.apply(0);

    if (arg.computeTraceLog) {
      cudaDeviceSynchronize();
      comm_allreduce_array((double*)arg.result_h, 2);
      trlog[0] = arg.result_h[0].x;
      trlog[1] = arg.result_h[0].y;
    }
  }

  template <typename Float>
  void cloverInvert(const CloverField &clover, bool computeTraceLog, QudaFieldLocation location) {

    if (clover.isNative()) {
      typedef typename clover_mapper<Float>::type C;
      cloverInvert<Float>(C(clover, 1), C(clover, 0), computeTraceLog,
			  clover.TrLog(), clover, location);
    } else {
      errorQuda("Clover field %d order not supported", clover.Order());
    }

  }

#endif

  // this is the function that is actually called, from here on down we instantiate all required templates
  void cloverInvert(CloverField &clover, bool computeTraceLog, QudaFieldLocation location) {

#ifdef GPU_CLOVER_DIRAC
    if (clover.Precision() == QUDA_HALF_PRECISION && clover.Order() > 4) 
      errorQuda("Half precision not supported for order %d", clover.Order());

    if (clover.Precision() == QUDA_DOUBLE_PRECISION) {
      cloverInvert<double>(clover, computeTraceLog, location);
    } else if (clover.Precision() == QUDA_SINGLE_PRECISION) {
      cloverInvert<float>(clover, computeTraceLog, location);
    } else {
      errorQuda("Precision %d not supported", clover.Precision());
    }
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
