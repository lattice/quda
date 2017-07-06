#include <tune_quda.h>
#include <clover_field_order.h>

namespace quda {

  using namespace clover;

#ifdef GPU_CLOVER_DIRAC

  template <typename real, typename Clover>
  struct CloverRhoArg  {
    Clover clover;
    real rho;
    CloverRhoArg(Clover &clover, real rho) : clover(clover), rho(rho) {}
  };

  template <typename Arg>
  __global__ void cloverRhoKernel(Arg arg) {  
    int x_cb = blockIdx.x*blockDim.x + threadIdx.x;
    int parity = threadIdx.y;

    if (x_cb < arg.clover.volumeCB) {
      for (int chi=0; chi<2; chi++) {
	decltype(arg.rho) A[36];
	arg.clover.load(A, x_cb, parity,chi);
	for (int i=0; i<6; i++) A[i] += arg.rho;
	arg.clover.save(A, x_cb, parity,chi);
      }
    }
  }

  template <typename Arg>
  class CloverRho : TunableLocalParity {
    Arg arg;
    const CloverField &meta; // used for meta data only

  private:
    bool tuneSharedBytes() const { return false; } // Don't tune the shared memory
    unsigned int minThreads() const { return arg.clover.volumeCB; }

  public:
    CloverRho(Arg &arg, const CloverField &meta) : arg(arg), meta(meta) {
      writeAuxString("_");
    }
    virtual ~CloverRho() { ; }
  
    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	cloverRhoKernel<<<tp.grid,tp.block,tp.shared_bytes,stream>>>(arg);
      } else {
	errorQuda("Not implemented");
      }
    }

    TuneKey tuneKey() const {
      return TuneKey(meta.VolString(), typeid(*this).name(), aux);
    }

    long long flops() const { return 0; } 
    long long bytes() const { return 2*2*arg.clover.volumeCB*(6*arg.clover.Bytes()/72); } 

    void preTune() { arg.clover.save(); }
    void postTune() { arg.clover.load(); }

  };

  template <typename Float, typename Clover>
  void cloverRho(Clover clover, const CloverField &meta, double rho) {
    CloverRhoArg<Float,Clover> arg(clover, rho);
    CloverRho<CloverRhoArg<Float,Clover> > clover_rho(arg, meta);
    clover_rho.apply(0);
  }

  template <typename Float>
  void cloverRho(const CloverField &clover, double rho) {

    if (clover.isNative()) {
      typedef typename clover_mapper<Float>::type C;
      cloverRho<Float>(C(clover, 0), clover, rho);
    } else {
      errorQuda("Clover field %d order not supported", clover.Order());
    }

  }

#endif

  void cloverRho(CloverField &clover, double rho) {

#ifdef GPU_CLOVER_DIRAC
    if (clover.Precision() == QUDA_DOUBLE_PRECISION) {
      cloverRho<double>(clover, rho);
    } else if (clover.Precision() == QUDA_SINGLE_PRECISION) {
      cloverRho<float>(clover, rho);
    } else {
      errorQuda("Precision %d not supported", clover.Precision());
    }
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
