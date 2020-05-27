#include <cstdio>
#include <cstdlib>

#include <tune_quda.h>
#include <gauge_field.h>
#include <jitify_helper.cuh>
#include <kernels/clover_deriv.cuh>

/**
   @file clover_deriv_quda.cu

   @brief This kernel has been a bit of a pain to optimize since it is
   excessively register bound.  To reduce register pressure we use
   shared memory to help offload some of this pressure.  Annoyingly,
   the optimal approach for CUDA 8.0 is not the same as CUDA 7.5, so
   implementation is compiler version dependent.  The CUDA 8.0 optimal
   code runs 10x slower on 7.5, though the 7.5 code runs fine on 8.0.

   CUDA >= 8.0
   - Used shared memory for force accumulator matrix
   - Template mu / nu to prevent register spilling of indexing arrays
   - Force the computation routine to inline

   CUDA <= 7.5
   - Used shared memory for force accumulator matrix
   - Keep mu/nu dynamic and use shared memory to store indexing arrays
   - Do not inline computation routine

   For the shared-memory dynamic indexing arrays, we use chars, since
   the array is 4-d, a 4-d coordinate can be stored in a single word
   which means that we will not have to worry about bank conflicts,
   and the shared array can be passed to the usual indexing routines
   (getCoordsExtended and linkIndexShift) with no code changes.  This
   strategy works as long as each local lattice coordinate is less
   than 256.
 */


namespace quda {

#ifdef GPU_CLOVER_DIRAC

  template<typename Float, typename Arg>
  class CloverDerivative : public TunableVectorY {

  private:
    Arg arg;
    const GaugeField &meta;

#if defined(SHARED_ACCUMULATOR) && defined(SHARED_ARRAY)
    unsigned int sharedBytesPerThread() const { return 18*sizeof(Float) + 8; }
#elif defined(SHARED_ACCUMULATOR)
    unsigned int sharedBytesPerThread() const { return 18*sizeof(Float); }
#else
    unsigned int sharedBytesPerThread() const { return 0; }
#endif
    unsigned int sharedBytesPerBlock(const TuneParam &) const { return 0; }

    unsigned int minThreads() const { return arg.volumeCB; }
    bool tuneGridDim() const { return false; } // don't tune the grid dimension

  public:
    CloverDerivative(const Arg &arg, const GaugeField &meta) : TunableVectorY(2), arg(arg), meta(meta) {
      writeAuxString("threads=%d,prec=%lu,fstride=%d,gstride=%d,ostride=%d",
		     arg.volumeCB,sizeof(Float),arg.force.stride,
		     arg.gauge.stride,arg.oprod.stride);
#ifdef JITIFY
      create_jitify_program("kernels/clover_deriv.cuh");
#endif
    }
    virtual ~CloverDerivative() {}

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::cloverDerivativeKernel")
                         .instantiate(Type<Float>(), Type<Arg>())
                         .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                         .launch(arg);
#else
      cloverDerivativeKernel<Float><<<tp.grid,tp.block,tp.shared_bytes>>>(arg);
#endif
    } // apply

    bool advanceBlockDim(TuneParam &param) const {
      dim3 block = param.block;
      dim3 grid = param.grid;
      bool rtn = TunableVectorY::advanceBlockDim(param);
      param.block.z = block.z;
      param.grid.z = grid.z;

      if (!rtn) {
	if (param.block.z < 4) {
	  param.block.z++;
	  param.grid.z = (4 + param.block.z - 1) / param.block.z;
	  rtn = true;
	} else {
	  param.block.z = 1;
	  param.grid.z = 4;
	  rtn = false;
	}
      }
      return rtn;
    }

    void initTuneParam(TuneParam &param) const {
      TunableVectorY::initTuneParam(param);
      param.block.y = 1;
      param.block.z = 1;
      param.grid.y = 2;
      param.grid.z = 4;
    }

    void defaultTuneParam(TuneParam &param) const { initTuneParam(param); }

    // The force field is updated so we must preserve its initial state
    void preTune() { arg.force.save(); }
    void postTune() { arg.force.load(); }

    long long flops() const { return 16 * 198 * 3 * 4 * 2 * (long long)arg.volumeCB; }
    long long bytes() const { return ((8*arg.gauge.Bytes() + 4*arg.oprod.Bytes())*3 + 2*arg.force.Bytes()) * 4 * 2 * arg.volumeCB; }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }
  };

  template<typename Float>
  void cloverDerivative(cudaGaugeField &force,
			cudaGaugeField &gauge,
			cudaGaugeField &oprod,
			double coeff, int parity) {

    if (oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");

    if (force.Order() != oprod.Order()) errorQuda("Force and Oprod orders must match");

    if (force.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");

    if (force.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      typedef gauge::FloatNOrder<Float, 18, 2, 18> F;
      typedef gauge::FloatNOrder<Float, 18, 2, 18> O;

      if (gauge.isNative()) {
	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  typedef CloverDerivArg<Float,F,G,O> Arg;
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), oprod.X(), coeff, parity);
	  CloverDerivative<Float, Arg> deriv(arg, gauge);
	  deriv.apply(0);
#if 0
	} else if (gauge.Reconstruct() == QUDA_RECONSTRUCT_12) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_12>::type G;
	  typedef CloverDerivArg<Float,F,G,O> Arg;
	  Arg arg(F(force), G(gauge), O(oprod), force.X(), oprod.X(), coeff, parity);
	  CloverDerivative<Float, Arg> deriv(arg, gauge);
	  deriv.apply(0);
#endif
	} else {
	  errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
	}
      } else {
	errorQuda("Gauge order %d not supported", gauge.Order());
      }
    } else {
      errorQuda("Force order %d not supported", force.Order());
    } // force / oprod order

    qudaDeviceSynchronize();
  }
#endif // GPU_CLOVER

  void cloverDerivative(
      cudaGaugeField &force, cudaGaugeField &gauge, cudaGaugeField &oprod, double coeff, QudaParity parity)
  {
#ifdef GPU_CLOVER_DIRAC
  assert(oprod.Geometry() == QUDA_TENSOR_GEOMETRY);
  assert(force.Geometry() == QUDA_VECTOR_GEOMETRY);

  for (int d=0; d<4; d++) {
    if (oprod.X()[d] != gauge.X()[d])
      errorQuda("Incompatible extended dimensions d=%d gauge=%d oprod=%d", d, gauge.X()[d], oprod.X()[d]);
  }

  int device_parity = (parity == QUDA_EVEN_PARITY) ? 0 : 1;

  if(force.Precision() == QUDA_DOUBLE_PRECISION){
    cloverDerivative<double>(force, gauge, oprod, coeff, device_parity);
#if 0
  } else if (force.Precision() == QUDA_SINGLE_PRECISION){
    cloverDerivative<float>(force, gauge, oprod, coeff, device_parity);
#endif
  } else {
    errorQuda("Precision %d not supported", force.Precision());
  }

  return;
#else
  errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
