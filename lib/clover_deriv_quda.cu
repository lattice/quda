#include <tunable_nd.h>
#include <gauge_field.h>
#include <kernels/clover_deriv.cuh>

namespace quda {

  template <typename Arg>
  class DerivativeClover : TunableKernel3D {

    Arg arg;
    const GaugeField &meta;
    unsigned int minThreads() const { return meta.LocalVolumeCB(); }

  public:
    DerivativeClover(const Arg &arg, const GaugeField &meta) :
      TunableKernel3D(meta, 2, 4),
      arg(arg),
      meta(meta) { }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CloverDerivative>(tp, stream, arg);
    }

    // The force field is updated so we must preserve its initial state
    void preTune() { arg.force.save(); }
    void postTune() { arg.force.load(); }

    long long flops() const { return 16 * 198 * 3 * 4 * meta.LocalVolume(); }
    long long bytes() const { return ((8*arg.gauge.Bytes() + 4*arg.oprod.Bytes())*3 + 2*arg.force.Bytes()) * 4 * meta.LocalVolume(); }
  };

  template<typename Float>
  void cloverDerivative(GaugeField &force, GaugeField &gauge, GaugeField &oprod, double coeff, int parity)
  {
    if (oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");
    if (force.Order() != oprod.Order()) errorQuda("Force and Oprod orders must match");
    if (force.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");

    if (force.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      typedef gauge::FloatNOrder<Float, 18, 2, 18> F;
      typedef gauge::FloatNOrder<Float, 18, 2, 18> O;

      if (gauge.isNative()) {
	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  typedef typename gauge_mapper<Float,QUDA_RECONSTRUCT_NO>::type G;
	  CloverDerivArg<Float,F,G,O> arg(force, gauge, oprod, coeff, parity);
	  DerivativeClover<decltype(arg)> deriv(arg, gauge);
	  deriv.apply(0);
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

  void cloverDerivative(GaugeField &force, GaugeField &gauge, GaugeField &oprod, double coeff, QudaParity parity)
  {
#ifdef GPU_CLOVER_DIRAC
    assert(oprod.Geometry() == QUDA_TENSOR_GEOMETRY);
    assert(force.Geometry() == QUDA_VECTOR_GEOMETRY);

    for (int d=0; d<4; d++) {
      if (oprod.X()[d] != gauge.X()[d])
        errorQuda("Incompatible extended dimensions d=%d gauge=%d oprod=%d", d, gauge.X()[d], oprod.X()[d]);
    }

    int device_parity = (parity == QUDA_EVEN_PARITY) ? 0 : 1;

    if (force.Precision() == QUDA_DOUBLE_PRECISION) {
      cloverDerivative<double>(force, gauge, oprod, coeff, device_parity);
    } else {
      errorQuda("Precision %d not supported", force.Precision());
    }
#else
    errorQuda("Clover has not been built");
#endif
  }

} // namespace quda
