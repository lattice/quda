#include <tunable_nd.h>
#include <gauge_field.h>
#include <kernels/clover_deriv.cuh>

namespace quda {

  template <typename Float, QudaReconstructType recon>
  class DerivativeClover : TunableKernel3D {
    GaugeField &force;
    GaugeField &gauge;
    GaugeField &oprod;
    double coeff;
    int parity;
    unsigned int minThreads() const { return gauge.LocalVolumeCB(); }

  public:
    DerivativeClover(GaugeField &force, GaugeField &gauge, GaugeField &oprod, double coeff, int parity) :
      TunableKernel3D(gauge, 2, 4),
      force(force),
      gauge(gauge),
      oprod(oprod),
      coeff(coeff),
      parity(parity)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream){
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<CloverDerivative>(tp, stream, CloverDerivArg<Float, recon>(force, gauge, oprod, coeff, parity));
    }

    // The force field is updated so we must preserve its initial state
    void preTune() { force.backup(); }
    void postTune() { force.restore(); }

    long long flops() const { return 16 * 198 * 3 * 4 * gauge.LocalVolume(); }
    long long bytes() const
    {
      return ((8 * gauge.Reconstruct() + 4 * oprod.Reconstruct()) * 3 + 2 * force.Reconstruct()) * 4 * gauge.LocalVolume() * gauge.Precision();
    }
  };

  template<typename Float>
  void cloverDerivative(GaugeField &force, GaugeField &gauge, GaugeField &oprod, double coeff, int parity)
  {
    if (oprod.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");
    if (force.Order() != oprod.Order()) errorQuda("Force and Oprod orders must match");
    if (force.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Force field does not support reconstruction");

    if (force.Order() == QUDA_FLOAT2_GAUGE_ORDER) {
      if (gauge.isNative()) {
	if (gauge.Reconstruct() == QUDA_RECONSTRUCT_NO) {
	  DerivativeClover<Float, QUDA_RECONSTRUCT_NO> deriv(force, gauge, oprod, coeff, parity);
	} else {
	  errorQuda("Reconstruction type %d not supported",gauge.Reconstruct());
	}
      } else {
	errorQuda("Gauge order %d not supported", gauge.Order());
      }
    } else {
      errorQuda("Force order %d not supported", force.Order());
    } // force / oprod order
  }

#if defined(GPU_CLOVER_DIRAC) && (QUDA_PRECISION & 8)
  void cloverDerivative(GaugeField &force, GaugeField &gauge, GaugeField &oprod, double coeff, QudaParity parity)
  {
    assert(oprod.Geometry() == QUDA_TENSOR_GEOMETRY);
    assert(force.Geometry() == QUDA_VECTOR_GEOMETRY);

    for (int d=0; d<4; d++) {
      if (oprod.X()[d] != gauge.X()[d])
        errorQuda("Incompatible extended dimensions d=%d gauge=%d oprod=%d", d, gauge.X()[d], oprod.X()[d]);
    }

    if (force.Precision() == QUDA_DOUBLE_PRECISION) {
      cloverDerivative<double>(force, gauge, oprod, coeff, (parity == QUDA_EVEN_PARITY) ? 0 : 1);
    } else {
      errorQuda("Precision %d not supported", force.Precision());
    }
  }
#else
  void cloverDerivative(GaugeField &, GaugeField &, GaugeField &, double, QudaParity)
  {
#ifdef GPU_CLOVER_DIRAC
    errorQuda("QUDA_PRECISION=%d does not enable double precision", QUDA_PRECISION);
#else
    errorQuda("Clover has not been built");
#endif
  }
#endif

} // namespace quda
