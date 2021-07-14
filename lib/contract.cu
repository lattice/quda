#include <color_spinor_field.h>
#include <contract_quda.h>
#include <tunable_nd.h>
#include <instantiate.h>
#include <kernels/contraction.cuh>

namespace quda {

  template <typename Float, int nColor> class Contraction : TunableKernel2D
  {
    complex<Float> *result;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;
    unsigned int minThreads() const { return x.VolumeCB(); }

public:
    Contraction(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType cType) :
      TunableKernel2D(x, 2),
      result(static_cast<complex<Float>*>(result)),
      x(x),
      y(y),
      cType(cType)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN: strcat(aux, "open,"); break;
      case QUDA_CONTRACT_TYPE_DR: strcat(aux, "degrand-rossi,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      ContractionArg<Float, nColor> arg(x, y, result);
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN: launch<ColorContract>(tp, stream, arg); break;
      case QUDA_CONTRACT_TYPE_DR:   launch<DegrandRossiContract>(tp, stream, arg); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
    }

    long long flops() const
    {
      if (cType == QUDA_CONTRACT_TYPE_OPEN)
        return 16 * 3 * 6ll * x.Volume();
      else
        return ((16 * 3 * 6ll) + (16 * (4 + 12))) * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };

#ifdef GPU_CONTRACT
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType cType)
  {
    checkPrecision(x, y);
    if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    instantiate<Contraction>(x, y, result, cType);
  }
#else
  void contractQuda(const ColorSpinorField &, const ColorSpinorField &, void *, const QudaContractType)
  {
    errorQuda("Contraction code has not been built");
  }
#endif

} // namespace quda
