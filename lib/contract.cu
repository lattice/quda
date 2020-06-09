#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <contract_quda.h>
#include <jitify_helper.cuh>
#include <kernels/contraction.cuh>

namespace quda {

#ifdef GPU_CONTRACT
  template <typename real, typename Arg> class Contraction : TunableVectorY
  {
protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;

private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    Contraction(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, const QudaContractType cType) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      y(y),
      cType(cType)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN: strcat(aux, "open,"); break;
      case QUDA_CONTRACT_TYPE_DR: strcat(aux, "degrand-rossi,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/contraction.cuh");
#endif
    }
    virtual ~Contraction() {}

    void apply(const qudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        std::string function_name;
        switch (cType) {
        case QUDA_CONTRACT_TYPE_OPEN: function_name = "quda::computeColorContraction"; break;
        case QUDA_CONTRACT_TYPE_DR: function_name = "quda::computeDegrandRossiContraction"; break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }

        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
                         .instantiate(Type<real>(), Type<Arg>())
                         .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                         .launch(arg);
#else
        switch (cType) {
        case QUDA_CONTRACT_TYPE_OPEN: computeColorContraction<real><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        case QUDA_CONTRACT_TYPE_DR:
          computeDegrandRossiContraction<real><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
          break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }
#endif
      } else {
        errorQuda("CPU not supported yet\n");
      }
    }

    TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(*this).name(), aux); }

    void preTune() {}
    void postTune() {}

    long long flops() const
    {
      if (cType == QUDA_CONTRACT_TYPE_OPEN)
        return 16 * 3 * 6ll * x.Volume();
      else
        return ((16 * 3 * 6ll) + (16 * (4 + 12))) * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<real>);
    }
  };

  template <typename real>
  void contract_quda(const ColorSpinorField &x, const ColorSpinorField &y, complex<real> *result,
                     const QudaContractType cType)
  {
    ContractionArg<real> arg(x, y, result);
    Contraction<real, ContractionArg<real>> contraction(arg, x, y, cType);
    contraction.apply(0);
    qudaDeviceSynchronize();
  }

#endif

  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType cType)
  {
#ifdef GPU_CONTRACT
    checkPrecision(x, y);

    if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      contract_quda<float>(x, y, (complex<float> *)result, cType);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      contract_quda<double>(x, y, (complex<double> *)result, cType);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }

#else
    errorQuda("Contraction code has not been built");
#endif
  }
} // namespace quda
