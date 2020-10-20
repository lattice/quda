#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <launch_kernel.cuh>
#include <contract_quda.h>
#include <jitify_helper.cuh>
#include <kernels/contraction.cuh>

namespace quda {

#ifdef GPU_CONTRACT
  // Summed contraction type kernels: Inherit from TunableLocalParityReduction
  // so there is a reduction class built in
  //----------------------------------------------------------------------------
  template <typename Arg> class ContractionSummedCompute : TunableLocalParityReduction
  {
  protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;

  private:
    bool staticGridDim() const { return true; } // Maintain grid dims set in this class.
    unsigned int minThreads() const { return arg.threads; }
    bool tuneSharedBytes() const { return false; }
    void initTuneParam(TuneParam &param) const {
      TunableLocalParityReduction::initTuneParam(param);
      param.block.y = 2;
      param.grid.z = x.X(Arg::reduction_dim); // Reduction dimension is mapped to different blocks in the Z dimension
    }

    void defaultTuneParam(TuneParam &param) const {
      TunableLocalParityReduction::defaultTuneParam(param);
      param.block.y = 2;
      param.grid.z = x.X(Arg::reduction_dim); // Reduction dimension is mapped to different blocks in the Z dimension
    }

  public:
    ContractionSummedCompute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, const QudaContractType cType) :
      TunableLocalParityReduction(),
      arg(arg),
      x(x),
      y(y),
      cType(cType)
    {
      switch (cType) {
      case QUDA_CONTRACT_TYPE_OPEN_SUM_T: strcat(aux, "open-sum-t,"); break;
      case QUDA_CONTRACT_TYPE_OPEN_SUM_Z: strcat(aux, "open-sum-z,"); break;
      case QUDA_CONTRACT_TYPE_DR_FT_T: strcat(aux, "degrand-rossi-ft-t,"); break;
      case QUDA_CONTRACT_TYPE_DR_FT_Z: strcat(aux, "degrand-rossi-ft-z,"); break;
      default: errorQuda("Unexpected contraction type %d", cType);
      }
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/contraction.cuh");
#endif
    }
    virtual ~ContractionSummedCompute() {}

    void apply(const qudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        std::string function_name;
        switch (cType) {
	case QUDA_CONTRACT_TYPE_OPEN_SUM_T: function_name = "quda::computeColorContractionSummed"; break;
	case QUDA_CONTRACT_TYPE_OPEN_SUM_Z: function_name = "quda::computeColorContractionSummed"; break;
        default: errorQuda("Unexpected contraction type %d", cType);
        }
	
        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<real>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
        switch (cType) {
	case QUDA_CONTRACT_TYPE_DR_FT_T:
        case QUDA_CONTRACT_TYPE_DR_FT_Z:
	  LAUNCH_KERNEL_LOCAL_PARITY(computeDegrandRossiContractionFT, (*this), tp, stream, arg, Arg);
	  break;
	case QUDA_CONTRACT_TYPE_OPEN_SUM_T:
        case QUDA_CONTRACT_TYPE_OPEN_SUM_Z:
	  LAUNCH_KERNEL_LOCAL_PARITY(computeColorContractionSummed, (*this), tp, stream, arg, Arg);
	  break;
	case QUDA_CONTRACT_TYPE_OPEN_FT_T:
        case QUDA_CONTRACT_TYPE_OPEN_FT_Z:
	  errorQuda("Contraction type %d not implemented", cType);
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
      if (cType != QUDA_CONTRACT_TYPE_OPEN || cType != QUDA_CONTRACT_TYPE_DR)
        return x.Nspin() * x.Nspin() * 3 * 6ll * x.Volume();
      else
        return ((x.Nspin() * x.Nspin() * 3 * 6ll) + (x.Nspin() * x.Nspin() * (4 + 12))) * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes();
    }
  };

  template <typename real>
  void contract_summed_quda(const ColorSpinorField &x, const ColorSpinorField &y,
			    std::vector<Complex> &result, const QudaContractType cType,
                            const int *const source_position, const int *const pxpypzpt, const size_t s1, const size_t b1)
  {
    if (cType == QUDA_CONTRACT_TYPE_DR_FT_Z || cType == QUDA_CONTRACT_TYPE_OPEN_SUM_Z){
      ContractionSummedArg<real, 2> arg(x, y, source_position, pxpypzpt, s1, b1); // reduce in the z direction
      ContractionSummedCompute<decltype(arg)> contraction_with_sum(arg, x, y, cType);
      contraction_with_sum.apply(0);
      arg.complete(result);
    } else if (cType == QUDA_CONTRACT_TYPE_DR_FT_T || cType == QUDA_CONTRACT_TYPE_OPEN_SUM_T) {
      ContractionSummedArg<real, 3> arg(x, y, source_position, pxpypzpt, s1, b1); // reduce in the t direction
      ContractionSummedCompute<decltype(arg)> contraction_with_sum(arg, x, y, cType);
      contraction_with_sum.apply(0);
      arg.complete(result);
    } else {
      errorQuda("Unexpected contraction type %d given for summed contraaction kernel.", cType);
    }
  }
#endif //GPU_CONTRACT
  
  void contractSummedQuda(const ColorSpinorField &x, const ColorSpinorField &y,
			  std::vector<Complex> &result, const QudaContractType cType,
			  const int *const source_position, const int *const pxpypzpt, const size_t s1, const size_t b1)
  {
#ifdef GPU_CONTRACT
    checkPrecision(x, y);

    if (x.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS || y.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      errorQuda("Unexpected gamma basis x=%d y=%d", x.GammaBasis(), y.GammaBasis());
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 4) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      contract_summed_quda<float>(x, y, result, cType, source_position, pxpypzpt, s1, b1);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      contract_summed_quda<double>(x, y, result, cType, source_position, pxpypzpt, s1, b1);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
#else
    errorQuda("Contraction code has not been built");
#endif
  }
  //----------------------------------------------------------------------------



  // Non-summed contraction type kernels
  //----------------------------------------------------------------------------
#ifdef GPU_CONTRACT
  template <typename real, typename Arg> class Contraction : TunableVectorY
  {
protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const QudaContractType cType;

private:
    bool tuneSharedBytes() const { return false; }
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
        case QUDA_CONTRACT_TYPE_OPEN: qudaLaunchKernel(computeColorContraction<real, Arg>, tp, stream, arg); break;
        case QUDA_CONTRACT_TYPE_DR:   qudaLaunchKernel(computeDegrandRossiContraction<real, Arg>, tp, stream, arg); break;
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
    if(cType == QUDA_CONTRACT_TYPE_OPEN || cType == QUDA_CONTRACT_TYPE_DR) {
      ContractionArg<real> arg(x, y, result);
      Contraction<real, ContractionArg<real>> contraction(arg, x, y, cType);
      contraction.apply(0);
      qudaDeviceSynchronize();
    } else {
      errorQuda("Unexpected contraction type %d given for non-summed contraaction kernel.", cType);
    }
  }
#endif //GPU_CONTRACT
  
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
