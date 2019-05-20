#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <contract_quda.h>
#include <kernels/contraction.cuh>

namespace quda {
  
#ifdef GPU_CONTRACT
  template <typename Float, typename Arg> class Contraction : TunableVectorY
  {
    
  protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    Float *result;
    const QudaContractType cType;
    
  private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    Contraction(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, Float *result, const QudaContractType cType) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      y(y),
      result(result),
      cType(cType)
    {
#ifdef JITIFY
      create_jitify_program("kernels/contraction.cuh");
#endif
    }
    virtual ~Contraction() {}
    
    void apply(const cudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
	TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
	std::string function_name;
	switch(cType) {
	case QUDA_CONTRACT_TYPE_OPEN:
	  function_name = "quda::computeColorContraction";
	  break;
	case QUDA_CONTRACT_TYPE_DR:
	  function_name = "quda::computeDegrandRossiContraction";
	  break;
	case QUDA_CONTRACT_TYPE_DP:
	  errorQuda("Contraction type not implemented");
	  //function_name = "quda::computeDiracPauliContraction";
	  break;
	default:
	  function_name = "quda::computeColorContraction";
	  break;
	}
	
	using namespace jitify::reflection;
	jitify_error = program->kernel(function_name)
	  .instantiate(Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	switch(cType) {
	case QUDA_CONTRACT_TYPE_OPEN:
	  computeColorContraction<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
	  break;
	case QUDA_CONTRACT_TYPE_DR:	  
	  computeDegrandRossiContraction<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
	  break;
	case QUDA_CONTRACT_TYPE_DP:
	  errorQuda("Contraction type not implemented");
	  //computeDiracPauliContraction<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
	  break;
	default:
	  computeColorContraction<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
	  break;
	}	
#endif
      } else {
	errorQuda("CPU not supported yet\n");
	//computeContractionCPU(arg);
      }
    }
    
    TuneKey tuneKey() const
    {
      std::stringstream aux;
      aux << "threads=" << arg.threads << ",prec=" << sizeof(Float);
      return TuneKey(x.VolString(), typeid(*this).name(), aux.str().c_str());
    }
    
    void preTune() {} 
    void postTune() {}
    
    long long flops() const {
      if (cType == QUDA_CONTRACT_TYPE_OPEN) return 16 * 3 * 6ll * arg.threads;
      else return ((16 * 3 * 6ll) + (16 * (4+12))) * arg.threads;
    }
    long long bytes() const {
      return  24 * 2 * arg.threads;
    }
  }; 
  
  template<typename Float>
  void contract_quda(const ColorSpinorField &x, const ColorSpinorField &y,
		     Float *result, const QudaContractType cType) {
    ContractionArg<Float> arg(x, y, result);
    Contraction<Float, ContractionArg<Float>> contraction(arg, x, y, result, cType);
    contraction.apply(0);
    qudaDeviceSynchronize();
  }
  
#endif
  
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, const QudaContractType cType)
  {
#ifdef GPU_CONTRACT
    
    if(x.Precision() != y.Precision()) {
      errorQuda("Contracted fields must have the same precision\n");
    }
    
    if (x.Precision() == QUDA_SINGLE_PRECISION){
      contract_quda<float>(x, y, (float*)result, cType);
    } else if(x.Precision() == QUDA_DOUBLE_PRECISION) {
      contract_quda<double>(x, y, (double*)result, cType);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
    return;
    
#else
    errorQuda("Contraction code has not been built");
#endif
  }
} // namespace quda
