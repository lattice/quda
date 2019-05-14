#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <contract_quda.h>
#include <kernels/contraction.cuh>

namespace quda {
  
#ifdef GPU_CONTRACT
  template <typename Float, typename Arg> class Contraction : TunableVectorYZ
  {
    
  protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    Float *result;
    
  private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    Contraction(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, Float *result) :
      TunableVectorYZ(2,3),
      arg(arg),
      x(x),
      y(y),
      result(result)
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
	using namespace jitify::reflection;
	jitify_error = program->kernel("quda::computeContraction")
	  .instantiate(Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	computeContraction<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
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

    //DMH FIXME: Work out what these should be
    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 3 * (1 + 2 * 6) * arg.threads; }
  }; 
  
  template<typename Float>
  void contract_quda(const ColorSpinorField &x, const ColorSpinorField &y,
		     Float *result) {
    ContractionArg<Float> arg(x, y, result);
    Contraction<Float, ContractionArg<Float>> contraction(arg, x, y, result);
    contraction.apply(0);
    qudaDeviceSynchronize();
  }
  
#endif
  
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y,
		    void *result)
  {
#ifdef GPU_CONTRACT
    
    if(x.Precision() != y.Precision()) {
      errorQuda("Contracted fields must have the same precision\n");
    }
    
    if (x.Precision() == QUDA_SINGLE_PRECISION){
      contract_quda<float>(x, y, (float*)result);
    } else if(x.Precision() == QUDA_DOUBLE_PRECISION) {
      contract_quda<double>(x, y, (double*)result);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
    return;
    
#else
    errorQuda("Contraction code has not been built");
#endif
  }
} // namespace quda
