#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <launch_kernel.cuh>
#include <kernels/color_contract.cuh>
#include <jitify_helper.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, typename Arg> class ColorContractCompute : TunableLocalParity
  {
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;

  private:
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    ColorContractCompute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y) :
      TunableLocalParity(),
      arg(arg),
      x(x),
      y(y)
    {
      strcat(aux, "color_contract,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/color_contract.cuh");
#endif
    }

    virtual ~ColorContractCompute() {}
    
    void apply(const cudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	
#ifdef JITIFY
        std::string function_name = "quda::spin";
	
        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	computeColorContract<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
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
      // 1 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 1 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };
  
  template <typename Float, int nColor>
  void colorContract(const ColorSpinorField &x, const ColorSpinorField &y, complex<Float> *result, cudaStream_t stream)
  {
    ColorContractArg<Float, nColor> arg(x, y, result);
    ColorContractCompute<Float, ColorContractArg<Float, nColor>> color_contract(arg, x, y);
    color_contract.apply(stream);
    //qudaDeviceSynchronize();
  }
  
  void colorContractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, cudaStream_t stream)
  {
    checkPrecision(x, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 1 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      colorContract<float, 3>(x, y, (complex<float> *)result, stream);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      colorContract<double, 3>(x, y, (complex<double> *)result, stream);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
  }
} // namespace quda
