#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <launch_kernel.cuh>
#include <kernels/color_cross.cuh>
#include <jitify_helper.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, typename Arg> class ColorCrossCompute : TunableVectorY
  {
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    ColorSpinorField &result;

  private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }
    
  public:
    ColorCrossCompute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      y(y),
      result(result)
    {
      strcat(aux, "color_cross,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/color_cross.cuh");
#endif
    }

    virtual ~ColorCrossCompute() {}
    
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
	computeColorCross<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
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
      // 1 spin, 3 color, 6 complex, lattice volume
      return 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };
  
  template <typename Float, int nColor>
  void colorCross(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result)
  {
    ColorCrossArg<Float, nColor> arg(x, y, result);
    ColorCrossCompute<Float, ColorCrossArg<Float, nColor>> color_cross(arg, x, y, result);
    color_cross.apply(0);
    qudaDeviceSynchronize();
  }
  
  void colorCrossQuda(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result)
  {
    checkPrecision(x, y);
    checkPrecision(result, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3 || result.Ncolor() != 3) errorQuda("Unexpected number of colors x = %d y = %d result = %d", x.Ncolor(), y.Ncolor(), result.Ncolor());
    if (x.Nspin() != 1 || y.Nspin() != 1 || result.Nspin() != 1) errorQuda("Unexpected number of spins x = %d y = %d result = %d", x.Nspin(), y.Nspin(), result.Nspin());
    
    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      colorCross<float, 3>(x, y, result);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      colorCross<double, 3>(x, y, result);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
  }
} // namespace quda
