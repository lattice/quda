#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <launch_kernel.cuh>
#include <kernels/evec_project.cuh>
#include <jitify_helper.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, typename Arg> class EvecProjectCompute : TunableVectorY
  {    
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;

  private:
    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }
    
public:
    EvecProjectCompute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      y(y)
    {
      strcat(aux, "evec_project,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/evec_project.cuh");
#endif
    }
    virtual ~EvecProjectCompute() {}
    
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
	computeEvecProject<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
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
      // 4 prop spins, 1 evec spin, 3 color, 6 complex, lattice volume
      return 4 * 3 * 6ll * x.Volume();
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };
  
  template <typename Float, int nColor>
  void evecProject(const ColorSpinorField &x, const ColorSpinorField &y, Float *result)
  {
    EvecProjectArg<Float, nColor> arg(x, y, result);
    EvecProjectCompute<Float, EvecProjectArg<Float, nColor>> evec_project(arg, x, y);
    evec_project.apply(0);
    qudaDeviceSynchronize();
  }
  
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result)
  {
    checkPrecision(x, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());

    
    if(x.Ncolor() == 3) {
      if (x.Precision() == QUDA_SINGLE_PRECISION) {
	evecProject<float, 3>(x, y, (float*)result);
      } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	evecProject<double, 3>(x, y, (double*)result);
      } else {
	errorQuda("Precision %d not supported", x.Precision());
      }
    } else {
      errorQuda("nColors = %d is not supported", x.Ncolor());
    }
  }
} // namespace quda
