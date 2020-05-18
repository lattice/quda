#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <launch_kernel.cuh>
#include <kernels/evec_project_sum.cuh>
#include <jitify_helper.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Arg> class EvecProjectSumCompute : TunableLocalParity
  {
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;

  private:
    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

  public:
    EvecProjectSumCompute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y) :
      TunableLocalParity(),
      arg(arg),
      x(x),
      y(y)
    {
      strcat(aux, "evec_project_sum,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/evec_project_sum.cuh");
#endif
    }
    
    void apply(const cudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
	for (int i=0; i<2*x.Nspin()*x.X(3); i++) ((double*)arg.result_h)[i] = 0.0; 
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
        tp.grid.z = x.X(3);

#ifdef JITIFY
        std::string function_name = "quda::spin";
	
        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	LAUNCH_KERNEL_LOCAL_PARITY(computeEvecProjectSum, (*this), tp, stream, arg, Arg);
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
      return x.Bytes() + y.Bytes();
    }
  };
  
  template <typename Float, int nColor>
  void evecProjectSum(const ColorSpinorField &x, const ColorSpinorField &y, double *result)
  {
    EvecProjectSumArg<Float, nColor> arg(x, y);
    EvecProjectSumCompute<decltype(arg)> evec_project_sum(arg, x, y);
    evec_project_sum.apply(0);
    qudaDeviceSynchronize();

    for (int i=0; i<2*x.Nspin()*x.X(3); i++) result[i] = ((Float*)arg.result_h)[i];
  }
  
  void evecProjectSumQuda(const ColorSpinorField &x, const ColorSpinorField &y, double *result)
  {
    checkPrecision(x, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    if (x.Ncolor() == 3) {
      if (x.Precision() == QUDA_SINGLE_PRECISION) {
        evecProjectSum<float, 3>(x, y, result);
      } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
        evecProjectSum<double, 3>(x, y, result);
      } else {
        errorQuda("Precision %d not supported", x.Precision());
      }
    } else {
      errorQuda("nColors = %d is not supported", x.Ncolor());
    }
  }
} // namespace quda
