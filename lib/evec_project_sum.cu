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

  template <typename Float, typename Arg> class EvecProjectSumCompute : TunableLocalParity
  {
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;

  private:
    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }
    unsigned int tuneBlockDimMultiple() const { return x.X()[0]*x.X()[1]*x.X()[2]/2; } // Only tune multiples 32 that are factors of this number
    
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
	for (int i=0; i<2*x.Nspin()*Arg::t; i++) ((double*)arg.result_h)[i] = 0.0; 
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
	
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
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<Float>);
    }
  };
  
  template <typename Float, int nColor, int t>
  void evecProjectSum(const ColorSpinorField &x, const ColorSpinorField &y, double *result)
  {
    EvecProjectSumArg<Float, nColor, t> arg(x, y);
    EvecProjectSumCompute<Float, EvecProjectSumArg<Float, nColor, t>> evec_project_sum(arg, x, y);
    evec_project_sum.apply(0);
    qudaDeviceSynchronize();

    for(int i=0; i<2*x.Nspin()*t; i++) result[i] = ((Float*)arg.result_h)[i];
  }
  
  void evecProjectSumQuda(const ColorSpinorField &x, const ColorSpinorField &y, double *result)
  {
    checkPrecision(x, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    if(x.Ncolor() == 3) {

      int t_dim_size = x.X()[3];

      switch(t_dim_size) {
      case 4 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 4>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 4>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;

      case 8 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 8>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 8>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;

      case 12 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 12>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 12>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;

      case 16 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 16>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 16>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;

      case 24 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 24>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 24>(x, y, result);
	  printfQuda("Double prec!\n");
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;

      case 32 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 32>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 32>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;

      case 48 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 48>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 48>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;


      case 64 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 64>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 64>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;

      case 96 :
	if (x.Precision() == QUDA_SINGLE_PRECISION) {
	  evecProjectSum<float, 3, 96>(x, y, result);
	} else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
	  evecProjectSum<double, 3, 96>(x, y, result);
	} else {
	  errorQuda("Precision %d not supported", x.Precision());
	}
	break;
	
      default :
	errorQuda("Uninstantiated T dim %d", t_dim_size);
	
      }
    } else {
      errorQuda("nColors = %d is not supported", x.Ncolor());
    }
  }
} // namespace quda
