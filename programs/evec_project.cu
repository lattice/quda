#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <kernels/evec_project.cuh>
//#include <stoch_laph_quark_smear.h>
#include <jitify_helper.cuh>


namespace quda {

  template <typename real, typename Arg> class EvecProject : TunableVectorY
  {
protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;

private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    EvecProject(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      y(y),
    {
      strcat(aux, "evec_project,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/evec_project.cuh");
#endif
    }
    virtual ~EvecProject() {}

    void apply(const cudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        std::string function_name = "quda::spin";
	
        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<real>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	computeEvecProject<real><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
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
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<real>);
    }
  };
  
  template <typename real>
  void evec_project_quda(const ColorSpinorField &x, const ColorSpinorField &y, const int alpha)
  {
    EvecProjectArg<real> arg(x, y, result);
    EvecProject<real, EvecProjectArg<real>> evec_project(arg, x, y);
    evec_project.apply(0);
    qudaDeviceSynchronize();
  }
  
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result)
  {
    checkPrecision(x, y);
    
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      evec_project_quda<float>(x, y, (complex<float> *)result);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      evec_project_quda<double>(x, y, (complex<double> *)result);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
  }
} // namespace quda
