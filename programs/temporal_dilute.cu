#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <kernels/temporal_dilute.cuh>
#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <instantiate.h>

namespace quda {

  template <typename real, typename Arg> class TemporalDilute : TunableVectorY
  {
protected:
    Arg &arg;
    ColorSpinorField &x;
    const int t;
    
private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    TemporalDilute(Arg &arg, ColorSpinorField &x, const int t) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      t(t)
    {
      strcat(aux, "temporal_dilute,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/temporal_dilute.cuh");
#endif
    }
    virtual ~TemporalDilute() {}

    void apply(const cudaStream_t &stream)
    {
      if (x.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        std::string function_name = "quda::temporalDilute";
	
        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate(Type<real>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	computeTemporalDilute<real><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
      } else {
        errorQuda("CPU not supported yet\n");
      }
    }
    
    TuneKey tuneKey() const { return TuneKey(x.VolString(), typeid(*this).name(), aux); }

    void preTune() {}
    void postTune() {}

    long long flops() const { return 0; }
    
    long long bytes() const
    {
      return x.Bytes() + x.Nspin() * x.Ncolor() * x.Volume() * sizeof(complex<real>);
    }
  };
  
  template <typename real>
  void temporal_dilute_quda(ColorSpinorField &x, const ColorSpinorField &y, const int alpha)
  {
    TemporalDiluteArg<real> arg(x, t);
    TemporalDilute<real, TemporalDiluteArg<real>> temporal_dilute(arg, x, t);
    temporal_dilute.apply(0);
    qudaDeviceSynchronize();
    // Check quarks
    for(int j=0; j<256; j++) {
      //printfQuda("QUARK CU spin = %d\n", alpha);
      //x.PrintVector(j);
    }      
  }
  
  void temporalDiluteQuda(ColorSpinorField &x, const int t)
  {

    if (x.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d", x.Ncolor());
    if (x.Nspin() != 4) errorQuda("Unexpected number of spins x=%d", x.Nspin());
    
    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      temporal_dilute_quda<float>(x, t);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      temporal_dilute_quda<double>(x, t);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
  }
} // namespace quda
