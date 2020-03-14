#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <kernels/spin_dilute.cuh>
//#include <stoch_laph_quark_smear.h>
#include <jitify_helper.cuh>


namespace quda {

  template <typename real, typename Arg> class SpinDilute : TunableVectorY
  {
protected:
    Arg &arg;
    const ColorSpinorField &x;
    const ColorSpinorField &y;
    const int alpha;

private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    SpinDilute(Arg &arg, const ColorSpinorField &x, const ColorSpinorField &y, const int alpha) :
      TunableVectorY(2),
      arg(arg),
      x(x),
      y(y),
      alpha(alpha)
    {
      strcat(aux, "spin_dilute,");
      strcat(aux, x.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/spin_dilute.cuh");
#endif
    }
    virtual ~SpinDilute() {}

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
	computeSpinDilute<real><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
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
      return 0;
    }

    long long bytes() const
    {
      return x.Bytes() + y.Bytes() + x.Nspin() * x.Nspin() * x.Volume() * sizeof(complex<real>);
    }
  };
  
  template <typename real>
  void spin_dilute_quda(const ColorSpinorField &x, const ColorSpinorField &y, const int alpha)
  {
    SpinDiluteArg<real> arg(x, y, alpha);
    SpinDilute<real, SpinDiluteArg<real>> spin_dilute(arg, x, y, alpha);
    spin_dilute.apply(0);
    qudaDeviceSynchronize();
  }
  
  void spinDiluteQuda(const ColorSpinorField &x, const ColorSpinorField &y, const int alpha)
  {
    checkPrecision(x, y);

    if(alpha < 0 || alpha > 3) errorQuda("Unexpected spin index alpha=%d", alpha);
    if (x.Ncolor() != 3 || y.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d", x.Ncolor(), y.Ncolor());
    if (x.Nspin() != 4 || y.Nspin() != 1) errorQuda("Unexpected number of spins x=%d y=%d", x.Nspin(), y.Nspin());
    
    if (x.Precision() == QUDA_SINGLE_PRECISION) {
      spin_dilute_quda<float>(x, y, alpha);
    } else if (x.Precision() == QUDA_DOUBLE_PRECISION) {
      spin_dilute_quda<double>(x, y, alpha);
    } else {
      errorQuda("Precision %d not supported", x.Precision());
    }
  }
} // namespace quda
