#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#include <launch_kernel.cuh>
#include <jitify_helper.cuh>
#include <kernels/gauge_energy.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Arg> class EnergyCompute : TunableLocalParity
  {
    Arg &arg;
    const GaugeField &meta;

private:
    bool tuneGridDim() const { return true; }
    unsigned int minThreads() const { return arg.threads; }

public:
    EnergyCompute(Arg &arg, const GaugeField &meta) : 
      TunableLocalParity(),
      arg(arg), 
      meta(meta)
    {
#ifdef JITIFY
      create_jitify_program("kernels/gauge_energy.cuh");
#endif
      strcpy(aux,compile_type_str(meta));
    }

    void apply(const cudaStream_t &stream)
    {
      if (meta.Location() == QUDA_CUDA_FIELD_LOCATION) {
	for (int i=0; i<2; i++) ((double*)arg.result_h)[i] = 0.0;   
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        using namespace jitify::reflection;
        jitify_error = program->kernel("quda::energyComputeKernel")
                         .instantiate((int)tp.block.x, Type<Arg>())
                         .configure(tp.grid, tp.block, tp.shared_bytes, stream)
                         .launch(arg);
#else
	//LAUNCH_KERNEL(energyComputeKernel, (*this), tp, stream, arg, Arg);
	LAUNCH_KERNEL_LOCAL_PARITY(energyComputeKernel, (*this), tp, stream, arg, Arg);
#endif
      } else { // run the CPU code
        errorQuda("energyComputeKernel not supported on CPU");
      }
    }

    TuneKey tuneKey() const
    {
      return TuneKey(meta.VolString(), typeid(*this).name(), meta.AuxString());
    }

    long long flops() const { return 2 * arg.threads * (3 * 198 + 9); }
    long long bytes() const { return 2 * arg.threads * ((6 * 18)) * sizeof(typename Arg::Float); }
  }; // EnergyCompute

  template <typename Float, int nColor, QudaReconstructType recon> struct Energy {
    Energy(const GaugeField &Fmunu, double2 &energy)
    {
      if (!Fmunu.isNative()) errorQuda("Energy computation only supported on native ordered fields");
      
      EnergyArg<Float, nColor, recon> arg(Fmunu);
      EnergyCompute<decltype(arg)> energyCompute(arg, Fmunu);
      energyCompute.apply(0);
      qudaDeviceSynchronize();
      
      checkCudaError();
      comm_allreduce_array((double *)arg.result_h, 2);
      //Volume normalisation
      for (int i=0; i<2; i++) ((double*)&energy)[i] = ((double*)arg.result_h)[i] / (2.0*arg.threads*comm_size());
    }
  };
  
  double3 computeEnergy(const GaugeField &Fmunu)
  {
    double2 energy = make_double2(0.0,0.0);
#ifdef GPU_GAUGE_TOOLS
    instantiate<Energy, ReconstructNone>(Fmunu, energy);
#else
    errorQuda("Gauge tools are not built");
#endif // GPU_GAUGE_TOOLS
    double3 energy3 = make_double3(energy.x+energy.y,energy.x,energy.y);
    return energy3;
  }
} // namespace quda
