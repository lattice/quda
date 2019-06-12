#include <tune_quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <contract_quda.h>
#include <kernels/sink_project.cuh>

namespace quda {

#ifdef GPU_CONTRACT
  template <typename Float, typename Arg> class SinkProject : TunableVectorY
  {
  protected:
    Arg &arg;
    const ColorSpinorField &evecs;
    const ColorSpinorField &psi;
    
private:
    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:
    SinkProject(Arg &arg, const ColorSpinorField &evecs, const ColorSpinorField &psi) :
      TunableVectorY(2),
      arg(arg),
      evecs(evecs),
      psi(psi)
    {
      strcat(aux, "spin_project,");      
      strcat(aux, psi.AuxString());
#ifdef JITIFY
      create_jitify_program("kernels/spin_project.cuh");
#endif
    }
    virtual ~SinkProject() {}

    void apply(const cudaStream_t &stream)
    {
      if (psi.Location() == QUDA_CUDA_FIELD_LOCATION) {
        TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
        std::string function_name;
	function_name = "quda::computeSinkProjection";
	
        using namespace jitify::reflection;
        jitify_error = program->kernel(function_name)
	  .instantiate((int)tp.block.x, Type<Float>(), Type<Arg>())
	  .configure(tp.grid, tp.block, tp.shared_bytes, stream)
	  .launch(arg);
#else
	//LAUNCH_KERNEL(computeSinkProjection, tp, stream, arg, Float);
	computeSinkProjection<Float><<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
      } else {
        errorQuda("CPU not supported yet\n");
      }
    }
    
    TuneKey tuneKey() const { return TuneKey(psi.VolString(), typeid(*this).name(), aux); }
    
    void preTune() {}
    void postTune() {}
    
    long long flops() const
    {
      // FIXME
      return ((16 * 3 * 6ll) + (16 * (4 + 12))) * psi.Volume();
    }
    
    long long bytes() const
    {
      return evecs.Bytes() + psi.Bytes() + evecs.Nspin() * psi.Nspin() * psi.Volume() * sizeof(complex<Float>);
    }
  };

  template <typename Float>
  void sink_project_quda(const ColorSpinorField &evecs, const ColorSpinorField &psi, complex<Float> *q)
  {
    SinkProjectArg<Float> arg(evecs, psi, q);
    SinkProject<Float, SinkProjectArg<Float>> sink_project(arg, evecs, psi);
    sink_project.apply(0);
    qudaDeviceSynchronize();
  }

#endif

  void sinkProjectQuda(const ColorSpinorField &evecs, const ColorSpinorField &psi, void *q)
  {
#ifdef GPU_CONTRACT
    checkPrecision(evecs, psi);

    if (psi.GammaBasis() != QUDA_DEGRAND_ROSSI_GAMMA_BASIS)
      errorQuda("Unexpected gamma %d", psi.GammaBasis());
    if (evecs.Ncolor() != 3 || psi.Ncolor() != 3) errorQuda("Unexpected number of colors x=%d y=%d (expect (x=3 y=3)", evecs.Ncolor(), psi.Ncolor());
    if (evecs.Nspin()  != 1 || psi.Nspin()  != 4) errorQuda("Unexpected number of spins  x=%d y=%d (expect (x=1 y=4)", evecs.Nspin(),  psi.Nspin());
    
    if (psi.Precision() == QUDA_SINGLE_PRECISION) {
      sink_project_quda<float>(evecs, psi, (complex<float> *)q);
    } else if (psi.Precision() == QUDA_DOUBLE_PRECISION) {
      sink_project_quda<double>(evecs, psi, (complex<double> *)q);
    } else {
      errorQuda("Precision %d not supported", psi.Precision());
    }
    
#else
    errorQuda("LapH-NN code has not been built");
#endif
  }
} // namespace quda
