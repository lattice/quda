#include <quda_internal.h>
#include <tune_quda.h>
#include <gauge_field.h>

#include <jitify_helper.cuh>
#include <kernels/gauge_wilson_flow.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon>
  class GaugeWFlowStep : TunableVectorYZ  
  {
    GaugeWFlowArg<Float, nColor, recon> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

public:

    GaugeWFlowStep(GaugeField &out, GaugeField &temp, const GaugeField &in, const double epsilon, const WFlowStepType stepType) :
      TunableVectorYZ(2, 3),
      arg(out, temp, in, epsilon, stepType),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
      if      (stepType == WFLOW_STEP_W1) strcat(aux,",computeWFlowStepW1");
      else if (stepType == WFLOW_STEP_W2) strcat(aux,",computeWFlowStepW2");
      else if (stepType == WFLOW_STEP_VT) strcat(aux,",computeWFlowStepVT");
#ifdef JITIFY
      create_jitify_program("kernels/gauge_wilson_flow.cuh");
#endif
      apply(0);
      qudaDeviceSynchronize();
    }
    
    void apply(const cudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeWFlowStep").instantiate(Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      computeWFlowStep<<<tp.grid, tp.block, tp.shared_bytes>>>(arg);
#endif
    }
    
    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() { arg.out.save(); } // defensive measure in case they alias
    void postTune() { arg.out.load(); }

    //DMH: FIXME Re-evaluate these
    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const { return 3 * ((1 + 2 * 6) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; }
  }; // GaugeWFlowW1
  
  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, double epsilon)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, in);
    checkReconstruct(out, in);
    
    if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());

    //Set each step type as an arg parameter, update halos if needed
    instantiate<GaugeWFlowStep>(out, temp, in, epsilon, WFLOW_STEP_W1);
    if (comm_partitioned()) out.exchangeExtendedGhost(out.R(), false);
    instantiate<GaugeWFlowStep>(in, temp, out, epsilon, WFLOW_STEP_W2);
    if (comm_partitioned()) in.exchangeExtendedGhost(out.R(), false);
    instantiate<GaugeWFlowStep>(out, temp, in, epsilon, WFLOW_STEP_VT);
    if (comm_partitioned()) out.exchangeExtendedGhost(out.R(), false);
    
#else
    errorQuda("Gauge tools are not built");
#endif
  }  
}
