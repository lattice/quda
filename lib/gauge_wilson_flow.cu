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
    static constexpr int wflow_dim = 4; // apply flow in all dims
    GaugeWFlowArg<Float, nColor, recon, wflow_dim> arg;
    const GaugeField &meta;

    bool tuneGridDim() const { return false; } // Don't tune the grid dimensions.
    unsigned int minThreads() const { return arg.threads; }

  public:
    GaugeWFlowStep(GaugeField &out, GaugeField &temp, const GaugeField &in, const double epsilon, const QudaWFlowType wflow_type, const WFlowStepType step_type) :
      TunableVectorYZ(2, wflow_dim),
      arg(out, temp, in, epsilon, wflow_type, step_type),
      meta(in)
    {
      strcpy(aux, meta.AuxString());
      strcat(aux, comm_dim_partitioned_string());
      switch (wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON: strcat(aux,",computeWFlowStepWilson"); break;
      case QUDA_WFLOW_TYPE_SYMANZIK: strcat(aux,",computeWFlowStepSymanzik"); break;
      default : errorQuda("Unknown Wilson Flow type %d", wflow_type);
      }
      switch (step_type) {
      case WFLOW_STEP_W1: strcat(aux, "_W1"); break;
      case WFLOW_STEP_W2: strcat(aux, "_W2"); break;
      case WFLOW_STEP_VT: strcat(aux, "_VT"); break;
      default : errorQuda("Unknown Wilson Flow step type %d", step_type);
      }

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
      jitify_error = program->kernel("quda::computeWFlowStep").instantiate(wflow_type,Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      switch (arg.wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON: computeWFlowStep<QUDA_WFLOW_TYPE_WILSON><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
      case QUDA_WFLOW_TYPE_SYMANZIK: computeWFlowStep<QUDA_WFLOW_TYPE_SYMANZIK><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
      default : errorQuda("Unknown Wilson Flow type %d", arg.wflow_type);
      }
#endif
    }

    TuneKey tuneKey() const { return TuneKey(meta.VolString(), typeid(*this).name(), aux); }

    void preTune() {
      arg.out.save(); // defensive measure in case out aliases in
      arg.temp.save();
    }
    void postTune() {
      arg.out.load();
      arg.temp.load();
    }

    //DMH: FIXME Re-evaluate these
    long long flops() const { return 3 * (2 + 2 * 4) * 198ll * arg.threads; } // just counts matrix multiplication
    long long bytes() const {
      int links = 0;
      switch(arg.wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON: links = 6; break;
      case QUDA_WFLOW_TYPE_SYMANZIK: links = 24; break;
      default : errorQuda("Unknown Wilson Flow type");
      }
      return ((1 + wflow_dim * links) * arg.in.Bytes() + arg.out.Bytes()) * arg.threads; }
  }; // GaugeWFlowStep

  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, const double epsilon, const QudaWFlowType wflow_type)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, in);
    checkReconstruct(out, in);

    if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());

    //Set each step type as an arg parameter, update halos if needed
    // Step W1
    instantiate<GaugeWFlowStep>(out, temp, in, epsilon, wflow_type, WFLOW_STEP_W1);
    out.exchangeExtendedGhost(out.R(), false);

    // Step W2
    instantiate<GaugeWFlowStep>(in, temp, out, epsilon, wflow_type, WFLOW_STEP_W2);
    in.exchangeExtendedGhost(in.R(), false);

    // Step Vt
    instantiate<GaugeWFlowStep>(out, temp, in, epsilon, wflow_type, WFLOW_STEP_VT);
    out.exchangeExtendedGhost(out.R(), false);
#else
    errorQuda("Gauge tools are not built");
#endif
  }
}
