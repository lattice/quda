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

    bool tuneSharedBytes() const { return false; }
    bool tuneGridDim() const { return false; }
    unsigned int minThreads() const { return arg.threads; }
    unsigned int maxBlockSize(const TuneParam &param) const { return 32; }
    int blockStep() const { return 8; }
    int blockMin() const { return 8; }

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

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
#ifdef JITIFY
      using namespace jitify::reflection;
      jitify_error = program->kernel("quda::computeWFlowStep").instantiate(arg.wflow_type,arg.step_type,Type<decltype(arg)>())
        .configure(tp.grid, tp.block, tp.shared_bytes, stream).launch(arg);
#else
      switch (arg.wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON:
        switch (arg.step_type) {
        case WFLOW_STEP_W1: computeWFlowStep<QUDA_WFLOW_TYPE_WILSON, WFLOW_STEP_W1><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        case WFLOW_STEP_W2: computeWFlowStep<QUDA_WFLOW_TYPE_WILSON, WFLOW_STEP_W2><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        case WFLOW_STEP_VT: computeWFlowStep<QUDA_WFLOW_TYPE_WILSON, WFLOW_STEP_VT><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        }
        break;
      case QUDA_WFLOW_TYPE_SYMANZIK:
        switch (arg.step_type) {
        case WFLOW_STEP_W1: computeWFlowStep<QUDA_WFLOW_TYPE_SYMANZIK, WFLOW_STEP_W1><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        case WFLOW_STEP_W2: computeWFlowStep<QUDA_WFLOW_TYPE_SYMANZIK, WFLOW_STEP_W2><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        case WFLOW_STEP_VT: computeWFlowStep<QUDA_WFLOW_TYPE_SYMANZIK, WFLOW_STEP_VT><<<tp.grid, tp.block, tp.shared_bytes>>>(arg); break;
        }
        break;
      default: errorQuda("Unknown Wilson Flow type %d", arg.wflow_type);
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

    long long flops() const
    {
      // only counts number of mat-muls per thread
      long long threads = 2ll * arg.threads * wflow_dim;
      long long mat_flops = arg.nColor * arg.nColor * (8 * arg.nColor - 2);
      long long mat_muls = 1; // 1 comes from Z * conj(U) term
      switch(arg.wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON: mat_muls += 4 * (wflow_dim - 1); break;
      case QUDA_WFLOW_TYPE_SYMANZIK: mat_muls += 28 * (wflow_dim - 1); break;
      default : errorQuda("Unknown Wilson Flow type");
      }
      return mat_muls * mat_flops * threads;
    }

    long long bytes() const
    {
      int links = 0;
      switch(arg.wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON: links = 6; break;
      case QUDA_WFLOW_TYPE_SYMANZIK: links = 24; break;
      default : errorQuda("Unknown Wilson Flow type");
      }
      auto temp_io = arg.step_type == WFLOW_STEP_W2 ? 2 : arg.step_type == WFLOW_STEP_VT ? 1 : 0;
      return ((1 + (wflow_dim-1) * links) * arg.in.Bytes() + arg.out.Bytes() + temp_io*arg.temp.Bytes()) * 2ll * arg.threads * wflow_dim;
    }
  }; // GaugeWFlowStep

  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, const double epsilon, const QudaWFlowType wflow_type)
  {
#ifdef GPU_GAUGE_TOOLS
    checkPrecision(out, temp, in);
    checkReconstruct(out, in);
    if (temp.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Temporary vector must not use reconstruct");
    if (!out.isNative()) errorQuda("Order %d with %d reconstruct not supported", in.Order(), in.Reconstruct());
    if (!in.isNative()) errorQuda("Order %d with %d reconstruct not supported", out.Order(), out.Reconstruct());

    // Set each step type as an arg parameter, update halos if needed
    // Step W1
    instantiate<GaugeWFlowStep,WilsonReconstruct>(out, temp, in, epsilon, wflow_type, WFLOW_STEP_W1);
    out.exchangeExtendedGhost(out.R(), false);

    // Step W2
    instantiate<GaugeWFlowStep,WilsonReconstruct>(in, temp, out, epsilon, wflow_type, WFLOW_STEP_W2);
    in.exchangeExtendedGhost(in.R(), false);

    // Step Vt
    instantiate<GaugeWFlowStep,WilsonReconstruct>(out, temp, in, epsilon, wflow_type, WFLOW_STEP_VT);
    out.exchangeExtendedGhost(out.R(), false);
#else
    errorQuda("Gauge tools are not built");
#endif
  }
}
