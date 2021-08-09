#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <kernels/gauge_wilson_flow.cuh>
#include <instantiate.h>

namespace quda {

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeWFlowStep : TunableKernel3D
  {
    using real = typename mapper<Float>::type;
    static constexpr int wflow_dim = 4; // apply flow in all dims
    GaugeField &out;
    GaugeField &temp;
    const GaugeField &in;
    const real epsilon;
    const QudaWFlowType wflow_type;
    const WFlowStepType step_type;

    bool tuneSharedBytes() const { return false; }
    unsigned int minThreads() const { return in.LocalVolumeCB(); }
    unsigned int maxBlockSize(const TuneParam &) const { return 32; }
    int blockStep() const { return 8; }
    int blockMin() const { return 8; }

  public:
    GaugeWFlowStep(GaugeField &out, GaugeField &temp, const GaugeField &in, const double epsilon, const QudaWFlowType wflow_type, const WFlowStepType step_type) :
      TunableKernel3D(in, 2, wflow_dim),
      out(out),
      temp(temp),
      in(in),
      epsilon(epsilon),
      wflow_type(wflow_type),
      step_type(step_type)
    {
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

      apply(device::get_default_stream());
    }

    template <QudaWFlowType wflow_type, WFlowStepType step_type> using Arg =
      GaugeWFlowArg<Float, nColor, recon, wflow_dim, wflow_type, step_type>;

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      switch (wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON:
        switch (step_type) {
        case WFLOW_STEP_W1:
          launch<WFlow>(tp, stream, Arg<QUDA_WFLOW_TYPE_WILSON, WFLOW_STEP_W1>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_W2:
          launch<WFlow>(tp, stream, Arg<QUDA_WFLOW_TYPE_WILSON, WFLOW_STEP_W2>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_VT:
          launch<WFlow>(tp, stream, Arg<QUDA_WFLOW_TYPE_WILSON, WFLOW_STEP_VT>(out, temp, in, epsilon));
          break;
        }
        break;
      case QUDA_WFLOW_TYPE_SYMANZIK:
        switch (step_type) {
        case WFLOW_STEP_W1:
          launch<WFlow>(tp, stream, Arg<QUDA_WFLOW_TYPE_SYMANZIK, WFLOW_STEP_W1>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_W2:
          launch<WFlow>(tp, stream, Arg<QUDA_WFLOW_TYPE_SYMANZIK, WFLOW_STEP_W2>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_VT:
          launch<WFlow>(tp, stream, Arg<QUDA_WFLOW_TYPE_SYMANZIK, WFLOW_STEP_VT>(out, temp, in, epsilon));
          break;
        }
        break;
      default: errorQuda("Unknown Wilson Flow type %d", wflow_type);
      }
    }

    void preTune() { out.backup(); temp.backup(); }
    void postTune() { out.restore(); temp.restore(); }

    long long flops() const
    {
      // only counts number of mat-muls per thread
      long long threads = in.LocalVolume() * wflow_dim;
      long long mat_flops = nColor * nColor * (8 * nColor - 2);
      long long mat_muls = 1; // 1 comes from Z * conj(U) term
      switch (wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON: mat_muls += 4 * (wflow_dim - 1); break;
      case QUDA_WFLOW_TYPE_SYMANZIK: mat_muls += 28 * (wflow_dim - 1); break;
      default : errorQuda("Unknown Wilson Flow type");
      }
      return mat_muls * mat_flops * threads;
    }

    long long bytes() const
    {
      int links = 0;
      switch (wflow_type) {
      case QUDA_WFLOW_TYPE_WILSON: links = 6; break;
      case QUDA_WFLOW_TYPE_SYMANZIK: links = 24; break;
      default : errorQuda("Unknown Wilson Flow type");
      }
      auto temp_io = step_type == WFLOW_STEP_W2 ? 2 : step_type == WFLOW_STEP_VT ? 1 : 0;
      return ((1 + (wflow_dim - 1) * links) * in.Bytes() + out.Bytes() + temp_io * temp.Bytes());
    }
  }; // GaugeWFlowStep

#ifdef GPU_GAUGE_TOOLS
  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, const double epsilon, const QudaWFlowType wflow_type)
  {
    checkPrecision(out, temp, in);
    checkReconstruct(out, in);
    checkNative(out, in);
    if (temp.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Temporary vector must not use reconstruct");

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
  }
#else
  void WFlowStep(GaugeField &, GaugeField &, GaugeField &, const double, const QudaWFlowType)
  {
    errorQuda("Gauge tools are not built");
  }
#endif

}
