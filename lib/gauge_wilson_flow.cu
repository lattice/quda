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
    const QudaGaugeSmearType wflow_type;
    const WFlowStepType step_type;

    unsigned int minThreads() const { return in.LocalVolumeCB(); }
    unsigned int maxSharedBytesPerBlock() const {
      return wflow_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW ? maxDynamicSharedBytesPerBlock() : TunableKernel3D::maxSharedBytesPerBlock();
    }

    unsigned int sharedBytesPerThread() const
    {
      // use SharedMemoryCache if using Symanzik improvement for two Link fields
      return wflow_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW ? 2 * in.Ncolor() * in.Ncolor() * 2 * sizeof(typename mapper<Float>::type) : 0;
    }

  public:
    GaugeWFlowStep(GaugeField &out, GaugeField &temp, const GaugeField &in, const double epsilon, const QudaGaugeSmearType wflow_type, const WFlowStepType step_type) :
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
      case QUDA_GAUGE_SMEAR_WILSON_FLOW: strcat(aux,",computeWFlowStepWilson"); break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: strcat(aux,",computeWFlowStepSymanzik"); break;
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

    template <QudaGaugeSmearType wflow_type, WFlowStepType step_type> using Arg =
      GaugeWFlowArg<Float, nColor, recon, wflow_dim, wflow_type, step_type>;

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      switch (wflow_type) {
      case QUDA_GAUGE_SMEAR_WILSON_FLOW:
        switch (step_type) {
        case WFLOW_STEP_W1:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_W1>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_W2:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_W2>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_VT:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_VT>(out, temp, in, epsilon));
          break;
        }
        break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW:
        tp.set_max_shared_bytes = true;
        switch (step_type) {
        case WFLOW_STEP_W1:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_STEP_W1>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_W2:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_STEP_W2>(out, temp, in, epsilon));
          break;
        case WFLOW_STEP_VT:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_STEP_VT>(out, temp, in, epsilon));
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
      case QUDA_GAUGE_SMEAR_WILSON_FLOW: mat_muls += 4 * (wflow_dim - 1); break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: mat_muls += 28 * (wflow_dim - 1); break;
      default : errorQuda("Unknown Wilson Flow type");
      }
      return mat_muls * mat_flops * threads;
    }

    long long bytes() const
    {
      int links = 0;
      switch (wflow_type) {
      case QUDA_GAUGE_SMEAR_WILSON_FLOW: links = 6; break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: links = 24; break;
      default : errorQuda("Unknown Wilson Flow type");
      }
      auto temp_io = step_type == WFLOW_STEP_W2 ? 2 : step_type == WFLOW_STEP_VT ? 1 : 0;
      return ((1 + (wflow_dim - 1) * links) * in.Bytes() + out.Bytes() + temp_io * temp.Bytes());
    }
  }; // GaugeWFlowStep

  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, const double epsilon, const QudaGaugeSmearType smear_type)
  {
    checkPrecision(out, temp, in);
    checkReconstruct(out, in);
    checkNative(out, in);
    if (temp.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Temporary vector must not use reconstruct");
    if (!(smear_type == QUDA_GAUGE_SMEAR_WILSON_FLOW || smear_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW))
      errorQuda("Gauge smear type %d not supported for flow kernels", smear_type);
    
    // Set each step type as an arg parameter, update halos if needed
    // Step W1
    instantiate<GaugeWFlowStep>(out, temp, in, epsilon, smear_type, WFLOW_STEP_W1);
    out.exchangeExtendedGhost(out.R(), false);

    // Step W2
    instantiate<GaugeWFlowStep>(in, temp, out, epsilon, smear_type, WFLOW_STEP_W2);
    in.exchangeExtendedGhost(in.R(), false);

    // Step Vt
    instantiate<GaugeWFlowStep>(out, temp, in, epsilon, smear_type, WFLOW_STEP_VT);
    out.exchangeExtendedGhost(out.R(), false);
  }

}
