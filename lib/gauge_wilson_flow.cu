#include <quda_internal.h>
#include <gauge_field.h>
#include <tunable_nd.h>
#include <kernels/gauge_wilson_flow.cuh>
#include <instantiate.h>

namespace quda
{

  template <typename Float, int nColor, QudaReconstructType recon> class GaugeWFlowStep : TunableKernel3D
  {
    static constexpr int wflow_dim = 4; // apply flow in all dims
    GaugeField &out;
    GaugeField &temp;
    const GaugeField &in;
    const real_t epsilon;
    const real_t anisotropy;
    const QudaGaugeSmearType wflow_type;
    const QudaWFlowStepType step_type;

    unsigned int minThreads() const { return in.LocalVolumeCB(); }
    unsigned int maxSharedBytesPerBlock() const
    {
      return wflow_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW ? maxDynamicSharedBytesPerBlock() :
                                                            TunableKernel3D::maxSharedBytesPerBlock();
    }

    unsigned int sharedBytesPerThread() const
    {
      // use ThreadLocalCache if using Symanzik improvement for two Link fields
      return wflow_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW ?
        2 * in.Ncolor() * in.Ncolor() * 2 * sizeof(typename mapper<Float>::type) :
        0;
    }

  public:
    GaugeWFlowStep(GaugeField &out, GaugeField &temp, const GaugeField &in, real_t eps, real_t aniso,
                   QudaGaugeSmearType wflow_type, QudaWFlowStepType step_type) :
      TunableKernel3D(in, 2, wflow_dim),
      out(out),
      temp(temp),
      in(in),
      epsilon(eps),
      anisotropy(aniso),
      wflow_type(wflow_type),
      step_type(step_type)
    {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      strcat(aux, comm_dim_partitioned_string());
      switch (wflow_type) {
      case QUDA_GAUGE_SMEAR_WILSON_FLOW: strcat(aux, ",computeWFlowStepWilson"); break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: strcat(aux, ",computeWFlowStepSymanzik"); break;
      default: errorQuda("Unknown Wilson Flow type %d", wflow_type);
      }
      switch (step_type) {
      case WFLOW_STEP_W1: strcat(aux, "_W1"); break;
      case WFLOW_STEP_W2: strcat(aux, "_W2"); break;
      case WFLOW_STEP_VT: strcat(aux, "_VT"); break;
      case WFLOW_FOURTH_ORDER_STEP_1: strcat(aux, "_F1"); break;
      case WFLOW_FOURTH_ORDER_STEP_2: strcat(aux, "_F2"); break;
      case WFLOW_FOURTH_ORDER_STEP_3: strcat(aux, "_F3"); break;
      case WFLOW_FOURTH_ORDER_STEP_4: strcat(aux, "_F4"); break;
      case WFLOW_FOURTH_ORDER_STEP_5: strcat(aux, "_F5"); break;
      case WFLOW_FOURTH_ORDER_STEP_6: strcat(aux, "_F6"); break;
      default: errorQuda("Unknown Wilson Flow step type %d", step_type);
      }

      apply(device::get_default_stream());
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    }

    template <QudaGaugeSmearType wflow_type, QudaWFlowStepType step_type>
    using Arg = GaugeWFlowArg<Float, nColor, recon, wflow_dim, wflow_type, step_type>;

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());

      switch (wflow_type) {
      case QUDA_GAUGE_SMEAR_WILSON_FLOW:
        switch (step_type) {
        case WFLOW_STEP_W1:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_W1>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_STEP_W2:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_W2>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_STEP_VT:
          launch<WFlow>(tp, stream, Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_STEP_VT>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_1:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_FOURTH_ORDER_STEP_1>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_2:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_FOURTH_ORDER_STEP_2>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_3:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_FOURTH_ORDER_STEP_3>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_4:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_FOURTH_ORDER_STEP_4>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_5:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_FOURTH_ORDER_STEP_5>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_6:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_WILSON_FLOW, WFLOW_FOURTH_ORDER_STEP_6>(out, temp, in, epsilon, anisotropy));
          break;
        }
        break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW:
        tp.set_max_shared_bytes = true;
        switch (step_type) {
        case WFLOW_STEP_W1:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_STEP_W1>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_STEP_W2:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_STEP_W2>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_STEP_VT:
          launch<WFlow>(tp, stream,
                        Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_STEP_VT>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_1:
          launch<WFlow>(
            tp, stream,
            Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_FOURTH_ORDER_STEP_1>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_2:
          launch<WFlow>(
            tp, stream,
            Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_FOURTH_ORDER_STEP_2>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_3:
          launch<WFlow>(
            tp, stream,
            Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_FOURTH_ORDER_STEP_3>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_4:
          launch<WFlow>(
            tp, stream,
            Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_FOURTH_ORDER_STEP_4>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_5:
          launch<WFlow>(
            tp, stream,
            Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_FOURTH_ORDER_STEP_5>(out, temp, in, epsilon, anisotropy));
          break;
        case WFLOW_FOURTH_ORDER_STEP_6:
          launch<WFlow>(
            tp, stream,
            Arg<QUDA_GAUGE_SMEAR_SYMANZIK_FLOW, WFLOW_FOURTH_ORDER_STEP_6>(out, temp, in, epsilon, anisotropy));
          break;
        }
        break;
      default: errorQuda("Unknown Wilson Flow type %d", wflow_type);
      }
    }

    void preTune()
    {
      out.backup();
      temp.backup();
    }
    void postTune()
    {
      out.restore();
      temp.restore();
    }

    long long flops() const
    {
      // only counts number of mat-muls per thread
      long long threads = in.LocalVolume() * wflow_dim;
      long long mat_flops = nColor * nColor * (8 * nColor - 2);
      long long mat_muls = 4; // 1 from Z * conj(U) term, 2 in exponentiate_iQ(Z), and 1 from exponentiate_iQ(Z) * U
      switch (wflow_type) {   // Add mat-muls coming from staple calculation
      case QUDA_GAUGE_SMEAR_WILSON_FLOW: mat_muls += 4 * (wflow_dim - 1); break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: mat_muls += 28 * (wflow_dim - 1); break;
      default: errorQuda("Unknown Wilson Flow type");
      }
      return mat_muls * mat_flops * threads;
    }

    long long bytes() const
    {
      int links = 0;
      switch (wflow_type) {
      case QUDA_GAUGE_SMEAR_WILSON_FLOW: links = 6; break;
      case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: links = 24; break;
      default: errorQuda("Unknown Wilson Flow type");
      }
      // Leon: I am not certain that the byte counting is correct here!
      // First and last steps have 1 store (retrieve) to (from) temp
      auto temp_io = 1;
      // Middle steps have an additional store or retrieve to or from temp
      if (step_type == WFLOW_STEP_W2 || step_type == WFLOW_FOURTH_ORDER_STEP_2 || step_type == WFLOW_FOURTH_ORDER_STEP_3
          || step_type == WFLOW_FOURTH_ORDER_STEP_4 || step_type == WFLOW_FOURTH_ORDER_STEP_5)
        temp_io += 1;
      return ((1 + (wflow_dim - 1) * links) * in.Bytes() + out.Bytes() + temp_io * temp.Bytes());
    }
  }; // GaugeWFlowStep

  void WFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, real_t epsilon, QudaGaugeSmearType smear_type,
                 real_t smear_anisotropy, int rk_order)
  {
    checkPrecision(out, temp, in);
    checkReconstruct(out, in);
    checkNative(out, in);
    if (temp.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Temporary vector must not use reconstruct");
    if (!(smear_type == QUDA_GAUGE_SMEAR_WILSON_FLOW || smear_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW))
      errorQuda("Gauge smear type %d not supported for flow kernels", smear_type);

    // Set each step type as an arg parameter, update halos if needed
    switch (rk_order) {
    case 3: // Use 3-stage third-order Runga-Kutta integration
      // Step W1
      instantiate<GaugeWFlowStep>(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_STEP_W1);
      out.exchangeExtendedGhost(out.R(), false);

      // Step W2
      instantiate<GaugeWFlowStep>(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_STEP_W2);
      in.exchangeExtendedGhost(in.R(), false);

      // Step Vt
      instantiate<GaugeWFlowStep>(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_STEP_VT);
      out.exchangeExtendedGhost(out.R(), false);
      break;
    case 4: // Use 6-stage fourth-order Runga-Kutta integration
      instantiate<GaugeWFlowStep>(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_1);
      out.exchangeExtendedGhost(out.R(), false);

      instantiate<GaugeWFlowStep>(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_2);
      in.exchangeExtendedGhost(in.R(), false);

      instantiate<GaugeWFlowStep>(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_3);
      out.exchangeExtendedGhost(out.R(), false);

      instantiate<GaugeWFlowStep>(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_4);
      in.exchangeExtendedGhost(in.R(), false);

      instantiate<GaugeWFlowStep>(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_5);
      out.exchangeExtendedGhost(out.R(), false);

      instantiate<GaugeWFlowStep>(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_6);
      in.exchangeExtendedGhost(in.R(), false);

      out = in;
      break;
    default: errorQuda("Unsupported Runga-Kutta order %d", rk_order);
    }
  }

  void GFlowStep(GaugeField &out, GaugeField &temp, GaugeField &in, real_t epsilon, QudaGaugeSmearType smear_type,
                 QudaWFlowStepType step_type)
  {
    checkPrecision(out, temp, in);
    checkReconstruct(out, in);
    checkNative(out, in);
    if (temp.Reconstruct() != QUDA_RECONSTRUCT_NO) errorQuda("Temporary vector must not use reconstruct");
    if (!(smear_type == QUDA_GAUGE_SMEAR_WILSON_FLOW || smear_type == QUDA_GAUGE_SMEAR_SYMANZIK_FLOW))
      errorQuda("Gauge smear type %d not supported for flow kernels", smear_type);

    instantiate<GaugeWFlowStep>(out, temp, in, epsilon, static_cast<real_t>(1.0), smear_type, step_type);
    out.exchangeExtendedGhost(out.R(), false);
  }
} // namespace quda
