#include <gauge_field.h>
#include <instantiate.h>
#include <gauge_tools.h>

namespace quda
{

  template <typename Float, QudaReconstructType recon>
  void applyGaugeWFlowStep(GaugeField &out, GaugeField &temp, const GaugeField &in, real_t epsilon, real_t anisotropy,
                           QudaGaugeSmearType wflow_type, QudaWFlowStepType step_type);

  void applyGaugeWFlowStep(GaugeField &out, GaugeField &temp, const GaugeField &in, real_t epsilon, real_t anisotropy,
                           QudaGaugeSmearType wflow_type, QudaWFlowStepType step_type)
  {
    instantiatePrecReconNo12(out, [&]<typename Float, QudaReconstructType recon>() {
      applyGaugeWFlowStep<Float, recon>(out, temp, in, epsilon, anisotropy, wflow_type, step_type);
    });
  }

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
      applyGaugeWFlowStep(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_STEP_W1);
      out.exchangeExtendedGhost(out.R(), false);

      applyGaugeWFlowStep(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_STEP_W2);
      in.exchangeExtendedGhost(in.R(), false);

      applyGaugeWFlowStep(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_STEP_VT);
      out.exchangeExtendedGhost(out.R(), false);
      break;
    case 4: // Use 6-stage fourth-order Runga-Kutta integration
      applyGaugeWFlowStep(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_1);
      out.exchangeExtendedGhost(out.R(), false);

      applyGaugeWFlowStep(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_2);
      in.exchangeExtendedGhost(in.R(), false);

      applyGaugeWFlowStep(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_3);
      out.exchangeExtendedGhost(out.R(), false);

      applyGaugeWFlowStep(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_4);
      in.exchangeExtendedGhost(in.R(), false);

      applyGaugeWFlowStep(out, temp, in, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_5);
      out.exchangeExtendedGhost(out.R(), false);

      applyGaugeWFlowStep(in, temp, out, epsilon, smear_anisotropy, smear_type, WFLOW_FOURTH_ORDER_STEP_6);
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

    applyGaugeWFlowStep(out, temp, in, epsilon, static_cast<real_t>(1.0), smear_type, step_type);
    out.exchangeExtendedGhost(out.R(), false);
  }

} // namespace quda
