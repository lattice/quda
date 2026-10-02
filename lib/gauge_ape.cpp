#include <gauge_field.h>
#include <instantiate.h>
#include <gauge_tools.h>

namespace quda
{

  template <typename Float, QudaReconstructType recon>
  void applyGaugeAPE(GaugeField &out, const GaugeField &in, real_t alpha, int dir_ignore, real_t anisotropy);

  void APEStep(GaugeField &out, GaugeField &in, real_t alpha, int dir_ignore, real_t smear_anisotropy)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
    in.exchangeExtendedGhost(in.R(), false);
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecReconNo12(out, [&]<typename Float, QudaReconstructType recon>() {
      applyGaugeAPE<Float, recon>(out, in, alpha, dir_ignore, smear_anisotropy);
    });
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    out.exchangeExtendedGhost(out.R(), false);
  }

} // namespace quda
