#include <gauge_field.h>
#include <instantiate.h>
#include <gauge_tools.h>

namespace quda
{

  template <typename Float, QudaReconstructType recon>
  void applyGaugeHYP(GaugeField &out, GaugeField *tmp[4], const GaugeField &in, real_t alpha, int level, int dir_ignore);

  void HYPStep(GaugeField &out, GaugeField &in, real_t alpha1, real_t alpha2, real_t alpha3, int dir_ignore)
  {
    checkPrecision(out, in);
    checkReconstruct(out, in);
    checkNative(out, in);

    if (dir_ignore < 0 || dir_ignore > 3) { dir_ignore = 4; }

    GaugeFieldParam gParam(out);
    gParam.location = QUDA_CUDA_FIELD_LOCATION;
    gParam.geometry = QUDA_TENSOR_GEOMETRY;

    GaugeField *tmp[4];
    if (dir_ignore == 4) { // hypDim == 4
      for (int i = 0; i < 4; ++i) { tmp[i] = new GaugeField(gParam); }
    } else { // hypDim == 3
      tmp[0] = new GaugeField(gParam);
      // tmp[1], tmp[2] and tmp[3] will not be used for hypDim == 3
      gParam.create = QUDA_REFERENCE_FIELD_CREATE;
      for (int i = 1; i < 4; ++i) { tmp[i] = new GaugeField(gParam); }
    }

    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    if (dir_ignore == 4) {
      copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
      in.exchangeExtendedGhost(in.R(), false);
      instantiatePrecReconNo12(out, [&]<typename Float, QudaReconstructType recon>() {
        applyGaugeHYP<Float, recon>(out, tmp, in, alpha3, 1, dir_ignore);
      });
      tmp[0]->exchangeExtendedGhost(tmp[0]->R(), false);
      tmp[1]->exchangeExtendedGhost(tmp[1]->R(), false);
      instantiatePrecReconNo12(out, [&]<typename Float, QudaReconstructType recon>() {
        applyGaugeHYP<Float, recon>(out, tmp, in, alpha2, 2, dir_ignore);
      });
      tmp[2]->exchangeExtendedGhost(tmp[2]->R(), false);
      tmp[3]->exchangeExtendedGhost(tmp[3]->R(), false);
      instantiatePrecReconNo12(out, [&]<typename Float, QudaReconstructType recon>() {
        applyGaugeHYP<Float, recon>(out, tmp, in, alpha1, 3, dir_ignore);
      });
      out.exchangeExtendedGhost(out.R(), false);
    } else {
      copyExtendedGauge(in, out, QUDA_CUDA_FIELD_LOCATION);
      in.exchangeExtendedGhost(in.R(), false);
      instantiatePrecReconNo12(out, [&]<typename Float, QudaReconstructType recon>() {
        applyGaugeHYP<Float, recon>(out, tmp, in, alpha3, 1, dir_ignore);
      });
      tmp[0]->exchangeExtendedGhost(tmp[0]->R(), false);
      instantiatePrecReconNo12(out, [&]<typename Float, QudaReconstructType recon>() {
        applyGaugeHYP<Float, recon>(out, tmp, in, alpha2, 2, dir_ignore);
      });
      out.exchangeExtendedGhost(out.R(), false);
    }
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);

    for (int i = 0; i < 4; i++) delete tmp[i];
  }

} // namespace quda
