#include <ks_qsmear.h>
#include <instantiate.h>

namespace quda
{

  template <typename Float, QudaReconstructType recon>
  void applyComputeTwoLink(GaugeField &twoLink, const GaugeField &link);

  void computeTwoLink(GaugeField &twoLink, const GaugeField &link)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      checkNative(twoLink, link);
      checkLocation(twoLink, link);
      checkPrecision(twoLink, link);
      // FIXME: enable link-12/8 reconstruction
      instantiatePrecReconNone(twoLink, [&]<typename Float, QudaReconstructType recon>() {
        applyComputeTwoLink<Float, recon>(twoLink, link);
      });
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Two-link computation requires staggered operator to be enabled");
    }
  }

} // namespace quda
