#include <gauge_field.h>
#include <llfat.h>
#include <instantiate.h>
#include <cmath>

#define MIN_COEFF 1e-7

namespace quda
{

  template <typename Float, QudaReconstructType recon>
  void applyLongLink(const GaugeField &u, GaugeField &lng, double coeff);

  template <typename Float, QudaReconstructType recon>
  void applyOneLink(const GaugeField &u, GaugeField &fat, double coeff);

  template <typename Float, QudaReconstructType recon>
  void applyStaple(const GaugeField &u, GaugeField &fat, GaugeField &staple, const GaugeField &mulink, int nu, int dir1,
                   int dir2, double coeff, bool save_staple);

  void computeLongLink(GaugeField &lng, const GaugeField &u, double coeff)
  {
    instantiatePrecReconNo12(u, [&]<typename Float, QudaReconstructType recon>() { // u first so we pick its recon
      applyLongLink<Float, recon>(u, lng, coeff);
    });
  }

  void computeOneLink(GaugeField &fat, const GaugeField &u, double coeff)
  {
    if (u.StaggeredPhase() != QUDA_STAGGERED_PHASE_MILC && u.Reconstruct() != QUDA_RECONSTRUCT_NO)
      errorQuda("Staggered phase type %d not supported", u.StaggeredPhase());
    instantiatePrecReconNo12(
      u, [&]<typename Float, QudaReconstructType recon>() { applyOneLink<Float, recon>(u, fat, coeff); });
  }

  // Compute the staple field for direction nu, excluding the directions dir1 and dir2.
  void computeStaple(GaugeField &fat, GaugeField &staple, const GaugeField &mulink, const GaugeField &u, int nu,
                     int dir1, int dir2, double coeff, bool save_staple)
  {
    instantiatePrecReconNo12(u, [&]<typename Float, QudaReconstructType recon>() {
      applyStaple<Float, recon>(u, fat, staple, mulink, nu, dir1, dir2, coeff, save_staple);
    });
  }

  void longKSLink(GaugeField &lng, const GaugeField &u, const double *coeff)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
      computeLongLink(lng, u, coeff[1]);
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Long-link computation requires staggered operator to be enabled");
    }
  }

  void fatKSLink(GaugeField &fat, const GaugeField &u, const double *coeff)
  {
    if constexpr (is_enabled<QUDA_STAGGERED_DSLASH>()) {
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);

      GaugeFieldParam gParam(u);
      gParam.reconstruct = QUDA_RECONSTRUCT_NO;
      gParam.setPrecision(gParam.Precision());
      gParam.create = QUDA_NULL_FIELD_CREATE;
      GaugeField staple(gParam);
      GaugeField staple1(gParam);

      if (((fat.X()[0] % 2 != 0) || (fat.X()[1] % 2 != 0) || (fat.X()[2] % 2 != 0) || (fat.X()[3] % 2 != 0))
          && (u.Reconstruct() != QUDA_RECONSTRUCT_NO)) {
        errorQuda("Reconstruct %d and odd dimension size is not supported by link fattening code (yet)", u.Reconstruct());
      }

      computeOneLink(fat, u, coeff[0] - 6.0 * coeff[5]);

      // Check the coefficients. If all of the following are zero, return.
      if (fabs(coeff[2]) >= MIN_COEFF || fabs(coeff[3]) >= MIN_COEFF || fabs(coeff[4]) >= MIN_COEFF
          || fabs(coeff[5]) >= MIN_COEFF) {

        for (int nu = 0; nu < 4; nu++) {
          computeStaple(fat, staple, u, u, nu, -1, -1, coeff[2], 1);

          if (coeff[5] != 0.0) computeStaple(fat, staple, staple, u, nu, -1, -1, coeff[5], 0);

          for (int rho = 0; rho < 4; rho++) {
            if (rho != nu) {

              computeStaple(fat, staple1, staple, u, rho, nu, -1, coeff[3], 1);

              if (fabs(coeff[4]) > MIN_COEFF) {
                for (int sig = 0; sig < 4; sig++) {
                  if (sig != nu && sig != rho) { computeStaple(fat, staple, staple1, u, sig, nu, rho, coeff[4], 0); }
                } // sig
              }   // MIN_COEFF
            }
          } // rho
        }   // nu
      }

      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
    } else {
      errorQuda("Fat-link computation requires staggered oprator to be enabled");
    }
  }

} // namespace quda

#undef MIN_COEFF
