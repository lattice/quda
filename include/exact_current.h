#pragma once

#include <vector>
#include <color_spinor_field.h>
#include <gauge_field.h>

namespace quda
{

  /**
     @brief Exact staggered vector current, in native field storage.

     Eigenvectors are already wrapped as device-resident parity fields.
     Masses and eigenvalues are already in QUDA's host real type. The
     returned accumulators are single-parity, nSpin = 1, nColor = 1
     fields, one per direction. acc2 is filled only when mass.size() == 3.

     @param[in] src_even Even-parity eigenvectors.
     @param[in] src_odd Odd-parity eigenvectors.
     @param[in] eval Real parts of the even-parity eigenvalues.
     @param[in] mass Staggered masses. Length 1, 2, or 3.
     @param[in,out] invert_param Invert-param metadata. dslash_type is
                    switched to the covariant derivative on the copy
                    passed in from the interface.
     @param[out] acc_even Even-parity current, native storage, length 4.
     @param[out] acc_odd Odd-parity current, native storage, length 4.
     @param[out] acc2_even Second even-parity current when three masses are given.
     @param[out] acc2_odd Second odd-parity current when three masses are given.
   */
  void exactCurrent(std::vector<ColorSpinorField> &src_even, std::vector<ColorSpinorField> &src_odd,
                    const std::vector<real_t> &eval, const std::vector<real_t> &mass, QudaInvertParam &invert_param,
                    std::vector<ColorSpinorField> &acc_even, std::vector<ColorSpinorField> &acc_odd,
                    std::vector<ColorSpinorField> &acc2_even, std::vector<ColorSpinorField> &acc2_odd);

  /**
     @brief Exact low-mode contribution to the sea-quark loop with the
     insertion of a Dslash difference operator,

       sum_n -2 Re < v_n | dD_eo D_oe v_n > / (lambda_n + 4 m^2),

     summed over the whole lattice (all ranks). v_n are the even-parity
     eigenvectors of the massless operator with eigenvalues lambda_n, D is
     the improved staggered Dslash built from (fat, lng), and dD is the
     one built from the difference links (dfat, dlng). Emulates MILC's
     exact_sea_loop() in generic_ks/f_meas_current.c.

     @param[in] evec_even Even-parity eigenvectors (device fields).
     @param[in] eval Real parts of the even-parity eigenvalues.
     @param[in] mass Staggered mass.
     @param[in] fat Fat links of the Dirac operator D.
     @param[in] lng Long links of the Dirac operator D.
     @param[in] dfat Fat links of the difference operator dD.
     @param[in] dlng Long links of the difference operator dD.
     @return The global (all-rank) sum.
   */
  double exactSeaLoop(const std::vector<ColorSpinorField> &evec_even, const std::vector<real_t> &eval, real_t mass,
                      GaugeField &fat, GaugeField &lng, GaugeField &dfat, GaugeField &dlng);

} // namespace quda
