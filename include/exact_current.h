#pragma once

#include <vector>
#include <color_spinor_field.h>

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

} // namespace quda
