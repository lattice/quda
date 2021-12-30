#pragma once

#include <quda_internal.h>
#include <quda.h>
#include <complex_quda.h>

#include <vector>

#include <device_vector.h>

namespace quda
{
  namespace madwf_ml
  {

    enum class transfer_5D_t { Wilson = 144, Spin = 16, chiral = 2 };

    template <class transfer_float, transfer_5D_t type>
    void transfer_5d_hh(ColorSpinorField &out, const ColorSpinorField &in, const device_vector<float> &tp,
                        bool dagger);

    template <class transfer_float, transfer_5D_t type>
    void tensor_5d_hh(const ColorSpinorField &x, const ColorSpinorField &y, device_vector<float> &tp);

  } // namespace madwf_ml
} // namespace quda

