#pragma once

#include <quda_internal.h>
#include <quda.h>
#include <complex_quda.h>

#include <vector>

#include <device_vector.h>
#include <madwf_ml.h>

namespace quda
{
  namespace madwf_ml
  {

    void transfer_5d_hh(ColorSpinorField &out, const ColorSpinorField &in,
                        const device_vector<MadwfAcc::transfer_float> &tp, bool dagger);

    void tensor_5d_hh(ColorSpinorField &x, const ColorSpinorField &y, device_vector<MadwfAcc::transfer_float> &tp);

  } // namespace madwf_ml
} // namespace quda
