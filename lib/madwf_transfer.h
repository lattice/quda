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
    /**
      @brief Perform a transfer in the fifth dimension, i.e. out_s = T_st * in_t, s,t <- fifth dimension indices
      @param out the output field
      @param in the input field
      @param tp The device_vector that contains the transfer matrix
      @param dagger whether or not perform dagger for the transfer matrix
     */
    void transfer_5d_hh(ColorSpinorField &out, const ColorSpinorField &in,
                        const device_vector<MadwfAcc::transfer_float> &tp, bool dagger);

    /**
      @brief Perform a tensor product in the fifth dimension, i.e. T_st = conj(conj(x_s) * y_t), s,t <- fifth dimension indices
      @param x the x field
      @param y the y field
      @param tp The device_vector that contains the output transfer matrix
     */
    void tensor_5d_hh(ColorSpinorField &x, const ColorSpinorField &y, device_vector<MadwfAcc::transfer_float> &tp);

  } // namespace madwf_ml
} // namespace quda
