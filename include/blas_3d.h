#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

namespace quda
{

  namespace blas3d
  {

    // Local enum for the 3D copy type
    enum class copyType { COPY_TO_3D, COPY_FROM_3D, SWAP_3D };

    /**
       @brief Extract / insert / swap a timeslice between a 4-d field and a 3-d field
       @param[in] slice Which slice
       @param[in] type Extracting a time slice (COPY_TO_3D) or
       inserting a timeslice (COPY_FROM_3D) or swapping time slices
       (SWAP_3D)
       @param[in,out] x 3-d field
       @param[in,out] y 4-d field
    */
    void copy(int slice, copyType type, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Swap the slice in two given fields
       @param[in] slice The slice we wish to swap in the fields
       @param[in,out] x Field whose slice we wish to swap
       @param[in,out] y Field whose slice we wish to swap
     */
    void swap(int slice, ColorSpinorField &x, ColorSpinorField &y);

    /**
       @brief Compute a set of real-valued inner products <x, y>, where each inner
       product is restricted to a timeslice.
       @param[out] result Vector of spatial inner products
       @param[in] x Left vector field
       @param[in] y Right vector field
     */
    void reDotProduct(std::vector<double> &result, const ColorSpinorField &x, const ColorSpinorField &y);

    /**
       @brief Compute a set of complex-valued inner products <x, y>, where each inner
       product is restricted to a timeslice.
       @param[out] result Vector of spatial inner products
       @param[in] x Left vector field
       @param[in] y Right vector field
     */
    void cDotProduct(std::vector<Complex> &result, const ColorSpinorField &a, const ColorSpinorField &b);

    /**
       @brief Timeslice real-valued scaling of the field
       @param[in] a Vector of scale factors (length = local temporary extent)
       @param[in] x Field we we wish to scale
     */
    void ax(std::vector<double> &a, ColorSpinorField &x);

    /**
       @brief Timeslice real-valued axpby computation
       @param[in] a Vector of scale factors (length = local temporary extent)
       @param[in] x Input field
       @param[in] b Vector of scale factors (length = local temporary extent)
       @param[in,out] y Field we are updating
     */
    void axpby(const std::vector<double> &a, const ColorSpinorField &x, const std::vector<double> &b,
               ColorSpinorField &y);

    /**
       @brief Timeslice complex-valued axpby computation
       @param[in] a Vector of scale factors (length = local temporary extent)
       @param[in] x Input field
       @param[in] b Vector of scale factors (length = local temporary extent)
       @param[in,out] y Field we are updating
     */
    void caxpby(const std::vector<Complex> &a, const ColorSpinorField &x, const std::vector<Complex> &b,
                ColorSpinorField &y);

  } // namespace blas3d
} // namespace quda
