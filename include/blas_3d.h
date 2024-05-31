#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

namespace quda
{

  namespace blas3d
  {

    // Local enum for the 3D copy type
    enum copy3dType { COPY_TO_3D, COPY_FROM_3D };
    void copy(const int slice, const copy3dType type, ColorSpinorField &x, ColorSpinorField &y);

    // Reductions
    void reDotProduct(std::vector<double> &result, const ColorSpinorField &a, const ColorSpinorField &b);
    void cDotProduct(std::vector<Complex> &result, const ColorSpinorField &a, const ColorSpinorField &b);

    // (c)axpby
    void axpby(std::vector<double> &a, ColorSpinorField &x, std::vector<double> &b, ColorSpinorField &y);
    void caxpby(std::vector<Complex> &a, ColorSpinorField &x, std::vector<Complex> &b, ColorSpinorField &y);
  } // namespace blas3d
} // namespace quda
