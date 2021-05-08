#pragma once

#include <quda_internal.h>
#include <color_spinor_field.h>

namespace quda {

  namespace blas3d {

    void copy(const int dim, const int slice, ColorSpinorField &x, const ColorSpinorField &y);
    void reDotProduct(const int dim, std::vector<double> &result, const ColorSpinorField &a, const ColorSpinorField &b);
    void cDotProduct(const int dim, std::vector<Complex> &result, const ColorSpinorField &a, const ColorSpinorField &b);
    
    void axpby(const int dim, std::vector<double> &a, ColorSpinorField &x, std::vector<double> &b, ColorSpinorField &y);
    void caxpby(const int dim, std::vector<Complex> &a, ColorSpinorField &x, std::vector<Complex> &b, ColorSpinorField &y);
  }
}
