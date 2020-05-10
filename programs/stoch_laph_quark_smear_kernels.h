#pragma once

#include <complex>
#include <quda.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
namespace quda
{
  void spinDiluteQuda(ColorSpinorField &x, const ColorSpinorField &y, const int alpha);
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, const int t, void *result);
} // namespace quda

