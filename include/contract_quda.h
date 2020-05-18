#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);  
  void evecProjectSumQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::complex<double> *result);
} // namespace quda
