#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda {

  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result);
  
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
} // namespace quda
