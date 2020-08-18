#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  // called in interface_quda.cpp, defined in contract.cu
  void contractSummedQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result,
                          QudaContractType cType, const size_t s1 = 0, const size_t b1 = 0);
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
} // namespace quda
