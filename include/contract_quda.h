#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  //called in interface_quda.cpp, defined in contract.cu
  void contractSpatialQuda(const ColorSpinorField &x, const ColorSpinorField &y, size_t s1, size_t b1, std::vector<Complex> &result, QudaContractType cType, int local_corr_length);
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
} // namespace quda
