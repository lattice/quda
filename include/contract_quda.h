#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  //called in interface_quda.cpp, defined in contract.cu
  void contractSpatialQuda(const ColorSpinorField &x, const ColorSpinorField &y, size_t s1, size_t b1, void *result, QudaContractType cType);
    void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
} // namespace quda
