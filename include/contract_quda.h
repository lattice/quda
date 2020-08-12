#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  //called in interface_quda.cpp
  void contractQuda(std::vector<ColorSpinorField *> h_prop_array_flavor_1, std::vector<ColorSpinorField *> h_prop_array_flavor_2, void *h_result, QudaContractType cType,
                    QudaInvertParam *param, ColorSpinorParam *cs_param, const int *X);

  //called in contract.cu
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, size_t s1, size_t c1, size_t s2, size_t c2, void *result, QudaContractType cType);
} // namespace quda
