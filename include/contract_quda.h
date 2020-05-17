#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);  
  void projectReduceQuda(void *result, void *contractions, const int t_dim_size, const QudaPrecision prec);
} // namespace quda
