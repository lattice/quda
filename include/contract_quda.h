#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda
{
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, const int t, void *result);
  
  void laphSinkProject(void *host_quark, void *host_evec, void *host_sinks,
		       QudaInvertParam inv_param, const int X[4], int t_size);
  
} // namespace quda
