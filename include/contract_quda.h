#pragma once

#include <quda_internal.h>
#include <quda.h>

namespace quda {
  
  void evecProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);
  
  void colorContractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);

  void innerProductQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result);
  
  void momentumProjectQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *cc_array,
			   std::vector<Complex> &mom_proj, std::vector<int> &momenta, const int n_mom);
  
  void colorCrossQuda(const ColorSpinorField &x, const ColorSpinorField &y, ColorSpinorField &result);
  
  void contractSummedQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result,
                          QudaContractType cType, const int *const source_position, const int *const mom_mode, const size_t s1, const size_t b1);
  
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);
} // namespace quda
