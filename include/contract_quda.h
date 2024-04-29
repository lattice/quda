#pragma once

#include <quda_internal.h>

namespace quda
{
  void contractQuda(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);

  /**
     @brief Contract the quark field x against the 3-d Laplace eigenvector
     set y.  At present, this expects a spin-4 fermion field, and the
     Laplace eigenvector is a spin-1 field.
     @param[out] result array of length 4 * Lt
     @param[in] x Input fermion field set we are contracting
     @param[in] y Input eigenvector field set we are contracting against
   */
  void evecProjectLaplace3D(std::vector<Complex> &result, cvector_ref<const ColorSpinorField> &x,
                            cvector_ref<const ColorSpinorField> &y);

} // namespace quda
