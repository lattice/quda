#pragma once

#include <quda_internal.h>

namespace quda
{
   /**
   * Interface function that launch contraction compute kernels,
   * used in interface_quda.cpp 
   * @param[in] x               input source field
   * @param[in] y               input source field
   * @param[out] result         container of complex contraction results for
   *                            all decay slices and spins
   * @param[in] cType           contraction types as defined in QudaContractType enum
   * @param[in] source_position 4d array of source position
   * @param[in] mom_mode        4d array of momentum
   * @param[in] fft_type        Fourier phase factor type 
   *                            as defined in QudaFFTSymmType enum 
   * @param[in] s1              spin component index (0 for staggered)
   * @param[in] b1              spin component index (0 for staggered)
   */

  void contractSummedQuda(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<Complex> &result,
                          QudaContractType cType, const int *const source_position,
			  const int *const mom_mode, const QudaFFTSymmType *const fft_type,
			  const size_t s1, const size_t b1);
  /**
   * @param[in] x       input color spinor
   * @param[in] y       input color spinor
   * @param[out] result pointer to the spinxspin projections per lattice site
   * @param[in] cType   contraction type
   */

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
