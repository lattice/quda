#pragma once

#include <quda_internal.h>

namespace quda
{
  /**
   * Summed contraction of two color-spinor fields. Called from contractFTQuda.
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

  void contractSummed(const ColorSpinorField &x, const ColorSpinorField &y, std::vector<complex_t> &result,
                          QudaContractType cType, const int *const source_position, const int *const mom_mode,
                          const QudaFFTSymmType *const fft_type, const size_t s1, const size_t b1);
  /**
   * @param[in] x       input color spinor
   * @param[in] y       input color spinor
   * @param[out] result pointer to the spinxspin projections per lattice site
   * @param[in] cType   contraction type
   */

  /**
     @param[out] result Per-site contraction in native field storage at the
                        precision of x and y. One nSpin*nSpin complex per site,
                        even parity then odd. Not an IEEE host buffer.
   */
  void contractField(const ColorSpinorField &x, const ColorSpinorField &y, void *result, QudaContractType cType);

  /**
     @brief Copy n complexes from native field storage to an IEEE host buffer.
     Single precision is a device-to-host copy. Double precision converts
     fp32mp2 to IEEE complex<double> when QUDA_FPMP_FLOATFLOAT is enabled.
   */
  void copyInternalComplexToHost(QudaPrecision precision, void *dst, const void *src, size_t n_complex);

  /**
     @brief Contract the quark field x against the 3-d Laplace eigenvector
     set y.  At present, this expects a spin-4 fermion field, and the
     Laplace eigenvector is a spin-1 field.
     @param[out] result array of length 4 * Lt
     @param[in] x Input fermion field set we are contracting
     @param[in] y Input eigenvector field set we are contracting against
   */
  void evecProjectLaplace3D(std::vector<complex_t> &result, cvector_ref<const ColorSpinorField> &x,
                            cvector_ref<const ColorSpinorField> &y);

} // namespace quda
