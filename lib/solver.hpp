
#include <quda_internal.h>
#include <invert_quda.h>
#include <blas_quda.h>
#include <cmath>

namespace quda
{

  /**
     @brief Compute power iterations on a Dirac matrix
     @param[in] diracm Dirac matrix used for power iterations
     @param[in] start Starting rhs for power iterations; value preserved unless it aliases tempvec1 or tempvec2
     @param[in,out] tempvec1 Temporary vector used for power iterations
     @param[in,out] tempvec2 Temporary vector used for power iterations
     @param[in] niter Total number of power iteration iterations
     @param[in] normalize_freq Frequency with which intermediate vector gets normalized
     @param[in] args Parameter pack of ColorSpinorFields used as temporary passed to Dirac
     @return Norm of final power iteration result
  */
  template <typename... Args>
  double Solver::performPowerIterations(const DiracMatrix &diracm, const ColorSpinorField &start,
                                        ColorSpinorField &tempvec1, ColorSpinorField &tempvec2, int niter,
                                        int normalize_freq, Args &&...args)
  {
    checkPrecision(tempvec1, tempvec2);
    blas::copy(tempvec1, start); // no-op if fields alias

    // Do niter iterations, normalize every normalize_freq
    for (int i = 0; i < niter; i++) {
      if (normalize_freq > 0 && i % normalize_freq == 0) {
        double tmpnrm = sqrt(blas::norm2(tempvec1));
        blas::ax(1.0 / tmpnrm, tempvec1);
      }
      diracm(tempvec2, tempvec1, args...);
      if (normalize_freq > 0 && i % normalize_freq == 0) {
        logQuda(QUDA_VERBOSE, "Current Rayleigh Quotient step %d is %e\n", i, sqrt(blas::norm2(tempvec2)));
      }
      std::swap(tempvec1, tempvec2);
    }
    // Get Rayleigh quotient
    double tmpnrm = sqrt(blas::norm2(tempvec1));
    blas::ax(1.0 / tmpnrm, tempvec1);
    diracm(tempvec2, tempvec1, args...);
    double lambda_max = sqrt(blas::norm2(tempvec2));
    logQuda(QUDA_VERBOSE, "Power iterations approximate max = %e\n", lambda_max);

    return lambda_max;
  }

  /**
     @brief Generate a Krylov space in a given basis
     @param[in] diracm Dirac matrix used to generate the Krylov space
     @param[out] Ap dirac matrix times the Krylov basis vectors
     @param[in,out] p Krylov basis vectors; assumes p[0] is in place
     @param[in] n_krylov Size of krylov space
     @param[in] basis Basis type
     @param[in] m_map Slope mapping for Chebyshev basis; ignored for power basis
     @param[in] b_map Intercept mapping for Chebyshev basis; ignored for power basis
     @param[in] args Parameter pack of ColorSpinorFields used as temporary passed to Dirac
  */
  template <typename... Args>
  void Solver::computeCAKrylovSpace(const DiracMatrix &diracm, std::vector<ColorSpinorField> &Ap,
                                    std::vector<ColorSpinorField> &p, int n_krylov, QudaCABasis basis, double m_map,
                                    double b_map, Args &&...args)
  {
    // in some cases p or Ap may be larger
    if (static_cast<int>(p.size()) < n_krylov) errorQuda("Invalid p.size() %lu < n_krylov %d", p.size(), n_krylov);
    if (static_cast<int>(Ap.size()) < n_krylov) errorQuda("Invalid Ap.size() %lu < n_krylov %d", Ap.size(), n_krylov);

    if (basis == QUDA_POWER_BASIS) {
      for (int k = 0; k < n_krylov; k++) {
        diracm(Ap[k], p[k], args...);
        if (k < (n_krylov - 1)) blas::copy(p[k + 1], Ap[k]); // no op if fields alias, which is often the case
      }
    } else { // chebyshev basis
      diracm(Ap[0], p[0], args...);

      if (n_krylov > 1) {
        // p_1 = m Ap_0 + b p_0
        blas::axpbyz(m_map, Ap[0], b_map, p[0], p[1]);
        diracm(Ap[1], p[1], args...);

        // Enter recursion relation
        if (n_krylov > 2) {
          // p_k = 2 m A[_{k-1} + 2 b p_{k-1} - p_{k-2}
          for (int k = 2; k < n_krylov; k++) {
            blas::axpbypczw(2. * m_map, Ap[k - 1], 2. * b_map, p[k - 1], -1., p[k - 2], p[k]);
            diracm(Ap[k], p[k], args...);
          }
        }
      }
    }
  }

} // namespace quda
