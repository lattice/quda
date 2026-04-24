#pragma once

/**
 * @file cg_ritz_extractor.h
 * @brief Extract approximate Ritz pairs from a CG solution via short TRLM.
 *
 * Uses Option B: short TRLM warm-started from CG solution, no CG modification.
 */

#include <color_spinor_field.h>
#include <dirac_quda.h>
#include <vector>

namespace quda
{

  /**
   * @brief Extract approximate Ritz pairs from a CG solution via short TRLM.
   *
   * Seeds a TRLM eigensolver with the CG solution vector (which is rich
   * in low-mode content) and runs a small number of iterations to extract
   * the best nRitz Ritz pairs. Much cheaper than a full TRLM because
   * the starting vector already approximates the low-mode subspace.
   */
  class CGRitzExtractor {
  public:
    /**
     * @brief Extract Ritz pairs from a CG solution.
     *
     * @param ritzVecs  Output: extracted Ritz eigenvectors (GPU)
     * @param ritzVals  Output: extracted Ritz eigenvalues
     * @param sol       CG solution vector (used as starting vector)
     * @param mat       Normal operator (D_hat^dag D_hat)
     * @param nRitz     Number of Ritz pairs to extract
     * @param nKr       Krylov space size (default: 3*nRitz)
     * @param maxRestarts Maximum restarts for short TRLM (default: 5)
     * @param tol       Convergence tolerance (default: 1e-4, relaxed)
     * @param usePolyAcc Enable Chebyshev polynomial acceleration (default false)
     * @param polyDeg   Chebyshev polynomial degree (default 50)
     * @param aMin      Lower suppression bound for poly-acc (~10x smallest target eval)
     * @param aMax      Upper bound for poly-acc; 0 => auto-estimate via power iteration
     */
    static void extract(std::vector<ColorSpinorField> &ritzVecs, std::vector<Complex> &ritzVals,
                        const ColorSpinorField &sol, const DiracMatrix &mat, int nRitz, int nKr = 0,
                        int maxRestarts = 5, double tol = 1e-4,
                        bool usePolyAcc = false, int polyDeg = 50, double aMin = 0.0, double aMax = 0.0);
  };

} // namespace quda
