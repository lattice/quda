#pragma once

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <coarse_deflation_manager.h>
#include <dirac_quda.h>
#include <quda.h>

namespace quda
{

  /**
   * @brief Cheap "inner" fermion force using coarse-grid deflation
   *        projection plus optional MR smoothing.
   *
   * Algorithm:
   *   restrict → project onto coarse eigenvectors → prolong → [MR smooth]
   *   → fermion force.
   *
   * Used by NestedFGIIntegrator's inner timescale to approximate the full
   * fermion force at low cost.
   */
  class LowModeForce
  {
  private:
    CoarseDeflationManager &deflManager;

    int nMRSmooth;
    double mrOmega;

    const DiracMatrix *matFine;

    /** Workspace fields (allocated once, reused) */
    ColorSpinorField coarseTmp;
    ColorSpinorField coarseSol;
    ColorSpinorField fineSol;
    ColorSpinorField mrResid;
    ColorSpinorField mrAr;
    ColorSpinorField fineSrcBuf;     // fine-grid buffer at Transfer::NullPrecision() for restrict
    ColorSpinorField fineProlongBuf; // fine-grid buffer at Transfer::NullPrecision() for prolong

  public:
    /**
     * @brief Construct the low-mode force calculator.
     * @param deflManager  Coarse deflation manager (provides eigenvectors and transfer)
     * @param matFine      Fine-grid normal operator (for MR smoothing)
     * @param nMRSmooth    Number of MR smoothing iterations (0 = no smoothing)
     * @param mrOmega      MR relaxation parameter
     */
    LowModeForce(CoarseDeflationManager &deflManager, const DiracMatrix &matFine, int nMRSmooth = 0,
                 double mrOmega = 1.0);

    ~LowModeForce() = default;

    /**
     * @brief Project source onto low modes and return the approximate solution.
     * @param xLow  Output: low-mode approximate solution (fine-grid)
     * @param src   Input: source vector (fine-grid, e.g. pseudofermion)
     */
    void projectLowModes(ColorSpinorField &xLow, const ColorSpinorField &src);

    /**
     * @brief Compute low-mode fermion force and accumulate into momentum.
     *
     * Calls projectLowModes, then the existing fermion force kernel
     * (e.g. computeCloverForce) with the projected solution.
     *
     * @param mom        Momentum field (accumulated into)
     * @param src        Source pseudofermion vector
     * @param coeff      Step-size coefficient (dt)
     * @param gauge      Current gauge field
     * @param clover     Clover field (if Wilson-clover)
     * @param gaugeParam Gauge metadata
     * @param invParam   Inverter metadata
     */
    void computeForce(GaugeField &mom, const ColorSpinorField &src, double coeff, GaugeField &gauge,
                      const CloverField *clover, QudaGaugeParam &gaugeParam, QudaInvertParam &invParam);

    /**
     * @brief Rebind the fine-grid normal operator after an MG refresh.
     *
     * updateMultigridQuda(refresh) destroys mg_solver->m and replaces it
     * with a new DiracM, leaving any cached pointer dangling. NestedFGI
     * calls this from its afterAccepted hook so the next computeForce /
     * projectLowModes call uses the fresh operator.
     */
    void rebindFineMatrix(const DiracMatrix &matFine_) { matFine = &matFine_; }
  };

} // namespace quda
