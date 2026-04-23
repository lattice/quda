#pragma once

#include <vector>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <eigensolve_quda.h>
#include <coarse_deflation_manager.h>
#include <multigrid.h>
#include <transfer.h>
#include <clover_field.h>
#include <dirac_quda.h>
#include <quda.h>

namespace quda
{

  /**
   * @brief Computes the cheap "inner" fermion force using coarse-grid deflation
   *        projection plus optional MR smoothing.
   *
   * Algorithm: restrict -> project onto coarse eigenvectors -> prolong -> [MR smooth] -> fermion force
   */
  class LowModeForce {
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
    ColorSpinorField fineSrcBuf;      // fine-grid buffer at Transfer::NullPrecision() for restrict
    ColorSpinorField fineProlongBuf;  // fine-grid buffer at Transfer::NullPrecision() for prolong

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
  };

  /**
   * @brief Nested force-gradient integrator with coarse-grid deflation.
   *
   * Outer integrator: PQPQP_FGI (4th order), coefficients lambda=1/6, xi=1/72.
   * Inner integrator: Leapfrog on cheap force (low-mode deflation + gauge force).
   * Force-gradient step: Hessian-free via gauge save/restore using GaugeBundleBackup.
   */
  class NestedFGIIntegrator {
  private:
    CoarseDeflationManager deflManager;
    LowModeForce lowModeForce;

    double lambda;
    double xi;
    double beta;
    int nInnerSteps;
    QudaIntegratorType innerIntegrator;
    double innerOmelyanLambda;

    /** External references for gauge/momentum/clover management */
    GaugeField *&gaugePrecise;
    GaugeField *&gaugeSloppy;
    GaugeField *&gaugePrecondition;
    GaugeField *&gaugeRefinement;
    GaugeField *&gaugeEigensolver;
    GaugeField *&gaugeExtended;
    GaugeField &momResident;
    CloverField *&cloverPrecise;
    GaugeField *&extendedGaugeResident;

    /** MG preconditioner for outer (expensive) force */
    void *mgPreconditioner;

    /** Parameter structs */
    QudaGaugeParam &gaugeParam;
    QudaInvertParam &invParam;

    /** Wilson plaquette gauge force paths */
    int **gaugePaths[4];
    int gaugePathLength[6];
    double gaugePathCoeff[6];

    /** @brief Set up Wilson plaquette gauge force paths */
    void setupGaugePaths();

    /** @brief Free gauge force paths */
    void freeGaugePaths();

    /** @brief Perform gauge drift + clover rebuild */
    void gaugeStep(double dt);

    /** @brief Compute the expensive outer force: full MG-prec solve minus low-mode */
    void computeOuterForce(double coeff, const ColorSpinorField &phi);

    /** @brief Compute the cheap inner force: low-mode + gauge */
    void computeInnerForce(double coeff, const ColorSpinorField &phi);

    /** @brief Perform inner leapfrog sub-integration for duration dt */
    void innerLeapfrog(double dt, const ColorSpinorField &phi);

    /** @brief Perform inner Omelyan (PQPQP) sub-integration for duration dt */
    void innerOmelyan(double dt, const ColorSpinorField &phi);

    /** @brief Hessian-free force-gradient sub-step */
    void forceGradientStep(double h, const ColorSpinorField &phi);

  public:
    /**
     * @brief Construct the nested FGI integrator.
     * @param hmcParam       HMC parameters (integrator config, deflation config, etc.)
     * @param mg             MG hierarchy (provides Transfer + coarse operator)
     * @param matFine        Fine-grid normal operator for MR smoothing in LowModeForce
     * @param mgPrec         MG preconditioner instance (for outer force CG solves)
     * @param gaugeParam     Gauge metadata
     * @param invParam       Inverter parameters
     * @param gaugePrecise   Reference to global precise gauge pointer
     * @param gaugeSloppy    Reference to global sloppy gauge pointer
     * @param gaugePrecondition Reference to global precondition gauge pointer
     * @param gaugeRefinement   Reference to global refinement gauge pointer
     * @param gaugeEigensolver  Reference to global eigensolver gauge pointer
     * @param gaugeExtended     Reference to global extended gauge pointer
     * @param momResident       Reference to global resident momentum
     */
    NestedFGIIntegrator(const QudaHMCParam &hmcParam, MG &mg, const DiracMatrix &matFine, void *mgPrec,
                        QudaGaugeParam &gaugeParam, QudaInvertParam &invParam, GaugeField *&gaugePrecise,
                        GaugeField *&gaugeSloppy, GaugeField *&gaugePrecondition, GaugeField *&gaugeRefinement,
                        GaugeField *&gaugeEigensolver, GaugeField *&gaugeExtended, GaugeField &momResident,
                        CloverField *&cloverPrecise, GaugeField *&extendedGaugeResident);

    ~NestedFGIIntegrator();

    /**
     * @brief Execute one full MD trajectory with the nested FGI scheme.
     * @param tau     Trajectory length
     * @param nSteps  Number of outer integration steps
     * @param phi     Pseudofermion field
     */
    void trajectory(double tau, int nSteps, const ColorSpinorField &phi);

    /** @brief Access the deflation manager (e.g. for inter-trajectory RR update) */
    CoarseDeflationManager &getDeflationManager() { return deflManager; }
  };

} // namespace quda
