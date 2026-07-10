#pragma once

#include <quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <coarse_deflation_manager.h>
#include <dirac_quda.h>
#include <low_mode_force.h>
#include <multigrid.h>
#include <transfer.h>

namespace quda
{

  // Forward declarations. EigenTrackingState is only held by pointer in the
  // base; the full definition is only pulled in by translation units that
  // actually dereference it.
  class EigenTrackingState;

  /**
   * @brief Abstract molecular-dynamics integrator for HMC.
   *
   * Derived classes implement a specific symplectic schedule (leapfrog,
   * Omelyan, FGI, nested-FGI) over shared momentum-kick / gauge-drift
   * primitives. The base class owns the scheduling-invariant bookkeeping
   * (parameter refs, eigentracking hook, helper methods) so that derived
   * class bodies reduce to schedule expressions.
   *
   * Mirrors the quda::Solver / quda::EigenSolver pattern: pure-virtual
   * operator(), static create() factory returning a raw new (caller owns
   * lifetime), enum-based type introspection.
   */
  class Integrator
  {
  public:
    /**
     * @brief Advance (U, π) over a trajectory of total time @p tau using
     *        @p nSteps outer steps, given the pseudofermion @p phi.
     */
    virtual void operator()(double tau, int nSteps, ColorSpinorField &phi) = 0;

    /**
     * @brief Factory. Dispatches on hmc_param.integrator. Raw new; caller
     *        owns the returned pointer.
     *
     * @param hmc_param    HMC configuration (integrator type, lambda, xi, ...)
     * @param gauge_param  Gauge parameters (used by gauge-force kernels)
     * @param inv_param    Inverter parameters (used by fermion-force CG)
     * @param tracking     Optional eigentracking state; threaded into every
     *                     derived integrator uniformly
     * @param mg_instance  Required for QUDA_NESTED_FGI_INTEGRATOR; ignored
     *                     otherwise
     */
    static Integrator *create(QudaHMCParam &hmc_param, QudaGaugeParam &gauge_param, QudaInvertParam &inv_param,
                              EigenTrackingState *tracking = nullptr, void *mg_instance = nullptr);

    /**
     * @brief Which integrator this instance implements (for caching and
     *        tests).
     */
    virtual QudaIntegratorType getIntegratorType() const = 0;

    /**
     * @brief Hook invoked by hmcRunQuda after an accepted trajectory.
     *
     * Default no-op. NestedFGIIntegrator overrides to refresh its
     * CoarseDeflationManager based on whether the MG hierarchy underwent a
     * full re-setup or a thin update.
     *
     * @param full_mg_update true  → MG null-space rebuilt; caller should
     *                               fully re-solve the coarse eigenproblem.
     *                       false → MG thin update only; a Rayleigh–Ritz
     *                               evolve of the existing coarse
     *                               eigenvectors is sufficient.
     */
    virtual void afterAccepted(bool full_mg_update) { (void)full_mg_update; }

    virtual ~Integrator() = default;

  protected:
    // Stored by reference: the integrator cache in getOrCreateIntegrator
    // invalidates on hmc_param/inv_param address change, so the references
    // here stay valid for the lifetime of the cached instance (one
    // hmcRunQuda invocation, where the C-API caller's params live).
    QudaHMCParam &hmc_param;
    QudaGaugeParam &gauge_param;
    QudaInvertParam &inv_param;
    EigenTrackingState *tracking;

    Integrator(QudaHMCParam &hmc_param_, QudaGaugeParam &gauge_param_, QudaInvertParam &inv_param_,
               EigenTrackingState *tracking_) :
      hmc_param(hmc_param_), gauge_param(gauge_param_), inv_param(inv_param_), tracking(tracking_)
    {
    }

    /**
     * @brief P-step: π += dt × F(U, φ), where F is the total gauge + fermion
     *        force. Updates eigentracking state if enabled.
     */
    void kick(double dt, ColorSpinorField &phi);

    /**
     * @brief Q-step: U ← exp(i·dt·π) U, plus sloppy/precondition copies and
     *        clover rebuild where needed.
     */
    void drift(double dt);

    /**
     * @brief Force-gradient sub-step used by FGI and (internally) nested-FGI.
     *
     * Saves gauge and momentum; zeros momentum; kicks with a small
     * coefficient derived from @p h (FGI's ξ·h³ term); displaces gauge with
     * the resulting δπ; restores momentum; full kick at (1-2λ)·h; restores
     * gauge.
     */
    void fgStep(double h, ColorSpinorField &phi);
  };

  /**
   * @brief 2nd-order leapfrog (PQPQ…QP).
   */
  class LeapfrogIntegrator : public Integrator
  {
  public:
    using Integrator::Integrator;
    void operator()(double tau, int nSteps, ColorSpinorField &phi) override;
    QudaIntegratorType getIntegratorType() const override { return QUDA_LEAPFROG_INTEGRATOR; }
  };

  /**
   * @brief 2nd-order Omelyan (2MN): P(λh) [ Q(h/2) P((1-2λ)h) Q(h/2) P(2λh) ]^{n-1}
   *        Q(h/2) P((1-2λ)h) Q(h/2) P(λh).
   *
   * λ is the minimum-norm coefficient 0.1931833275037836, matching the
   * value used by the free-function predecessor.
   */
  class OmelyanIntegrator : public Integrator
  {
  public:
    using Integrator::Integrator;
    void operator()(double tau, int nSteps, ColorSpinorField &phi) override;
    QudaIntegratorType getIntegratorType() const override { return QUDA_OMELYAN_INTEGRATOR; }
  };

  /**
   * @brief 4th-order force-gradient (PQPQP_FG).
   *
   * Same skeleton as Omelyan but replaces the centre P((1-2λ)h) with a
   * force-gradient sub-step that folds in the ξ·h³ correction. λ = 1/6,
   * ξ = 1/72 follow the defaults set in hmc_param.
   */
  class FGIntegrator : public Integrator
  {
  public:
    using Integrator::Integrator;
    void operator()(double tau, int nSteps, ColorSpinorField &phi) override;
    QudaIntegratorType getIntegratorType() const override { return QUDA_FORCE_GRADIENT_INTEGRATOR; }
  };

  /**
   * @brief Nested force-gradient integrator with coarse-grid deflation.
   *
   * Outer integrator: PQPQP_FGI (4th order), coefficients lambda=1/6, xi=1/72.
   * Inner integrator: Leapfrog or Omelyan on cheap force (low-mode deflation
   *                   + gauge force).
   * Force-gradient step: Hessian-free via gauge save/restore.
   *
   * Inherits Integrator so the HMC entry-point can dispatch uniformly via
   * Integrator::create() and treat all integrators (including eigentracking
   * plumbing) through the same interface. The class's internal kick/drift
   * helpers remain private because the two-timescale structure (outer MG
   * force vs. inner low-mode + gauge) does not map onto the base's
   * single-timescale kick/drift primitives.
   */
  class NestedFGIIntegrator : public Integrator
  {
  private:
    CoarseDeflationManager deflManager;
    LowModeForce lowModeForce;

    double lambda;
    double xi;
    double beta;
    int nInnerSteps;
    QudaIntegratorType innerIntegrator;
    double innerOmelyanLambda;

    /** External references for gauge/momentum/clover management.
     *  The gauge-hierarchy entries at refinement / eigensolver / extended
     *  precisions and the extendedGaugeResident halo are accessed by the
     *  integrator's force code through file-scope externs in
     *  hmc_integrator.cpp (see the `extern quda::GaugeField *gauge*;`
     *  block at the top of that file). They were originally captured as
     *  members for symmetry with the precise/sloppy/precondition entries
     *  but never read through the member references — clang -Werror
     *  -Wunused-private-field flags that. Members removed; the externs
     *  are the single source of truth. */
    GaugeField *&gaugePrecise;
    GaugeField *&gaugeSloppy;
    GaugeField *&gaugePrecondition;
    GaugeField &momResident;
    CloverField *&cloverPrecise;

    /** MG preconditioner for outer (expensive) force */
    void *mgPreconditioner;

    // Wilson plaquette gauge-force paths are shared with kick(); see the
    // file-local g_gauge_paths state in hmc_integrator.cpp, populated by
    // setupWilsonGaugePaths().

    /** @brief Perform gauge drift + clover rebuild */
    void gaugeStep(double dt);

    /** @brief Compute the expensive outer force: full MG-prec solve minus low-mode */
    void computeOuterForce(double coeff, const ColorSpinorField &phi);

    /** @brief Compute the cheap inner force: low-mode + gauge */
    void computeInnerForce(double coeff, const ColorSpinorField &phi);

    /** @brief Inner leapfrog sub-integration for duration dt */
    void innerLeapfrog(double dt, ColorSpinorField &phi);

    /** @brief Inner Omelyan (PQPQP) sub-integration for duration dt */
    void innerOmelyan(double dt, ColorSpinorField &phi);

    /** @brief Hessian-free force-gradient sub-step */
    void forceGradientStep(double h, const ColorSpinorField &phi);

  public:
    NestedFGIIntegrator(QudaHMCParam &hmcParam, MG &mg, const DiracMatrix &matFine, void *mgPrec,
                        QudaGaugeParam &gaugeParam, QudaInvertParam &invParam, EigenTrackingState *tracking,
                        GaugeField *&gaugePrecise, GaugeField *&gaugeSloppy, GaugeField *&gaugePrecondition,
                        GaugeField &momResident, CloverField *&cloverPrecise);

    ~NestedFGIIntegrator();

    void operator()(double tau, int nSteps, ColorSpinorField &phi) override;

    QudaIntegratorType getIntegratorType() const override { return QUDA_NESTED_FGI_INTEGRATOR; }

    /**
     * @brief Refresh coarse-grid deflation after an accepted trajectory.
     *
     * Full MG re-setup → re-solve coarse eigenproblem; thin MG update →
     * Rayleigh–Ritz evolve of existing coarse eigenvectors.
     */
    void afterAccepted(bool full_mg_update) override;

    /** @brief Access the deflation manager (e.g. for inter-trajectory RR update). */
    CoarseDeflationManager &getDeflationManager() { return deflManager; }
  };

  /**
   * @brief Free Wilson plaquette gauge-force path tables owned by
   *        hmc_integrator.cpp.
   */
  void releaseIntegratorState();

  /**
   * @brief Get-or-build the persistent Integrator instance.
   *
   * Builds via Integrator::create on first call or whenever
   * @p hmc_param.integrator changes. Otherwise returns the cached
   * instance so nested-FGI's CoarseDeflationManager state survives
   * across trajectories. Caller does not own the returned reference.
   */
  Integrator &getOrCreateIntegrator(QudaHMCParam &hmc_param, QudaGaugeParam &gauge_param, QudaInvertParam &inv_param,
                                    EigenTrackingState *tracking, void *mg_instance);

  /**
   * @brief Pointer to the cached Integrator, or nullptr if none.
   *
   * Used by hmcRunQuda to forward post-accept hooks without rebuilding.
   */
  Integrator *currentIntegrator();

  /**
   * @brief Tear down the cached Integrator. Call before the device pool
   *        is flushed.
   */
  void releaseIntegrator();

  /**
   * @brief Reset per-trajectory CG iteration/solve counters.
   *
   * Call from hmcTrajectoryQuda before each trajectory; the counters
   * accumulate across every kick() within that trajectory.
   */
  void integratorResetCGStats();
  int integratorCGIters();
  int integratorCGSolves();

  /**
   * @brief Attribute @p iters additional CG iterations (one solve) to the
   *        current trajectory's stats. Used by non-integrator call sites
   *        (computeFermionAction) that also perform solves.
   */
  void integratorBumpCGStats(int iters);

} // namespace quda
