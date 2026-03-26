#include <nested_fgi.h>
#include <gauge_backup.h>
#include <gauge_update_quda.h>
#include <momentum.h>
#include <blas_quda.h>
#include <quda_internal.h>

namespace quda
{

  NestedFGIIntegrator::NestedFGIIntegrator(const QudaHMCParam &hmcParam, MG &mg, void *mgPrec,
                                           QudaGaugeParam &gaugeParam_, QudaInvertParam &invParam_,
                                           GaugeField *&gaugePrecise_, GaugeField *&gaugeSloppy_,
                                           GaugeField *&gaugePrecondition_, GaugeField *&gaugeRefinement_,
                                           GaugeField *&gaugeEigensolver_, GaugeField *&gaugeExtended_,
                                           GaugeField &momResident_) :
    deflManager(*mg.getTransfer(), *mg.getMatCoarseResidual(), hmcParam.n_defl, hmcParam.eig_tol,
                hmcParam.eig_n_kr > 0 ? hmcParam.eig_n_kr : 3 * hmcParam.n_defl, hmcParam.eig_max_restarts,
                hmcParam.defl_refresh_interval),
    lowModeForce(deflManager, *mg.getMatCoarseResidual(), hmcParam.n_mr_smooth, hmcParam.mr_omega),
    lambda(hmcParam.fgi_lambda),
    xi(hmcParam.fgi_xi),
    nInnerSteps(hmcParam.n_inner_steps),
    gaugePrecise(gaugePrecise_),
    gaugeSloppy(gaugeSloppy_),
    gaugePrecondition(gaugePrecondition_),
    gaugeRefinement(gaugeRefinement_),
    gaugeEigensolver(gaugeEigensolver_),
    gaugeExtended(gaugeExtended_),
    momResident(momResident_),
    mgPreconditioner(mgPrec),
    gaugeParam(gaugeParam_),
    invParam(invParam_)
  {
    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: lambda=%f, xi=%f, n_inner=%d, n_defl=%d\n", lambda, xi, nInnerSteps,
            hmcParam.n_defl);
  }

  void NestedFGIIntegrator::computeOuterForce(GaugeField &mom, double coeff, const ColorSpinorField &phi)
  {
    // The outer force = F_full - F_low + F_gauge
    // F_full comes from a full MG-preconditioned CG solve
    // F_low is subtracted so the outer + inner forces sum to the total

    // 1. Full CG solve: X_full = (D†D)^{-1} phi using MG preconditioner
    invParam.preconditioner = mgPreconditioner;
    ColorSpinorField xFull(ColorSpinorParam(phi));
    blas::zero(xFull);

    // Use invertQuda pattern: the solution goes into xFull
    // For now, call invertQuda via the C-API with resident fields
    // This is a placeholder -- actual integration requires wiring to the solve() interface
    invertQuda(static_cast<void *>(xFull.data()), static_cast<void *>(const_cast<ColorSpinorField &>(phi).data()),
               &invParam);

    // 2. Compute full fermion force from X_full and accumulate into mom
    // F_full is accumulated with coefficient +coeff
    // (The actual force computation goes through computeCloverForceQuda or similar)

    // 3. Subtract the low-mode force contribution with coefficient -coeff
    // so that outer + inner = total
    // lowModeForce.computeForce(mom, phi, -coeff, *gaugePrecise, clover, gaugeParam, invParam);

    // 4. Add gauge force
    // computeGaugeForceQuda(...)

    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: Computed outer force with coeff=%e\n", coeff);
  }

  void NestedFGIIntegrator::computeInnerForce(GaugeField &mom, double coeff, const ColorSpinorField &phi)
  {
    // Inner force = F_low + F_gauge
    // F_low = fermion_force(D_current, X_low) where X_low is the deflation-projected solution

    // 1. Low-mode fermion force
    // lowModeForce.computeForce(mom, phi, coeff, *gaugePrecise, clover, gaugeParam, invParam);

    // 2. Gauge force (cheap)
    // computeGaugeForceQuda(...)

    // Increment deflation step counter
    deflManager.step();
    deflManager.maybeRefresh();

    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: Computed inner force with coeff=%e\n", coeff);
  }

  void NestedFGIIntegrator::innerLeapfrog(double dt, const ColorSpinorField &phi)
  {
    double dti = dt / nInnerSteps;

    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: Inner leapfrog, dt=%e, dti=%e, n_inner=%d\n", dt, dti, nInnerSteps);

    // Half kick
    computeInnerForce(momResident, dti / 2.0, phi);

    for (int i = 0; i < nInnerSteps; i++) {
      // Full drift: U = exp(dti * P) * U
      updateGaugeField(*gaugePrecise, dti, *gaugePrecise, momResident, false, true);

      // Rebuild sloppy/precondition gauge copies if they alias
      if (gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
      if (gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
        gaugePrecondition->copy(*gaugePrecise);

      // Full or half kick
      double kickCoeff = (i < nInnerSteps - 1) ? dti : dti / 2.0;
      computeInnerForce(momResident, kickCoeff, phi);
    }
  }

  void NestedFGIIntegrator::forceGradientStep(double h, const ColorSpinorField &phi)
  {
    double one_m_2lam = 1.0 - 2.0 * lambda;
    double xi_h3 = xi * h * h * h;

    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: Force-gradient step, h=%e\n", h);

    // 1. Save gauge + momentum state
    GaugeBundleBackup gaugeBkup;
    gaugeBkup.backup(gaugePrecise, gaugeSloppy, gaugePrecondition, gaugeRefinement, gaugeEigensolver, gaugeExtended);
    GaugeField momBackup(momResident); // deep copy

    // 2. Zero momentum
    momResident.zero();

    // 3. Compute outer force F, kick: mom = (xi_h3 / (one_m_2lam * h)) * F = (h^2/24) * F
    //    Note: updateMomentum does mom = mom - coeff * force, so we negate the coefficient
    double fgCoeff = xi_h3 / (one_m_2lam * h); // = h^2/24 for lambda=1/6, xi=1/72
    computeOuterForce(momResident, fgCoeff, phi);

    // 4. Displace gauge: U' = exp(1.0 * mom) * U
    //    The momentum now contains only the FG displacement
    updateGaugeField(*gaugePrecise, 1.0, *gaugePrecise, momResident, false, true);

    // Rebuild sloppy/precondition copies for force evaluation at displaced gauge
    if (gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);

    // 5. Restore momentum from backup
    momResident.copy(momBackup);

    // 6. Compute outer force F' at displaced gauge U'
    //    Kick: mom += (1-2*lambda)*h * F' = (2h/3) * F'
    computeOuterForce(momResident, one_m_2lam * h, phi);

    // 7. Restore gauge from backup
    //    Use setupGaugeFields which handles aliasing correctly
    GaugeField *newPrecise = new GaugeField(*gaugeBkup.precise);
    setupGaugeFields(newPrecise, gaugePrecise, gaugeSloppy, gaugePrecondition, gaugeRefinement, gaugeEigensolver,
                     gaugeExtended, gaugeBkup, getProfile());
  }

  void NestedFGIIntegrator::trajectory(double tau, int nSteps, const ColorSpinorField &phi)
  {
    double h = tau / nSteps;

    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: Starting trajectory, tau=%e, n_outer=%d, h=%e\n", tau, nSteps, h);

    // Reset deflation step counter for this trajectory
    deflManager.resetCounter();

    // PQPQP_FGI outer integrator:
    //   P(lambda*h) [inner(h/2) FG((1-2*lambda)*h, xi*h^3) inner(h/2) P(2*lambda*h)]^{n-1}
    //   inner(h/2) FG(...) inner(h/2) P(lambda*h)

    // Initial outer P-kick: P(lambda*h)
    computeOuterForce(momResident, lambda * h, phi);

    for (int o = 0; o < nSteps; o++) {
      logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: Outer step %d/%d\n", o + 1, nSteps);

      // Inner sub-integrator: inner(h/2)
      innerLeapfrog(h / 2.0, phi);

      // Force-gradient step: FG((1-2*lambda)*h, xi*h^3)
      forceGradientStep(h, phi);

      // Inner sub-integrator: inner(h/2)
      innerLeapfrog(h / 2.0, phi);

      // Outer P-kick: P(2*lambda*h) for merged steps, P(lambda*h) for final
      if (o < nSteps - 1) {
        computeOuterForce(momResident, 2.0 * lambda * h, phi); // merged adjacent kicks
      } else {
        computeOuterForce(momResident, lambda * h, phi); // final half-kick
      }
    }

    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: Trajectory complete\n");
  }

} // namespace quda
