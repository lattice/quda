#include <nested_fgi.h>
#include <gauge_update_quda.h>
#include <gauge_path_quda.h>
#include <momentum.h>
#include <blas_quda.h>
#include <malloc_quda.h>
#include <quda_internal.h>
#include <quda.h>

namespace quda
{

  // Forward declarations for file-scope globals in interface_quda.cpp
  extern std::vector<ColorSpinorField> solutionResident;

  NestedFGIIntegrator::NestedFGIIntegrator(const QudaHMCParam &hmcParam, MG &mg, const DiracMatrix &matFine,
                                           void *mgPrec, QudaGaugeParam &gaugeParam_, QudaInvertParam &invParam_,
                                           GaugeField *&gaugePrecise_, GaugeField *&gaugeSloppy_,
                                           GaugeField *&gaugePrecondition_, GaugeField *&gaugeRefinement_,
                                           GaugeField *&gaugeEigensolver_, GaugeField *&gaugeExtended_,
                                           GaugeField &momResident_, CloverField *&cloverPrecise_,
                                           GaugeField *&extendedGaugeResident_) :
    deflManager(*mg.getTransfer(), *mg.getMatCoarseResidual(), hmcParam.n_defl, hmcParam.eig_tol,
                hmcParam.eig_n_kr > 0 ? hmcParam.eig_n_kr : 3 * hmcParam.n_defl, hmcParam.eig_max_restarts,
                hmcParam.defl_refresh_interval),
    lowModeForce(deflManager, matFine, hmcParam.n_mr_smooth, hmcParam.mr_omega),
    lambda(hmcParam.fgi_lambda),
    xi(hmcParam.fgi_xi),
    beta(hmcParam.beta),
    nInnerSteps(hmcParam.n_inner_steps),
    innerIntegrator(hmcParam.inner_integrator),
    innerOmelyanLambda(hmcParam.inner_omelyan_lambda),
    gaugePrecise(gaugePrecise_),
    gaugeSloppy(gaugeSloppy_),
    gaugePrecondition(gaugePrecondition_),
    gaugeRefinement(gaugeRefinement_),
    gaugeEigensolver(gaugeEigensolver_),
    gaugeExtended(gaugeExtended_),
    momResident(momResident_),
    cloverPrecise(cloverPrecise_),
    extendedGaugeResident(extendedGaugeResident_),
    mgPreconditioner(mgPrec),
    gaugeParam(gaugeParam_),
    invParam(invParam_)
  {
    setupGaugePaths();
    logQuda(QUDA_SUMMARIZE,
            "NestedFGIIntegrator: lambda=%f, xi=%f, n_inner=%d, n_defl=%d, inner=%s\n", lambda, xi, nInnerSteps,
            hmcParam.n_defl, innerIntegrator == QUDA_OMELYAN_INTEGRATOR ? "Omelyan" : "Leapfrog");
  }

  NestedFGIIntegrator::~NestedFGIIntegrator() { freeGaugePaths(); }

  void NestedFGIIntegrator::setupGaugePaths()
  {
    for (int i = 0; i < 6; i++) {
      gaugePathLength[i] = 3;
      gaugePathCoeff[i] = 1.0;
    }
    for (int dir = 0; dir < 4; dir++) {
      gaugePaths[dir] = static_cast<int **>(safe_malloc(6 * sizeof(int *)));
      int idx = 0;
      for (int i = 0; i < 4; i++) {
        if (i == dir) continue;
        int opp_dir = 7 - dir;
        int opp_i = 7 - i;
        gaugePaths[dir][idx] = static_cast<int *>(safe_malloc(3 * sizeof(int)));
        gaugePaths[dir][idx][0] = i;
        gaugePaths[dir][idx][1] = opp_dir;
        gaugePaths[dir][idx][2] = opp_i;
        idx++;
        gaugePaths[dir][idx] = static_cast<int *>(safe_malloc(3 * sizeof(int)));
        gaugePaths[dir][idx][0] = opp_i;
        gaugePaths[dir][idx][1] = opp_dir;
        gaugePaths[dir][idx][2] = i;
        idx++;
      }
    }
  }

  void NestedFGIIntegrator::freeGaugePaths()
  {
    for (int dir = 0; dir < 4; dir++) {
      if (gaugePaths[dir]) {
        for (int i = 0; i < 6; i++) host_free(gaugePaths[dir][i]);
        host_free(gaugePaths[dir]);
        gaugePaths[dir] = nullptr;
      }
    }
  }

  void NestedFGIIntegrator::gaugeStep(double dt)
  {
    // Gauge update: U_out = exp(dt * P) * U_in (separate in/out to avoid race)
    GaugeFieldParam gParam(*gaugePrecise);
    gParam.create = QUDA_NULL_FIELD_CREATE;
    GaugeField u_out(gParam);
    updateGaugeField(u_out, dt, *gaugePrecise, momResident, false, true);
    gaugePrecise->copy(u_out);

    // Rebuild sloppy/precondition copies
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);

    // Rebuild clover from updated gauge (also handles extended gauge internally)
    if (cloverPrecise) createCloverQuda(&invParam);
  }

  void NestedFGIIntegrator::computeOuterForce(double coeff, const ColorSpinorField &phi)
  {
    // Outer force = F_gauge + F_full_fermion - F_low
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: computeOuterForce coeff=%e\n", coeff);

    // --- Gauge force ---
    QudaGaugeParam gp = gaugeParam;
    gp.use_resident_gauge = 1;
    gp.use_resident_mom = 1;
    gp.make_resident_gauge = 1;
    gp.make_resident_mom = 1;
    gp.return_result_mom = 0;
    gp.overwrite_mom = 0;

    double eb3 = coeff * beta / 3.0;
    computeGaugeForceQuda(nullptr, nullptr, gaugePaths, gaugePathLength, gaugePathCoeff, 6, 4, eb3, &gp);

    // --- Full fermion force (CG solve + clover force) ---
    QudaInvertParam ip = invParam;
    ip.use_resident_solution = 1;
    ip.make_resident_solution = 1;
    ip.preconditioner = mgPreconditioner;

    ColorSpinorParam cpuParam(nullptr, ip, gaugePrecise->X(), true, QUDA_CPU_FIELD_LOCATION);
    cpuParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField h_x(cpuParam);
    ColorSpinorField h_b(cpuParam);
    h_b = phi; // device → host copy

    invertQuda(h_x.data(), h_b.data(), &ip);

    double fCoeff = 1.0;
    double kappa2 = ip.kappa * ip.kappa;
    double ck = ip.clover_csw * ip.kappa;
    ip.use_resident_solution = 1;
    gp.overwrite_mom = 0; // accumulate
    computeCloverForceQuda(nullptr, coeff, nullptr, nullptr, &fCoeff, kappa2, ck, 1, 1.0, nullptr, &gp, &ip);
    solutionResident.clear();

    // --- Subtract low-mode force ---
    // The inner force contains F_low, so we subtract it here so outer + inner = total
    lowModeForce.computeForce(momResident, phi, -coeff, *gaugePrecise, cloverPrecise, gaugeParam, invParam);
  }

  void NestedFGIIntegrator::computeInnerForce(double coeff, const ColorSpinorField &phi)
  {
    // Inner force = F_gauge + F_low
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: computeInnerForce coeff=%e\n", coeff);

    // --- Gauge force ---
    QudaGaugeParam gp = gaugeParam;
    gp.use_resident_gauge = 1;
    gp.use_resident_mom = 1;
    gp.make_resident_gauge = 1;
    gp.make_resident_mom = 1;
    gp.return_result_mom = 0;
    gp.overwrite_mom = 0;

    double eb3 = coeff * beta / 3.0;
    computeGaugeForceQuda(nullptr, nullptr, gaugePaths, gaugePathLength, gaugePathCoeff, 6, 4, eb3, &gp);

    // --- Low-mode fermion force ---
    lowModeForce.computeForce(momResident, phi, coeff, *gaugePrecise, cloverPrecise, gaugeParam, invParam);

    // Increment deflation refresh counter
    deflManager.step();
    deflManager.maybeRefresh();
  }

  void NestedFGIIntegrator::innerLeapfrog(double dt, const ColorSpinorField &phi)
  {
    double dti = dt / nInnerSteps;
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: innerLeapfrog dt=%e, dti=%e, n=%d\n", dt, dti, nInnerSteps);

    // Half kick
    computeInnerForce(dti / 2.0, phi);

    for (int i = 0; i < nInnerSteps; i++) {
      // Full drift + clover rebuild
      gaugeStep(dti);

      // Full or half kick
      double kickDt = (i < nInnerSteps - 1) ? dti : dti / 2.0;
      computeInnerForce(kickDt, phi);
    }
  }

  void NestedFGIIntegrator::innerOmelyan(double dt, const ColorSpinorField &phi)
  {
    double dti = dt / nInnerSteps;
    double lam = innerOmelyanLambda;
    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: innerOmelyan dt=%e, dti=%e, n=%d, lam=%f\n", dt, dti, nInnerSteps,
            lam);

    // Initial P(lambda*dti)
    computeInnerForce(lam * dti, phi);

    for (int i = 0; i < nInnerSteps; i++) {
      // Q(dti/2)
      gaugeStep(dti / 2.0);

      // P((1-2*lambda)*dti)
      computeInnerForce((1.0 - 2.0 * lam) * dti, phi);

      // Q(dti/2)
      gaugeStep(dti / 2.0);

      // P(2*lambda*dti) for merged, P(lambda*dti) for final
      double kickDt = (i < nInnerSteps - 1) ? 2.0 * lam * dti : lam * dti;
      computeInnerForce(kickDt, phi);
    }
  }

  void NestedFGIIntegrator::forceGradientStep(double h, const ColorSpinorField &phi)
  {
    double one_m_2lam = 1.0 - 2.0 * lambda;
    double xi_h3 = xi * h * h * h;

    logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: forceGradientStep h=%e\n", h);

    // 1. Save gauge + momentum state (deep copies on device)
    GaugeField gaugeSaved(*gaugePrecise);
    GaugeField momSaved(momResident);

    // 2. Zero momentum
    momResident.zero();

    // 3. Compute outer force F, kick: mom = (xi_h3 / (one_m_2lam * h)) * F = (h^2/24) * F
    double fgCoeff = xi_h3 / (one_m_2lam * h);
    computeOuterForce(fgCoeff, phi);

    // 4. Displace gauge: U' = exp(1.0 * mom) * U (separate in/out)
    {
      GaugeFieldParam gfParam(*gaugePrecise);
      gfParam.create = QUDA_NULL_FIELD_CREATE;
      GaugeField u_out(gfParam);
      updateGaugeField(u_out, 1.0, *gaugePrecise, momResident, false, true);
      gaugePrecise->copy(u_out);
    }
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);

    // 5. Restore momentum from saved copy
    momResident.copy(momSaved);

    // 6. Compute outer force F' at displaced gauge, kick: mom += (1-2*lambda)*h * F'
    computeOuterForce(one_m_2lam * h, phi);

    // 7. Restore gauge from saved copy
    gaugePrecise->copy(gaugeSaved);
    if (gaugeSloppy && gaugeSloppy != gaugePrecise) gaugeSloppy->copy(*gaugePrecise);
    if (gaugePrecondition && gaugePrecondition != gaugePrecise && gaugePrecondition != gaugeSloppy)
      gaugePrecondition->copy(*gaugePrecise);
  }

  void NestedFGIIntegrator::trajectory(double tau, int nSteps, const ColorSpinorField &phi)
  {
    double h = tau / nSteps;

    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: trajectory tau=%e, n_outer=%d, h=%e, inner=%s\n", tau, nSteps, h,
            innerIntegrator == QUDA_OMELYAN_INTEGRATOR ? "Omelyan" : "Leapfrog");

    deflManager.resetCounter();

    // Select inner sub-integrator
    auto runInner = [&](double dt) {
      if (innerIntegrator == QUDA_OMELYAN_INTEGRATOR)
        innerOmelyan(dt, phi);
      else
        innerLeapfrog(dt, phi);
    };

    // PQPQP_FGI outer integrator:
    //   P(lambda*h) [inner(h/2) FG inner(h/2) P(2*lambda*h)]^{n-1} inner(h/2) FG inner(h/2) P(lambda*h)

    // Initial outer P-kick
    computeOuterForce(lambda * h, phi);

    for (int o = 0; o < nSteps; o++) {
      logQuda(QUDA_VERBOSE, "NestedFGIIntegrator: outer step %d/%d\n", o + 1, nSteps);

      runInner(h / 2.0);
      forceGradientStep(h, phi);
      runInner(h / 2.0);

      // Merged P-kick for adjacent steps, half-kick for final
      double kickCoeff = (o < nSteps - 1) ? 2.0 * lambda * h : lambda * h;
      computeOuterForce(kickCoeff, phi);
    }

    logQuda(QUDA_SUMMARIZE, "NestedFGIIntegrator: trajectory complete\n");
  }

} // namespace quda
