#pragma once

#include <color_spinor_field.h>
#include <gauge_field.h>
#include <quda.h>

namespace quda
{

  /**
   * @brief Gauge action: S_g = beta * (1 - plaq) * V * 6.
   */
  double computeGaugeActionHMC(double beta);

  /**
   * @brief Generate EO pseudofermion: φ_e = D̂ η_e, η_e ~ N(0,1).
   *
   * Creates a single-parity (even) pseudofermion field on the GPU.
   */
  ColorSpinorField generateEOPseudofermion(QudaInvertParam &inv_param, unsigned long long seed);

  /**
   * @brief EO fermion action: S_f = Re(φ_e† x_e) where D̂†D̂ x_e = φ_e.
   *
   * Solves the even-odd preconditioned normal equations via CG.
   */
  double computeEOFermionAction(ColorSpinorField &phi_even, QudaInvertParam &inv_param);

  /**
   * @brief EO fermion force: accumulates dS_eo/dU into momentum.
   *
   * Implements the verified Schwinger EO force algorithm:
   *   Y_e = D̂ x_e, p_o = Dslash(x_e), q_o = Dslash^dag(Y_e)
   *   X_full = (x_e, p_o), chi_full = (Y_e, q_o)
   *   force = -outer_product(chi_full, X_full)
   *
   * @param mom      Momentum field (accumulated into)
   * @param x_even   CG solution: D̂†D̂ x_e = φ_e (even parity)
   * @param inv_param Inverter parameters (provides kappa, dslash type)
   * @param dt       Integration step size
   */
  void computeEOFermionForce(GaugeField &mom, ColorSpinorField &x_even,
                             QudaInvertParam &inv_param, double dt);

  /**
   * @brief Seed the eigentracker from MG null-space vectors.
   *
   * Extracts even-parity components of the MG null vectors and feeds
   * them into the eigentracker pool, avoiding the expensive initial TRLM.
   * Must be called after MG setup and before the first HMC trajectory.
   *
   * @param mg_instance  MG preconditioner (multigrid_solver *)
   * @param hmc_param    HMC parameters (for eigentracking config)
   * @param inv_param    Inverter parameters (for Dirac operator creation)
   */
  void seedEigenTrackingFromMG(void *mg_instance, QudaHMCParam *hmc_param, QudaInvertParam *inv_param);

} // namespace quda
