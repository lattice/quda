#pragma once

/**
 * @file inv_tracker.h
 * @brief Shared lifecycle utilities for solver-side eigentracking trackers.
 *
 * CGTracker (cg_tracker.h, cg_ritz_extractor.cpp) and GCRTracker
 * (gcr_tracker.h, gcr_tracker.cpp) implement two unrelated algorithms
 * for harvesting low-mode information out of a fermion solve --- CG via
 * the implicit Lanczos tridiagonal, GCR via a residual stash --- but
 * they share the same lifecycle:
 *
 *   1. caller allocates a tracker on the stack
 *   2. install it on the solver-loop's extern global pointer
 *   3. run the solve
 *   4. uninstall (restoring whatever was there before)
 *   5. take whatever the tracker captured and absorb into the ET pool
 *
 * The shared bits (steps 2/4 install/uninstall, step 5 absorb) live
 * here. The algorithms themselves stay in their own translation units.
 */

#include <color_spinor_field.h>
#include <quda_internal.h>
#include <vector>

#include <cg_tracker.h>
#include <gcr_tracker.h>

namespace quda
{

  /**
   * @brief Scope-bound install / restore for a tracker's extern global slot.
   *
   * Replaces the manual save / restore boilerplate at every install
   * site:
   *   T *prev = activeT; activeT = &t; ... activeT = prev;
   *
   * Pass nullptr for the tracker pointer to install a "no tracker for
   * this solve" hole inside an outer scope that had one installed --- the
   * solver loop sees nullptr for the duration of this scope and reverts
   * to the outer install on destruction.
   *
   * Same template instantiated against different concrete tracker types
   * (CGTracker, GCRTracker) so each tracker keeps its own algorithm
   * surface and its own extern global, without sharing virtual-call
   * overhead in the solver hot path.
   */
  template <typename T> class TrackerScope
  {
  private:
    T *&slot_;
    T *prev_;

  public:
    TrackerScope(T *&slot, T *t) : slot_(slot), prev_(slot) { slot_ = t; }
    ~TrackerScope() { slot_ = prev_; }

    TrackerScope(const TrackerScope &) = delete;
    TrackerScope &operator=(const TrackerScope &) = delete;
  };

  /**
   * @brief Drain captured residuals out of a GCRTracker as Ritz-equivalent vectors.
   *
   * GCR cannot do Lanczos-tridiag Ritz extraction (its T_m is the QR
   * factor of A·P, not the Hessenberg of A in any basis we have); we
   * instead hand the raw normalised residuals to the EigenTracker pool,
   * whose downstream Rayleigh-Ritz step picks the low-mode content out
   * of them. Returns empty when the tracker is inactive (cap=0) so the
   * call site can use the same drain pattern as the CG case below.
   */
  inline std::vector<ColorSpinorField> takeRitzVectors(GCRTracker &t)
  {
    if (!t.isActive()) return {};
    return t.takeResiduals();
  }

  /**
   * @brief Extract zero-cost Ritz vectors from a CGTracker.
   *
   * Builds the Lanczos tridiagonal from the recorded α/β, eigendecomposes
   * it on the host, reconstructs the Ritz vectors via v_p = Q_m u_p.
   * Returns empty when the tracker is inactive (cap=0) or insufficient
   * iterations were captured. The Ritz eigenvalues are discarded by this
   * helper --- absorbtion into the EigenTracker pool only needs vectors;
   * if a future caller wants the eigenvalues, use t.extractRitzPairs
   * directly.
   */
  inline std::vector<ColorSpinorField> takeRitzVectors(CGTracker &t)
  {
    if (!t.isActive()) return {};
    std::vector<ColorSpinorField> vecs;
    std::vector<Complex> vals;
    t.extractRitzPairs(vecs, vals);
    return vecs;
  }

} // namespace quda
