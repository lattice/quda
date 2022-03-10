#pragma once

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <util_quda.h>

namespace quda
{

  /**
    @brief a struct that includes the parameters that determines how reliable updates (aka defect correction, etc)
      should be performed. There are two variants:
      - The "original" or the naive one (alternative_reliable = false), where higher precision corrections are
        made every time the redidual is decreased to
          (delta) * (the residual when the last reliable update is performed)
      - A more sophisticated approach where the lower precision inaccuracy is estimated and accumulated from
        iteration to iteration, and higher precision corrections are made when this inaccuracy is higher than a
        threashhold. See https://doi.org/10.1137/S1064827599353865 for reference.
    @param alternative_reliable determines which approach we use: alternative_reliable = false means the first
      approach, alternative_reliable = true means the second approach
    @param u the lower precision tolerance, only relevant for the second approach
    @param uhigh the higher precision tolerance, only relevant for the second approach
    @param Anorm The normal of the underlying operator that is to be used to estimate the lower precision inaccuracy,
      only relevant for the second approach
    @param maxResIncrease number of consecutive residual increases between reliable updates allowed: should only matter
    for the first approach
    @param maxResIncreaseTotal total number of residual increases between reliable updates allowed: should only matter
    for the first approach
    @param use_heavy_quark_res whether or not using heavy quark residual
    @param hqmaxresIncrease same as maxResIncrease, but for heavy quark
    @param hqmaxresRestartTotal total number of heavy quark restarts allowed
   */
  struct ReliableUpdatesParams {

    bool alternative_reliable;

    double u;
    double uhigh;
    double Anorm;
    double delta;

    // this parameter determines how many consective reliable update
    // residual increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    int maxResIncrease; //  check if we reached the limit of our tolerance
    int maxResIncreaseTotal;

    bool use_heavy_quark_res;

    int hqmaxresIncrease;
    int hqmaxresRestartTotal; // this limits the number of heavy quark restarts we can do
  };

  struct ReliableUpdates {

    const ReliableUpdatesParams params;

    const double deps;
    static constexpr double dfac = 1.1;
    double d_new = 0;
    double d = 0;
    double dinit = 0;
    double xNorm = 0;
    double xnorm = 0;
    double pnorm = 0;
    double ppnorm = 0;
    double delta;
    double beta = 0.0;

    double rNorm;
    double r0Norm;
    double maxrx;
    double maxrr;
    double maxr_deflate; // The maximum residual since the last deflation

    int resIncrease = 0;
    int resIncreaseTotal = 0;
    int hqresIncrease = 0;
    int hqresRestartTotal = 0;

    int updateX = 0;
    int updateR = 0;

    int steps_since_reliable = 1;

    int rUpdate = 0;

    /**
      @brief constructor
      @param params the parameters
      @param r2 the residual norm squared
     */
    ReliableUpdates(ReliableUpdatesParams params, double r2) :
      params(params),
      deps(sqrt(params.u)),
      delta(params.delta),
      rNorm(sqrt(r2)),
      r0Norm(rNorm),
      maxrx(rNorm),
      maxrr(rNorm),
      maxr_deflate(rNorm)
    {
      // alternative reliable updates
      if (params.alternative_reliable) {
        dinit = params.uhigh * (rNorm + params.Anorm * xNorm);
        d = dinit;
      }
    }

    /**
      @brief Update the norm squared for p (thus ppnorm)
     */
    void update_ppnorm(double ppnorm_) { ppnorm = ppnorm_; }

    /**
      @brief Update the norm for r (thus rNorm)
     */
    void update_rNorm(double rNorm_) { rNorm = rNorm_; }

    /**
      @brief Update maxr_deflate
     */
    void update_maxr_deflate(double r2) { maxr_deflate = sqrt(r2); }

    /**
      @brief Evaluate whether a reliable update is needed
      @param r2_old the old residual norm squared
     */
    void evaluate(double r2_old)
    {
      if (params.alternative_reliable) {
        // alternative reliable updates
        updateX = ((d <= deps * sqrt(r2_old)) or (dfac * dinit > deps * r0Norm)) and (d_new > deps * rNorm)
          and (d_new > dfac * dinit);
        updateR = 0;
      } else {
        if (rNorm > maxrx) maxrx = rNorm;
        if (rNorm > maxrr) maxrr = rNorm;
        updateX = (rNorm < params.delta * r0Norm && r0Norm <= maxrx) ? 1 : 0;
        updateR = ((rNorm < params.delta * maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
      }
    }

    /**
      @brief Set updateX to 1
     */
    void set_updateX() { updateX = 1; }

    /**
      @brief Whether it is time to do reliable update
     */
    bool trigger() { return updateR || updateX; }

    /**
      @brief Accumulate the estimate for error - used when reliable update is not performed
      @param alpha the alpha that is used in CG to update the solution vector x, given p
     */
    void accumulate_norm(double alpha)
    {
      // accumulate norms
      if (params.alternative_reliable) {
        d = d_new;
        pnorm = pnorm + alpha * alpha * ppnorm;
        xnorm = sqrt(pnorm);
        d_new = d + params.u * rNorm + params.uhigh * params.Anorm * xnorm;
        if (steps_since_reliable == 0 && getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("New dnew: %e (r %e , y %e)\n", d_new, params.u * rNorm, params.uhigh * params.Anorm * xnorm);
      }
      steps_since_reliable++;
    }

    /**
      @brief Reset the estimate for error - used when reliable update is performed
      @param r2 the residual norm squared
      @param y2 the solution vector norm squared
     */
    void update_norm(double r2, ColorSpinorField &y)
    {
      // update_norms
      if (params.alternative_reliable) {
        double y2 = blas::norm2(y);
        dinit = params.uhigh * (sqrt(r2) + params.Anorm * sqrt(y2));
        d = d_new;
        xnorm = 0; // sqrt(norm2(x));
        pnorm = 0; // pnorm + alpha * sqrt(norm2(p));
        if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
          printfQuda("New dinit: %e (r %e , y %e)\n", dinit, params.uhigh * sqrt(r2),
                     params.uhigh * params.Anorm * sqrt(y2));
        d_new = dinit;
      } else {
        rNorm = sqrt(r2);
        maxrr = rNorm;
        maxrx = rNorm;
      }
    }

    /**
      @brief Whether we should break out, i.e. we have reached the limit of the precisions
      @param r2 residual norm squared
      @param stop the stopping condition
      @param[in/out] L2breakdown whether or not L2 breakdown
      @param L2breakdown_eps L2 breakdown epsilon
     */
    bool reliable_break(double r2, double stop, bool &L2breakdown, double L2breakdown_eps)
    {
      // break-out check if we have reached the limit of the precision
      if (sqrt(r2) > r0Norm && updateX and not L2breakdown) { // reuse r0Norm for this
        resIncrease++;
        resIncreaseTotal++;
        warningQuda("new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
                    sqrt(r2), r0Norm, resIncreaseTotal);

        if ((params.use_heavy_quark_res and sqrt(r2) < L2breakdown_eps) or resIncrease > params.maxResIncrease
            or resIncreaseTotal > params.maxResIncreaseTotal or r2 < stop) {
          if (params.use_heavy_quark_res) {
            L2breakdown = true;
            warningQuda("L2 breakdown %e, %e", sqrt(r2), L2breakdown_eps);
          } else {
            if (resIncrease > params.maxResIncrease or resIncreaseTotal > params.maxResIncreaseTotal or r2 < stop) {
              warningQuda("solver exiting due to too many true residual norm increases");
              return true;
            }
          }
        }
      } else {
        resIncrease = 0;
      }
      return false;
    }

    /**
      @brief Whether we should break out for heavy quark
      @param L2breakdown whether or not L2 breakdown
      @param heavy_quark_res the heavy quark residual
      @param heavy_quark_res_old the old heavy quark residual
      @param[in/out] heavy_quark_restart whether should restart the heavy quark
     */
    bool reliable_heavy_quark_break(bool L2breakdown, double heavy_quark_res, double heavy_quark_res_old,
                                    bool &heavy_quark_restart)
    {
      if (params.use_heavy_quark_res and L2breakdown) {
        hqresRestartTotal++; // count the number of heavy quark restarts we've done
        delta = 0;
        warningQuda("CG: Restarting without reliable updates for heavy-quark residual (total #inc %i)",
                    hqresRestartTotal);
        heavy_quark_restart = true;

        if (heavy_quark_res > heavy_quark_res_old) { // check if new hq residual is greater than previous
          hqresIncrease++;                           // count the number of consecutive increases
          warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e",
                      heavy_quark_res, heavy_quark_res_old);
          // break out if we do not improve here anymore
          if (hqresIncrease > params.hqmaxresIncrease) {
            warningQuda("CG: solver exiting due to too many heavy quark residual norm increases (%i/%i)", hqresIncrease,
                        params.hqmaxresIncrease);
            return true;
          }
        } else {
          hqresIncrease = 0;
        }
      }

      if (hqresRestartTotal > params.hqmaxresRestartTotal) {
        warningQuda("CG: solver exiting due to too many heavy quark residual restarts (%i/%i)", hqresRestartTotal,
                    params.hqmaxresRestartTotal);
        return true;
      }
      return false;
    }

    /**
      @brief Reset the counters - after a reliable update has been performed
      @param r2 residual norm squared
    */
    void reset(double r2)
    {
      steps_since_reliable = 0;
      r0Norm = sqrt(r2);
      rUpdate++;
    }
  };

} // namespace quda
