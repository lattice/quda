#pragma once

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <util_quda.h>

struct reliable_updates {

  const bool alternative_reliable;

  const double u;
  const double uhigh;

  const double deps;
  static constexpr double dfac = 1.1;
  double d_new = 0;
  double d = 0;
  double dinit = 0;
  double xNorm = 0;
  double xnorm = 0;
  double pnorm = 0;
  double ppnorm = 0;
  double Anorm = 0;
  double beta = 0.0;

  double rNorm;
  double r0Norm;
  double maxrx;
  double maxrr;
  double maxr_deflate; // The maximum residual since the last deflation
  double delta;

  int resIncrease = 0;
  int resIncreaseTotal = 0;
  int hqresIncrease = 0;
  int hqresRestartTotal = 0;

  // this parameter determines how many consective reliable update
  // residual increases we tolerate before terminating the solver,
  // i.e., how long do we want to keep trying to converge
  int maxResIncrease; //  check if we reached the limit of our tolerance
  int maxResIncreaseTotal;

  bool use_heavy_quark_res;

  int hqmaxresIncrease;
  int hqmaxresRestartTotal; // this limits the number of heavy quark restarts we can do

  int updateX = 0;
  int updateR = 0;

  int steps_since_reliable = 1;

  int rUpdate = 0;

  reliable_updates(bool alternative_reliable, double u, double uhigh, double Anorm, double r2, double delta,
                   int max_res_increase, int max_res_increase_total, bool use_heavy_quark_res, int max_hq_res_increase,
                   int max_hq_res_restart_total) :
    alternative_reliable(alternative_reliable),
    u(u),
    uhigh(uhigh),
    deps(sqrt(u)),
    Anorm(Anorm),
    rNorm(sqrt(r2)),
    r0Norm(rNorm),
    maxrx(rNorm),
    maxrr(rNorm),
    maxr_deflate(rNorm),
    delta(delta),
    maxResIncrease(max_res_increase), //  check if we reached the limit of our tolerance
    maxResIncreaseTotal(max_res_increase_total),
    use_heavy_quark_res(use_heavy_quark_res),
    hqmaxresIncrease(max_hq_res_increase),
    hqmaxresRestartTotal(max_hq_res_restart_total)

  {
    // alternative reliable updates
    if (alternative_reliable) {
      dinit = uhigh * (rNorm + Anorm * xNorm);
      d = dinit;
    }
  }

  void update_ppnorm(double ppnorm_) { ppnorm = ppnorm_; }

  void update_rNorm(double rNorm_) { rNorm = rNorm_; }

  void update_maxr_deflate(double r2) { maxr_deflate = sqrt(r2); }

  // Evaluate whether a reliable update is needed
  void evaluate(double r2_old)
  {
    if (alternative_reliable) {
      // alternative reliable updates
      updateX = ((d <= deps * sqrt(r2_old)) or (dfac * dinit > deps * r0Norm)) and (d_new > deps * rNorm)
        and (d_new > dfac * dinit);
      updateR = 0;
    } else {
      if (rNorm > maxrx) maxrx = rNorm;
      if (rNorm > maxrr) maxrr = rNorm;
      updateX = (rNorm < delta * r0Norm && r0Norm <= maxrx) ? 1 : 0;
      updateR = ((rNorm < delta * maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
    }
  }

  // Set updateX to 1
  void set_updateX() { updateX = 1; }

  // Whether it is time to do reliable update
  bool trigger() { return updateR || updateX; }

  // Accumulate the estimate for error - used when reliable update is not performed
  void accumulate_norm(double alpha)
  {
    // accumulate norms
    if (alternative_reliable) {
      d = d_new;
      pnorm = pnorm + alpha * alpha * ppnorm;
      xnorm = sqrt(pnorm);
      d_new = d + u * rNorm + uhigh * Anorm * xnorm;
      if (steps_since_reliable == 0 && getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("New dnew: %e (r %e , y %e)\n", d_new, u * rNorm, uhigh * Anorm * xnorm);
    }
    steps_since_reliable++;
  }

  // Reset the estimate for error - used when reliable update is performed
  void update_norm(double r2, ColorSpinorField &y)
  {
    // update_norms
    if (alternative_reliable) {
      double y2 = blas::norm2(y);
      dinit = uhigh * (sqrt(r2) + Anorm * sqrt(y2));
      d = d_new;
      xnorm = 0; // sqrt(norm2(x));
      pnorm = 0; // pnorm + alpha * sqrt(norm2(p));
      if (getVerbosity() >= QUDA_DEBUG_VERBOSE)
        printfQuda("New dinit: %e (r %e , y %e)\n", dinit, uhigh * sqrt(r2), uhigh * Anorm * sqrt(y2));
      d_new = dinit;
    } else {
      rNorm = sqrt(r2);
      maxrr = rNorm;
      maxrx = rNorm;
    }
  }

  // Whether we should break out
  bool reliable_break(double r2, double stop, bool &L2breakdown, double L2breakdown_eps)
  {
    // break-out check if we have reached the limit of the precision
    if (sqrt(r2) > r0Norm && updateX and not L2breakdown) { // reuse r0Norm for this
      resIncrease++;
      resIncreaseTotal++;
      warningQuda("new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
                  sqrt(r2), r0Norm, resIncreaseTotal);

      if ((use_heavy_quark_res and sqrt(r2) < L2breakdown_eps) or resIncrease > maxResIncrease
          or resIncreaseTotal > maxResIncreaseTotal or r2 < stop) {
        if (use_heavy_quark_res) {
          L2breakdown = true;
          warningQuda("L2 breakdown %e, %e", sqrt(r2), L2breakdown_eps);
        } else {
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal or r2 < stop) {
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

  // Whether we should break out for heavy quark
  bool reliable_heavy_quark_break(bool L2breakdown, double heavy_quark_res, double heavy_quark_res_old,
                                  bool &heavy_quark_restart)
  {
    if (use_heavy_quark_res and L2breakdown) {
      hqresRestartTotal++; // count the number of heavy quark restarts we've done
      delta = 0;
      warningQuda("CG: Restarting without reliable updates for heavy-quark residual (total #inc %i)", hqresRestartTotal);
      heavy_quark_restart = true;

      if (heavy_quark_res > heavy_quark_res_old) { // check if new hq residual is greater than previous
        hqresIncrease++;                           // count the number of consecutive increases
        warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e",
                    heavy_quark_res, heavy_quark_res_old);
        // break out if we do not improve here anymore
        if (hqresIncrease > hqmaxresIncrease) {
          warningQuda("CG: solver exiting due to too many heavy quark residual norm increases (%i/%i)", hqresIncrease,
                      hqmaxresIncrease);
          return true;
        }
      } else {
        hqresIncrease = 0;
      }
    }

    if (hqresRestartTotal > hqmaxresRestartTotal) {
      warningQuda("CG: solver exiting due to too many heavy quark residual restarts (%i/%i)", hqresRestartTotal,
                  hqmaxresRestartTotal);
      return true;
    }
    return false;
  }

  // Reset the counters - after a reliable update has been performed
  void reset(double r2)
  {
    steps_since_reliable = 0;
    r0Norm = sqrt(r2);
    rUpdate++;
  }
};
