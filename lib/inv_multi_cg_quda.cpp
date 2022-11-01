#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

/*!
 * Generic Multi Shift Solver
 *
 * For staggered, the mass is folded into the dirac operator
 * Otherwise the matrix mass is 'unmodified'.
 *
 * The lowest offset is in offsets[0]
 *
 */

#include <worker.h>

namespace quda {

  /**
     This worker class is used to update the shifted p and x vectors.
     These updates take place in the subsequent dslash application in
     the next iteration, while we're waiting on communication to
     complete.  This results in improved strong scaling of the
     multi-shift solver.

     Since the matrix-vector consists of multiple dslash applications,
     we partition the shifts between these successive dslash
     applications for optimal communications hiding.

     In general, when using a Worker class to hide communication in
     the dslash, one must be aware whether auto-tuning on the dslash
     policy that envelops the dslash will occur.  If so, then the
     Worker class instance will be called multiple times during this
     tuning, potentially rendering the results wrong.  This isn't a
     problem in the multi-shift solve, since we are guaranteed to not
     run the worker class on the first iteration (when the dslash
     policy tuning will take place), but this is something that will
     need to be addressed in the future as the Worker idea to applied
     elsewhere.
   */
  class ShiftUpdate : public Worker {

    ColorSpinorField &r;
    std::vector<ColorSpinorField> &p;
    std::vector<ColorSpinorField> &x;

    std::vector<double> &alpha;
    std::vector<double> &beta;
    std::vector<double> &zeta;
    std::vector<double> &zeta_old;

    const int j_low;
    int n_shift;

    /**
       How much to partition the shifted update.  Assuming the
       operator is (M^\dagger M), this means four applications of
       dslash for Wilson type operators and two applications for
       staggered
    */
    int n_update;

  public:
    ShiftUpdate(ColorSpinorField &r, std::vector<ColorSpinorField> &p, std::vector<ColorSpinorField> &x,
                std::vector<double> &alpha, std::vector<double> &beta, std::vector<double> &zeta,
                std::vector<double> &zeta_old, int j_low, int n_shift) :
      r(r),
      p(p),
      x(x),
      alpha(alpha),
      beta(beta),
      zeta(zeta),
      zeta_old(zeta_old),
      j_low(j_low),
      n_shift(n_shift),
      n_update((r.Nspin() == 4) ? 4 : 2)
    {
    }

    void updateNshift(int new_n_shift) { n_shift = new_n_shift; }
    void updateNupdate(int new_n_update) { n_update = new_n_update; }

    // note that we can't set the stream parameter here so it is
    // ignored.  This is more of a future design direction to consider
    void apply(const qudaStream_t = device::get_default_stream())
    {
      static int count = 0;
      auto n_upper = std::min(((count + 1) * n_shift) / n_update + 1, n_shift);
      auto n_lower = (count * n_shift) / n_update + 1;

      for (int j = n_lower; j < n_upper; j++) {
        beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
      }

      if (n_upper > n_lower)
        blas::axpyBzpcx({alpha.begin() + n_lower, alpha.begin() + n_upper}, {p.begin() + n_lower, p.begin() + n_upper},
                        {x.begin() + n_lower, x.begin() + n_upper}, {zeta.begin() + n_lower, zeta.begin() + n_upper}, r,
                        {beta.begin() + n_lower, beta.begin() + n_upper});

      if (++count == n_update) count = 0;
    }
  };

  // this is the Worker pointer that the dslash uses to launch the shifted updates
  namespace dslash {
    extern Worker* aux_worker;
  }

  MultiShiftCG::MultiShiftCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, SolverParam &param,
                             TimeProfile &profile) :
    MultiShiftSolver(mat, matSloppy, param, profile), init(false)
  {
  }

  void MultiShiftCG::create(std::vector<ColorSpinorField> &x, const ColorSpinorField &b, std::vector<ColorSpinorField> &p)
  {
    if (!init) {
      profile.TPSTART(QUDA_PROFILE_INIT);
      MultiShiftSolver::create(x, b);
      num_offset = param.num_offset;

      reliable = false;
      for (int j = 0; j < num_offset; j++)
        if (param.tol_offset[j] < param.delta) reliable = true;

      r = b;

      ColorSpinorParam csParam(b);
      csParam.create = QUDA_NULL_FIELD_CREATE;

      mixed = param.precision_sloppy != param.precision;
      group_update = false; // do we want to do group updates regardless of precision?

      x_sloppy.resize(num_offset);
      if ((mixed && param.use_sloppy_partial_accumulator) || group_update) {
        if (param.use_sloppy_partial_accumulator) csParam.setPrecision(param.precision_sloppy);
        group_update = true;
      }
      for (int i = 0; i < num_offset; i++) {
        x_sloppy[i] = group_update ? ColorSpinorField(csParam) : x[i].create_alias(csParam);
        blas::zero(x_sloppy[i]);
      }

      csParam.setPrecision(param.precision_sloppy);
      csParam.field = &r;
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = mixed ? ColorSpinorField(csParam) : r.create_alias(csParam);

      p.resize(num_offset);
      for (auto &pi : p) pi = r_sloppy;

      csParam.create = QUDA_NULL_FIELD_CREATE;
      Ap = ColorSpinorField(csParam);

      profile.TPSTOP(QUDA_PROFILE_INIT);
    }
  }

  /**
     Compute the new values of alpha and zeta
   */
  void updateAlphaZeta(std::vector<double> &alpha, std::vector<double> &zeta, std::vector<double> &zeta_old,
                       const std::vector<double> &r2, const std::vector<double> &beta, double pAp, const double *offset,
                       const int nShift, const int j_low)
  {
    std::vector<double> alpha_old(alpha);

    alpha[0] = r2[0] / pAp;
    zeta[0] = 1.0;
    for (int j=1; j<nShift; j++) {
      double c0 = zeta[j] * zeta_old[j] * alpha_old[j_low];
      double c1 = alpha[j_low] * beta[j_low] * (zeta_old[j]-zeta[j]);
      double c2 = zeta_old[j] * alpha_old[j_low] * (1.0 + (offset[j] - offset[0]) * alpha[j_low]);

      zeta_old[j] = zeta[j];
      zeta[j] = (c1 + c2 != 0.0) ? c0 / (c1 + c2) : 0.0;
      alpha[j] = (zeta[j] != 0.0) ? alpha[j_low] * zeta[j] / zeta_old[j] : 0.0;
    }
  }

  void MultiShiftCG::operator()(std::vector<ColorSpinorField> &x, ColorSpinorField &b, std::vector<ColorSpinorField> &p,
                                std::vector<double> &r2_old_array)
  {
    pushOutputPrefix("MultiShiftCG: ");
    create(x, b, p);

    if (num_offset == 0) return;

    double *offset = param.offset;

    const double b2 = blas::norm2(b);
    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0) {
      warningQuda("inverting on zero-field source");
      for (int i = 0; i < num_offset; i++) {
        x[i] = b;
        param.true_res_offset[i] = 0.0;
        param.true_res_hq_offset[i] = 0.0;
      }
      return;
    }

    bool exit_early = false;
    // whether we will switch to refinement on unshifted system after other shifts have converged
    bool zero_refinement = param.precision_refinement_sloppy != param.precision;

    // this is the limit of precision possible
    const double sloppy_tol= param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon() :
      ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon() : pow(2.,-17));
    const double fine_tol = pow(10.,(-2*(int)b.Precision()+1));
    std::vector<double> prec_tol(num_offset);

    prec_tol[0] = mixed ? sloppy_tol : fine_tol;
    for (int i=1; i<num_offset; i++) {
      prec_tol[i] = std::min(sloppy_tol,std::max(fine_tol,sqrt(param.tol_offset[i]*sloppy_tol)));
    }

    std::vector<double> zeta(num_offset, 1.0);
    std::vector<double> zeta_old(num_offset, 1.0);
    std::vector<double> alpha(num_offset, 1.0);
    std::vector<double> beta(num_offset, 0.0);

    int j_low = 0;
    int num_offset_now = num_offset;

    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    // stopping condition of each shift
    std::vector<double> r2(num_offset, b2);
    std::vector<double> stop(num_offset);
    for (int i = 0; i < num_offset; i++) stop[i] = Solver::stopping(param.tol_offset[i], b2, param.residual_type);

    std::vector<int> iter(num_offset + 1, 0); // record how many iterations for each shift
    iter[num_offset] = 1;                     // this initial condition ensures that the heaviest shift can be removed

    double r2_old;
    double pAp;

    std::vector<double> rNorm(num_offset);
    for (int i = 0; i < num_offset; i++) rNorm[i] = sqrt(r2[i]);
    std::vector<double> r0Norm(rNorm);
    std::vector<double> maxrx(rNorm);
    std::vector<double> maxrr(rNorm);
    double delta = param.delta;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease =  param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    int resIncrease = 0;
    std::vector<int> resIncreaseTotal(num_offset, 0);

    int k = 0;
    int rUpdate = 0;
    blas::flops = 0;

    // now create the worker class for updating the shifted solutions and gradient vectors
    bool aux_update = false;
    ShiftUpdate shift_update(r_sloppy, p, x_sloppy, alpha, beta, zeta, zeta_old, j_low, num_offset_now);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);

    logQuda(QUDA_VERBOSE, "%d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0] / b2));

    while ( !convergence(r2, stop, num_offset_now) &&  !exit_early && k < param.maxiter) {

      if (aux_update) dslash::aux_worker = &shift_update;
      matSloppy(Ap, p[0]);
      dslash::aux_worker = nullptr;
      aux_update = false;

      // update number of shifts now instead of end of previous
      // iteration so that all shifts are updated during the dslash
      shift_update.updateNshift(num_offset_now);

      // at some point we should curry these into the Dirac operator
      if (r.Nspin() == 4)
        pAp = blas::axpyReDot(offset[0], p[0], Ap);
      else
        pAp = blas::reDotProduct(p[0], Ap);

      // compute zeta and alpha
      for (int j=1; j<num_offset_now; j++) r2_old_array[j] = zeta[j] * zeta[j] * r2[0];
      updateAlphaZeta(alpha, zeta, zeta_old, r2, beta, pAp, offset, num_offset_now, j_low);

      r2_old = r2[0];
      r2_old_array[0] = r2_old;

      auto cg_norm = blas::axpyCGNorm(-alpha[j_low], Ap, r_sloppy);
      r2[0] = cg_norm.x;
      double zn = cg_norm.y;

      // reliable update conditions
      rNorm[0] = sqrt(r2[0]);
      for (int j=1; j<num_offset_now; j++) rNorm[j] = rNorm[0] * zeta[j];

      int updateX=0, updateR=0;
      //fixme: with the current implementation of the reliable update it is sufficient to trigger it only for shift 0
      //fixme: The loop below is unnecessary but I don't want to delete it as we still might find a better reliable update
      int reliable_shift = -1; // this is the shift that sets the reliable_shift
      for (int j=0; j>=0; j--) {
        if (rNorm[j] > maxrx[j]) maxrx[j] = rNorm[j];
        if (rNorm[j] > maxrr[j]) maxrr[j] = rNorm[j];
        updateX = (rNorm[j] < delta*r0Norm[j] && r0Norm[j] <= maxrx[j]) ? 1 : updateX;
        updateR = ((rNorm[j] < delta*maxrr[j] && r0Norm[j] <= maxrr[j]) || updateX) ? 1 : updateR;
        if ((updateX || updateR) && reliable_shift == -1) reliable_shift = j;
      }

      if ( !(updateR || updateX) || !reliable) {
        // beta[0] = r2[0] / r2_old;
        beta[0] = zn / r2_old;
	// update p[0] and x[0]
        blas::axpyZpbx(alpha[0], p[0], x_sloppy[0], r_sloppy, beta[0]);

        // this should trigger the shift update in the subsequent sloppy dslash
	aux_update = true;
        /*
          for (int j=1; j<num_offset_now; j++) {
          beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
          // update p[i] and x[i]
          blas::axpyBzpcx(alpha[j], p[j], x_sloppy[j], zeta[j], r_sloppy, beta[j]);
          }
        */
      } else {
	for (int j=0; j<num_offset_now; j++) {
          blas::axpy(alpha[j], p[j], x_sloppy[j]);
          if (group_update) {
            if (rUpdate == 0)
              x[j] = x_sloppy[j];
            else
              blas::xpy(x_sloppy[j], x[j]);
          }
        }

        mat(r, x[0]);
        if (r.Nspin() == 4) blas::axpy(offset[0], x[0], r);

        r2[0] = blas::xmyNorm(b, r);
        for (int j = 1; j < num_offset_now; j++) r2[j] = zeta[j] * zeta[j] * r2[0];
        for (int j = 0; j < num_offset_now; j++)
          if (group_update) blas::zero(x_sloppy[j]);

        blas::copy(r_sloppy, r);

        // break-out check if we have reached the limit of the precision
	if (sqrt(r2[reliable_shift]) > r0Norm[reliable_shift]) { // reuse r0Norm for this
	  resIncrease++;
	  resIncreaseTotal[reliable_shift]++;
          warningQuda("Shift %d, updated residual %e is greater than previous residual %e (total #inc %i)",
                      reliable_shift, sqrt(r2[reliable_shift]), r0Norm[reliable_shift], resIncreaseTotal[reliable_shift]);

          if (resIncrease > maxResIncrease or resIncreaseTotal[reliable_shift] > maxResIncreaseTotal) {
            warningQuda("solver exiting due to too many true residual norm increases");
            break;
          }
	} else {
	  resIncrease = 0;
	}

	// explicitly restore the orthogonality of the gradient vector
	for (int j=0; j<num_offset_now; j++) {
          Complex rp = blas::cDotProduct(r_sloppy, p[j]) / (r2[0]);
          blas::caxpy(-rp, r_sloppy, p[j]);
        }

        // update beta and p
        beta[0] = r2[0] / r2_old;
        blas::xpay(r_sloppy, beta[0], p[0]);
        for (int j = 1; j < num_offset_now; j++) {
          beta[j] = beta[j_low] * zeta[j] * alpha[j] / (zeta_old[j] * alpha[j_low]);
          blas::axpby(zeta[j], r_sloppy, beta[j], p[j]);
        }

        // update reliable update parameters for the system that triggered the update
	int m = reliable_shift;
	rNorm[m] = sqrt(r2[0]) * zeta[m];
	maxrr[m] = rNorm[m];
	maxrx[m] = rNorm[m];
        r0Norm[m] = rNorm[m];
        rUpdate++;
      }

      // now we can check if any of the shifts have converged and remove them
      int converged = 0;
      for (int j=num_offset_now-1; j>=1; j--) {
        if (zeta[j] == 0.0 && r2[j+1] < stop[j+1]) {
          converged++;
          logQuda(QUDA_VERBOSE, "Shift %d converged after %d iterations\n", j, k + 1);
        } else {
	  r2[j] = zeta[j] * zeta[j] * r2[0];
	  // only remove if shift above has converged
	  if ((r2[j] < stop[j] || sqrt(r2[j] / b2) < prec_tol[j]) && iter[j+1] ) {
	    converged++;
	    iter[j] = k+1;
            logQuda(QUDA_VERBOSE, "Shift %d converged after %d iterations\n", j, k + 1);
          }
	}
      }
      num_offset_now -= converged;

      // exit early so that we can finish of shift 0 using CG and allowing for mixed precison refinement
      if ( (mixed || zero_refinement) and param.compute_true_res and num_offset_now==1) {
        exit_early=true;
        num_offset_now--;
      }

      k++;

      // this ensure we do the update on any shifted systems that
      // happen to converge when the un-shifted system converges
      if ( (convergence(r2, stop, num_offset_now) || exit_early || k == param.maxiter) && aux_update == true) {
        logQuda(QUDA_VERBOSE, "Convergence of unshifted system so trigger shiftUpdate\n");

        // set worker to do all updates at once
	shift_update.updateNupdate(1);
        shift_update.apply();

        for (int j = 0; j < num_offset_now; j++) iter[j] = k;
      }

      logQuda(QUDA_VERBOSE, "%d iterations, <r,r> = %e, |r|/|b| = %e\n", k, r2[0], sqrt(r2[0] / b2));
    }

    for (int i=0; i<num_offset; i++) {
      if (iter[i] == 0) iter[i] = k;
      if (group_update) blas::xpy(x_sloppy[i], x[i]);
    }

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    logQuda(QUDA_VERBOSE, "Reliable updates = %d\n", rUpdate);
    if (k==param.maxiter) warningQuda("Exceeded maximum iterations %d\n", param.maxiter);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (param.compute_true_res) {
      for (int i = 0; i < num_offset; i++) {
        // only calculate true residual if we need to:
        // 1.) For higher shifts if we did not use mixed precision
        // 2.) For shift 0 if we did not exit early  (we went to the full solution)
        if ( (i > 0 and not mixed) or (i == 0 and not exit_early) ) {
          mat(r, x[i]);
          if (r.Nspin() == 4) {
            blas::axpy(offset[i], x[i], r); // Offset it.
          } else if (i != 0) {
            blas::axpy(offset[i] - offset[0], x[i], r); // Offset it.
          }
          double true_res = blas::xmyNorm(b, r);
          param.true_res_offset[i] = sqrt(true_res / b2);
          param.true_res_hq_offset[i] = sqrt(blas::HeavyQuarkResidualNorm(x[i], r).z);
        } else {
          param.true_res_offset[i] = std::numeric_limits<double>::infinity();
          param.true_res_hq_offset[i] = std::numeric_limits<double>::infinity();
        }
        param.iter_res_offset[i] = sqrt(r2[i] / b2);
      }

      logQuda(QUDA_SUMMARIZE, "Converged after %d iterations\n", k);
      for (int i = 0; i < num_offset; i++) {
        if (std::isinf(param.true_res_offset[i])) {
          logQuda(QUDA_SUMMARIZE, " shift=%d, %d iterations, relative residual: iterated = %e\n", i, iter[i],
                  param.iter_res_offset[i]);
        } else {
          logQuda(QUDA_SUMMARIZE, " shift=%d, %d iterations, relative residual: iterated = %e, true = %e\n", i, iter[i],
                  param.iter_res_offset[i], param.true_res_offset[i]);
        }
      }

    } else {
      logQuda(QUDA_SUMMARIZE, "Converged after %d iterations\n", k);
      for (int i = 0; i < num_offset; i++) {
        param.iter_res_offset[i] = sqrt(r2[i] / b2);
        logQuda(QUDA_SUMMARIZE, " shift=%d, %d iterations, relative residual: iterated = %e\n", i, iter[i],
                param.iter_res_offset[i]);
      }
    }

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    popOutputPrefix();
  }

} // namespace quda
