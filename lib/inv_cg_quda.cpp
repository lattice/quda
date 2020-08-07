#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <iostream>

#ifdef BLOCKSOLVER
#include <Eigen/Dense>
#endif

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <eigensolve_quda.h>

namespace quda {

  CG::CG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, param, profile),
    yp(nullptr),
    rp(nullptr),
    rnewp(nullptr),
    pp(nullptr),
    App(nullptr),
    tmpp(nullptr),
    tmp2p(nullptr),
    tmp3p(nullptr),
    rSloppyp(nullptr),
    xSloppyp(nullptr),
    init(false)
  {
  }

  CG::~CG()
  {
    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_FREE);
    if ( init ) {
      for (auto pi : p) if (pi) delete pi;
      if (rp) delete rp;
      if (pp) delete pp;
      if (yp) delete yp;
      if (App) delete App;
      if (param.precision != param.precision_sloppy) {
        if (rSloppyp) delete rSloppyp;
        if (xSloppyp) delete xSloppyp;
      }
      if (tmpp) delete tmpp;
      if (!mat.isStaggered()) {
        if (tmp2p && tmpp != tmp2p) delete tmp2p;
        if (tmp3p && tmpp != tmp3p && param.precision != param.precision_sloppy) delete tmp3p;
      }
      if (rnewp) delete rnewp;
      init = false;

      destroyDeflationSpace();
    }
    if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_FREE);
  }

  CGNE::CGNE(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    CG(mmdag, mmdagSloppy, mmdagPrecon, param, profile),
    mmdag(mat.Expose()),
    mmdagSloppy(matSloppy.Expose()),
    mmdagPrecon(matPrecon.Expose()),
    xp(nullptr),
    yp(nullptr),
    init(false)
  {
  }

  CGNE::~CGNE() {
    if ( init ) {
      if (xp) delete xp;
      if (yp) delete yp;
      init = false;
    }
  }

  // CGNE: M Mdag y = b is solved; x = Mdag y is returned as solution.
  void CGNE::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    const int iter0 = param.iter;

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      xp = ColorSpinorField::Create(x, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(x, csParam);
      init = true;
    }

    double b2 = blas::norm2(b);

    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {

      // compute initial residual
      mmdag.Expose()->M(*xp,x);
      double r2 = blas::xmyNorm(b,*xp);
      if (b2 == 0.0) b2 = r2;

      // compute solution to residual equation
      CG::operator()(*yp,*xp);

      mmdag.Expose()->Mdag(*xp,*yp);

      // compute full solution
      blas::xpy(*xp, x);

    } else {

      CG::operator()(*yp,b);
      mmdag.Expose()->Mdag(x,*yp);

    }

    // future optimization: with preserve_source == QUDA_PRESERVE_SOURCE_NO; b is already
    // expected to be the CG residual which matches the CGNE residual
    // (but only with zero initial guess).  at the moment, CG does not respect this convention
    if (param.compute_true_res || param.preserve_source == QUDA_PRESERVE_SOURCE_NO) {

      // compute the true residual
      mmdag.Expose()->M(*xp, x);

      ColorSpinorField &A = param.preserve_source == QUDA_PRESERVE_SOURCE_YES ? b : *xp;
      ColorSpinorField &B = param.preserve_source == QUDA_PRESERVE_SOURCE_YES ? *xp : b;
      blas::axpby(-1.0, A, 1.0, B);

      double r2;
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        double3 h3 = blas::HeavyQuarkResidualNorm(x, B);
        r2 = h3.y;
        param.true_res_hq = sqrt(h3.z);
      } else {
        r2 = blas::norm2(B);
      }
      param.true_res = sqrt(r2 / b2);

      PrintSummary("CGNE", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);
    }

  }

  CGNR::CGNR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    CG(mdagm, mdagmSloppy, mdagmPrecon, param, profile),
    mdagm(mat.Expose()),
    mdagmSloppy(matSloppy.Expose()),
    mdagmPrecon(matPrecon.Expose()),
    bp(nullptr),
    init(false)
  {
  }

  CGNR::~CGNR() {
    if ( init ) {
      if (bp) delete bp;
      init = false;
    }
  }

  // CGNR: Mdag M x = Mdag b is solved.
  void CGNR::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    const int iter0 = param.iter;

    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      bp = ColorSpinorField::Create(csParam);
      init = true;
    }

    double b2 = blas::norm2(b);
    if (b2 == 0.0) { // compute initial residual vector
      mdagm.Expose()->M(*bp,x);
      b2 = blas::norm2(*bp);
    }

    mdagm.Expose()->Mdag(*bp,b);
    CG::operator()(x,*bp);

    if ( param.compute_true_res || param.preserve_source == QUDA_PRESERVE_SOURCE_NO ) {

      // compute the true residual
      mdagm.Expose()->M(*bp, x);

      ColorSpinorField &A = param.preserve_source == QUDA_PRESERVE_SOURCE_YES ? b : *bp;
      ColorSpinorField &B = param.preserve_source == QUDA_PRESERVE_SOURCE_YES ? *bp : b;
      blas::axpby(-1.0, A, 1.0, B);

      double r2;
      if (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) {
        double3 h3 = blas::HeavyQuarkResidualNorm(x, B);
        r2 = h3.y;
        param.true_res_hq = sqrt(h3.z);
      } else {
        r2 = blas::norm2(B);
      }
      param.true_res = sqrt(r2 / b2);
      PrintSummary("CGNR", param.iter - iter0, r2, b2, stopping(param.tol, b2, param.residual_type), param.tol_hq);

    } else if (param.preserve_source == QUDA_PRESERVE_SOURCE_NO) {
      mdagm.Expose()->M(*bp, x);
      blas::axpby(-1.0, *bp, 1.0, b);
    }

  }

  void CG::operator()(ColorSpinorField &x, ColorSpinorField &b, ColorSpinorField *p_init, double r2_old_init)
  {
    if (param.is_preconditioner && param.global_reduction == false) commGlobalReductionSet(false);

    if (checkLocation(x, b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");
    if (checkPrecision(x, b) != param.precision)
      errorQuda("Precision mismatch: expected=%d, received=%d", param.precision, x.Precision());

    if (param.maxiter == 0 || param.Nsteps == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    const int Np = (param.solution_accumulator_pipeline == 0 ? 1 : param.solution_accumulator_pipeline);
    if (Np < 0 || Np > 16) errorQuda("Invalid value %d for solution_accumulator_pipeline\n", Np);

    // whether to select alternative reliable updates
    bool alternative_reliable = param.use_alternative_reliable;

    if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_INIT);

    double b2 = blas::norm2(b);

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0 && param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
      if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      rp = ColorSpinorField::Create(csParam);
      yp = ColorSpinorField::Create(csParam);

      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      App = ColorSpinorField::Create(csParam);
      if(param.precision != param.precision_sloppy) {
	rSloppyp = ColorSpinorField::Create(csParam);
	xSloppyp = ColorSpinorField::Create(csParam);
      } else {
	rSloppyp = rp;
	param.use_sloppy_partial_accumulator = false;
      }

      // temporary fields
      tmpp = ColorSpinorField::Create(csParam);
      if(!mat.isStaggered()) {
	// tmp2 only needed for multi-gpu Wilson-like kernels
	tmp2p = ColorSpinorField::Create(csParam);
	// additional high-precision temporary if Wilson and mixed-precision
	csParam.setPrecision(param.precision);
	tmp3p = (param.precision != param.precision_sloppy) ?
	  ColorSpinorField::Create(csParam) : tmpp;
      } else {
	tmp3p = tmp2p = tmpp;
      }

      init = true;
    }

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      constructDeflationSpace(b, matPrecon);
      if (deflate_compute) {
        // compute the deflation space.
        if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_INIT);
        (*eig_solve)(evecs, evals);
        if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_INIT);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(matPrecon, evecs, evals);
        recompute_evals = false;
      }
    }

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &Ap = *App;
    ColorSpinorField &tmp = *tmpp;
    ColorSpinorField &tmp2 = *tmp2p;
    ColorSpinorField &tmp3 = *tmp3p;
    ColorSpinorField &rSloppy = *rSloppyp;
    ColorSpinorField &xSloppy = param.use_sloppy_partial_accumulator ? *xSloppyp : x;

    {
      ColorSpinorParam csParam(r);
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);

      if (Np != (int)p.size()) {
	for (auto &pi : p) delete pi;
	p.resize(Np);
	for (auto &pi : p) pi = ColorSpinorField::Create(csParam);
      }
    }

    // alternative reliable updates
    // alternative reliable updates - set precision - does not hurt performance here

    const double u = param.precision_sloppy == 8 ?
      std::numeric_limits<double>::epsilon() / 2. :
      param.precision_sloppy == 4 ? std::numeric_limits<float>::epsilon() / 2. :
                                    param.precision_sloppy == 2 ? pow(2., -13) : pow(2., -6);
    const double uhigh = param.precision == 8 ? std::numeric_limits<double>::epsilon() / 2. :
                                                param.precision == 4 ? std::numeric_limits<float>::epsilon() / 2. :
                                                                       param.precision == 2 ? pow(2., -13) : pow(2., -6);
    const double deps=sqrt(u);
    constexpr double dfac = 1.1;
    double d_new = 0;
    double d = 0;
    double dinit = 0;
    double xNorm = 0;
    double xnorm = 0;
    double pnorm = 0;
    double ppnorm = 0;
    double Anorm = 0;
    double beta = 0.0;

    // for alternative reliable updates
    if (alternative_reliable) {
      // estimate norm for reliable updates
      mat(r, b, y, tmp3);
      Anorm = sqrt(blas::norm2(r)/b2);
    }

    // compute initial residual
    double r2 = 0.0;
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x, y, tmp3);
      r2 = blas::xmyNorm(b, r);
      if (b2 == 0) b2 = r2;
      // y contains the original guess.
      blas::copy(y, x);
    } else {
      if (&r != &b) blas::copy(r, b);
      r2 = b2;
      blas::zero(y);
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate and accumulate to solution vector
      eig_solve->deflate(y, r, evecs, evals, true);
      mat(r, y, x, tmp3);
      r2 = blas::xmyNorm(b, r);
    }

    blas::zero(x);
    if (&x != &xSloppy) blas::zero(xSloppy);
    blas::copy(rSloppy,r);

    if (Np != (int)p.size()) {
      for (auto &pi : p) delete pi;
      p.resize(Np);
      ColorSpinorParam csParam(rSloppy);
      csParam.create = QUDA_COPY_FIELD_CREATE;
      for (auto &pi : p)
        pi = p_init ? ColorSpinorField::Create(*p_init, csParam) : ColorSpinorField::Create(rSloppy, csParam);
    } else {
      for (auto &p_i : p) *p_i = p_init ? *p_init : rSloppy;
    }

    double r2_old=0.0;
    if (r2_old_init != 0.0 and p_init) {
      r2_old = r2_old_init;
      Complex rp = blas::cDotProduct(rSloppy, *p[0]) / (r2);
      blas::caxpy(-rp, rSloppy, *p[0]);
      beta = r2 / r2_old;
      blas::xpayz(rSloppy, beta, *p[0], *p[0]);
    }

    const bool use_heavy_quark_res =
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    bool heavy_quark_restart = false;

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      profile.TPSTART(QUDA_PROFILE_PREAMBLE);
    }

    double stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver

    double heavy_quark_res = 0.0;  // heavy quark res idual
    double heavy_quark_res_old = 0.0;  // heavy quark residual

    if (use_heavy_quark_res) {
      heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
      heavy_quark_res_old = heavy_quark_res;   // heavy quark residual
    }
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double alpha[Np];
    double pAp;
    int rUpdate = 0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double maxr_deflate = rNorm; // The maximum residual since the last deflation
    double delta = param.delta;

    // this parameter determines how many consective reliable update
    // residual increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = param.max_res_increase; //  check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    // this means when using heavy quarks we will switch to simple hq restarts as soon as the reliable strategy fails
    const int hqmaxresIncrease = param.max_hq_res_increase;
    const int hqmaxresRestartTotal
      = param.max_hq_res_restart_total; // this limits the number of heavy quark restarts we can do

    int resIncrease = 0;
    int resIncreaseTotal = 0;
    int hqresIncrease = 0;
    int hqresRestartTotal = 0;

    // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
    // only used if we use the heavy_quark_res
    bool L2breakdown = false;
    const double L2breakdown_eps = 100. * uhigh;

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
      profile.TPSTART(QUDA_PROFILE_COMPUTE);
      blas::flops = 0;
    }

    int k = 0;
    int j = 0;

    PrintStats("CG", k, r2, b2, heavy_quark_res);

    int steps_since_reliable = 1;
    bool converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

    // alternative reliable updates
    if(alternative_reliable){
      dinit = uhigh * (rNorm + Anorm * xNorm);
      d = dinit;
    }

    while ( !converged && k < param.maxiter ) {
      matSloppy(Ap, *p[j], tmp, tmp2);  // tmp as tmp
      double sigma;

      bool breakdown = false;
      if (param.pipeline) {
        double Ap2;
        //TODO: alternative reliable updates - need r2, Ap2, pAp, p norm
        if(alternative_reliable){
          double4 quadruple = blas::quadrupleCGReduction(rSloppy, Ap, *p[j]);
          r2 = quadruple.x; Ap2 = quadruple.y; pAp = quadruple.z; ppnorm= quadruple.w;
        }
        else{
          double3 triplet = blas::tripleCGReduction(rSloppy, Ap, *p[j]);
          r2 = triplet.x; Ap2 = triplet.y; pAp = triplet.z;
        }
        r2_old = r2;
        alpha[j] = r2 / pAp;
        sigma = alpha[j]*(alpha[j] * Ap2 - pAp);
        if (sigma < 0.0 || steps_since_reliable == 0) { // sigma condition has broken down
          r2 = blas::axpyNorm(-alpha[j], Ap, rSloppy);
          sigma = r2;
          breakdown = true;
        }

        r2 = sigma;
      } else {
        r2_old = r2;

        // alternative reliable updates,
        if (alternative_reliable) {
          double3 pAppp = blas::cDotProductNormA(*p[j],Ap);
          pAp = pAppp.x;
          ppnorm = pAppp.z;
        } else {
          pAp = blas::reDotProduct(*p[j], Ap);
        }

        alpha[j] = r2 / pAp;

        // here we are deploying the alternative beta computation
        Complex cg_norm = blas::axpyCGNorm(-alpha[j], Ap, rSloppy);
        r2 = real(cg_norm);  // (r_new, r_new)
        sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2;  // use r2 if (r_k+1, r_k+1-r_k) breaks
      }

      // reliable update conditions
      rNorm = sqrt(r2);
      int updateX;
      int updateR;

      if (alternative_reliable) {
        // alternative reliable updates
        updateX = ( (d <= deps*sqrt(r2_old)) or (dfac * dinit > deps * r0Norm) ) and (d_new > deps*rNorm) and (d_new > dfac * dinit);
        updateR = 0;
      } else {
        if (rNorm > maxrx) maxrx = rNorm;
        if (rNorm > maxrr) maxrr = rNorm;
        updateX = (rNorm < delta * r0Norm && r0Norm <= maxrx) ? 1 : 0;
        updateR = ((rNorm < delta * maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
      }

      // force a reliable update if we are within target tolerance (only if doing reliable updates)
      if ( convergence(r2, heavy_quark_res, stop, param.tol_hq) && param.delta >= param.tol ) updateX = 1;

      // For heavy-quark inversion force a reliable update if we continue after
      if ( use_heavy_quark_res and L2breakdown and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq) and param.delta >= param.tol ) {
        updateX = 1;
      }

      if ( !(updateR || updateX )) {
        beta = sigma / r2_old;  // use the alternative beta computation

        if (param.pipeline && !breakdown) {

	  if (Np == 1) {
	    blas::tripleCGUpdate(alpha[j], beta, Ap, xSloppy, rSloppy, *p[j]);
	  } else {
	    errorQuda("Not implemented pipelined CG with Np > 1");
	  }
	} else {
	  if (Np == 1) {
	    // with Np=1 we just run regular fusion between x and p updates
	    blas::axpyZpbx(alpha[k%Np], *p[k%Np], xSloppy, rSloppy, beta);
	  } else {

	    if ( (j+1)%Np == 0 ) {
	      std::vector<ColorSpinorField*> x_;
	      x_.push_back(&xSloppy);
              blas::axpy(alpha, p, x_);
            }

            // p[(k+1)%Np] = r + beta * p[k%Np]
            blas::xpayz(rSloppy, beta, *p[j], *p[(j + 1) % Np]);
          }
        }

        if (use_heavy_quark_res && k % heavy_quark_check == 0) {
          if (&x != &xSloppy) {
	    blas::copy(tmp,y);
	    heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy, tmp, rSloppy).z);
	  } else {
	    blas::copy(r, rSloppy);
	    heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(x, y, r).z);
	  }
        }

        // alternative reliable updates
	if (alternative_reliable) {
	  d = d_new;
	  pnorm = pnorm + alpha[j] * alpha[j]* (ppnorm);
	  xnorm = sqrt(pnorm);
	  d_new = d + u*rNorm + uhigh*Anorm * xnorm;
	  if (steps_since_reliable==0 && getVerbosity() >= QUDA_DEBUG_VERBOSE)
            printfQuda("New dnew: %e (r %e , y %e)\n",d_new,u*rNorm,uhigh*Anorm * sqrt(blas::norm2(y)) );
	}
	steps_since_reliable++;

      } else {

	{
	  std::vector<ColorSpinorField*> x_;
	  x_.push_back(&xSloppy);
	  std::vector<ColorSpinorField*> p_;
	  for (int i=0; i<=j; i++) p_.push_back(p[i]);
          blas::axpy(alpha, p_, x_);
        }

        blas::copy(x, xSloppy); // nop when these pointers alias

        blas::xpy(x, y); // swap these around?
        mat(r, y, x, tmp3); //  here we can use x as tmp
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y, x, tmp3);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2);
        }

        blas::copy(rSloppy, r); //nop when these pointers alias
        blas::zero(xSloppy);

        // alternative reliable updates
        if (alternative_reliable) {
          dinit = uhigh*(sqrt(r2) + Anorm * sqrt(blas::norm2(y)));
          d = d_new;
          xnorm = 0;//sqrt(norm2(x));
          pnorm = 0;//pnorm + alpha * sqrt(norm2(p));
          if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printfQuda("New dinit: %e (r %e , y %e)\n",dinit,uhigh*sqrt(r2),uhigh*Anorm*sqrt(blas::norm2(y)));
          d_new = dinit;
        } else {
          rNorm = sqrt(r2);
          maxrr = rNorm;
          maxrx = rNorm;
        }

        // calculate new reliable HQ resididual
        if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(y, r).z);

        // break-out check if we have reached the limit of the precision
        if (sqrt(r2) > r0Norm && updateX and not L2breakdown) { // reuse r0Norm for this
          resIncrease++;
          resIncreaseTotal++;
          warningQuda(
            "CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2), r0Norm, resIncreaseTotal);

          if ((use_heavy_quark_res and sqrt(r2) < L2breakdown_eps) or resIncrease > maxResIncrease
              or resIncreaseTotal > maxResIncreaseTotal or r2 < stop) {
            if (use_heavy_quark_res) {
              L2breakdown = true;
              warningQuda("CG: L2 breakdown %e, %e", sqrt(r2), L2breakdown_eps);
            } else {
              if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal or r2 < stop) {
                warningQuda("CG: solver exiting due to too many true residual norm increases");
                break;
              }
            }
          }
        } else {
          resIncrease = 0;
        }

        // if L2 broke down already we turn off reliable updates and restart the CG
        if (use_heavy_quark_res and L2breakdown) {
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
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG: solver exiting due to too many heavy quark residual norm increases (%i/%i)",
                          hqresIncrease, hqmaxresIncrease);
              break;
            }
          } else {
            hqresIncrease = 0;
          }

          if (hqresRestartTotal > hqmaxresRestartTotal) {
            warningQuda("CG: solver exiting due to too many heavy quark residual restarts (%i/%i)", hqresRestartTotal,
                        hqmaxresRestartTotal);
            break;
          }
        }

        if (use_heavy_quark_res and heavy_quark_restart) {
          // perform a restart
          blas::copy(*p[0], rSloppy);
          heavy_quark_restart = false;
        } else {
          // explicitly restore the orthogonality of the gradient vector
          Complex rp = blas::cDotProduct(rSloppy, *p[j]) / (r2);
          blas::caxpy(-rp, rSloppy, *p[j]);

          beta = r2 / r2_old;
          blas::xpayz(rSloppy, beta, *p[j], *p[0]);
        }

        steps_since_reliable = 0;
        r0Norm = sqrt(r2);
        rUpdate++;

        heavy_quark_res_old = heavy_quark_res;
      }

      breakdown = false;
      k++;

      PrintStats("CG", k, r2, b2, heavy_quark_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged = convergence(r2, heavy_quark_res, stop, param.tol_hq);

      // check for recent enough reliable updates of the HQ residual if we use it
      if (use_heavy_quark_res) {
        // L2 is converged or precision maxed out for L2
        bool L2done = L2breakdown or convergenceL2(r2, heavy_quark_res, stop, param.tol_hq);
        // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
        bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq);
        converged = L2done and HQdone;
      }

      // if we have converged and need to update any trailing solutions
      if (converged && steps_since_reliable > 0 && (j+1)%Np != 0 ) {
	std::vector<ColorSpinorField*> x_;
	x_.push_back(&xSloppy);
	std::vector<ColorSpinorField*> p_;
	for (int i=0; i<=j; i++) p_.push_back(p[i]);
        blas::axpy(alpha, p_, x_);
      }

      j = steps_since_reliable == 0 ? 0 : (j+1)%Np; // if just done a reliable update then reset j
    }

    blas::copy(x, xSloppy);
    blas::xpy(y, x);

    if (!param.is_preconditioner) {
      profile.TPSTOP(QUDA_PROFILE_COMPUTE);
      profile.TPSTART(QUDA_PROFILE_EPILOGUE);

      param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
      double gflops = (blas::flops + mat.flops() + matSloppy.flops() + matPrecon.flops()) * 1e-9;
      param.gflops = gflops;
      param.iter += k;

      if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);
    }

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);

    if (param.compute_true_res) {
      // compute the true residuals
      mat(r, x, y, tmp3);
      param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
    }

    PrintSummary("CG", k, r2, b2, stop, param.tol_hq);

    if (!param.is_preconditioner) {
      // reset the flops counters
      blas::flops = 0;
      mat.flops();
      matSloppy.flops();
      matPrecon.flops();

      profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    }

    if (param.is_preconditioner && param.global_reduction == false) commGlobalReductionSet(true);
  }

// use BlockCGrQ algortithm or BlockCG (with / without GS, see BLOCKCG_GS option)
#define BCGRQ 1
#if BCGRQ
void CG::blocksolve(ColorSpinorField& x, ColorSpinorField& b) {
  #ifndef BLOCKSOLVER
  errorQuda("QUDA_BLOCKSOLVER not built.");
  #else

  if (checkLocation(x, b) != QUDA_CUDA_FIELD_LOCATION)
  errorQuda("Not supported");

  profile.TPSTART(QUDA_PROFILE_INIT);

  using Eigen::MatrixXcd;

  // Check to see that we're not trying to invert on a zero-field source
  //MW: it might be useful to check what to do here.
  double b2[QUDA_MAX_MULTI_SHIFT];
  double b2avg=0;
  for(int i=0; i< param.num_src; i++){
    b2[i]=blas::norm2(b.Component(i));
    b2avg += b2[i];
    if(b2[i] == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      errorQuda("Warning: inverting on zero-field source - undefined for block solver\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }
  }

  b2avg = b2avg / param.num_src;

  ColorSpinorParam csParam(x);
  if (!init) {
    csParam.setPrecision(param.precision);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    rp = ColorSpinorField::Create(csParam);
    yp = ColorSpinorField::Create(csParam);

    // sloppy fields
    csParam.setPrecision(param.precision_sloppy);
    pp = ColorSpinorField::Create(csParam);
    App = ColorSpinorField::Create(csParam);
    if(param.precision != param.precision_sloppy) {
      rSloppyp = ColorSpinorField::Create(csParam);
      xSloppyp = ColorSpinorField::Create(csParam);
    } else {
      rSloppyp = rp;
      param.use_sloppy_partial_accumulator = false;
    }

    // temporary fields
    tmpp = ColorSpinorField::Create(csParam);
    if(!mat.isStaggered()) {
      // tmp2 only needed for multi-gpu Wilson-like kernels
      tmp2p = ColorSpinorField::Create(csParam);
      // additional high-precision temporary if Wilson and mixed-precision
      csParam.setPrecision(param.precision);
      tmp3p = (param.precision != param.precision_sloppy) ?
	ColorSpinorField::Create(csParam) : tmpp;
    } else {
      tmp3p = tmp2p = tmpp;
    }

    init = true;
  }

  if(!rnewp) {
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    // ColorSpinorField *rpnew = ColorSpinorField::Create(csParam);
  }

  ColorSpinorField &r = *rp;
  ColorSpinorField &y = *yp;
  ColorSpinorField &p = *pp;
  ColorSpinorField &Ap = *App;
  ColorSpinorField &rnew = *rnewp;
  ColorSpinorField &tmp = *tmpp;
  ColorSpinorField &tmp2 = *tmp2p;
  ColorSpinorField &tmp3 = *tmp3p;
  ColorSpinorField &rSloppy = *rSloppyp;
  ColorSpinorField &xSloppy = param.use_sloppy_partial_accumulator ? *xSloppyp : x;

  // calculate residuals for all vectors
  // and initialize r2 matrix
  double r2avg=0;
  MatrixXcd r2(param.num_src, param.num_src);
  for(int i=0; i<param.num_src; i++){
    mat(r.Component(i), x.Component(i), y.Component(i));
    r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
    r2avg += r2(i,i).real();
    printfQuda("r2[%i] %e\n", i, r2(i,i).real());
  }
  for(int i=0; i<param.num_src; i++){
    for(int j=i+1; j < param.num_src; j++){
      r2(i,j) = blas::cDotProduct(r.Component(i),r.Component(j));
      r2(j,i) = std::conj(r2(i,j));
    }
  }

  blas::copy(rSloppy, r);
  blas::copy(p, rSloppy);
  blas::copy(rnew, rSloppy);

  if (&x != &xSloppy) {
    blas::copy(y, x);
    blas::zero(xSloppy);
  } else {
    blas::zero(y);
  }

  const bool use_heavy_quark_res =
  (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
  if(use_heavy_quark_res) errorQuda("ERROR: heavy quark residual not supported in block solver");

  profile.TPSTOP(QUDA_PROFILE_INIT);
  profile.TPSTART(QUDA_PROFILE_PREAMBLE);

  double stop[QUDA_MAX_MULTI_SHIFT];

  for(int i = 0; i < param.num_src; i++){
    stop[i] = stopping(param.tol, b2[i], param.residual_type);  // stopping condition of solver
  }

  // Eigen Matrices instead of scalars
  MatrixXcd alpha = MatrixXcd::Zero(param.num_src,param.num_src);
  MatrixXcd beta = MatrixXcd::Zero(param.num_src,param.num_src);
  MatrixXcd C = MatrixXcd::Zero(param.num_src,param.num_src);
  MatrixXcd S = MatrixXcd::Identity(param.num_src,param.num_src);
  MatrixXcd pAp = MatrixXcd::Identity(param.num_src,param.num_src);
  quda::Complex * AC = new quda::Complex[param.num_src*param.num_src];

  #ifdef MWVERBOSE
  MatrixXcd pTp =  MatrixXcd::Identity(param.num_src,param.num_src);
  #endif




  //FIXME:reliable updates currently not implemented
  /*
  double rNorm[QUDA_MAX_MULTI_SHIFT];
  double r0Norm[QUDA_MAX_MULTI_SHIFT];
  double maxrx[QUDA_MAX_MULTI_SHIFT];
  double maxrr[QUDA_MAX_MULTI_SHIFT];

  for(int i = 0; i < param.num_src; i++){
    rNorm[i] = sqrt(r2(i,i).real());
    r0Norm[i] = rNorm[i];
    maxrx[i] = rNorm[i];
    maxrr[i] = rNorm[i];
  }
  bool L2breakdown = false;
  int rUpdate = 0;
  nt steps_since_reliable = 1;
  */

  profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
  profile.TPSTART(QUDA_PROFILE_COMPUTE);
  blas::flops = 0;

  int k = 0;

  PrintStats("CG", k, r2avg / param.num_src, b2avg, 0.);
  bool allconverged = true;
  bool converged[QUDA_MAX_MULTI_SHIFT];
  for(int i=0; i<param.num_src; i++){
    converged[i] = convergence(r2(i,i).real(), 0., stop[i], param.tol_hq);
    allconverged = allconverged && converged[i];
  }

  // CHolesky decomposition
  MatrixXcd L = r2.llt().matrixL();//// retrieve factor L  in the decomposition
  C = L.adjoint();
  MatrixXcd Linv = C.inverse();

  #ifdef MWVERBOSE
  std::cout << "r2\n " << r2 << std::endl;
  std::cout << "L\n " << L.adjoint() << std::endl;
  #endif

  // set p to QR decompsition of r
  // temporary hack - use AC to pass matrix arguments to multiblas
  for(int i=0; i<param.num_src; i++){
    blas::zero(p.Component(i));
    for(int j=0;j<param.num_src; j++){
      AC[i*param.num_src + j] = Linv(i,j);
    }
  }
  blas::caxpy(AC,r,p);

  // set rsloppy to to QR decompoistion of r (p)
  for(int i=0; i< param.num_src; i++){
    blas::copy(rSloppy.Component(i), p.Component(i));
  }

  #ifdef MWVERBOSE
  for(int i=0; i<param.num_src; i++){
    for(int j=0; j<param.num_src; j++){
      pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
    }
  }
  std::cout << " pTp  " << std::endl << pTp << std::endl;
  std::cout << " L " << std::endl << L.adjoint() << std::endl;
  std::cout << " C " << std::endl << C << std::endl;
  #endif

  while ( !allconverged && k < param.maxiter ) {
    // apply matrix
    for(int i=0; i<param.num_src; i++){
      matSloppy(Ap.Component(i), p.Component(i), tmp.Component(i), tmp2.Component(i));  // tmp as tmp
    }

    // calculate pAp
    for(int i=0; i<param.num_src; i++){
      for(int j=i; j < param.num_src; j++){
        pAp(i,j) = blas::cDotProduct(p.Component(i), Ap.Component(j));
        if (i!=j) pAp(j,i) = std::conj(pAp(i,j));
      }
    }

    // update Xsloppy
    alpha = pAp.inverse() * C;
    // temporary hack using AC
    for(int i=0; i<param.num_src; i++){
      for(int j=0;j<param.num_src; j++){
        AC[i*param.num_src + j] = alpha(i,j);
      }
    }
    blas::caxpy(AC,p,xSloppy);

    // update rSloppy
    beta = pAp.inverse();
    // temporary hack
    for(int i=0; i<param.num_src; i++){
      for(int j=0;j<param.num_src; j++){
        AC[i*param.num_src + j] = -beta(i,j);
      }
    }
    blas::caxpy(AC,Ap,rSloppy);

    // orthorgonalize R
    // copy rSloppy to rnew as temporary
    for(int i=0; i< param.num_src; i++){
      blas::copy(rnew.Component(i), rSloppy.Component(i));
    }
    for(int i=0; i<param.num_src; i++){
      for(int j=i; j < param.num_src; j++){
        r2(i,j) = blas::cDotProduct(r.Component(i),r.Component(j));
        if (i!=j) r2(j,i) = std::conj(r2(i,j));
      }
    }
    // Cholesky decomposition
    L = r2.llt().matrixL();// retrieve factor L  in the decomposition
    S = L.adjoint();
    Linv = S.inverse();
    // temporary hack
    for(int i=0; i<param.num_src; i++){
      blas::zero(rSloppy.Component(i));
      for(int j=0;j<param.num_src; j++){
        AC[i*param.num_src + j] = Linv(i,j);
      }
    }
    blas::caxpy(AC,rnew,rSloppy);

    #ifdef MWVERBOSE
    for(int i=0; i<param.num_src; i++){
      for(int j=0; j<param.num_src; j++){
        pTp(i,j) = blas::cDotProduct(rSloppy.Component(i), rSloppy.Component(j));
      }
    }
    std::cout << " rTr " << std::endl << pTp << std::endl;
    std::cout <<  "QR" << S<<  std::endl << "QP " << S.inverse()*S << std::endl;;
    #endif

    // update p
    // use rnew as temporary again for summing up
    for(int i=0; i<param.num_src; i++){
      blas::copy(rnew.Component(i),rSloppy.Component(i));
    }
    // temporary hack
    for(int i=0; i<param.num_src; i++){
      for(int j=0;j<param.num_src; j++){
        AC[i*param.num_src + j] = std::conj(S(j,i));
      }
    }
    blas::caxpy(AC,p,rnew);
    // set p = rnew
    for(int i=0; i < param.num_src; i++){
      blas::copy(p.Component(i),rnew.Component(i));
    }

    // update C
    C = S * C;

    #ifdef MWVERBOSE
    for(int i=0; i<param.num_src; i++){
      for(int j=0; j<param.num_src; j++){
        pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
      }
    }
    std::cout << " pTp " << std::endl << pTp << std::endl;
    std::cout <<  "S " << S<<  std::endl << "C " << C << std::endl;
    #endif

    // calculate the residuals for all shifts
    r2avg=0;
    for (int j=0; j<param.num_src; j++ ){
      r2(j,j) = C(0,j)*conj(C(0,j));
      for(int i=1; i < param.num_src; i++)
      r2(j,j) += C(i,j) * conj(C(i,j));
      r2avg += r2(j,j).real();
    }

    k++;
    PrintStats("CG", k, r2avg / param.num_src, b2avg, 0);
    // check convergence
    allconverged = true;
    for(int i=0; i<param.num_src; i++){
      converged[i] = convergence(r2(i,i).real(), 0, stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }


  }

  for(int i=0; i<param.num_src; i++){
    blas::xpy(y.Component(i), xSloppy.Component(i));
  }

  profile.TPSTOP(QUDA_PROFILE_COMPUTE);
  profile.TPSTART(QUDA_PROFILE_EPILOGUE);

  param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
  double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
  param.gflops = gflops;
  param.iter += k;

  if (k == param.maxiter)
  warningQuda("Exceeded maximum iterations %d", param.maxiter);

  // if (getVerbosity() >= QUDA_VERBOSE)
  // printfQuda("CG: Reliable updates = %d\n", rUpdate);

  // compute the true residuals
  for(int i=0; i<param.num_src; i++){
    mat(r.Component(i), x.Component(i), y.Component(i), tmp3.Component(i));
    param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
    param.true_res_offset[i] = param.true_res;
    param.true_res_hq_offset[i] = param.true_res_hq;

    PrintSummary("CG", k, r2(i,i).real(), b2[i], stop[i], 0.0);
  }

  // reset the flops counters
  blas::flops = 0;
  mat.flops();
  matSloppy.flops();

  profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
  profile.TPSTART(QUDA_PROFILE_FREE);

  delete[] AC;
  profile.TPSTOP(QUDA_PROFILE_FREE);

  return;

  #endif
}

#else

// use Gram Schmidt in Block CG ?
#define BLOCKCG_GS 1
void CG::solve(ColorSpinorField& x, ColorSpinorField& b) {
  #ifndef BLOCKSOLVER
  errorQuda("QUDA_BLOCKSOLVER not built.");
  #else
  #ifdef BLOCKCG_GS
  printfQuda("BCGdQ Solver\n");
  #else
  printfQuda("BCQ Solver\n");
  #endif
  const bool use_block = true;
  if (checkLocation(x, b) != QUDA_CUDA_FIELD_LOCATION)
  errorQuda("Not supported");

  profile.TPSTART(QUDA_PROFILE_INIT);

  using Eigen::MatrixXcd;
  MatrixXcd mPAP(param.num_src,param.num_src);
  MatrixXcd mRR(param.num_src,param.num_src);


  // Check to see that we're not trying to invert on a zero-field source
  //MW: it might be useful to check what to do here.
  double b2[QUDA_MAX_MULTI_SHIFT];
  double b2avg=0;
  double r2avg=0;
  for(int i=0; i< param.num_src; i++){
    b2[i]=blas::norm2(b.Component(i));
    b2avg += b2[i];
    if(b2[i] == 0){
      profile.TPSTOP(QUDA_PROFILE_INIT);
      errorQuda("Warning: inverting on zero-field source\n");
      x=b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }
  }

  #ifdef MWVERBOSE
  MatrixXcd b2m(param.num_src,param.num_src);
  // just to check details of b
  for(int i=0; i<param.num_src; i++){
    for(int j=0; j<param.num_src; j++){
      b2m(i,j) = blas::cDotProduct(b.Component(i), b.Component(j));
    }
  }
  std::cout << "b2m\n" <<  b2m << std::endl;
  #endif

  ColorSpinorParam csParam(x);
  if (!init) {
    csParam.setPrecision(param.precision);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    rp = ColorSpinorField::Create(csParam);
    yp = ColorSpinorField::Create(csParam);

    // sloppy fields
    csParam.setPrecision(param.precision_sloppy);
    pp = ColorSpinorField::Create(csParam);
    App = ColorSpinorField::Create(csParam);
    if(param.precision != param.precision_sloppy) {
      rSloppyp = ColorSpinorField::Create(csParam);
      xSloppyp = ColorSpinorField::Create(csParam);
    } else {
      rSloppyp = rp;
      param.use_sloppy_partial_accumulator = false;
    }

    // temporary fields
    tmpp = ColorSpinorField::Create(csParam);
    if(!mat.isStaggered()) {
      // tmp2 only needed for multi-gpu Wilson-like kernels
      tmp2p = ColorSpinorField::Create(csParam);
      // additional high-precision temporary if Wilson and mixed-precision
      csParam.setPrecision(param.precision);
      tmp3p = (param.precision != param.precision_sloppy) ?
	ColorSpinorField::Create(csParam) : tmpp;
    } else {
      tmp3p = tmp2p = tmpp;
    }

    init = true;
  }

  if(!rnewp) {
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);
    // ColorSpinorField *rpnew = ColorSpinorField::Create(csParam);
  }

  ColorSpinorField &r = *rp;
  ColorSpinorField &y = *yp;
  ColorSpinorField &p = *pp;
  ColorSpinorField &pnew = *rnewp;
  ColorSpinorField &Ap = *App;
  ColorSpinorField &tmp = *tmpp;
  ColorSpinorField &tmp2 = *tmp2p;
  ColorSpinorField &tmp3 = *tmp3p;
  ColorSpinorField &rSloppy = *rSloppyp;
  ColorSpinorField &xSloppy = param.use_sloppy_partial_accumulator ? *xSloppyp : x;

  //  const int i = 0;  // MW: hack to be able to write Component(i) instead and try with i=0 for now

  for(int i=0; i<param.num_src; i++){
    mat(r.Component(i), x.Component(i), y.Component(i));
  }

  // double r2[QUDA_MAX_MULTI_SHIFT];
  MatrixXcd r2(param.num_src,param.num_src);
  for(int i=0; i<param.num_src; i++){
    r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
    printfQuda("r2[%i] %e\n", i, r2(i,i).real());
  }
  if(use_block){
    // MW need to initalize the full r2 matrix here
    for(int i=0; i<param.num_src; i++){
      for(int j=i+1; j<param.num_src; j++){
        r2(i,j) = blas::cDotProduct(r.Component(i), r.Component(j));
        r2(j,i) = std::conj(r2(i,j));
      }
    }
  }

  blas::copy(rSloppy, r);
  blas::copy(p, rSloppy);
  blas::copy(pnew, rSloppy);

  if (&x != &xSloppy) {
    blas::copy(y, x);
    blas::zero(xSloppy);
  } else {
    blas::zero(y);
  }

  const bool use_heavy_quark_res =
  (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
  bool heavy_quark_restart = false;

  profile.TPSTOP(QUDA_PROFILE_INIT);
  profile.TPSTART(QUDA_PROFILE_PREAMBLE);

  MatrixXcd r2_old(param.num_src, param.num_src);
  double heavy_quark_res[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark res idual
  double heavy_quark_res_old[QUDA_MAX_MULTI_SHIFT] = {0.0};  // heavy quark residual
  double stop[QUDA_MAX_MULTI_SHIFT];

  for(int i = 0; i < param.num_src; i++){
    stop[i] = stopping(param.tol, b2[i], param.residual_type);  // stopping condition of solver
    if (use_heavy_quark_res) {
      heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
      heavy_quark_res_old[i] = heavy_quark_res[i];   // heavy quark residual
    }
  }
  const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

  MatrixXcd alpha = MatrixXcd::Zero(param.num_src,param.num_src);
  MatrixXcd beta = MatrixXcd::Zero(param.num_src,param.num_src);
  MatrixXcd gamma = MatrixXcd::Identity(param.num_src,param.num_src);
  //  gamma = gamma * 2.0;

  MatrixXcd pAp(param.num_src, param.num_src);
  MatrixXcd pTp(param.num_src, param.num_src);
  int rUpdate = 0;

  double rNorm[QUDA_MAX_MULTI_SHIFT];
  double r0Norm[QUDA_MAX_MULTI_SHIFT];
  double maxrx[QUDA_MAX_MULTI_SHIFT];
  double maxrr[QUDA_MAX_MULTI_SHIFT];

  for(int i = 0; i < param.num_src; i++){
    rNorm[i] = sqrt(r2(i,i).real());
    r0Norm[i] = rNorm[i];
    maxrx[i] = rNorm[i];
    maxrr[i] = rNorm[i];
  }

  double delta = param.delta;//MW: hack no reliable updates param.delta;

  // this parameter determines how many consective reliable update
  // reisudal increases we tolerate before terminating the solver,
  // i.e., how long do we want to keep trying to converge
  const int maxResIncrease = (use_heavy_quark_res ? 0 : param.max_res_increase); //  check if we reached the limit of our tolerance
  const int maxResIncreaseTotal = param.max_res_increase_total;
  // 0 means we have no tolerance
  // maybe we should expose this as a parameter
  const int hqmaxresIncrease = maxResIncrease + 1;

  int resIncrease = 0;
  int resIncreaseTotal = 0;
  int hqresIncrease = 0;

  // set this to true if maxResIncrease has been exceeded but when we use heavy quark residual we still want to continue the CG
  // only used if we use the heavy_quark_res
  bool L2breakdown = false;

  profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
  profile.TPSTART(QUDA_PROFILE_COMPUTE);
  blas::flops = 0;

  int k = 0;

  for(int i=0; i<param.num_src; i++){
    r2avg+=r2(i,i).real();
  }
  PrintStats("CG", k, r2avg, b2avg, heavy_quark_res[0]);
  int steps_since_reliable = 1;
  bool allconverged = true;
  bool converged[QUDA_MAX_MULTI_SHIFT];
  for(int i=0; i<param.num_src; i++){
    converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
    allconverged = allconverged && converged[i];
  }
  MatrixXcd sigma(param.num_src,param.num_src);

  #ifdef BLOCKCG_GS
  // begin ignore Gram-Schmidt for now

  for(int i=0; i < param.num_src; i++){
    double n = blas::norm2(p.Component(i));
    blas::ax(1/sqrt(n),p.Component(i));
    for(int j=i+1; j < param.num_src; j++) {
      std::complex<double> ri=blas::cDotProduct(p.Component(i),p.Component(j));
      blas::caxpy(-ri,p.Component(i),p.Component(j));
    }
  }

  gamma = MatrixXcd::Zero(param.num_src,param.num_src);
  for ( int i = 0; i < param.num_src; i++){
    for (int j=i; j < param.num_src; j++){
      gamma(i,j) = blas::cDotProduct(p.Component(i),pnew.Component(j));
    }
  }
  #endif
  // end ignore Gram-Schmidt for now

  #ifdef MWVERBOSE
  for(int i=0; i<param.num_src; i++){
    for(int j=0; j<param.num_src; j++){
      pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
    }
  }

  std::cout << " pTp " << std::endl << pTp << std::endl;
  std::cout <<  "QR" << gamma<<  std::endl << "QP " << gamma.inverse()*gamma << std::endl;;
  #endif
  while ( !allconverged && k < param.maxiter ) {
    for(int i=0; i<param.num_src; i++){
      matSloppy(Ap.Component(i), p.Component(i), tmp.Component(i), tmp2.Component(i));  // tmp as tmp
    }


    bool breakdown = false;
    // FIXME: need to check breakdown
    // current implementation sets breakdown to true for pipelined CG if one rhs triggers breakdown
    // this is probably ok


    if (param.pipeline) {
      errorQuda("pipeline not implemented");
    } else {
      r2_old = r2;
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j < param.num_src; j++){
          if(use_block or i==j)
          pAp(i,j) = blas::cDotProduct(p.Component(i), Ap.Component(j));
          else
          pAp(i,j) = 0.;
        }
      }

      alpha = pAp.inverse() * gamma.adjoint().inverse() * r2;
      #ifdef MWVERBOSE
      std::cout << "alpha\n" << alpha << std::endl;

      if(k==1){
        std::cout << "pAp " << std::endl <<pAp << std::endl;
        std::cout << "pAp^-1 " << std::endl <<pAp.inverse() << std::endl;
        std::cout << "r2 " << std::endl <<r2 << std::endl;
        std::cout << "alpha " << std::endl <<alpha << std::endl;
        std::cout << "pAp^-1r2" << std::endl << pAp.inverse()*r2 << std::endl;
      }
      #endif
      // here we are deploying the alternative beta computation
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j < param.num_src; j++){

          blas::caxpy(-alpha(j,i), Ap.Component(j), rSloppy.Component(i));
        }
      }
      // MW need to calculate the full r2 matrix here, after update. Not sure how to do alternative sigma yet ...
      for(int i=0; i<param.num_src; i++){
        for(int j=0; j<param.num_src; j++){
          if(use_block or i==j)
          r2(i,j) = blas::cDotProduct(r.Component(i), r.Component(j));
          else
          r2(i,j) = 0.;
        }
      }
      sigma = r2;
    }


    bool updateX=false;
    bool updateR=false;
    //      int updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? true : false;
    //      int updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? true : false;
    //
    // printfQuda("Checking reliable update %i %i\n",updateX,updateR);
    // reliable update conditions
    for(int i=0; i<param.num_src; i++){
      rNorm[i] = sqrt(r2(i,i).real());
      if (rNorm[i] > maxrx[i]) maxrx[i] = rNorm[i];
      if (rNorm[i] > maxrr[i]) maxrr[i] = rNorm[i];
      updateX = (rNorm[i] < delta * r0Norm[i] && r0Norm[i] <= maxrx[i]) ? true : false;
      updateR = ((rNorm[i] < delta * maxrr[i] && r0Norm[i] <= maxrr[i]) || updateX) ? true : false;
    }
    if ( (updateR || updateX )) {
      // printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
      updateX=false;
      updateR=false;
      // printfQuda("Suppressing reliable update %i %i\n",updateX,updateR);
    }

    if ( !(updateR || updateX )) {

      beta = gamma * r2_old.inverse() * sigma;
      #ifdef MWVERBOSE
      std::cout << "beta\n" << beta << std::endl;
      #endif
      if (param.pipeline && !breakdown)
      errorQuda("pipeline not implemented");

      else{
        for(int i=0; i<param.num_src; i++){
          for(int j=0; j<param.num_src; j++){
            blas::caxpy(alpha(j,i),p.Component(j),xSloppy.Component(i));
          }
        }

        // set to zero
        for(int i=0; i < param.num_src; i++){
          blas::ax(0,pnew.Component(i)); // do we need components here?
        }
        // add r
        for(int i=0; i<param.num_src; i++){
          // for(int j=0;j<param.num_src; j++){
          // order of updating p might be relevant here
          blas::axpy(1.0,r.Component(i),pnew.Component(i));
          // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
          // }
        }
        // beta = beta * gamma.inverse();
        for(int i=0; i<param.num_src; i++){
          for(int j=0;j<param.num_src; j++){
            double rcoeff= (j==0?1.0:0.0);
            // order of updating p might be relevant hereq
            blas::caxpy(beta(j,i),p.Component(j),pnew.Component(i));
            // blas::axpby(rcoeff,rSloppy.Component(i),beta(i,j),p.Component(j));
          }
        }
        // now need to do something with the p's

        for(int i=0; i< param.num_src; i++){
          blas::copy(p.Component(i), pnew.Component(i));
        }


        #ifdef BLOCKCG_GS
        for(int i=0; i < param.num_src; i++){
          double n = blas::norm2(p.Component(i));
          blas::ax(1/sqrt(n),p.Component(i));
          for(int j=i+1; j < param.num_src; j++) {
            std::complex<double> ri=blas::cDotProduct(p.Component(i),p.Component(j));
            blas::caxpy(-ri,p.Component(i),p.Component(j));

          }
        }


        gamma = MatrixXcd::Zero(param.num_src,param.num_src);
        for ( int i = 0; i < param.num_src; i++){
          for (int j=i; j < param.num_src; j++){
            gamma(i,j) = blas::cDotProduct(p.Component(i),pnew.Component(j));
          }
        }
        #endif

        #ifdef MWVERBOSE
        for(int i=0; i<param.num_src; i++){
          for(int j=0; j<param.num_src; j++){
            pTp(i,j) = blas::cDotProduct(p.Component(i), p.Component(j));
          }
        }
        std::cout << " pTp " << std::endl << pTp << std::endl;
        std::cout <<  "QR" << gamma<<  std::endl << "QP " << gamma.inverse()*gamma << std::endl;;
        #endif
      }


      if (use_heavy_quark_res && (k % heavy_quark_check) == 0) {
        if (&x != &xSloppy) {
          blas::copy(tmp, y);   //  FIXME: check whether copy works here
          for(int i=0; i<param.num_src; i++){
            heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy.Component(i), tmp.Component(i), rSloppy.Component(i)).z);
          }
        } else {
          blas::copy(r, rSloppy);  //  FIXME: check whether copy works here
          for(int i=0; i<param.num_src; i++){
            heavy_quark_res[i] = sqrt(blas::xpyHeavyQuarkResidualNorm(x.Component(i), y.Component(i), r.Component(i)).z);
          }
        }
      }

      steps_since_reliable++;
    } else {
      printfQuda("reliable update\n");
      for(int i=0; i<param.num_src; i++){
        blas::axpy(alpha(i,i).real(), p.Component(i), xSloppy.Component(i));
      }
      blas::copy(x, xSloppy); // nop when these pointers alias

      for(int i=0; i<param.num_src; i++){
        blas::xpy(x.Component(i), y.Component(i)); // swap these around?
      }
      for(int i=0; i<param.num_src; i++){
        mat(r.Component(i), y.Component(i), x.Component(i), tmp3.Component(i)); //  here we can use x as tmp
      }
      for(int i=0; i<param.num_src; i++){
        r2(i,i) = blas::xmyNorm(b.Component(i), r.Component(i));
      }

      for(int i=0; i<param.num_src; i++){
        blas::copy(rSloppy.Component(i), r.Component(i)); //nop when these pointers alias
        blas::zero(xSloppy.Component(i));
      }

      // calculate new reliable HQ resididual
      if (use_heavy_quark_res){
        for(int i=0; i<param.num_src; i++){
          heavy_quark_res[i] = sqrt(blas::HeavyQuarkResidualNorm(y.Component(i), r.Component(i)).z);
        }
      }

      // MW: FIXME as this probably goes terribly wrong right now
      for(int i = 0; i<param.num_src; i++){
        // break-out check if we have reached the limit of the precision
        if (sqrt(r2(i,i).real()) > r0Norm[i] && updateX) { // reuse r0Norm for this
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
          sqrt(r2(i,i).real()), r0Norm[i], resIncreaseTotal);
          if ( resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            if (use_heavy_quark_res) {
              L2breakdown = true;
            } else {
              warningQuda("CG: solver exiting due to too many true residual norm increases");
              break;
            }
          }
        } else {
          resIncrease = 0;
        }
      }
      // if L2 broke down already we turn off reliable updates and restart the CG
      for(int i = 0; i<param.num_src; i++){
        if (use_heavy_quark_res and L2breakdown) {
          delta = 0;
          warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
          heavy_quark_restart = true;
          if (heavy_quark_res[i] > heavy_quark_res_old[i]) {
            hqresIncrease++;
            warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res[i], heavy_quark_res_old[i]);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
              break;
            }
          }
        }
      }

      for(int i=0; i<param.num_src; i++){
        rNorm[i] = sqrt(r2(i,i).real());
        maxrr[i] = rNorm[i];
        maxrx[i] = rNorm[i];
        r0Norm[i] = rNorm[i];
        heavy_quark_res_old[i] = heavy_quark_res[i];
      }
      rUpdate++;

      if (use_heavy_quark_res and heavy_quark_restart) {
        // perform a restart
        blas::copy(p, rSloppy);
        heavy_quark_restart = false;
      } else {
        // explicitly restore the orthogonality of the gradient vector
        for(int i=0; i<param.num_src; i++){
          double rp = blas::reDotProduct(rSloppy.Component(i), p.Component(i)) / (r2(i,i).real());
          blas::axpy(-rp, rSloppy.Component(i), p.Component(i));

          beta(i,i) = r2(i,i) / r2_old(i,i);
          blas::xpay(rSloppy.Component(i), beta(i,i).real(), p.Component(i));
        }
      }

      steps_since_reliable = 0;
    }

    breakdown = false;
    k++;

    allconverged = true;
    r2avg=0;
    for(int i=0; i<param.num_src; i++){
      r2avg+= r2(i,i).real();
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged[i] = convergence(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
      allconverged = allconverged && converged[i];
    }
    PrintStats("CG", k, r2avg, b2avg, heavy_quark_res[0]);

    // check for recent enough reliable updates of the HQ residual if we use it
    if (use_heavy_quark_res) {
      for(int i=0; i<param.num_src; i++){
        // L2 is concverged or precision maxed out for L2
        bool L2done = L2breakdown or convergenceL2(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
        // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
        bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2(i,i).real(), heavy_quark_res[i], stop[i], param.tol_hq);
        converged[i] = L2done and HQdone;
      }
    }

  }

  blas::copy(x, xSloppy);
  for(int i=0; i<param.num_src; i++){
    blas::xpy(y.Component(i), x.Component(i));
  }

  profile.TPSTOP(QUDA_PROFILE_COMPUTE);
  profile.TPSTART(QUDA_PROFILE_EPILOGUE);

  param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
  double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
  param.gflops = gflops;
  param.iter += k;

  if (k == param.maxiter)
  warningQuda("Exceeded maximum iterations %d", param.maxiter);

  if (getVerbosity() >= QUDA_VERBOSE)
  printfQuda("CG: Reliable updates = %d\n", rUpdate);

  // compute the true residuals
  for(int i=0; i<param.num_src; i++){
    mat(r.Component(i), x.Component(i), y.Component(i), tmp3.Component(i));
    param.true_res = sqrt(blas::xmyNorm(b.Component(i), r.Component(i)) / b2[i]);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x.Component(i), r.Component(i)).z);
    param.true_res_offset[i] = param.true_res;
    param.true_res_hq_offset[i] = param.true_res_hq;

    PrintSummary("CG", k, r2(i,i).real(), b2[i], stop[i], 0.0);
  }

  // reset the flops counters
  blas::flops = 0;
  mat.flops();
  matSloppy.flops();

  profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
  profile.TPSTART(QUDA_PROFILE_FREE);

  profile.TPSTOP(QUDA_PROFILE_FREE);

  return;

  #endif

}
#endif


}  // namespace quda
