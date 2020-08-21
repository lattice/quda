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

}  // namespace quda
