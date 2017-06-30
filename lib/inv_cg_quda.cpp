#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <sys/time.h>
#include <limits>
#include <cmath>

#include <face_quda.h>

#include <iostream>


namespace quda {

  CG::CG(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), yp(nullptr), rp(nullptr), App(nullptr), tmpp(nullptr),
    init(false) {
  }

  CG::~CG() {
    if ( init ) {
      for (auto pi : p) delete pi;
      if (rp) delete rp;
      if (yp) delete yp;
      if (App) delete App;
      if (tmpp) delete tmpp;
      init = false;
    }
  }

  CGNE::CGNE(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    CG(mmdag, mmdagSloppy, param, profile), mmdag(mat.Expose()), mmdagSloppy(mat.Expose()), init(false) {
  }

  CGNE::~CGNE() {
    if ( init ) {
      delete xp;
      init = false;
    }
  }

  // CGNE: M Mdag y = b is solved; x = Mdag y is returned as solution.
  void CGNE::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    if (!init) {
      ColorSpinorParam csParam(x);
      csParam.create = QUDA_COPY_FIELD_CREATE;
      xp = ColorSpinorField::Create(x, csParam);

      init = true;

    } else if(param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      warningQuda("Initial guess may not work as expected with CGNE\n");
      *xp = x;
    }

    CG::operator()(*xp,b);

    mmdag.Expose()->Mdag(x,*xp);
  }

  CGNR::CGNR(DiracMatrix &mat, DiracMatrix &matSloppy, SolverParam &param, TimeProfile &profile) :
    CG(mdagm, mdagmSloppy, param, profile), mdagm(mat.Expose()), mdagmSloppy(mat.Expose()), init(false) {
  }

  CGNR::~CGNR() {
    if ( init ) {
      delete bp;
      init = false;
    }
  }

  // CGNR: Mdag M x = Mdag b is solved.
  void CGNR::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    const int iter0 = param.iter;

    if (!init) {
      ColorSpinorParam csParam(b);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      bp = ColorSpinorField::Create(csParam);

      init = true;

    }

    mdagm.Expose()->Mdag(*bp,b);
    CG::operator()(x,*bp);

    if (param.compute_true_res) {
      // compute the true residuals
      const double b2 = blas::norm2(b);
      mdagm.Expose()->M(*bp, x);
      const double r2 = blas::xmyNorm(b, *bp) / b2;
      param.true_res = sqrt(r2);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, *bp).z);

      PrintSummary("CGNR", param.iter - iter0, r2, b2);
    }
  }

  void CG::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)
      errorQuda("Not supported");

    const int Np = (param.solution_accumulator_pipeline == 0 ? 1 : param.solution_accumulator_pipeline);
    if (Np < 0 || Np > 16) errorQuda("Invalid value %d for solution_accumulator_pipeline\n", Np);

#ifdef ALTRELIABLE
    // hack to select alternative reliable updates
    constexpr bool alternative_reliable = true;
    warningQuda("Using alternative reliable updates. This feature is mostly ok but needs a little more testing in the real world.\n");
#else
    constexpr bool alternative_reliable = false;
#endif
    profile.TPSTART(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source
    double b2 = blas::norm2(b);

    // Check to see that we're not trying to invert on a zero-field source
    if (b2 == 0 && param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_NO) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return;
    }

    ColorSpinorParam csParam(x);
    if (!init) {
      csParam.create = QUDA_NULL_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);
      // sloppy fields
      csParam.setPrecision(param.precision_sloppy);
      App = ColorSpinorField::Create(csParam);
      tmpp = ColorSpinorField::Create(csParam);

      init = true;

    }
    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &Ap = *App;
    ColorSpinorField &tmp = *tmpp;

    csParam.setPrecision(param.precision_sloppy);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    // tmp2 only needed for multi-gpu Wilson-like kernels
    ColorSpinorField *tmp2_p = !mat.isStaggered() ? ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp2 = *tmp2_p;

    // additional high-precision temporary if Wilson and mixed-precision
    csParam.setPrecision(param.precision);
    ColorSpinorField *tmp3_p = (x.Precision() != param.precision_sloppy && !mat.isStaggered()) ?
      ColorSpinorField::Create(x, csParam) : &tmp;
    ColorSpinorField &tmp3 = *tmp3_p;

    // alternative reliable updates
    // alternative reliable updates - set precision - does not hurt performance here

    const double u= param.precision_sloppy == 8 ? std::numeric_limits<double>::epsilon()/2. : ((param.precision_sloppy == 4) ? std::numeric_limits<float>::epsilon()/2. : pow(2.,-13));
    const double uhigh= param.precision == 8 ? std::numeric_limits<double>::epsilon()/2. : ((param.precision == 4) ? std::numeric_limits<float>::epsilon()/2. : pow(2.,-13));
    const double deps=sqrt(u);
    constexpr double dfac = 1.1;
    double d_new =0 ;
    double d =0 ;
    double dinit =0;
    double xNorm = 0;
    double xnorm = 0;
    double pnorm = 0;
    double ppnorm = 0;
    double Anorm = 0;
    
    // for alternative reliable updates
    if(alternative_reliable){
      // estimate norm for reliable updates
      mat(r, b, y, tmp3);
      Anorm = sqrt(blas::norm2(r)/b2);
    }


    // compute initial residual
    mat(r, x, y, tmp3);
    double r2 = blas::xmyNorm(b, r);
    if (b2 == 0) {
      b2 = r2;
    }

    csParam.setPrecision(param.precision_sloppy);
    ColorSpinorField *r_sloppy;
    if (param.precision_sloppy == x.Precision()) {
      r_sloppy = &r;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      r_sloppy = ColorSpinorField::Create(r, csParam);
    }

    ColorSpinorField *x_sloppy;
    if (param.precision_sloppy == x.Precision() ||
        !param.use_sloppy_partial_accumulator) {
      x_sloppy = &x;
    } else {
      csParam.create = QUDA_COPY_FIELD_CREATE;
      x_sloppy = ColorSpinorField::Create(x, csParam);
    }

    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    csParam.create = QUDA_COPY_FIELD_CREATE;
    csParam.setPrecision(param.precision_sloppy);

    if (Np != (int)p.size()) {
      for (auto &pi : p) delete pi;
      p.resize(Np);
      for (auto &pi : p) pi = ColorSpinorField::Create(rSloppy, csParam);
    } else {
      for (auto &pi : p) *pi = rSloppy;
    }

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

    double r2_old;

    double stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver

    double heavy_quark_res = 0.0;  // heavy quark res idual
    double heavy_quark_res_old = 0.0;  // heavy quark residual

    if (use_heavy_quark_res) {
      heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
      heavy_quark_res_old = heavy_quark_res;   // heavy quark residual
    }
    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double alpha[Np];
    double beta = 0.0;
    double pAp;
    int rUpdate = 0;

    double rNorm = sqrt(r2);
    double r0Norm = rNorm;
    double maxrx = rNorm;
    double maxrr = rNorm;
    double delta = param.delta;


    // this parameter determines how many consective reliable update
    // residual increases we tolerate before terminating the solver,
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
        if(alternative_reliable){
          double3 pAppp = blas::cDotProductNormA(*p[j],Ap);
          pAp = pAppp.x;
          ppnorm = pAppp.z;
        }
        else{
          pAp = blas::reDotProduct(*p[j], Ap);
        }

        alpha[j] = r2 / pAp;

        // here we are deploying the alternative beta computation
        Complex cg_norm = blas::axpyCGNorm(-alpha[j], Ap, rSloppy);
        r2 = real(cg_norm);  // (r_new, r_new)
        std::cout << "altS " << imag(cg_norm) << " regS" << r2 << " deltaS " << (r2-imag(cg_norm))/r2 << std::endl;
        sigma = r2;//imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2;  // use r2 if (r_k+1, r_k+1-r_k) breaks
      }

      // reliable update conditions
      rNorm = sqrt(r2);
      int updateX;
      int updateR;

      if(alternative_reliable){
        // alternative reliable updates
        updateX = ( (d <= deps*sqrt(r2_old)) or (dfac * dinit > deps * r0Norm) ) and (d_new > deps*rNorm) and (d_new > dfac * dinit);
        updateR = 0;
        // if(updateX)
          // printfQuda("new reliable update conditions (%i) d_n-1 < eps r2_old %e %e;\t dn > eps r_n %e %e;\t (dnew > 1.1 dinit %e %e)\n",
        // updateX,d,deps*sqrt(r2_old),d_new,deps*rNorm,d_new,dinit);
      }
      else{
        if (rNorm > maxrx) maxrx = rNorm;
        if (rNorm > maxrr) maxrr = rNorm;
        updateX = (rNorm < delta*r0Norm && r0Norm <= maxrx) ? 1 : 0;
        updateR = ((rNorm < delta*maxrr && r0Norm <= maxrr) || updateX) ? 1 : 0;
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
	      Complex alpha_[Np];
	      for (int i=0; i<Np; i++) alpha_[i] = alpha[i];
	      std::vector<ColorSpinorField*> x_;
	      x_.push_back(&xSloppy);
	      blas::caxpy(alpha_, p, x_);
	      blas::flops -= 4*j*xSloppy.RealLength(); // correct for over flop count since using caxpy
	    }

	    //p[(k+1)%Np] = r + beta * p[k%Np]
	    blas::xpayz(rSloppy, beta, *p[j], *p[(j+1)%Np]);
	  }
	}

	if (use_heavy_quark_res && k%heavy_quark_check==0) {
	  if (&x != &xSloppy) {
	    blas::copy(tmp,y);
	    heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(xSloppy, tmp, rSloppy).z);
	  } else {
	    blas::copy(r, rSloppy);
	    heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(x, y, r).z);
	  }
	}

	// alternative reliable updates
	if(alternative_reliable){
	  d = d_new;
	  pnorm = pnorm + alpha[j] * alpha[j]* (ppnorm);
	  xnorm = sqrt(pnorm);
	  d_new = d + u*rNorm + uhigh*Anorm * xnorm;
	  if(steps_since_reliable==0)
	    printfQuda("New dnew: %e (r %e , y %e)\n",d_new,u*rNorm,uhigh*Anorm * sqrt(blas::norm2(y)) );
	}
	steps_since_reliable++;

      } else {

	{
	  Complex alpha_[Np];
	  for (int i=0; i<=j; i++) alpha_[i] = alpha[i];
	  std::vector<ColorSpinorField*> x_;
	  x_.push_back(&xSloppy);
	  std::vector<ColorSpinorField*> p_;
	  for (int i=0; i<=j; i++) p_.push_back(p[i]);
	  blas::caxpy(alpha_, p_, x_);
	  blas::flops -= 4*j*xSloppy.RealLength(); // correct for over flop count since using caxpy
	}

        blas::copy(x, xSloppy); // nop when these pointers alias

        blas::xpy(x, y); // swap these around?
        mat(r, y, x, tmp3); //  here we can use x as tmp
        r2 = blas::xmyNorm(b, r);

        blas::copy(rSloppy, r); //nop when these pointers alias
        blas::zero(xSloppy);

        // alternative reliable updates
        if(alternative_reliable){
          dinit = uhigh*(sqrt(r2) + Anorm * sqrt(blas::norm2(y)));
          d = d_new;
          xnorm = 0;//sqrt(norm2(x));
          pnorm = 0;//pnorm + alpha * sqrt(norm2(p));
          printfQuda("New dinit: %e (r %e , y %e)\n",dinit,uhigh*sqrt(r2),uhigh*Anorm*sqrt(blas::norm2(y)));
          d_new = dinit;
          r0Norm = sqrt(r2);
        }
        else{
          rNorm = sqrt(r2);
          maxrr = rNorm;
          maxrx = rNorm;
          r0Norm = rNorm;
        }


        // calculate new reliable HQ resididual
        if (use_heavy_quark_res) heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(y, r).z);

        // break-out check if we have reached the limit of the precision
        if (sqrt(r2) > r0Norm && updateX) { // reuse r0Norm for this
          resIncrease++;
          resIncreaseTotal++;
          warningQuda("CG: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
          sqrt(r2), r0Norm, resIncreaseTotal);
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
        // if L2 broke down already we turn off reliable updates and restart the CG
        if (use_heavy_quark_res and L2breakdown) {
          delta = 0;
          warningQuda("CG: Restarting without reliable updates for heavy-quark residual");
          heavy_quark_restart = true;
          if (heavy_quark_res > heavy_quark_res_old) {
            hqresIncrease++;
            warningQuda("CG: new reliable HQ residual norm %e is greater than previous reliable residual norm %e", heavy_quark_res, heavy_quark_res_old);
            // break out if we do not improve here anymore
            if (hqresIncrease > hqmaxresIncrease) {
              warningQuda("CG: solver exiting due to too many heavy quark residual norm increases");
              break;
            }
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
        // L2 is concverged or precision maxed out for L2
        bool L2done = L2breakdown or convergenceL2(r2, heavy_quark_res, stop, param.tol_hq);
        // HQ is converged and if we do reliable update the HQ residual has been calculated using a reliable update
        bool HQdone = (steps_since_reliable == 0 and param.delta > 0) and convergenceHQ(r2, heavy_quark_res, stop, param.tol_hq);
        converged = L2done and HQdone;
      }

      // if we have converged and need to update any trailing solutions
      if (converged && steps_since_reliable > 0 && (j+1)%Np != 0 ) {
	Complex alpha_[Np];
	for (int i=0; i<=j; i++) alpha_[i] = alpha[i];
	std::vector<ColorSpinorField*> x_;
	x_.push_back(&xSloppy);
	std::vector<ColorSpinorField*> p_;
	for (int i=0; i<=j; i++) p_.push_back(p[i]);
	blas::caxpy(alpha_, p_, x_);
	blas::flops -= 4*j*xSloppy.RealLength(); // correct for over flop count since using caxpy
      }

      j = steps_since_reliable == 0 ? 0 : (j+1)%Np; // if just done a reliable update then reset j
    }

    blas::copy(x, xSloppy);
    blas::xpy(y, x);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);

    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops() + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    { // temporary addition for SC'17
      comm_allreduce(&gflops);
      printfQuda("CG: Convergence in %d iterations, %f seconds, GFLOPS = %g\n", k, param.secs, gflops / param.secs);
    }

    if (k == param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("CG: Reliable updates = %d\n", rUpdate);

    if (param.compute_true_res) {
      // compute the true residuals
      mat(r, x, y, tmp3);
      param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
      param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);
    }

    PrintSummary("CG", k, r2, b2);

    // reset the flops counters
    blas::flops = 0;
    mat.flops();
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (&tmp3 != &tmp) delete tmp3_p;
    if (&tmp2 != &tmp) delete tmp2_p;

    if (&rSloppy != &r) delete r_sloppy;
    if (&xSloppy != &x) delete x_sloppy;

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return;
  }


}  // namespace quda
