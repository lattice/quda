#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <string.h>

#ifdef MAGMA_LIB 
#include <blas_magma.h>
#endif


#include <Eigen/Dense>

#include <deflation.h>


/*
Based on  eigCG(nev, m) algorithm:
A. Stathopolous and K. Orginos, arXiv:0707.0131
*/

namespace quda {

   using namespace blas;
   using namespace Eigen;

   using DynamicStride   = Stride<Dynamic, Dynamic>;
   using DenseMatrix     = MatrixXcd;
   using VectorSet       = MatrixXcd;
   using Vector          = VectorXcd;
   using RealVector      = VectorXd;

//special types needed for compatibility with QUDA blas:
   using RowMajorDenseMatrix = Matrix<Complex, Dynamic, Dynamic, RowMajor>;

   static int max_eigcg_cycles = 4;//how many eigcg cycles do we allow?

   enum  class libtype {eigen_lib, magma_lib, lapack_lib, mkl_lib};

   class EigCGArgs
   {

   public:
     // host Lanczos matrice, and its eigenvalue/vector arrays:
     DenseMatrix Tm; // VH A V,
     // eigenvectors:
     VectorSet ritzVecs; // array of  (m)  ritz and of m length
     // eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)
     RealVector Tmvals; // eigenvalues of T[m,  m  ] and T[m-1, m-1] (re-used)
     // Aux matrix for computing 2k Ritz vectors:
     DenseMatrix H2k;

     int m;
     int k;
     int id; // cuurent search spase index

     int restarts;
     double global_stop;

     bool run_residual_correction; // used in mixed precision cycles

     ColorSpinorFieldSet
       *V2k; // eigCG accumulation vectors needed to update Tm (spinor matrix of size eigen_vector_length x (2*k))

     EigCGArgs(int m, int k) :
       Tm(DenseMatrix::Zero(m, m)),
       ritzVecs(VectorSet::Zero(m, m)),
       Tmvals(m),
       H2k(2 * k, 2 * k),
       m(m),
       k(k),
       id(0),
       restarts(0),
       global_stop(0.0),
       run_residual_correction(false),
       V2k(nullptr)
     {
     }

     ~EigCGArgs()
     {
       if (V2k) delete V2k;
     }

     // method for constructing Lanczos matrix :
     inline void SetLanczos(Complex diag_val, Complex offdiag_val)
     {
       if (run_residual_correction) return;

       Tm.diagonal<0>()[id] = diag_val;

       if (id < (m - 1)) { // Load Lanczos off-diagonals:
         Tm.diagonal<+1>()[id] = offdiag_val;
         Tm.diagonal<-1>()[id] = offdiag_val;
       }

       id += 1;

       return;
     }

     inline void ResetArgs()
     {
       id = 0;
       Tm.setZero();
       Tmvals.setZero();
       ritzVecs.setZero();

       if (V2k) delete V2k;
       V2k = nullptr;
     }

     inline void ResetSearchIdx()
     {
       id = 2 * k;
       restarts += 1;
     }

     void RestartLanczos(ColorSpinorField *w, ColorSpinorFieldSet *v, const double inv_sqrt_r2)
     {
       Tm.setZero();

       auto s = std::make_unique<Complex[]>(2 * k);

       for (int i = 0; i < 2 * k; i++) Tm(i, i) = Tmvals(i); //??

       std::vector<ColorSpinorField *> w_;
       w_.push_back(w);

       std::vector<ColorSpinorField *> v_(v->Components().begin(), v->Components().begin() + 2 * k);

       blas::cDotProduct(s.get(), w_, v_);

       Map<VectorXcd, Unaligned> s_(s.get(), 2 * k);
       s_ *= inv_sqrt_r2;

       Tm.col(2 * k).segment(0, 2 * k) = s_;
       Tm.row(2 * k).segment(0, 2 * k) = s_.adjoint();
     }
   };

   //Rayleigh Ritz procedure:
   template <libtype which_lib> void ComputeRitz(EigCGArgs &args) { errorQuda("\nUnknown library type."); }

   //pure eigen version: 
   template <> void ComputeRitz<libtype::eigen_lib>(EigCGArgs &args)
   {
     const int m = args.m;
     const int k = args.k;
     //Solve m dim eigenproblem:
     SelfAdjointEigenSolver<MatrixXcd> es_tm(args.Tm);
     args.ritzVecs.leftCols(k) = es_tm.eigenvectors().leftCols(k);
     //Solve m-1 dim eigenproblem:
     SelfAdjointEigenSolver<MatrixXcd> es_tm1(Map<MatrixXcd, Unaligned, DynamicStride >(args.Tm.data(), (m-1), (m-1), DynamicStride(m, 1)));
     Block<MatrixXcd>(args.ritzVecs.derived(), 0, k, m-1, k) = es_tm1.eigenvectors().leftCols(k);
     args.ritzVecs.block(m-1, k, 1, k).setZero();

     MatrixXcd Q2k(MatrixXcd::Identity(m, 2*k));
     HouseholderQR<MatrixXcd> ritzVecs2k_qr( Map<MatrixXcd, Unaligned >(args.ritzVecs.data(), m, 2*k) );
     Q2k.applyOnTheLeft( ritzVecs2k_qr.householderQ() );

     //2. Construct H = QH*Tm*Q :
     args.H2k = Q2k.adjoint()*args.Tm*Q2k;

     /* solve the small evecm1 2nev x 2nev eigenproblem */
     SelfAdjointEigenSolver<MatrixXcd> es_h2k(args.H2k);
     Block<MatrixXcd>(args.ritzVecs.derived(), 0, 0, m, 2*k) = Q2k * es_h2k.eigenvectors();
     args.Tmvals.segment(0,2*k) = es_h2k.eigenvalues();//this is ok

     return;
   }

   //(supposed to be a pure) magma version: 
   template <> void ComputeRitz<libtype::magma_lib>(EigCGArgs &args)
   {
#ifdef MAGMA_LIB
     const int m = args.m;
     const int k = args.k;
     //Solve m dim eigenproblem:
     args.ritzVecs = args.Tm;
     Complex *evecm = static_cast<Complex*>( args.ritzVecs.data());
     double  *evalm = static_cast<double *>( args.Tmvals.data());

     cudaHostRegister(static_cast<void *>(evecm), m*m*sizeof(Complex),  cudaHostRegisterDefault);
     magma_Xheev(evecm, m, m, evalm, sizeof(Complex));
     //Solve m-1 dim eigenproblem:
     DenseMatrix ritzVecsm1(args.Tm);
     Complex *evecm1 = static_cast<Complex*>( ritzVecsm1.data());

     cudaHostRegister(static_cast<void *>(evecm1), m*m*sizeof(Complex),  cudaHostRegisterDefault);
     magma_Xheev(evecm1, (m-1), m, evalm, sizeof(Complex));
     // fill 0s in mth element of old evecs:
     for(int l = 1; l <= m ; l++) evecm1[l*m-1] = 0.0 ;
     // Attach the first nev old evecs at the end of the nev latest ones:
     memcpy(&evecm[k*m], evecm1, k*m*sizeof(Complex));
//?
    // Orthogonalize the 2*nev (new+old) vectors evecm=QR:

     MatrixXcd Q2k(MatrixXcd::Identity(m, 2*k));
     HouseholderQR<MatrixXcd> ritzVecs2k_qr( Map<MatrixXcd, Unaligned >(args.ritzVecs.data(), m, 2*k) );
     Q2k.applyOnTheLeft( ritzVecs2k_qr.householderQ() );

     //2. Construct H = QH*Tm*Q :
     args.H2k = Q2k.adjoint()*args.Tm*Q2k;

     /* solve the small evecm1 2nev x 2nev eigenproblem */
     SelfAdjointEigenSolver<MatrixXcd> es_h2k(args.H2k);
     Block<MatrixXcd>(args.ritzVecs.derived(), 0, 0, m, 2*k) = Q2k * es_h2k.eigenvectors();
     args.Tmvals.segment(0,2*k) = es_h2k.eigenvalues();//this is ok
//?
     cudaHostUnregister(evecm);
     cudaHostUnregister(evecm1);
#else
     errorQuda("Magma library was not built.");
#endif
     return;
  }

  // set the required parameters for the inner solver
  static void fillEigCGInnerSolverParam(SolverParam &inner, const SolverParam &outer, bool use_sloppy_partial_accumulator = true)
  {
    inner.tol = outer.tol_precondition;
    inner.maxiter = outer.maxiter_precondition;
    inner.delta = 1e-20; // no reliable updates within the inner solver
    inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.iter   = 0;
    inner.gflops = 0;
    inner.secs   = 0;

    inner.inv_type_precondition = QUDA_INVALID_INVERTER;
    inner.is_preconditioner = true; // used to tell the inner solver it is an inner solver

    inner.use_sloppy_partial_accumulator= use_sloppy_partial_accumulator;

    if(outer.inv_type == QUDA_EIGCG_INVERTER && outer.precision_sloppy != outer.precision_precondition)
      inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
    else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  }

  // set the required parameters for the initCG solver
  static void fillInitCGSolverParam(SolverParam &inner, const SolverParam &outer) {
    inner.iter   = 0;
    inner.gflops = 0;
    inner.secs   = 0;

    inner.tol              = outer.tol;
    inner.tol_restart      = outer.tol_restart;
    inner.maxiter          = outer.maxiter;
    inner.delta            = outer.delta; 
    inner.precision        = outer.precision; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.inv_type        = QUDA_CG_INVERTER;       // use CG solver
    inner.use_init_guess  = QUDA_USE_INIT_GUESS_YES;// use deflated initial guess...

    inner.use_sloppy_partial_accumulator= false;//outer.use_sloppy_partial_accumulator;
  }


  IncEigCG::IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(nullptr), Kparam(param), Vm(nullptr), r_pre(nullptr), p_pre(nullptr), eigcg_args(nullptr), profile(profile), init(false)
  {

    if (2 * param.nev >= param.m)
      errorQuda(
        "Incorrect number of the requested low modes: m= %d while nev=%d (note that 2*nev must be less then m).",
        param.m, param.nev);

    if (param.rhs_idx < param.deflation_grid)
      printfQuda("\nInitialize eigCG(m=%d, nev=%d) solver.", param.m, param.nev);
    else {
      printfQuda("\nDeflation space is complete, running initCG solver.");
      fillInitCGSolverParam(Kparam, param);
      //K = new CG(mat, matPrecon, Kparam, profile);//Preconditioned Mat has comms flag on
      return;
    }

    if ( param.inv_type == QUDA_EIGCG_INVERTER ) {
      fillEigCGInnerSolverParam(Kparam, param);
    } else if ( param.inv_type == QUDA_INC_EIGCG_INVERTER ) {
      if (param.inv_type_precondition != QUDA_INVALID_INVERTER)
        errorQuda("preconditioning is not supported for the incremental solver.");
      fillInitCGSolverParam(Kparam, param);
    }

    if(param.inv_type_precondition == QUDA_CG_INVERTER){
      K = new CG(matPrecon, matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_MR_INVERTER){
      K = new MR(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_SD_INVERTER){
      K = new SD(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition != QUDA_INVALID_INVERTER){ // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
    }
    return;
  }

  IncEigCG::~IncEigCG() {

    if(init)
    {
      if(Vm)  delete Vm;

      delete tmpp;
      delete rp;
      delete yp;
      delete Ap;
      delete p;

      if(Az) delete Az;

      if(K) {
        delete r_pre;
        delete p_pre;

        delete K;
      }
      delete eigcg_args;
    } else if (K) {
      //delete K; //hack for the init CG solver 
    }
  }

 void IncEigCG::RestartVT(const double beta, const double rho)
  {
    EigCGArgs &args = *eigcg_args;

    if ( param.extlib_type == QUDA_MAGMA_EXTLIB ) {
      ComputeRitz<libtype::magma_lib>(args);
    } else if( param.extlib_type == QUDA_EIGEN_EXTLIB ) {
      ComputeRitz<libtype::eigen_lib>(args);//if args.m > 128, one may better use libtype::magma_lib
    } else {
      errorQuda("Library type %d is currently not supported.", param.extlib_type);
    }

    //Restart V:

    blas::zero(*args.V2k);

    std::vector<ColorSpinorField*> vm (Vm->Components());
    std::vector<ColorSpinorField*> v2k(args.V2k->Components());

    RowMajorDenseMatrix Alpha(args.ritzVecs.topLeftCorner(args.m, 2*args.k));
    blas::caxpy( static_cast<Complex*>(Alpha.data()), vm , v2k);

    for(int i = 0; i < 2*args.k; i++)  blas::copy(Vm->Component(i), args.V2k->Component(i));

    //Restart T:
    ColorSpinorField *omega = nullptr;

    //Compute Az = Ap - beta*Ap_old(=Az):
    blas::xpay(*Ap, -beta, *Az);

    if(Vm->Precision() != Az->Precision())//we may not need this if multiprec blas is used
    {
      Vm->Component(args.m-1) = *Az;//use the last vector as a temporary
      omega = &Vm->Component(args.m-1);
    }
    else omega = Az;

    args.RestartLanczos(omega, Vm, 1.0 / rho);
    return;
  }

  void IncEigCG::UpdateVm(ColorSpinorField &res, double beta, double sqrtr2)
  {
    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) return;

    if (args.id == param.m){//Begin Rayleigh-Ritz block: 
      //
      RestartVT(beta, sqrtr2);
      args.ResetSearchIdx();
    } else if (args.id == (param.m-1)) {
      blas::copy(*Az, *Ap);//save current mat-vec result if ready for the restart in the next cycle
    }

    //load Lanczos basis vector:
    blas::copy(Vm->Component(args.id), res);//convert arrays
    //rescale the vector
    blas::ax(1.0 / sqrtr2, Vm->Component(args.id)); 
    return;
  }

/*
 * This is a solo precision solver.
*/
  int IncEigCG::eigCGsolve(ColorSpinorField &x, ColorSpinorField &b) {

    int k=0;

    if (checkLocation(x, b) != QUDA_CUDA_FIELD_LOCATION)  errorQuda("Not supported");

    profile.TPSTART(QUDA_PROFILE_INIT);

    // Check to see that we're not trying to invert on a zero-field source
    const double b2 = blas::norm2(b);
    if (b2 == 0) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      printfQuda("Warning: inverting on zero-field source\n");
      x = b;
      param.true_res = 0.0;
      param.true_res_hq = 0.0;
      return 0;
    }

    ColorSpinorParam csParam(x);

    if (!init) {
      eigcg_args = new EigCGArgs(param.m, param.nev);//need only deflation meta structure

      csParam.create = QUDA_COPY_FIELD_CREATE;
      rp = ColorSpinorField::Create(b, csParam);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      yp = ColorSpinorField::Create(b, csParam);

      Ap = ColorSpinorField::Create(csParam);
      p  = ColorSpinorField::Create(csParam);

      tmpp = ColorSpinorField::Create(csParam);

      Az = ColorSpinorField::Create(csParam);

      if (K && param.precision_precondition != param.precision_sloppy) {
        csParam.setPrecision(param.precision_precondition);
        p_pre = ColorSpinorField::Create(csParam);
        r_pre = ColorSpinorField::Create(csParam);
      } 

      //Create a search vector set:
      csParam.setPrecision(param.precision_ritz);//eigCG internal search space precision may not coincide with the solver precision!
      csParam.is_composite  = true;
      csParam.composite_dim = param.m;

      Vm = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors

      eigcg_args->global_stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver

      init = true;
    }

    double local_stop = x.Precision() == QUDA_DOUBLE_PRECISION ? b2*param.tol*param.tol :  b2*1e-11;

    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction && param.inv_type == QUDA_INC_EIGCG_INVERTER) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      (*K)(x, b);
      return Kparam.iter; 
    }

//!
    csParam.setPrecision(QUDA_DOUBLE_PRECISION);

    csParam.create        = QUDA_ZERO_FIELD_CREATE;
    csParam.is_composite  = true;
    csParam.composite_dim = (2*args.k);

    args.V2k = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors
//!
    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &tmp = *tmpp;

    csParam.setPrecision(param.precision_sloppy);
    csParam.is_composite  = false;

    // compute initial residual
    matSloppy(r, x, y);
    double r2 = blas::xmyNorm(b, r);

    ColorSpinorField *z  = (K != nullptr) ? ColorSpinorField::Create(csParam) : rp;//

    if( K ) {//apply preconditioner
      if (param.precision_precondition == param.precision_sloppy) { r_pre = rp; p_pre = z; }

      ColorSpinorField &rPre = *r_pre;
      ColorSpinorField &pPre = *p_pre;

      blas::copy(rPre, r);
      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);
      blas::copy(*z, pPre);
    }

    *p = *z;
    blas::zero(y);

    const bool use_heavy_quark_res =
      (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double heavy_quark_res = 0.0;  // heavy quark res idual

    if (use_heavy_quark_res)  heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

    double pAp;
    double alpha=1.0, alpha_inv=1.0, beta=0.0, alpha_old_inv = 1.0;

    double lanczos_diag, lanczos_offdiag;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    double rMinvr = blas::reDotProduct(r,*z);
    //Begin EigCG iterations:
    args.restarts = 0;

    PrintStats("eigCG", k, r2, b2, heavy_quark_res);

    bool converged = convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq);

    while ( !converged && k < param.maxiter ) {
      matSloppy(*Ap, *p, tmp);  // tmp as tmp

      pAp    = blas::reDotProduct(*p, *Ap);
      alpha_old_inv =  alpha_inv;
      alpha         = rMinvr / pAp;
      alpha_inv     = 1.0 / alpha;

      lanczos_diag  = (alpha_inv + beta*alpha_old_inv);

      UpdateVm(*z, beta, sqrt(r2));

      r2 = blas::axpyNorm(-alpha, *Ap, r);
      if( K ) {//apply preconditioner
        ColorSpinorField &rPre = *r_pre;
        ColorSpinorField &pPre = *p_pre;

        blas::copy(rPre, r);
        commGlobalReductionSet(false);
        (*K)(pPre, rPre);
        commGlobalReductionSet(true);
        blas::copy(*z, pPre);
      }
      //
      double rMinvr_old   = rMinvr;
      rMinvr = K ? blas::reDotProduct(r,*z) : r2;
      beta                = rMinvr / rMinvr_old;
      blas::axpyZpbx(alpha, *p, y, *z, beta);

      //
      lanczos_offdiag  = (-sqrt(beta)*alpha_inv);
      args.SetLanczos(lanczos_diag, lanczos_offdiag);

      k++;

      PrintStats("eigCG", k, r2, b2, heavy_quark_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged = convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq) or convergence(r2, heavy_quark_res, local_stop, param.tol_hq);
    }

    args.ResetArgs();//eigCG cycle finished, this cleans V2k as well

    blas::xpy(y, x);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residuals
    matSloppy(r, x, y);
    param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

    PrintSummary("eigCG", k, r2, b2, args.global_stop, param.tol_hq);

    // reset the flops counters
    blas::flops = 0;
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    profile.TPSTOP(QUDA_PROFILE_FREE);
    return k;
  }

  int IncEigCG::initCGsolve(ColorSpinorField &x, ColorSpinorField &b) {
    int k = 0;
    //Start init CG iterations:
    deflated_solver *defl_p = static_cast<deflated_solver*>(param.deflation_op);
    Deflation &defl         = *(defl_p->defl);

    const double full_tol    = Kparam.tol;
    Kparam.tol         = Kparam.tol_restart;

    ColorSpinorParam csParam(x);

    csParam.create = QUDA_ZERO_FIELD_CREATE;

    ColorSpinorField *tmpp2 = ColorSpinorField::Create(csParam);//full precision accumulator
    ColorSpinorField &tmp2  = *tmpp2;
    ColorSpinorField *rp = ColorSpinorField::Create(csParam);//full precision residual
    ColorSpinorField &r = *rp;

    csParam.setPrecision(param.precision_ritz);

    ColorSpinorField *xp_proj =  ( param.precision_ritz == param.precision ) ? &x : ColorSpinorField::Create(csParam);
    ColorSpinorField &xProj = *xp_proj;

    ColorSpinorField *rp_proj =  ( param.precision_ritz == param.precision ) ? rp : ColorSpinorField::Create(csParam);
    ColorSpinorField &rProj = *rp_proj;

    int restart_idx  = 0;

    xProj = x;
    rProj = b; 
    //launch initCG:
    while((Kparam.tol >= full_tol) && (restart_idx < param.max_restart_num)) {
      restart_idx += 1;

      defl(xProj, rProj);
      x = xProj;

      K = new CG(mat, matPrecon, matPrecon, Kparam, profile);
      (*K)(x, b);
      delete K;

      mat(r, x, tmp2);
      blas::xpay(b, -1.0, r);

      xProj = x;
      rProj = r; 

      if(getVerbosity() >= QUDA_VERBOSE) printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", Kparam.iter, Kparam.secs, Kparam.gflops);

      Kparam.tol *= param.inc_tol;

      if(restart_idx == (param.max_restart_num-1)) Kparam.tol = full_tol;//do the last solve in the next cycle to full tolerance

      param.secs   += Kparam.secs;
    }

    if(getVerbosity() >= QUDA_VERBOSE) printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", Kparam.iter, Kparam.secs, Kparam.gflops);
    //
    param.secs   += Kparam.secs;
    param.gflops += Kparam.gflops;

    k   += Kparam.iter;

    delete rp;
    delete tmpp2;

    if( param.precision_ritz != param.precision ) {
      delete xp_proj;
      delete rp_proj;
    }
    return k;
  }

  void IncEigCG::operator()(ColorSpinorField &out, ColorSpinorField &in)
  {
     if(param.rhs_idx == 0) max_eigcg_cycles = param.eigcg_max_restarts;

     const bool mixed_prec = (param.precision != param.precision_sloppy);
     const double b2       = norm2(in);

     deflated_solver *defl_p = static_cast<deflated_solver*>(param.deflation_op);
     Deflation &defl         = *(defl_p->defl);

     //If deflation space is complete: use initCG solver
     if( defl.is_complete() ) {

       if (K) errorQuda("\nInitCG does not (yet) support preconditioning.");

       int iters = initCGsolve(out, in);
       param.iter += iters;

       return;
     } 

     //Start (incremental) eigCG solver:
     ColorSpinorParam csParam(in);
     csParam.create = QUDA_ZERO_FIELD_CREATE;

     ColorSpinorField *ep = ColorSpinorField::Create(csParam);//full precision accumulator
     ColorSpinorField &e = *ep;
     ColorSpinorField *rp = ColorSpinorField::Create(csParam);//full precision residual
     ColorSpinorField &r = *rp;

     //deflate initial guess ('out'-field):
     mat(r, out, e);
     //
     double r2 = xmyNorm(in, r);

     csParam.setPrecision(param.precision_sloppy);

     ColorSpinorField *ep_sloppy = ( mixed_prec ) ? ColorSpinorField::Create(csParam) : ep;
     ColorSpinorField &eSloppy = *ep_sloppy;
     ColorSpinorField *rp_sloppy = ( mixed_prec ) ? ColorSpinorField::Create(csParam) : rp;
     ColorSpinorField &rSloppy = *rp_sloppy;

     const double stop = b2*param.tol*param.tol;
     //start iterative refinement cycles (or just one eigcg call for full (solo) precision solver):
     int logical_rhs_id = 0;
     bool dcg_cycle    = false; 
     do {
       blas::zero(e);
       defl(e, r);
       //
       eSloppy = e, rSloppy = r;

       if( dcg_cycle ) { //run DCG instead
         if(!K) {
           Kparam.precision   = param.precision_sloppy;
           Kparam.tol         = 5*param.inc_tol;//former cg_iterref_tol param
           K = new CG(matSloppy, matPrecon, matPrecon, Kparam, profile);
         }

         eigcg_args->run_residual_correction = true;      
         printfQuda("Running DCG correction cycle.\n");
       }

       int iters = eigCGsolve(eSloppy, rSloppy);

       bool update_ritz = !dcg_cycle && (eigcg_args->restarts > 1) && !defl.is_complete(); //too uglyyy

       if( update_ritz ) { 

         defl.increment(*Vm, param.nev);
         logical_rhs_id += 1;

         dcg_cycle = (logical_rhs_id >= max_eigcg_cycles);

       } else { //run DCG instead
         dcg_cycle = true;
       }

       // use mixed blas ??
       e = eSloppy;
       blas::xpy(e, out);
       // compute the true residuals
       blas::zero(e);
       mat(r, out, e);
       //
       r2 = blas::xmyNorm(in, r);

       param.true_res = sqrt(r2 / b2);
       param.true_res_hq = sqrt(HeavyQuarkResidualNorm(out,r).z);
       PrintSummary( !dcg_cycle ? "EigCG:" : "DCG (correction cycle):", iters, r2, b2, stop, param.tol_hq);

       if( getVerbosity() >= QUDA_VERBOSE ) { 
         if( !dcg_cycle &&  (eigcg_args->restarts > 1) && !defl.is_complete() ) defl.verify();
       }
     } while ((r2 > stop) && mixed_prec);

     delete ep;
     delete rp;

     if(mixed_prec){
       delete ep_sloppy;
       delete rp_sloppy;
     }

     if (mixed_prec && max_eigcg_cycles > logical_rhs_id) {
       printfQuda("Reset maximum eigcg cycles to %d (was %d)\n", logical_rhs_id, max_eigcg_cycles);
       max_eigcg_cycles = logical_rhs_id;//adjust maximum allowed cycles based on the actual information
     }

     param.rhs_idx += logical_rhs_id;

     if(defl.is_complete()) {
       if(param.rhs_idx != param.deflation_grid) warningQuda("\nTotal rhs number (%d) does not match the deflation grid size (%d).\n", param.rhs_idx, param.deflation_grid);
       if(Vm) delete Vm;//safe some space
       Vm = nullptr;

       const int max_nev = defl.size();//param.m;
       printfQuda("\nRequested to reserve %d eigenvectors with max tol %le.\n", max_nev, param.eigenval_tol);
       defl.reduce(param.eigenval_tol, max_nev);
     }
     return;
  }

} // namespace quda
