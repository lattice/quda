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
#include <string.h>

#include <face_quda.h>

#include <iostream>

#include <blas_magma.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <deflation.h>

//#define USE_MAGMA_HEEV

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

   class EigCGArgs{

     public:
       //host Lanczos matrice, and its eigenvalue/vector arrays:
       DenseMatrix Tm;//VH A V,
       //eigenvectors:
       VectorSet ritzVecs;//array of  (m)  ritz and of m length
       //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)
       RealVector Tmvals;//eigenvalues of T[m,  m  ] and T[m-1, m-1] (re-used)
       //Aux matrix for computing 2k Ritz vectors:
       DenseMatrix H2k;

       int m;
       int k;

       int restarts;
       double global_stop;

       EigCGArgs(int m, int k) : Tm(DenseMatrix::Zero(m,m)), ritzVecs(VectorSet::Zero(m,m)), Tmvals(m), H2k(2*k, 2*k),  
       m(m), k(k), restarts(0), global_stop(0.0) { }

       ~EigCGArgs() { }

       //methods for constructing Lanczos matrix :
       inline void ResetArgs() { 
         Tm.setZero();  
         Tmvals.setZero();  
         ritzVecs.setZero(); 
       }

       template<int diag>
       inline void SetLanczos(int idx, Complex val){ Tm.diagonal<diag>()[idx] = val; } 
       inline void RestartLanczos(ColorSpinorField *w, ColorSpinorFieldSet *v, const double inv_sqrt_r2);
       //methods for Rayleigh Ritz procedure:
       void ComputeRitz();

   };


   void EigCGArgs::RestartLanczos(ColorSpinorField *w, ColorSpinorFieldSet *v, const double inv_sqrt_r2)
   {
     Tm.setZero();

     Complex *s  = new Complex[2*k];

     for(int i = 0; i < 2*k; i++) Tm(i,i) = Tmvals(i);//??
    
     const int cdot_pipeline_length  = 5;
     int offset = 0;
   
     do {
        const int local_length = (2*k - offset) > cdot_pipeline_length  ? cdot_pipeline_length : (2*k - offset) ;

        std::vector<cudaColorSpinorField*> v_;
        std::vector<cudaColorSpinorField*> w_;
        v_.reserve(local_length);
        w_.reserve(local_length);

        for(int i = 0; i < local_length; i++)
        {
          v_.push_back(static_cast<cudaColorSpinorField*>(&v->Component(offset+i)));
          w_.push_back(static_cast<cudaColorSpinorField*>(w));
        }
        //Warning! this won't work with arbitrary (big) param.cur_dim. That's why pipelining is needed.
        blas::cDotProduct(&s[offset], w_, v_);//<i, b>
    
        offset += cdot_pipeline_length; 

     } while (offset < 2*k);

     Map<VectorXcd, Unaligned > s_(s, 2*k);
     s_ *= inv_sqrt_r2;

     Tm.col(2*k).segment(0, 2*k) = s_;
     Tm.row(2*k).segment(0, 2*k) = s_.adjoint();

     delete [] s;

     return;
   }

   void EigCGArgs::ComputeRitz()
   {
#if (not defined USE_MAGMA_HEEV) //Eigen version
     //Solve m dim eigenproblem:
     SelfAdjointEigenSolver<MatrixXcd> es_tm(Tm); 
     ritzVecs.leftCols(k) = es_tm.eigenvectors().leftCols(k);
     //Solve m-1 dim eigenproblem:
     SelfAdjointEigenSolver<MatrixXcd> es_tm1(Map<MatrixXcd, Unaligned, DynamicStride >(Tm.data(), (m-1), (m-1), DynamicStride(m, 1)));
     Block<MatrixXcd>(ritzVecs.derived(), 0, k, m-1, k) = es_tm1.eigenvectors().leftCols(k);
     ritzVecs.block(m-1, k, 1, k).setZero();
#else
     //Solve m dim eigenproblem:
     ritzVecs = Tm;
     Complex *evecm = static_cast<Complex*>( ritzVecs.data());
     double  *evalm = static_cast<double *>(Tmvals.data()); 

     cudaHostRegister(static_cast<void *>(evecm), m*m*sizeof(Complex),  cudaHostRegisterDefault);
     magma_Xheev(evecm, m, m, evalm, sizeof(Complex));
     //Solve m-1 dim eigenproblem:
     DenseMatrix ritzVecsm1(Tm);
     Complex *evecm1 = static_cast<Complex*>( ritzVecsm1.data());

     cudaHostRegister(static_cast<void *>(evecm1), m*m*sizeof(Complex),  cudaHostRegisterDefault);
     magma_Xheev(evecm1, (m-1), m, evalm, sizeof(Complex));
     // fill 0s in mth element of old evecs:
     for(int l = 1; l <= m ; l++) evecm1[l*m-1] = 0.0 ;
     // Attach the first nev old evecs at the end of the nev latest ones:
     memcpy(&evecm[k*m], evecm1, k*m*sizeof(Complex));
#endif
    // Orthogonalize the 2*nev (new+old) vectors evecm=QR:

     MatrixXcd Q2k(MatrixXcd::Identity(m, 2*k));
     HouseholderQR<MatrixXcd> ritzVecs2k_qr( Map<MatrixXcd, Unaligned >(ritzVecs.data(), m, 2*k) );
     Q2k.applyOnTheLeft( ritzVecs2k_qr.householderQ() );

     //2. Construct H = QH*Tm*Q :
     H2k = Q2k.adjoint()*Tm*Q2k;

     /* solve the small evecm1 2nev x 2nev eigenproblem */
     SelfAdjointEigenSolver<MatrixXcd> es_h2k(H2k);
     Block<MatrixXcd>(ritzVecs.derived(), 0, 0, m, 2*k) = Q2k * es_h2k.eigenvectors(); 
     Tmvals.segment(0,2*k) = es_h2k.eigenvalues();//this is ok

#ifdef USE_MAGMA_HEEV
     cudaHostUnregister(evecm);
     cudaHostUnregister(evecm1);
#endif

     return;
  }


  // set the required parameters for the inner solver
  static void fillInnerSolverParam(SolverParam &inner, const SolverParam &outer, bool use_sloppy_partial_accumulator = true)
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

  IncEigCG::IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(nullptr), Vm(nullptr), r_pre(nullptr), p_pre(nullptr), eigcg_args(nullptr), profile(profile), init(false)
  {
    if((param.rhs_idx < param.deflation_grid))  printfQuda("\nInitialize eigCG(m=%d, nev=%d) solver.\n", param.m, param.nev);
    else  errorQuda("\nDeflation space is complete, nothing to do for the eigCG solver.\n");

    fillInnerSolverParam(Kparam, param);

    if(param.inv_type_precondition == QUDA_CG_INVERTER){
      K = new CG(matPrecon, matPrecon, Kparam, profile);
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
      if(Vm) delete Vm;

      delete tmpp;
      delete rp;
      delete yp;
      delete Ap;
      delete p;

      if(Az) delete Az;

      if(K) {
        delete r_pre;
        delete p_pre;
      }
  
      delete eigcg_args;
    }

  }

  void IncEigCG::RestartVT(const double beta, const double rho)
  {
    EigCGArgs &args = *eigcg_args;

    args.ComputeRitz();

    //Create intermediate model:
    ColorSpinorParam csParam(Vm->Component(0));
    //
    csParam.is_composite  = true;
    csParam.composite_dim = (2*args.k);
    csParam.setPrecision(QUDA_DOUBLE_PRECISION);
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    ColorSpinorFieldSet *V2k = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors
    //Restart V:
    {
      std::vector<ColorSpinorField*> vm(Vm->Components().begin(),Vm->Components().end());
      for(int i = 0; i < 2*args.k; i++) 
      {
         std::vector<ColorSpinorField*> v2k(V2k->Components().begin()+i,V2k->Components().begin()+i+1);
         blas::caxpy( static_cast<Complex*>(args.ritzVecs.col(i).data()), vm , v2k ); 
      }
    }

    for(int i = 0; i < 2*args.k; i++)  blas::copy(Vm->Component(i), V2k->Component(i));

    delete V2k;
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


/*
 * This is a solo precision solver.
*/
  int IncEigCG::eigCGsolve(ColorSpinorField &x, ColorSpinorField &b) {
    if (Location(x, b) != QUDA_CUDA_FIELD_LOCATION)  errorQuda("Not supported");

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

    double local_stop = b2*1e-11;

    EigCGArgs &args = *eigcg_args;

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &tmp = *tmpp;

    csParam.setPrecision(param.precision_sloppy);
    csParam.is_composite  = false;
    csParam.create        = QUDA_ZERO_FIELD_CREATE;

    // compute initial residual
    matSloppy(r, x, y);
    double r2 = blas::xmyNorm(b, r);

    ColorSpinorField *z  = (K != nullptr) ? ColorSpinorField::Create(csParam) : rp;//

    if( K ) {//apply preconditioner
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

    const int heavy_quark_check = param.heavy_quark_check; // how often to check the heavy quark residual

    double pAp;
    double alpha=1.0, alpha_inv=1.0, beta=0.0, alpha_old_inv = 1.0;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    double rMinvr = blas::reDotProduct(r,*z);
    //Begin EigCG iterations:
    int k=0, l=0;
    args.restarts = 0; 

    PrintStats("eigCG", k, r2, b2, heavy_quark_res);

    bool converged = convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq);

    while ( !converged && k < param.maxiter ) {
      matSloppy(*Ap, *p, tmp);  // tmp as tmp

      pAp    = blas::reDotProduct(*p, *Ap);
      alpha_old_inv =  alpha_inv;
      alpha         = rMinvr / pAp;
      alpha_inv     = 1.0 / alpha;

      if (l == param.m){//Begin Rayleigh-Ritz block: 
         //
         RestartVT(beta, sqrt(r2));
         l = 2*args.k;
         //
         args.restarts += 1;
      }
      //load Lanczos basis vector:
      blas::copy(Vm->Component(l), *z);//convert arrays
      //rescale the vector
      blas::ax(1.0 / sqrt(r2), Vm->Component(l));
      //Load Lanczos diagonal (off-diagonals will be loaded after beta computation)
      args.SetLanczos<0>(l, (alpha_inv + beta*alpha_old_inv));

      r2 = blas::axpyNorm(-alpha, *Ap, r);
#if 0
      {
        double r2_old = r2;
        // here we are deploying the alternative beta computation
        Complex cg_norm = blas::axpyCGNorm(-alpha, *Ap, r);
        r2 = real(cg_norm);  // (r_new, r_new)
        double sigma = imag(cg_norm) >= 0.0 ? imag(cg_norm) : r2;  // use r2 if (r_k+1, r_k+1-r_k) breaks
        //
        beta = sigma / r2_old;  // use the alternative beta computation
	blas::axpyZpbx(alpha, *p, y, r, beta);

	if (use_heavy_quark_res && k%heavy_quark_check==0) heavy_quark_res = sqrt(blas::xpyHeavyQuarkResidualNorm(x, y, r).z);
      }
#endif
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
      l += 1;       

      if ( l == param.m )  blas::copy(*Az, *Ap);//save previous mat-vec result if ready for the restart 
      else { //Load Lanczos off-diagonals:
        double off_diagonal = (-sqrt(beta)*alpha_inv);
        args.SetLanczos<+1>(l-1,off_diagonal);
        args.SetLanczos<-1>(l-1,off_diagonal);
      }

      k++;

      PrintStats("eigCG", k, r2, b2, heavy_quark_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently
      converged = convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq) or convergence(r2, heavy_quark_res, local_stop, param.tol_hq);
    }

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

    PrintSummary("eigCG", k, r2, b2);

    // reset the flops counters
    blas::flops = 0;
    matSloppy.flops();

    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    profile.TPSTOP(QUDA_PROFILE_FREE);

    return k;
  }

  void IncEigCG::operator()(ColorSpinorField &out, ColorSpinorField &in)
  {
     const bool mixed_prec = (param.precision != param.precision_sloppy);
     const double b2 = norm2(in);

     //deflate initial guess ('out'-field):
     deflated_solver *defl_p = static_cast<deflated_solver*>(param.deflation_op); 
     Deflation &defl         = *(defl_p->defl);

     ColorSpinorParam csParam(in);
     csParam.create = QUDA_ZERO_FIELD_CREATE;

     ColorSpinorField *ep = ColorSpinorField::Create(csParam);//full precision accumulator
     ColorSpinorField &e = *ep;
     ColorSpinorField *rp = ColorSpinorField::Create(csParam);//full precision residual
     ColorSpinorField &r = *rp;

     //y = out; r = in;
     mat(r, out, e);
     //
     double r2 = xmyNorm(in, r);

     csParam.setPrecision(param.precision_sloppy);

     ColorSpinorField *ep_sloppy = ( mixed_prec ) ? ColorSpinorField::Create(csParam) : ep;
     ColorSpinorField &eSloppy = *ep_sloppy;
     ColorSpinorField *rp_sloppy = ( mixed_prec ) ? ColorSpinorField::Create(csParam) : rp;
     ColorSpinorField &rSloppy = *rp_sloppy;

     csParam.setPrecision(param.precision_ritz);
     ColorSpinorField *ep_proj = ( param.precision_ritz == param.precision_sloppy ) ? ep_sloppy : ( ( param.precision_ritz == param.precision ) ? ep : ColorSpinorField::Create(csParam) );
     ColorSpinorField &eProj = *ep_proj;

     ColorSpinorField *rp_proj = ( param.precision_ritz == param.precision_sloppy ) ? rp_sloppy : ( ( param.precision_ritz == param.precision ) ? rp : ColorSpinorField::Create(csParam) );
     ColorSpinorField &rProj = *rp_proj;

     const double stop = b2*param.tol*param.tol;
     //start reliable updates (or just one cycle for full precision solver):
     do {
       rProj = r;
       blas::zero(eProj);
       defl(eProj, rProj);
       //
       eSloppy = eProj, rSloppy = rProj; 

       int iters = eigCGsolve(eSloppy, rSloppy);
       if( eigcg_args->restarts > 0 ) defl.increment(*Vm, param.nev);
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
       PrintSummary("EigCG:", iters, r2, b2);

       defl.verify();
     } while ((r2 > stop) && mixed_prec); 

     if( ep_proj != ep && ep_proj != ep_sloppy ) {
       delete ep_proj;
       delete rp_proj;
     }

     delete ep;
     delete rp;

     if(mixed_prec){
       delete ep_sloppy;
       delete rp_sloppy;
     }
      
     param.rhs_idx += 1;

     return;
  }


} // namespace quda
