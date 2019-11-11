#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>

#ifdef MAGMA_LIB
#include <blas_magma.h>
#endif

#include <algorithm>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>


/*
GMRES-DR algorithm:
R. B. Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24 (2002) p. 20-37
See also: A.Frommer et al, "Deflation and Flexible SAP-Preconditioning of GMRES in Lattice QCD simulations" ArXiv hep-lat/1204.5463
*/

namespace quda {

    using namespace blas;
    using namespace std;

    using namespace Eigen;

    using DynamicStride   = Stride<Dynamic, Dynamic>;

    using DenseMatrix     = MatrixXcd;
    using VectorSet       = MatrixXcd;
    using Vector          = VectorXcd;

//special types needed for compatibility with QUDA blas:
    using RowMajorDenseMatrix = Matrix<Complex, Dynamic, Dynamic, RowMajor>;

    struct SortedEvals{

      double _val;
      int    _idx;

      SortedEvals(double val, int idx) : _val(val), _idx(idx) {};
      static bool SelectSmall (SortedEvals v1, SortedEvals v2) { return (v1._val < v2._val);}
    };


    enum class libtype {eigen_lib, magma_lib, lapack_lib, mkl_lib};

    class GMResDRArgs{

      public:
       VectorSet   ritzVecs;
       DenseMatrix H;
       Vector      eta;

       int m;
       int k;
       int restarts;

       Complex      *c;

       ColorSpinorFieldSet *Vkp1;//high-precision accumulation array

       GMResDRArgs(int m, int nev) : ritzVecs(VectorSet::Zero(m+1,nev+1)), H(DenseMatrix::Zero(m+1,m)),
       eta(Vector::Zero(m)), m(m), k(nev), restarts(0), Vkp1(nullptr) { c = static_cast<Complex*> (ritzVecs.col(k).data()); }

       inline void ResetArgs() {
         ritzVecs.setZero();
         H.setZero();
         eta.setZero();
       }

       ~GMResDRArgs(){
          if(Vkp1) delete Vkp1;
       }
   };

   template<libtype which_lib> void ComputeHarmonicRitz(GMResDRArgs &args) {errorQuda("\nUnknown library type.\n");}

   template <> void ComputeHarmonicRitz<libtype::magma_lib>(GMResDRArgs &args)
   {
#ifdef MAGMA_LIB
     DenseMatrix cH = args.H.block(0, 0, args.m, args.m).adjoint();
     DenseMatrix Gk = args.H.block(0, 0, args.m, args.m);

     VectorSet  harVecs = MatrixXcd::Zero(args.m, args.m);
     Vector     harVals = VectorXcd::Zero(args.m);

     Vector em = VectorXcd::Zero(args.m);

     em(args.m-1) = norm( args.H(args.m, args.m-1) );

     cudaHostRegister(static_cast<void *>(cH.data()), args.m*args.m*sizeof(Complex), cudaHostRegisterDefault);
     magma_Xgesv(static_cast<void*>(em.data()), args.m, args.m, static_cast<void*>(cH.data()), args.m, sizeof(Complex));
     cudaHostUnregister(cH.data());

     Gk.col(args.m-1) += em;

     cudaHostRegister(static_cast<void *>(Gk.data()), args.m*args.m*sizeof(Complex), cudaHostRegisterDefault);
     magma_Xgeev(static_cast<void*>(Gk.data()), args.m, args.m, static_cast<void*>(harVecs.data()), static_cast<void*>(harVals.data()), args.m, sizeof(Complex));
     cudaHostUnregister(Gk.data());

     std::vector<SortedEvals> sorted_evals;
     sorted_evals.reserve(args.m);

     for(int e = 0; e < args.m; e++) sorted_evals.push_back( SortedEvals( abs(harVals.data()[e]), e ));
     std::stable_sort(sorted_evals.begin(), sorted_evals.end(), SortedEvals::SelectSmall);

     for(int e = 0; e < args.k; e++) memcpy(args.ritzVecs.col(e).data(), harVecs.col(sorted_evals[e]._idx).data(), (args.m)*sizeof(Complex));
#else
    errorQuda("Magma library was not built.\n");
#endif
     return;
   }


   template <> void ComputeHarmonicRitz<libtype::eigen_lib>(GMResDRArgs &args)
   {

     DenseMatrix cH = args.H.block(0, 0, args.m, args.m).adjoint();
     DenseMatrix Gk = args.H.block(0, 0, args.m, args.m);

     VectorSet  harVecs = MatrixXcd::Zero(args.m, args.m);
     Vector     harVals = VectorXcd::Zero(args.m);

     Vector em = VectorXcd::Zero(args.m);

     em(args.m-1) = norm( args.H(args.m, args.m-1) );
     Gk.col(args.m-1) += cH.colPivHouseholderQr().solve(em);

     ComplexEigenSolver<DenseMatrix> es( Gk );
     harVecs = es.eigenvectors();
     harVals = es.eigenvalues ();     

     std::vector<SortedEvals> sorted_evals;
     sorted_evals.reserve(args.m);

     for(int e = 0; e < args.m; e++) sorted_evals.push_back( SortedEvals( abs(harVals.data()[e]), e ));
     std::stable_sort(sorted_evals.begin(), sorted_evals.end(), SortedEvals::SelectSmall);

     for(int e = 0; e < args.k; e++) memcpy(args.ritzVecs.col(e).data(), harVecs.col(sorted_evals[e]._idx).data(), (args.m)*sizeof(Complex));

     return;
   }


    template<libtype which_lib> void ComputeEta(GMResDRArgs &args) {errorQuda("\nUnknown library type.\n");}

    template <> void ComputeEta<libtype::magma_lib>(GMResDRArgs &args) {
#ifdef MAGMA_LIB
       DenseMatrix Htemp(DenseMatrix::Zero(args.m+1,args.m));
       Htemp = args.H; 

       Complex *ctemp = static_cast<Complex*> (args.ritzVecs.col(0).data());
       memcpy(ctemp, args.c, (args.m+1)*sizeof(Complex));

       cudaHostRegister(static_cast<void*>(Htemp.data()), (args.m+1)*args.m*sizeof(Complex), cudaHostRegisterDefault);
       magma_Xgels(static_cast<void*>(Htemp.data()), ctemp, args.m+1, args.m, args.m+1, sizeof(Complex));
       cudaHostUnregister(Htemp.data());

       memcpy(args.eta.data(), ctemp, args.m*sizeof(Complex));
       memset(ctemp, 0, (args.m+1)*sizeof(Complex));
#else
       errorQuda("MAGMA library was not built.\n");
#endif
       return;
    }

    template <> void ComputeEta<libtype::eigen_lib>(GMResDRArgs &args) {

        Map<VectorXcd, Unaligned> c_(args.c, args.m+1);
        args.eta = args.H.jacobiSvd(ComputeThinU | ComputeThinV).solve(c_);

       return;
    }

    void fillFGMResDRInnerSolveParam(SolverParam &inner, const SolverParam &outer) {
      inner.tol = outer.tol_precondition;
      inner.maxiter = outer.maxiter_precondition;
      inner.delta = 1e-20; 
      inner.inv_type = outer.inv_type_precondition;

      inner.precision = outer.precision_precondition; 
      inner.precision_sloppy = outer.precision_precondition;

      inner.iter = 0;
      inner.gflops = 0;
      inner.secs = 0;

      inner.inv_type_precondition = QUDA_INVALID_INVERTER;
      inner.is_preconditioner = true; 
      inner.global_reduction  = false;
      if(inner.global_reduction) warningQuda("Set global reduction flag for preconditioner to true.\n");

      if (outer.precision_sloppy != outer.precision_precondition)
        inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
      else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
    }


 GMResDR::GMResDR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(nullptr), Kparam(param),
    Vm(nullptr), Zm(nullptr), profile(profile), gmresdr_args(nullptr), init(false)
 {
     fillFGMResDRInnerSolveParam(Kparam, param);

     if (param.inv_type_precondition == QUDA_CG_INVERTER)
       K = new CG(matPrecon, matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) 
       K = new BiCGstab(matPrecon, matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_MR_INVERTER) 
       K = new MR(matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_SD_INVERTER) 
       K = new SD(matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_INVALID_INVERTER) 
       K = nullptr;
     else
       errorQuda("Unsupported preconditioner %d\n", param.inv_type_precondition);


     return;
 }

 GMResDR::GMResDR(DiracMatrix &mat, Solver &K, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(&K), Kparam(param),
    Vm(nullptr), Zm(nullptr), profile(profile), gmresdr_args(nullptr), init(false) { }


 GMResDR::~GMResDR() {
    profile.TPSTART(QUDA_PROFILE_FREE);

    if(init)
    {
      delete Vm;
      Vm = nullptr;

      delete r_sloppy;

      if(K && (param.precision_precondition != param.precision_sloppy))
      {
        delete r_pre;
        delete p_pre;
      }

      if(K) {
        delete Zm;
        delete K;
      }

      delete tmpp;
      delete yp;
      delete rp;

      delete gmresdr_args;
    }

   profile.TPSTOP(QUDA_PROFILE_FREE);
 }


 void GMResDR::UpdateSolution(ColorSpinorField *x, ColorSpinorField *r, bool do_gels)
 {
   GMResDRArgs &args = *gmresdr_args;

   if(do_gels) {
     if (  param.extlib_type == QUDA_MAGMA_EXTLIB ) { 
       ComputeEta<libtype::magma_lib>(args);
     } else if (  param.extlib_type == QUDA_EIGEN_EXTLIB ) {
       ComputeEta<libtype::eigen_lib>(args);
     } else {
       errorQuda("Library type %d is currently not supported.\n", param.extlib_type);
     }
   }

   std::vector<ColorSpinorField*> Z_(Zm->Components().begin(),Zm->Components().begin()+args.m);
   std::vector<ColorSpinorField*> V_(Vm->Components());

   std::vector<ColorSpinorField*> x_, r_;
   x_.push_back(x), r_.push_back(r);

   blas::caxpy( static_cast<Complex*> ( args.eta.data()), Z_, x_);

   VectorXcd minusHeta = - (args.H * args.eta);
   Map<VectorXcd, Unaligned> c_(args.c, args.m+1);
   c_ += minusHeta;

   blas::caxpy(static_cast<Complex*>(minusHeta.data()), V_, r_);
   return;
 }


 void GMResDR::RestartVZH()
 {
   GMResDRArgs &args = *gmresdr_args;

   if ( param.extlib_type == QUDA_MAGMA_EXTLIB ) { 
     ComputeHarmonicRitz<libtype::magma_lib>(args);
   } else if(param.extlib_type == QUDA_EIGEN_EXTLIB) {
     ComputeHarmonicRitz<libtype::eigen_lib>(args);
   } else {
     errorQuda("Library type %d is currently not supported.\n", param.extlib_type);
   }

   DenseMatrix Qkp1(MatrixXcd::Identity((args.m+1), (args.k+1)));

   HouseholderQR<MatrixXcd> qr(args.ritzVecs);
   Qkp1.applyOnTheLeft( qr.householderQ());

   DenseMatrix Res = Qkp1.adjoint()*args.H*Qkp1.topLeftCorner(args.m, args.k);
   args.H.setZero();
   args.H.topLeftCorner(args.k+1, args.k) = Res;

   blas::zero( *args.Vkp1 );

   std::vector<ColorSpinorField*> vkp1(args.Vkp1->Components());
   std::vector<ColorSpinorField*> vm  (Vm->Components());

   RowMajorDenseMatrix Alpha(Qkp1);//convert Qkp1 to Row-major format first  
   blas::caxpy(static_cast<Complex*>(Alpha.data()), vm , vkp1); 

   for(int i = 0; i < (args.m+1); i++)
   {
     if(i < (args.k+1) )
     {
       blas::copy(Vm->Component(i), args.Vkp1->Component(i));
       blas::zero(args.Vkp1->Component(i));
     }
     else blas::zero(Vm->Component(i));
   }

   if( Zm->V() != Vm->V() )
   {
     std::vector<ColorSpinorField*> z (Zm->Components());
     std::vector<ColorSpinorField*> vk(args.Vkp1->Components().begin(),args.Vkp1->Components().begin()+args.k);

     RowMajorDenseMatrix Beta(Qkp1.topLeftCorner(args.m,args.k));
     blas::caxpy(static_cast<Complex*>(Beta.data()), z , vk);

     for(int i = 0; i < (args.m); i++)
     {
       if( i < (args.k) ) blas::copy(Zm->Component(i), args.Vkp1->Component(i));
       else               blas::zero(Zm->Component(i));
     }
   }

   checkCudaError();

   for(int j = 0; j < args.k; j++)
   {
     Complex alpha = cDotProduct(Vm->Component(j), Vm->Component(args.k));
     caxpy(-alpha, Vm->Component(j), Vm->Component(args.k));
   }

   blas::ax(1.0/ sqrt(blas::norm2(Vm->Component(args.k))), Vm->Component(args.k));

   args.ritzVecs.setZero();
   return;
 }


int GMResDR::FlexArnoldiProcedure(const int start_idx, const bool do_givens = false)
 {
   int j = start_idx;
   GMResDRArgs &args = *gmresdr_args;
   ColorSpinorField &tmp = *tmpp;

   std::unique_ptr<Complex[] > givensH((do_givens) ? new Complex[(args.m+1)*args.m] : nullptr);
   std::unique_ptr<Complex[] > cn((do_givens) ? new Complex[args.m] : nullptr);
   std::unique_ptr<double[]  > sn((do_givens) ? new double[args.m] : nullptr);

   Complex c0 = args.c[0];

   while( j < args.m ) 
   {
     if(K) {
       ColorSpinorField &inPre  = (param.precision_precondition != param.precision_sloppy) ? *r_pre : Vm->Component(j);
       ColorSpinorField &outPre = (param.precision_precondition != param.precision_sloppy) ? *p_pre : Zm->Component(j);

       if(param.precision_precondition != param.precision_sloppy) inPre = Vm->Component(j);
       zero(outPre);
       pushVerbosity(param.verbosity_precondition);
       (*K)( outPre ,inPre );
       popVerbosity();

       if(param.precision_precondition != param.precision_sloppy) Zm->Component(j) = outPre;
     }
     matSloppy(Vm->Component(j+1), Zm->Component(j), tmp);

     args.H(0, j) = cDotProduct(Vm->Component(0), Vm->Component(j+1));
     caxpy(-args.H(0, j), Vm->Component(0), Vm->Component(j+1));

     Complex h0 = do_givens ? args.H(0, j) : 0.0;

     for(int i = 1; i <= j; i++)
     {
        args.H(i, j) = cDotProduct(Vm->Component(i), Vm->Component(j+1));
        caxpy(-args.H(i, j), Vm->Component(i), Vm->Component(j+1));

        if(do_givens) {
           givensH[(args.m+1)*j+(i-1)] = conj(cn[i-1])*h0 + sn[i-1]*args.H(i,j);
           h0 = -sn[i-1]*h0 + cn[i-1]*args.H(i,j);
        }
     }

     args.H(j+1, j) = Complex(sqrt(norm2(Vm->Component(j+1))), 0.0);
     blas::ax( 1.0 / args.H(j+1, j).real(), Vm->Component(j+1));
     if(do_givens)
     {
       double inv_denom = 1.0 / sqrt(norm(h0)+norm(args.H(j+1,j)));
       cn[j] = h0 * inv_denom;
       sn[j] = args.H(j+1,j).real() * inv_denom;
       givensH[j*(args.m+1)+j] = conj(cn[j])*h0 + sn[j]*args.H(j+1,j);

       args.c[j+1] = -sn[j]*args.c[j];
       args.c[j]  *= conj(cn[j]);
     }

     j += 1;
   }

   if(do_givens)
   {
     Map<MatrixXcd, Unaligned, DynamicStride> givensH_(givensH.get(), args.m, args.m, DynamicStride(args.m+1,1) );
     memcpy(args.eta.data(),  args.c, args.m*sizeof(Complex));
     memset((void *)args.c, 0, (args.m + 1) * sizeof(Complex));
     args.c[0] = c0;

     givensH_.triangularView<Upper>().solveInPlace<OnTheLeft>(args.eta);

   } else {
     memset((void *)args.c, 0, (args.m + 1) * sizeof(Complex));

     std::vector<ColorSpinorField*> v_(Vm->Components().begin(), Vm->Components().begin()+args.k+1);
     std::vector<ColorSpinorField*> r_;
     r_.push_back(static_cast<ColorSpinorField*>(r_sloppy));

     blas::cDotProduct(args.c, v_, r_);

   }
   return (j-start_idx);
 }

 void GMResDR::operator()(ColorSpinorField &x, ColorSpinorField &b)
 {
    profile.TPSTART(QUDA_PROFILE_INIT);

    const double tol_threshold     = 1.2;
    const double det_max_deviation = 0.4;

    ColorSpinorField *ep = nullptr;

    if (!init) {

      gmresdr_args = new GMResDRArgs(param.m, param.nev);

      ColorSpinorParam csParam(b);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      rp = ColorSpinorField::Create(csParam);
      yp = ColorSpinorField::Create(csParam); 
      ep = ColorSpinorField::Create(csParam); 

      csParam.setPrecision(param.precision_sloppy);

      tmpp     = ColorSpinorField::Create(csParam);
      r_sloppy = ColorSpinorField::Create(csParam);

      if ( K && (param.precision_precondition != param.precision_sloppy) ) {

        csParam.setPrecision(param.precision_precondition);
        p_pre = ColorSpinorField::Create(csParam);
        r_pre = ColorSpinorField::Create(csParam);

      }

      csParam.setPrecision(param.precision_sloppy);
      csParam.is_composite  = true;
      csParam.composite_dim = param.m+1;

      Vm   = ColorSpinorFieldSet::Create(csParam); 

      csParam.composite_dim = param.m;

      Zm = K ? ColorSpinorFieldSet::Create(csParam) : Vm;


      csParam.composite_dim = (param.nev+1);
      csParam.setPrecision(QUDA_DOUBLE_PRECISION);

      gmresdr_args->Vkp1 = ColorSpinorFieldSet::Create(csParam);

      init = true;
    }

    GMResDRArgs &args = *gmresdr_args;

    ColorSpinorField &r   = *rp;
    ColorSpinorField &y   = *yp;
    ColorSpinorField &e   = *ep;

    ColorSpinorField &rSloppy = *r_sloppy;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    int tot_iters = 0;

    double normb = norm2( b );
    double stop  = param.tol*param.tol* normb;  

    mat(r, x);
    
    double r2 = xmyNorm(b, r);
    double b2 = r2;
    args.c[0] = Complex(sqrt(r2), 0.0);

    printfQuda("\nInitial residual squared: %1.16e, source %1.16e, tolerance %1.16e\n", r2, sqrt(normb), param.tol);

    rSloppy = r;

    if(param.precision_sloppy != param.precision) {
      blas::axpy(1.0 / args.c[0].real(), r, y);
      Vm->Component(0) = y;
      blas::zero(y);
    } else {
      blas::axpy(1.0 / args.c[0].real(), r, Vm->Component(0));   
    }

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    double heavy_quark_res = 0.0;  
    if (use_heavy_quark_res)  heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);


    int restart_idx = 0, j = 0, check_interval = 4;

    DenseMatrix Gm = DenseMatrix::Zero(args.k+1, args.k+1);

    while(restart_idx < param.deflation_grid && !(convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop)))
    {
      tot_iters += FlexArnoldiProcedure(j, (j == 0));
      UpdateSolution(&e, r_sloppy, !(j == 0));

      r2 = norm2(rSloppy);

      bool   do_clean_restart = false;
      double ext_r2 = 1.0;

      if((restart_idx+1) % check_interval) {
        mat(y, e);
        ext_r2 = xmyNorm(r, y);

	// can this be done as a single 2-d reduction?
        for(int l = 0; l < args.k+1; l++) {

          Complex *col = Gm.col(l).data();

	  std::vector<ColorSpinorField*> v1_(Vm->Components().begin(), Vm->Components().begin()+args.k+1);
	  std::vector<ColorSpinorField*> v2_;
	  v2_.push_back(static_cast<ColorSpinorField*>(&Vm->Component(l)));

	  blas::cDotProduct(col, v1_, v2_);

	}//end l-loop

	Complex detGm = Gm.determinant();

	PrintStats("FGMResDR:", tot_iters, r2, b2, heavy_quark_res);
	printfQuda("\nCheck cycle %d, true residual squared %1.15e, Gramm det : (%le, %le)\n", restart_idx, ext_r2, detGm.real(), detGm.imag());

	Gm.setZero();

	do_clean_restart = ((sqrt(ext_r2) / sqrt(r2)) > tol_threshold) || fabs(1.0 - (norm(detGm)) > det_max_deviation);
      }

      if( ((restart_idx != param.deflation_grid-1) && !do_clean_restart) ) {

	RestartVZH();
	j = args.k;

      } else {

       printfQuda("\nClean restart for cycle %d, true residual squared %1.15e\n", restart_idx, ext_r2);
       args.ResetArgs();

       //update solution:
       xpy(e, x);
       r = y;
       zero(e);

       args.c[0] = Complex(sqrt(ext_r2), 0.0);
       blas::zero(Vm->Component(0));
       blas::axpy(1.0 / args.c[0].real(), rSloppy, Vm->Component(0));

       j = 0;
     }

     restart_idx += 1;

   }

   //final solution:
   xpy(e, x);

   profile.TPSTOP(QUDA_PROFILE_COMPUTE);
   profile.TPSTART(QUDA_PROFILE_EPILOGUE);

   param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
   double gflops = (blas::flops + mat.flops())*1e-9;
   param.gflops = gflops;
   param.iter += tot_iters;

   mat(r, x);

   param.true_res = sqrt(xmyNorm(b, r) / b2);

   PrintSummary("FGMResDR:", tot_iters, r2, b2, stop, param.tol_hq);

   blas::flops = 0;
   mat.flops();

   profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

   param.rhs_idx += 1;

   if(ep) delete ep;
   return;
 }

} // namespace quda
