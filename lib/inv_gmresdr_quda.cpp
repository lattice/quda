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
#include <algorithm>

#include <Eigen/Dense>


/*
GMRES-DR algorithm:
R. B. Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24 (2002) p. 20-37
See also: A.Frommer et al, "Deflation and Flexible SAP-Preconditioning of GMRES in Lattice QCD simulations" ArXiv hep-lat/1204.5463
*/

namespace quda {
//Notes:
//GMResDR does not require large m (and esp. nev), so this will use normal LAPACK routines.
    using namespace blas;

    using namespace Eigen;
    using namespace std;

    using DynamicStride   = Stride<Dynamic, Dynamic>;

    using DenseMatrix     = MatrixXcd;
    using VectorSet       = MatrixXcd;
    using Vector          = VectorXcd;

    struct SortedEvals{

      double _val;
      int    _idx;

      SortedEvals(double val, int idx) : _val(val), _idx(idx) {};
      static bool SelectSmall (SortedEvals v1, SortedEvals v2) { return (v1._val < v2._val);}
    };

    enum class libtype {eigen_lib, magma_lib, lapack_lib, mkl_lib};

    class GMResDRArgs{

      public:

       VectorSet   ritzVecs;//array of sorted harmonic eigenvectors: (nev+1) length      
       DenseMatrix H;//Hessenberg matrix
       Vector      eta;
      
       int m;
       int k;
       int restarts;

       Complex      *c;//  c = &ritzVecs[(m+1)*k] => ritzVecs.col(k)

       GMResDRArgs(int m, int nev) : ritzVecs(VectorSet::Zero(m+1,nev+1)), H(DenseMatrix::Zero(m+1,m)), 
       eta(Vector::Zero(m)), m(m), k(nev), restarts(0) { c = static_cast<Complex*> (ritzVecs.col(k).data()); }

       void ComputeHarmonicRitz();

       inline void ResetArgs() { 
         ritzVecs.setZero();
         H.setZero();  
         eta.setZero();  
       }

       ~GMResDRArgs(){ }
   };

#define USE_MAGMA

   void GMResDRArgs::ComputeHarmonicRitz()
   {
     //1. initialize Eigen objects:
     DenseMatrix cH = H.block(0, 0, m, m).adjoint();
     DenseMatrix Gk = H.block(0, 0, m, m);
     //
     VectorSet  harVecs = MatrixXcd::Zero(m, m);
     Vector     harVals = VectorXcd::Zero(m);

     //2. construct H + beta*H^{-H} e_m*e_m^{T}
     //2.a need to solve H^{H}y = beta e_m;
     Vector em = VectorXcd::Zero(m);//in fact, em^{T}=(0,....,1)
     //2.b construct beta*em
     em(m-1) = norm( H(m, m-1) );//in fact, we construct beta*em
     //2.c Compute y = H^{-H} beta*e_m:   
#ifdef USE_MAGMA 
     cudaHostRegister(static_cast<void *>(cH.data()), m*m*sizeof(Complex), cudaHostRegisterDefault);
     magma_Xgesv(static_cast<void*>(em.data()), m, m, static_cast<void*>(cH.data()), m, sizeof(Complex));
     cudaHostUnregister(cH.data());
#else
//     RowVectorXcd sol = _cH.fullPivLu().solve(em);
//or
//     RowVectorXcd sol = _cH.colPivHouseholderQr().solve(_em);
//     em = sol;     
#endif
//     em.applyOnTheLeft(_Gk);
     //2.d Adjust last column with ((H^{-H}*beta*em)=em, em^{T}=[0,....,1]):
     Gk.col(m-1) += em;

     //3.  Compute harmonic eigenpairs:
#ifdef USE_MAGMA 
     cudaHostRegister(static_cast<void *>(Gk.data()), m*m*sizeof(Complex), cudaHostRegisterDefault);
     magma_Xgeev(static_cast<void*>(Gk.data()), m, m, static_cast<void*>(harVecs.data()), static_cast<void*>(harVals.data()), m, sizeof(Complex));//check it!
     cudaHostUnregister(Gk.data());
#else
     //
#endif
     //4. do sort:
     std::vector<SortedEvals> sorted_evals;
     sorted_evals.reserve(m);

     for(int e = 0; e < m; e++) sorted_evals.push_back( SortedEvals( abs(harVals.data()[e]), e ));
     //
     std::stable_sort(sorted_evals.begin(), sorted_evals.end(), SortedEvals::SelectSmall);

     //5. Copy sorted eigenvectors:
     for(int e = 0; e < k; e++) memcpy(ritzVecs.col(e).data(), harVecs.col(sorted_evals[e]._idx).data(), (m)*sizeof(Complex));//CHECK!

     return;
   }

    //Solve least sq:
    template<libtype which_lib> void ComputeEta(GMResDRArgs &args) {errorQuda("\nUnknown library type.\n");}

    //pure magma version: 
    template <> void ComputeEta<libtype::magma_lib>(GMResDRArgs &args) {

       DenseMatrix Htemp(DenseMatrix::Zero(args.m+1,args.m));
       Htemp = args.H; //need this copy

       Complex *ctemp = static_cast<Complex*> (args.ritzVecs.col(0).data());
       memcpy(ctemp, args.c, (args.m+1)*sizeof(Complex));

       cudaHostRegister(static_cast<void*>(Htemp.data()), (args.m+1)*args.m*sizeof(Complex), cudaHostRegisterDefault);
       magma_Xgels(static_cast<void*>(Htemp.data()), ctemp, args.m+1, args.m, args.m+1, sizeof(Complex));
       cudaHostUnregister(Htemp.data());

       memcpy(args.eta.data(), ctemp, args.m*sizeof(Complex));
       memset(ctemp, 0, (args.m+1)*sizeof(Complex));

       return;
    }

    //pure magma version: 
    template <> void ComputeEta<libtype::eigen_lib>(GMResDRArgs &args) {

        Map<VectorXcd, Unaligned> c_(args.c, args.m+1);
        args.eta = args.H.jacobiSvd(ComputeThinU | ComputeThinV).solve(c_);

       return;
    }


    // set the required parameters for the inner solver
    void fillInnerSolveParam_(SolverParam &inner, const SolverParam &outer) {
      inner.tol = outer.tol_precondition;
      inner.maxiter = outer.maxiter_precondition;
      inner.delta = 1e-20; // no reliable updates within the inner solver
  
      inner.precision = outer.precision_precondition; // preconditioners are uni-precision solvers
      inner.precision_sloppy = outer.precision_precondition;
  
      inner.iter = 0;
      inner.gflops = 0;
      inner.secs = 0;

      inner.inv_type_precondition = QUDA_INVALID_INVERTER;
      inner.is_preconditioner = true; // tell inner solver it is a preconditionis_re

      inner.global_reduction = false;

      inner.use_init_guess = QUDA_USE_INIT_GUESS_NO;

      if (outer.inv_type == QUDA_GCR_INVERTER && outer.precision_sloppy != outer.precision_precondition) 
        inner.preserve_source = QUDA_PRESERVE_SOURCE_NO;
      else inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;
    }

 GMResDR::GMResDR(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(nullptr), 
    Vm(nullptr), Zm(nullptr), profile(profile), gmresdr_args(nullptr), init(false)
 {
     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMResDR is not currently supported.\n");
     //
     fillInnerSolveParam_(Kparam, param);

     if (param.inv_type_precondition == QUDA_CG_INVERTER) // inner CG preconditioner
       K = new CG(matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
       K = new BiCGstab(matPrecon, matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_MR_INVERTER) // inner MR preconditioner
       K = new MR(matPrecon, matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_SD_INVERTER) // inner MR preconditioner
       K = new SD(matPrecon, Kparam, profile);
     else if (param.inv_type_precondition == QUDA_INVALID_INVERTER) // unknown preconditioner
       K = nullptr;
     else 
       errorQuda("Unsupported preconditioner %d\n", param.inv_type_precondition);

     return;
 }

 GMResDR::GMResDR(DiracMatrix &mat, Solver &K, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(&K), 
    Vm(nullptr), Zm(nullptr), profile(profile), gmresdr_args(nullptr), init(false) { }


 GMResDR::~GMResDR() {
    profile.TPSTART(QUDA_PROFILE_FREE);

    if(init)
    {
      delete Vm;
      Vm = nullptr;

      if(K) delete Zm;

      if (param.precision_sloppy != param.precision)
      {
        delete r_sloppy;
        delete x_sloppy;
      }

      if(K && (param.precision_precondition != param.precision_sloppy)) 
      {
        delete r_pre;
        delete p_pre;
      }

      delete tmpp;
      delete yp;
      delete rp;

      delete gmresdr_args;
    }

   profile.TPSTOP(QUDA_PROFILE_FREE);
 }
#define EIGEN_GELS
 void GMResDR::UpdateSolution(ColorSpinorField *x, ColorSpinorField *r, bool do_gels)
 {
   GMResDRArgs &args = *gmresdr_args;
#ifdef EIGEN_GELS
   if(do_gels) ComputeEta<libtype::eigen_lib>(args);
#else
   if(do_gels) ComputeEta<libtype::magma_lib>(args);
#endif

   //compute x = x0+Wm*lsqSol, where Wm is Vm or Zm (if preconditioned):
   std::vector<ColorSpinorField*> Z_(Zm->Components().begin(),Zm->Components().begin()+args.m);//warning: without preconditioner, Zm is an alias of Vm and has length args.m+1
   std::vector<ColorSpinorField*> V_(Vm->Components());

   std::vector<ColorSpinorField*> x_, r_;
   x_.push_back(x), r_.push_back(r);

   blas::caxpy( static_cast<Complex*> ( args.eta.data()), Z_, x_);

   VectorXcd minusHeta = - (args.H * args.eta);
   //Compute "short" residual:
   Map<VectorXcd, Unaligned> c_(args.c, args.m+1);
   c_ += minusHeta; 

   //Compute iterative residual:
   blas::caxpy(static_cast<Complex*>(minusHeta.data()), V_, r_);

   return;
 }

 void GMResDR::RestartVZH()
 {
   GMResDRArgs &args = *gmresdr_args;

   args.ComputeHarmonicRitz(); 

   //1. QR of the eigenvctor matrix:
   DenseMatrix Qkp1(MatrixXcd::Identity((args.m+1), (args.k+1)));
  
   HouseholderQR<MatrixXcd> qr(args.ritzVecs);
   Qkp1.applyOnTheLeft( qr.householderQ());

   //2. Update H:
   DenseMatrix Res = Qkp1.adjoint()*args.H*Qkp1.topLeftCorner(args.m, args.k);
   args.H.setZero();
   args.H.topLeftCorner(args.k+1, args.k) = Res;

   //2. Update Vm+1 : Vk+1 = Vm+1 Qk+1
   //Call multiblas:
   //Create intermediate model:
   ColorSpinorParam csParam(Vm->Component(0));

   csParam.is_composite  = true;
   csParam.composite_dim = (args.k+1);
   csParam.create = QUDA_ZERO_FIELD_CREATE;  
   csParam.setPrecision(QUDA_DOUBLE_PRECISION);

   ColorSpinorFieldSet *Vkp1 = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors

   std::vector<ColorSpinorField*> V(Vm->Components());

   for(int i = 0; i < (args.k+1); i++) 
   {
     std::vector<ColorSpinorField*> Vi(Vkp1->Components().begin()+i,Vkp1->Components().begin()+i+1);
     blas::caxpy(static_cast<Complex*>(Qkp1.col(i).data()), V , Vi);//use mixed multiblas here
   }

   for(int i = 0; i < (args.m+1); i++)  
   {
     if(i < (args.k+1) ) 
     {
       blas::copy(Vm->Component(i), Vkp1->Component(i));  
       blas::zero(Vkp1->Component(i));
     }
     else blas::zero(Vm->Component(i));
   }

   //3. Zk = Zm Qk
   if( Zm->V() != Vm->V() )
   {
     DenseMatrix Qk = Qkp1.topLeftCorner(args.m,args.k);

     std::vector<ColorSpinorField*> Z(Zm->Components());

     for(int i = 0; i < args.k; i++) 
     {
       std::vector<ColorSpinorField*> Vi(Vkp1->Components().begin()+i,Vkp1->Components().begin()+i+1);
       blas::caxpy(static_cast<Complex*>(Qkp1.col(i).data()), Z , Vi);//use mixed multiblas here
     }

     for(int i = 0; i < (args.m); i++)  
     {
       if( i < (args.k) ) blas::copy(Zm->Component(i), Vkp1->Component(i));  
       else               blas::zero(Zm->Component(i));
     }
   }

   delete Vkp1;

   checkCudaError();

   /*****REORTH V_{nev+1}:****/

   for(int j = 0; j < args.k; j++)
   {
     Complex alpha = cDotProduct(Vm->Component(j), Vm->Component(args.k));//
     caxpy(-alpha, Vm->Component(j), Vm->Component(args.k));
   }

   blas::ax(1.0/ sqrt(blas::norm2(Vm->Component(args.k))), Vm->Component(args.k));

   args.ritzVecs.setZero();

   return;
 }

 int GMResDR::RunFlexArnoldiProcess(const int start_idx, bool do_givens = false)//bool => const bool
 {
   GMResDRArgs &args = *gmresdr_args;
   ColorSpinorField &tmp = *tmpp;

   Complex *givensH = (do_givens) ? new Complex[(args.m+1)*args.m] : nullptr;
   Complex *cn      = (do_givens) ? new Complex[args.m]            : nullptr;
   double  *sn      = (do_givens) ? new double [args.m]            : nullptr;

   Complex c0 = args.c[0];//keep the first element 

   int j = start_idx;

   while( j < args.m ) //run full cycle
   {
     if(K) {
       ColorSpinorField &inPre  = (param.precision_precondition != param.precision_sloppy) ? *r_pre : Vm->Component(j);
       ColorSpinorField &outPre = (param.precision_precondition != param.precision_sloppy) ? *p_pre : Zm->Component(j);
       //aliases hadled automatically.
       inPre = Vm->Component(j);
       (*K)( outPre ,inPre );
       Zm->Component(j) = outPre;
     }

     matSloppy(Vm->Component(j+1), Zm->Component(j), tmp);

     args.H(0, j) = cDotProduct(Vm->Component(0), Vm->Component(j+1));//
     caxpy(-args.H(0, j), Vm->Component(0), Vm->Component(j+1));     

     Complex h0 = do_givens ? args.H(0, j) : 0.0;
     
     for(int i = 1; i <= j; i++)
     {
        args.H(i, j) = cDotProduct(Vm->Component(i), Vm->Component(j+1));//
        caxpy(-args.H(i, j), Vm->Component(i), Vm->Component(j+1));

        if(do_givens) {
           givensH[(args.m+1)*j+(i-1)] = conj(cn[i-1])*h0 + sn[i-1]*args.H(i,j);
           h0 = -sn[i-1]*h0 + cn[i-1]*args.H(i,j); 
        }
     }

     //6. Compute h(j+1,j):
     args.H(j+1, j) = Complex(sqrt(norm2(Vm->Component(j+1))), 0.0);
     //7. Scale the last arnoldi vector:
     blas::ax( 1.0 / args.H(j+1, j).real(), Vm->Component(j+1));
     //
     if(do_givens)
     {
       double inv_denom = 1.0 / sqrt(norm(h0)+norm(args.H(j+1,j)));
       //8. Update Givens coefficients:
       cn[j] = h0 * inv_denom;
       //
       sn[j] = args.H(j+1,j).real() * inv_denom;
       //9. Compute diagonal element in G:
       givensH[j*(args.m+1)+j] = conj(cn[j])*h0 + sn[j]*args.H(j+1,j);
      
       //10. Compute iter residual:
       args.c[j+1] = -sn[j]*args.c[j];
       args.c[j]  *= conj(cn[j]);
       //printfQuda("\nResidual: %le\n", std::norm(args.c[j+1]));
     }

     j += 1;
   }//end of main loop.

   //11 Update solution:
   //11.a Solve least squares:

   if(do_givens)
   {
     Map<MatrixXcd, Unaligned, DynamicStride> givensH_(givensH, args.m, args.m, DynamicStride(args.m+1,1) );//extract triangular part
     memcpy(args.eta.data(),  args.c, args.m*sizeof(Complex));
     //restore the orig array:
     memset(args.c, 0, (args.m+1)*sizeof(Complex));
     args.c[0] = c0;

     givensH_.triangularView<Upper>().solveInPlace<OnTheLeft>(args.eta);

     delete[] givensH;
     delete[] cn;
     delete[] sn;

   } else {
     const int cdot_pipeline_length  = 5;
     int offset = 0;
     memset(args.c, 0, (args.m+1)*sizeof(Complex));
 
     do {
        const int local_length = ((args.k+1) - offset) > cdot_pipeline_length  ? cdot_pipeline_length : ((args.k+1) - offset) ;

        std::vector<cudaColorSpinorField*> v_;
        std::vector<cudaColorSpinorField*> r_;
        v_.reserve(local_length);
        r_.reserve(local_length);

        for(int i = 0; i < local_length; i++)
        {
          v_.push_back(static_cast<cudaColorSpinorField*>(&Vm->Component(offset+i)));
          r_.push_back(static_cast<cudaColorSpinorField*>(r_sloppy));
        }
        //Warning! this won't work with arbitrary (big) param.cur_dim. That's why pipelining is needed.
        blas::cDotProduct(&args.c[offset], v_, r_);//<i, b>
        //
        offset += cdot_pipeline_length; 

     } while (offset < (args.k+1));
   }

   UpdateSolution(x_sloppy, r_sloppy, !do_givens);

   return (j-start_idx);
 }

 void GMResDR::operator()(ColorSpinorField &out, ColorSpinorField &in)
 {
    profile.TPSTART(QUDA_PROFILE_INIT);

    const double tol_threshold = 6.0; 

    ColorSpinorField &x   = out;

    if (!init) {

      gmresdr_args = new GMResDRArgs(param.m, param.nev);

      ColorSpinorParam csParam(in);//create spinor parameters

      yp = ColorSpinorField::Create(in); //high precision accumulation field
      rp = ColorSpinorField::Create(in); //high precision residual
      
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);

      if(param.precision_sloppy != param.precision){

        r_sloppy = ColorSpinorField::Create(csParam);
        x_sloppy = ColorSpinorField::Create(csParam);

      } else {

	x_sloppy = &x;
	r_sloppy = rp;
      }

      tmpp   = ColorSpinorField::Create(csParam);

      if ( K && (param.precision_precondition != param.precision_sloppy) ) {

        csParam.setPrecision(param.precision_precondition);
        p_pre = ColorSpinorField::Create(csParam);
        r_pre = ColorSpinorField::Create(csParam);

      }

      csParam.setPrecision(param.precision_sloppy);
      csParam.is_composite  = true;
      csParam.composite_dim = param.m+1;

      Vm   = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors

      csParam.composite_dim = param.m;

      Zm = K ? ColorSpinorFieldSet::Create(csParam) : Vm;

    }


    GMResDRArgs &args = *gmresdr_args;

    ColorSpinorField &r   = *rp;
    ColorSpinorField &y   = *yp;
    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    int tot_iters = 0;

    //1. Compute initial residual:
    double normb = norm2( in );
    double stop  = param.tol*param.tol* normb;	/* Relative to b tolerance */

    mat(r, out, y);
    //
    double r2 = xmyNorm(in, r);//compute residual
    double b2     = r2;

    args.c[0] = Complex(sqrt(r2), 0.0);

    printfQuda("\nInitial residual squared: %1.16e, source %1.16e, tolerance %1.16e\n", r2, sqrt(normb), param.tol);
    //2. Compute the first Arnoldi vector:
    if(param.precision_sloppy != param.precision) {
      blas::copy(rSloppy, r);

      blas::ax(1.0 / args.c[0].real(), r);   
      Vm->Component(0) = r;
    } else {
      blas::zero(Vm->Component(0));//no need for this op
      blas::axpy(1.0 / args.c[0].real(), rSloppy, Vm->Component(0));   
    }

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    double heavy_quark_res = 0.0;  // heavy quark res idual
    if (use_heavy_quark_res)  heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

   /************************Run Flex Arnoldi cycles*****************************/
   int restart_idx = 0, j = 0, check_interval = 10;

   bool do_givens = true; //do Givens transforms for the initial cycle

   while(restart_idx < param.deflation_grid && !(convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop)))
   {

     tot_iters += RunFlexArnoldiProcess(j, ((j == 0) and do_givens));

     r2 = norm2(rSloppy);

     //Update residual for the next cycle:
     if (param.precision_sloppy != param.precision)
     {
       y = xSloppy;
       blas::xpy(y, x);
       blas::zero(xSloppy);
     }
     //
     PrintStats("FGMResDR:", tot_iters, r2, b2, heavy_quark_res);

     if( (restart_idx != param.deflation_grid-1) ) {

       RestartVZH();
       j = args.k;

     } else {

       mat(r, x, y);
       //
       double ext_r2 = xmyNorm(in, r);//compute full precision residual

       printfQuda("\nDone for cycle %d, true residual squared %1.15e\n", restart_idx, ext_r2);

       args.ResetArgs();
       args.c[0] = Complex(sqrt(ext_r2), 0.0);
//@       blas::ax(1.0 / args.c[0].real(), r); 
//@       Vm->Component(0) = r;
       blas::zero(Vm->Component(0));//no need for this op
       blas::axpy(1.0 / args.c[0].real(), rSloppy, Vm->Component(0));

       j = 0; 
     }

     restart_idx += 1;

   }//end of deflated restarts

   profile.TPSTOP(QUDA_PROFILE_COMPUTE);
   profile.TPSTART(QUDA_PROFILE_EPILOGUE);

   param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
   double gflops = (blas::flops + mat.flops())*1e-9;
   param.gflops = gflops;
   param.iter += tot_iters;

   // compute the true residuals
   mat(r, x, y);

   param.true_res = sqrt(xmyNorm(in, r) / b2);

   PrintSummary("FGMResDR:", tot_iters, r2, b2);

   // reset the flops counters
   blas::flops = 0;
   mat.flops();

   profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

   param.rhs_idx += 1;

   return;
 }

} // namespace quda
