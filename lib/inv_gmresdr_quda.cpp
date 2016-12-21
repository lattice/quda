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
      static bool SelectSmall (SortEvals v1, SortEvals v2) { return (v1._val < v2._val);}
    };

    class GMResDRArgs{

      public:

       VectorSet   ritzVecs;//array of sorted harmonic eigenvectors: (nev+1) length      
       DenseMatrix H;//Hessenberg matrix
       Vector      eta;
       //Vector      *c;//  c = &ritzVecs[(m+1)*k] => ritzVecs.col(k)
      
       int m;
       int k;

       int restarts;

       GMResDRArgs(int m, int nev) : ritzVecs(VectorSet::Zero(m+1,nev+1)), H(DenseMatrix::Zero(m+1,m)), 
       eta(m), m(m), k(nev), restarts(0) 
       { }

       //inline Complex& Hess(int row, int col) {return H[col*(m+1)+row];}
       //inline Complex& const Hess(int row, int col) const {return H[col*(m+1)+row];}

       void ComputeHarmonicRitz();

       inline void AddShortResidual() { ritzVecs.col(k) -= H * eta;}
       inline void ResetArgs() { 
         H.setZero();  
         eta.setZero();  
         ritzVecs.setZero(); 
       }

       ~GMResDRArgs(){ }
   };

   void GMResDRArgs::ComputeHarmonicRitz()
   {
     //1. initialize Eigen objects:
     DenseMatrix cH = H.block(0, 0, m, m).adjoint();
     DenseMatrix Gk = H.block(0, 0, m, m);
     //
     VectorSet  harVecs = MatrixXcd::Zero(m,m);
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
     magma_Xgesv(static_cast<void*>(Gk.data()), m, m, static_cast<void*>(harVecs.data()), static_cast<void*>(harVals.data()), m, sizeof(Complex));//check it!
     cudaHostUnregister(Gk.data());
#else
     //
#endif
     //4. do sort:
     std::vector<SortEvals> sorted_evals(m);

     for(int e = 0; e < m; e++) sorted_evals.push_back( SortEvals( abs(harVals[e]), e ));
     //
     std::stable_sort(sorted_evals.begin(), sorted_evals.end(), SortEvals::SelectSmall);

     //5. Copy sorted eigenvectors:
     for(int e = 0; e < k; e++) memcpy(ritzVecs.col((m+1)*e).data(), harVecs.col((m+1)*( sorted_evals[e].idx)).data(), (m)*sizeof(Complex));

     return;
   }

    // set the required parameters for the inner solver
    void fillInnerSolveParam(SolverParam &inner, const SolverParam &outer) {
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

 GMResDR::GMResDR(DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matDefl, DiracMatrix *matPrecon, SolverParam &param, TimeProfile *profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), matPrecon(matPrecon), K(nullptr), gmres_space_prec(QUDA_INVALID_PRECISION),
    Vm(nullptr), Zm(nullptr), profile(profile), args(nullptr), gmres_alloc(false)
 {
     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMResDR is not currently supported.\n");
     //
     fillInnerSolveParam(Kparam, param);

     if (param.inv_type_precondition == QUDA_CG_INVERTER) // inner CG preconditioner
       K = new CG(*matPrecon, *matPrecon, Kparam, *profile);
     else if (param.inv_type_precondition == QUDA_BICGSTAB_INVERTER) // inner BiCGstab preconditioner
       K = new BiCGstab(*matPrecon, *matPrecon, *matPrecon, Kparam, *profile);
     else if (param.inv_type_precondition == QUDA_MR_INVERTER) // inner MR preconditioner
       K = new MR(*matPrecon, *matPrecon, Kparam, *profile);
     else if (param.inv_type_precondition == QUDA_SD_INVERTER) // inner MR preconditioner
       K = new SD(*matPrecon, Kparam, *profile);
     else if (param.inv_type_precondition == QUDA_INVALID_INVERTER) // unknown preconditioner
       K = nullptr;
     else 
       errorQuda("Unsupported preconditioner %d\n", param.inv_type_precondition);

     return;
 }

 GMResDR::GMResDR(DiracMatrix *mat, Solver &K, DiracMatrix *matSloppy, DiracMatrix *matDefl, DiracMatrix *matPrecon, SolverParam &param, TimeProfile *profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), matPrecon(matPrecon), K(&K), gmres_space_prec(QUDA_INVALID_PRECISION),  init(false)
    Vm(nullptr), Zm(nullptr), profile(profile), args(nullptr) { }

 GMResDR::GMResDR(SolverParam &param) :
    DeflatedSolver(param, nullptr), mat(nullptr), matSloppy(nullptr), matDefl(nullptr), matPrecon(nullptr), gmres_space_prec(QUDA_INVALID_PRECISION),
    Vm(nullptr), profile(nullptr), args(nullptr), init(false) { }


 GMResDR::~GMResDR() {
    profile->TPSTART(QUDA_PROFILE_FREE);

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

      delete tmp1_p;
      delete yp;
      delete rp;

      delete gmresdr_args;
    }

   profile->TPSTOP(QUDA_PROFILE_FREE);
 }

 void GMResDR::UpdateSolution(ColorSpinorField *x, ColorSpinorField *r, bool do_gels = true)
 {
   GMResDRArgs &args = *gmresdr_args;

   void *c = static_cast<void*>(args.ritzVecs.col(args.k).data());

#ifdef USE_MAGMA
   cudaHostRegister(static_cast<void *>(args.H.data()), (args.m+1)*args.m*sizeof(Complex), cudaHostRegisterDefault)
   if(do_gels) magma_gels(static_cast<void*>(args.H.data()), c, static_cast<void*>(args.eta.data()), (args.m+1), args.m, (args.m+1));
   cudaHostUnregister(args.H.data());
#else
   //
#endif
   //compute x = x0+Wm*lsqSol, where Wm is Vm or Zm (if preconditioned):
   std::vector<ColorSpinorField*> Z_(Zm->Components().begin(),Zm->Components().end());
   std::vector<ColorSpinorField*> V_(Vm->Components().begin(),Vm->Components().end());

   std::vector<ColorSpinorField*> x_(&x), r_(&r);

   blas::caxpy( static_cast<Complex*>(args.eta.data()), Z_, x_);

   Complex Heta = new Complex[args.m+1]; 
   Map<VectorXcd, Unaligned> Heta_(Heta, args.m+1);
 
   Heta_ -= (args.H*args.eta);

   blas::caxpy(Heta,V_,r_);

   delete[] Heta;

   return;
 }

 void GMResDRArgs::RestartVZH()
 {
   GMResDRArgs &args = *gmresdr_args;

   args.ComputeHarmonicRitz(); 
   args.AddShortResidual();

   //Map<MatrixXcd, Unaligned> _H   (_args.H, (args.m+1), args.m);
   //Map<MatrixXcd, Unaligned> _Gkp1(_args.ritzVecs, (args.m+1), (args.k+1));

   //1. QR of the eigenvctor matrix:
   DenseMatrix Qkp1(MatrixXcd::Identity((args.m+1), (args.k+1)));
  
   HouseholderQR<MatrixXcd> qr(args.ritzVecs);
   Qkp1.applyOnTheLeft( qr.householderQ());

   //2. Update H:
   DenseMatrix Res = Qkp1.adjoint()*args.H*Qkp1.topLeftCorner(args.m, args.k);
   H.setZero();
   H.topLeftCorner(args.k+1, args.k) = Res;

   //2. Update Vm+1 : Vk+1 = Vm+1 Qk+1

   //Call multiblas:
   //Create intermediate model:
   ColorSpinorFieldParam csParam(Vm->Component(0));

   csParam.is_composite  = true;
   csParam.composite_dim = (args.k+1);
   csParam.setPrecision(QUDA_DOUBLE_PRECISION);

   ColorSpinorFieldSet Vkp1 = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors

   std::vector<ColorSpinorField*> V(Vm->Components().begin(), Vm->Components().end());

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
       blas::zero(Vkp1->Compnent(i));
     }
     else blas::zero(Vm->Component(i));
   }

   //3. Zk = Zm Qk
   if( Zm->V() &&  Zm->V() != Vm->V() )
   {
     DenseMatrix Qk = Qkp1.topLeftCorner(args.m,args.k);

     std::vector<ColorSpinorField*> Z(Zm->Components().begin(), Zm->Components().end());

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

   blas::ax(1.0/ norm2(Vm->Component(args.k)), Vm->Component(args.k));

   return;
 }

 int GMResDRArgs::RunFlexArnoldiProcess(const int start_idx, const bool do_givens = false)
 {
   GMResDRArgs &args = *gmresdr_args;

   Complex *givensH = nullptr, 
   Complex *cn      = nullptr;
   double  *sn      = nullptr;

   if(do_givens)
   {
     if (start_idx != 0) errorQuda("\nStart index must be zero.\n"); 

     givensH = new Complex[(args.m+1)*args.m];//Keep Givens matrix
     //Givens coefficients:
     cn = new Complex[args.m];
     sn = new double [args.m];
   }

   int j = start_idx;

   while( j < args.m ) //we allow full cycle
   {
     if(K) {
       ColorSpinorField &rPre = (param.precision_precondition != param.precision_sloppy ? *r_pre : &Vm->Component(j));
       ColorSpinorField &pPre = (param.precision_precondition != param.precision_sloppy ? *p_pre : &Zm->Component(j));

       if( !precMatch ) copy(rPre, Vm->Component(j));

       (*K)( pPre ,rPre );

       if( !precMatch ) copy(Zm->Component(j), pPre);
     }

     matSloppy(Vm->Component(j+1), Zm->Component(j), tmp);
     //
     Complex h0(0.0, 0.0);
     //
     for(int i = (start_idx+1); i <= j; i++)
     {
        args.H(i, j) = cDotProduct(Vm->Component(i), Vm->Component(j+1));//
        //
        caxpy(-args.H(i, j), Vm->Component(i), Vm->Component(j+1));

        if(do_givens)  //lets do Givens rotations:
        {
           if(i > 0) givensH[args->ldm*j+(i-1)] = conj(Cn[i-1])*h0 + Sn[i-1]*args.H(i,j);
           //
           h0 = (i == 0) ?  args.H(0,j) : -Sn[i-1]*h0 + Cn[i-1]*args.H(i,j); 
        }
     }
     //6. Compute h(j+1,j):
     args.H(j+1, j) = Complex(sqrt(norm2(*Vm->Component(j+1))), 0.0);
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
       givensH[j*args->ldm+j] = conj(Cn[j])*h0 + Sn[j]*args.H(j+1,j);
      
       //10. Compute iter residual:???
       args.ritzVecs.col(args.k).data()[j+1] = -Sn[j]*args.ritzVecs.col(args.k).data()[j];
       //
       args.ritzVecs.col(args.k).data()[j] *= conj(Cn[j]);
       //
       r2 = norm(args.ritzVecs.col(args.k).data()[j+1]);//stopping criterio
     }

     j += 1;
   }//end of main loop.

   //11 Update solution:
   //11.a Solve LSQR problem:
   if(do_givens)
   {
     Map<MatrixXcd, Unaligned, DynamicStride> _givensH(givensH,args.m,args.m, DynamicStride<args.m+1,1>);//extract triangular part
     _givensH.triangularView<Upper>().solveInPlace<OnTheLeft>(args.eta);

     delete[] givensH;
     delete[] cn;
     delete[] sn;

   }  else {
     //express old residual in terms of new basis:
     std::vector<ColorSpinorField*> V_, r_;
     V_.reserve(args.k+1);
     r_.reserve(args.k+1);
     for(int i = 0; i < args.k+1; i++)
     {
       V_.push_back(&Vm->Component(i));
       r_.push_back(r_sloppy);
     }

     blas::cDotProduct( static_cast<Complex*>(args.ritzVecs.col(args.k).data()), V_, r_ );

   }

   UpdateSolution(x_sloppy, r_sloppy, !do_givens);

   return j;
 }

 void GMResDR::operator()(ColorSpinorField *out, ColorSpinorField *in)
 {
    profile->TPSTART(QUDA_PROFILE_INIT);

    const double tol_threshold = 6.0; 

    if (!init) {

      gmresdr_args = new GMResDRArgs(param.m, param.nev);

      ColorSpinorParam csParam(*in);//create spinor parameters

      yp = ColorSpinorField::Create(*in); //high precision accumulation field
      rp = ColorSpinorField::Create(*in); //high precision residual
      
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.setPrecision(param.precision_sloppy);

      if(param.precision_sloppy != param.precision){

        r_sloppy = ColorSpinorField::Create(csParam);
        x_sloppy = ColorSpinorField::Create(csParam);

      } else {

	x_sloppy = &x;
	r_sloppy = rp;
      }

      tmp1_p   = ColorSpinorField::Create(csParam);

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
    ColorSpinorField &x   = *out;
    ColorSpinorField &y   = *yp;
    ColorSpinorField &xSloppy = *x_sloppy;
    ColorSpinorField &rSloppy = *r_sloppy;

    ColorSpinorField &tmp = *tmp1_p;

    profile->TPSTOP(QUDA_PROFILE_INIT);
    profile->TPSTART(QUDA_PROFILE_PREAMBLE);

    int tot_iters = 0;

    //1. Compute initial residual:
    double normb = norm2(*in);
    double stop  = param.tol*param.tol* normb;	/* Relative to b tolerance */

    (*mat)(r, *out, y);
    //
    double r2 = xmyNorm(*in, r);//compute residual
    double b2 = r2;

    printfQuda("\nInitial residual squared: %1.16e, source %1.16e, tolerance %1.16e\n", sqrt(r2), sqrt(normb), param.tol);
    //2. Compute the first Arnoldi vector:
    args.c[0] = sqrt(r2);
    //
    if(param.precision_sloppy != param.precision) blas::copy(rSloppy, r);

    blas::ax(1.0 / args.c[0], r);   
    blas::copy(Vm->Component(0), r);

    profile->TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile->TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

   /************************Run Flex Arnoldi cycles*****************************/
   int restart_idx = 0, j = 0;

   bool run_deflated_cycles = true; //deflated or projected cycles?
   bool do_givens  = true;

   while(restart_idx < param.deflation_grid && !(convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop)))
   {

     tot_iters += RunFlexArnoldiProcess(j, do_givens);

     r2 = norm(rSloppy);

     //Update residual for the next cycle:
     if (param.precision_sloppy != param.precision)
     {
       copy(y, *x_sloppy);
       xpy(y, x);
       zero(*x_sloppy);
     }

     (*mat)(r, x, y);
     //
     double ext_r2 = xmyNorm(*in, r);//compute full precision residual

     if(run_deflated_cycles && ((param.precision_sloppy != param.precision) && ((sqrt(ext_r2) / sqrt(r2)) > tol_threshold)))
     {
       run_deflated_cycles = false;
     }

     //printfQuda("\nDone for cycle %d, true residual squared %1.15e\n", restart_idx, ext_r2);
     //
     PrintStats("FGMResDR:", tot_iters, r2, b2, heavy_quark_res);

     if(param.precision_sloppy != param.precision) blas::copy(rSloppy, r);

     if( run_deflated_cycles ) {

       RestartVZH();
       j = args.k;

     } else {

       args.ResetData();
       blas::ax(1.0 / args.c[0], r);   
       blas::copy(Vm->Component(0), r);

       j = 0; 
     }

     restart_idx += 1;

   }//end of deflated restarts

   profile->TPSTOP(QUDA_PROFILE_COMPUTE);
   profile->TPSTART(QUDA_PROFILE_EPILOGUE);

   param.secs = profile->Last(QUDA_PROFILE_COMPUTE);
   double gflops = (blas::flops + mat->flops())*1e-9;
   param.gflops = gflops;
   param.iter += tot_iters;

   // compute the true residuals
   (*mat)(r, x, y);
   //matSloppy(r, x, tmp, tmp2);

   param.true_res = sqrt(xmyNorm(*in, r) / b2);

   PrintSummary("FGMResDR:", tot_iters, r2, b2);

   // reset the flops counters
   blas::flops = 0;
   mat->flops();

   profile->TPSTOP(QUDA_PROFILE_EPILOGUE);

   param.rhs_idx += 1;

   return;
 }

} // namespace quda
