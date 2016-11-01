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


    static GMResDRDeflationParam *defl_param = 0;

    struct SortedEvals{

      double nrm;
      int    idx;

      SortedEvals(double val, int idx) : nrm(val), idx(idx) {};
    };

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

    class GMResDRArgs{

      private:

       Complex *ritzVecs;//array of sorted harmonic eigenvectors: (nev+1) length

      public:
      
       Complex *H;//Hessenberg matrix
       Complex *c;//
       Complex *eta;
      
       int m;
       int k;

       GMResDRArgs( ) { };

       GMResDRArgs(int m, int nev) ritzVecs(0), m(m), k(nev)
       {
         if(nev == 0 || nev > m) errorQuda("\nError: incorrect deflation size.\n");

         H   = new Complex[(m+1)*m];//Hessenberg matrix
         eta = new Complex[m];
         ritzVecs  = new Complex[(m+1)*(k+1)];//

         c = &ritzVecs[(m+1)*k];
       }

       inline Complex& Hess(int row, int col) {return H[col*(m+1)+row];}
       inline Complex& const Hess(int row, int col) const {return H[col*(m+1)+row];}

       void ComputeHarmonicRitz();
       void AddShortResidual();

       void ResetData()
       {
         std::memset(H, 0, (m+1)*m*sizeof(Complex) );
         std::memset(eta, 0, m*sizeof(Complex) );
         std::memset(ritzVecs, 0, (m+1)*(k+1)*sizeof(Complex) );
       }

       ~GMResDRArgs()
       {
         delete[] eta;
         delete[] H;

         delete[] ritzVecs;
       }
   };

 void GMResDRArgs::ComputeHarmonicRitz()
 {
   //1. initialize Eigen objects:
   Map<MatrixXcd, Unaligned, Stride<Dynamic, Dynamic> > _H(H, m, m, Stride(m+1, 1));

   MatrixXcd _cH(_H.adjoint());
   MatrixXcd _Gk(_H);

   MatrixXcd  _ritzVecs = MatrixXcd::Zero((m+1)*m);
   VectorXcd  _ritzVals = VectorXcd::Zero(m);

   //2. construct H + beta*H^{-H} e_m*e_m^{T}
   //2.a need to solve H^{H}y = beta e_m;
   VectorXcd _em = VectorXcd::Zero(m);//in fact, em^{T}=(0,....,1)
   //2.b construct beta*em
   _em(m-1) = norm( Hess(m, m-1) );//in fact, we construct beta*em
   //2.c Compute y = H^{-H} beta*e_m:   
#ifdef USE_MAGMA 
   magma_Xgesv(static_cast<void*>(_em.data()), m, m, static_cast<void*>(_cH.data()), m, sizeof(Complex));
#else
//   RowVectorXcd sol = _cH.fullPivLu().solve(em);
//   RowVectorXcd _sol = _cH.colPivHouseholderQr().solve(_em);
//   _em = _sol;     
#endif
//   _em.applyOnTheLeft(_Gk);
   //2.d Adjust last column with ((H^{-H}*beta*em)=em, em^{T}=[0,....,1]):
   _Gk.col(m-1) += em;

   //3.  Compute harmonic eigenpairs:
#ifdef USE_MAGMA 
   magma_Xgesv(static_cast<void*>(_Gk.data()), m, (m+1), static_cast<void*>(_ritzVecs.data()), static_cast<void*>(_ritzVals.data()), (m+1), sizeof(Complex));//check it!
#else
   //
#endif
   //4. do sort:
   std::vector<SortedEvals> sorted_evals(m);

   for(int e = 0; e < m; e++) sorted_evals.push_back( SortEvals( abs(_ritzVals[e]), e ));
   //
   std::stable_sort(sorted_evals.begin(), sorted_evals.end(),  [](SortedEvals &v1, SortedEvals &v2) { return (v1.nrm < v2.nrm);});

   //5. Copy sorted eigenvectors:
   for(int e = 0; e < k; e++) memcpy(&ritzVecs[(m+1)*e], _ritzVecs.col((m+1)*( sorted_evals[e].idx)).data(), (m+1)*sizeof(Complex));
//   std::for_each(sorted_evals.begin(), sorted_evals.end(), std::memcpy(&ritzVecs[(m+1)*e], _ritzVecs.col((m+1)*( sorted_evals[e].idx)).data(), (m+1)*sizeof(Complex)));

   return;
 }

 void GMResDRArgs::AddShortResidual()
 {
   Map<MatrixXcd, Unaligned> _H(H, m+1, m);
   Map<VectorXcd, Unaligned> _eta(eta, m);

   Map<VectorXcd, Unaligned> _Gkp1(&ritzVecs[(m+1)*k], (m+1));

   _Gkp1 -= (_H*_eta);

   return;
 }



 GMResDR::GMResDR(DiracMatrix *mat, DiracMatrix *matSloppy, DiracMatrix *matDefl, DiracMatrix *matPrecon, SolverParam &param, TimeProfile *profile) :
    DeflatedSolver(param, profile), mat(mat), matSloppy(matSloppy), matDefl(matDefl), matPrecon(matPrecon), K(nullptr), gmres_space_prec(QUDA_INVALID_PRECISION),
    Vm(nullptr), Zm(nullptr), profile(profile), args(nullptr), gmres_alloc(false)
 {
     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMResDR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//We don't allow half precision, do we?

     args = new GMResDRArgs(param.m, param.nev);//use_deflated_cycles flag is true by default

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
    Vm(nullptr), Zm(nullptr), profile(profile), args(nullptr)
 {
     //if(param.precision != param.precision_sloppy) errorQuda("\nMixed precision GMResDR is not currently supported.\n");
     //
     gmres_space_prec = param.precision_sloppy;//We don't allow half precision, do we?

     args = new GMResDRArgs(param.m, param.nev);//use_deflated_cycles flag is true by default

     return;
 }

 GMResDR::GMResDR(SolverParam &param) :
    DeflatedSolver(param, nullptr), mat(nullptr), matSloppy(nullptr), matDefl(nullptr), matPrecon(nullptr), gmres_space_prec(QUDA_INVALID_PRECISION),
    Vm(nullptr), profile(nullptr), args(nullptr), init(false) { }


 GMResDR::~GMResDR() {
    profile->TPSTART(QUDA_PROFILE_FREE);

    if(args) delete args;

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
    }

   profile->TPSTOP(QUDA_PROFILE_FREE);
 }

 void GMResDR::UpdateSolution(ColorSpinorField *x, ColorSpinorField *r, bool do_gels = true)
 {
   GMResDRArgs &args = *gmresdr_args;

   Map<MatrixXcd, Unaligned> _H(_args.H, (args.m+1), args.m);
   Map<VectorXcd, Unaligned> _eta(args.eta, args.m);
   Map<VectorXcd, Unaligned> _c(args.c, args.m+1);

#ifdef USE_MAGMA
   if(do_gels) magma_gels(static_cast<void*>(_H.data()), static_cast<void*>(_c.data()), static_cast<void*>(_eta.data()), (args.m+1), args.m, (args.m+1));
#else
   //
#endif
   //compute x = x0+Wm*lsqSol, where Wm is Vm or Zm (if preconditioned)
   std::vector<ColorSpinorField*> _Zm(args.m), _x(args.m), _Vm(args.m+1), _r(args.m+1);
   for(int i = 0; i < args.m; i++)
   {
     _Zm.push_back(&Zm->Component(i));
     _Vm.push_back(&Vm->Component(i));
     _x.push_back(&x);
     _r.push_back(&r);
   }

   _Vm.push_back(&Vm->Component(args.m));
   _r.push_back(&r);

   blas::caxpy(args.eta, _Zm, _x);

   Complex Heta = new Complex[args.m+1]; 
   Map<VectorXcd, Unaligned> __Heta(Heta, args.m+1);
 
   _Heta -= (_H*_eta);

   blas::caxpy(__Heta, _Vm, _r);

   delete[] Heta;

   return;
 }

 void GMResDRArgs::RestartVZH()
 {
   GMResDRArgs &args = *gmresdr_args;

   args.ComputeHarmonicRitz(); 
   args.AddShortResidual();

   Map<MatrixXcd, Unaligned> _H   (_args.H, (args.m+1), args.m);
   Map<MatrixXcd, Unaligned> _Gkp1(_args.ritzVecs, (args.m+1), (args.k+1));

   //1. QR of the eigenvctor matrix:
   MatrixXcd _Qkp1(MatrixXcd::Identity((args.m+1), (args.k+1)));
  
   HouseholderQR<MatrixXcd> _qr(_Gkp1);
   _Qkp1.applyOnTheLeft( _qr.householderQ());

   //2. Update H:
   MatrixXcd _Res = _Qkp1.adjoint()*_H*_Qkp1.topLeftCorner(args.m, args.k);
   _H.setZero();
   _H.topLeftCorner(args.k+1, args.k) = _Res;

   //2. Update Vm+1 : Vk+1 = Vm+1 Qk+1

   //Call multiblas:
   //Create intermediate model:
   ColorSpinorFieldParam csParam(Vm->Component(0));

   csParam.is_composite  = true;
   csParam.composite_dim = (args.k+1);
   csParam.setPrecision(QUDA_DOUBLE_PRECISION);

   ColorSpinorFieldSet Vkp1 = ColorSpinorFieldSet::Create(csParam); //search space for Ritz vectors

   std::vector<ColorSpinorField*> _Vm(Vm->CompositeDim());
   for(int i = 0; i < Vm->CompositeDim(); i++) _Vm.push_back(&Vm->Component(i)); 

   for(int i = 0; i < (args.k+1); i++) blas::caxpy(_Qkp1.col(i).data(), _Vm , Vkp1->Component(i));//use mixed multiblas here

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
     MatixXcd _Qk = _Qkp1.topLeftCorner(args.m,args.k);

     std::vector<ColorSpinorField*> _Zm(Zm->CompositeDim());
     for(int i = 0; i < Zm->CompositeDim(); i++) _Zm.push_back(&Zm->Component(i)); 

     blas::caxpy(Qk.col(i).data(), _Zm , Vkp1->Component(i) ); 

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
        args.Hess(i, j) = cDotProduct(Vm->Component(i), Vm->Component(j+1));//
        //
        caxpy(-args.Hess(i, j), Vm->Component(i), Vm->Component(j+1));

        if(do_givens)  //lets do Givens rotations:
        {
           if(i > 0) givensH[args->ldm*j+(i-1)] = conj(Cn[i-1])*h0 + Sn[i-1]*args.Hess(i,j);
           //
           h0 = (i == 0) ?  args.Hess(0,j) : -Sn[i-1]*h0 + Cn[i-1]*args.Hess(i,j); 
        }
     }
     //6. Compute h(j+1,j):
     args.Hess(j+1, j) = Complex(sqrt(norm2(*Vm->Component(j+1))), 0.0);
     //7. Scale the last arnoldi vector:
     blas::ax( 1.0 / args.Hess(j+1, j).real(), Vm->Component(j+1));
     //
     if(do_givens)
     {
       double inv_denom = 1.0 / sqrt(norm(h0)+norm(args.Hess(j+1,j)));
       //8. Update Givens coefficients:
       cn[j] = h0 * inv_denom;
       //
       sn[j] = args.Hess(j+1,j).real() * inv_denom;
       //9. Compute diagonal element in G:
       givensH[j*args->ldm+j] = conj(Cn[j])*h0 + Sn[j]*args.Hess(j+1,j);
      
       //10. Compute iter residual:
       args.c[j+1] = -Sn[j]*args.c[j];
       //
       args.c[j] *= conj(Cn[j]);
       //
       r2 = norm(args.c[j+1]);//stopping criterio
     }

     j += 1;
   }//end of main loop.

   //11 Update solution:
   //11.a Solve LSQR problem:
   if(do_givens)
   {
     Map<MatrixXcd, Unaligned, Stride<args.m+1,1>> _givensH(givensH,args.m,args.m);//extract triangular part
     Map<VectorXcd, Unaligned> _eta(args.eta,args.m);

     _givensH.triangularView<Upper>().solveInPlace<OnTheLeft>(_eta);

     delete[] givensH;
     delete[] cn;
     delete[] sn;

   }  else {
     //express old residual in terms of new basis:
     std::vector<ColorSpinorField*> _Vkp1(args.k+1), _r(args.k+1);
     for(int i = 0; i < args.k+1; i++)
     {
       _Vkp1.push_back(&Vm->Component(i));
       _r.push_back(r_sloppy);
     }

     blas::cDotProduct( args.c, _Vkp1, _r );

   }

   UpdateSolution(x_sloppy, r_sloppy, !do_givens);

   return j;
 }

 void GMResDR::operator()(ColorSpinorField *out, ColorSpinorField *in)
 {
    profile->TPSTART(QUDA_PROFILE_INIT);

    const double tol_threshold = 6.0; 

    GMResDRArgs &args = *gmresdr_args;

    if (!init) {

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
