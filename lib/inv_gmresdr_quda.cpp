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

#include <algorithm>
#include <memory>

#include <eigen_helper.h>

/*
GMRES-DR algorithm:
R. B. Morgan, "GMRES with deflated restarting", SIAM J. Sci. Comput. 24 (2002) p. 20-37
See also: A.Frommer et al, "Deflation and Flexible SAP-Preconditioning of GMRES in Lattice QCD simulations" ArXiv hep-lat/1204.5463
*/

namespace quda {

    using namespace blas;
    using namespace std;

    using DynamicStride   = Stride<Dynamic, Dynamic>;

    using DenseMatrix     = MatrixXcd;
    using VectorSet       = MatrixXcd;
    using Vector          = VectorXcd;

//special types needed for compatibility with QUDA blas:
    using RowMajorDenseMatrix = Matrix<Complex, Dynamic, Dynamic, RowMajor>;

    // helper for a smart pointer creation

    std::shared_ptr<ColorSpinorField> MakeSharedPtr2(const ColorSpinorParam &param)
    {
      if (param.location == QUDA_CPU_FIELD_LOCATION) {
        auto cpu_sptr = std::make_shared<cpuColorSpinorField>(param);
        return cpu_sptr;
      } else {
        auto gpu_sptr = std::make_shared<cudaColorSpinorField>(param);
        return gpu_sptr;
      }
    }

    class GMResDRArgs{

      public:
       VectorSet   ritzVecs;
       DenseMatrix H;
       Vector      eta;

       MatrixXcd givensH;
       VectorXcd cn;
       VectorXd sn;

       int m;
       int k;
       int restarts;

       ColorSpinorFieldSet* Vkp1; // high-precision accumulation array

       GMResDRArgs(ColorSpinorField &meta, int m, int nev) :
         ritzVecs(VectorSet::Zero(m + 1, nev + 1)),
         H(DenseMatrix::Zero(m + 1, m)),
         eta(Vector::Zero(m)),
         givensH(MatrixXcd::Zero(m + 1, m)),
         cn(VectorXcd::Zero(m)),
         sn(VectorXd::Zero(m)),
         m(m),
         k(nev),
         restarts(0),
         Vkp1(nullptr)
       {
         ColorSpinorParam csParam(meta);

         csParam.composite_dim = (k + 1);
         csParam.setPrecision(QUDA_DOUBLE_PRECISION);

         Vkp1 = ColorSpinorField::Create(csParam);

         printfQuda("GMResDR parameters nKrylov = %d, nDefl = %d\n", m, k);
       }

       inline void ResetArgs() {
         ritzVecs.setZero();
         H.setZero();
         eta.setZero();
         givensH.setZero();
         cn.setZero();
         sn.setZero();
       }

       ~GMResDRArgs() {
         delete Vkp1;
       }

       void Givens(const int j)
       {
         Ref<VectorXcd> c = ritzVecs.col(k);

         Complex h0 = H(0, j - 1);

         for (int i = 1; i < j; i++) {
           givensH(i - 1, j - 1) = conj(cn(i - 1)) * h0 + sn(i - 1) * H(i, j - 1);
           h0 = -sn(i - 1) * h0 + cn(i - 1) * H(i, j - 1);
         }

         const double inv_denom = 1.0 / sqrt(norm(h0) + norm(H(j, j - 1)));

         cn(j - 1) = h0 * inv_denom;
         sn(j - 1) = H(j, j - 1).real() * inv_denom;
         givensH(j - 1, j - 1) = conj(cn(j - 1)) * h0 + sn(j - 1) * H(j, j - 1);

         c(j) = -sn(j - 1) * c(j - 1);
         c(j - 1) *= conj(cn(j - 1));
         //
         printfQuda("Residual %le :: %le \n", c(j).real(), c(j).imag());

         return;
       }

       void LeastSquaresSolve(const Complex c0, const ColorSpinorField &r_sloppy, const ColorSpinorFieldSet &vm,
                              const bool do_givens)
       {
	 Ref<VectorXcd> c = ritzVecs.col(k);

         if (do_givens) {
           // recosntruct the last col
           Complex h0 = H(0, m - 1);

           for (int i = 1; i <= m - 1; i++) {
             givensH(i - 1, m - 1) = conj(cn(i - 1)) * h0 + sn(i - 1) * H(i, m - 1);
             h0 = -sn(i - 1) * h0 + cn(i - 1) * H(i, m - 1);
           }

           const double inv_denom = 1.0 / sqrt(norm(h0) + norm(H(m, m - 1)));

           cn(m - 1) = h0 * inv_denom;
           sn(m - 1) = H(m, m - 1).real() * inv_denom;
           givensH(m - 1, m - 1) = conj(cn(m - 1)) * h0 + sn(m - 1) * H(m, m - 1);

           c(m) = -sn(m - 1) * c(m - 1);
           c(m - 1) *= conj(cn(m - 1));
           //
           printfQuda("Last cycle residual %le :: %le \n", c(m).real(), c(m).imag());

           memcpy(eta.data(), c.data(), m * sizeof(Complex));
           c.setZero();
           c(0) = c0;

           givensH.block(0, 0, m, m).triangularView<Upper>().solveInPlace<OnTheLeft>(eta);

         } else {
           c.setZero();

           std::vector<ColorSpinorField *> v_(const_cast<ColorSpinorField &>(vm)(0, k + 1));
           std::vector<ColorSpinorField *> r_;
           r_.push_back(const_cast<ColorSpinorField *>(&r_sloppy));

           blas::cDotProduct(c.data(), v_, r_);
         }

         return;
       }
    };

    void ComputeHarmonicRitz(GMResDRArgs &args)
    {

      DenseMatrix cH = args.H.block(0, 0, args.m, args.m).adjoint();
      DenseMatrix Gk = args.H.block(0, 0, args.m, args.m);

      VectorSet harVecs = MatrixXcd::Zero(args.m, args.m);
      Vector harVals = VectorXcd::Zero(args.m);

      Vector em = VectorXcd::Zero(args.m);

      em(args.m - 1) = norm(args.H(args.m, args.m - 1));
      // Gk.col(args.m-1) += cH.colPivHouseholderQr().solve(em);
      // Gk.col(args.m-1) += cH.fullPivHouseholderQr().solve(em);
      Gk.col(args.m - 1) += cH.householderQr().solve(em);

      ComplexEigenSolver<DenseMatrix> es(Gk);
      harVecs = es.eigenvectors();
      harVals = es.eigenvalues();

      std::vector<std::pair<double, Complex*>> sort_ev(args.m);

      for(int i = 0; i < args.m; i++) sort_ev[i] = std::make_pair(abs(harVals.data()[i]), harVecs.col(i).data());

      std::sort(sort_ev.begin(), sort_ev.end(),
                       [](const std::pair<double, Complex *> &x1, const std::pair<double, Complex *> &x2) {
                            return (x1.first < x2.first);} );

      for(int i = 0; i < args.k; i++) memcpy(args.ritzVecs.col(i).data(), sort_ev[i].second, (args.m) * sizeof(Complex));

      return;
   }

   void ComputeEta(GMResDRArgs &args)
   {

     Ref<VectorXcd> c = args.ritzVecs.col(args.k);
     args.eta = args.H.jacobiSvd(ComputeThinU | ComputeThinV).solve(c);

     return;
   }

   // set the required parameters for the inner solver
   void fillFGMResDRInnerSolveParam(SolverParam &inner, const SolverParam &outer)
   {
     inner.tol = outer.tol_precondition;
     inner.delta = 1e-20; // no reliable updates within the inner solver

     inner.precision = outer.precision_precondition; // precision_sloppy
     inner.precision_sloppy = outer.precision_precondition;

     // this sets a fixed iteration count if we're using the MR solver
     inner.residual_type
       = (outer.inv_type_precondition == QUDA_MR_INVERTER) ? QUDA_INVALID_RESIDUAL : QUDA_L2_RELATIVE_RESIDUAL;

     inner.iter = 0;
     inner.gflops = 0;
     inner.secs = 0;

     inner.inv_type_precondition = QUDA_INVALID_INVERTER;
     inner.is_preconditioner = true; // tell inner solver it is a preconditioner

     inner.schwarz_type = outer.schwarz_type;
     inner.global_reduction = inner.schwarz_type == QUDA_INVALID_SCHWARZ ? true : false;

     inner.use_init_guess = QUDA_USE_INIT_GUESS_NO;

     inner.maxiter = outer.maxiter_precondition;
     if (outer.inv_type_precondition == QUDA_CA_GCR_INVERTER) {
       inner.Nkrylov = inner.maxiter / outer.precondition_cycle;
     } else {
       inner.Nsteps = outer.precondition_cycle;
       if (outer.inv_type_precondition == QUDA_GMRESDR_INVERTER) { inner.max_restart_num = outer.precondition_cycle; }
     }

     inner.preserve_source = QUDA_PRESERVE_SOURCE_YES;

     inner.verbosity_precondition = outer.verbosity_precondition;
     inner.compute_true_res = false;
     inner.sloppy_converge = true;
   }

   GMResDR::GMResDR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                    SolverParam &param, TimeProfile &profile) :
     Solver(mat, matSloppy, matPrecon, matSloppy, param, profile),
     K(nullptr),
     Kparam(param),
     nKrylov(param.Nkrylov),
     gmresdr_args(nullptr),
     Vm(nullptr),
     Zm(nullptr),
     profile(profile),
     init(false)
   {
     fillFGMResDRInnerSolveParam(Kparam, param);

     if (param.inv_type_precondition == QUDA_CG_INVERTER) {
       K = new CG(matPrecon, matPrecon, matPrecon, matPrecon, Kparam, profile);
     } else if (param.inv_type_precondition == QUDA_MR_INVERTER) {
       K = new MR(matPrecon, matPrecon, Kparam, profile);
     } else if (param.inv_type_precondition == QUDA_SD_INVERTER) {
       K = new SD(matPrecon, Kparam, profile);
     } else if (param.inv_type_precondition == QUDA_GMRESDR_INVERTER) {
       K = new GMResDR(matPrecon, matPrecon, matPrecon, Kparam, profile);
     } else if (param.inv_type_precondition != QUDA_INVALID_INVERTER) { // unknown preconditioner
       errorQuda("Unknown inner solver %d", param.inv_type_precondition);
     }

     if (!K) warningQuda("Running without preconditioning...\n");

     return;
 }

 GMResDR::GMResDR(const DiracMatrix &mat, Solver &K_, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                  SolverParam &param, TimeProfile &profile) :
   Solver(mat, matSloppy, matPrecon, matSloppy, param, profile),
   K(&K_),
   Kparam(param),
   nKrylov(param.Nkrylov),
   gmresdr_args(nullptr),
   Vm(nullptr),
   Zm(nullptr),
   profile(profile),
   init(false)
 {
 }

 GMResDR::~GMResDR()
 {
   if (K && param.inv_type_precondition != QUDA_MG_INVERTER) delete K;
 }

 void GMResDR::UpdateSolution(ColorSpinorField &x, ColorSpinorField &r, bool do_gels)
 {
   GMResDRArgs &args = *gmresdr_args;

   if (do_gels) ComputeEta(args);

   std::vector<ColorSpinorField *> zm((*Zm)(0, args.m));
   std::vector<ColorSpinorField *> vm((*Vm)());

   std::vector<ColorSpinorField*> x_, r_;
   x_.push_back(&x), r_.push_back(&r);

   blas::caxpy(static_cast<Complex *>(args.eta.data()), zm, x_);

   VectorXcd minusHeta = - (args.H * args.eta);
   Ref<VectorXcd> c = args.ritzVecs.col(args.k);
   c += minusHeta;

   blas::caxpy(static_cast<Complex *>(minusHeta.data()), vm, r_);

   return;
 }


 void GMResDR::RestartVZH()
 {
   GMResDRArgs &args = *gmresdr_args;

   ComputeHarmonicRitz(args);

   DenseMatrix Qkp1(MatrixXcd::Identity((args.m+1), (args.k+1)));

   HouseholderQR<MatrixXcd> qr(args.ritzVecs);
   Qkp1.applyOnTheLeft( qr.householderQ());

   DenseMatrix Res = Qkp1.adjoint()*args.H*Qkp1.topLeftCorner(args.m, args.k);
   args.H.setZero();
   args.H.topLeftCorner(args.k+1, args.k) = Res;

   blas::zero( *args.Vkp1 );

   std::vector<ColorSpinorField *> vkp1((*args.Vkp1)());
   std::vector<ColorSpinorField *> vm((*Vm)());

   RowMajorDenseMatrix Alpha(Qkp1); // convert Qkp1 to Row-major format first
   blas::caxpy(static_cast<Complex *>(Alpha.data()), vm, vkp1);

   for(int i = 0; i < (args.m+1); i++)
   {
     if(i < (args.k+1) )
     {
       blas::copy((*Vm)[i], (*args.Vkp1)[i]);
       blas::zero((*args.Vkp1)[i]);
     } else
       blas::zero((*Vm)[i]);
   }

   if( Zm->V() != Vm->V() )
   {
     std::vector<ColorSpinorField *> zm((*Zm)());
     std::vector<ColorSpinorField *> vk((*args.Vkp1)(0, args.k));

     RowMajorDenseMatrix Beta(Qkp1.topLeftCorner(args.m,args.k));
     blas::caxpy(static_cast<Complex *>(Beta.data()), zm, vk);

     for(int i = 0; i < (args.m); i++)
     {
       if (i < (args.k))
         blas::copy((*Zm)[i], (*args.Vkp1)[i]);
       else
         blas::zero((*Zm)[i]);
     }
   }

   //checkCudaError();

   for(int j = 0; j < args.k; j++)
   {
     Complex alpha = cDotProduct((*Vm)[j], (*Vm)[args.k]);
     caxpy(-alpha, (*Vm)[j], (*Vm)[args.k]);
   }

   blas::ax(1.0 / sqrt(blas::norm2((*Vm)[args.k])), (*Vm)[args.k]);

   args.ritzVecs.setZero();
   return;
 }

 //#define CAOPT_ORTH

 int GMResDR::FlexArnoldiProcedure(const int start_idx, const bool do_givens = false)
 {
   int j = start_idx;


   GMResDRArgs &args = *gmresdr_args;

   ColorSpinorFieldSet &vm = *Vm;
   ColorSpinorFieldSet &zm = *Zm;
   ColorSpinorField &tmp = *tmpp;

   if (do_givens) {
     args.givensH.setZero();
     args.cn.setZero();
     args.sn.setZero();
   }

#ifdef CAOPT_ORTH
   // Advanced Ortho objects:
   MatrixXcd R(MatrixXcd::Identity(args.m + 1, args.m + 1));
   MatrixXcd T(MatrixXcd::Identity(args.m + 1, args.m + 1));
   RowMajorDenseMatrix L(args.m, 2);

   while (j < args.m) {

     if(K) {
       ColorSpinorField &inPre = (param.precision_precondition != param.precision_sloppy) ? *r_pre : vm[j];
       ColorSpinorField &outPre = (param.precision_precondition != param.precision_sloppy) ? *p_pre : zm[j];

       if (param.precision_precondition != param.precision_sloppy) inPre = vm[j];
       zero(outPre);
       pushVerbosity(param.verbosity_precondition);
       (*K)( outPre ,inPre );
       popVerbosity();

       if (param.precision_precondition != param.precision_sloppy) zm[j] = outPre;
     }

     matSloppy(vm[j + 1], zm[j], tmp);

     std::vector<ColorSpinorField *> vmj(vm(0, j + 1));
     std::vector<ColorSpinorField *> vm2(vm(j, j + 2));

     blas::cDotProduct(L.block(0, 0, j + 1, 2).data(), vmj, vm2); // single reduction for the iteration

     if (j > start_idx) { // no need to apply normalization scaling for the first iteration (the first basis vector is already normalized)

       if (L(j, 0).real() <= 0) errorQuda("Breakdown detected at itaration %d", j);

       R(j, j) = sqrt(L(j, 0).real()); // extract norm of vm[j] vector

       L.block(0, 0, j, 2) = L.block(0, 0, j, 2) / R(j, j).real();
       L(j, 1) = L(j, 1) / L(j, 0);
       L(j, 0) = 1.0;

       blas::ax(1.0 / R(j, j).real(), vm2);
       // restore the last entry of the Hessenberg
       args.H(j, j - 1) = R(j, j).real();
     }

     R(j, j + 1) = L(j, 1);

     if (j > 0) {
       T.col(j).head(j) = L.col(0).head(j);
       R.col(j + 1).head(j) = L.col(1).head(j);
       T.col(j).head(j) = (-1.0) * T.block(0, 0, j, j) * T.col(j).head(j);
     }

     R.col(j + 1).head(j + 1) = T.block(0, 0, j + 1, j + 1).adjoint() * R.col(j + 1).head(j + 1);

     VectorXcd Rjp1(R.col(j + 1).head(j + 1));

     for (int i = 0; i <= j; i++) Rjp1[i] = -Rjp1[i];

     std::vector<ColorSpinorField *> vmjp1;
     vmjp1.push_back(&vm[j + 1]);

     blas::caxpy(Rjp1.data(), vmj, vmjp1);

     args.H.col(j).head(j + 1) = R.col(j + 1).head(j + 1);

     if (do_givens && j > start_idx) args.Givens(j);

     j += 1;
   }

   R(args.m, args.m) = sqrt(blas::norm2(vm[args.m]));

   // rescale zm vectors
   if (K) {                                              // works only if we have a preconditioner
     std::vector<ColorSpinorField *> zmj(zm(1, args.m)); // we don't need to rescale the first vector zm[0]
     VectorXd invRii(args.m - 1);
     for (int i = 0; i < args.m - 1; i++) invRii[i] = 1.0 / R(i + 1, i + 1).real();
     blas::ax(invRii.data(), zmj);
   }

   // normalize the last vector
   blas::ax(1.0 / R(args.m, args.m).real(), vm[args.m]);
   // set m+1 entry in the last col of the Hessenberg
   args.H(args.m, args.m - 1) = R(args.m, args.m);
#else
   while (j < args.m) {
     if (K) {
       ColorSpinorField &inPre = (param.precision_precondition != param.precision_sloppy) ? *r_pre : vm[j];
       ColorSpinorField &outPre = (param.precision_precondition != param.precision_sloppy) ? *p_pre : zm[j];

       if (param.precision_precondition != param.precision_sloppy) inPre = vm[j];
       zero(outPre);
       pushVerbosity(param.verbosity_precondition);
       (*K)(outPre, inPre);
       popVerbosity();

       if (param.precision_precondition != param.precision_sloppy) zm[j] = outPre;
     }

     matSloppy(vm[j + 1], zm[j], tmp);

     args.H(0, j) = cDotProduct(vm[0], vm[j + 1]);
     caxpy(-args.H(0, j), vm[0], vm[j + 1]);

     for (int i = 1; i <= j; i++) {
       args.H(i, j) = cDotProduct(vm[i], vm[j + 1]);
       caxpy(-args.H(i, j), vm[i], vm[j + 1]);
     }

     args.H(j + 1, j) = Complex(sqrt(norm2(vm[j + 1])), 0.0);
     blas::ax(1.0 / args.H(j + 1, j).real(), vm[j + 1]);

     if (do_givens && j > start_idx) args.Givens(j);

     j += 1;
   }
#endif
   Ref<VectorXcd> c = args.ritzVecs.col(args.k);
   Complex c0 = c(0);

   args.LeastSquaresSolve(c0, *r_sloppy, *Vm, do_givens);

   return (j-start_idx);
 }

 void GMResDR::operator()(ColorSpinorField &x, ColorSpinorField &b)
 {
   if (!param.is_preconditioner) profile.TPSTART(QUDA_PROFILE_INIT);

   const double tol_threshold = 1.2;
   const double det_max_deviation = 0.4;

   if (!init) {

     ColorSpinorParam csParam(b);
     csParam.create = QUDA_ZERO_FIELD_CREATE;
     rp = ColorSpinorField::Create(csParam);
     yp = ColorSpinorField::Create(csParam);
     ep = ColorSpinorField::Create(csParam);

     csParam.setPrecision(param.precision_sloppy);

     tmpp     = ColorSpinorField::Create(csParam);
     r_sloppy = ColorSpinorField::Create(csParam);

     if (K && (param.precision_precondition != param.precision_sloppy)) {

       csParam.setPrecision(param.precision_precondition);
       p_pre = ColorSpinorField::Create(csParam);
       r_pre = ColorSpinorField::Create(csParam);
     }

     csParam.setPrecision(param.precision_sloppy);
     csParam.is_composite = true;
     csParam.composite_dim = nKrylov + 1;

     Vm = ColorSpinorField::Create(csParam);

     csParam.composite_dim = nKrylov;

     Zm = K ? ColorSpinorField::Create(csParam) : Vm;

     csParam.composite_dim = (param.eig_param.n_ev + 1);

     csParam.setPrecision(QUDA_DOUBLE_PRECISION);

     gmresdr_args = std::make_shared<GMResDRArgs>(*Vm, nKrylov, param.eig_param.n_ev);

     init = true;
   }

   GMResDRArgs &args = *gmresdr_args;
   Ref<VectorXcd> c  = args.ritzVecs.col(args.k);

   ColorSpinorField &r   = *rp;
   ColorSpinorField &y   = *yp;
   ColorSpinorField &e   = *ep;

   ColorSpinorField &rSloppy = *r_sloppy;

   if (!param.is_preconditioner) {
     profile.TPSTOP(QUDA_PROFILE_INIT);
     profile.TPSTART(QUDA_PROFILE_PREAMBLE);
   }

   int tot_iters = 0;

   double normb = norm2( b );
   double stop = param.tol * param.tol * normb;

   mat(r, x);

   double r2 = xmyNorm(b, r);
   double b2 = r2;
   c(0) = Complex(sqrt(r2), 0.0);

   printfQuda("\nInitial residual squared: %1.16e, source %1.16e, tolerance %1.16e\n", r2, sqrt(normb), param.tol);

   rSloppy = r;

   if(param.precision_sloppy != param.precision) {
     blas::axpy(1.0 / c(0).real(), r, y);
     Vm->Component(0) = y;
     blas::zero(y);
   } else {
     blas::axpy(1.0 / c(0).real(), r, Vm->Component(0));
   }

   if (!param.is_preconditioner) {
     profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
     profile.TPSTART(QUDA_PROFILE_COMPUTE);
     blas::flops = 0;
   }

   const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

   double heavy_quark_res = 0.0;
   if (use_heavy_quark_res)  heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

   int restart_idx = 0, j = 0, check_interval = 8;

   DenseMatrix Gm = DenseMatrix::Zero(param.eig_param.n_ev + 1, param.eig_param.n_ev + 1);

   while (restart_idx < param.max_restart_num
           && !(convergence(r2, heavy_quark_res, stop, param.tol_hq) || !(r2 > stop))) {
     tot_iters += FlexArnoldiProcedure(j, (j == 0));
     UpdateSolution(e, rSloppy, !(j == 0));

     r2 = norm2(rSloppy);

     bool   do_clean_restart = false;
     double ext_r2 = 1.0;

     if ((restart_idx + 1) % check_interval == 0) {
       mat(y, e);
       ext_r2 = xmyNorm(r, y);

	// can this be done as a single 2-d reduction?
       for (int l = 0; l < param.eig_param.n_ev + 1; l++) {
         Complex *col = Gm.col(l).data();
         std::vector<ColorSpinorField *> v1_(Vm->Components().begin(),
                                              Vm->Components().begin() + param.eig_param.n_ev + 1);
         std::vector<ColorSpinorField*> v2_;
	       v2_.push_back(static_cast<ColorSpinorField*>(&Vm->Component(l)));

	       blas::cDotProduct(col, v1_, v2_);
       } // end l-loop

       Complex detGm = Gm.determinant();

	     PrintStats("FGMResDR:", tot_iters, r2, b2, heavy_quark_res);
	     printfQuda("\nCheck cycle %d, true residual squared %1.15e, Gramm det : (%le, %le)\n", restart_idx, ext_r2, detGm.real(), detGm.imag());

	     Gm.setZero();

	     do_clean_restart = ((sqrt(ext_r2) / sqrt(r2)) > tol_threshold) || fabs(1.0 - (norm(detGm)) > det_max_deviation);
     }

     if ((param.max_restart_num != 1) && ((restart_idx != param.max_restart_num - 1) && !do_clean_restart)) {

       RestartVZH();
	     j = args.k;

     } else {

       printfQuda("\nClean restart for cycle %d, true residual squared %1.15e\n", restart_idx, ext_r2);
       args.ResetArgs();

       // update solution:
       xpy(e, x);
       r = y;
       zero(e);

       c(0) = Complex(sqrt(ext_r2), 0.0);
       blas::zero(Vm->Component(0));
       blas::axpy(1.0 / c(0).real(), rSloppy, Vm->Component(0));

       j = 0;
     }

    restart_idx += 1;
  }

  //final solution:
  xpy(e, x);

  if (!param.is_preconditioner) {
    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + mat.flops()) * 1e-9;
    param.gflops = gflops;
    param.iter += tot_iters;
  }

  mat(r, x);

  param.true_res = sqrt(xmyNorm(b, r) / b2);

  PrintSummary("FGMResDR:", tot_iters, r2, b2, stop, param.tol_hq);

  printfQuda("Done with %d cycles..", restart_idx);

  blas::flops = 0;
  mat.flops();

  if (!param.is_preconditioner) profile.TPSTOP(QUDA_PROFILE_EPILOGUE);

  param.rhs_idx += 1;

  if(init) {
    delete rp;
    delete yp;
    delete ep;

    delete tmpp;
    delete r_sloppy;

    delete Vm;
    if(K) {
      if(p_pre) delete p_pre;
      if(r_pre) delete r_pre;
      delete Zm; 
    }
  }

   return;
}

} // namespace quda
