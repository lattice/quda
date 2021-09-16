#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory>
#include <future>
#include <iostream>

#include <string.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <eigen_helper.h>

/*
Based on eigCG(n_ev, m) algorithm:
A. Stathopolous and K. Orginos, arXiv:0707.0131

Pipelined version is based on :
T. Chen and E. Carson, arXiv:1905.01549
See also: G. Meurant, Parellel Computing Vol 5, Issue 3, p. 267, 1987
*/

namespace quda {

   using namespace blas;

   using DynamicStride = Stride<Dynamic, Dynamic>;
   using RealMatrix = MatrixXd;
   using RealMatrixF = MatrixXf;
   using RealVectorSet = MatrixXd;
   using RealVector = VectorXd;
   using RealVectorF = VectorXf;
   using RealDiagonalMatrix = DiagonalMatrix<double, Dynamic>;

   // special types needed for compatibility with QUDA blas:
   using RowMajorRealMatrix = Matrix<double, Dynamic, Dynamic, RowMajor>;
   using RowMajorComplexMatrix = Matrix<Complex, Dynamic, Dynamic, RowMajor>;

   constexpr int default_eigcg_cycles = 4;
   static int max_eigcg_cycles = default_eigcg_cycles;//how many eigcg cycles do we allow?
   static std::array<int, default_eigcg_cycles> search_space_restarts{4,4,3,3};//minimum number of the search space restarts

   constexpr bool host_flag = false;

   static int init_k = 0;

   int Lm = 0;
   int L2kE = 0;
   int L2kO = 0;

   EigCGArgs *IncEigCG::eigcg_args = nullptr;

   class EigCGArgs
   {

   public:
     // host Lanczos matrice, and its eigenvalue/vector arrays:
     RealMatrix Tm; // VH A V,
     // host global proj matrix:
     Complex *projMat;
     // eigenvectors:
     RealVectorSet eigenVecs; // array of  (m)  ritz and of m length
     // eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)
     RealVector Tmvals; // eigenvalues of T[m,  m  ] and T[m-1, m-1] (re-used)

     int id; // current search spase index

     int restarts;
     double global_stop;

     bool run_residual_correction; // used in mixed precision cycles
     // Cached invert residual norm:
     RealDiagonalMatrix inv_normr_m;
     // Array of (local) dot products
     RealVector s;

     /** Pipeline specific parameters:	*/
     const bool is_host_location; // where to perform computations (host or device)
     int shift;

     // Cached lanczos elements (diagonal and off-diagonal elements)
     RealVectorSet cached_lanczos_elems;

     // Cached inverse residual norms:
     RealVector cached_inv_normr;

     // C++ task-based object for the (asynchron.) RayleighRitz computations
     std::future<void> rr_task;

     ColorSpinorField    *Az;     // mat * conjugate vector from the previous iteration
     ColorSpinorFieldSet *Vm;  // eigCG search vectors  (spinor matrix of size eigen_vector_length x m)
     ColorSpinorFieldSet *V2k; // temp vector set

     ColorSpinorField    *hAz;    // mat * conjugate vector from the previous iteration
     ColorSpinorFieldSet *hVm; // eigCG search vectors  (spinor matrix of size eigen_vector_length x m)

     SolverParam &solver_param;

     int logical_rhs_id; //current eigcg cycle

     EigCGArgs(ColorSpinorField &meta, SolverParam &solver_param, const bool is_host_location = false) :
       Tm(RealMatrix::Zero(solver_param.eig_param.n_kr, solver_param.eig_param.n_kr)),
       projMat(nullptr),
       eigenVecs(RealVectorSet::Zero(solver_param.eig_param.n_kr, solver_param.eig_param.n_kr)),
       Tmvals(solver_param.eig_param.n_kr),
       id(0),
       restarts(0),
       global_stop(0.0),
       run_residual_correction(false),
       inv_normr_m(solver_param.eig_param.n_kr),
       s(RealVector::Zero(2 * solver_param.eig_param.nLockedMax)),
       is_host_location(is_host_location),
       shift(0),
       cached_lanczos_elems(RealVectorSet::Zero(solver_param.pipeline, 2)),
       cached_inv_normr(RealVector::Zero(solver_param.pipeline + 1)),
       Az(nullptr),
       Vm(nullptr),
       V2k(nullptr),
       hAz(nullptr),
       hVm(nullptr),
       solver_param(solver_param),
       logical_rhs_id(0)
     {
       const int m = solver_param.eig_param.n_kr;
       const int k = solver_param.eig_param.nLockedMax;

       printfQuda("\n\nAllocating local resources ... \n");
       projMat = (Complex *)safe_malloc(solver_param.eig_param.n_ev * solver_param.eig_param.n_ev * sizeof(Complex));

       ColorSpinorParam csParam(meta);
       csParam.create = QUDA_ZERO_FIELD_CREATE;
       csParam.is_composite = false;
       csParam.composite_dim = 1;
       // csParam.setPrecision(solver_param.precision_sloppy);
       csParam.setPrecision(solver_param.precision);
       //
       Az = ColorSpinorField::Create(csParam);

       csParam.is_composite = true;
       csParam.composite_dim = is_host_location ?
         2 * solver_param.pipeline :
         m + solver_param.pipeline; // solver_param.pipeline = 0 for the legacy version
       // csParam.composite_dim = m;
       csParam.setPrecision(
         solver_param.eig_param
           .cuda_prec_ritz); // eigCG internal search space precision may not coincide with the solver precision!

       // Create a search vector set:
       Vm = ColorSpinorField::Create(csParam);

       if (is_host_location) printfQuda("Running eigenvalue computations on the host.\n");

       char *enabled_managed_memory = getenv("QUDA_ENABLE_MANAGED_MEMORY");
       if (enabled_managed_memory && strcmp(enabled_managed_memory, "1") == 0) {
         // nothing to do
       } else {
         if (is_host_location) csParam.mem_type = QUDA_MEMORY_MAPPED;
       }

       // csParam.mem_type      = solver_param.pipeline != 0 ? QUDA_MEMORY_MAPPED : QUDA_MEMORY_DEVICE;
       // csParam.setPrecision(solver_param.eig_param.cuda_prec_ritz);
       // csParam.setPrecision(solver_param.precision);
       csParam.setPrecision(is_host_location ? (solver_param.precision_sloppy == QUDA_HALF_PRECISION ? QUDA_SINGLE_PRECISION : solver_param.precision_sloppy) : solver_param.precision);
       csParam.composite_dim = (2 * k);
       // Create a search vector set:
       V2k = ColorSpinorField::Create(csParam);

       if (solver_param.pipeline != 0 && is_host_location) {
         // set up pipelined shifts:
         int leftover_elems = (m - (m / solver_param.pipeline) * solver_param.pipeline);
         // adjust Lm shift:
         if (leftover_elems != 0) Lm = (solver_param.pipeline - leftover_elems);

         const int am = ((m + Lm) / solver_param.pipeline) & 1;

         // Adjust Odd cycle shift
         leftover_elems = (2 * k - ((2 * k) / solver_param.pipeline) * solver_param.pipeline);

         if (leftover_elems != 0) L2kO = (solver_param.pipeline - leftover_elems);

         const int a2k = ((2 * k + solver_param.pipeline + L2kO) / solver_param.pipeline) & 1;

         L2kO = am == 0 ? L2kO + (1 - a2k) * solver_param.pipeline : L2kO + a2k * solver_param.pipeline;

         // Adjust Even cycle shift:
         const int am2k = ((m - 2 * k - solver_param.pipeline) / solver_param.pipeline) & 1;

         if (am2k == 0)
           L2kE = am == 0 ? L2kE + a2k * solver_param.pipeline : L2kE + (1 - a2k) * solver_param.pipeline;
         else
           L2kE = L2kO;

         printfQuda("\nPipeline shift parameters : Lm = % d, L2kO = %d, L2kE = %d\n", Lm, L2kO, L2kE);

         csParam.composite_dim = m;
         // csParam.setPrecision(solver_param.eig_param.cuda_prec_ritz);
         // eigCG internal search space precision may not coincide with the solver precision
	 csParam.setPrecision(solver_param.precision_sloppy == QUDA_HALF_PRECISION ? QUDA_SINGLE_PRECISION : solver_param.precision_sloppy);
         // Create a search vector set:
         hVm = ColorSpinorField::Create(csParam);
         //
         csParam.is_composite = false;
         csParam.composite_dim = 1;
         // csParam.setPrecision(solver_param.precision);
         hAz = ColorSpinorField::Create(csParam);
       }

       Eigen::initParallel();
     }

     virtual ~EigCGArgs() {
       host_free(projMat);
       if(Az) {delete Az; Az=nullptr;}
       if(Vm) {delete Vm; Vm=nullptr;}
       if(V2k){delete V2k; V2k=nullptr;}
       if(hVm){delete hVm; hVm=nullptr;}
       if(hAz){delete hAz; hAz=nullptr;}
     }

     template <bool is_pipelined = false> inline void UpdateLanczosMatrix(double diag_val, double offdiag_val)
     {
       const int m = solver_param.eig_param.n_kr;

       if (run_residual_correction || id == 0) return;
       const int cid = id - 1;

       if (is_pipelined) {
         if (cid >= m) { // store lanczos coeff in the buffers
           cached_lanczos_elems(cid - m, 0) = diag_val;
           cached_lanczos_elems(cid - m, 1) = offdiag_val;
           return;
         }
       }
       // Load Lanczos off-diagonals:
       if (cid < (m - 1)) {
         Tm.diagonal<+1>()[cid] = offdiag_val;
         Tm.diagonal<-1>()[cid] = offdiag_val;
       }
       if (cid < m) Tm.diagonal<0>()[cid] = diag_val;

       return;
     }

     inline void CacheInvRNorm(const double &normr)
     {
       const int m = solver_param.eig_param.n_kr;

       if (id < m) inv_normr_m.diagonal()[id] = 1.0 / normr;
       // store in the intermediate buffer otherwise:
       else
         cached_inv_normr[(id % m)] = 1.0 / normr;
     }

     inline void CleanArgs()
     {
       id = 0;
       Tm.setZero();
       Tmvals.setZero();
       eigenVecs.setZero();
       shift = 0;
       if (rr_task.valid()) rr_task.wait();
     }

     inline void UpdateShift()
     {
       const int m = solver_param.eig_param.n_kr;

       if (!is_host_location) return;

       if (id == m)
         shift = Lm;
       else
         shift = restarts & 1 ? L2kE : L2kO;
     }

     void RestartArgs()
     {
       const int k = solver_param.eig_param.nLockedMax;

       Tm.setZero();
       inv_normr_m.setIdentity();

       for (int i = 0; i < 2 * k; i++) Tm(i, i) = Tmvals(i);

       if (solver_param.pipeline
           != 0) { // need to finish global reduction and restore search space for the pipelined version

         for (int i = 0; i < solver_param.pipeline; i++) {
           Tm.diagonal<0>()[2 * k + i] = cached_lanczos_elems(i, 0);
           Tm.diagonal<+1>()[2 * k + i] = cached_lanczos_elems(i, 1);
           Tm.diagonal<-1>()[2 * k + i] = cached_lanczos_elems(i, 1);
           inv_normr_m.diagonal()[2 * k + i] = cached_inv_normr[i];
         }
       }
       inv_normr_m.diagonal()[2 * k + solver_param.pipeline]
         = cached_inv_normr[solver_param.pipeline]; // extra term even for the non-pipelined version

       s *= cached_inv_normr[0]; // this correspond to the residual norm for the first after the (postponed) restart

       Tm.col(2 * k).segment(0, 2 * k) = s;
       Tm.row(2 * k).segment(0, 2 * k) = s;

       id = 2 * k + solver_param.pipeline;
       restarts += 1;
       s.setZero();

       return;
     }

     void RayleighRitz()
     {
       const int m = solver_param.eig_param.n_kr;
       const int k = solver_param.eig_param.nLockedMax;

       // 1.Solve m dim eigenproblem:
       SelfAdjointEigenSolver<MatrixXd> es_tm(Tm);

       eigenVecs.leftCols(k) = es_tm.eigenvectors().leftCols(k);

       // 2.Solve m-1 dim eigenproblem:
       SelfAdjointEigenSolver<RealMatrix> es_tm1(Tm.block(0, 0, (m - 1), (m - 1)));

       eigenVecs.block(0, k, m - 1, k) = es_tm1.eigenvectors().leftCols(k);

       eigenVecs.block(m - 1, k, 1, k).setZero();

       MatrixXd Q2k(RealMatrix::Identity(m, 2 * k));

       HouseholderQR<RealMatrix> eigenVecs2k_qr(eigenVecs.block(0, 0, m, 2 * k));
       Q2k.applyOnTheLeft(eigenVecs2k_qr.householderQ());
       // ColPivHouseholderQR<RealMatrix> eigenVecs2k_qr( eigenVecs.block(0, 0, m, 2*k) );
       // Q2k.applyOnTheLeft(eigenVecs2k_qr.householderQ().setLength(eigenVecs2k_qr.nonzeroPivots()));

       // 3. Construct H = QH*Tm*Q :
       RealMatrix H2k2 = Q2k.transpose() * Tm * Q2k;

       /* solve the small evecm1 2nev x 2nev eigenproblem */
       SelfAdjointEigenSolver<RealMatrix> es_h2k(H2k2);

       // Block<MatrixXd>(eigenVecs.derived(), 0, 0, m, 2*k) = Q2k * es_h2k.eigenvectors();
       RealMatrix Qm2k = Q2k * es_h2k.eigenvectors();

       Tmvals.segment(0, 2 * k) = es_h2k.eigenvalues();

       // 4. Rescale eigenvectors since we did not rescale Vm
       eigenVecs.block(0, 0, m, 2 * k) = inv_normr_m * Qm2k;

       // 5. Synchronize an aux compute stream for the pipelined version:
       //if (is_host_location) blas::synchronizeAuxBlasStream();

       // 6. Compute VQ
       ColorSpinorFieldSet &v2k = *V2k;
       ColorSpinorFieldSet &vm = is_host_location ? *hVm : *Vm;

       if (is_host_location) {
         ColorSpinorFieldSet &haz = *hAz;

         RealMatrix Alpha(eigenVecs.block(0, 0, m, 2 * k));

         if (vm.Precision() == QUDA_DOUBLE_PRECISION) {
           Map<RealMatrix, Unaligned> eigenv2kmat(static_cast<double *>(v2k.V()), v2k[0].RealLength(), 2 * k);
           Map<RealMatrix, Unaligned> eigenvmmat(static_cast<double *>(vm.V()), vm[0].RealLength(), m);

           eigenv2kmat.setZero();
           eigenv2kmat.noalias() += eigenvmmat * Alpha;

           Map<RealVector, Unaligned> az(static_cast<double *>(haz.V()), haz.RealLength());

           s.setZero();
           s.noalias() += eigenv2kmat.adjoint() * az;
         } else {
           Map<RealMatrixF, Unaligned> eigenv2kmat(static_cast<float *>(v2k.V()), v2k[0].RealLength(), 2 * k);
           Map<RealMatrixF, Unaligned> eigenvmmat(static_cast<float *>(vm.V()), vm[0].RealLength(), m);

           RealMatrixF AlphaF(m, 2 * k);
           for (int row = 0; row < m; row++)
             for (int col = 0; col < 2 * k; col++) AlphaF(row, col) = Alpha(row, col);

           eigenv2kmat.setZero();
           eigenv2kmat.noalias() += eigenvmmat * AlphaF;

           Map<RealVectorF, Unaligned> az(static_cast<float *>(haz.V()), haz.RealLength());

           RealVectorF sf(RealVectorF::Zero(2 * k));
           sf.noalias() += eigenv2kmat.adjoint() * az;
           for (int el = 0; el < 2 * k; el++) s(el) = sf(el);
         }

       } else {
         std::vector<ColorSpinorField *> v2k_(v2k());

         blas::zero(v2k);

         std::vector<ColorSpinorField *> vm_(vm(0, m));

         RowMajorRealMatrix Alpha(eigenVecs.block(0, 0, m, 2 * k));

         blas::axpy(Alpha.data(), vm_, v2k_);

         std::vector<ColorSpinorField *> Az_;

         Az_.push_back(Az);

         blas::reDotProduct(s.data(), Az_, v2k_);
       }

       for (int j = 0; j < 2 * k; j++) blas::copy(vm[j], v2k[j]);
       // vm.CopySubset(v2k, 2*k, 0);

       return;
     }

     inline void StagedPrefetch(const ColorSpinorField &w0)
     {
       const int m = solver_param.eig_param.n_kr;
       const int k = solver_param.eig_param.nLockedMax;

       static int ref_pnt = 0;

       if (id == 0) {
         ref_pnt = 0;
         return;
       }

       const bool last_stage = (id == m); // ready for the restart, before that we need to prefech leftover basis vectors
       const int cid = id % (m + 1) + (id > m ? 2 * k : 0);
       const int d = (cid - ref_pnt) % solver_param.pipeline;

       if (last_stage) {
         blas::copy(*Az, w0);         // using a regular stream
         if (ref_pnt == 0) ref_pnt = 2 * k; // redefine the reference point...
       }

       if (!is_host_location) {

         if (id == (2 * k + solver_param.pipeline) && restarts > 0) {
           ColorSpinorFieldSet &vm = *Vm;
           for (int i = 0; i < solver_param.pipeline; i++) vm[2 * k + i] = vm[m + i];
         }

         return;
       }

       const int l = (id + shift) % Vm->CompositeDim();

       if ((d == 0 && (cid - ref_pnt) != 0) || last_stage) { // prefetch gpu vectors on the host:

         ColorSpinorFieldSet &vm = *Vm;
         ColorSpinorFieldSet &hvm = *hVm;

         const int elements = d == 0 ? solver_param.pipeline : d; // might be d for the last copy stage
         //
         //blas::registerAuxBlasStream();

         const int dst_offset = cid - elements;
         const int src_offset = l - elements + (l == 0 ? Vm->CompositeDim() : 0);

         // hvm.CopySubset(vm, elements, dst_offset, src_offset);
         for (int i = 0; i < elements; i++) blas::copy(hvm[dst_offset + i], vm[src_offset + i]);

         if (last_stage) blas::copy(*hAz, *Az);
         //blas::unregisterAuxBlasStream();
         //
       }

       return;
     }

     inline void UpdateLanczosBasisAndSearchIdx(const ColorSpinorField &r)
     {
       ColorSpinorFieldSet &vm = *Vm;
       // load Lanczos basis vector:
       const int cid = (id + shift) % vm.CompositeDim();

       blas::copy(vm[cid], r); // convert arrays

       id += 1;

       return;
     }
   };

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
    inner.deflate = false;

    inner.delta = 1e-2;                       // outer.delta;
    inner.precision        = outer.precision; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.inv_type        = QUDA_CG_INVERTER;       // use CG solver
    inner.use_init_guess  = QUDA_USE_INIT_GUESS_YES;// use deflated initial guess...

    inner.use_sloppy_partial_accumulator= false;//outer.use_sloppy_partial_accumulator;
  }

  IncEigCG::IncEigCG(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
                     SolverParam &param, TimeProfile &profile) :
    Solver(mat, matSloppy, matPrecon, matSloppy, param, profile),
    matDefl(matSloppy),
    K(nullptr),
    Kparam(param),
    m(param.eig_param.n_kr),
    ep(nullptr),
    ep_sloppy(nullptr),
    rp(nullptr),
    rp_sloppy(nullptr),
    work_space(nullptr),
    r_pre(nullptr),
    p_pre(nullptr),
    profile(profile),
    init(false)
  {

    if (2 * param.eig_param.nLockedMax >= param.eig_param.n_kr)
      errorQuda(
        "Incorrect number of the requested low modes: m= %d while nev=%d (note that 2*nev must be less then m).",
        param.eig_param.n_kr, param.eig_param.nLockedMax);

    if (param.eig_param.n_conv < param.eig_param.n_ev)
      printfQuda("\nInitialize eigCG(m=%d, nev=%d) solver.", param.eig_param.n_kr, param.eig_param.nLockedMax);
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
      K = std::make_shared<CG>(matPrecon, matPrecon, matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_MR_INVERTER){
      K = std::make_shared<MR>(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_SD_INVERTER){
      K = std::make_shared<SD>(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition != QUDA_INVALID_INVERTER){ // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
    }

    if (param.rhs_idx > 0)
      injectDeflationSpace(reinterpret_cast<deflation_space *>(param.eig_param.preserve_deflation_space)->evecs, false);

    return;
  }

  IncEigCG::~IncEigCG() { }

  void IncEigCG::EigenSolve() { eigcg_args->RayleighRitz(); }

  void IncEigCG::LegacySearchSpaceUpdate(const double &lanczos_diag, const double &lanczos_offdiag, const double &beta,
                                         const double &rnorm)
  {

    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) return;

    ColorSpinorField &Ap = (*work_space)[0];
    // whether we want preconditiond (id = 5) or unpreconditioned (id = 2) residual:
    const int r_idx = K ? 5 : 2;
    ColorSpinorField &z = (*work_space)[r_idx];

    args.UpdateLanczosMatrix(lanczos_diag, lanczos_offdiag);
    args.CacheInvRNorm(rnorm);

    if (args.id == (m - 1)) blas::copy(*args.Az, Ap);
    // run RR block:
    else if (args.id == m) {
      // Compute (update) Ap = Ap - beta*Ap_old
      blas::xpay(Ap, -beta, *args.Az);

      args.RayleighRitz();

      args.RestartArgs();
    }

    args.UpdateLanczosBasisAndSearchIdx(z);

    return;
  }

  void IncEigCG::PipelinedSearchSpaceUpdate(const double &lanczos_diag, const double &lanczos_offdiag, const double &normr)
  {
    constexpr bool is_pipelined = true;

    EigCGArgs &args = *eigcg_args;

    if (args.run_residual_correction) return;

    args.UpdateLanczosMatrix<is_pipelined>(lanczos_diag, lanczos_offdiag);
    args.CacheInvRNorm(normr);
    args.StagedPrefetch((*work_space)[0]);

    // launch RR block for args.id == param.m, otherwise is nop
    if (args.id == m) {
      // launch async task (current hack for GPU based computing)
      args.rr_task
        = std::async(args.is_host_location ? std::launch::async : std::launch::deferred, &IncEigCG::EigenSolve, this);

      if (args.restarts == 0) args.UpdateShift();

    } else if (args.id == (m + args.solver_param.pipeline)) {
      // do synchronization with EIGENV
      if (args.rr_task.valid())
        args.rr_task.wait();
      else
        warningQuda("Tried to synchronize an invalid task...");
      args.UpdateShift();
      //
      args.RestartArgs();
      //
      args.StagedPrefetch((*work_space)[0]);
    }

    args.UpdateLanczosBasisAndSearchIdx((*work_space)[4]);

    return;
  }

  void IncEigCG::Increment()
  {
    EigCGArgs &args = *eigcg_args;

    const unsigned int first_idx = param.eig_param.n_conv;
    const unsigned int n_evec = param.eig_param.n_ev;
    const unsigned int k = param.eig_param.nLockedMax;

    ColorSpinorFieldSet &vm = *args.Vm;
    ColorSpinorFieldSet &ws = *work_space;

    if (evecs.size() < (first_idx + k) || n_evec < (first_idx + k)) { //!
      warningQuda("\nNot enough space to add %u vectors. Keep deflation space unchanged.\n", k);
      return;
    }

    printfQuda("\nOrthonormalize new eigenvectors..\n");

#if 1 // communication optimized version
    MatrixXcd R ( MatrixXcd::Identity(first_idx+k, first_idx+k) );
    MatrixXcd T ( MatrixXcd::Identity(first_idx+k, first_idx+k) );
    RowMajorComplexMatrix L (first_idx+k-1, 2);

    for (unsigned int j = 0; j < k; j++) {//extra step to include the last vector normalization

      const unsigned int i = first_idx+j;

      printfQuda("Working with vector  %u .\n", i);

      *evecs[i] = vm[j];
      // skip the first vector

      if (first_idx == 0 && j == 0) continue;

      std::vector<ColorSpinorField*> rvj(evecs.begin(), evecs.begin() + i);
      std::vector<ColorSpinorField*> rv2(evecs.begin() + (i-1), evecs.begin() + (i+1));

      blas::cDotProduct(L.block(0,0,i,2).data(), rvj, rv2);

      R(i-1,i-1) = sqrt( L(i-1, 0).real() );
      R(i-1,i  ) = L(i-1, 1) / R(i-1,i-1);

      if( i > 1 ) {
	T.col(i-1).head(i-1) = L.col(0).head(i-1) / R(i-1, i-1);
	R.col(i).head(i-1)   = L.col(1).head(i-1);
	T.col(i-1).head(i-1) = (-1.0)*T.block(0,0,i-1,i-1)*T.col(i-1).head(i-1);
      }

      R.col(i).head(i) = T.block(0,0,i,i).adjoint() * R.col(i).head(i);

      //Normalization of the previous vectors:
      if(R(i-1, i-1).real() < 1e-8)  errorQuda("\nCannot orthogonalize %dth vector\n", i-1);

      blas::ax(1.0 / R(i-1, i-1).real(), *evecs[i-1]);

      VectorXcd Rj( R.col(i).head(i) );
      std::vector<ColorSpinorField*> rvjp1{evecs[i]};

      for(unsigned int l = 0; l < i; l++) Rj[l] = -Rj[l];

      blas::caxpy( Rj.data(), rvj, rvjp1);
    } // end for loop over j

    //extra step to include the last vector normalization

    R(first_idx+k-1,first_idx+k-1) = sqrt(blas::norm2(*evecs[first_idx+k-1]));
    blas::ax(1.0 / R(first_idx+k-1,first_idx+k-1).real(), *evecs[first_idx+k-1]);

#else // old legacy version

    // Block MGS orthogonalization
    // The degree to which we interpolate between modified GramSchmidt and GramSchmidt (performance vs stability)
    const unsigned int cdot_pipeline_length = 4;

    for (unsigned int j = 0; j < k; j++) {

      const unsigned int i = first_idx + j;
      *evecs[i] = vm[j];

      if(i>0) {
	//std::unique_ptr<Complex[]> alpha(new Complex[i]);
	//printfQuda("i = %d\n", i);
	std::vector<Complex> alpha(i);// = new Complex[i];
	unsigned int offset = 0;

	while (offset < i) {
	  const unsigned int local_length = (i - offset) > cdot_pipeline_length ? cdot_pipeline_length : (i - offset);

	  std::vector<ColorSpinorField *> vj_(evecs.begin() + offset, evecs.begin() + offset + local_length);
	  std::vector<ColorSpinorField *> vi_ {evecs[i]};

	  //blas::cDotProduct(alpha.get(), vj_, vi_);
	  blas::cDotProduct(alpha.data(), vj_, vi_);
	  for (unsigned int l = 0; l < local_length; l++) alpha[l] = -alpha[l];
	  //blas::caxpy(alpha.get(), vj_, vi_);
	  blas::caxpy(alpha.data(), vj_, vi_);
	  offset += cdot_pipeline_length;
	}

	alpha[0].real(blas::norm2(*evecs[i]));
	alpha[0].imag(0.0);

	if (alpha[0].real() > 1e-16)
	  blas::ax(1.0 / sqrt(alpha[0].real()), *evecs[i]);
	else
	  errorQuda("\nCannot orthogonalize %dth vector\n", i);
      }
    }

#endif
    printfQuda("\nConstruct projection matrix..\n");

    for (unsigned int i = first_idx; i < (first_idx + k); i++) {
      std::unique_ptr<Complex[]> alpha(new Complex[i + 1]);

      ColorSpinorField &vk0 = (*work_space)[0];
      ColorSpinorField &swp = work_space->Precision() != evecs[i]->Precision() ? (*work_space)[1] : *evecs[i];

      std::vector<ColorSpinorField *> vj_(evecs.begin(), evecs.begin() + i + 1);
      // std::vector<ColorSpinorField*> av_(vk(0));
      std::vector<ColorSpinorField *> av_(ws(0));

      swp = *evecs[i];
      matDefl(vk0, swp);
      *evecs[i] = swp;

      blas::cDotProduct(alpha.get(), vj_, av_);

      args.projMat[i * param.eig_param.n_ev + i] = alpha[i]; //!

      for (unsigned int j = 0; j < i; j++) {
        args.projMat[i * n_evec + j] = alpha[j];
        args.projMat[j * n_evec + i] = conj(alpha[j]);
      } //!
    }   // end for loop

    param.eig_param.n_conv += k;
    printfQuda("\nNew curr deflation space dim = %d\n", param.eig_param.n_conv);
    return;
  }

  void IncEigCG::Deflate(ColorSpinorField &x, ColorSpinorField &b)
  {
    EigCGArgs &args = *eigcg_args;

    if (param.eig_param.n_conv == 0) return; // nothing to do

    std::unique_ptr<Complex[]> vec(new Complex[param.eig_param.n_ev]);

    double check_nrm2 = norm2(b);

    printfQuda("\nSource norm (gpu): %1.15e, curr deflation space dim = %d\n", sqrt(check_nrm2), param.eig_param.n_conv);

    std::vector<ColorSpinorField *> rv_(evecs.begin(), evecs.begin() + param.eig_param.n_conv);
    std::vector<ColorSpinorField *> in_ {static_cast<ColorSpinorField *>(&b)};

    blas::cDotProduct(vec.get(), rv_, in_); //<i, b>

    Map<VectorXcd, Unaligned> vec_(vec.get(), param.eig_param.n_conv);
    Map<MatrixXcd, Unaligned, DynamicStride> projm_(args.projMat, param.eig_param.n_conv, param.eig_param.n_conv,
                                                    DynamicStride(param.eig_param.n_ev, 1));

    vec_ = projm_.fullPivHouseholderQr().solve(vec_);

    std::vector<ColorSpinorField *> out_ {&x};
    blas::caxpy(vec.get(), rv_, out_); // multiblas

    check_nrm2 = norm2(x);
    printfQuda("\nDeflated guess spinor norm (gpu): %1.15e\n", sqrt(check_nrm2));

    return;
  }

  void IncEigCG::Reduce(double tol, int max_nev)
  {
    EigCGArgs &args = *eigcg_args;

    blas::zero(*args.Vm);
    if(args.is_host_location) blas::zero(*args.hVm);

    if (max_nev == 0 || param.eig_param.n_conv == 0) {
      printfQuda("Deflation space is empty.\n");
      return;
    }

    if (param.eig_param.n_conv < max_nev) {

      printf("\nToo big number of eigenvectors was requested, switched to maximum available number %d\n",
             param.eig_param.n_conv);
      max_nev = param.eig_param.n_conv;
    }

    std::unique_ptr<double[]> devals(new double[param.eig_param.n_conv]);
    std::unique_ptr<Complex[]> projm(new Complex[param.eig_param.n_ev * param.eig_param.n_conv]);

    memcpy(projm.get(), args.projMat, param.eig_param.n_ev * param.eig_param.n_conv * sizeof(Complex)); //!

    Map<MatrixXcd, Unaligned, DynamicStride> projm_(projm.get(), param.eig_param.n_conv, param.eig_param.n_conv,
                                                    DynamicStride(param.eig_param.n_ev, 1));
    Map<VectorXd, Unaligned> devals_(devals.get(), param.eig_param.n_conv);

    SelfAdjointEigenSolver<MatrixXcd> es(projm_);

    projm_ = es.eigenvectors(), devals_ = es.eigenvalues();

    ColorSpinorParam csParam(*evecs[0]);

    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.is_composite = false;
    std::unique_ptr<ColorSpinorField> r_sloppy(ColorSpinorField::Create(csParam));

    csParam.setPrecision(QUDA_DOUBLE_PRECISION);
    std::unique_ptr<ColorSpinorField> r(ColorSpinorField::Create(csParam));
    std::unique_ptr<ColorSpinorField> t(ColorSpinorField::Create(csParam));

    csParam.is_composite = true;
    csParam.composite_dim = max_nev;

    csParam.setPrecision(evecs[0]->Precision());
    csParam.mem_type = QUDA_MEMORY_MAPPED;
    std::unique_ptr<ColorSpinorField> buff(ColorSpinorField::Create(csParam));

    int idx = 0;
    double relerr = 0.0;
    bool do_residual_check = (tol != 0.0);

    while ((relerr < tol) && (idx < max_nev)) {

      if (fabs(devals[idx]) < 1e-16) errorQuda("\n .. zero Ritz value..\n");

      std::vector<ColorSpinorField *> rv_(evecs.begin(), evecs.begin() + param.eig_param.n_conv);
      std::vector<ColorSpinorField *> res {r.get()};

      blas::zero(*r);
      blas::caxpy(&projm.get()[idx * param.eig_param.n_ev], rv_, res);

      blas::copy(buff->Component(idx), *r);

      if (do_residual_check) // if tol=0.0 then disable relative residual norm check
      {
        mat(*t, *r);

        double3 dotnorm = cDotProductNormA(*r, *t);
        double curr_eval = dotnorm.x / dotnorm.z;

        blas::xpay(*t, -curr_eval, *r);
        relerr = sqrt(norm2(*r) / dotnorm.z);
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Eigenvalue: %1.12e Residual: %1.12e\n", curr_eval, relerr);
      }

      idx += 1;
    }

    // reset current dimension:
    param.eig_param.n_conv = idx;
    param.eig_param.n_ev_deflate = param.eig_param.n_conv;

    printfQuda("\nReserved eigenvectors: %d\n", param.eig_param.n_conv);

    deflation_space *reserved_space = reinterpret_cast<deflation_space *>(param.eig_param.preserve_deflation_space);

    reserved_space->evals.resize(param.eig_param.n_conv);
    reserved_space->evecs.resize(0);
    // first ensure that any existing space is freed
    for (int i = param.eig_param.n_conv; i < param.eig_param.n_ev; i++) if (evecs[i]) delete evecs[i];//??
    evecs.resize(param.eig_param.n_conv);

    for (int i = 0; i < param.eig_param.n_conv; i++) {
      reserved_space->evals[i] = Complex(devals[i]);
      evals[i] = Complex(devals[i]);
      blas::copy(*evecs[i], buff->Component(i));
    }

    extractDeflationSpace(reserved_space->evecs);

    blas::zero(*args.V2k);

    return;
  }

  void IncEigCG::Verify()
  {
    EigCGArgs &args = *eigcg_args;

    const int nevs_to_print = param.eig_param.n_conv;
    if (nevs_to_print == 0) errorQuda("\nIncorrect size of current deflation space. \n");

    std::unique_ptr<Complex[]> projm(new Complex[param.eig_param.n_ev * param.eig_param.n_conv]);

    Map<MatrixXcd, Unaligned, DynamicStride> projm_(args.projMat, param.eig_param.n_conv, param.eig_param.n_conv,
                                                    DynamicStride(param.eig_param.n_ev, 1));
    Map<MatrixXcd, Unaligned, DynamicStride> evecs_(projm.get(), param.eig_param.n_conv, param.eig_param.n_conv,
                                                    DynamicStride(param.eig_param.n_ev, 1));

    SelfAdjointEigenSolver<MatrixXcd> es_projm(projm_);
    evecs_.block(0, 0, param.eig_param.n_conv, param.eig_param.n_conv) = es_projm.eigenvectors();

    ColorSpinorParam csParam(*evecs[0]);
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.location = param.eig_param.location;
    csParam.mem_type = QUDA_MEMORY_DEVICE;
    csParam.setPrecision(QUDA_DOUBLE_PRECISION);

    if (csParam.location == QUDA_CUDA_FIELD_LOCATION) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
      if (csParam.nSpin != 1) csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    }

    std::unique_ptr<ColorSpinorField> r(ColorSpinorField::Create(csParam));

    csParam.setPrecision(param.precision_sloppy);
    if (csParam.location == QUDA_CUDA_FIELD_LOCATION && param.precision_sloppy != QUDA_DOUBLE_PRECISION)
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;

    std::unique_ptr<ColorSpinorField> r_sloppy(ColorSpinorField::Create(csParam));
    std::unique_ptr<ColorSpinorField> Av_sloppy(ColorSpinorField::Create(csParam));

    std::vector<ColorSpinorField *> rv_(evecs.begin(), evecs.begin() + param.eig_param.n_conv);
    std::vector<ColorSpinorField *> res_ {r.get()};

    for (int i = 0; i < nevs_to_print; i++) {
      zero(*r);
      blas::caxpy(&projm.get()[i * param.eig_param.n_ev], rv_, res_); // multiblas
      *r_sloppy = *r;

      matDefl(*Av_sloppy, *r_sloppy);
      double3 dotnorm = cDotProductNormA(*r_sloppy, *Av_sloppy);

      double eval = dotnorm.x / dotnorm.z;
      blas::xpay(*Av_sloppy, -eval, *r_sloppy);

      double relerr = sqrt(norm2(*r_sloppy) / dotnorm.z);
      printfQuda("Eigenvalue %d: %1.12e Residual: %1.12e\n", i + 1, eval, relerr);
    }

    return;
  }

/*
 * This is a solo precision solver.
*/
  int IncEigCG::EigCGsolve(ColorSpinorField &x, ColorSpinorField &b)
  {
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

    double local_stop = x.Precision() == QUDA_DOUBLE_PRECISION ? b2*param.tol*param.tol :  b2*1e-11;
    const double local_rel_stop = param.delta;

    EigCGArgs &args = *eigcg_args;

    if (args.run_residual_correction) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      (*K)(x, b);
      return Kparam.iter;
    }

    ColorSpinorField &Ap = (*work_space)[0];
    ColorSpinorField &p = (*work_space)[1];
    ColorSpinorField &r = (*work_space)[2];
    ColorSpinorField &y = (*work_space)[3];
    ColorSpinorField &tmp = (*work_space)[4];
    ColorSpinorField &z = K ? (*work_space)[5] : r;
    ColorSpinorField &tmp2 = !matSloppy.isStaggered() ? (*work_space)[K ? 6 : 5] : tmp;

    // compute initial residual
    //
    matSloppy(r, x, tmp, tmp2);
    double r2 = blas::xmyNorm(b, r);

    if( K ) {//apply preconditioner

      ColorSpinorField &rPre = *r_pre;
      ColorSpinorField &pPre = *p_pre;

      blas::copy(rPre, r);
      commGlobalReductionPush(false);
      (*K)(pPre, rPre);
      commGlobalReductionPop();
      blas::copy(z, pPre);
    }

    p = z;

    blas::zero(y);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    double heavy_quark_res = 0.0;  // heavy quark res idual

    if (use_heavy_quark_res)  heavy_quark_res = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

    double pAp;
    double alpha=1.0, alpha_inv=1.0, beta=0.0, alpha_old_inv = 1.0;

    double lanczos_diag = 0.0, lanczos_offdiag = 0.0;

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    double rMinvr = blas::reDotProduct(r, z);
    //Begin EigCG iterations:
    args.restarts = 0;

    PrintStats("eigCG", k, r2, b2, heavy_quark_res);

    bool converged = convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq);

    while ( (!converged && k < param.maxiter) || (args.restarts < 3) ) {

      matSloppy(Ap, p, tmp, tmp2); // tmp as tmp

      pAp = blas::reDotProduct(p, Ap);
      alpha_old_inv = alpha_inv;
      alpha         = rMinvr / pAp;
      alpha_inv     = 1.0 / alpha;

      LegacySearchSpaceUpdate(lanczos_diag, lanczos_offdiag, beta, sqrt(r2));

      r2 = blas::axpyNorm(-alpha, Ap, r);

      if( K ) {//apply preconditioner
        ColorSpinorField &rPre = *r_pre;

        ColorSpinorField &pPre = *p_pre;

        blas::copy(rPre, r);
        commGlobalReductionPush(false);
        (*K)(pPre, rPre);
        commGlobalReductionPop();
        blas::copy(z, pPre);
      }
      //

      lanczos_diag = (alpha_inv + beta * alpha_old_inv);

      double rMinvr_old   = rMinvr;

      rMinvr = K ? blas::reDotProduct(r, z) : r2;
      beta                = rMinvr / rMinvr_old;

      blas::axpyZpbx(alpha, p, y, z, beta);

      lanczos_offdiag  = (-sqrt(beta)*alpha_inv);

      k++;

      PrintStats("eigCG", k, r2, b2, heavy_quark_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently

      double rel_r2 = r2 / b2;
      converged = (convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq) or convergence(r2, heavy_quark_res, local_stop, param.tol_hq))
                  or convergence(rel_r2, heavy_quark_res, local_rel_stop, param.tol_hq);
    }

    blas::zero(*args.V2k);
    //
    args.CleanArgs();

    blas::xpy(y, x);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);
    double gflops = (blas::flops + matSloppy.flops())*1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residuals
    matSloppy(r, x, tmp, tmp2);
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

  int IncEigCG::CAEigCGsolve(ColorSpinorField &x, ColorSpinorField &b)
  {

    if (param.pipeline < param.eig_param.nLockedMax / 2)
      errorQuda("\nPipeline length is too short (%d).\n", param.pipeline);

    if (((param.eig_param.n_kr - 2 * param.eig_param.nLockedMax) % param.pipeline != 0)
        || ((param.eig_param.n_kr - 2 * param.eig_param.nLockedMax) < param.pipeline))
      errorQuda("Pipeline length %d is not supported.", param.pipeline);

    if (checkLocation(x, b) != QUDA_CUDA_FIELD_LOCATION) errorQuda("Not supported");

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

    EigCGArgs &args = *eigcg_args;

    if (args.run_residual_correction) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      (*K)(x, b);
      return Kparam.iter;
    }

    ColorSpinorField &w = (*work_space)[0];
    ColorSpinorField &u = (*work_space)[1];
    ColorSpinorField &p = (*work_space)[2];
    ColorSpinorField &s = (*work_space)[3];
    ColorSpinorField &r = (*work_space)[4];
    ColorSpinorField &y = (*work_space)[5];
    ColorSpinorField &tmp = (*work_space)[6];
    ColorSpinorField &tmp2 = !matSloppy.isStaggered() ? (*work_space)[7] : tmp;

    // compute initial residual

    matSloppy(r, x, tmp, tmp2);
    blas::xpay(b, -1.0, r); // r2

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    // MPI specific:
    std::unique_ptr<double[]> recvbuff(new double[4]);
    // MPI_Request iallreduce_request_handle;
    MsgHandle *iallreduce_request_handle = nullptr;

    double4 local_buffer;

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_res
      = (use_heavy_quark_res) ? sqrt(blas::HeavyQuarkResidualNorm(x, r).z) : 0.0; // heavy quark res idual

    blas::zero(y);

    matSloppy(w, r, tmp, tmp2);
    p = r;
    matSloppy(s, p);
    double alpha = 0.0;
    double beta = 0.0;
    double beta_old = 0.0;
    double nunew = 0.0;

    double alpha_inv = 1.0;
    double alpha_old_inv = 1.0;

    double lanczos_diag = 0.0, lanczos_offdiag = 0.0;

    double &nu = local_buffer.x;
    double &gamma = local_buffer.y;
    double &delta = local_buffer.z;
    double &mu = local_buffer.w;

    commGlobalReductionPush(false);
    nu = blas::norm2(r);
    gamma = blas::norm2(s);
    delta = blas::reDotProduct(r, s);
    mu = blas::reDotProduct(p, s);
    commGlobalReductionPop();//restore global reduction

    if (comm_size() > 1)
      comm_nonblocking_allreduce_array(iallreduce_request_handle, reinterpret_cast<double *>(&local_buffer),
                                       recvbuff.get(), 4);

    matSloppy(u, s, tmp, tmp2);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    // Begin EigCG iterations (init cycle):
    args.restarts = 0;

    int k = 0;

    const double local_stop = x.Precision() == QUDA_DOUBLE_PRECISION ?
      b2 * param.tol * param.tol :
      (x.Precision() == QUDA_SINGLE_PRECISION ? b2 * 1e-12 : b2 * 5e-10);

    const double local_rel_stop = param.delta;

    double rel_r2 = nu / b2;

    bool converged = convergence(gamma, heavy_quark_res, args.global_stop, param.tol_hq);

    constexpr int prediction_correction_interval = 4;
    int correction_count = 0;

    printfQuda("\nRunning CA eigCG with %d correction interval, local relative tolerance %le, local stop %le, global stop %le.\n", prediction_correction_interval, local_rel_stop, local_stop, args.global_stop);

    if (comm_size() > 1) {
      comm_wait(iallreduce_request_handle);
      memcpy(reinterpret_cast<double *>(&local_buffer), recvbuff.get(), 4 * sizeof(double));
    }

    //blas::createAuxBlasStream();

    PrintStats("CAeigCG", k, nu, b2, heavy_quark_res);

    while ((!converged && k < param.maxiter) || (args.restarts < search_space_restarts[args.logical_rhs_id] )) {
      // Update search space
      PipelinedSearchSpaceUpdate(lanczos_diag, lanczos_offdiag, sqrt(nu));

      alpha_old_inv = alpha_inv;
      beta_old = beta;

      alpha = nu / mu;
      alpha_inv = 1.0 / alpha;

      nunew = nu - 2 * alpha * delta + alpha * alpha * gamma;
      beta = nunew / nu;

      lanczos_diag = (alpha_inv + beta_old * alpha_old_inv);
      lanczos_offdiag = (-sqrt(beta) * alpha_inv);

      blas::axpy(+alpha, p, y);
      commGlobalReductionPush(false);
      local_buffer = quadrupleEigCGUpdate(alpha, beta, r, s, u, w, p);
      commGlobalReductionPop();

      if (comm_size() > 1)
        comm_nonblocking_allreduce_array(iallreduce_request_handle, reinterpret_cast<double *>(&local_buffer),
                                         recvbuff.get(), 4);

      matSloppy(u, s, tmp, tmp2);

      if (k % prediction_correction_interval == 0) {
        matSloppy(w, r, tmp, tmp2);
        correction_count += 1;
      }

      if (comm_size() > 1) {
        comm_wait(iallreduce_request_handle);
        memcpy(reinterpret_cast<double *>(&local_buffer), recvbuff.get(), 4 * sizeof(double));
      }

      rel_r2 = nu / b2;

      k++;

      PrintStats("CAeigCG", k, nu, b2, heavy_quark_res);

      converged = (convergence(nu, heavy_quark_res, args.global_stop, param.tol_hq)
                   or convergence(nu, heavy_quark_res, local_stop, param.tol_hq))
        or convergence(rel_r2, heavy_quark_res, local_rel_stop, param.tol_hq);
    }

    // blas::zero(*args.V2k);
    if (args.is_host_location) {
      for (int i = 0; i < param.eig_param.nLockedMax; i++) blas::copy((*args.Vm)[i], (*args.hVm)[i]);
    }

    //blas::destroyAuxBlasStream();

    args.CleanArgs();
    if (comm_size() > 1) comm_free( iallreduce_request_handle );

    blas::xpy(y, x);

    profile.TPSTOP(QUDA_PROFILE_COMPUTE);
    profile.TPSTART(QUDA_PROFILE_EPILOGUE);

    param.secs = profile.Last(QUDA_PROFILE_COMPUTE);

    printfQuda("Compute time is %1.8le\n", param.secs);

    double gflops = (blas::flops + matSloppy.flops()) * 1e-9;
    param.gflops = gflops;
    param.iter += k;

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    // compute the true residuals

    matSloppy(r, x, tmp, tmp2);
    param.true_res = sqrt(blas::xmyNorm(b, r) / b2);
    param.true_res_hq = sqrt(blas::HeavyQuarkResidualNorm(x, r).z);

    PrintSummary("CAeigCG", k, nu, b2, args.global_stop, param.tol_hq);

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
    fillInitCGSolverParam(Kparam, param);

    const double full_tol = param.tol; // check for 5e-8
    const int max_restart_num = 5; //param.max_restart_num;
    double inc_tol = param.inc_tol;
    Kparam.tol = param.tol_restart; // Kparam.tol_restart;
    Kparam.deflate = false;

    ColorSpinorParam csParam(x);

    csParam.create = QUDA_ZERO_FIELD_CREATE;

    ColorSpinorField *tmpp2 = ColorSpinorField::Create(csParam);//full precision accumulator
    ColorSpinorField &tmp2  = *tmpp2;
    ColorSpinorField *rp2 = ColorSpinorField::Create(csParam); // full precision residual
    ColorSpinorField &r = *rp2;

    csParam.setPrecision(param.precision_ritz);

    ColorSpinorField *xp_proj = ColorSpinorField::Create(csParam);
    ColorSpinorField &xProj = *xp_proj;

    ColorSpinorField *rp_proj2 = ColorSpinorField::Create(csParam);
    ColorSpinorField &rProj = *rp_proj2;

    int restart_idx  = 0;

    // xProj = x;
    rProj = b;
    //launch initCG:
    while ((Kparam.tol >= full_tol) && (restart_idx < max_restart_num)) {
      restart_idx += 1;

      if (restart_idx > 1) warningQuda("\n\nRestart :: id %d (max restarts %d) with local tol %le.\n", restart_idx, max_restart_num,  Kparam.tol);

      Deflate(xProj, rProj);
      x = xProj;

      if (!K)
        K = std::make_shared<CG>(mat, matPrecon, matPrecon, matPrecon, Kparam, profile);
      else
        K.reset(new CG(mat, matPrecon, matPrecon, matPrecon, Kparam, profile));

      (*K)(x, b);

      mat(r, x, tmp2);
      blas::xpay(b, -1.0, r);

      xProj = x;
      rProj = r;

      if(getVerbosity() >= QUDA_VERBOSE) printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", Kparam.iter, Kparam.secs, Kparam.gflops);

      Kparam.tol *= inc_tol;
      inc_tol *= (inc_tol < 1e-1 ? 1e+1 : 1.0);

      if (Kparam.tol <= full_tol && (restart_idx < max_restart_num)) restart_idx = (max_restart_num - 1); // durty hack
      if (restart_idx == (max_restart_num-1))
        Kparam.tol = full_tol; // do the last solve in the next cycle to full tolerance

      warningQuda("Reset tol to %le and full  tol is %le, inc tol %le \n", Kparam.tol, full_tol, inc_tol);

      param.secs   += Kparam.secs;
    }

    if(getVerbosity() >= QUDA_VERBOSE) printfQuda("\ninitCG stat: %i iter / %g secs = %g Gflops. \n", Kparam.iter, Kparam.secs, Kparam.gflops);
    //
    param.secs   += Kparam.secs;
    param.gflops += Kparam.gflops;

    k   += Kparam.iter;

    delete rp2;
    delete tmpp2;
    delete xp_proj;
    delete rp_proj2;

    return k;
  }

  void IncEigCG::operator()(ColorSpinorField &out, ColorSpinorField &in)
  {
    if (param.rhs_idx == 0) max_eigcg_cycles = param.eig_param.max_restarts;

    const bool mixed_prec = (param.precision != param.precision_sloppy);
    const double b2 = norm2(in);

    if (!init) {
      ColorSpinorParam csParam(in);

      csParam.create = QUDA_ZERO_FIELD_CREATE;

      ep = ColorSpinorField::Create(csParam); // full precision accumulator
      rp = ColorSpinorField::Create(csParam); // full precision residual

      csParam.setPrecision(param.precision_sloppy);

      ep_sloppy = (mixed_prec) ? ColorSpinorField::Create(csParam) : ep;
      rp_sloppy = (mixed_prec) ? ColorSpinorField::Create(csParam) : rp;

      csParam.is_composite = true;
      csParam.composite_dim
        = (param.pipeline == 0) ? (K ? 6 : 5) : 7; // needs an extra field for the preconditioned version

      if (!matSloppy.isStaggered()) csParam.composite_dim += 1; // add an extra tmp field for the wilson-like dslash

      // A work space to keep CG fields
      work_space = ColorSpinorField::Create(csParam);

      if (K) {
        csParam.is_composite = false;
        csParam.setPrecision(param.precision_precondition);
        p_pre = ColorSpinorField::Create(csParam);
        r_pre = ColorSpinorField::Create(csParam);
      }

      const bool host_computing = param.pipeline == 0 ? false : host_flag;

      constructDeflationSpace(in, param.eig_param.n_ev);

      if (eigcg_args == nullptr) eigcg_args = new EigCGArgs(in, param, host_computing);

      eigcg_args->global_stop = stopping(param.tol, b2, param.residual_type); // stopping condition of solver
      init = true;
    }

    // If deflation space is complete: use initCG solver
    if (param.eig_param.n_conv == param.eig_param.n_ev) return;

    ColorSpinorField &e = *ep;
    ColorSpinorField &eSloppy = *ep_sloppy;
    ColorSpinorField &r = *rp;
    ColorSpinorField &rSloppy = *rp_sloppy;

    // deflate initial guess ('out'-field):
    mat(r, out, e);
    //
    double r2 = xmyNorm(in, r);

    const double stop = b2 * param.tol * param.tol;
    // start iterative refinement cycles (or just one eigcg call for full (solo) precision solver):
    bool dcg_cycle = false;

    do {
      if (getVerbosity() >= QUDA_VERBOSE) {	    
        printfQuda("Running logical RHS : %d with min search space restarts %d, %d, %d, %d \n", eigcg_args->logical_rhs_id, 
												search_space_restarts[0], 
												search_space_restarts[1], 
												search_space_restarts[2], 
												search_space_restarts[3] );
      }

      blas::zero(e);
      Deflate(e, r);
      //
      eSloppy = e, rSloppy = r;

      int iters = param.pipeline == 0 ? EigCGsolve(eSloppy, rSloppy) : CAEigCGsolve(eSloppy, rSloppy);

      bool update_ritz = !dcg_cycle && (eigcg_args->restarts >= 1) && (param.eig_param.n_conv < param.eig_param.n_ev);

      if (update_ritz) {
        Increment();

        eigcg_args->logical_rhs_id += 1;

        dcg_cycle = (eigcg_args->logical_rhs_id >= default_eigcg_cycles);

      } else { // run DCG instead
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
      param.true_res_hq = sqrt(HeavyQuarkResidualNorm(out, r).z);
      PrintSummary(!dcg_cycle ? "EigCG:" : "DCG (correction cycle):", iters, r2, b2, stop, param.tol_hq);

      if (getVerbosity() >= QUDA_VERBOSE) { // disable intermediate checks
        if (!dcg_cycle && (eigcg_args->restarts >= 1) && (param.eig_param.n_conv < param.eig_param.n_ev)) Verify();
      }

    } while ((r2 > stop && eigcg_args->logical_rhs_id < default_eigcg_cycles) /*&& mixed_prec*/);

    // adjust search space restart number
    if (param.rhs_idx == 0) init_k = param.iter;
    else {
      if(init_k / param.iter  >= 2) {
        search_space_restarts = {3, 3, 3, 2};
      } else if (init_k / param.iter >= 3) {
        search_space_restarts = {2, 2, 2, 1};
      }
    }

    // we need to update rhs index and number of computed eigenvectors
    param.rhs_idx += 1;

    if (eigcg_args->logical_rhs_id == 0) {
      warningQuda("Cannot expand the deflation space.\n");
      param.eig_param.n_ev = param.eig_param.n_conv;
    }

    eigcg_args->logical_rhs_id = 0;

    if (param.eig_param.n_conv == param.eig_param.n_ev || param.eig_param.is_last_rhs) {
      if (getVerbosity() == QUDA_DEBUG_VERBOSE) {
        blas::zero(out);
        initCGsolve(out, in);
      }
      printfQuda("\nRequested to reserve %d eigenvectors (requested %d) with max tol %le.\n", param.eig_param.n_conv, param.eig_param.n_ev,  param.eig_param.tol);
      Reduce(param.eig_param.tol, param.eig_param.n_conv);
      param.eig_param.is_complete = QUDA_BOOLEAN_TRUE;

      destroyDeflationSpace();

      if(param.eig_param.n_conv == 0) param.eig_param.preserve_deflation = QUDA_BOOLEAN_FALSE; // No eigenvectors found!


      delete eigcg_args;
      eigcg_args = nullptr;
    }

    delete ep;
    delete rp;

    delete work_space;

    if(mixed_prec){
      delete ep_sloppy;
      delete rp_sloppy;
    }

    if(K) {
      delete p_pre;
      delete r_pre;
    }

    return;
  }

} // namespace quda
