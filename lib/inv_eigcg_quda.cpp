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
#include <deflation.h>

#include <Eigen/Dense>

#include <mpi.h>

#define EIGCG_MPI_CHECK_(mpi_call) do {                   		\
  int status = comm_size() == 1 ? MPI_SUCCESS : mpi_call;	\
  if (status != MPI_SUCCESS) {                      		\
    char err_string[128];                           		\
    int err_len;                                    		\
    MPI_Error_string(status, err_string, &err_len); 		\
    err_string[127] = '\0';                         		\
    errorQuda("(MPI) %s", err_string);              		\
  }                                                 		\
} while (0)


/*
Based on  eigCG(nev, m) algorithm:
A. Stathopolous and K. Orginos, arXiv:0707.0131
*/

namespace quda {

   using namespace blas;
   using namespace Eigen;

   using DynamicStride   = Stride<Dynamic, Dynamic>;
   using RealDenseMatrix     = MatrixXd;
   using RealVectorSet       = MatrixXd;
   using RealVector          = VectorXd;

//special types needed for compatibility with QUDA blas:
   using RowMajorRealDenseMatrix = Matrix<double, Dynamic, Dynamic, RowMajor>;

   static int max_eigcg_cycles = 4;//how many eigcg cycles do we allow?

   class EigCGArgs{

     public:
       //host Lanczos matrice, and its eigenvalue/vector arrays:
       RealDenseMatrix Tm;//VH A V,
       //eigenvectors:
       RealVectorSet eigenVecs;//array of  (m)  ritz and of m length
       //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)
       RealVector Tmvals;//eigenvalues of T[m,  m  ] and T[m-1, m-1] (re-used)

       int m;
       int k;
       int id;//cuurent search spase index

       int restarts;
       double global_stop;

       bool run_residual_correction;//used in mixed precision cycles
       //Cached invert residual norm:
       double inv_normr_m;
       //Array of (local) dot products
       RealVector s;

       /** Pipeline specific parameters:
	*/
       Vector3i pipel_per_task;//pipeline length per compute task (for 3 tasks)
       int      tot_pipel;     //total pipeline length
       //
       RealVector pipelanczos_diag;
       RealVector pipelanczos_offdiag;
       // C++ task-based object for the (asynchron.) RayleighRitz computations
       std::future<void> rr_task;
       /** Global reduction stuff:
	 @rr_task_recvbuff recieve buffer for local dot product obtained in the RR block
         @caeigcg_recvbuff recieve buffer for the eigcg internal local dot products
       */
       RealVector rr_task_recvbuff;
       //
       MPI_Request rr_task_request_handle;
       //
       RealVector caeigcg_recvbuff;
       //
       MPI_Request caeigcg_request_handle;

       EigCGArgs(int m, int k, int pipe_l[3]) :
	  Tm(RealDenseMatrix::Zero(m,m)),
	  eigenVecs(RealVectorSet::Zero(m,m)),
	  Tmvals(m),
	  m(m),
	  k(k),
	  id(0),
	  restarts(0),
	  global_stop(0.0),
	  run_residual_correction(false),
	  inv_normr_m(0.0),
	  s(RealVector::Zero(2*k)),
	  pipel_per_task(pipe_l),
	  tot_pipel(pipel_per_task.sum()),
	  pipelanczos_diag(RealVector::Zero(tot_pipel == 0 ? 0 : tot_pipel)),
	  pipelanczos_offdiag(RealVector::Zero(tot_pipel == 0 ? 0 : tot_pipel)),
	  rr_task_recvbuff(RealVector::Zero(tot_pipel == 0 ? 0 : 2*k)),
	  caeigcg_recvbuff(RealVector::Zero(tot_pipel == 0 ? 0 : 4))
       { }

       ~EigCGArgs() { }

       template <bool is_pipelined = false>
       inline void SetLanczos(double diag_val, double offdiag_val) {
         if(run_residual_correction || id == 0) return;
	 const int cid = id - 1;

	 if(is_pipelined) {
           if(cid >= m) {//store lanczos coeff in the buffers
             pipelanczos_diag[cid-m]    = diag_val;
	     pipelanczos_offdiag[cid-m] = offdiag_val;
	     return;
	   }
	 }
	 //Load Lanczos off-diagonals:
	 if (cid < (m-1)) { Tm.diagonal<+1>()[cid] = offdiag_val; Tm.diagonal<-1>()[cid] = offdiag_val;}
	 if (cid < m)       Tm.diagonal< 0>()[cid] = diag_val;

	 return;
       }

       inline void CacheNormR(const double &normr) {inv_normr_m = 1.0/normr;}

       inline void CleanArgs() {
	 id = 0; Tm.setZero(); Tmvals.setZero(); eigenVecs.setZero();
	 if(rr_task.valid()) rr_task.wait();
       }

       inline void ResetSearchIdx() {  id = 2*k+tot_pipel;  restarts += 1; }
       inline void UpdateSearchIdx(){  id += 1; }

       template <bool is_pipelined = false>
       void RestartArgs(){
         Tm.setZero();
	 for(int i = 0; i < 2*k; i++) Tm(i,i) = Tmvals(i);//??

	 s *= inv_normr_m;

	 Tm.col(2*k).segment(0, 2*k) = s;
	 Tm.row(2*k).segment(0, 2*k) = s;

	 id = 2*k+tot_pipel;  restarts += 1; s.setZero();

	 if (!is_pipelined) return;

	 rr_task_recvbuff.setZero();

         // For the pipelined version we need to restore cached coefficinets:
	 for(int i = 0; i < tot_pipel; i++) {
	   Tm.diagonal< 0>()[2*k+i] = pipelanczos_diag[i];
	   Tm.diagonal<+1>()[2*k+i] = pipelanczos_offdiag[i];
	   Tm.diagonal<-1>()[2*k+i] = pipelanczos_offdiag[i];
	 }

	 return;
       }


       void ComputeEv() {
         //Solve m dim eigenproblem:
	 SelfAdjointEigenSolver<MatrixXd> es_tm(Tm);

	 eigenVecs.leftCols(k) = es_tm.eigenvectors().leftCols(k);

	 //Solve m-1 dim eigenproblem:
	 SelfAdjointEigenSolver<MatrixXd> es_tm1(Map<MatrixXd, Unaligned, DynamicStride >(Tm.data(), (m-1), (m-1), DynamicStride(m, 1)));

	 Block<MatrixXd>(eigenVecs.derived(), 0, k, m-1, k) = es_tm1.eigenvectors().leftCols(k);

	 eigenVecs.block(m-1, k, 1, k).setZero();

	 MatrixXd Q2k(MatrixXd::Identity(m, 2*k));

	 HouseholderQR<MatrixXd> eigenVecs2k_qr( Map<MatrixXd, Unaligned >(eigenVecs.data(), m, 2*k) );

 	 Q2k.applyOnTheLeft( eigenVecs2k_qr.householderQ() );

	 //2. Construct H = QH*Tm*Q :
	 RealDenseMatrix H2k2 = Q2k.transpose()*Tm*Q2k;

	 /* solve the small evecm1 2nev x 2nev eigenproblem */
	 SelfAdjointEigenSolver<MatrixXd> es_h2k(H2k2);

	 Block<MatrixXd>(eigenVecs.derived(), 0, 0, m, 2*k) = Q2k * es_h2k.eigenvectors();

	 Tmvals.segment(0,2*k) = es_h2k.eigenvalues();

	 return;
       }

   };

   //helper for a smart pointer creation
   
   std::shared_ptr<ColorSpinorField> MakeSharedPtr(const ColorSpinorParam &param)
   {
     if (param.location == QUDA_CPU_FIELD_LOCATION )  return std::move(std::make_shared<cpuColorSpinorField>(param) );
     else					      return std::move(std::make_shared<cudaColorSpinorField>(param));
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
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(nullptr), Kparam(param), Vm(nullptr), V2k(nullptr),
    work_space(nullptr), Az(nullptr), r_pre(nullptr), p_pre(nullptr), eigcg_args(nullptr), profile(profile), init(false)
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
      K = std::make_shared<CG>(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_MR_INVERTER){
      K = std::make_shared<MR>(matPrecon, matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition == QUDA_SD_INVERTER){
      K = std::make_shared<SD>(matPrecon, Kparam, profile);
    }else if(param.inv_type_precondition != QUDA_INVALID_INVERTER){ // unknown preconditioner
      errorQuda("Unknown inner solver %d", param.inv_type_precondition);
    }
    return;
  }

  IncEigCG::~IncEigCG() { }

  template <bool is_pipelined, CAEigCGComputeTasks compute_task_id>
  void IncEigCG::RayleighRitz()
  {
    EigCGArgs &args = *eigcg_args;
    if ( compute_task_id & COMPUTE_EIGENV ) {
      args.ComputeEv();
      //
      if (compute_task_id == COMPUTE_EIGENV) return;
    } // end compute task


    if( compute_task_id & COMPUTE_QV ) {
      constexpr int compute_qv_idx = COMPUTE_QV >> 1; //1
      //Restart V task:
      ColorSpinorFieldSet &vm  = *Vm;
      ColorSpinorFieldSet &v2k = *V2k;

      std::vector<ColorSpinorField*> v2k_(v2k());

      blas::zero(v2k);

      const int n_qv_updates = args.pipel_per_task[compute_qv_idx];

      const int block_size   = args.m / n_qv_updates;//WARNING args.n_updates < args.m

      for(int i = 0; i < n_qv_updates; i++) {

	const int current_block_size = (!is_pipelined || i < (n_qv_updates -1)) ? block_size : args.m - i*block_size;

	const int offset             = i*block_size;

        std::vector<ColorSpinorField*> vm_(vm(offset, offset + current_block_size));

        RowMajorRealDenseMatrix Alpha(args.eigenVecs.block(i*block_size,0,current_block_size, 2*args.k));//args.m->work_sz

        blas::axpy( Alpha.data(), vm_ , v2k_);
      }

      for(int i = 0; i < 2*args.k; i++)  blas::copy(vm[i], v2k[i]);
    }

    if( compute_task_id & COMPUTE_ZAV ) {
      ColorSpinorFieldSet &v2k = *V2k;

      std::vector<ColorSpinorField*> v2k_(v2k());
      std::vector<ColorSpinorField*> Az_;

      Az_.push_back(Az.get());

      commGlobalReductionSet(!is_pipelined); //disable when is_pipelined is true
      blas::reDotProduct(args.s.data(), Az_, v2k_);
      commGlobalReductionSet(true);

      //Start async communications:
      if( is_pipelined ) {
        EIGCG_MPI_CHECK_(MPI_Iallreduce(reinterpret_cast<double*>(args.s.data()),
				        reinterpret_cast<double*>(args.rr_task_recvbuff.data()),
					2*args.k,
					MPI_DOUBLE,
					MPI_SUM, MPI_COMM_WORLD,
					&args.rr_task_request_handle));
      }
    }

    return;
  }

  void IncEigCG::SearchSpaceUpdate(ColorSpinorField &z,  const double &lanczos_diag, const double &lanczos_offdiag, const double &beta, const double &normr)
  {
    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) return;

    ColorSpinorFieldSet &vm  = *Vm;
    ColorSpinorField    &Ap  = (*work_space)[0];

    args.SetLanczos(lanczos_diag, lanczos_offdiag);

    if (args.id == (args.m-1)) blas::copy(*Az, Ap);

    // launch RR block for args.id == param.m, otherwise is nop
    if(args.id == args.m) {
      args.CacheNormR(normr);
      //Compute (update) Ap = Ap - beta*Ap_old:
      blas::xpay(Ap, -beta, *Az);
      RayleighRitz();
      args.RestartArgs();
    }

    //load Lanczos basis vector:
    blas::copy(vm[args.id], z);//convert arrays
    //rescale the vector
    blas::ax(1.0 / normr, vm[args.id]);

    args.UpdateSearchIdx();

    return;
  }

  void IncEigCG::PipelinedSearchSpaceUpdate(const double &lanczos_diag, const double &lanczos_offdiag, const double &beta, const double &normr)
  {
    constexpr bool is_pipelined	= true;

    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) return;

    ColorSpinorFieldSet &vm  = *Vm;

    args.SetLanczos<is_pipelined>(lanczos_diag, lanczos_offdiag);

    // launch RR block for args.id == param.m, otherwise is nop
    if(args.id == args.m) {
      args.CacheNormR(normr);
      //Store updated Ap
      blas::copy(*Az, (*work_space)[0]);
      //args.rr_task = std::async(std::launch::async, &IncEigCG::RayleighRitz<is_pipelined, COMPUTE_EIGENV_AND_QV>, this);
      //ok, this works but does not launch a parallel task
      args.rr_task = std::async(std::launch::deferred, &IncEigCG::RayleighRitz<is_pipelined, COMPUTE_EIGENV_AND_QV>, this);
    } else if (args.id == (args.m+args.tot_pipel-1)) {
      //do synchronization first
      if(args.rr_task.valid()) args.rr_task.wait();
      else warningQuda("Tried to synchronize an invalid task...");

      RayleighRitz<is_pipelined, COMPUTE_ZAV>();
    } else if (args.id == (args.m+args.tot_pipel)) {
      // load cached basis vectors (nop for the non-pipelined version)
      for(int i = 0; i < args.tot_pipel; i++) blas::copy(vm[2*args.k + i], vm[args.m + i]);
      //
      EIGCG_MPI_CHECK_(MPI_Wait(&args.rr_task_request_handle, MPI_STATUS_IGNORE));
      if (comm_size() > 1) args.s = args.rr_task_recvbuff;
      //
      args.RestartArgs<is_pipelined>();
    }

    //load Lanczos basis vector:
    blas::copy(vm[args.id], *rp);//convert arrays
    //rescale the vector
    blas::ax(1.0 / normr, vm[args.id]);

    args.UpdateSearchIdx();

    return;
  }


/*
 * This is a solo precision solver.
*/
  int IncEigCG::EigCGsolve(ColorSpinorField &x, ColorSpinorField &b) {
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

    int default_pipel[3] = {0,0,0};

    if (!init) {
      eigcg_args = std::make_shared <EigCGArgs> (param.m, param.nev, default_pipel);//need only deflation meta

      csParam.create = QUDA_ZERO_FIELD_CREATE;
      rp   = MakeSharedPtr(csParam);
      yp   = MakeSharedPtr(csParam);
      Az   = MakeSharedPtr(csParam);
      tmpp = MakeSharedPtr(csParam);

      csParam.is_composite  = true;
      csParam.composite_dim = 2;
      // An auxiliary work space:
      work_space = MakeSharedPtr(csParam);
      //Create a search vector set:
      csParam.setPrecision(param.precision_ritz);
      csParam.composite_dim = param.m;
      Vm         = MakeSharedPtr(csParam);

      csParam.setPrecision(QUDA_DOUBLE_PRECISION);
      csParam.composite_dim = (2*param.nev);
      V2k        = MakeSharedPtr(csParam);

      if (K && param.precision_precondition != param.precision_sloppy) {
        csParam.is_composite  = false;
	csParam.setPrecision(param.precision_precondition);
	p_pre = MakeSharedPtr(csParam);
	r_pre = MakeSharedPtr(csParam);
      }

      eigcg_args->global_stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver
      init = true;
    }

    double local_stop = x.Precision() == QUDA_DOUBLE_PRECISION ? b2*param.tol*param.tol :  b2*1e-11;

    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      (*K)(x, b);
      return Kparam.iter;
    }

    csParam.setPrecision(param.precision_sloppy);
    csParam.is_composite  = false;

    std::shared_ptr<ColorSpinorField> zp  = (K != nullptr) ? MakeSharedPtr(csParam) : rp;

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &z = *zp;
    ColorSpinorField &tmp = *tmpp;
				ColorSpinorField &Ap= (*work_space)[0];
    ColorSpinorField &p = (*work_space)[1];

    // compute initial residual
    //
    matSloppy(r, x, y);
    double r2 = blas::xmyNorm(b, r);

    if( K ) {//apply preconditioner
      if (param.precision_precondition == param.precision_sloppy) { r_pre = rp; p_pre = zp; }
      ColorSpinorField &rPre = *r_pre;
      ColorSpinorField &pPre = *p_pre;

      blas::copy(rPre, r);
      commGlobalReductionSet(false);
      (*K)(pPre, rPre);
      commGlobalReductionSet(true);
      blas::copy(z, pPre);
    }

    p = z;

    blas::zero(y);

    const bool use_heavy_quark_res =
				(param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

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

    double rMinvr = blas::reDotProduct(r,z);
    //Begin EigCG iterations:
    args.restarts = 0;

    PrintStats("eigCG", k, r2, b2, heavy_quark_res);

    bool converged = convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq);

    while ( !converged && k < param.maxiter ) {

      matSloppy(Ap, p, tmp);  // tmp as tmp

      pAp           = blas::reDotProduct(p, Ap);
      alpha_old_inv = alpha_inv;
      alpha         = rMinvr / pAp;
      alpha_inv     = 1.0 / alpha;

      SearchSpaceUpdate(z, lanczos_diag, lanczos_offdiag, beta, sqrt(r2));

      r2 = blas::axpyNorm(-alpha, Ap, r);

      if( K ) {//apply preconditioner
        ColorSpinorField &rPre = *r_pre;

	ColorSpinorField &pPre = *p_pre;

	blas::copy(rPre, r);

	commGlobalReductionSet(false);
	(*K)(pPre, rPre);
	commGlobalReductionSet(true);

	blas::copy(z, pPre);
      }
      //

      lanczos_diag  = (alpha_inv + beta*alpha_old_inv);

      double rMinvr_old   = rMinvr;

      rMinvr = K ? blas::reDotProduct(r,z) : r2;
      beta                = rMinvr / rMinvr_old;

      blas::axpyZpbx(alpha, p, y, z, beta);

      lanczos_offdiag  = (-sqrt(beta)*alpha_inv);

      k++;

      PrintStats("eigCG", k, r2, b2, heavy_quark_res);
      // check convergence, if convergence is satisfied we only need to check that we had a reliable update for the heavy quarks recently

      converged = convergence(r2, heavy_quark_res, args.global_stop, param.tol_hq) or convergence(r2, heavy_quark_res, local_stop, param.tol_hq);
    }

    blas::zero(*V2k);
    //
    args.CleanArgs();

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

  constexpr int qv_pipe_l  = 8;
  constexpr int zav_pipe_l = 8;

  int IncEigCG::CAEigCGsolve(ColorSpinorField &x, ColorSpinorField &b) {

    const int leftover_iters = 4;
    const int rr_pipe_l      = std::max(std::min((param.m-2*param.nev-(qv_pipe_l+zav_pipe_l)-leftover_iters), param.pipeline), param.nev);
    const int tot_pipe_l     = rr_pipe_l+qv_pipe_l+zav_pipe_l;

    if(tot_pipe_l < param.nev) errorQuda("Pipeline length is too short (%d).", tot_pipe_l);

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

    int caeigcg_pipe_l[3] = {rr_pipe_l, qv_pipe_l, zav_pipe_l};

    if (!init) {
      eigcg_args = std::make_shared <EigCGArgs> (param.m, param.nev, caeigcg_pipe_l);

      csParam.create = QUDA_ZERO_FIELD_CREATE;
      rp = MakeSharedPtr(csParam);
      yp = MakeSharedPtr(csParam);
      Az = MakeSharedPtr(csParam);

      tmpp = MakeSharedPtr(csParam);

      csParam.is_composite  = true;
      csParam.composite_dim = 4;

      // auxiliary work space to keep w, p, s, u fields
      work_space = MakeSharedPtr(csParam);

      //Create a search vector set:
      csParam.composite_dim = param.m+tot_pipe_l;
      csParam.setPrecision(param.precision_ritz);//eigCG internal search space precision may not coincide with the solver precision!

      Vm = MakeSharedPtr(csParam);

      csParam.setPrecision(QUDA_DOUBLE_PRECISION);
      csParam.composite_dim = (2*param.nev);

      V2k = MakeSharedPtr(csParam);


      eigcg_args->global_stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver

      init = true;
    }

    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) {
      profile.TPSTOP(QUDA_PROFILE_INIT);
      (*K)(x, b);
      return Kparam.iter;
    }

    ColorSpinorField &r = *rp;
    ColorSpinorField &y = *yp;
    ColorSpinorField &tmp = *tmpp;
    // compute initial residual

    matSloppy(r, x, y);
    blas::xpay(b, -1.0, r); //r2

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    double heavy_quark_res = (use_heavy_quark_res) ? sqrt(blas::HeavyQuarkResidualNorm(x, r).z) : 0.0;  // heavy quark res idual

    ColorSpinorField &w = (*work_space)[0];
    ColorSpinorField &p = (*work_space)[1];
    ColorSpinorField &s = (*work_space)[2];
    ColorSpinorField &u = (*work_space)[3];

    blas::zero(y);

    matSloppy(w, r);
    p = r;
    matSloppy(s, p);
    double alpha    = 0.0;
    double beta     = 0.0;
    double beta_old = 0.0;
    double nunew    = 0.0;

    double alpha_inv     = 1.0;
    double alpha_old_inv = 1.0;

    double lanczos_diag  = 0.0, lanczos_offdiag = 0.0;

    double4 local_buffer;

    double &nu    = local_buffer.x;
    double &gamma = local_buffer.y;
    double &delta = local_buffer.z;
    double &mu    = local_buffer.w;

    nu     = blas::norm2(r);
    gamma  = blas::norm2(s);
    delta  = blas::reDotProduct(r, s);
    mu     = blas::reDotProduct(p, s);

    EIGCG_MPI_CHECK_(MPI_Iallreduce(reinterpret_cast<double*>(&local_buffer),
			            args.caeigcg_recvbuff.data(),
				    4,
				    MPI_DOUBLE,
			    	    MPI_SUM, MPI_COMM_WORLD,
				    &args.caeigcg_request_handle));


    matSloppy(u, s);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    //Begin EigCG iterations (init cycle):
    args.restarts = 0;

    int k = 0;

    PrintStats("pipeEigCG", k, nu, b2, heavy_quark_res);

    const double local_stop = x.Precision() == QUDA_DOUBLE_PRECISION ? b2*param.tol*param.tol :  b2*1e-8;
    bool converged = convergence(gamma, heavy_quark_res, args.global_stop, param.tol_hq);

    constexpr int prediction_correction_interval = 16;
    int correction_count;

    while ( !converged && k < param.maxiter ) {
      //Update search space
      PipelinedSearchSpaceUpdate(lanczos_diag, lanczos_offdiag, beta, sqrt(nu));

      //EIGCG_MPI_CHECK_(MPI_Wait(&args.pipecg_task_request_handle, MPI_STATUS_IGNORE));
      //memcpy(local_buffer.get(), args.pipecg_task_recvbuff.data(), 4*sizeof(double));

      alpha_old_inv = alpha_inv;
      beta_old      = beta;

      alpha         = nu / mu;
      alpha_inv     = 1.0 / alpha;

      nunew = nu - 2*alpha*delta + alpha*alpha*gamma;
      beta  = nunew / nu;

      lanczos_diag     = (alpha_inv + beta_old*alpha_old_inv);
      lanczos_offdiag  = (-sqrt(beta)*alpha_inv);

      blas::axpy(+alpha, p, y);
      local_buffer = quadrupleEigCGUpdate(alpha, beta, r, s, u, w, p);

      matSloppy(u, s, tmp);

      if (k % prediction_correction_interval == 0){
        matSloppy(w, r, tmp);
	correction_count += 1;
      }

      k++;

      PrintStats("pipeEigCG", k, nu, b2, heavy_quark_res);
      converged = convergence(gamma, heavy_quark_res, args.global_stop, param.tol_hq) or convergence(gamma, heavy_quark_res, local_stop, param.tol_hq);
    }

    blas::zero(*V2k);

    args.CleanArgs();

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

    PrintSummary("eigCG", k, gamma, b2, args.global_stop, param.tol_hq);

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
     if(param.rhs_idx == 0) max_eigcg_cycles = param.eigcg_max_restarts;

     const bool mixed_prec = (param.precision != param.precision_sloppy);
     const double b2       = norm2(in);

     deflated_solver *defl_p = static_cast<deflated_solver*>(param.deflation_op);
     Deflation &defl         = *(defl_p->defl);

     //If deflation space is complete: use initCG solver
     if( defl.is_complete() ) return;

     //Start (incremental) eigCG solver:
     ColorSpinorParam csParam(in);
     csParam.create = QUDA_ZERO_FIELD_CREATE;

     std::shared_ptr<ColorSpinorField> ep = MakeSharedPtr(csParam);//full precision accumulator
     std::shared_ptr<ColorSpinorField> rp = MakeSharedPtr(csParam);//full precision residual

     ColorSpinorField &e = *ep;
     ColorSpinorField &r = *rp;

     //deflate initial guess ('out'-field):
     mat(r, out, e);
     //
     double r2 = xmyNorm(in, r);

     csParam.setPrecision(param.precision_sloppy);

     std::shared_ptr<ColorSpinorField> ep_sloppy = ( mixed_prec ) ? MakeSharedPtr(csParam) : ep;
     std::shared_ptr<ColorSpinorField> rp_sloppy = ( mixed_prec ) ? MakeSharedPtr(csParam) : rp;

     ColorSpinorField &eSloppy = *ep_sloppy;
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
           K.reset( new CG(matSloppy, matPrecon, Kparam, profile) );
         }

         eigcg_args->run_residual_correction = true;
         printfQuda("Running DCG correction cycle.\n");
       }

       int iters = param.pipeline == 0 ? EigCGsolve(eSloppy, rSloppy) : CAEigCGsolve(eSloppy, rSloppy);

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

     if (mixed_prec && max_eigcg_cycles > logical_rhs_id) {
       printfQuda("Reset maximum eigcg cycles to %d (was %d)\n", logical_rhs_id, max_eigcg_cycles);
       max_eigcg_cycles = logical_rhs_id;//adjust maximum allowed cycles based on the actual information
     }

     param.rhs_idx += logical_rhs_id;

     if(logical_rhs_id == 0) {
       warningQuda("Cannot expand the deflation space.\n");
       defl.reset_deflation_space();
       param.rhs_idx += 1; //we still solved the system
     }

     if(defl.is_complete()) {
       if(param.rhs_idx != param.deflation_grid) warningQuda("\nTotal rhs number (%d) does not match the requested deflation grid size (%d).\n", param.rhs_idx, param.deflation_grid);

       const int max_nev = defl.size();//param.m;
       printfQuda("\nRequested to reserve %d eigenvectors with max tol %le.\n", max_nev, param.eigenval_tol);
       defl.reduce(param.eigenval_tol, max_nev);
     }
     return;
  }

} // namespace quda
