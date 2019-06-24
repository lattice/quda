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

extern MPI_Comm MPI_COMM_HANDLE;

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

   //helper for a smart pointer creation

   std::shared_ptr<ColorSpinorField> MakeSharedPtr(const ColorSpinorParam &param)
   {
     if (param.location == QUDA_CPU_FIELD_LOCATION )  return std::move(std::make_shared<cpuColorSpinorField>(param) );
     else					      return std::move(std::make_shared<cudaColorSpinorField>(param));
   }

   using EigCGComputeTasks = enum EigCGComputeTasks_s { COMPUTE_NOTHING   = 0,
	                                                COMPUTE_QV        = 1,
							COMPUTE_ZAV       = 2,
							COMPUTE_ALL       = 3
   };

   // this is the Worker pointer that the dslash uses to launch the shifted updates
   namespace dslash {
     extern Worker* aux_worker;
   }

   class EigCGArgs : public Worker {

     public:
       //host Lanczos matrice, and its eigenvalue/vector arrays:
       RealDenseMatrix Tm;//VH A V,
       //eigenvectors:
       RealVectorSet eigenVecs;//array of  (m)  ritz and of m length
       //eigenvalues of both T[m,  m  ] and T[m-1, m-1] (re-used)
       RealVector Tmvals;//eigenvalues of T[m,  m  ] and T[m-1, m-1] (re-used)

       const int m;
       const int k;
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

       const int pipe_l; // total pipeline length

       const int ev_task_offset;//pipeline length for the eigenvec computations

       const int n_update;

       //Cached lanczos elements (diagonal and off-diagonal elements)
       RealVectorSet lanczos_cached_elems;

       // C++ task-based object for the (asynchron.) RayleighRitz computations
       std::future<void> rr_task;
       /** Global reduction stuff:
	   @rr_task_recvbuff recieve buffer for local dot product obtained in the RR block
           @caeigcg_recvbuff recieve buffer for the eigcg internal local dot products
       */
       RealVector rr_task_recvbuff;
       //
       MPI_Request rr_task_request_handle;

       std::shared_ptr<ColorSpinorField> Az;       // mat * conjugate vector from the previous iteration

       std::shared_ptr<ColorSpinorFieldSet> Vm;          //eigCG search vectors  (spinor matrix of size eigen_vector_length x m)

       std::shared_ptr<ColorSpinorFieldSet> V2k;         //temp vector set

       EigCGComputeTasks ctask_id;

       EigCGArgs(const int m, const int k, const int pipe_l, const int ev_task_offset, const int n_update, const ColorSpinorField &meta, QudaPrecision search_space_prec ) :
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
	  pipe_l(pipe_l),
	  ev_task_offset(ev_task_offset),
	  n_update(n_update),
	  lanczos_cached_elems(RealVectorSet::Zero(pipe_l, 2)),
	  rr_task_recvbuff(RealVector::Zero(2*k)),
	  Az(nullptr),
	  Vm(nullptr),
	  V2k(nullptr),
	  ctask_id(EigCGComputeTasks::COMPUTE_QV)
	  {
	    ColorSpinorParam csParam(meta);
	    csParam.create        = QUDA_ZERO_FIELD_CREATE;
	    //
	    Az = MakeSharedPtr(csParam);

	    csParam.is_composite  = true;
	    csParam.composite_dim = m+pipe_l;
	    csParam.setPrecision(search_space_prec);//eigCG internal search space precision may not coincide with the solver precision!

	    //Create a search vector set:
	    Vm = MakeSharedPtr(csParam);
	    csParam.setPrecision(QUDA_DOUBLE_PRECISION);
	    csParam.composite_dim = (2*k);
	    //Create a search vector set:
	    V2k = MakeSharedPtr(csParam);
	  }

       virtual ~EigCGArgs() { }

       template <bool is_pipelined = false>
       inline void UpdateLanczosMatrix(double diag_val, double offdiag_val) {
         if(run_residual_correction || id == 0) return;
	 const int cid = id - 1;

	 if(is_pipelined) {
           if(cid >= m) {//store lanczos coeff in the buffers
             lanczos_cached_elems(cid-m,0)    = diag_val;
	     lanczos_cached_elems(cid-m,1) = offdiag_val;
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

       void RestartArgs(){
         Tm.setZero();
	 for(int i = 0; i < 2*k; i++) Tm(i,i) = Tmvals(i);

         if(pipe_l != 0) { //need to finish global reduction and restore search space for the pipelined version

           //for(int i = 0; i < pipe_l; i++) blas::copy(vm[2*args.k + i], vm[args.m + i]);
	   ColorSpinorFieldSet &vm = *Vm;

           for(int i = 0; i < pipe_l; i++) {
             blas::copy(vm[2*k + i], vm[m + i]);
             Tm.diagonal< 0>()[2*k+i] = lanczos_cached_elems(i,0);
             Tm.diagonal<+1>()[2*k+i] = lanczos_cached_elems(i,1);
             Tm.diagonal<-1>()[2*k+i] = lanczos_cached_elems(i,1);
           }

	   if (comm_size() > 1) {
             EIGCG_MPI_CHECK_(MPI_Wait(&rr_task_request_handle, MPI_STATUS_IGNORE));

             s = rr_task_recvbuff;//?
	     rr_task_recvbuff.setZero();
	   }

	 }

	 s *= inv_normr_m;

	 Tm.col(2*k).segment(0, 2*k) = s;
	 Tm.row(2*k).segment(0, 2*k) = s;

	 id = 2*k+pipe_l;  restarts += 1; s.setZero();

       	 // Reset compute tasks:

	 ctask_id = EigCGComputeTasks::COMPUTE_QV;

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

       inline void UpdateLanczosBasisAndSearchIdx(const ColorSpinorField &r, const double rnorm){
	  ColorSpinorFieldSet &vm = *Vm;
	  //load Lanczos basis vector:

	  blas::copy(vm[id], r);//convert arrays
	  //rescale the vector

	  blas::ax(1.0 / rnorm, vm[id]);
          //
          id += 1;

	  return;
       }


       // Using worker framework:

       void apply(const cudaStream_t &stream) {

	 ColorSpinorFieldSet &vm  = *Vm;
  	 ColorSpinorFieldSet &v2k = *V2k;

     	 static int count = 0;

	 const int tot_n_updates         =  pipe_l ? n_update * (pipe_l - ev_task_offset) : 2;
		//Think about this:
		const int n_zav_subtask_updates =  tot_n_updates > 2 ? 2 : 1;
	 const int n_qv_subtask_updates  =  tot_n_updates - n_zav_subtask_updates;


     	 if (ctask_id == EigCGComputeTasks::COMPUTE_QV)
	 {
	   std::vector<ColorSpinorField*> v2k_(v2k());

	   if(count == 0) blas::zero(v2k);

	   const int bsize   = m / n_qv_subtask_updates;

       	   const int wsize = (count < (n_qv_subtask_updates -1)) ? bsize : m - count*bsize;

       	   const int offset    = count*bsize;

       	   std::vector<ColorSpinorField*> vm_(vm(offset, offset + wsize));

       	   RowMajorRealDenseMatrix Alpha(eigenVecs.block(count*bsize,0,wsize, 2*k));//args.m->work_sz

       	   blas::axpy( Alpha.data(), vm_ , v2k_);

     	   if(++count == n_qv_subtask_updates) {
	     count = 0;
	     ctask_id = EigCGComputeTasks::COMPUTE_ZAV;
	   }
     	 } else if(ctask_id == EigCGComputeTasks::COMPUTE_ZAV) {
	   std::vector<ColorSpinorField*> Az_;

	   Az_.push_back(Az.get());

	   const int bsize  = std::max(2*k / n_zav_subtask_updates, 1);//WARNING args.n_updates < args.m

	   const int wsize  = (count < (n_zav_subtask_updates -1)) ? bsize : 2*k - count*bsize;

	   const int offset = count*bsize;

	   std::vector<ColorSpinorField*> v2k_(v2k(offset, offset + wsize));

	   for(int j = offset; j < offset+wsize; j++)  blas::copy(vm[j], v2k[j]);
	   commGlobalReductionSet(pipe_l ? false : true); //disable when is_pipelined is true
	   blas::reDotProduct(s.segment(offset, wsize).data(), Az_, v2k_);
	   commGlobalReductionSet(true);

	   if(++count == n_zav_subtask_updates) {
	     count = 0;
	     ctask_id = EigCGComputeTasks::COMPUTE_NOTHING;
	     //Start async communications:

	     if( (pipe_l != 0) && (comm_size() > 1) ) {
		EIGCG_MPI_CHECK_(MPI_Iallreduce(reinterpret_cast<double*>(s.data()),
				                reinterpret_cast<double*>(rr_task_recvbuff.data()),
						2*k,
						MPI_DOUBLE,
						MPI_SUM, MPI_COMM_WORLD,
						&rr_task_request_handle));
	     }
	   }
	 }

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
    inner.delta            = outer.delta;
    inner.precision        = outer.precision; // preconditioners are uni-precision solvers
    inner.precision_sloppy = outer.precision_precondition;

    inner.inv_type        = QUDA_CG_INVERTER;       // use CG solver
    inner.use_init_guess  = QUDA_USE_INIT_GUESS_YES;// use deflated initial guess...

    inner.use_sloppy_partial_accumulator= false;//outer.use_sloppy_partial_accumulator;
  }

  IncEigCG::IncEigCG(DiracMatrix &mat, DiracMatrix &matSloppy, DiracMatrix &matPrecon, SolverParam &param, TimeProfile &profile) :
    Solver(param, profile), mat(mat), matSloppy(matSloppy), matPrecon(matPrecon), K(nullptr), Kparam(param),
    work_space(nullptr), r_pre(nullptr), p_pre(nullptr), eigcg_args(nullptr), profile(profile), init(false)
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


  void IncEigCG::EigenSolve() { eigcg_args->ComputeEv(); }

  void IncEigCG::LegacySearchSpaceUpdate(const double &lanczos_diag, const double &lanczos_offdiag, const double &beta, const double &rnorm) {

     EigCGArgs &args = *eigcg_args;

     ColorSpinorField &Ap = (*work_space)[0];
					//whether we want preconditiond (id = 5) or unpreconditioned (id = 2) residual:
     const int r_idx = K ? 5 : 2;
     ColorSpinorField &z  = (*work_space)[r_idx];

     args.UpdateLanczosMatrix(lanczos_diag, lanczos_offdiag);

     if (args.id == (args.m-1)) blas::copy(*args.Az, Ap);
     //run RR block:
     else if(args.id == args.m) {
	args.CacheNormR(rnorm);
	//Compute (update) Ap = Ap - beta*Ap_old
	blas::xpay(Ap, -beta, *args.Az);

	args.ComputeEv();
        // apply QV task:
	args.apply(0);
	// apply ZAV task:
	args.apply(0);

	args.RestartArgs();
     }

     args.UpdateLanczosBasisAndSearchIdx(z, rnorm);

     return;
  }

  void IncEigCG::PipelinedSearchSpaceUpdate(const double &lanczos_diag, const double &lanczos_offdiag, const double &beta, const double &normr)
  {
    constexpr bool is_pipelined	= true;

    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) return;

    args.UpdateLanczosMatrix<is_pipelined>(lanczos_diag, lanczos_offdiag);

    // launch RR block for args.id == param.m, otherwise is nop
    if(args.id == args.m) {
      args.CacheNormR(normr);
      //Store updated Ap
      blas::copy(*args.Az, (*work_space)[0]);
						// launch async task
      args.rr_task = std::async(std::launch::async, &IncEigCG::EigenSolve, this);
    } else if (args.id == (args.m+args.ev_task_offset)) {
      // do synchronization with EIGENV
      if(args.rr_task.valid()) args.rr_task.wait();
      else warningQuda("Tried to synchronize an invalid task...");
      dslash::aux_worker = eigcg_args.get();
      //args.apply(0);
    } else if (args.id == (args.m+args.pipe_l)) {

      dslash::aux_worker = nullptr;
      args.RestartArgs();
    }

    args.UpdateLanczosBasisAndSearchIdx((*work_space)[4], normr);

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

    if (!init) {
      eigcg_args = std::make_shared <EigCGArgs> (param.m, param.nev, 0, 0, matSloppy.getStencilSteps(), x, param.precision_ritz);//need only deflation meta

      csParam.create        = QUDA_ZERO_FIELD_CREATE;
      csParam.is_composite  = true;
      csParam.composite_dim = K ? 6 : 5;//an extra field for the preconditioned version
      // An auxiliary work space:
      work_space = MakeSharedPtr(csParam);

      if (K) {
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

    ColorSpinorField &Ap  = (*work_space)[0];
    ColorSpinorField &p   = (*work_space)[1];
    ColorSpinorField &r   = (*work_space)[2];
    ColorSpinorField &y   = (*work_space)[3];
    ColorSpinorField &tmp = (*work_space)[4];
    ColorSpinorField &z   = K ? (*work_space)[5] : r;

    // compute initial residual
    //
    matSloppy(r, x, y);
    double r2 = blas::xmyNorm(b, r);

    if( K ) {//apply preconditioner
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

      LegacySearchSpaceUpdate(lanczos_diag, lanczos_offdiag, beta, sqrt(r2));

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

  int IncEigCG::CAEigCGsolve(ColorSpinorField &x, ColorSpinorField &b) {

    const int ev_task_offset  = 16;
    const int tot_pipe_l      = std::min((param.m-2*param.nev), ev_task_offset+param.pipeline);

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

    if (!init) {
      eigcg_args = std::make_shared <EigCGArgs> (param.m, param.nev, tot_pipe_l, ev_task_offset, matSloppy.getStencilSteps(), x, param.precision_ritz);

      csParam.create = QUDA_ZERO_FIELD_CREATE;
      csParam.is_composite  = true;
      csParam.composite_dim = 7;
      // auxiliary work space to keep w, p, s, u fields
      work_space = MakeSharedPtr(csParam);

      eigcg_args->global_stop = stopping(param.tol, b2, param.residual_type);  // stopping condition of solver

      init = true;
    }

    EigCGArgs &args = *eigcg_args;

    if(args.run_residual_correction) {
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

    // compute initial residual

    matSloppy(r, x, y);
    blas::xpay(b, -1.0, r); //r2

    profile.TPSTOP(QUDA_PROFILE_INIT);
    profile.TPSTART(QUDA_PROFILE_PREAMBLE);

//MPI specific:
    std::unique_ptr<double[]> recvbuff(new double[4]);
    MPI_Request iallreduce_request_handle;

    double4 local_buffer;

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;
    double heavy_quark_res = (use_heavy_quark_res) ? sqrt(blas::HeavyQuarkResidualNorm(x, r).z) : 0.0;  // heavy quark res idual

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

    double &nu    = local_buffer.x;
    double &gamma = local_buffer.y;
    double &delta = local_buffer.z;
    double &mu    = local_buffer.w;

    commGlobalReductionSet(false);
    nu     = blas::norm2(r);
    gamma  = blas::norm2(s);
    delta  = blas::reDotProduct(r, s);
    mu     = blas::reDotProduct(p, s);
    commGlobalReductionSet(true);

    if (comm_size() > 1) {

      EIGCG_MPI_CHECK_(MPI_Iallreduce(reinterpret_cast<double*>(&local_buffer),
			            recvbuff.get(),
				    4,
				    MPI_DOUBLE,
			    	    MPI_SUM, MPI_COMM_HANDLE,
				    &iallreduce_request_handle));
    }

    matSloppy(u, s);

    profile.TPSTOP(QUDA_PROFILE_PREAMBLE);
    profile.TPSTART(QUDA_PROFILE_COMPUTE);
    blas::flops = 0;

    //Begin EigCG iterations (init cycle):
    args.restarts = 0;

    int k = 0;

    const double local_stop = x.Precision() == QUDA_DOUBLE_PRECISION ? b2*param.tol*param.tol :  b2*1e-11;
    bool converged = convergence(gamma, heavy_quark_res, args.global_stop, param.tol_hq);

    constexpr int prediction_correction_interval = 16;
    int correction_count;

    printfQuda("\nRunning CA eigCG with %d correction interval\n", prediction_correction_interval);

    if (comm_size() > 1){
      EIGCG_MPI_CHECK_(MPI_Wait(&iallreduce_request_handle, MPI_STATUS_IGNORE));
      memcpy(reinterpret_cast<double*>(&local_buffer), recvbuff.get(), 4*sizeof(double));
    }

    dslash::aux_worker = nullptr;

    blas::createAuxBlasStream();

    while ( !converged && k < param.maxiter ) {
      //Update search space
      PipelinedSearchSpaceUpdate(lanczos_diag, lanczos_offdiag, beta, sqrt(nu));

      PrintStats("CAEigCG", k, nu, b2, heavy_quark_res);

      alpha_old_inv = alpha_inv;
      beta_old      = beta;

      alpha         = nu / mu;
      alpha_inv     = 1.0 / alpha;

      nunew = nu - 2*alpha*delta + alpha*alpha*gamma;
      beta  = nunew / nu;

      lanczos_diag     = (alpha_inv + beta_old*alpha_old_inv);
      lanczos_offdiag  = (-sqrt(beta)*alpha_inv);

      blas::axpy(+alpha, p, y);
      commGlobalReductionSet(false);
      local_buffer = quadrupleEigCGUpdate(alpha, beta, r, s, u, w, p);
      commGlobalReductionSet(true);

      if (comm_size() > 1){
	EIGCG_MPI_CHECK_(MPI_Iallreduce(reinterpret_cast<double*>(&local_buffer),
				recvbuff.get(),
				4,
				MPI_DOUBLE,
				MPI_SUM, MPI_COMM_HANDLE,
				&iallreduce_request_handle));
      }

      if(dslash::aux_worker) {
	 blas::synchronizeAuxBlasStream();     
	 blas::registerAuxBlasStream();
      }

      matSloppy(u, s, tmp);

      if (k % prediction_correction_interval == 0){
        matSloppy(w, r, tmp);
	correction_count += 1;
      }

      if(dslash::aux_worker) blas::unregisterAuxBlasStream();

      if (comm_size() > 1){
        EIGCG_MPI_CHECK_(MPI_Wait(&iallreduce_request_handle, MPI_STATUS_IGNORE));
        memcpy(reinterpret_cast<double*>(&local_buffer), recvbuff.get(), 4*sizeof(double));
      }

      k++;

      converged = convergence(nu, heavy_quark_res, args.global_stop, param.tol_hq) or convergence(nu, heavy_quark_res, local_stop, param.tol_hq);
    }

    blas::zero(*args.V2k);

    dslash::aux_worker = nullptr;
    blas::destroyAuxBlasStream();

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

         defl.increment(*eigcg_args->Vm, param.nev);
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

       eigcg_args->Vm.reset(); eigcg_args->V2k.reset();

       const int max_nev = defl.size();//param.m;
       printfQuda("\nRequested to reserve %d eigenvectors with max tol %le.\n", max_nev, param.eigenval_tol);
       defl.reduce(param.eigenval_tol, max_nev);
     }
     return;
  }

} // namespace quda
