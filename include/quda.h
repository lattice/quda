#pragma once

/**
 * @file  quda.h
 * @brief Main header file for the QUDA library
 *
 * Note to QUDA developers: When adding new members to QudaGaugeParam
 * and QudaInvertParam, be sure to update lib/check_params.h as well
 * as the Fortran interface in lib/quda_fortran.F90.
 */

#include <enum_quda.h>
#include <stdio.h> /* for FILE */
#include <quda_define.h>
#include <quda_constants.h>

#ifndef __CUDACC_RTC__
#define double_complex double _Complex
#else // keep NVRTC happy since it can't handle C types
#define double_complex double2
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /**
   * Parameters having to do with the gauge field or the
   * interpretation of the gauge field by various Dirac operators
   */
  typedef struct QudaGaugeParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size */
    QudaFieldLocation location; /**< The location of the gauge field */

    int X[4];             /**< The local space-time dimensions (without checkboarding) */

    double anisotropy;    /**< Used for Wilson and Wilson-clover */
    double tadpole_coeff; /**< Used for staggered only */
    double scale; /**< Used by staggered long links */

    QudaLinkType type; /**< The link type of the gauge field (e.g., Wilson, fat, long, etc.) */
    QudaGaugeFieldOrder gauge_order; /**< The ordering on the input gauge field */

    QudaTboundary t_boundary;  /**< The temporal boundary condition that will be used for fermion fields */

    QudaPrecision cpu_prec; /**< The precision used by the caller */

    QudaPrecision cuda_prec; /**< The precision of the cuda gauge field */
    QudaReconstructType reconstruct; /**< The reconstruction type of the cuda gauge field */

    QudaPrecision cuda_prec_sloppy; /**< The precision of the sloppy gauge field */
    QudaReconstructType reconstruct_sloppy; /**< The recontruction type of the sloppy gauge field */

    QudaPrecision cuda_prec_refinement_sloppy; /**< The precision of the sloppy gauge field for the refinement step in multishift */
    QudaReconstructType reconstruct_refinement_sloppy; /**< The recontruction type of the sloppy gauge field for the refinement step in multishift*/

    QudaPrecision cuda_prec_precondition; /**< The precision of the preconditioner gauge field */
    QudaReconstructType reconstruct_precondition; /**< The recontruction type of the preconditioner gauge field */

    QudaPrecision cuda_prec_eigensolver;         /**< The precision of the eigensolver gauge field */
    QudaReconstructType reconstruct_eigensolver; /**< The recontruction type of the eigensolver gauge field */

    QudaGaugeFixed gauge_fix; /**< Whether the input gauge field is in the axial gauge or not */

    int ga_pad;       /**< The pad size that the cudaGaugeField will use (default=0) */

    int site_ga_pad;  /**< Used by link fattening and the gauge and fermion forces */

    int staple_pad;   /**< Used by link fattening */
    int llfat_ga_pad; /**< Used by link fattening */
    int mom_ga_pad;   /**< Used by the gauge and fermion forces */

    QudaStaggeredPhase staggered_phase_type; /**< Set the staggered phase type of the links */
    int staggered_phase_applied; /**< Whether the staggered phase has already been applied to the links */

    double i_mu; /**< Imaginary chemical potential */

    int overlap; /**< Width of overlapping domains */

    int overwrite_gauge; /**< When computing gauge, should we overwrite it or accumulate to it */
    int overwrite_mom;   /**< When computing momentum, should we overwrite it or accumulate to it */

    int use_resident_gauge;  /**< Use the resident gauge field as input */
    int use_resident_mom;    /**< Use the resident momentum field as input*/
    int make_resident_gauge; /**< Make the result gauge field resident */
    int make_resident_mom;   /**< Make the result momentum field resident */
    int return_result_gauge; /**< Return the result gauge field */
    int return_result_mom;   /**< Return the result momentum field */

    size_t gauge_offset; /**< Offset into MILC site struct to the gauge field (only if gauge_order=MILC_SITE_GAUGE_ORDER) */
    size_t mom_offset; /**< Offset into MILC site struct to the momentum field (only if gauge_order=MILC_SITE_GAUGE_ORDER) */
    size_t site_size; /**< Size of MILC site struct (only if gauge_order=MILC_SITE_GAUGE_ORDER) */
  } QudaGaugeParam;


  /**
   * Parameters relating to the solver and the choice of Dirac operator.
   */
  typedef struct QudaInvertParam_s {

    /** Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size */
    size_t struct_size;

    QudaFieldLocation input_location; /**< The location of the input field */
    QudaFieldLocation output_location; /**< The location of the output field */

    QudaDslashType dslash_type; /**< The Dirac Dslash type that is being used */
    QudaInverterType inv_type; /**< Which linear solver to use */

    double mass;  /**< Used for staggered only */
    double kappa; /**< Used for Wilson and Wilson-clover */

    double m5;    /**< Domain wall height */
    int Ls;       /**< Extent of the 5th dimension (for domain wall) */

    double_complex b_5[QUDA_MAX_DWF_LS]; /**< Mobius coefficients - only real part used if regular Mobius */
    double_complex c_5[QUDA_MAX_DWF_LS]; /**< Mobius coefficients - only real part used if regular Mobius */

    /**<
     * The following specifies the EOFA parameters. Notation follows arXiv:1706.05843
     * eofa_shift: the "\beta" in the paper
     * eofa_pm: plus or minus for the EOFA operator
     * mq1, mq2, mq3 are the three masses corresponds to Hasenbusch mass spliting.
     * As far as I know mq1 is always the same as "mass" but it's here just for consistence.
     * */
    double eofa_shift;
    int eofa_pm;
    double mq1;
    double mq2;
    double mq3;

    double mu;    /**< Twisted mass parameter */
    double tm_rho;  /**< Hasenbusch mass shift applied like twisted mass to diagonal (but not inverse) */
    double epsilon; /**< Twisted mass parameter */

    QudaTwistFlavorType twist_flavor;  /**< Twisted mass flavor */

    int laplace3D; /**< omit this direction from laplace operator: x,y,z,t -> 0,1,2,3 (-1 is full 4D) */

    double tol;    /**< Solver tolerance in the L2 residual norm */
    double tol_restart;   /**< Solver tolerance in the L2 residual norm (used to restart InitCG) */
    double tol_hq; /**< Solver tolerance in the heavy quark residual norm */

    int compute_true_res; /** Whether to compute the true residual post solve */
    double true_res; /**< Actual L2 residual norm achieved in solver */
    double true_res_hq; /**< Actual heavy quark residual norm achieved in solver */
    int maxiter; /**< Maximum number of iterations in the linear solver */
    double reliable_delta; /**< Reliable update tolerance */
    double reliable_delta_refinement; /**< Reliable update tolerance used in post multi-shift solver refinement */
    int use_alternative_reliable; /**< Whether to use alternative reliable updates */
    int use_sloppy_partial_accumulator; /**< Whether to keep the partial solution accumuator in sloppy precision */

    /**< This parameter determines how often we accumulate into the
       solution vector from the direction vectors in the solver.
       E.g., running with solution_accumulator_pipeline = 4, means we
       will update the solution vector every four iterations using the
       direction vectors from the prior four iterations.  This
       increases performance of mixed-precision solvers since it means
       less high-precision vector round-trip memory travel, but
       requires more low-precision memory allocation. */
    int solution_accumulator_pipeline;

    /**< This parameter determines how many consecutive reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase;

    /**< This parameter determines how many total reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase_total;

    /**< This parameter determines how many consecutive heavy-quark
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_hq_res_increase;

    /**< This parameter determines how many total heavy-quark residual
    restarts we tolerate before terminating the solver, i.e., how long
    do we want to keep trying to converge */
    int max_hq_res_restart_total;

    /**< After how many iterations shall the heavy quark residual be updated */
    int heavy_quark_check;

    int pipeline; /**< Whether to use a pipelined solver with less global sums */

    int num_offset; /**< Number of offsets in the multi-shift solver */

    int num_src; /**< Number of sources in the multiple source solver */

    int num_src_per_sub_partition; /**< Number of sources in the multiple source solver, but per sub-partition */

    /**< The grid of sub-partition according to which the processor grid will be partitioned.
    Should have:
      split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3] * num_src_per_sub_partition == num_src. **/
    int split_grid[QUDA_MAX_DIM];

    int overlap; /**< Width of domain overlaps */

    /** Offsets for multi-shift solver */
    double offset[QUDA_MAX_MULTI_SHIFT];

    /** Solver tolerance for each offset */
    double tol_offset[QUDA_MAX_MULTI_SHIFT];

    /** Solver tolerance for each shift when refinement is applied using the heavy-quark residual */
    double tol_hq_offset[QUDA_MAX_MULTI_SHIFT];

    /** Actual L2 residual norm achieved in solver for each offset */
    double true_res_offset[QUDA_MAX_MULTI_SHIFT];

    /** Iterated L2 residual norm achieved in multi shift solver for each offset */
    double iter_res_offset[QUDA_MAX_MULTI_SHIFT];

    /** Actual heavy quark residual norm achieved in solver for each offset */
    double true_res_hq_offset[QUDA_MAX_MULTI_SHIFT];

    /** Residuals in the partial faction expansion */
    double residue[QUDA_MAX_MULTI_SHIFT];

    /** Whether we should evaluate the action after the linear solver*/
    int compute_action;

    /** Computed value of the bilinear action (complex-valued)
	invert: \phi^\dagger A^{-1} \phi
	multishift: \phi^\dagger r(x) \phi = \phi^\dagger (sum_k residue[k] * (A + offset[k])^{-1} ) \phi */
    double action[2];

    QudaSolutionType solution_type;  /**< Type of system to solve */
    QudaSolveType solve_type;        /**< How to solve it */
    QudaMatPCType matpc_type;        /**< The preconditioned matrix type */
    QudaDagType dagger;              /**< Whether we are using the Hermitian conjugate system or not */
    QudaMassNormalization mass_normalization; /**< The mass normalization is being used by the caller */
    QudaSolverNormalization solver_normalization; /**< The normalization desired in the solver */

    QudaPreserveSource preserve_source;       /**< Preserve the source or not in the linear solver (deprecated) */

    QudaPrecision cpu_prec;                /**< The precision used by the input fermion fields */
    QudaPrecision cuda_prec;               /**< The precision used by the QUDA solver */
    QudaPrecision cuda_prec_sloppy;        /**< The precision used by the QUDA sloppy operator */
    QudaPrecision cuda_prec_refinement_sloppy; /**< The precision of the sloppy gauge field for the refinement step in multishift */
    QudaPrecision cuda_prec_precondition;  /**< The precision used by the QUDA preconditioner */
    QudaPrecision cuda_prec_eigensolver;   /**< The precision used by the QUDA eigensolver */

    QudaDiracFieldOrder dirac_order;       /**< The order of the input and output fermion fields */

    QudaGammaBasis gamma_basis;            /**< Gamma basis of the input and output host fields */

    QudaFieldLocation clover_location;     /**< The location of the clover field */
    QudaPrecision clover_cpu_prec;         /**< The precision used for the input clover field */
    QudaPrecision clover_cuda_prec;        /**< The precision used for the clover field in the QUDA solver */
    QudaPrecision clover_cuda_prec_sloppy; /**< The precision used for the clover field in the QUDA sloppy operator */
    QudaPrecision clover_cuda_prec_refinement_sloppy; /**< The precision of the sloppy clover field for the refinement step in multishift */
    QudaPrecision clover_cuda_prec_precondition; /**< The precision used for the clover field in the QUDA preconditioner */
    QudaPrecision clover_cuda_prec_eigensolver;  /**< The precision used for the clover field in the QUDA eigensolver */

    QudaCloverFieldOrder clover_order;     /**< The order of the input clover field */
    QudaUseInitGuess use_init_guess;       /**< Whether to use an initial guess in the solver or not */

    double clover_csw;                     /**< Csw coefficient of the clover term */
    double clover_coeff;                   /**< Coefficient of the clover term */
    double clover_rho;                     /**< Real number added to the clover diagonal (not to inverse) */

    int compute_clover_trlog;              /**< Whether to compute the trace log of the clover term */
    double trlogA[2];                      /**< The trace log of the clover term (even/odd computed separately) */

    int compute_clover;                    /**< Whether to compute the clover field */
    int compute_clover_inverse;            /**< Whether to compute the clover inverse field */
    int return_clover;                     /**< Whether to copy back the clover matrix field */
    int return_clover_inverse;             /**< Whether to copy back the inverted clover matrix field */

    QudaVerbosity verbosity;               /**< The verbosity setting to use in the solver */

    int iter;                              /**< The number of iterations performed by the solver */
    double gflops;                         /**< The Gflops rate of the solver */
    double secs;                           /**< The time taken by the solver */

    QudaTune tune; /**< Enable auto-tuning? (default = QUDA_TUNE_YES) */

    /** Number of steps in s-step algorithms */
    int Nsteps;

    /** Maximum size of Krylov space used by solver */
    int gcrNkrylov;

    /*
     * The following parameters are related to the solver
     * preconditioner, if enabled.
     */

    /**
     * The inner Krylov solver used in the preconditioner.  Set to
     * QUDA_INVALID_INVERTER to disable the preconditioner entirely.
     */
    QudaInverterType inv_type_precondition;

    /** Preconditioner instance, e.g., multigrid */
    void *preconditioner;

    /** Deflation instance */
    void *deflation_op;

    /** defines deflation */
    void *eig_param;

    /** If true, deflate the initial guess */
    QudaBoolean deflate;

    /** Dirac Dslash used in preconditioner */
    QudaDslashType dslash_type_precondition;
    /** Verbosity of the inner Krylov solver */
    QudaVerbosity verbosity_precondition;

    /** Tolerance in the inner solver */
    double tol_precondition;

    /** Maximum number of iterations allowed in the inner solver */
    int maxiter_precondition;

    /** Relaxation parameter used in GCR-DD (default = 1.0) */
    double omega;

    /** Basis for CA algorithms */
    QudaCABasis ca_basis;

    /** Minimum eigenvalue for Chebyshev CA basis */
    double ca_lambda_min;

    /** Maximum eigenvalue for Chebyshev CA basis */
    double ca_lambda_max;

    /** Basis for CA algorithms in a preconditioned solver */
    QudaCABasis ca_basis_precondition;

    /** Minimum eigenvalue for Chebyshev CA basis in a preconditioner solver */
    double ca_lambda_min_precondition;

    /** Maximum eigenvalue for Chebyshev CA basis in a preconditioner solver */
    double ca_lambda_max_precondition;

    /** Number of preconditioner cycles to perform per iteration */
    int precondition_cycle;

    /** Whether to use additive or multiplicative Schwarz preconditioning */
    QudaSchwarzType schwarz_type;

    /** The type of accelerator type to use for preconditioner */
    QudaAcceleratorType accelerator_type_precondition;

    /**
     * The following parameters are the ones used to perform the adaptive MADWF in MSPCG
     * See section 3.3 of [arXiv:2104.05615]
     */

    /** The diagonal constant to suppress the low modes when performing 5D transfer */
    double madwf_diagonal_suppressor;

    /** The target MADWF Ls to be used in the accelerator */
    int madwf_ls;

    /** The minimum number of iterations after which to generate the null vectors for MADWF */
    int madwf_null_miniter;

    /** The maximum tolerance after which to generate the null vectors for MADWF */
    double madwf_null_tol;

    /** The maximum number of iterations for the training iterations */
    int madwf_train_maxiter;

    /** Whether to load the MADWF parameters from the file system */
    QudaBoolean madwf_param_load;

    /** Whether to save the MADWF parameters to the file system */
    QudaBoolean madwf_param_save;

    /** Path to load from the file system */
    char madwf_param_infile[256];

    /** Path to save to the file system */
    char madwf_param_outfile[256];

    /**
     * Whether to use the L2 relative residual, Fermilab heavy-quark
     * residual, or both to determine convergence.  To require that both
     * stopping conditions are satisfied, use a bitwise OR as follows:
     *
     * p.residual_type = (QudaResidualType) (QUDA_L2_RELATIVE_RESIDUAL
     *                                     | QUDA_HEAVY_QUARK_RESIDUAL);
     */
    QudaResidualType residual_type;

    /**Parameters for deflated solvers*/
    /** The precision of the Ritz vectors */
    QudaPrecision cuda_prec_ritz;
    /** How many vectors to compute after one solve
     *  for eigCG recommended values 8 or 16
    */
    int n_ev;
    /** EeigCG  : Search space dimension
     *  gmresdr : Krylov subspace dimension
     */
    int max_search_dim;
    /** For systems with many RHS: current RHS index */
    int rhs_idx;
    /** Specifies deflation space volume: total number of eigenvectors is n_ev*deflation_grid */
    int deflation_grid;
    /** eigCG: selection criterion for the reduced eigenvector set */
    double eigenval_tol;
    /** mixed precision eigCG tuning parameter:  minimum search vector space restarts */
    int eigcg_max_restarts;
    /** initCG tuning parameter:  maximum restarts */
    int max_restart_num;
    /** initCG tuning parameter:  tolerance for cg refinement corrections in the deflation stage */
    double inc_tol;

    /** Whether to make the solution vector(s) after the solve */
    int make_resident_solution;

    /** Whether to use the resident solution vector(s) */
    int use_resident_solution;

    /** Whether to use the solution vector to augment the chronological basis */
    int chrono_make_resident;

    /** Whether the solution should replace the last entry in the chronology */
    int chrono_replace_last;

    /** Whether to use the resident chronological basis */
    int chrono_use_resident;

    /** The maximum length of the chronological history to store */
    int chrono_max_dim;

    /** The index to indicate which chrono history we are augmenting */
    int chrono_index;

    /** Precision to store the chronological basis in */
    QudaPrecision chrono_precision;

    /** Which external library to use in the linear solvers (Eigen) */
    QudaExtLibType extlib_type;

    /** Whether to use the platform native or generic BLAS / LAPACK */
    QudaBoolean native_blas_lapack;

    /** Whether to use fused kernels for mobius */
    QudaBoolean use_mobius_fused_kernel;

  } QudaInvertParam;

  // Parameter set for solving eigenvalue problems.
  typedef struct QudaEigParam_s {
    /** Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size */
    size_t struct_size;

    // EIGENSOLVER PARAMS
    //-------------------------------------------------
    /** Used to store information pertinent to the operator **/
    QudaInvertParam *invert_param;

    /** Type of eigensolver algorithm to employ **/
    QudaEigType eig_type;

    /** Use Polynomial Acceleration **/
    QudaBoolean use_poly_acc;

    /** Degree of the Chebysev polynomial **/
    int poly_deg;

    /** Range used in polynomial acceleration **/
    double a_min;
    double a_max;

    /** Whether to preserve the deflation space between solves.  If
        true, the space will be stored in an instance of the
        deflation_space struct, pointed to by preserve_deflation_space */
    QudaBoolean preserve_deflation;

    /** This is where we store the deflation space.  This will point
        to an instance of deflation_space. When a deflated solver is enabled, the deflation space will be obtained from this.  */
    void *preserve_deflation_space;

    /** If we restore the deflation space, this boolean indicates
        whether we are also preserving the evalues or recomputing
        them.  For example if a different mass shift is being used
        than the one used to generate the space, then this should be
        false, but preserve_deflation would be true */
    QudaBoolean preserve_evals;

    /** What type of Dirac operator we are using **/
    /** If !(use_norm_op) && !(use_dagger) use M. **/
    /** If use_dagger, use Mdag **/
    /** If use_norm_op, use MdagM **/
    /** If use_norm_op && use_dagger use MMdag. **/
    /** If use_pc for any, then use the even-odd pc version **/
    QudaBoolean use_dagger;
    QudaBoolean use_norm_op;
    QudaBoolean use_pc;

    /** Use Eigen routines to eigensolve the upper Hessenberg via QR **/
    QudaBoolean use_eigen_qr;

    /** Performs an MdagM solve, then constructs the left and right SVD. **/
    QudaBoolean compute_svd;

    /** Performs the \gamma_5 OP solve by Post multipling the eignvectors with
        \gamma_5 before computing the eigenvalues */
    QudaBoolean compute_gamma5;

    /** If true, the solver will error out if the convergence criteria are not met **/
    QudaBoolean require_convergence;

    /** Which part of the spectrum to solve **/
    QudaEigSpectrumType spectrum;

    /** Size of the eigenvector search space **/
    int n_ev;
    /** Total size of Krylov space **/
    int n_kr;
    /** Max number of locked eigenpairs (deduced at runtime) **/
    int nLockedMax;
    /** Number of requested converged eigenvectors **/
    int n_conv;
    /** Number of requested converged eigenvectors to use in deflation **/
    int n_ev_deflate;
    /** Tolerance on the least well known eigenvalue's residual **/
    double tol;
    /** Tolerance on the QR iteration **/
    double qr_tol;
    /** For IRLM/IRAM, check every nth restart **/
    int check_interval;
    /** For IRLM/IRAM, quit after n restarts **/
    int max_restarts;
    /** For the Ritz rotation, the maximal number of extra vectors the solver may allocate **/
    int batched_rotate;
    /** For block method solvers, the block size **/
    int block_size;
    /** For block method solvers, quit after n attempts at block orthonormalisation **/
    int max_ortho_attempts;
    /** For hybrid modifeld Gram-Schmidt orthonormalisations **/
    int ortho_block_size;

    /** In the test function, cross check the device result against ARPACK **/
    QudaBoolean arpack_check;
    /** For Arpack cross check, name of the Arpack logfile **/
    char arpack_logfile[512];

    /** Name of the QUDA logfile (residua, upper Hessenberg/tridiag matrix updates) **/
    char QUDA_logfile[512];

    //-------------------------------------------------

    // EIG-CG PARAMS
    //-------------------------------------------------
    int nk;
    int np;

    /** Whether to load eigenvectors */
    QudaBoolean import_vectors;

    /** The precision of the Ritz vectors */
    QudaPrecision cuda_prec_ritz;

    /** The memory type used to keep the Ritz vectors */
    QudaMemoryType mem_type_ritz;

    /** Location where deflation should be done */
    QudaFieldLocation location;

    /** Whether to run the verification checks once set up is complete */
    QudaBoolean run_verify;

    /** Filename prefix where to load the null-space vectors */
    char vec_infile[256];

    /** Filename prefix for where to save the null-space vectors */
    char vec_outfile[256];

    /** The precision with which to save the vectors */
    QudaPrecision save_prec;

    /** Whether to inflate single-parity eigen-vector I/O to a full
        field (e.g., enabling this is required for compatability with
        MILC I/O) */
    QudaBoolean io_parity_inflate;

    /** The Gflops rate of the eigensolver setup */
    double gflops;

    /**< The time taken by the eigensolver setup */
    double secs;

    /** Which external library to use in the deflation operations (Eigen) */
    QudaExtLibType extlib_type;
    //-------------------------------------------------
  } QudaEigParam;

  typedef struct QudaMultigridParam_s {

    /** Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct size */
    size_t struct_size;

    QudaInvertParam *invert_param;

    QudaEigParam *eig_param[QUDA_MAX_MG_LEVEL];

    /** Number of multigrid levels */
    int n_level;

    /** Geometric block sizes to use on each level */
    int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];

    /** Spin block sizes to use on each level */
    int spin_block_size[QUDA_MAX_MG_LEVEL];

    /** Number of null-space vectors to use on each level */
    int n_vec[QUDA_MAX_MG_LEVEL];

    /** Precision to store the null-space vectors in (post block orthogonalization) */
    QudaPrecision precision_null[QUDA_MAX_MG_LEVEL];

    /** Number of times to repeat Gram-Schmidt in block orthogonalization */
    int n_block_ortho[QUDA_MAX_MG_LEVEL];

    /** Whether to do passes at block orthogonalize in fixed point for improved accuracy */
    QudaBoolean block_ortho_two_pass[QUDA_MAX_MG_LEVEL];

    /** Verbosity on each level of the multigrid */
    QudaVerbosity verbosity[QUDA_MAX_MG_LEVEL];

    /** Setup MMA usage on each level of the multigrid */
    QudaBoolean setup_use_mma[QUDA_MAX_MG_LEVEL];

    /** Dslash MMA usage on each level of the multigrid */
    QudaBoolean dslash_use_mma[QUDA_MAX_MG_LEVEL];

    /** Inverter to use in the setup phase */
    QudaInverterType setup_inv_type[QUDA_MAX_MG_LEVEL];

    /** Number of setup iterations */
    int num_setup_iter[QUDA_MAX_MG_LEVEL];

    /** Tolerance to use in the setup phase */
    double setup_tol[QUDA_MAX_MG_LEVEL];

    /** Maximum number of iterations for each setup solver */
    int setup_maxiter[QUDA_MAX_MG_LEVEL];

    /** Maximum number of iterations for refreshing the null-space vectors */
    int setup_maxiter_refresh[QUDA_MAX_MG_LEVEL];

    /** Basis to use for CA solver setup */
    QudaCABasis setup_ca_basis[QUDA_MAX_MG_LEVEL];

    /** Basis size for CA solver setup */
    int setup_ca_basis_size[QUDA_MAX_MG_LEVEL];

    /** Minimum eigenvalue for Chebyshev CA basis */
    double setup_ca_lambda_min[QUDA_MAX_MG_LEVEL];

    /** Maximum eigenvalue for Chebyshev CA basis */
    double setup_ca_lambda_max[QUDA_MAX_MG_LEVEL];

    /** Null-space type to use in the setup phase */
    QudaSetupType setup_type;

    /** Pre orthonormalize vectors in the setup phase */
    QudaBoolean pre_orthonormalize;

    /** Post orthonormalize vectors in the setup phase */
    QudaBoolean post_orthonormalize;

    /** The solver that wraps around the coarse grid correction and smoother */
    QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL];

    /** Tolerance for the solver that wraps around the coarse grid correction and smoother */
    double coarse_solver_tol[QUDA_MAX_MG_LEVEL];

    /** Maximum number of iterations for the solver that wraps around the coarse grid correction and smoother */
    int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL];

    /** Basis to use for CA coarse solvers */
    QudaCABasis coarse_solver_ca_basis[QUDA_MAX_MG_LEVEL];

    /** Basis size for CA coarse solvers */
    int coarse_solver_ca_basis_size[QUDA_MAX_MG_LEVEL];

    /** Minimum eigenvalue for Chebyshev CA basis */
    double coarse_solver_ca_lambda_min[QUDA_MAX_MG_LEVEL];

    /** Maximum eigenvalue for Chebyshev CA basis */
    double coarse_solver_ca_lambda_max[QUDA_MAX_MG_LEVEL];

    /** Smoother to use on each level */
    QudaInverterType smoother[QUDA_MAX_MG_LEVEL];

    /** Tolerance to use for the smoother / solver on each level */
    double smoother_tol[QUDA_MAX_MG_LEVEL];

    /** Number of pre-smoother applications on each level */
    int nu_pre[QUDA_MAX_MG_LEVEL];

    /** Number of post-smoother applications on each level */
    int nu_post[QUDA_MAX_MG_LEVEL];

    /** Basis to use for CA smoother solvers */
    QudaCABasis smoother_solver_ca_basis[QUDA_MAX_MG_LEVEL];

    /** Minimum eigenvalue for Chebyshev CA smoother basis */
    double smoother_solver_ca_lambda_min[QUDA_MAX_MG_LEVEL];

    /** Maximum eigenvalue for Chebyshev CA smoother basis */
    double smoother_solver_ca_lambda_max[QUDA_MAX_MG_LEVEL];

    /** Over/under relaxation factor for the smoother at each level */
    double omega[QUDA_MAX_MG_LEVEL];

    /** Precision to use for halo communication in the smoother */
    QudaPrecision smoother_halo_precision[QUDA_MAX_MG_LEVEL];

    /** Whether to use additive or multiplicative Schwarz preconditioning in the smoother */
    QudaSchwarzType smoother_schwarz_type[QUDA_MAX_MG_LEVEL];

    /** Number of Schwarz cycles to apply */
    int smoother_schwarz_cycle[QUDA_MAX_MG_LEVEL];

    /** The type of residual to send to the next coarse grid, and thus the
	type of solution to receive back from this coarse grid */
    QudaSolutionType coarse_grid_solution_type[QUDA_MAX_MG_LEVEL];

    /** The type of smoother solve to do on each grid (e/o preconditioning or not)*/
    QudaSolveType smoother_solve_type[QUDA_MAX_MG_LEVEL];

    /** The type of multigrid cycle to perform at each level */
    QudaMultigridCycleType cycle_type[QUDA_MAX_MG_LEVEL];

    /** Whether to use global reductions or not for the smoother / solver at each level */
    QudaBoolean global_reduction[QUDA_MAX_MG_LEVEL];

    /** Location where each level should be done */
    QudaFieldLocation location[QUDA_MAX_MG_LEVEL];

    /** Location where the coarse-operator construction will be computedn */
    QudaFieldLocation setup_location[QUDA_MAX_MG_LEVEL];

    /** Whether to use eigenvectors for the nullspace or, if the coarsest instance deflate*/
    QudaBoolean use_eig_solver[QUDA_MAX_MG_LEVEL];

    /** Minimize device memory allocations during the adaptive setup,
        placing temporary fields in mapped memory instad of device
        memory */
    QudaBoolean setup_minimize_memory;

    /** Whether to compute the null vectors or reload them */
    QudaComputeNullVector compute_null_vector;

    /** Whether to generate on all levels or just on level 0 */
    QudaBoolean generate_all_levels;

    /** Whether to run the verification checks once set up is complete */
    QudaBoolean run_verify;

    /** Whether to run null Vs eigen vector overlap checks once set up is complete */
    QudaBoolean run_low_mode_check;

    /** Whether to run null vector oblique checks once set up is complete */
    QudaBoolean run_oblique_proj_check;

    /** Whether to load the null-space vectors to disk (requires QIO) */
    QudaBoolean vec_load[QUDA_MAX_MG_LEVEL];

    /** Filename prefix where to load the null-space vectors */
    char vec_infile[QUDA_MAX_MG_LEVEL][256];

    /** Whether to store the null-space vectors to disk (requires QIO) */
    QudaBoolean vec_store[QUDA_MAX_MG_LEVEL];

    /** Filename prefix for where to save the null-space vectors */
    char vec_outfile[QUDA_MAX_MG_LEVEL][256];

    /** Whether to use and initial guess during coarse grid deflation */
    QudaBoolean coarse_guess;

    /** Whether to preserve the deflation space during MG update */
    QudaBoolean preserve_deflation;

    /** The Gflops rate of the multigrid solver setup */
    double gflops;

    /**< The time taken by the multigrid solver setup */
    double secs;

    /** Multiplicative factor for the mu parameter */
    double mu_factor[QUDA_MAX_MG_LEVEL];

    /** Boolean for aggregation type, implies staggered or not */
    QudaTransferType transfer_type[QUDA_MAX_MG_LEVEL];

    /** Whether or not to let MG coarsening drop improvements, for ex dropping long links in small aggregation dimensions */
    QudaBoolean allow_truncation;

    /** Whether or not to use the dagger approximation for the KD preconditioned operator */
    QudaBoolean staggered_kd_dagger_approximation;

    /** Whether to do a full (false) or thin (true) update in the context of updateMultigridQuda */
    QudaBoolean thin_update_only;
  } QudaMultigridParam;

  typedef struct QudaGaugeObservableParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/
    QudaBoolean su_project;               /**< Whether to project onto the manifold prior to measurement */
    QudaBoolean compute_plaquette;        /**< Whether to compute the plaquette */
    double plaquette[3];                  /**< Total, spatial and temporal field energies, respectively */
    QudaBoolean compute_polyakov_loop;    /**< Whether to compute the temporal Polyakov loop */
    double ploop[2];                      /**< Real and imaginary part of temporal Polyakov loop */
    QudaBoolean compute_gauge_loop_trace; /**< Whether to compute gauge loop traces */
    double_complex *traces;               /**< Individual complex traces of each loop */
    int **input_path_buff;                /**< Array of paths */
    int *path_length;                     /**< Length of each path */
    double *loop_coeff;                   /**< Multiplicative factor for each loop */
    int num_paths;                        /**< Total number of paths */
    int max_length;                       /**< Maximum length of any path */
    double factor;                        /**< Global multiplicative factor to apply to each loop trace */
    QudaBoolean compute_qcharge;          /**< Whether to compute the topological charge and field energy */
    double qcharge;                       /**< Computed topological charge */
    double energy[3];                     /**< Total, spatial and temporal field energies, respectively */
    QudaBoolean compute_qcharge_density;  /**< Whether to compute the topological charge density */
    void *qcharge_density; /**< Pointer to host array of length volume where the q-charge density will be copied */
    QudaBoolean
      remove_staggered_phase; /**< Whether or not the resident gauge field has staggered phases applied and if they should
                                 be removed; this was needed for the Polyakov loop calculation when called through MILC,
                                 with the underlying issue documented https://github.com/lattice/quda/issues/1315 */
  } QudaGaugeObservableParam;

  typedef struct QudaGaugeSmearParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/
    unsigned int n_steps; /**< The total number of smearing steps to perform. */
    double epsilon;       /**< Serves as one of the coefficients in Over Improved Stout smearing, or as the step size in
                             Wilson/Symanzik flow */
    double alpha;         /**< The single coefficient used in APE smearing */
    double rho; /**< Serves as one of the coefficients used in Over Improved Stout smearing, or as the single coefficient used in Stout */
    unsigned int meas_interval;    /**< Perform the requested measurements on the gauge field at this interval */
    QudaGaugeSmearType smear_type; /**< The smearing type to perform */
  } QudaGaugeSmearParam;

  typedef struct QudaBLASParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/

    QudaBLASType blas_type; /**< Type of BLAS computation to perfrom */

    // GEMM params
    QudaBLASOperation trans_a; /**< operation op(A) that is non- or (conj.) transpose. */
    QudaBLASOperation trans_b; /**< operation op(B) that is non- or (conj.) transpose. */
    int m;                     /**< number of rows of matrix op(A) and C. */
    int n;                     /**< number of columns of matrix op(B) and C. */
    int k;                     /**< number of columns of op(A) and rows of op(B). */
    int lda;                   /**< leading dimension of two-dimensional array used to store the matrix A. */
    int ldb;                   /**< leading dimension of two-dimensional array used to store matrix B. */
    int ldc;                   /**< leading dimension of two-dimensional array used to store matrix C. */
    int a_offset;              /**< position of the A array from which begin read/write. */
    int b_offset;              /**< position of the B array from which begin read/write. */
    int c_offset;              /**< position of the C array from which begin read/write. */
    int a_stride;              /**< stride of the A array in strided(batched) mode */
    int b_stride;              /**< stride of the B array in strided(batched) mode */
    int c_stride;              /**< stride of the C array in strided(batched) mode */
    double_complex alpha; /**< scalar used for multiplication. */
    double_complex beta;  /**< scalar used for multiplication. If beta==0, C does not have to be a valid input. */

    // LU inversion params
    int inv_mat_size; /**< The rank of the square matrix in the LU inversion */

    // Common params
    int batch_count;              /**< number of pointers contained in arrayA, arrayB and arrayC. */
    QudaBLASDataType data_type;   /**< Specifies if using S(C) or D(Z) BLAS type */
    QudaBLASDataOrder data_order; /**< Specifies if using Row or Column major */

  } QudaBLASParam;

  /*
   * Interface functions, found in interface_quda.cpp
   */

  /**
   * Set parameters related to status reporting.
   *
   * In typical usage, this function will be called once (or not at
   * all) just before the call to initQuda(), but it's valid to call
   * it any number of times at any point during execution.  Prior to
   * the first time it's called, the parameters take default values
   * as indicated below.
   *
   * @param verbosity  Default verbosity, ranging from QUDA_SILENT to
   *                   QUDA_DEBUG_VERBOSE.  Within a solver, this
   *                   parameter is overridden by the "verbosity"
   *                   member of QudaInvertParam.  The default value
   *                   is QUDA_SUMMARIZE.
   *
   * @param prefix     String to prepend to all messages from QUDA.  This
   *                   defaults to the empty string (""), but you may
   *                   wish to specify something like "QUDA: " to
   *                   distinguish QUDA's output from that of your
   *                   application.
   *
   * @param outfile    File pointer (such as stdout, stderr, or a handle
   *                   returned by fopen()) where messages should be
   *                   printed.  The default is stdout.
   */
  void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[],
                        FILE *outfile);

  /**
   * initCommsGridQuda() takes an optional "rank_from_coords" argument that
   * should be a pointer to a user-defined function with this prototype.
   *
   * @param coords  Node coordinates
   * @param fdata   Any auxiliary data needed by the function
   * @return        MPI rank or QMP node ID cooresponding to the node coordinates
   *
   * @see initCommsGridQuda
   */
  typedef int (*QudaCommsMap)(const int *coords, void *fdata);

  /**
   * @param mycomm User provided MPI communicator in place of MPI_COMM_WORLD
   */

  void qudaSetCommHandle(void *mycomm);

  /**
   * Declare the grid mapping ("logical topology" in QMP parlance)
   * used for communications in a multi-GPU grid.  This function
   * should be called prior to initQuda().  The only case in which
   * it's optional is when QMP is used for communication and the
   * logical topology has already been declared by the application.
   *
   * @param nDim   Number of grid dimensions.  "4" is the only supported
   *               value currently.
   *
   * @param dims   Array of grid dimensions.  dims[0]*dims[1]*dims[2]*dims[3]
   *               must equal the total number of MPI ranks or QMP nodes.
   *
   * @param func   Pointer to a user-supplied function that maps coordinates
   *               in the communication grid to MPI ranks (or QMP node IDs).
   *               If the pointer is NULL, the default mapping depends on
   *               whether QMP or MPI is being used for communication.  With
   *               QMP, the existing logical topology is used if it's been
   *               declared.  With MPI or as a fallback with QMP, the default
   *               ordering is lexicographical with the fourth ("t") index
   *               varying fastest.
   *
   * @param fdata  Pointer to any data required by "func" (may be NULL)
   *
   * @see QudaCommsMap
   */

  void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata);

  /**
   * Initialize the library.  This is a low-level interface that is
   * called by initQuda.  Calling initQudaDevice requires that the
   * user also call initQudaMemory before using QUDA.
   *
   * @param device CUDA device number to use.  In a multi-GPU build,
   *               this parameter may either be set explicitly on a
   *               per-process basis or set to -1 to enable a default
   *               allocation of devices to processes.
   */
  void initQudaDevice(int device);

  /**
   * Initialize the library persistant memory allocations (both host
   * and device).  This is a low-level interface that is called by
   * initQuda.  Calling initQudaMemory requires that the user has
   * previously called initQudaDevice.
   */
  void initQudaMemory();

  /**
   * Initialize the library.  This function is actually a wrapper
   * around calls to initQudaDevice() and initQudaMemory().
   *
   * @param device  CUDA device number to use.  In a multi-GPU build,
   *                this parameter may either be set explicitly on a
   *                per-process basis or set to -1 to enable a default
   *                allocation of devices to processes.
   */
  void initQuda(int device);

  /**
   * Finalize the library.
   */
  void endQuda(void);

  /**
   * @brief update the radius for halos.
   * @details This should only be needed for automated testing when
   * different partitioning is applied within a single run.
   */
  void updateR();

  /**
   * A new QudaGaugeParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   QudaGaugeParam gauge_param = newQudaGaugeParam();
   */
  QudaGaugeParam newQudaGaugeParam(void);

  /**
   * A new QudaInvertParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   QudaInvertParam invert_param = newQudaInvertParam();
   */
  QudaInvertParam newQudaInvertParam(void);

  /**
   * A new QudaMultigridParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   QudaMultigridParam mg_param = newQudaMultigridParam();
   */
  QudaMultigridParam newQudaMultigridParam(void);

  /**
   * A new QudaEigParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   QudaEigParam eig_param = newQudaEigParam();
   */
  QudaEigParam newQudaEigParam(void);

  /**
   * A new QudaGaugeObservableParam should always be initialized
   * immediately after it's defined (and prior to explicitly setting
   * its members) using this function.  Typical usage is as follows:
   *
   *   QudaGaugeObservalbeParam obs_param = newQudaGaugeObservableParam();
   */
  QudaGaugeObservableParam newQudaGaugeObservableParam(void);

  /**
   * A new QudaGaugeSmearParam should always be initialized
   * immediately after it's defined (and prior to explicitly setting
   * its members) using this function.  Typical usage is as follows:
   *
   *   QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
   */
  QudaGaugeSmearParam newQudaGaugeSmearParam(void);

  /**
   * A new QudaBLASParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   QudaBLASParam blas_param = newQudaBLASParam();
   */
  QudaBLASParam newQudaBLASParam(void);

  /**
   * Print the members of QudaGaugeParam.
   * @param param The QudaGaugeParam whose elements we are to print.
   */
  void printQudaGaugeParam(QudaGaugeParam *param);

  /**
   * Print the members of QudaInvertParam.
   * @param param The QudaInvertParam whose elements we are to print.
   */
  void printQudaInvertParam(QudaInvertParam *param);

  /**
   * Print the members of QudaMultigridParam.
   * @param param The QudaMultigridParam whose elements we are to print.
   */
  void printQudaMultigridParam(QudaMultigridParam *param);

  /**
   * Print the members of QudaEigParam.
   * @param param The QudaEigParam whose elements we are to print.
   */
  void printQudaEigParam(QudaEigParam *param);

  /**
   * Print the members of QudaGaugeObservableParam.
   * @param param The QudaGaugeObservableParam whose elements we are to print.
   */
  void printQudaGaugeObservableParam(QudaGaugeObservableParam *param);

  /**
   * Print the members of QudaBLASParam.
   * @param param The QudaBLASParam whose elements we are to print.
   */
  void printQudaBLASParam(QudaBLASParam *param);

  /**
   * Load the gauge field from the host.
   * @param h_gauge Base pointer to host gauge field (regardless of dimensionality)
   * @param param   Contains all metadata regarding host and device storage
   */
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param);

  /**
   * Free QUDA's internal copy of the gauge field.
   */
  void freeGaugeQuda(void);

  /**
   * Free a unique type (Wilson, HISQ fat, HISQ long, smeared) of internal gauge field.
   * @param link_type[in] Type of link type to free up
   */
  void freeUniqueGaugeQuda(QudaLinkType link_type);

  /**
   * Free QUDA's internal smeared gauge field.
   */
  void freeGaugeSmearedQuda(void);
  
  /**
   * Save the gauge field to the host.
   * @param h_gauge Base pointer to host gauge field (regardless of dimensionality)
   * @param param   Contains all metadata regarding host and device storage
   */
  void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param);

  /**
   * Load the clover term and/or the clover inverse from the host.
   * Either h_clover or h_clovinv may be set to NULL.
   * @param h_clover    Base pointer to host clover field
   * @param h_cloverinv Base pointer to host clover inverse field
   * @param inv_param   Contains all metadata regarding host and device storage
   */
  void loadCloverQuda(void *h_clover, void *h_clovinv,
      QudaInvertParam *inv_param);

  /**
   * Free QUDA's internal copy of the clover term and/or clover inverse.
   */
  void freeCloverQuda(void);

  /**
   * Perform the solve, according to the parameters set in param.  It
   * is assumed that the gauge field has already been loaded via
   * loadGaugeQuda().
   * @param h_x    Solution spinor field
   * @param h_b    Source spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void lanczosQuda(int k0, int m, void *hp_Apsi, void *hp_r, void *hp_V, void *hp_alpha, void *hp_beta,
                   QudaEigParam *eig_param);

  /**
   * Perform the eigensolve. The problem matrix is defined by the invert param, the
   * mode of solution is specified by the eig param. It is assumed that the gauge
   * field has already been loaded via  loadGaugeQuda().
   * @param h_evecs  Array of pointers to application eigenvectors
   * @param h_evals  Host side eigenvalues
   * @param param Contains all metadata regarding the type of solve.
   */
  void eigensolveQuda(void **h_evecs, double_complex *h_evals, QudaEigParam *param);

  /**
   * Perform the solve, according to the parameters set in param.  It
   * is assumed that the gauge field has already been loaded via
   * loadGaugeQuda().
   * @param h_x    Solution spinor field
   * @param h_b    Source spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);

  /**
   * @brief Perform the solve like @invertQuda but for multiple rhs by spliting the comm grid into
   * sub-partitions: each sub-partition invert one or more rhs'.
   * The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
   * Unlike @invertQuda, the interface also takes the host side gauge as input. The gauge pointer and
   * gauge_param are used if for inv_param split_grid[0] * split_grid[1] * split_grid[2] * split_grid[3]
   * is larger than 1, in which case gauge field is not required to be loaded beforehand; otherwise
   * this interface would just work as @invertQuda, which requires gauge field to be loaded beforehand,
   * and the gauge field pointer and gauge_param are not used.
   * @param _hp_x       Array of solution spinor fields
   * @param _hp_b       Array of source spinor fields
   * @param param       Contains all metadata regarding host and device storage and solver parameters
   * @param h_gauge     Base pointer to host gauge field (regardless of dimensionality)
   * @param gauge_param Contains all metadata regarding host and device storage for gauge field
   */
  void invertMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, void *h_gauge, QudaGaugeParam *gauge_param);

  /**
   * @brief Really the same with @invertMultiSrcQuda but for staggered-style fermions, by accepting pointers
   * to fat links and long links.
   * @param _hp_x       Array of solution spinor fields
   * @param _hp_b       Array of source spinor fields
   * @param param       Contains all metadata regarding host and device storage and solver parameters
   * @param milc_fatlinks     Base pointer to host **fat** gauge field (regardless of dimensionality)
   * @param milc_longlinks    Base pointer to host **long** gauge field (regardless of dimensionality)
   * @param gauge_param Contains all metadata regarding host and device storage for gauge field
   */
  void invertMultiSrcStaggeredQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, void *milc_fatlinks,
                                   void *milc_longlinks, QudaGaugeParam *gauge_param);

  /**
   * @brief Really the same with @invertMultiSrcQuda but for clover-style fermions, by accepting pointers
   * to direct and inverse clover field pointers.
   * @param _hp_x       Array of solution spinor fields
   * @param _hp_b       Array of source spinor fields
   * @param param       Contains all metadata regarding host and device storage and solver parameters
   * @param h_gauge     Base pointer to host gauge field (regardless of dimensionality)
   * @param gauge_param Contains all metadata regarding host and device storage for gauge field
   * @param h_clover    Base pointer to the direct clover field
   * @param h_clovinv   Base pointer to the inverse clover field
   */
  void invertMultiSrcCloverQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, void *h_gauge,
                                QudaGaugeParam *gauge_param, void *h_clover, void *h_clovinv);

  /**
   * Solve for multiple shifts (e.g., masses).
   * @param _hp_x    Array of solution spinor fields
   * @param _hp_b    Source spinor fields
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param);

  /**
   * Setup the multigrid solver, according to the parameters set in param.  It
   * is assumed that the gauge field has already been loaded via
   * loadGaugeQuda().
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void* newMultigridQuda(QudaMultigridParam *param);

  /**
   * @brief Free resources allocated by the multigrid solver
   * @param mg_instance Pointer to instance of multigrid_solver
   * @param param Contains all metadata regarding host and device
   * storage and solver parameters
   */
  void destroyMultigridQuda(void *mg_instance);

  /**
   * @brief Updates the multigrid preconditioner for the new gauge / clover field
   * @param mg_instance Pointer to instance of multigrid_solver
   * @param param Contains all metadata regarding host and device
   * storage and solver parameters, of note contains a flag specifying whether
   * to do a full update or a thin update.
   */
  void updateMultigridQuda(void *mg_instance, QudaMultigridParam *param);

  /**
   * @brief Dump the null-space vectors to disk
   * @param[in] mg_instance Pointer to the instance of multigrid_solver
   * @param[in] param Contains all metadata regarding host and device
   * storage and solver parameters (QudaMultigridParam::vec_outfile
   * sets the output filename prefix).
   */
  void dumpMultigridQuda(void *mg_instance, QudaMultigridParam *param);

  /**
   * Apply the Dslash operator (D_{eo} or D_{oe}).
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   * @param parity The destination parity of the field
   */
  void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity);

  /**
   * @brief Perform the solve like @dslashQuda but for multiple rhs by spliting the comm grid into
   * sub-partitions: each sub-partition does one or more rhs'.
   * The QudaInvertParam object specifies how the solve should be performed on each sub-partition.
   * Unlike @invertQuda, the interface also takes the host side gauge as
   * input - gauge field is not required to be loaded beforehand.
   * @param _hp_x       Array of solution spinor fields
   * @param _hp_b       Array of source spinor fields
   * @param param       Contains all metadata regarding host and device storage and solver parameters
   * @param parity      Parity to apply dslash on
   * @param h_gauge     Base pointer to host gauge field (regardless of dimensionality)
   * @param gauge_param Contains all metadata regarding host and device storage for gauge field
   */
  void dslashMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity, void *h_gauge,
                          QudaGaugeParam *gauge_param);
  /**
   * @brief Really the same with @dslashMultiSrcQuda but for staggered-style fermions, by accepting pointers
   * to fat links and long links.
   * @param _hp_x       Array of solution spinor fields
   * @param _hp_b       Array of source spinor fields
   * @param param       Contains all metadata regarding host and device storage and solver parameters
   * @param parity      Parity to apply dslash on
   * @param milc_fatlinks     Base pointer to host **fat** gauge field (regardless of dimensionality)
   * @param milc_longlinks    Base pointer to host **long** gauge field (regardless of dimensionality)
   * @param gauge_param Contains all metadata regarding host and device storage for gauge field
   */

  void dslashMultiSrcStaggeredQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity,
                                   void *milc_fatlinks, void *milc_longlinks, QudaGaugeParam *gauge_param);

  /**
   * @brief Really the same with @dslashMultiSrcQuda but for clover-style fermions, by accepting pointers
   * to direct and inverse clover field pointers.
   * @param _hp_x       Array of solution spinor fields
   * @param _hp_b       Array of source spinor fields
   * @param param       Contains all metadata regarding host and device storage and solver parameters
   * @param parity      Parity to apply dslash on
   * @param h_gauge     Base pointer to host gauge field (regardless of dimensionality)
   * @param gauge_param Contains all metadata regarding host and device storage for gauge field
   * @param h_clover    Base pointer to the direct clover field
   * @param h_clovinv   Base pointer to the inverse clover field
   */
  void dslashMultiSrcCloverQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param, QudaParity parity, void *h_gauge,
                                QudaGaugeParam *gauge_param, void *h_clover, void *h_clovinv);

  /**
   * Apply the clover operator or its inverse.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   * @param parity The source and destination parity of the field
   * @param inverse Whether to apply the inverse of the clover term
   */
  void cloverQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity, int inverse);

  /**
   * Apply the full Dslash matrix, possibly even/odd preconditioned.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   */
  void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);

  /**
   * Apply M^{\dag}M, possibly even/odd preconditioned.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   */
  void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);


  /*
   * The following routines are temporary additions used by the HISQ
   * link-fattening code.
   */

  void set_dim(int *);
  void pack_ghost(void **cpuLink, void **cpuGhost, int nFace,
      QudaPrecision precision);

  void computeKSLinkQuda(void* fatlink, void* longlink, void* ulink, void* inlink,
                         double *path_coeff, QudaGaugeParam *param);
                         
  /**
   * Compute two-link field
   *
   * @param[out] twolink computed two-link field
   * @param[in] inlink  the external field
   * @param[in] param  Contains all metadata regarding host and device
   *               storage
   */
  void computeTwoLinkQuda(void *twolink, void *inlink, QudaGaugeParam *param);

  /**
   * Either downloads and sets the resident momentum field, or uploads
   * and returns the resident momentum field
   *
   * @param[in,out] mom The external momentum field
   * @param[in] param The parameters of the external field
   */
  void momResidentQuda(void *mom, QudaGaugeParam *param);

  /**
   * Compute the gauge force and update the momentum field
   *
   * @param[in,out] mom The momentum field to be updated
   * @param[in] sitelink The gauge field from which we compute the force
   * @param[in] input_path_buf[dim][num_paths][path_length]
   * @param[in] path_length One less that the number of links in a loop (e.g., 3 for a staple)
   * @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
   * @param[in] num_paths How many contributions from path_length different "staples"
   * @param[in] max_length The maximum number of non-zero of links in any path in the action
   * @param[in] dt The integration step size (for MILC this is dt*beta/3)
   * @param[in] param The parameters of the external fields and the computation settings
   */
  int computeGaugeForceQuda(void *mom, void *sitelink, int ***input_path_buf, int *path_length, double *loop_coeff,
                            int num_paths, int max_length, double dt, QudaGaugeParam *qudaGaugeParam);

  /**
   * Compute the product of gauge links along a path and add to/overwrite the output field
   *
   * @param[in,out] out The output field to be updated
   * @param[in] sitelink The gauge field from which we compute the products of gauge links
   * @param[in] input_path_buf[dim][num_paths][path_length]
   * @param[in] path_length One less that the number of links in a loop (e.g., 3 for a staple)
   * @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
   * @param[in] num_paths How many contributions from path_length different "staples"
   * @param[in] max_length The maximum number of non-zero of links in any path in the action
   * @param[in] dt The integration step size (for MILC this is dt*beta/3)
   * @param[in] param The parameters of the external fields and the computation settings
   */
  int computeGaugePathQuda(void *out, void *sitelink, int ***input_path_buf, int *path_length, double *loop_coeff,
                           int num_paths, int max_length, double dt, QudaGaugeParam *qudaGaugeParam);

  /**
   * Compute the traces of products of gauge links along paths using the resident field
   *
   * @param[in,out] traces The computed traces
   * @param[in] sitelink The gauge field from which we compute the products of gauge links
   * @param[in] path_length The number of links in each loop
   * @param[in] loop_coeff Multiplicative coefficients for each loop
   * @param[in] num_paths Total number of loops
   * @param[in] max_length The maximum number of non-zero of links in any path in the action
   * @param[in] factor An overall normalization factor
   */
  void computeGaugeLoopTraceQuda(double_complex *traces, int **input_path_buf, int *path_length, double *loop_coeff,
                                 int num_paths, int max_length, double factor);

  /**
   * Evolve the gauge field by step size dt, using the momentum field
   * I.e., Evalulate U(t+dt) = e(dt pi) U(t)
   *
   * @param gauge The gauge field to be updated
   * @param momentum The momentum field
   * @param dt The integration step size step
   * @param conj_mom Whether to conjugate the momentum matrix
   * @param exact Whether to use an exact exponential or Taylor expand
   * @param param The parameters of the external fields and the computation settings
   */
  void updateGaugeFieldQuda(void* gauge, void* momentum, double dt,
      int conj_mom, int exact, QudaGaugeParam* param);

  /**
   * Apply the staggered phase factors to the gauge field.  If the
   * imaginary chemical potential is non-zero then the phase factor
   * exp(imu/T) will be applied to the links in the temporal
   * direction.
   *
   * @param gauge_h The gauge field
   * @param param The parameters of the gauge field
   */
  void staggeredPhaseQuda(void *gauge_h, QudaGaugeParam *param);

  /**
   * Project the input field on the SU(3) group.  If the target
   * tolerance is not met, this routine will give a runtime error.
   *
   * @param gauge_h The gauge field to be updated
   * @param tol The tolerance to which we iterate
   * @param param The parameters of the gauge field
   */
  void projectSU3Quda(void *gauge_h, double tol, QudaGaugeParam *param);

  /**
   * Evaluate the momentum contribution to the Hybrid Monte Carlo
   * action.
   *
   * @param momentum The momentum field
   * @param param The parameters of the external fields and the computation settings
   * @return momentum action
   */
  double momActionQuda(void* momentum, QudaGaugeParam* param);

  /**
   * Allocate a gauge (matrix) field on the device and optionally download a host gauge field.
   *
   * @param gauge The host gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scalar, 4 - vector, 6 - tensor)
   * @param param The parameters of the external field and the field to be created
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* createGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param);

  /**
   * Copy the QUDA gauge (matrix) field on the device to the CPU
   *
   * @param outGauge Pointer to the host gauge field
   * @param inGauge Pointer to the device gauge field (QUDA device field)
   * @param param The parameters of the host and device fields
   */
  void  saveGaugeFieldQuda(void* outGauge, void* inGauge, QudaGaugeParam* param);

  /**
   * Reinterpret gauge as a pointer to cudaGaugeField and call destructor.
   *
   * @param gauge Gauge field to be freed
   */
  void destroyGaugeFieldQuda(void* gauge);

  /**
   * Compute the clover field and its inverse from the resident gauge field.
   *
   * @param param The parameters of the clover field to create
   */
  void createCloverQuda(QudaInvertParam* param);

  /**
   * Compute the clover force contributions in each dimension mu given
   * the array of solution fields, and compute the resulting momentum
   * field.
   *
   * @param mom Force matrix
   * @param dt Integrating step size
   * @param x Array of solution vectors
   * @param p Array of intermediate vectors
   * @param coeff Array of residues for each contribution (multiplied by stepsize)
   * @param kappa2 -kappa*kappa parameter
   * @param ck -clover_coefficient * kappa / 8
   * @param nvec Number of vectors
   * @param multiplicity Number fermions this bilinear reresents
   * @param gauge Gauge Field
   * @param gauge_param Gauge field meta data
   * @param inv_param Dirac and solver meta data
   */
  void computeCloverForceQuda(void *mom, double dt, void **x, void **p, double *coeff, double kappa2, double ck,
			      int nvector, double multiplicity, void *gauge,
			      QudaGaugeParam *gauge_param, QudaInvertParam *inv_param);

  /**
   * Compute the naive staggered force.  All fields must be in the same precision.
   *
   * @param mom Momentum field
   * @param dt Integrating step size
   * @param delta Additional scale factor when updating momentum (mom += delta * [force]_TA
   * @param gauge Gauge field (at present only supports resident gauge field)
   * @param x Array of single-parity solution vectors (at present only supports resident solutions)
   * @param gauge_param Gauge field meta data
   * @param invert_param Dirac and solver meta data
   */
  void computeStaggeredForceQuda(void *mom, double dt, double delta, void *gauge, void **x, QudaGaugeParam *gauge_param,
                                 QudaInvertParam *invert_param);

  /**
   * Compute the fermion force for the HISQ quark action and integrate the momentum.
   * @param momentum        The momentum field we are integrating
   * @param dt              The stepsize used to integrate the momentum
   * @param level2_coeff    The coefficients for the second level of smearing in the quark action.
   * @param fat7_coeff      The coefficients for the first level of smearing (fat7) in the quark action.
   * @param w_link          Unitarized link variables obtained by applying fat7 smearing and unitarization to the original links.
   * @param v_link          Fat7 link variables.
   * @param u_link          SU(3) think link variables.
   * @param quark           The input fermion field.
   * @param num             The number of quark fields
   * @param num_naik        The number of naik contributions
   * @param coeff           The coefficient multiplying the fermion fields in the outer product
   * @param param.          The field parameters.
   */
  void computeHISQForceQuda(void* momentum,
                            double dt,
                            const double level2_coeff[6],
                            const double fat7_coeff[6],
                            const void* const w_link,
                            const void* const v_link,
                            const void* const u_link,
                            void** quark,
                            int num,
                            int num_naik,
                            double** coeff,
                            QudaGaugeParam* param);

  /**
     @brief Generate Gaussian distributed fields and store in the
     resident gauge field. We create a Gaussian-distributed su(n)
     field and exponentiate it, e.g., U = exp(sigma * H), where H is
     the distributed su(n) field and sigma is the width of the
     distribution (sigma = 0 results in a free field, and sigma = 1 has
     maximum disorder).

     @param seed The seed used for the RNG
     @param sigma Width of Gaussian distrubution
  */
  void gaussGaugeQuda(unsigned long long seed, double sigma);

  /**
   * @brief Generate Gaussian distributed fields and store in the
   * resident momentum field. We create a Gaussian-distributed su(n)
   * field, e.g., sigma * H, where H is the distributed su(n) field
   * and sigma is the width of the distribution (sigma = 0 results
   * in a free field, and sigma = 1 has maximum disorder).
   *
   * @param seed The seed used for the RNG
   * @param sigma Width of Gaussian distrubution
   */
  void gaussMomQuda(unsigned long long seed, double sigma);

  /**
   * Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
   * @param[out] Array for storing the averages (total, spatial, temporal)
   */
  void plaqQuda(double plaq[3]);

  /**
     @brief Computes the trace of the Polyakov loop of the current resident field
     in a given direction.

     @param[out] ploop Trace of the Polyakov loop in direction dir
     @param[in] dir Direction of Polyakov loop
  */
  void polyakovLoopQuda(double ploop[2], int dir);

  /**
   * Performs a deep copy from the internal extendedGaugeResident field.
   * @param Pointer to externally allocated GaugeField
   */
  void copyExtendedResidentGaugeQuda(void *resident_gauge);

  /**
   * Performs Wuppertal smearing on a given spinor using the gauge field
   * gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage and operator which will be applied to the spinor
   * @param n_steps Number of steps to apply.
   * @param alpha  Alpha coefficient for Wuppertal smearing.
   */
  void performWuppertalnStep(void *h_out, void *h_in, QudaInvertParam *param, unsigned int n_steps, double alpha);

  /**
   * Performs APE, Stout, or Over Imroved STOUT smearing on gaugePrecise and stores it in gaugeSmeared
   * @param[in] smear_param Parameter struct that defines the computation parameters
   * @param[in,out] obs_param Parameter struct that defines which
   * observables we are making and the resulting observables.
   */
  void performGaugeSmearQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param);

  /**
   * Performs Wilson Flow on gaugePrecise and stores it in gaugeSmeared
   * @param[in] smear_param Parameter struct that defines the computation parameters
   * @param[in,out] obs_param Parameter struct that defines which
   * observables we are making and the resulting observables.
   */
  void performWFlowQuda(QudaGaugeSmearParam *smear_param, QudaGaugeObservableParam *obs_param);

  /**
   * @brief Calculates a variety of gauge-field observables.  If a
   * smeared gauge field is presently loaded (in gaugeSmeared) the
   * observables are computed on this, else the resident gauge field
   * will be used.
   * @param[in,out] param Parameter struct that defines which
   * observables we are making and the resulting observables.
   */
  void gaugeObservablesQuda(QudaGaugeObservableParam *param);

  /**
   * Public function to perform color contractions of the host spinors x and y.
   * @param[in] x pointer to host data
   * @param[in] y pointer to host data
   * @param[out] result pointer to the 16 spin projections per lattice site
   * @param[in] cType Which type of contraction (open, degrand-rossi, etc)
   * @param[in] param meta data for construction of ColorSpinorFields.
   * @param[in] X spacetime data for construction of ColorSpinorFields.
   */
  void contractQuda(const void *x, const void *y, void *result, const QudaContractType cType, QudaInvertParam *param,
                    const int *X);

  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in,out] gauge, gauge field to be fixed
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when
   * iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
   * @param[in] stopWtheta, 0 for MILC criterion and 1 to use the theta value
   * @param[in] param The parameters of the external fields and the computation settings
   * @param[out] timeinfo
   */
  int computeGaugeFixingOVRQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                                const unsigned int verbose_interval, const double relax_boost, const double tolerance,
                                const unsigned int reunit_interval, const unsigned int stopWtheta,
                                QudaGaugeParam *param, double *timeinfo);

  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in,out] gauge, gauge field to be fixed
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
   * @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when
   * iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] stopWtheta, 0 for MILC criterion and 1 to use the theta value
   * @param[in] param The parameters of the external fields and the computation settings
   * @param[out] timeinfo
   */
  int computeGaugeFixingFFTQuda(void *gauge, const unsigned int gauge_dir, const unsigned int Nsteps,
                                const unsigned int verbose_interval, const double alpha, const unsigned int autotune,
                                const double tolerance, const unsigned int stopWtheta, QudaGaugeParam *param,
                                double *timeinfo);

  /**
   * @brief Strided Batched GEMM
   * @param[in] arrayA The array containing the A matrix data
   * @param[in] arrayB The array containing the B matrix data
   * @param[in] arrayC The array containing the C matrix data
   * @param[in] native Boolean to use either the native or generic version
   * @param[in] param The data defining the problem execution.
   */
  void blasGEMMQuda(void *arrayA, void *arrayB, void *arrayC, QudaBoolean native, QudaBLASParam *param);

  /**
   * @brief Strided Batched in-place matrix inversion via LU
   * @param[in] Ainv The array containing the A inverse matrix data
   * @param[in] A The array containing the A matrix data
   * @param[in] use_native Boolean to use either the native or generic version
   * @param[in] param The data defining the problem execution.
   */
  void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *param);

  /**
   * @brief Flush the chronological history for the given index
   * @param[in] index Index for which we are flushing
   */
  void flushChronoQuda(int index);


  /**
  * Create deflation solver resources.
  *
  **/

  void* newDeflationQuda(QudaEigParam *param);

  /**
   * Free resources allocated by the deflated solver
   */
  void destroyDeflationQuda(void *df_instance);

  void setMPICommHandleQuda(void *mycomm);
  
  // Parameter set for quark smearing operations
  typedef struct QudaQuarkSmearParam_s {
    //-------------------------------------------------
    /** Used to store information pertinent to the operator **/
    QudaInvertParam *inv_param;

    /** Number of steps to apply **/
    int  n_steps;
    /** The width of the Gaussian **/
    double  width;
    /** if nonzero then compute two-link, otherwise reuse gaugeSmeared**/
    int compute_2link;
    /** if nonzero then delete two-link, otherwise keep two-link for future use**/
    int delete_2link;
    /** Set if the input spinor is on a time slice **/
    int t0;
    /** Flops count for the smearing operations **/
    int gflops;
    
  } QudaQuarkSmearParam;

  /**
   * Performs two-link Gaussian smearing on a given spinor (for staggered fermions).
   * @param[in,out] h_in Input spinor field to smear
   * @param[in] smear_param   Contains all metadata the operator which will be applied to the spinor
   */
  void performTwoLinkGaussianSmearNStep(void *h_in, QudaQuarkSmearParam *smear_param);

#ifdef __cplusplus
}
#endif

// remove NVRTC WAR
#undef double_complex

/* #include <quda_new_interface.h> */

