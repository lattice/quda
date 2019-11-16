#ifndef _QUDA_H
#define _QUDA_H

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

    int overwrite_mom; /**< When computing momentum, should we overwrite it or accumulate to to */

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

    double mu;    /**< Twisted mass parameter */
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

    QudaDiracFieldOrder dirac_order;       /**< The order of the input and output fermion fields */

    QudaGammaBasis gamma_basis;            /**< Gamma basis of the input and output host fields */

    QudaFieldLocation clover_location;            /**< The location of the clover field */
    QudaPrecision clover_cpu_prec;         /**< The precision used for the input clover field */
    QudaPrecision clover_cuda_prec;        /**< The precision used for the clover field in the QUDA solver */
    QudaPrecision clover_cuda_prec_sloppy; /**< The precision used for the clover field in the QUDA sloppy operator */
    QudaPrecision clover_cuda_prec_refinement_sloppy; /**< The precision of the sloppy clover field for the refinement step in multishift */
    QudaPrecision clover_cuda_prec_precondition; /**< The precision used for the clover field in the QUDA preconditioner */

    QudaCloverFieldOrder clover_order;     /**< The order of the input clover field */
    QudaUseInitGuess use_init_guess;       /**< Whether to use an initial guess in the solver or not */

    double clover_coeff;                   /**< Coefficient of the clover term */
    double clover_rho;                     /**< Real number added to the clover diagonal (not to inverse) */

    int compute_clover_trlog;              /**< Whether to compute the trace log of the clover term */
    double trlogA[2];                      /**< The trace log of the clover term (even/odd computed separately) */

    int compute_clover;                    /**< Whether to compute the clover field */
    int compute_clover_inverse;            /**< Whether to compute the clover inverse field */
    int return_clover;                     /**< Whether to copy back the clover matrix field */
    int return_clover_inverse;             /**< Whether to copy back the inverted clover matrix field */

    QudaVerbosity verbosity;               /**< The verbosity setting to use in the solver */

    int sp_pad;                            /**< The padding to use for the fermion fields */
    int cl_pad;                            /**< The padding to use for the clover fields */

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

    /** Number of preconditioner cycles to perform per iteration */
    int precondition_cycle;

    /** Whether to use additive or multiplicative Schwarz preconditioning */
    QudaSchwarzType schwarz_type;

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
    int nev;
    /** EeigCG  : Search space dimension
     *  gmresdr : Krylov subspace dimension
    */
    int max_search_dim;
    /** For systems with many RHS: current RHS index */
    int rhs_idx;
    /** Specifies deflation space volume: total number of eigenvectors is nev*deflation_grid */
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

    /** Which external library to use in the linear solvers (MAGMA or Eigen) */
    QudaExtLibType extlib_type;

  } QudaInvertParam;

  // Parameter set for solving eigenvalue problems.
  typedef struct QudaEigParam_s {

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
    QudaBoolean use_dagger;
    QudaBoolean use_norm_op;

    /** Performs an MdagM solve, then constructs the left and right SVD. **/
    QudaBoolean compute_svd;

    /** If true, the solver will error out if the convergence criteria are not met **/
    QudaBoolean require_convergence;

    /** Which part of the spectrum to solve **/
    QudaEigSpectrumType spectrum;

    /** Size of the eigenvector search space **/
    int nEv;
    /** Total size of Krylov space **/
    int nKr;
    /** Max number of locked eigenpairs (deduced at runtime) **/
    int nLockedMax;
    /** Number of requested converged eigenvectors **/
    int nConv;
    /** Tolerance on the least well known eigenvalue's residual **/
    double tol;
    /** For IRLM/IRAM, check every nth restart **/
    int check_interval;
    /** For IRLM/IRAM, quit after n restarts **/
    int max_restarts;
    /** For the Ritz rotation, the maximal number of extra vectors the solver may allocate **/
    int batched_rotate;

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

    /** The Gflops rate of the eigensolver setup */
    double gflops;

    /**< The time taken by the eigensolver setup */
    double secs;

    /** Which external library to use in the deflation operations (MAGMA or Eigen) */
    QudaExtLibType extlib_type;
    //-------------------------------------------------

  } QudaEigParam;

  typedef struct QudaMultigridParam_s {

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

    /** Verbosity on each level of the multigrid */
    QudaVerbosity verbosity[QUDA_MAX_MG_LEVEL];

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

    /** Basis to use for CA-CGN(E/R) setup */
    QudaCABasis setup_ca_basis[QUDA_MAX_MG_LEVEL];

    /** Basis size for CACG setup */
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

    /** Basis to use for CA-CGN(E/R) coarse solver */
    QudaCABasis coarse_solver_ca_basis[QUDA_MAX_MG_LEVEL];

    /** Basis size for CACG coarse solver */
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

  } QudaMultigridParam;



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
   * Perform the solve like @invertQuda but for multiples right hand sides.
   *
   * @param _hp_x    Array of solution spinor fields
   * @param _hp_b    Array of source spinor fields
   * @param param  Contains all metadata regarding
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void invertMultiSrcQuda(void **_hp_x, void **_hp_b, QudaInvertParam *param);

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
   * storage and solver parameters
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
  void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param,
      QudaParity parity);

  /**
   * Apply the clover operator or its inverse.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   * @param parity The source and destination parity of the field
   * @param inverse Whether to apply the inverse of the clover term
   */
  void cloverQuda(void *h_out, void *h_in, QudaInvertParam *inv_param,
      QudaParity *parity, int inverse);

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
   * Compute the gauge force and update the mometum field
   *
   * @param mom The momentum field to be updated
   * @param sitelink The gauge field from which we compute the force
   * @param input_path_buf[dim][num_paths][path_length]
   * @param path_length One less that the number of links in a loop (e.g., 3 for a staple)
   * @param loop_coeff Coefficients of the different loops in the Symanzik action
   * @param num_paths How many contributions from path_length different "staples"
   * @param max_length The maximum number of non-zero of links in any path in the action
   * @param dt The integration step size (for MILC this is dt*beta/3)
   * @param param The parameters of the external fields and the computation settings
   */
  int computeGaugeForceQuda(void* mom, void* sitelink,  int*** input_path_buf, int* path_length,
			    double* loop_coeff, int num_paths, int max_length, double dt,
			    QudaGaugeParam* qudaGaugeParam);

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
  void computeStaggeredForceQuda(void* mom, double dt, double delta, void **x, void *gauge,
				 QudaGaugeParam *gauge_param, QudaInvertParam *invert_param);

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
     resident gauge field.  We create a Gaussian-distributed su(n)
     field and exponentiate it, e.g., U = exp(sigma * H), where H is
     the distributed su(n) field and beta is the width of the
     distribution (beta = 0 results in a free field, and sigma = 1 has
     maximum disorder).

     @param seed The seed used for the RNG
     @param sigma Width of Gaussian distrubution
  */
  void gaussGaugeQuda(unsigned long long seed, double sigma);

  /**
   * Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
   * @param Array for storing the averages (total, spatial, temporal)
   */
  void plaqQuda(double plaq[3]);

  /*
   * Performs a deep copy from the internal extendedGaugeResident field.
   * @param Pointer to externalGaugeResident cudaGaugeField
   * @param Location of gauge field
   */
  void copyExtendedResidentGaugeQuda(void* resident_gauge, QudaFieldLocation loc);

  /**
   * Performs Wuppertal smearing on a given spinor using the gauge field
   * gaugeSmeared, if it exist, or gaugePrecise if no smeared field is present.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage and operator which will be applied to the spinor
   * @param nSteps Number of steps to apply.
   * @param alpha  Alpha coefficient for Wuppertal smearing.
   */
  void performWuppertalnStep(void *h_out, void *h_in, QudaInvertParam *param, unsigned int nSteps, double alpha);

  /**
   * Performs APE smearing on gaugePrecise and stores it in gaugeSmeared
   * @param nSteps Number of steps to apply.
   * @param alpha  Alpha coefficient for APE smearing.
   */
  void performAPEnStep(unsigned int nSteps, double alpha);

  /**
   * Performs STOUT smearing on gaugePrecise and stores it in gaugeSmeared
   * @param nSteps Number of steps to apply.
   * @param rho    Rho coefficient for STOUT smearing.
   */
  void performSTOUTnStep(unsigned int nSteps, double rho);

  /**
   * Performs Over Imroved STOUT smearing on gaugePrecise and stores it in gaugeSmeared
   * @param nSteps Number of steps to apply.
   * @param rho    Rho coefficient for STOUT smearing.
   * @param epsilon Epsilon coefficient for Over Improved STOUT smearing.
   */
  void performOvrImpSTOUTnStep(unsigned int nSteps, double rho, double epsilon);

  /**
   * Calculates the topological charge from gaugeSmeared, if it exist, or from gaugePrecise if no smeared fields are present.
   */
  double qChargeQuda();

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
     @brief Calculates the topological charge from gaugeSmeared, if it exist,
     or from gaugePrecise if no smeared fields are present.
     @param[out] qDensity array holding Q charge density
  */
  double qChargeDensityQuda(void *qDensity);

  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in,out] gauge, gauge field to be fixed
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   * @param[in] param The parameters of the external fields and the computation settings
   * @param[out] timeinfo
   */
  int computeGaugeFixingOVRQuda(void* gauge,
                      const unsigned int gauge_dir,
                      const unsigned int Nsteps,
                      const unsigned int verbose_interval,
                      const double relax_boost,
                      const double tolerance,
                      const unsigned int reunit_interval,
                      const unsigned int stopWtheta,
                      QudaGaugeParam* param,
                      double* timeinfo);
  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in,out] gauge, gauge field to be fixed
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
   * @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   * @param[in] param The parameters of the external fields and the computation settings
   * @param[out] timeinfo
   */
  int computeGaugeFixingFFTQuda(void* gauge,
                      const unsigned int gauge_dir,
                      const unsigned int Nsteps,
                      const unsigned int verbose_interval,
                      const double alpha,
                      const unsigned int autotune,
                      const double tolerance,
                      const unsigned int stopWtheta,
                      QudaGaugeParam* param,
                      double* timeinfo);

  /**
   * @brief Flush the chronological history for the given index
   * @param[in] index Index for which we are flushing
   */
  void flushChronoQuda(int index);


  /**
  * Open/Close MAGMA library
  *
  **/
  void openMagma();

  void closeMagma();

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

#ifdef __cplusplus
}
#endif

// remove NVRTC WAR
#undef double_complex

/* #include <quda_new_interface.h> */

#endif /* _QUDA_H */
