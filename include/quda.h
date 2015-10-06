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
#include <quda_constants.h>

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

    QudaPrecision cuda_prec_precondition; /**< The precision of the preconditioner gauge field */
    QudaReconstructType reconstruct_precondition; /**< The recontruction type of the preconditioner gauge field */

    QudaGaugeFixed gauge_fix; /**< Whether the input gauge field is in the axial gauge or not */

    int ga_pad;       /**< The pad size that the cudaGaugeField will use (default=0) */ 

    int site_ga_pad;  /**< Used by link fattening and the gauge and fermion forces */

    int staple_pad;   /**< Used by link fattening */
    int llfat_ga_pad; /**< Used by link fattening */
    int mom_ga_pad;   /**< Used by the gauge and fermion forces */
    double gaugeGiB;  /**< The storage used by the gauge fields */

    int preserve_gauge; /**< Used by link fattening */

    QudaStaggeredPhase staggered_phase_type; /**< Set the staggered phase type of the links */
    int staggered_phase_applied; /**< Whether the staggered phase has already been applied to the links */

    double i_mu; /**< Imaginary chemical potential */

    int overlap; /**< Width of overlapping domains */

    int use_resident_gauge;  /**< Use the resident gauge field */
    int use_resident_mom;    /**< Use the resident mom field */
    int make_resident_gauge; /**< Make the gauge field resident */
    int make_resident_mom;   /**< Make the mom field resident */
    int return_gauge;        /**< Return the new gauge field */
    int return_mom;          /**< Return the new mom field */

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

    double b_5[QUDA_MAX_DWF_LS];  /**< MDWF coefficients */
    double c_5[QUDA_MAX_DWF_LS];  /**< will be used only for the mobius type of Fermion */

    double mu;    /**< Twisted mass parameter */
    double epsilon; /**< Twisted mass parameter */

    QudaTwistFlavorType twist_flavor;  /**< Twisted mass flavor */

    double tol;    /**< Solver tolerance in the L2 residual norm */
    double tol_restart;   /**< Solver tolerance in the L2 residual norm (used to restart InitCG) */
    double tol_hq; /**< Solver tolerance in the heavy quark residual norm */
    double true_res; /**< Actual L2 residual norm achieved in solver */
    double true_res_hq; /**< Actual heavy quark residual norm achieved in solver */
    int maxiter; /**< Maximum number of iterations in the linear solver */
    double reliable_delta; /**< Reliable update tolerance */
    int use_sloppy_partial_accumulator; /**< Whether to keep the partial solution accumuator in sloppy precision */

    /**< This parameter determines how many consective reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase;

    /**< This parameter determines how many total reliable update
    residual increases we tolerate before terminating the solver,
    i.e., how long do we want to keep trying to converge */
    int max_res_increase_total;

    /**< After how many iterations shall the heavy quark residual be updated */
    int heavy_quark_check;

    int pipeline; /**< Whether to use a pipelined solver with less global sums */

    int num_offset; /**< Number of offsets in the multi-shift solver */

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
    QudaPrecision cuda_prec_precondition;  /**< The precision used by the QUDA preconditioner */

    QudaDiracFieldOrder dirac_order;       /**< The order of the input and output fermion fields */

    QudaGammaBasis gamma_basis;            /**< Gamma basis of the input and output host fields */

    QudaFieldLocation clover_location;            /**< The location of the clover field */
    QudaPrecision clover_cpu_prec;         /**< The precision used for the input clover field */
    QudaPrecision clover_cuda_prec;        /**< The precision used for the clover field in the QUDA solver */
    QudaPrecision clover_cuda_prec_sloppy; /**< The precision used for the clover field in the QUDA sloppy operator */
    QudaPrecision clover_cuda_prec_precondition; /**< The precision used for the clover field in the QUDA preconditioner */

    QudaCloverFieldOrder clover_order;     /**< The order of the input clover field */
    QudaUseInitGuess use_init_guess;       /**< Whether to use an initial guess in the solver or not */

    double clover_coeff;                   /**< Coefficient of the clover term */

    int compute_clover_trlog;              /**< Whether to compute the trace log of the clover term */
    double trlogA[2];                      /**< The trace log of the clover term (even/odd computed separately) */

    QudaVerbosity verbosity;               /**< The verbosity setting to use in the solver */

    int sp_pad;                            /**< The padding to use for the fermion fields */
    int cl_pad;                            /**< The padding to use for the clover fields */

    int iter;                              /**< The number of iterations performed by the solver */
    double spinorGiB;                      /**< The memory footprint of the fermion fields */
    double cloverGiB;                      /**< The memory footprint of the clover fields */
    double gflops;                         /**< The Gflops rate of the solver */
    double secs;                           /**< The time taken by the solver */

    QudaTune tune;                          /**< Enable auto-tuning? (default = QUDA_TUNE_YES) */


    /** Number of steps in s-step algorithms */
    int Nsteps; 

    /** Maximum size of Krylov space used by solver */
    int gcrNkrylov;

    /*
     * The following parameters are related to the domain-decomposed
     * preconditioner, if enabled.
     */

    /**
     * The inner Krylov solver used in the preconditioner.  Set to
     * QUDA_INVALID_INVERTER to disable the preconditioner entirely.
     */
    QudaInverterType inv_type_precondition;

    /**
      Dirac Dslash used in preconditioner
    */
    QudaDslashType dslash_type_precondition;
    /** Verbosity of the inner Krylov solver */
    QudaVerbosity verbosity_precondition;

    /** Tolerance in the inner solver */
    double tol_precondition;

    /** Maximum number of iterations allowed in the inner solver */
    int maxiter_precondition;

    /** Relaxation parameter used in GCR-DD (default = 1.0) */
    double omega;

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
    QudaPrecision cuda_prec_ritz; /**< The precision of the Ritz vectors */

    int nev;

    int max_search_dim;//for magma library this parameter must be multiple 16?

    int rhs_idx;

    int deflation_grid;//total deflation space is nev*deflation_grid

    /** Whether to make the solution vector(s) after the solve */
    int make_resident_solution;

    /** Whether to use the resident solution vector(s) */
    int use_resident_solution;

  } QudaInvertParam;


  // Parameter set for solving the eigenvalue problems.
  // Eigen problems are tightly related with Ritz algorithm.
  // And the Lanczos algorithm use the Ritz operator.
  // For Ritz matrix operation, 
  // we need to know about the solution type of dirac operator.
  // For acceleration, we are also using chevisov polynomial method.
  // And nk, np values are needed Implicit Restart Lanczos method
  // which is optimized form of Lanczos algorithm
  typedef struct QudaEigParam_s {

    QudaInvertParam *invert_param;
    QudaSolutionType  RitzMat_lanczos;
    QudaSolutionType  RitzMat_Convcheck;
    QudaEigType eig_type;

    double *MatPoly_param;
    int NPoly;
    double Stp_residual;
    int nk;
    int np;
    int f_size;
    double eigen_shift;

  } QudaEigParam;

    

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
   * Print the members of QudaGaugeParam.
   * @param param The QudaGaugeParam whose elements we are to print.
   */
  void printQudaInvertParam(QudaInvertParam *param);

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
  void lanczosQuda(int k0, int m, void *hp_Apsi, void *hp_r, void *hp_V, 
                   void *hp_alpha, void *hp_beta, QudaEigParam *eig_param);

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
   * Solve for multiple shifts (e.g., masses).
   * @param _hp_x    Array of solution spinor fields
   * @param _hp_b    Array of source spinor fields
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param);

  /**
   * Deflated solvers interface (e.g., based on invremental deflation space constructors, like incremental eigCG).
   * @param _h_x    Outnput: array of solution spinor fields (typically O(10))
   * @param _h_b    Input: array of source spinor fields (typically O(10))
   * @param _h_u    Input/Output: array of Ritz spinor fields (typically O(100))
   * @param _h_h    Input/Output: complex projection mutirx (typically O(100))
   * @param param  Contains all metadata regarding host and device
   *               storage and solver parameters
   */
  void incrementalEigQuda(void *_h_x, void *_h_b, QudaInvertParam *param, void *_h_u, double *inv_eigenvals, int last_rhs);

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
   * Apply the Dslash operator (D_{eo} or D_{oe}) for 4D EO preconditioned DWF.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   * @param parity The destination parity of the field
   * @param test_type Choose a type of dslash operators 
   */
  void dslashQuda_4dpc(void *h_out, void *h_in, QudaInvertParam *inv_param,
      QudaParity parity, int test_type);
  
  /**
   * Apply the Dslash operator (D_{eo} or D_{oe}) for Mobius DWF.
   * @param h_out  Result spinor field
   * @param h_in   Input spinor field
   * @param param  Contains all metadata regarding host and device
   *               storage
   * @param parity The destination parity of the field
   * @param test_type Choose a type of dslash operators 
   */
  void dslashQuda_mdwf(void *h_out, void *h_in, QudaInvertParam *inv_param,
      QudaParity parity, int test_type);

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
  void setFatLinkPadding(QudaComputeFatMethod method, QudaGaugeParam* param);

  void computeKSLinkQuda(void* fatlink, void* longlink, void* ulink, void* inlink, 
                         double *path_coeff, QudaGaugeParam *param, QudaComputeFatMethod method);



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
   * action.  The momentum field is assumed to be in MILC order.
   *
   * @param momentum The momentum field
   * @param param The parameters of the external fields and the computation settings
   * @return momentum action
   */
  double momActionQuda(void* momentum, QudaGaugeParam* param);

  /**
   * Take a gauge field on the host, load it onto the device and extend it.
   * Return a pointer to the extended gauge field object.
   *
   * @param gauge The CPU gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param param The parameters of the external field and the field to be created
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* createExtendedGaugeFieldQuda(void* gauge, int geometry, QudaGaugeParam* param);

  /**
   * Allocate a gauge (matrix) field on the device and optionally download a host gauge field.
   *
   * @param gauge The host gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
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
   * Take a gauge field on the device and copy to the extended gauge
   * field.  The precisions and reconstruct types can differ between
   * the input and output field, but they must be compatible (same volume, geometry).
   *
   * @param outGauge Pointer to the output extended device gauge field (QUDA extended device field)
   * @param inGauge Pointer to the input device gauge field (QUDA gauge field)
   */
  void  extendGaugeFieldQuda(void* outGauge, void* inGauge);

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
   * Compute the sigma trace field (part of clover force computation).
   * All the pointers here are for QUDA native device objects.  The
   * precisions of all fields must match.  This function requires that
   * there is a persistent clover field.
   * 
   * @param out Sigma trace field  (QUDA device field, geometry = 1)
   * @param dummy (not used)
   * @param mu mu direction
   * @param nu nu direction
   * @param dim array of local field dimensions
   */
  void computeCloverTraceQuda(void* out, void* dummy, int mu, int nu, int dim[4]);

  /**
   * Compute the derivative of the clover term (part of clover force
   * computation).  All the pointers here are for QUDA native device
   * objects.  The precisions of all fields must match.
   * 
   * @param out Clover derivative field (QUDA device field, geometry = 1)
   * @param gauge Gauge field (extended QUDA device field, gemoetry = 4)
   * @param oprod Matrix field (outer product) which is multiplied by the derivative
   * @param mu mu direction
   * @param nu nu direction
   * @param coeff Coefficient of the clover derviative (including stepsize and clover coefficient)
   * @param parity Parity for which we are computing
   * @param param Gauge field meta data
   * @param conjugate Whether to make the oprod field anti-hermitian prior to multiplication
   */
  void computeCloverDerivativeQuda(void* out, void* gauge, void* oprod, int mu, int nu,
				   double coeff,
				   QudaParity parity, QudaGaugeParam* param, int conjugate);
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
   * Compute the quark-field outer product needed for gauge generation
   *  
   * @param oprod The outer product to be computed.
   * @param quark The input fermion field.
   * @param num The number of quark fields
   * @param coeff The coefficient multiplying the fermion fields in the outer product
   * @param param The parameters of the outer-product field.
   */
  void computeStaggeredOprodQuda(void** oprod, void** quark, int num, double** coeff, QudaGaugeParam* param);

  /**
   * Compute the naive staggered force (experimental).  All fields are
   * QUDA device fields and must be in the same precision.
   *
   * mom Momentum field (QUDA device field)
   * quark Quark field solution vectors
   * coeff Step-size coefficient
   */
  void computeStaggeredForceQuda(void* mom, void* quark, double* coeff);

  /**
   * Compute the fermion force for the asqtad quark action. 
   * @param momentum          The momentum contribution from the quark action.
   * @param act_path_coeff    The coefficients that define the asqtad action.
   * @param one_link_src      The quark field outer product corresponding to the one-link term in the action. 
   * @param naik_src          The quark field outer product corresponding to the naik term in the action.
   * @param link              The gauge field.
   * @param param             The field parameters.
   */
  void computeAsqtadForceQuda(void* const momentum,
	long long* flops,
        const double act_path_coeff[6],
        const void* const one_link_src[4],
        const void* const naik_src[4],
        const void* const link,
        const QudaGaugeParam* param);


  /**
   * Compute the fermion force for the HISQ quark action. 
   * @param momentum        The momentum contribution from the quark action.
   * @param level2_coeff    The coefficients for the second level of smearing in the quark action.
   * @param fat7_coeff      The coefficients for the first level of smearing (fat7) in the quark action.
   * @param staple_src      Quark outer-product for the staple.
   * @param one_link_src    Quark outer-product for the one-link term in the action.
   * @param naik_src        Quark outer-product for the three-hop term in the action.
   * @param w_link          Unitarized link variables obtained by applying fat7 smearing and unitarization to the original links.
   * @param v_link          Fat7 link variables. 
   * @param u_link          SU(3) think link variables. 
   * @param param.          The field parameters.
   */

  void computeHISQForceQuda(void* momentum,
    long long* flops,
    const double level2_coeff[6],
    const double fat7_coeff[6],
    const void* const staple_src[4],
    const void* const one_link_src[4],
    const void* const naik_src[4],
    const void* const w_link,
    const void* const v_link,
    const void* const u_link,
    const QudaGaugeParam* param);



  void computeHISQForceCompleteQuda(void* momentum,
                      const double level2_coeff[6],
                      const double fat7_coeff[6],
                      void** quark_array,
                      int num_terms,
                      double** quark_coeff,
                      const void* const w_link,
                      const void* const v_link,
                      const void* const u_link,
                      const QudaGaugeParam* param);

  /**
   * Computes the total, spatial and temporal plaquette averages of the loaded gauge configuration.
   * @param Array for storing the averages (total, spatial, temporal)
   */
  void plaqQuda(double plaq[3]);

  /**
   * Performs APE smearing on gaugePrecise and stores it in gaugeSmeared
   * @param nSteps Number of steps to apply.
   * @param alpha  Alpha coefficient for APE smearing.
   */
  void performAPEnStep(unsigned int nSteps, double alpha);

  /**
   * Calculates the topological charge from gaugeSmeared, if it exist, or from gaugePrecise if no smeared fields are present.
   */
  double qChargeCuda();

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
  * Open/Close MAGMA library
  *
  **/
  void openMagma();

  void closeMagma();

#ifdef __cplusplus
}
#endif

/* #include <quda_new_interface.h> */

#endif /* _QUDA_H */
