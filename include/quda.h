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

#define QUDA_VERSION_MAJOR     0
#define QUDA_VERSION_MINOR     5
#define QUDA_VERSION_SUBMINOR  0

/**
 * @def   QUDA_VERSION
 * @brief This macro is deprecated.  Use QUDA_VERSION_MAJOR, etc., instead.
 */
#define QUDA_VERSION ((QUDA_VERSION_MAJOR<<16) | (QUDA_VERSION_MINOR<<8) | QUDA_VERSION_SUBMINOR)


/**
 * @def   QUDA_MAX_DIM
 * @brief Maximum number of dimensions supported by QUDA.  In practice, no
 *        routines make use of more than 5.
 */
#define QUDA_MAX_DIM 5

/**
 * @def QUDA_MAX_MULTI_SHIFT
 * @brief Maximum number of shifts supported by the multi-shift solver.
 *        This number may be changed if need be.
 */
#define QUDA_MAX_MULTI_SHIFT 32


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

    double mu;    /**< Twisted mass parameter */
    double epsilon; /**< Twisted mass parameter */
    
    QudaTwistFlavorType twist_flavor;  /**< Twisted mass flavor */

    double tol;   /**< Solver tolerance in the L2 residual norm */
    double tol_hq; /**< Solver tolerance in the heavy quark residual norm */
    double true_res; /**< Actual L2 residual norm achieved in solver */
    double true_res_hq; /**< Actual heavy quark residual norm achieved in solver */
    int maxiter; /**< Maximum number of iterations in the linear solver */
    double reliable_delta; /**< Reliable update tolerance */

    int pipeline; /**< Whether to use a pipelined solver with less global sums */

    int num_offset; /**< Number of offsets in the multi-shift solver */

    /** Offsets for multi-shift solver */
    double offset[QUDA_MAX_MULTI_SHIFT];

    /** Solver tolerance for each offset */
    double tol_offset[QUDA_MAX_MULTI_SHIFT];     

    /** Solver tolerance for each shift when refinement is applied using the heavy-quark residual */
    double tol_hq_offset[QUDA_MAX_MULTI_SHIFT];

    /** Actual L2 residual norm achieved in solver for each offset */
    double true_res_offset[QUDA_MAX_MULTI_SHIFT]; 

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

    QudaVerbosity verbosity;               /**< The verbosity setting to use in the solver */

    int sp_pad;                            /**< The padding to use for the fermion fields */
    int cl_pad;                            /**< The padding to use for the clover fields */

    int iter;                              /**< The number of iterations performed by the solver */
    double spinorGiB;                      /**< The memory footprint of the fermion fields */
    double cloverGiB;                      /**< The memory footprint of the clover fields */
    double gflops;                         /**< The Gflops rate of the solver */
    double secs;                           /**< The time taken by the solver */

    QudaTune tune;                          /**< Enable auto-tuning? (default = QUDA_TUNE_YES) */

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

  } QudaInvertParam;


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
  void setFatLinkPadding(QudaComputeFatMethod method, QudaGaugeParam* param);
  int computeKSLinkQuda(void* fatlink, void* longlink, void** sitelink,
			     double* act_path_coeff, QudaGaugeParam* param, 
			     QudaComputeFatMethod method);

  /**
   * Compute the gauge force.
   */
  int computeGaugeForceQuda(void* mom, void* sitelink,  int*** input_path_buf, int* path_length,
			    void* loop_coeff, int num_paths, int max_length, double eb3,
			    QudaGaugeParam* qudaGaugeParam, double* timeinfo);

  /**
   * Evolve the gauge field by step size dt, using the momentum field
   * I.e., Evalulate U(t+dt) = e(dt pi) U(t) 
   *
   * @param gauge The gauge field to be updated 
   * @param momentum The momentum field
   * @param dt The integration step size step
   * @param param The parameters of the external fields and the computation settings
   */
  void updateGaugeFieldQuda(void* gauge, void* momentum, double dt, QudaGaugeParam* param);

#ifdef __cplusplus
}
#endif

#include <quda_fortran.h>
/* #include <quda_new_interface.h> */

#endif /* _QUDA_H */
