#ifndef _QUDA_MILC_INTERFACE_H
#define _QUDA_MILC_INTERFACE_H

#include <enum_quda.h>
#include <quda.h>

/**
 * @file    quda_milc_interface.h
 *
 * @section Description
 *
 * The header file defines the milc interface to enable easy
 * interfacing between QUDA and the MILC software packed.
 */
#if __COMPUTE_CAPABILITY__ >= 600
#define USE_QUDA_MANAGED 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /**
   * Parameters related to MILC site struct
   */
  typedef struct {
    void *site; /** Pointer to beginning of site array */
    void *link; /** Pointer to link field (only used if site is not set) */
    size_t link_offset; /** Offset to link entry in site struct (bytes) */
    void *mom; /** Pointer to link field (only used if site is not set) */
    size_t mom_offset; /** Offset to mom entry in site struct (bytes) */
    size_t size; /** Size of site struct (bytes) */
  } QudaMILCSiteArg_t;

  /**
   * Parameters related to linear solvers.
   */
  typedef struct {
    int max_iter; /** Maximum number of iterations */
    QudaParity evenodd; /** Which parity are we working on ? (options are QUDA_EVEN_PARITY, QUDA_ODD_PARITY, QUDA_INVALID_PARITY */
    int mixed_precision; /** Whether to use mixed precision or not (1 - yes, 0 - no) */
    double boundary_phase[4]; /** Boundary conditions */
    int make_resident_solution; /** Make the solution resident and don't copy back */
    int use_resident_solution; /** Use the resident solution */
    QudaInverterType solver_type; /** Type of solver to use */
    double tadpole; /** Tadpole improvement factor - set to 1.0 for
                        HISQ fermions since the tadpole factor is
                        baked into the links during their construction */
    double naik_epsilon; /** Naik epsilon parameter (HISQ fermions only).*/
  } QudaInvertArgs_t;

  /**
   * Parameters related to deflated solvers.
   */

  typedef struct {
    QudaPrecision  prec_ritz;
    int nev;
    int max_search_dim;
    int deflation_grid;
    double tol_restart;

    int eigcg_max_restarts;
    int max_restart_num;
    double inc_tol;
    double eigenval_tol;

    QudaExtLibType   solver_ext_lib;
    QudaExtLibType   deflation_ext_lib;

    QudaFieldLocation location_ritz;
    QudaMemoryType    mem_type_ritz;

    char *vec_infile;
    char *vec_outfile;

  } QudaEigArgs_t;


  /**
   * Parameters related to problem size and machine topology.
   */
  typedef struct {
    const int* latsize; /** Local lattice dimensions */
    const int* machsize; /** Machine grid size */
    int device; /** GPU device  number */
  } QudaLayout_t;


  /**
   * Parameters used to create a QUDA context.
   */
  typedef struct {
    QudaVerbosity verbosity; /** How verbose QUDA should be (QUDA_SILENT, QUDA_VERBOSE or QUDA_SUMMARIZE) */
    QudaLayout_t layout; /** Layout for QUDA to use */
  } QudaInitArgs_t; // passed to the initialization struct


  /**
   * Parameters for defining HISQ calculations
   */
  typedef struct {
    int reunit_allow_svd;         /** Allow SVD for reuniarization */
    int reunit_svd_only;          /** Force use of SVD for reunitarization */
    double reunit_svd_abs_error;  /** Absolute error bound for SVD to apply */
    double reunit_svd_rel_error;  /** Relative error bound for SVD to apply */
    double force_filter;          /** UV filter to apply to force */
  } QudaHisqParams_t;


  /**
   * Parameters for defining fat-link calculations
   */
  typedef struct {
    int su3_source;          /** is the incoming gauge field SU(3) */
    int use_pinned_memory;   /** use page-locked memory in QUDA    */
  } QudaFatLinkArgs_t;

  /**
   * Optional: Set the MPI Comm Handle if it is not MPI_COMM_WORLD
   *
   * @param input Pointer to an MPI_Comm handle, static cast as a void *
   */
  void qudaSetMPICommHandle(void *mycomm);

  /**
   * Initialize the QUDA context.
   *
   * @param input Meta data for the QUDA context
   */
  void qudaInit(QudaInitArgs_t input);

  /**
   * Set set the local dimensions and machine topology for QUDA to use
   *
   * @param layout Struct defining local dimensions and machine topology
   */
  void qudaSetLayout(QudaLayout_t layout);

  /**
   * Destroy the QUDA context.
   */
  void qudaFinalize();

  /**
   * Allocate pinned memory suitable for CPU-GPU transfers
   * @param bytes The size of the requested allocation
   * @return Pointer to allocated memory
  */
  void* qudaAllocatePinned(size_t bytes);
 

  /**
   * Free pinned memory
   * @param ptr Pointer to memory to be free
   */
  void qudaFreePinned(void *ptr);
  
    /**
   * Allocate managed memory to reduce CPU-GPU transfers
   * @param bytes The size of the requested allocation
   * @return Pointer to allocated memory
  */
  void* qudaAllocateManaged(size_t bytes);
  
    /**
   * Free managed memory
   * @param ptr Pointer to memory to be free
   */
  void qudaFreeManaged(void *ptr);
  
  /**
   * Set the algorithms to use for HISQ fermion calculations, e.g.,
   * SVD parameters for reunitarization.
   *
   * @param hisq_params Meta data desribing the algorithms to use for HISQ fermions
   */
  void qudaHisqParamsInit(QudaHisqParams_t hisq_params);

  /**
   * Compute the fat and long links using the input gauge field.  All
   * fields passed here are host fields, that must be preallocated.
   * The precision of all fields must match.
   *
   * @param precision The precision of the fields
   * @param fatlink_args Meta data for the algorithms to deploy
   * @param act_path_coeff Array of coefficients for each path in the action
   * @param inlink Host gauge field used for input
   * @param fatlink Host fat-link field that is computed
   * @param longlink Host long-link field that is computed
   */
  void qudaLoadKSLink(int precision,
		      QudaFatLinkArgs_t fatlink_args,
		      const double act_path_coeff[6],
		      void* inlink,
		      void* fatlink,
		      void* longlink);

  /**
   * Compute the fat links and unitzarize using the input gauge field.
   * All fields passed here are host fields, that must be
   * preallocated.  The precision of all fields must match.
   *
   * @param precision The precision of the fields
   * @param fatlink_args Meta data for the algorithms to deploy
   * @param path_coeff Array of coefficients for each path in the action
   * @param inlink Host gauge field used for input
   * @param fatlink Host fat-link field that is computed
   * @param ulink Host unitarized field that is computed
   */
  void qudaLoadUnitarizedLink(int precision,
			      QudaFatLinkArgs_t fatlink_args,
			      const double path_coeff[6],
			      void* inlink,
			      void* fatlink,
			      void* ulink);


  /**
   * Apply the improved staggered operator to a field. All fields
   * passed and returned are host (CPU) field in MILC order.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param inv_args Struct setting some solver metadata
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   */
  void qudaDslash(int external_precision,
		  int quda_precision,
		  QudaInvertArgs_t inv_args,
		  const void* const milc_fatlink,
		  const void* const milc_longlink,
		  void* source,
		  void* solution,
		  int* num_iters);

  /**
   * Solve Ax=b using an improved staggered operator with a
   * domain-decomposition preconditioner.  All fields are fields
   * passed and returned are host (CPU) field in MILC order.  This
   * function requires that persistent gauge and clover fields have
   * been created prior.  This interface is experimental.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param precision Precision for QUDA to use (2 - double, 1 - single)
   * @param mass Fermion mass parameter
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Target residual
   * @param target_relative_residual Target Fermilab residual
   * @param domain_overlap Array specifying the overlap of the domains in each dimension
   * @param fatlink Fat-link field on the host
   * @param longlink Long-link field on the host
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param final_residual True residual
   * @param final_relative_residual True Fermilab residual
   * @param num_iters Number of iterations taken
   */
  void qudaDDInvert(int external_precision,
		    int quda_precision,
		    double mass,
		    QudaInvertArgs_t inv_args,
		    double target_residual,
		    double target_fermilab_residual,
		    const int * const domain_overlap,
		    const void* const fatlink,
		    const void* const longlink,
		    void* source,
		    void* solution,
		    double* const final_residual,
		    double* const final_fermilab_residual,
		    int* num_iters);



  /**
   * Solve Ax=b for an improved staggered operator. All fields are fields
   * passed and returned are host (CPU) field in MILC order.  This
   * function requires that persistent gauge and clover fields have
   * been created prior.  This interface is experimental.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param mass Fermion mass parameter
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Target residual
   * @param target_relative_residual Target Fermilab residual
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param final_residual True residual
   * @param final_relative_residual True Fermilab residual
   * @param num_iters Number of iterations taken
   */
  void qudaInvert(int external_precision,
		  int quda_precision,
		  double mass,
		  QudaInvertArgs_t inv_args,
		  double target_residual,
		  double target_fermilab_residual,
		  const void* const milc_fatlink,
		  const void* const milc_longlink,
		  void* source,
		  void* solution,
		  double* const final_resid,
		  double* const final_rel_resid,
		  int* num_iters);

  /**
   * Solve Ax=b for an improved staggered operator with many right hand sides. 
   * All fields are fields passed and returned are host (CPU) field in MILC order.
   * This function requires that persistent gauge and clover fields have
   * been created prior.  This interface is experimental.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param mass Fermion mass parameter
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Target residual
   * @param target_relative_residual Target Fermilab residual
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param source array of right-hand side source fields
   * @param solution array of solution spinor fields
   * @param final_residual True residual
   * @param final_relative_residual True Fermilab residual
   * @param num_iters Number of iterations taken
   * @param num_src Number of source fields
   */
  void qudaInvertMsrc(int external_precision,
                      int quda_precision,
                      double mass,
                      QudaInvertArgs_t inv_args,
                      double target_residual,
                      double target_fermilab_residual,
                      const void* const fatlink,
                      const void* const longlink,
                      void** sourceArray,
                      void** solutionArray,
                      double* const final_residual,
                      double* const final_fermilab_residual,
                      int* num_iters,
                      int num_src);

  /**
   * Solve for multiple shifts (e.g., masses) using an improved
   * staggered operator.  All fields are fields passed and returned
   * are host (CPU) field in MILC order.  This function requires that
   * persistent gauge and clover fields have been created prior.  When
   * a pure double-precision solver is requested no reliable updates
   * are used, else reliable updates are used with a reliable_delta
   * parameter of 0.1.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param precision Precision for QUDA to use (2 - double, 1 - single)
   * @param num_offsets Number of shifts to solve for
   * @param offset Array of shift offset values
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Array of target residuals per shift
   * @param target_relative_residual Array of target Fermilab residuals per shift
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param source Right-hand side source field
   * @param solutionArray Array of solution spinor fields
   * @param final_residual Array of true residuals
   * @param final_relative_residual Array of true Fermilab residuals
   * @param num_iters Number of iterations taken
   */
  void qudaMultishiftInvert(
      int external_precision,
      int precision,
      int num_offsets,
      double* const offset,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const double* target_fermilab_residual,
      const void* const milc_fatlink,
      const void* const milc_longlink,
      void* source,
      void** solutionArray,
      double* const final_residual,
      double* const final_fermilab_residual,
      int* num_iters);

 /**
   * Solve for a system with many RHS using an improved
   * staggered operator.
   * The solving procedure consists of two computation phases :
   * 1) incremental pahse : call eigCG solver to accumulate low eigenmodes
   * 2) deflation phase : use computed eigenmodes to deflate a regular CG
   * All fields are fields passed and returned
   * are host (CPU) field in MILC order.  This function requires that
   * persistent gauge and clover fields have been created prior.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param precision Precision for QUDA to use (2 - double, 1 - single)
   * @param num_offsets Number of shifts to solve for
   * @param offset Array of shift offset values
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Array of target residuals per shift
   * @param target_relative_residual Array of target Fermilab residuals per shift
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param source Right-hand side source field
   * @param solution Array of solution spinor fields
   * @param eig_args contains info about deflation space
   * @param rhs_idx  bookkeep current rhs
   * @param last_rhs_flag  is this the last rhs to solve?
   * @param final_residual Array of true residuals
   * @param final_relative_residual Array of true Fermilab residuals
   * @param num_iters Number of iterations taken
   */
  void qudaEigCGInvert(
      int external_precision,
      int quda_precision,
      double mass,
      QudaInvertArgs_t inv_args,
      double target_residual,
      double target_fermilab_residual,
      const void* const fatlink,
      const void* const longlink,
      void* source,
      void* solution,
      QudaEigArgs_t eig_args,
      const int rhs_idx,//current rhs
      const int last_rhs_flag,//is this the last rhs to solve?
      double* const final_residual,
      double* const final_fermilab_residual,
      int *num_iters);

  /**
   * Solve Ax=b using a Wilson-Clover operator.  All fields are fields
   * passed and returned are host (CPU) field in MILC order.  This
   * function creates the gauge and clover field from the host fields.
   * Reliable updates are used with a reliable_delta parameter of 0.1.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param kappa Kappa value
   * @param clover_coeff Clover coefficient
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Target residual
   * @param milc_link Gauge field on the host
   * @param milc_clover Clover field on the host
   * @param milc_clover_inv Inverse clover on the host
   * @param clover_coeff Clover coefficient
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param final_residual True residual returned by the solver
   * @param final_residual True Fermilab residual returned by the solver
   * @param num_iters Number of iterations taken
   */
  void qudaCloverInvert(int external_precision,
			int quda_precision,
			double kappa,
			double clover_coeff,
			QudaInvertArgs_t inv_args,
			double target_residual,
			double target_fermilab_residual,
			const void* milc_link,
			void* milc_clover,
			void* milc_clover_inv,
			void* source,
			void* solution,
			double* const final_residual,
			double* const final_fermilab_residual,
			int* num_iters);

  /**
   * Solve for a system with many RHS using using a Wilson-Clover operator.
   * The solving procedure consists of two computation phases :
   * 1) incremental pahse : call eigCG solver to accumulate low eigenmodes
   * 2) deflation phase : use computed eigenmodes to deflate a regular CG
   * All fields are fields passed and returned
   * are host (CPU) field in MILC order.  This function requires that
   * persistent gauge and clover fields have been created prior.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param kappa Kappa value
   * @param clover_coeff Clover coefficient
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Target residual
   * @param milc_link Gauge field on the host
   * @param milc_clover Clover field on the host
   * @param milc_clover_inv Inverse clover on the host
   * @param clover_coeff Clover coefficient
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param eig_args contains info about deflation space
   * @param rhs_idx  bookkeep current rhs
   * @param last_rhs_flag  is this the last rhs to solve?
   * @param final_residual Array of true residuals
   * @param final_relative_residual Array of true Fermilab residuals
   * @param num_iters Number of iterations taken
   */
  void qudaEigCGCloverInvert(
      int external_precision,
      int quda_precision,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      double target_residual,
      double target_fermilab_residual,
      const void* milc_link,
      void* milc_clover,
      void* milc_clover_inv,
      void* source,
      void* solution,
      QudaEigArgs_t eig_args,
      const int rhs_idx,//current rhs
      const int last_rhs_flag,//is this the last rhs to solve?
      double* const final_residual,
      double* const final_fermilab_residual,
      int *num_iters);


  /**
   * Load the gauge field from the host.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param inv_args Meta data
   * @param milc_link Base pointer to host gauge field (regardless of dimensionality)
   */
  void qudaLoadGaugeField(int external_precision,
			  int quda_precision,
			  QudaInvertArgs_t inv_args,
			  const void* milc_link) ;

  /**
     Free the gauge field allocated in QUDA.
   */
  void qudaFreeGaugeField();

  /**
   * Load the clover field and its inverse from the host.  If null
   * pointers are passed, the clover field and / or its inverse will
   * be computed dynamically from the resident gauge field.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param inv_args Meta data
   * @param milc_clover Pointer to host clover field.  If 0 then the
   * clover field is computed dynamically within QUDA.
   * @param milc_clover_inv Pointer to host inverse clover field.  If
   * 0 then the inverse if computed dynamically within QUDA.
   * @param solution_type The type of solution required  (mat, matpc)
   * @param solve_type The solve type to use (normal/direct/preconditioning)
   * @param clover_coeff Clover coefficient
   * @param compute_trlog Whether to compute the trlog of the clover field when inverting
   * @param Array for storing the trlog (length two, one for each parity)
   */
  void qudaLoadCloverField(int external_precision,
			   int quda_precision,
			   QudaInvertArgs_t inv_args,
			   void* milc_clover,
			   void* milc_clover_inv,
			   QudaSolutionType solution_type,
			   QudaSolveType solve_type,
			   double clover_coeff,
			   int compute_trlog,
			   double *trlog) ;

  /**
     Free the clover field allocated in QUDA.
   */
  void qudaFreeCloverField();

  /**
   * Solve for multiple shifts (e.g., masses) using a Wilson-Clover
   * operator with multi-shift CG.  All fields are fields passed and
   * returned are host (CPU) field in MILC order.  This function
   * requires that persistent gauge and clover fields have been
   * created prior.  When a pure double-precision solver is requested
   * no reliable updates are used, else reliable updates are used with
   * a reliable_delta parameter of 0.1.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param num_offsets Number of shifts to solve for
   * @param offset Array of shift offset values
   * @param kappa Kappa value
   * @param clover_coeff Clover coefficient
   * @param inv_args Struct setting some solver metadata
   * @param target_residual Array of target residuals per shift
   * @param milc_link Ignored
   * @param milc_clover Ignored
   * @param milc_clover_inv Ignored
   * @param clover_coeff Clover coefficient
   * @param source Right-hand side source field
   * @param solutionArray Array of solution spinor fields
   * @param final_residual Array of true residuals
   * @param num_iters Number of iterations taken
   */
  void qudaCloverMultishiftInvert(int external_precision,
      int quda_precision,
      int num_offsets,
      double* const offset,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const void* milc_link,
      void* milc_clover,
      void* milc_clover_inv,
      void* source,
      void** solutionArray,
      double* const final_residual,
      int* num_iters
      );

  /**
   * Compute the fermion force for the HISQ quark action.  All fields
   * are host fields in MILC order, and the precision of these fields
   * must match.
   *
   * @param precision       The precision of the fields
   * @param num_terms The number of quark fields
   * @param num_naik_terms The number of naik contributions
   * @param dt Integrating step size
   * @param coeff The coefficients multiplying the fermion fields in the outer product
   * @param quark_field The input fermion field.
   * @param level2_coeff    The coefficients for the second level of smearing in the quark action.
   * @param fat7_coeff      The coefficients for the first level of smearing (fat7) in the quark action.
   * @param w_link          Unitarized link variables obtained by applying fat7 smearing and unitarization to the original links.
   * @param v_link          Fat7 link variables.
   * @param u_link          SU(3) think link variables.
   * @param milc_momentum        The momentum contribution from the quark action.
   */
  void qudaHisqForce(int precision,
                     int num_terms,
                     int num_naik_terms,
                     double dt,
                     double** coeff,
                     void** quark_field,
		     const double level2_coeff[6],
		     const double fat7_coeff[6],
		     const void* const w_link,
		     const void* const v_link,
		     const void* const u_link,
		     void* const milc_momentum);

  /**
   * Compute the gauge force and update the mometum field.  All fields
   * here are CPU fields in MILC order, and their precisions should
   * match.
   *
   * @param precision The precision of the field (2 - double, 1 - single)
   * @param num_loop_types 1, 2 or 3
   * @param milc_loop_coeff Coefficients of the different loops in the Symanzik action
   * @param eb3 The integration step size (for MILC this is dt*beta/3)
   * @param arg Metadata for MILC's internal site struct array
   */
  void qudaGaugeForce(int precision,
		      int num_loop_types,
		      double milc_loop_coeff[3],
		      double eb3,
		      QudaMILCSiteArg_t *arg);

  /**
   * Evolve the gauge field by step size dt, using the momentum field
   * I.e., Evalulate U(t+dt) = e(dt pi) U(t).  All fields are CPU fields in MILC order.
   *
   * @param precision Precision of the field (2 - double, 1 - single)
   * @param dt The integration step size step
   * @param arg Metadata for MILC's internal site struct array
   */
  void qudaUpdateU(int precision,
		   double eps,
		   QudaMILCSiteArg_t *arg);

  /**
   * Download the momentum from MILC and place into QUDA's resident
   * momentum field.  The source momentum field can either be as part
   * of a MILC site struct (QUDA_MILC_SITE_GAUGE_ORDER) or as a
   * separate field (QUDA_MILC_GAUGE_ORDER).
   *
   * @param precision Precision of the field (2 - double, 1 - single)
   * @param arg Metadata for MILC's internal site struct array
   */
  void qudaMomLoad(int precision, QudaMILCSiteArg_t *arg);

  /**
   * Upload the momentum to MILC from QUDA's resident momentum field.
   * The destination momentum field can either be as part of a MILC site
   * struct (QUDA_MILC_SITE_GAUGE_ORDER) or as a separate field
   * (QUDA_MILC_GAUGE_ORDER).
   *
   * @param precision Precision of the field (2 - double, 1 - single)
   * @param arg Metadata for MILC's internal site struct array
   */
  void qudaMomSave(int precision, QudaMILCSiteArg_t *arg);

  /**
   * Evaluate the momentum contribution to the Hybrid Monte Carlo
   * action.  MILC convention is applied, subtracting 4.0 from each
   * momentum matrix to increase stability.
   *
   * @param precision Precision of the field (2 - double, 1 - single)
   * @param arg Metadata for MILC's internal site struct array
   * @return momentum action
   */
  double qudaMomAction(int precision, QudaMILCSiteArg_t *arg);

  /**
   * Apply the staggered phase factors to the gauge field.  If the
   * imaginary chemical potential is non-zero then the phase factor
   * exp(imu/T) will be applied to the links in the temporal
   * direction.
   *
   * @param prec Precision of the gauge field
   * @param gauge_h The gauge field
   * @param flag Whether to apply to remove the staggered phase
   * @param i_mu Imaginary chemical potential
   */
  void qudaRephase(int prec, void *gauge, int flag, double i_mu);

  /**
   * Project the input field on the SU(3) group.  If the target
   * tolerance is not met, this routine will give a runtime error.
   *
   * @param prec Precision of the gauge field
   * @param tol The tolerance to which we iterate
   * @param arg Metadata for MILC's internal site struct array
   */
  void qudaUnitarizeSU3(int prec, double tol, QudaMILCSiteArg_t *arg);

  /**
   * Compute the clover force contributions in each dimension mu given
   * the array solution fields, and compute the resulting momentum
   * field.
   *
   * @param mom Momentum matrix
   * @param dt Integrating step size
   * @param x Array of solution vectors
   * @param p Array of intermediate vectors
   * @param coeff Array of residues for each contribution
   * @param kappa kappa parameter
   * @param ck -clover_coefficient * kappa / 8
   * @param nvec Number of vectors
   * @param multiplicity Number of fermions represented by this bilinear
   * @param gauge Gauge Field
   * @param precision Precision of the fields
   * @param inv_args Struct setting some solver metadata
   */
  void qudaCloverForce(void *mom, double dt, void **x, void **p, double *coeff, double kappa,
		       double ck, int nvec, double multiplicity, void *gauge, int precision,
		       QudaInvertArgs_t inv_args);

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
   */
  void qudaCloverTrace(void* out,
		       void* dummy,
		       int mu,
		       int nu);


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
   * @param precision Precision of the fields (2 = double, 1 = single)
   * @param parity Parity for which we are computing
   * @param conjugate Whether to make the oprod field anti-hermitian prior to multiplication
   */
  void qudaCloverDerivative(void* out,
			    void* gauge,
			    void* oprod,
			    int mu,
			    int nu,
			    double coeff,
			    int precision,
			    int parity,
			    int conjugate);


  /**
   * Take a gauge field on the host, load it onto the device and extend it.
   * Return a pointer to the extended gauge field object.
   *
   * @param gauge The CPU gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the fields (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaCreateExtendedGaugeField(void* gauge,
				     int geometry,
				     int precision);

  /**
   * Take the QUDA resident gauge field and extend it.
   * Return a pointer to the extended gauge field object.
   *
   * @param gauge The CPU gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the fields (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaResidentExtendedGaugeField(void* gauge,
				       int geometry,
				       int precision);

  /**
   * Allocate a gauge (matrix) field on the device and optionally download a host gauge field.
   *
   * @param gauge The host gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the field to be created (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaCreateGaugeField(void* gauge,
			     int geometry,
			     int precision);

  /**
   * Copy the QUDA gauge (matrix) field on the device to the CPU
   *
   * @param outGauge Pointer to the host gauge field
   * @param inGauge Pointer to the device gauge field (QUDA device field)
   */
  void qudaSaveGaugeField(void* gauge,
			  void* inGauge);

  /**
   * Reinterpret gauge as a pointer to cudaGaugeField and call destructor.
   *
   * @param gauge Gauge field to be freed
   */
  void qudaDestroyGaugeField(void* gauge);


  /**
   * @brief Gauge fixing with overrelaxation with support for single and multi GPU.
   * @param[in] precision, 1 for single precision else for double precision
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] relax_boost, gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] reunit_interval, reunitarize gauge field when iteration count is a multiple of this
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   * @param[in,out] milc_sitelink, MILC gauge field to be fixed
   */
  void qudaGaugeFixingOVR( const int precision,
    const unsigned int gauge_dir,
    const int Nsteps,
    const int verbose_interval,
    const double relax_boost,
    const double tolerance,
    const unsigned int reunit_interval,
    const unsigned int stopWtheta,
    void* milc_sitelink
    );


  /**
   * @brief Gauge fixing with Steepest descent method with FFTs with support for single GPU only.
   * @param[in] precision, 1 for single precision else for double precision
   * @param[in] gauge_dir, 3 for Coulomb gauge fixing, other for Landau gauge fixing
   * @param[in] Nsteps, maximum number of steps to perform gauge fixing
   * @param[in] verbose_interval, print gauge fixing info when iteration count is a multiple of this
   * @param[in] alpha, gauge fixing parameter of the method, most common value is 0.08
   * @param[in] autotune, 1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
   * @param[in] tolerance, torelance value to stop the method, if this value is zero then the method stops when iteration reachs the maximum number of steps defined by Nsteps
   * @param[in] stopWtheta, 0 for MILC criterium and 1 to use the theta value
   * @param[in,out] milc_sitelink, MILC gauge field to be fixed
   */
  void qudaGaugeFixingFFT( int precision,
    unsigned int gauge_dir,
    int Nsteps,
    int verbose_interval,
    double alpha,
    unsigned int autotune,
    double tolerance,
    unsigned int stopWtheta,
    void* milc_sitelink
    );

  /* The below declarations are for removed functions from prior versions of QUDA. */

  /**
   * Note this interface function has been removed.  This stub remains
   * for compatibility only.
   */
  void qudaAsqtadForce(int precision,
		       const double act_path_coeff[6],
		       const void* const one_link_src[4],
		       const void* const naik_src[4],
		       const void* const link,
		       void* const milc_momentum);

  /**
   * Note this interface function has been removed.  This stub remains
   * for compatibility only.
   */
  void qudaComputeOprod(int precision,
			int num_terms,
			int num_naik_terms,
			double** coeff,
                        double scale,
			void** quark_field,
			void* oprod[3]);

#ifdef __cplusplus
}
#endif


#endif // _QUDA_MILC_INTERFACE_H
