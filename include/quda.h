#ifndef _QUDA_H
#define _QUDA_H

/**
 * @file  quda.h
 * @brief Main header file for the QUDA library
 *
 * Note to QUDA developers: When adding new members to QudaGaugeParam
 * and QudaInvertParam, be sure to update lib/check_params.h
 */

#include <enum_quda.h>

#define QUDA_VERSION_MAJOR     0
#define QUDA_VERSION_MINOR     4
#define QUDA_VERSION_SUBMINOR  0 

/**
 * @def   QUDA_VERSION
 * @brief This macro is deprecated.  Use QUDA_VERSION_MAJOR, etc., instead.
 */
#define QUDA_VERSION ((QUDA_VERSION_MAJOR<<16) | (QUDA_VERSION_MINOR<<8) | QUDA_VERSION_SUBMINOR)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def   QUDA_MAX_DIM
 * @brief Maximum number of dimensions supported by QUDA.  In practice, no
 *        routines make use of more than 5.
 */
#define QUDA_MAX_DIM 6

/**
 * @def QUDA_MAX_MULTI_SHIFT
 * @brief Maximum number of shifts supported by the multi-shift solver.
 *        This number may be changed if need be.
 */
#define QUDA_MAX_MULTI_SHIFT 32


  /**
   * Parameters having to do with the gauge field or the
   * interpretation of the gauge field by various Dirac operators
   */
  typedef struct QudaGaugeParam_s {

    int X[4];

    double anisotropy;    /**< Used for Wilson and Wilson-clover */
    double tadpole_coeff; /**< Used for staggered only */

    QudaLinkType type;
    QudaGaugeFieldOrder gauge_order;

    QudaTboundary t_boundary;

    QudaPrecision cpu_prec;

    QudaPrecision cuda_prec;
    QudaReconstructType reconstruct;

    QudaPrecision cuda_prec_sloppy;
    QudaReconstructType reconstruct_sloppy;

    QudaPrecision cuda_prec_precondition;
    QudaReconstructType reconstruct_precondition;

    QudaGaugeFixed gauge_fix;

    int ga_pad;

    /** Used by link fattening and the gauge and fermion forces */
    int site_ga_pad;

    int staple_pad;   /**< Used by link fattening */
    int llfat_ga_pad; /**< Used by link fattening */
    int mom_ga_pad;   /**< Used by the gauge and fermion forces */
    int packed_size;
    double gaugeGiB;

    int preserve_gauge; /**< Used by link fattening */
    
  } QudaGaugeParam;


  /*
   * Parameters relating to the solver and the choice of Dirac operator.
   */
  typedef struct QudaInvertParam_s {

    QudaDslashType dslash_type;
    QudaInverterType inv_type;

    double mass;  /**< Used for staggered only */
    double kappa; /**< Used for Wilson and Wilson-clover */

    double m5;    /**< Domain wall height */
    int Ls;       /**< Extent of the 5th dimension (for domain wall) */

    double mu;    /**< Twisted mass parameter */
    QudaTwistFlavorType twist_flavor;  /**< Twisted mass flavor */

    double tol;
    int maxiter;
    double reliable_delta; /**< Reliable update tolerance */

    int num_offset; /**< Number of offsets in the multi-shift solver */

    /** Offsets for multi-shift solver */
    double offset[QUDA_MAX_MULTI_SHIFT];

    /** Solver tolerance for each offset */
    double tol_offset[QUDA_MAX_MULTI_SHIFT];

    QudaSolutionType solution_type;  /**< Type of system to solve */
    QudaSolveType solve_type;        /**< How to solve it */
    QudaMatPCType matpc_type;
    QudaDagType dagger;
    QudaMassNormalization mass_normalization;

    QudaPreserveSource preserve_source;

    QudaPrecision cpu_prec;
    QudaPrecision cuda_prec;
    QudaPrecision cuda_prec_sloppy;
    QudaPrecision cuda_prec_precondition;

    QudaDiracFieldOrder dirac_order;

    /** Gamma basis of the input and output host fields */
    QudaGammaBasis gamma_basis;

    QudaPrecision clover_cpu_prec;
    QudaPrecision clover_cuda_prec;
    QudaPrecision clover_cuda_prec_sloppy;
    QudaPrecision clover_cuda_prec_precondition;

    QudaCloverFieldOrder clover_order;
    QudaUseInitGuess use_init_guess;

    QudaVerbosity verbosity;    

    int sp_pad;
    int cl_pad;

    int iter;
    double spinorGiB;
    double cloverGiB;
    double gflops;
    double secs;

    /** Enable auto-tuning? */
    QudaTune tune;

    /** Free the Dirac operator or keep it resident? */
    QudaPreserveDirac preserve_dirac;

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

    /** Precision used in the inner solver. */
    QudaPrecision prec_precondition;

    /** Relaxation parameter used in GCR-DD (default = 1.0) */
    double omega;

  } QudaInvertParam;


  /*
   * Interface functions, found in interface_quda.cpp
   */

  /**
   * Initialize the library.
   *
   * @param device  CUDA device number to use.  In a multi-GPU build,
   *                this parameter may be either set explicitly on a
   *                per-process basis or set to -1 to enable a default
   *                allocation of devices to processes.
   */
  void initQuda(int device);

  /**
   * Load the gauge field from the host.
   */
  void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param);

  /**
   * Free QUDA's internal copy of the gauge field.
   */
  void freeGaugeQuda(void);

  /**
   * Save the gauge field to the host.
   */
  void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param);

  /**
   * Load the clover term and/or the clover inverse from the host.
   * Either h_clover or h_clovinv may be set to NULL.
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
   */
  void invertQuda(void *h_x, void *h_b, QudaInvertParam *param);

  /**
   * Solve for multiple shifts (e.g., masses).
   */
  void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param,
			    double* offsets, int num_offsets,
			    double* residue_sq);

  /**
   * Mixed-precision multi-shift solver.  In the future, this functionality
   * will be folded into invertMultiShiftQuda().
   */
  void invertMultiShiftQudaMixed(void **_hp_x, void *_hp_b,
				 QudaInvertParam *param, double* offsets,
				 int num_offsets, double* residue_sq);
    
  /** Apply the Dslash operator (D_{eo} or D_{oe}) */
  void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param,
		  QudaParity parity);

  /** Apply the full Dslash matrix, possibly even/odd preconditioned */
  void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);

  /** Apply M^{\dag}M, possibly even/odd preconditioned */
  void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param);

  /** Finalize the library */
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
   */
  void printQudaGaugeParam(QudaGaugeParam *param);

  /**
   * Print the members of QudaGaugeParam.
   */
  void printQudaInvertParam(QudaInvertParam *param);


  /*
   * The following routines are temporary additions used by the HISQ
   * link-fattening code.
   */

  void  record_gauge(int* X, void *_fatlink, int _fatlink_pad, 
		     void* _longlink, int _longlink_pad, 
		     QudaReconstructType _longlink_recon,
		     QudaReconstructType _longlink_recon_sloppy,
		     QudaGaugeParam *_param);
  void set_dim(int *);
  void pack_ghost(void **cpuLink, void **cpuGhost, int nFace,
		  QudaPrecision precision);
  void setFatLinkPadding(QudaComputeFatMethod method, QudaGaugeParam* param);
  int computeFatLinkQuda(void* fatlink, void** sitelink,
			 double* act_path_coeff, QudaGaugeParam* param, 
			 QudaComputeFatMethod method);
  
  /*
   * The following routines are only used by the examples in tests/ .
   * They should generally not be called in a typical application.
   */  
  void initCommsQuda(int argc, char **argv, const int *X, const int nDim);
  void endCommsQuda();

#ifdef __cplusplus
}
#endif

#endif /* _QUDA_H */
