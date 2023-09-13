#pragma once

#include <enum_quda.h>
#include <quda.h>

/**
 * @file    quda_openqcd_interface.h
 *
 * @section Description
 *
 * The header file defines the interface to enable easy
 * interfacing between QUDA and the OpenQCD software.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Parameters related to problem size and machine topology. They should hold the
 * numbers in quda format, i.e. xyzt convention. For example L[0] = L1, L[1] =
 * L2, ...
 */
typedef struct {
  int L[4];         /** Local lattice dimensions L0, L1, L2, L3 */
  int nproc[4];     /** Machine grid size NPROC0, NPROC1, NPROC2, NPROC3*/
  int nproc_blk[4]; /** Blocking size NPROC0_BLK, NPROC1_BLK, NPROC2_BLK, NPROC3_BLK */
  int N[4];         /** Glocal lattice dimensions N0, N1, N2, N3 */
  int device;       /** GPU device number */
} openQCD_QudaLayout_t;

/**
 * Parameters used to create a QUDA context.
 */
typedef struct {
  QudaVerbosity verbosity;      /** How verbose QUDA should be (QUDA_SILENT, QUDA_VERBOSE or QUDA_SUMMARIZE) */
  openQCD_QudaLayout_t layout;  /** Layout for QUDA to use */
  FILE *logfile;                /** log file handler */
  void *gauge;                  /** base pointer to the gauge fields */
  int volume;                   /** VOLUME */
  int sizeof_su3_dble;          /** sizeof(su3_dble) */
  int sizeof_spinor_dble;       /** sizeof(spinor_dble) */
  void (*reorder_gauge_openqcd_to_quda)(void *in, void *out);
  void (*reorder_gauge_quda_to_openqcd)(void *in, void *out);
  void (*reorder_spinor_openqcd_to_quda)(void *in, void *out);
  void (*reorder_spinor_quda_to_openqcd)(void *in, void *out);
} openQCD_QudaInitArgs_t;


typedef struct {
  int initialized;    /** Whether openQCD_qudaInit() was called or not */
  int gauge_loaded;   /** Whether openQCD_qudaGaugeLoad() was called or not */
  int clover_loaded;  /** Whether openQCD_qudaCloverLoad() was called or not */
  int dslash_setup;   /** Whether openQCD_qudaSetDslashOptions() was called or not */
} openQCD_QudaState_t;


/**
 * Initialize the QUDA context.
 *
 * @param input Meta data for the QUDA context
 */
void openQCD_qudaInit(openQCD_QudaInitArgs_t input);

/**
 * Set set the local dimensions and machine topology for QUDA to use
 *
 * @param layout Struct defining local dimensions and machine topology
 */
void openQCD_qudaSetLayout(openQCD_QudaLayout_t layout);

/**
 * Destroy the QUDA context.
 */
void openQCD_qudaFinalize(void);


/**
 * Parameters related to linear solvers.
 */

typedef struct {
  // TODO: work out what we want to expose here
  int max_iter; /** Maximum number of iterations */
  QudaParity
    evenodd; /** Which parity are we working on ? (options are QUDA_EVEN_PARITY, QUDA_ODD_PARITY, QUDA_INVALID_PARITY */
  int mixed_precision;          /** Whether to use mixed precision or not (1 - yes, 0 - no) */
  double boundary_phase[4];     /** Boundary conditions */
  int make_resident_solution;   /** Make the solution resident and don't copy back */
  int use_resident_solution;    /** Use the resident solution */
  QudaInverterType solver_type; /** Type of solver to use */
  double tadpole;               /** Tadpole improvement factor - set to 1.0 for
                                    HISQ fermions since the tadpole factor is
                                    baked into the links during their construction */
  double naik_epsilon;          /** Naik epsilon parameter (HISQ fermions only).*/
  QudaDslashType dslash_type;
} openQCD_QudaInvertArgs_t;


/**
 * @brief      Setup Dirac operator
 *
 * @param[in]  kappa   kappa
 * @param[in]  mu      twisted mass
 */
void openQCD_qudaSetDwOptions(double kappa, double mu);


/**
 * @brief      Norm square on QUDA.
 *
 * @param[in]  h_in  Input field (from openQCD)
 *
 * @return     The norm
 */
double openQCD_qudaNorm(void *h_in);

void openQCD_qudaGamma(int dir, void *openQCD_in, void *openQCD_out);

/**
 * @brief      Apply the Wilson-Clover Dirac operator to a field. All fields
 *             passed and returned are host (CPU) fields in openQCD order.
 *
 * @param[in]  src     Source spinor field
 * @param[out] dst     Destination spinor field
 * @param[in]  dagger  Whether we are using the Hermitian conjugate system or
 *                     not (QUDA_DAG_NO or QUDA_DAG_YES)
 */
void openQCD_qudaDw(void *src, void *dst, QudaDagType dagger);


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
void openQCD_qudaInvert(int external_precision, int quda_precision, double mass, openQCD_QudaInvertArgs_t inv_args,
                        double target_residual, double target_fermilab_residual, const void *const milc_fatlink,
                        const void *const milc_longlink, void *source, void *solution, double *const final_resid,
                        double *const final_rel_resid, int *num_iters);

/**
 * @brief      Wrapper for the plaquette. We could call plaqQuda() directly in
 *             openQCD, but we have to make sure manually that the gauge field
 *             is loaded
 *
 * @return     Plaquette value
 * @see        https://github.com/lattice/quda/wiki/Gauge-Measurements#wilson-plaquette-action
 */
double openQCD_qudaPlaquette(void);


/**
 * @brief      Load the gauge fields from host to quda.
 *
 * @param[in]  gauge      The gauge fields (in openqcd order)
 */
void openQCD_qudaGaugeLoad(void *gauge);


/**
 * @brief      Save the gauge fields from quda to host.
 *
 * @param[out] gauge      The gauge fields (will be stored in openqcd order)
 */
void openQCD_qudaGaugeSave(void *gauge);


/**
 * @brief      Free the gauge field allocated in quda.
 */
void openQCD_qudaGaugeFree(void);


/**
 * @brief      Load the clover fields from host to quda.
 *
 * @param[in]  clover      The clover fields (in openqcd order)
 */
void openQCD_qudaCloverLoad(void *clover);


/**
 * @brief      Calculates the clover field and its inverse
 *
 * @param[in]  su3csw  The csw coefficient
 */
void openQCD_qudaCloverCreate(double su3csw);


/**
 * @brief      Free the clover field allocated in quda.
 */
void openQCD_qudaCloverFree(void);


#ifdef __cplusplus
}
#endif
