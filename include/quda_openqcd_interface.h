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
 * Parameters related to problem size and machine topology.
 */
typedef struct {
  const int *latsize; /** Local lattice dimensions L0, L1, L2, L3 */
  const int *machsize; /** Machine grid size NPROC0, NPROC1, NPROC2, NPROC3*/
  const int *blksize; /** Blocking size NPROC0_BLK, NPROC1_BLK, NPROC2_BLK, NPROC3_BLK */
  int device; /** GPU device number */
  // const int *ipt; // TODO: IN THE FUTURE
} openQCD_QudaLayout_t;

/**
 * Parameters used to create a QUDA context.
 */
typedef struct {
  QudaVerbosity verbosity;     /** How verbose QUDA should be (QUDA_SILENT, QUDA_VERBOSE or QUDA_SUMMARIZE) */
  openQCD_QudaLayout_t layout; /** Layout for QUDA to use */
  FILE *logfile;
  int volume; /* VOLUME */
  int sizeof_su3_dble; /* sizeof(su3_dble) */
  void (*reorder_gauge_openqcd_to_quda)(void *in, void *out);
  void (*reorder_gauge_quda_to_openqcd)(void *in, void *out);
} openQCD_QudaInitArgs_t;      // passed to the initialization struct


typedef struct {
  int initialized;
  int gauge_loaded;
  int dslash_setup;
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
 * @brief      Setup Dslash
 *
 * @param[in]  kappa  kappa
 * @param[in]  mu     twisted mass
 */
void openQCD_qudaSetDslashOptions(double kappa, double mu);


/**
 * @brief      Apply the Wilson-Clover Dirac operator to a field. All fields
 *             passed and returned are host (CPU) fields in openQCD order.
 *
 * @param[in]  src                 Source spinor field
 * @param[out] dst                 Destination spinor field
 */
void openQCD_qudaDslash(void *src, void *dst);


/**
 * @brief      Set metadata, options for Dslash.
 *
 * @param[in]  external_precision  Precision of host fields passed to QUDA (2 - double, 1 - single)
 * @param[in]  quda_precision      Precision for QUDA to use (2 - double, 1 - single)
 * @param[in]  inv_args            Struct containing arguments, metadata
 */
/*void openQCD_qudaSetDslashOptions(int external_precision, int quda_precision, openQCD_QudaInvertArgs_t inv_args);*/

/**
 * ALL the following except the Dirac operator application
 * Apply the improved staggered operator to a field. All fields
 * passed and returned are host (CPU) field in MILC order.
 *
 * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
 * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
 * @param inv_args Struct setting some solver metadata
 * @param source Right-hand side source field
 * @param solution Solution spinor field
 */
void openQCD_colorspinorloadsave(int external_precision, int quda_precision, openQCD_QudaInvertArgs_t inv_args, void *src,
                        void *dst, void *gauge);


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
 * @brief      Calculate the plaquette
 *
 * @param[out] plaq  array to store the 3 plaquette values
 */
void openQCD_qudaPlaquette(double plaq[3]);


/**
 * @brief      Load the gauge fields from host to quda
 *
 * @param[in]  precision  The precision
 * @param[in]  gauge      The gauge fields (in openqcd order)
 */
void openQCD_gaugeload(int precision, void *gauge);


/**
 * @brief      Save the gauge fields from quda to host
 *
 * @param[in]  precision  The precision
 * @param[out] gauge      The gauge fields (will be stored in openqcd order)
 */
void openQCD_gaugesave(int precision, void *gauge);


/**
   Free the gauge field allocated in QUDA.
 */
void openQCD_qudaFreeGaugeField(void);

#ifdef __cplusplus
}
#endif
