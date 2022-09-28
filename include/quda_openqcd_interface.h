#pragma once

#include <enum_quda.h>
#include <quda.h>

/**
 * @file    quda_openqcd_interface.h
 *
 * @section Description
 *
 * The header file defines the milc interface to enable easy
 * interfacing between QUDA and the OpenQCS software.
 */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Parameters related to problem size and machine topology.
 */
typedef struct {
  const int *latsize;  /** Local lattice dimensions L0, L1, L2, L3 */
  const int *machsize; /** Machine grid size NPROC0, NPROC1, NPROC2, NPROC3*/
  const int *blksize;  /** Blocking size NPROC0_BLK, NPROC1_BLK, NPROC2_BLK, NPROC3_BLK */
  int device;          /** GPU device  number */
} openQCD_QudaLayout_t;

/**
 * Parameters used to create a QUDA context.
 */
typedef struct {
  QudaVerbosity verbosity;     /** How verbose QUDA should be (QUDA_SILENT, QUDA_VERBOSE or QUDA_SUMMARIZE) */
  openQCD_QudaLayout_t layout; /** Layout for QUDA to use */
} openQCD_QudaInitArgs_t;      // passed to the initialization struct

/**
 * Initialize the QUDA context.
 *
 * @param input Meta data for the QUDA context
 */
void openQCD_qudaInit(openQCD_QudaInitArgs_t input);

// /**
//  * Set set the local dimensions and machine topology for QUDA to use
//  *
//  * @param layout Struct defining local dimensions and machine topology
//  */
// void openQCD_qudaSetLayout(openQCD_QudaLayout_t layout);

/**
 * Destroy the QUDA context.
 */
void openQCD_qudaFinalize(void);

#if 0
// leave that here for now
  /**
   * Allocate pinned memory suitable for CPU-GPU transfers
   * @param bytes The size of the requested allocation
   * @return Pointer to allocated memory
  */
  void* openQCD_qudaAllocatePinned(size_t bytes);

  /**
   * Free pinned memory
   * @param ptr Pointer to memory to be free
   */
  void openQCD_qudaFreePinned(void *ptr);

  /**
   * Allocate managed memory to reduce CPU-GPU transfers
   * @param bytes The size of the requested allocation
   * @return Pointer to allocated memory
   */
  void *openQCD_qudaAllocateManaged(size_t bytes);

  /**
   * Free managed memory
   * @param ptr Pointer to memory to be free
   */
  void openQCD_qudaFreeManaged(void *ptr);

#endif

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
} openQCD_QudaInvertArgs_t;

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
void openQCD_qudaDslash(int external_precision, int quda_precision, openQCD_QudaInvertArgs_t inv_args,
                        const void *const milc_fatlink, const void *const milc_longlink, void *source, void *solution,
                        int *num_iters);

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
 * Load the gauge field from the host.
 *
 * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
 * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
 * @param inv_args Meta data
 * @param milc_link Base pointer to host gauge field (regardless of dimensionality)
 */
void openQCD_qudaLoadGaugeField(int external_precision, int quda_precision, openQCD_QudaInvertArgs_t inv_args,
                                const void *milc_link);

void openQCD_qudaPlaquette(int precision, double plaq[3], void *gauge);

/**
   Free the gauge field allocated in QUDA.
 */
void openQCD_qudaFreeGaugeField();

#ifdef __cplusplus
}
#endif
