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
  openQCD_QudaInitArgs_t init;
  openQCD_QudaLayout_t layout;
} openQCD_QudaState_t;


typedef struct {
  double kappa;
  double mu;
  double su3csw;
  int dagger;
} openQCD_QudaDiracParam_t;


typedef struct {
  double tol;
  double nmx;
  int nkv;
  double reliable_delta;
} openQCD_QudaGCRParam_t;


/**
 * Initialize the QUDA context.
 *
 * @param[in]  init    Meta data for the QUDA context
 * @param[in]  layout  The layout
 */
void openQCD_qudaInit(openQCD_QudaInitArgs_t init, openQCD_QudaLayout_t layout);


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
 * @brief      Norm square on QUDA.
 *
 * @param[in]  h_in  Input field (from openQCD)
 *
 * @return     The norm
 */
double openQCD_qudaNorm(void *h_in);


/**
 * @brief      Applies Dirac matrix to spinor.
 *
 *             openQCD_out = gamma[dir] * openQCD_in
 *
 * @param[in]  dir          Dirac index, 0 <= dir <= 5, notice that dir is in
 *                          openQCD convention, ie. (0: t, 1: x, 2: y, 3: z, 4: 5, 5: 5)
 * @param[in]  openQCD_in   of type spinor_dble[NSPIN]
 * @param[out] openQCD_out  of type spinor_dble[NSPIN]
 */
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
void openQCD_qudaDw(void *src, void *dst, openQCD_QudaDiracParam_t p);


/**
 * Solve Ax=b for a Clover Wilson operator using QUDAs GCR algorithm. All fields
 * are fields passed and returned are host (CPU) field in openQCD order. This
 * function requires that persistent gauge and clover fields have been created
 * prior.
 *
 * @param[in]  source       Source spinor
 * @param[out] solution     Solution spinor
 * @param[in]  dirac_param  Dirac parameter struct
 * @param[in]  gcr_param    GCR parameter struct
 */
void openQCD_qudaGCR(void *source, void *solution,
  openQCD_QudaDiracParam_t dirac_param, openQCD_QudaGCRParam_t gcr_param);


/**
 * Solve Ax=b for an Clover Wilson operator. All fields are fields passed and
 * returned are host (CPU) field in openQCD order.  This function requires that
 * persistent gauge and clover fields have been created prior.
 *
 * @param[in]  source    Right-hand side source field
 * @param[out] solution  Solution spinor field
 * @param[in]  tol       The tolerance
 * @param[in]  maxiter   The maxiter
 */
void openQCD_qudaInvert(void *source, void *solution, openQCD_QudaDiracParam_t dirac_param);


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
 * @brief      Free the clover field allocated in quda.
 */
void openQCD_qudaCloverFree(void);


#ifdef __cplusplus
}
#endif
