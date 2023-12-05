#pragma once

/**
 * The macro battle below is to trick quda.h to think that double_complex is
 * defined to be the struct below. For this we need to set the __CUDACC_RTC__,
 * which makes double_complex to be defined as double2 (see quda.h), which we
 * redefine below as openqcd_complex_dble. The original definitions of
 * __CUDACC_RTC__ and double2 are recovered below. We do this to be able to
 * include this header file into a openQxD program and compile with flags
 * "-std=C89 -pedantic -Werror". Else the compiler trows an
 * "ISO C90 does not support complex types" error because of the
 * "double _Complex" data types exposed in quda.h.
 */

typedef struct
{
  double re,im;
} openqcd_complex_dble;

#ifdef __CUDACC_RTC__
#define __CUDACC_RTC_ORIGINAL__ __CUDACC_RTC__
#endif

#ifdef double2
#define double2_ORIGINAL double2
#endif

#define __CUDACC_RTC__
#define double2 openqcd_complex_dble
#include <quda.h>
#undef double2
#undef __CUDACC_RTC__

#ifdef double2_ORIGINAL
#define double2 double2_ORIGINAL
#undef double2_ORIGINAL
#endif

#ifdef __CUDACC_RTC_ORIGINAL__
#define __CUDACC_RTC__ __CUDACC_RTC_ORIGINAL__
#undef __CUDACC_RTC_ORIGINAL__
#endif

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
 * Copied from flags.h
 * #############################################
 */
#ifndef FLAGS_H
typedef struct
{
   int type;
   int cstar;
   double phi3[2][3];
   double phi1[2];
} bc_parms_t;

typedef struct
{
   int qhat;
   double m0,su3csw,u1csw,cF[2],theta[3];
} dirac_parms_t;

typedef struct
{
   int gauge;
   int nfl;
} flds_parms_t;
#endif
/**
 * #############################################
 */


typedef enum OpenQCDGaugeGroup_s {
  OPENQCD_GAUGE_SU3 = 1,
  OPENQCD_GAUGE_U1 = 2,
  OPENQCD_GAUGE_SU3xU1 = 3,
  OPENQCD_GAUGE_INVALID = QUDA_INVALID_ENUM
} OpenQCDGaugeGroup;


/**
 * Parameters related to problem size and machine topology. They should hold the
 * numbers in quda format, i.e. xyzt convention. For example L[0] = L1, L[1] =
 * L2, ...
 */
typedef struct {
  int L[4];         /** Local lattice dimensions L1, L2, L3, L0 */
  int nproc[4];     /** Machine grid size NPROC1, NPROC2, NPROC3, NPROC0*/
  int nproc_blk[4]; /** Blocking size NPROC0_BLK, NPROC1_BLK, NPROC2_BLK, NPROC3_BLK,
                        is assumed to be [1, 1, 1, 1] */
  int N[4];         /** Glocal lattice dimensions N1, N2, N3, N3 */
  int device;       /** GPU device number */
  int cstar;        /** number of cstar directions, equals bc_cstar() */
  int *data;        /** rank topology, length 5 + NPROC1*NPROC2*NPROC3*NPROC0:
                        data[0] = cstar;
                        data[1+i] = nproc[i] for 0 <= i < 4
                        data[5+lex(ix,iy,iz,it)] returns rank number in
                        openQCD, where lex stands for lexicographical
                        indexing (in QUDA order (xyzt)) */
  bc_parms_t bc_parms;       /** @see bc_parms() */
  dirac_parms_t dirac_parms; /** @see dirac_parms() */
  flds_parms_t flds_parms;   /** @see flds_parms() */
  void* (*h_gauge)(void);    /** function to return a pointer to the gauge fields */
  void* (*h_sw)(void);       /** function to return a pointer to the clover fields */
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
  int bndry;                    /** BNDRY */
  void (*reorder_gauge_quda_to_openqcd)(void *in, void *out);
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
  double kappa;   /* kappa: hopping parameter */
  double mu;      /* mu: twisted mass */
  double su3csw;  /* su3csw: csw coefficient for SU(3) fields */
  double u1csw;   /* u1csw: csw coefficient for U(1) fields, quda doesn't respect that parameter (yet) */
  int qhat;       /* qhat: quda doesn't respect that parameter (yet) */
} openQCD_QudaDiracParam_t;


/**
 * Initialize the QUDA context.
 *
 * @param[in]  init    Meta data for the QUDA context
 * @param[in]  layout  The layout struct
 */
void openQCD_qudaInit(openQCD_QudaInitArgs_t init, openQCD_QudaLayout_t layout);


/**
 * Destroy the QUDA context.
 */
void openQCD_qudaFinalize(void);


/**
 * Copy a spinor to GPU and back to CPU.
 * 
 * @param[in]   h_in  Spinor input field (from openQCD)
 * @param[out]  h_out Spinor output field
 */
void openQCD_back_and_forth(void *h_in, void *h_out);


/**
 * @brief      Norm square in QUDA.
 *
 * @param[in]  h_in  Spinor input field (from openQCD)
 *
 * @return     The norm
 */
double openQCD_qudaNorm(void *h_in);


/**
 * @brief      Prototype function for the norm-square in QUDA without loading
 *             the field.
 *
 * @param[in]  d_in  Spinor input field (device pointer)
 *
 * @return     The norm
 */
double openQCD_qudaNorm_NoLoads(void *d_in);


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
void openQCD_qudaGamma(const int dir, void *openQCD_in, void *openQCD_out);


void* openQCD_qudaH2D(void *openQCD_field);
void openQCD_qudaD2H(void *quda_field, void *openQCD_field);
void openQCD_qudaSpinorFree(void** quda_field);


/**
 * @brief      Apply the Wilson-Clover Dirac operator to a field. All fields
 *             passed and returned are host (CPU) fields in openQCD order.
 *
 * @param[in]  src   Source spinor field
 * @param[out] dst   Destination spinor field
 * @param[in]  p     Dirac parameter struct
 */
void openQCD_qudaDw(void *src, void *dst, openQCD_QudaDiracParam_t p);
void openQCD_qudaDdagD(void *src, void *dst, openQCD_QudaDiracParam_t p);
void openQCD_qudaDw2(void *param, double mu, void *src, void *dst);


/**
 * Setup the solver interface to quda.  This function parses the file given by
 * [infile] as an openQCD ini file.  The solver section given by the [section]
 * parameter must have a key-value pair like solver = QUDA and may contain every
 * member of the struct [QudaInvertParam].  If one sets inv_type_precondition =
 * QUDA_MG_INVERTER, one can additionally use all the members from the struct
 * [QudaMultigridParam] in a section called "{section} Multigrid", where
 * {section} is replaced by [section].  For every level given by n_level in the
 * above section, one has to provide a subsection called
 * "{section} Multigrid Level {level}", where {level} runs from 0 to n_level-1.
 * All these subsections may have keys given by all the array-valued members of
 * QudaMultigridParam, for example smoother_tol may appear in all subsections.
 *
 * @param[in]  infile   Ini-file containing sections about the solver
 * @param[in]  section  The section name
 *
 * @return     Pointer to the solver context
 */
void* openQCD_qudaSolverSetup(char *infile, char *section);


/**
 * @brief        Solve Ax=b for an Clover Wilson operator with a multigrid
 *               solver. All fields are fields passed and returned are host
 *               (CPU) field in openQCD order.  This function requires an
 *               existing solver context created with openQCD_qudaSolverSetup()
 *
 * @param[inout] param     Pointer returned by openQCD_qudaSolverSetup()
 * @param[in]    mu        Twisted mass
 * @param[in]    source    The source
 * @param[out]   solution  The solution
 * @param[out]   status    If the function is able to solve the Dirac equation
 *                         to the desired accuracy (invert_param->tol), status
 *                         reports the total number of iteration steps. -1
 *                         indicates that the inversion failed.
 *
 * @return       Residual
 */
double openQCD_qudaInvert(void *param, double mu, void *source, void *solution, int *status);


/**
 * @brief      Destroys an existing solver context and frees all involed
 *             structs.
 *
 * @param      param  Pointer to the context to destroy
 */
void openQCD_qudaSolverDestroy(void *param);


void* openQCD_qudaEigensolverSetup(char *infile, char *section, char *inv_section);
void openQCD_qudaEigensolve(void *param, void **h_evecs, void *h_evals);
void openQCD_qudaEigensolverDestroy(void *param);


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
 * @brief      Load the gauge fields from host to quda. Notice that the boundary
 *             fields have to be up2date; i.e. call copy_bnd_hd(), copy_bnd_ud()
 *             before pass fields into this function.
 *
 * @param[in]  gauge  The gauge fields (in openqcd order)
 * @param[in]  prec   Precision of the incoming gauge field
 */
void openQCD_qudaGaugeLoad(void *gauge, QudaPrecision prec);


/**
 * @brief      Save the gauge fields from quda to host.
 *
 * @param[out] gauge  The gauge fields (will be stored in openqcd order)
 * @param[in]  prec   Precision of the outgoing gauge field
 */
void openQCD_qudaGaugeSave(void *gauge, QudaPrecision prec);


/**
 * @brief      Free the gauge field allocated in quda.
 */
void openQCD_qudaGaugeFree(void);


/**
 * @brief      Load the clover fields from host to quda.
 *
 * @param[in]  clover  The clover fields (in openqcd order)
 * @param[in]  kappa   The kappa (we need this, because quda has its clover
 *                     field multiplied by kappa and we have to reverse this
 *                     when loading ours)
 * @param[in]  csw     The csw coefficient of the clover field
 */
void openQCD_qudaCloverLoad(void *clover, double kappa, double csw);


/**
 * @brief      Free the clover field allocated in quda.
 */
void openQCD_qudaCloverFree(void);


#ifdef __cplusplus
}
#endif
