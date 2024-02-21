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
  bc_parms_t (*bc_parms)(void); /** @see bc_parms() */
  flds_parms_t (*flds_parms)(void); /** @see flds_parms() */
  dirac_parms_t (*dirac_parms)(void); /** @see dirac_parms() */
  void* (*h_gauge)(void);    /** function to return a pointer to the gauge field */
  void* (*h_sw)(void);       /** function to return a pointer to the updated Clover field */
  void (*get_gfld_flags)(int *ud, int *ad);      /** function pointer to gauge field revision query function */
} openQCD_QudaLayout_t;


/**
 * Parameters used to create a QUDA context.
 */
typedef struct {
  QudaVerbosity verbosity;      /** How verbose QUDA should be (QUDA_SILENT, QUDA_VERBOSE or QUDA_SUMMARIZE) */
  FILE *logfile;                /** log file handler */
  void *gauge;                  /** base pointer to the gauge fields */
  int volume;                   /** VOLUME */
  int bndry;                    /** BNDRY */
  void (*reorder_gauge_quda_to_openqcd)(void *in, void *out);
} openQCD_QudaInitArgs_t;


typedef struct {
  int initialized;    /** Whether openQCD_qudaInit() was called or not */
  int ud_rev;         /** Revision of ud field from openqxd */
  int ad_rev;         /** Revision of ad field from openqxd */
  int swd_ud_rev;     /** Revision of ud field used to calc/transfer the SW field from openqxd */
  int swd_ad_rev;     /** Revision of ad field used to calc/transfer the SW field from openqxd */
  double swd_kappa;   /** kappa corresponding to the current SW field in QUDA */
  double swd_su3csw;  /** SU(3) csw coefficient corresponding to the current SW field in QUDA */
  double swd_u1csw;   /** U(1) csw coefficient corresponding to the current SW field in QUDA */
  openQCD_QudaInitArgs_t init;
  openQCD_QudaLayout_t layout;
  void* handles[32];  /** Array of void-pointers to QudaInvertParam structs for the solver(s) */
  void* dirac_handle; /** void-pointer to QudaInvertParam struct for the Dirac operator */
  char infile[1024];  /** Path to the input file (if given to quda_init()) */
} openQCD_QudaState_t;


typedef struct openQCD_QudaSolver_s {
  char infile[1024];              /** Path to the input file (if given to quda_init()) */
  int id;                         /** Solver section identifier in the input file */
  QudaMultigridParam* mg_param;   /** Pointer to the multigrid param struct */
  double u1csw;                   /** u1csw property */
  int mg_ud_rev;                  /** Revision of ud field from openqxd */
  int mg_ad_rev;                  /** Revision of ad field from openqxd */
  double mg_kappa;                /** kappa corresponding to the current mg-instance in QUDA */
  double mg_su3csw;               /** SU(3) csw coefficient corresponding to the current mg-instance in QUDA */
  double mg_u1csw;                /** U(1) csw coefficient corresponding to the current mg-instance in QUDA */
} openQCD_QudaSolver;


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
 * @param[in]  layout  Layout struct
 * @param      infile  Input file
 */
void openQCD_qudaInit(openQCD_QudaInitArgs_t init, openQCD_QudaLayout_t layout, char *infile);


/**
 * Destroy the QUDA context and deallocate all solvers.
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
 * @brief      Wrapper around openqcd::ipt
 *
 * @param[in]  x     Euclidean corrdinate in txyz convention
 *
 * @return     ipt[x]
 *
 * @see        openqcd::ipt()
 */
int openQCD_qudaIndexIpt(const int *x);


/**
 * @brief      Wrapper around openqcd::iup
 *
 * @param[in]  x     Euclidean corrdinate in txyz convention
 * @param[in]  mu    Direction
 *
 * @return     iup[x][mu]
 *
 * @see        openqcd::iup()
 */
int openQCD_qudaIndexIup(const int *x, const int mu);


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
void openQCD_qudaDw_deprecated(void *src, void *dst, openQCD_QudaDiracParam_t p);


/**
 * @brief      Apply the Dirac operator that corresponds to the current openQxD
 *             setup to a field. All fields passed and returned are host (CPU)
 *             fields in openQCD order.
 *
 * @param[in]  mu    Twisted mass
 * @param      in    Input spinor
 * @param      out   Output spinor
 */
void openQCD_qudaDw(double mu, void *in, void *out);


/**
 * Setup the solver interface to quda.  This function parses the file given by
 * [infile] as an openQCD ini file.  The solver section given by the [id]
 * parameter must have a key-value pair like solver = QUDA and may contain every
 * member of the struct [QudaInvertParam].  If one sets inv_type_precondition =
 * QUDA_MG_INVERTER, one can additionally use all the members from the struct
 * [QudaMultigridParam] in a section called "Solver {id} Multigrid", where {id}
 * is replaced by [id].  For every level given by n_level in the above section,
 * one has to provide a subsection called "Solver {id} Multigrid Level {level}",
 * where {level} runs from 0 to n_level-1. All these subsections may have keys
 * given by all the array-valued members of QudaMultigridParam, for example
 * smoother_tol may appear in all subsections. This function must be called on
 * all ranks simulaneously.
 *
 * @param[in]  id    The identifier of the solver section, i.e. "Solver #". The
 *                   input file is taken from the arguments of quda_init(). If
 *                   id is -1, then the section called "Lattice parameters" is
 *                   parsed in the same way.
 *
 * @return     Pointer to the solver context
 */

void* openQCD_qudaSolverGetHandle(int id);


/**
 * @brief      Return a hash from a subset of the settings in the
 *             QudaInvertParam struct. Return 0 if the struct is not initialized
 *             yet.
 *
 * @param[in]  id    The solver identifier
 *
 * @return     Hash value
 */
int openQCD_qudaSolverGetHash(int id);


/**
 * @brief      Print solver information about the QUDA solver. Print
 *             "Solver is not initialized yet" is the solver struct is nul
 *             initialized yet.
 *
 * @param[in]  id    The solver identifier
 */
void openQCD_qudaSolverPrintSetup(int id);


/**
 * @brief      Solve Ax=b for an Clover Wilson operator with a multigrid solver.
 *             All fields passed and returned are host (CPU) field in openQCD
 *             order.
 *
 * @param[in]  id        The solver identifier in the input file, i.e.
 *                       "Solver #". The input file is the one given by
 *                       quda_init
 * @param[in]  mu        Twisted mass parameter
 * @param[in]  source    The source
 * @param[out] solution  The solution
 * @param[out] status    If the function is able to solve the Dirac equation to
 *                       the desired accuracy (invert_param->tol), status
 *                       reports the total number of iteration steps. -1
 *                       indicates that the inversion failed.
 *
 * @return     Residual
 */
double openQCD_qudaInvert(int id, double mu, void *source, void *solution, int *status);


/**
 * @brief      Destroys an existing solver context and frees all involed
 *             structs.
 *
 * @param[in]  id    The solver identifier
 */
void openQCD_qudaSolverDestroy(int id);


/**
 * Setup the eigen-solver interface to quda.  This function parses the file
 * given by [infile] as an openQCD ini file.  The solver section given by the
 * [inv_section] parameter must have a key-value pair like solver = QUDA and may
 * contain every member of the struct [QudaInvertParam].  See
 * [openQCD_qudaSolverSetup] for more details about the solver. The eigen-solver
 * section given by the [section] parameter may contain every member of the
 * struct [QudaEigParam].
 *
 * @param[in]  infile     Ini-file containing sections about the eigen-solver,
 *                        if null we use the value of qudaState.infile
 * @param[in]  section    The section name of the eigen-solver
 * @param[in]  solver_id  The section id of the solver. If -1, the section is
 *                        not read in.
 *
 * @return     Pointer to the eigen-solver context
 */
void* openQCD_qudaEigensolverSetup(char *infile, char *section, int solver_id);


/**
 * @brief        Solve Ax=b for an Clover Wilson operator with a multigrid
 *               solver. All fields are fields passed and returned are host
 *               (CPU) field in openQCD order.  This function requires an
 *               existing solver context created with openQCD_qudaSolverSetup().
 *
 * @param[inout] param    Pointer returned by openQCD_qudaEigensolverSetup()
 * @param[inout] h_evecs  Allocated array of void-pointers to param->n_conf
 *                        fields
 * @param[out]   h_evals  Allocated array of param->n_conf complex_dbles
 */
void openQCD_qudaEigensolve(void *param, void **h_evecs, void *h_evals);


/**
 * @brief      Destroys an existing eigen-solver context and frees all involed
 *             structs.
 *
 * @param      param  Pointer to the context to destroy
 */
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
 * @param[in]  gauge       The gauge fields (in openqcd order)
 * @param[in]  prec        Precision of the incoming gauge field
 * @param[in]  rec         How the field should be stored internally in QUDA
 * @param[in]  t_boundary  Time boundary condition
 */
void openQCD_qudaGaugeLoad(void *gauge, QudaPrecision prec, QudaReconstructType rec, QudaTboundary t_boundary);


/**
 * @brief      Save the gauge fields from quda to host.
 *
 * @param[out] gauge       The gauge fields (will be stored in openqcd order)
 * @param[in]  prec        Precision of the outgoing gauge field
 * @param[in]  rec         How the field should be stored internally in QUDA
 * @param[in]  t_boundary  Time boundary condition
 */
void openQCD_qudaGaugeSave(void *gauge, QudaPrecision prec, QudaReconstructType rec, QudaTboundary t_boundary);


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
