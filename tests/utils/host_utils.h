#pragma once

#include <vector>
#include <array>
#include <quda.h>
#include <random_quda.h>
#include <color_spinor_field.h>

constexpr size_t gauge_site_size = 18;      // real numbers per link
constexpr size_t spinor_site_size = 24;     // real numbers per wilson spinor
constexpr size_t stag_spinor_site_size = 6; // real numbers per staggered 'spinor'
constexpr size_t clover_site_size = 72;     // real numbers per block-diagonal clover matrix
constexpr size_t mom_site_size = 10;        // real numbers per momentum

extern int Z[4];
extern int V;
extern int Vh;
extern int Vs_x, Vs_y, Vs_z, Vs_t;
extern int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
extern int faceVolume[4];
extern int E1, E1h, E2, E3, E4;
extern int E[4];
extern int V_ex, Vh_ex;

extern double kappa5;
extern int Ls;
extern int V5;
extern int V5h;

extern size_t host_gauge_data_type_size;
extern size_t host_spinor_data_type_size;
extern size_t host_clover_data_type_size;

// QUDA precisions
extern QudaPrecision &cpu_prec;
extern QudaPrecision &cuda_prec;
extern QudaPrecision &cuda_prec_sloppy;
extern QudaPrecision &cuda_prec_precondition;
extern QudaPrecision &cuda_prec_eigensolver;
extern QudaPrecision &cuda_prec_refinement_sloppy;
extern QudaPrecision &cuda_prec_ritz;

// Set some basic parameters via command line or use defaults
// Implemented in set_params.cpp
void setQudaStaggeredEigTestParams();
void setQudaStaggeredInvTestParams();

// Staggered gauge field utils
//------------------------------------------------------
void constructStaggeredHostDeviceGaugeField(void **qdp_inlink, void **qdp_longlink_cpu, void **qdp_longlink_gpu,
                                            void **qdp_fatlink_cpu, void **qdp_fatlink_gpu, QudaGaugeParam &gauge_param,
                                            int argc, char **argv, bool &gauge_loaded);
void constructStaggeredHostGaugeField(void **qdp_inlink, void **qdp_longlink, void **qdp_fatlink,
                                      QudaGaugeParam &gauge_param, int argc, char **argv);
void constructFatLongGaugeField(void **fatlink, void **longlink, int type, QudaPrecision precision, QudaGaugeParam *,
                                QudaDslashType dslash_type);
void loadFatLongGaugeQuda(void *milc_fatlink, void *milc_longlink, QudaGaugeParam &gauge_param);
void computeLongLinkCPU(void **longlink, void **sitelink, QudaPrecision prec, void *act_path_coeff);
void computeHISQLinksCPU(void **fatlink, void **longlink, void **fatlink_eps, void **longlink_eps, void **sitelink,
                         void *qudaGaugeParamPtr, double **act_path_coeffs, double eps_naik);
void computeTwoLinkCPU(void **twolink, void **sitelink, QudaGaugeParam *gauge_param);
void staggeredTwoLinkGaussianSmear(quda::ColorSpinorField &out, void *qdp_twolnk[], void** ghost_twolnk,  quda::ColorSpinorField &in, QudaGaugeParam *qudaGaugeParam, QudaInvertParam *inv_param, const int oddBit, const double width, const int t0, QudaPrecision prec);
template <typename Float>
void applyGaugeFieldScaling_long(Float **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type);
void applyGaugeFieldScaling_long(void **gauge, int Vh, QudaGaugeParam *param, QudaDslashType dslash_type,
                                 QudaPrecision local_prec);
template <typename Float> void applyStaggeredScaling(Float **res, QudaGaugeParam *param, int type);
//------------------------------------------------------

// Spinor utils
//------------------------------------------------------
void constructStaggeredTestSpinorParam(quda::ColorSpinorParam *csParam, const QudaInvertParam *inv_param,
                                       const QudaGaugeParam *gauge_param);
//------------------------------------------------------

// MILC Data reordering routines
//------------------------------------------------------
void reorderQDPtoMILC(void *milc_out, void **qdp_in, int V, int siteSize, QudaPrecision out_precision,
                      QudaPrecision in_precision);
void reorderMILCtoQDP(void **qdp_out, void *milc_in, int V, int siteSize, QudaPrecision out_precision,
                      QudaPrecision in_precision);
//------------------------------------------------------

// Set some basic parameters via command line or use defaults
void setQudaPrecisions();
void setQudaMgSolveTypes();
void setQudaDefaultMgTestParams();

// Wilson type gauge and clover fields
//------------------------------------------------------
void constructQudaGaugeField(void **gauge, int type, QudaPrecision precision, QudaGaugeParam *param);
void constructHostGaugeField(void **gauge, QudaGaugeParam &gauge_param, int argc, char **argv);
void constructHostCloverField(void *clover, void *clover_inv, QudaInvertParam &inv_param);
void constructQudaCloverField(void *clover, double norm, double diag, QudaPrecision precision);
template <typename Float> void constructCloverField(Float *res, double norm, double diag);
template <typename Float> void constructUnitGaugeField(Float **res, QudaGaugeParam *param);
template <typename Float>
void constructRandomGaugeField(Float **res, QudaGaugeParam *param, QudaDslashType dslash_type = QUDA_WILSON_DSLASH);
template <typename Float> void applyGaugeFieldScaling(Float **gauge, int Vh, QudaGaugeParam *param);
//------------------------------------------------------

// Spinor utils
//------------------------------------------------------
void constructWilsonTestSpinorParam(quda::ColorSpinorParam *csParam, const QudaInvertParam *inv_param,
                                    const QudaGaugeParam *gauge_param);
void constructRandomSpinorSource(void *v, int nSpin, int nColor, QudaPrecision precision, QudaSolutionType sol_type,
                                 const int *const x, int nDim, quda::RNG &rng);
//------------------------------------------------------

// Helper functions
//------------------------------------------------------
inline bool isPCSolution(QudaSolutionType solution_type)
{
  return (solution_type == QUDA_MATPC_SOLUTION || solution_type == QUDA_MATPC_DAG_SOLUTION
          || solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
}
//------------------------------------------------------

// Reports basic statistics of flops and solver iterations
void performanceStats(std::vector<double> &time, std::vector<double> &gflops, std::vector<int> &iter);

void initComms(int argc, char **argv, std::array<int, 4> &commDims);
void initComms(int argc, char **argv, int *const commDims);
void finalizeComms();
void initRand();

int lex_rank_from_coords_t(const int *coords, void *fdata);
int lex_rank_from_coords_x(const int *coords, void *fdata);

void get_size_from_env(int *const dims, const char env[]);
void setDims(int *X);
void dw_setDims(int *X, const int L5);
int dimPartitioned(int dim);

bool last_node_in_t();

int index_4d_cb_from_coordinate_4d(const int coordinate[4], const int dim[4]);
void coordinate_from_shrinked_index(int coordinate[4], int shrinked_index, const int shrinked_dim[4],
                                    const int shift[4], int parity);

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1);

int neighborIndex(int dim[4], int index, int oddBit, int dx[4]);
int neighborIndexFullLattice(int dim[4], int index, int dx[4]);

int neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
int neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1);

void printSpinorElement(void *spinor, int X, QudaPrecision precision);
void printGaugeElement(void *gauge, int X, QudaPrecision precision);
template <typename Float> void printVector(Float *v);

int fullLatticeIndex(int i, int oddBit);
int fullLatticeIndex(int dim[4], int index, int oddBit);
int getOddBit(int X);

// Custom "sitelink" enum used to create unphased, MILC phased, or continuous U(1) phased links
enum {
  SITELINK_PHASE_NO = 0,   // no phase, used to create SU(3) links
  SITELINK_PHASE_MILC = 1, // MILC phase, used to test staggered fermions
  SITELINK_PHASE_U1 = 2    // continuous phase, used to test reconstruct 13
};

/**
   @brief Host implementation of creating a random set of gauge links, with optional phases
   @param[out] link QDP-ordered gauge links
   @param[in] precision Precision of field
   @param[in] phase Type of phase; 0 == no additional phase, 1 == MILC phases, 2 == U(1) phase
 */
void createSiteLinkCPU(void **link, QudaPrecision precision, int phase);
void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision);
void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision,
                     QudaGaugeParam *param);

void compare_spinor(void *spinor_cpu, void *spinor_gpu, int len, QudaPrecision precision);
void strong_check(void *spinor, void *spinorGPU, int len, QudaPrecision precision);
int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision);
double compare_floats_v2(void *a, void *b, int len, double epsilon, QudaPrecision precision);

void check_gauge(void **, void **, double epsilon, QudaPrecision precision);

int strong_check_link(void **linkA, const char *msgA, void **linkB, const char *msgB, int len, QudaPrecision prec);
int strong_check_mom(void *momA, void *momB, int len, QudaPrecision prec);

/**
   @brief Host reference implementation of the momentum action
   contribution.
 */
double mom_action(void *mom, QudaPrecision prec, int len);

void createMomCPU(void *mom, QudaPrecision precision);

/**
   @brief Create four Staggered spinor fields, whose outer product is used for momentum calculations
   @param[out] stag_for_oprod Set of four contiguous host spinor fields
   @param[in] precision Precision of field
   @param[in] x Full lattice volume
   @param[in] rng RNG
*/
void createStagForOprodCPU(void *stag_for_oprod, QudaPrecision precision, const int *const x, quda::RNG &rng);

// used by link fattening code
int x4_from_full_index(int i);

// additions for dw (quickly hacked on)
int fullLatticeIndex_4d(int i, int oddBit);
int fullLatticeIndex_5d(int i, int oddBit);
int fullLatticeIndex_5d_4dpc(int i, int oddBit);
int process_command_line_option(int argc, char **argv, int *idx);
int process_options(int argc, char **argv);

// Implemented in face_gauge.cpp
void exchange_cpu_sitelink(quda::lat_dim_t &X, void **sitelink, void **ghost_sitelink, void **ghost_sitelink_diag,
                           QudaPrecision gPrecision, QudaGaugeParam *param, int optflag);
void exchange_cpu_sitelink_ex(quda::lat_dim_t &X, quda::lat_dim_t &R, void **sitelink, QudaGaugeFieldOrder cpu_order,
                              QudaPrecision gPrecision, int optflag, int geometry);
void exchange_cpu_staple(quda::lat_dim_t &X, void *staple, void **ghost_staple, QudaPrecision gPrecision);
void exchange_llfat_init(QudaPrecision prec);
void exchange_llfat_cleanup(void);

// Implemented in host_blas.cpp
double norm_2(void *vector, int len, QudaPrecision precision);
void mxpy(void *x, void *y, int len, QudaPrecision precision);
void ax(double a, void *x, int len, QudaPrecision precision);
void cax(double _Complex a, void *x, int len, QudaPrecision precision);
void axpy(double a, void *x, void *y, int len, QudaPrecision precision);
void caxpy(double _Complex a, void *x, void *y, int len, QudaPrecision precision);
void xpay(void *x, double a, void *y, int len, QudaPrecision precision);
void cxpay(void *x, double _Complex a, void *y, int len, QudaPrecision precision);
void cpu_axy(QudaPrecision prec, double a, void *x, void *y, int size);
void cpu_xpy(QudaPrecision prec, void *x, void *y, int size);

inline QudaPrecision getPrecision(int i)
{
  switch (i) {
  case 0: return QUDA_QUARTER_PRECISION;
  case 1: return QUDA_HALF_PRECISION;
  case 2: return QUDA_SINGLE_PRECISION;
  case 3: return QUDA_DOUBLE_PRECISION;
  }
  return QUDA_INVALID_PRECISION;
}

inline int getReconstructNibble(QudaReconstructType recon)
{
  switch (recon) {
  case QUDA_RECONSTRUCT_NO: return 4;
  case QUDA_RECONSTRUCT_13:
  case QUDA_RECONSTRUCT_12: return 2;
  case QUDA_RECONSTRUCT_9:
  case QUDA_RECONSTRUCT_8: return 1;
  default: return 0;
  }
}

/**
  @brief Return the negative exponent of a reasonable expected tolerance for a given precision
  @param[in] prec Precision
  @return Exponent of the base-10 reasonably expected tolerance
*/
inline int getNegLog10Tolerance(QudaPrecision prec)
{
  switch (prec) {
  case QUDA_QUARTER_PRECISION: return 1;
  case QUDA_HALF_PRECISION: return 3;
  case QUDA_SINGLE_PRECISION: return 4;
  case QUDA_DOUBLE_PRECISION: return 11;
  case QUDA_INVALID_PRECISION: return 0;
  }
  return 0;
}

/**
  @brief Return the expected tolerance for a given precision consistent with the
    integer values in getNegLog10Tolerance.
  @param[in] prec Precision
  @return Reasonable expected tolerance
*/
inline double getTolerance(QudaPrecision prec) { return pow(10, -getNegLog10Tolerance(prec)); }

/**
  @brief Check if the std::string has a size smaller than the limit: if yes, copy it to a C-string;
    if no, give an error based on the given name. The 256 is the C-string length for parameters in
    QUDA's C interface.
  @param cstr the destination C-string
  @param str the input std::string
  @param limit the limit for the size check
  @param name the name used for the error message
 */
inline void safe_strcpy(char *cstr, const std::string &str, size_t limit, const std::string &name)
{
  if (str.size() < limit) {
    strcpy(cstr, str.c_str());
  } else {
    errorQuda("%s is longer (%lu) than the %lu limit.", name.c_str(), str.size(), limit);
  }
}

// MG param types
void setMultigridParam(QudaMultigridParam &mg_param);
void setStaggeredMultigridParam(QudaMultigridParam &mg_param);

// Eig param types
void setDeflationParam(QudaEigParam &df_param);
void setMultigridEigParam(QudaEigParam &eig_param, int level);
void setEigParam(QudaEigParam &eig_param);

// Invert param types
void setInvertParam(QudaInvertParam &inv_param);
void setContractInvertParam(QudaInvertParam &inv_param);
void setMultigridInvertParam(QudaInvertParam &inv_param);
void setDeflatedInvertParam(QudaInvertParam &inv_param);
void setStaggeredInvertParam(QudaInvertParam &inv_param);
void setStaggeredMGInvertParam(QudaInvertParam &inv_param);

// Gauge param types
void setGaugeParam(QudaGaugeParam &gauge_param);
void setWilsonGaugeParam(QudaGaugeParam &gauge_param);
void setStaggeredGaugeParam(QudaGaugeParam &gauge_param);
