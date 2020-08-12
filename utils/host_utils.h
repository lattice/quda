#pragma once

#include <quda.h>
#include <random_quda.h>
#include <vector>
#include <color_spinor_field.h>

#define gauge_site_size 18      // real numbers per link
#define spinor_site_size 24     // real numbers per wilson spinor
#define stag_spinor_site_size 6 // real numbers per staggered 'spinor'
#define clover_site_size 72     // real numbers per block-diagonal clover matrix
#define mom_site_size 10        // real numbers per momentum
#define hw_site_size 12         // real numbers per half wilson

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

extern int my_spinor_site_size;
extern size_t host_gauge_data_type_size;
extern size_t host_spinor_data_type_size;
extern size_t host_clover_data_type_size;

// QUDA precisions
extern QudaPrecision &cpu_prec;
extern QudaPrecision &cuda_prec;
extern QudaPrecision &cuda_prec_sloppy;
extern QudaPrecision &cuda_prec_precondition;
extern QudaPrecision &cuda_prec_refinement_sloppy;
extern QudaPrecision &cuda_prec_ritz;

// Set some basic parameters via command line or use defaults
// Implemented in set_params.cpp
void setQudaStaggeredEigTestParams();
void setQudaStaggeredInvTestParams();

// Staggered gauge field utils
//------------------------------------------------------
void constructStaggeredHostGhostGaugeField(quda::GaugeField *cpuFat, quda::GaugeField *cpuLong, void *milc_fatlink,
                                           void *milc_longlink, QudaGaugeParam &gauge_param);
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
void constructRandomSpinorSource(void *v, int nSpin, int nColor, QudaPrecision precision, const int *const x,
                                 quda::RNG &rng);
//------------------------------------------------------

void performanceStats(double *time, double *gflops);

void initComms(int argc, char **argv, std::array<int, 4> &commDims);
void initComms(int argc, char **argv, int *const commDims);
void finalizeComms();
void initRand();

int lex_rank_from_coords_t(const int *coords, void *fdata);
int lex_rank_from_coords_x(const int *coords, void *fdata);

void get_gridsize_from_env(int *const dims);
void setDims(int *X);
void dw_setDims(int *X, const int L5);
void setSpinorSiteSize(int n);
int dimPartitioned(int dim);

bool last_node_in_t();

int index_4d_cb_from_coordinate_4d(const int coordinate[4], const int dim[4]);
void coordinate_from_shrinked_index(int coordinate[4], int shrinked_index, const int shrinked_dim[4],
                                    const int shift[4], int parity);

int neighborIndex(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1);

int neighborIndex(int dim[], int index, int oddBit, int dx[]);
int neighborIndexFullLattice(int dim[], int index, int dx[]);

int neighborIndex_mg(int i, int oddBit, int dx4, int dx3, int dx2, int dx1);
int neighborIndexFullLattice_mg(int i, int dx4, int dx3, int dx2, int dx1);

void printSpinorElement(void *spinor, int X, QudaPrecision precision);
void printGaugeElement(void *gauge, int X, QudaPrecision precision);
template <typename Float> void printVector(Float *v);

int fullLatticeIndex(int i, int oddBit);
int fullLatticeIndex(int dim[], int index, int oddBit);
int getOddBit(int X);

void createSiteLinkCPU(void **link, QudaPrecision precision, int phase);
void su3_construct(void *mat, QudaReconstructType reconstruct, QudaPrecision precision);
void su3_reconstruct(void *mat, int dir, int ga_idx, QudaReconstructType reconstruct, QudaPrecision precision,
                     QudaGaugeParam *param);

void compare_spinor(void *spinor_cpu, void *spinor_gpu, int len, QudaPrecision precision);
void strong_check(void *spinor, void *spinorGPU, int len, QudaPrecision precision);
int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision);

void check_gauge(void **, void **, double epsilon, QudaPrecision precision);

int strong_check_link(void **linkA, const char *msgA, void **linkB, const char *msgB, int len, QudaPrecision prec);
int strong_check_mom(void *momA, void *momB, int len, QudaPrecision prec);

/**
   @brief Host reference implementation of the momentum action
   contribution.
 */
double mom_action(void *mom, QudaPrecision prec, int len);

void createMomCPU(void *mom, QudaPrecision precision);
void createHwCPU(void *hw, QudaPrecision precision);

// used by link fattening code
int x4_from_full_index(int i);

// additions for dw (quickly hacked on)
int fullLatticeIndex_4d(int i, int oddBit);
int fullLatticeIndex_5d(int i, int oddBit);
int fullLatticeIndex_5d_4dpc(int i, int oddBit);
int process_command_line_option(int argc, char **argv, int *idx);
int process_options(int argc, char **argv);

#ifdef __cplusplus
extern "C" {
#endif

// Implemented in face_gauge.cpp
void exchange_cpu_sitelink(int *X, void **sitelink, void **ghost_sitelink, void **ghost_sitelink_diag,
                           QudaPrecision gPrecision, QudaGaugeParam *param, int optflag);
void exchange_cpu_sitelink_ex(int *X, int *R, void **sitelink, QudaGaugeFieldOrder cpu_order, QudaPrecision gPrecision,
                              int optflag, int geometry);
void exchange_cpu_staple(int *X, void *staple, void **ghost_staple, QudaPrecision gPrecision);
void exchange_llfat_init(QudaPrecision prec);
void exchange_llfat_cleanup(void);

// Implemented in host_blas.cpp
double norm_2(void *vector, int len, QudaPrecision precision);
void mxpy(void *x, void *y, int len, QudaPrecision precision);
void ax(double a, void *x, int len, QudaPrecision precision);
void axpy(double a, void *x, void *y, int len, QudaPrecision precision);
void xpay(void *x, double a, void *y, int len, QudaPrecision precision);
void cxpay(void *x, double _Complex a, void *y, int len, QudaPrecision precision);
void cpu_axy(QudaPrecision prec, double a, void *x, void *y, int size);
void cpu_xpy(QudaPrecision prec, void *x, void *y, int size);

#ifdef __cplusplus
}
#endif

// Use for profiling
void stopwatchStart();
double stopwatchReadSeconds();
void performanceStats(double *time, double *gflops);

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

inline double getTolerance(QudaPrecision prec)
{
  switch (prec) {
  case QUDA_QUARTER_PRECISION: return 1e-1;
  case QUDA_HALF_PRECISION: return 1e-3;
  case QUDA_SINGLE_PRECISION: return 1e-4;
  case QUDA_DOUBLE_PRECISION: return 1e-11;
  case QUDA_INVALID_PRECISION: return 1.0;
  }
  return 1.0;
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
