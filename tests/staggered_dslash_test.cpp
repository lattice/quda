#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <misc.h>
#include <test_util.h>
#include <dslash_util.h>
#include <staggered_dslash_reference.h>
#include <llfat_reference.h>
#include <gauge_field.h>
#include <unitarization_links.h>

#include <qio_field.h>

#include <assert.h>
#include <gtest/gtest.h>

using namespace quda;

#define MAX(a,b) ((a)>(b)?(a):(b))
#define staggeredSpinorSiteSize 6
// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)

extern void usage(char** argv );

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaPrecision cpu_prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;

extern QudaDslashType dslash_type;

extern int test_type;

void *qdp_inlink[4] = { nullptr, nullptr, nullptr, nullptr };

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef, *tmpCpu;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;

cudaColorSpinorField* tmp;


void *hostGauge[4];

// In the HISQ case, we include building fat/long links in this unit test

void *qdp_fatlink_cpu[4], *qdp_longlink_cpu[4];
void **ghost_fatlink_cpu, **ghost_longlink_cpu;

QudaParity parity = QUDA_EVEN_PARITY;
extern QudaDagType dagger;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];

extern int device;
extern bool verify_results;
extern int niter;

extern double mass; // the mass of the Dirac operator
extern double kappa; // will get overriden
extern int laplace3D;

extern bool compute_fatlong; // build the true fat/long links or use random numbers

extern double tadpole_factor;
// relativistic correction for naik term
extern double eps_naik;
// Number of naiks. If eps_naik is 0.0, we only need
// to construct one naik.
static int n_naiks = 1;

extern char latfile[];


int X[4];
extern int Nsrc; // number of spinors to apply to simultaneously

extern QudaVerbosity verbosity;

Dirac* dirac;


// Unitarization coefficients
static double unitarize_eps  = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only  = false;
static double svd_rel_error  = 1e-4;
static double svd_abs_error  = 1e-4;
static double max_allowed_error = 1e-11;

// For loading the gauge fields
int argc_copy;
char** argv_copy;

double getTolerance(QudaPrecision prec)
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

// Wrap everything for the GPU construction of fat/long links here
void computeHISQLinksGPU(void** qdp_fatlink, void** qdp_longlink,
        void** qdp_fatlink_eps, void** qdp_longlink_eps,
        void** qdp_inlink, void* qudaGaugeParamPtr,
        double** act_path_coeffs, double eps_naik) {

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  // inlink in different format
  void *milc_inlink = pinned_malloc(4*V*gaugeSiteSize*gSize);
  reorderQDPtoMILC(milc_inlink,qdp_inlink,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);

  // Paths for step 1:
  void* milc_vlink  = pinned_malloc(4*V*gaugeSiteSize*gSize); // V links
  void* milc_wlink  = pinned_malloc(4*V*gaugeSiteSize*gSize); // W links

  // Paths for step 2:
  void* milc_fatlink = pinned_malloc(4*V*gaugeSiteSize*gSize); // final fat ("X") links
  void* milc_longlink = pinned_malloc(4*V*gaugeSiteSize*gSize); // final long links

  // Place to accumulate Naiks, step 3:
  void* milc_fatlink_eps = nullptr;
  void* milc_longlink_eps = nullptr;
  if (n_naiks > 1) {
    milc_fatlink_eps = pinned_malloc(4*V*gaugeSiteSize*gSize); // epsilon fat links
    milc_longlink_eps = pinned_malloc(4*V*gaugeSiteSize*gSize); // epsilon long naiks
  }

  // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
  computeKSLinkQuda(milc_vlink, nullptr, milc_wlink, milc_inlink, act_path_coeffs[0], &gaugeParam);

  if (n_naiks > 1) {
    // Create Naiks, 3rd path table set
    computeKSLinkQuda(milc_fatlink, milc_longlink, nullptr, milc_wlink, act_path_coeffs[2], &gaugeParam);

    // Rescale+copy Naiks into Naik field
    cpu_axy(gaugeParam.cpu_prec, eps_naik, milc_fatlink, milc_fatlink_eps, V*4*gaugeSiteSize);
    cpu_axy(gaugeParam.cpu_prec, eps_naik, milc_longlink, milc_longlink_eps, V*4*gaugeSiteSize);
  } else {
    memset(milc_fatlink, 0, V*4*gaugeSiteSize*gSize);
    memset(milc_longlink, 0, V*4*gaugeSiteSize*gSize);
  }

  // Create X and long links, 2nd path table set
  computeKSLinkQuda(milc_fatlink, milc_longlink, nullptr, milc_wlink, act_path_coeffs[1], &gaugeParam);

  if (n_naiks > 1) {
    // Add into Naik field
    cpu_xpy(gaugeParam.cpu_prec, milc_fatlink, milc_fatlink_eps, V*4*gaugeSiteSize);
    cpu_xpy(gaugeParam.cpu_prec, milc_longlink, milc_longlink_eps, V*4*gaugeSiteSize);
  }

  // Copy back
  reorderMILCtoQDP(qdp_fatlink,milc_fatlink,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  reorderMILCtoQDP(qdp_longlink,milc_longlink,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);

  if (n_naiks > 1) {
    // Add into Naik field
    reorderMILCtoQDP(qdp_fatlink_eps,milc_fatlink_eps,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
    reorderMILCtoQDP(qdp_longlink_eps,milc_longlink_eps,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  }

  // Clean up GPU compute links
  host_free(milc_inlink);
  host_free(milc_vlink);
  host_free(milc_wlink);
  host_free(milc_fatlink);
  host_free(milc_longlink);

  if (n_naiks > 1) {
    host_free(milc_fatlink_eps);
    host_free(milc_longlink_eps);
  }

}

void init()
{

  initQuda(device);

  gaugeParam = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gaugeParam.X[0] = X[0] = xdim;
  gaugeParam.X[1] = X[1] = ydim;
  gaugeParam.X[2] = X[2] = zdim;
  gaugeParam.X[3] = X[3] = tdim;

  setDims(gaugeParam.X);
  dw_setDims(gaugeParam.X,Nsrc); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;

    // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH &&
    dslash_type != QUDA_ASQTAD_DSLASH &&
    dslash_type != QUDA_LAPLACE_DSLASH) {
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  gaugeParam.anisotropy = 1.0;

  // For asqtad:
  //gaugeParam->tadpole_coeff = tadpole_coeff;
  //gaugeParam->scale = dslash_type != QUDA_ASQTAD_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  // For HISQ, this must always be set to 1.0, since the tadpole
  // correction is baked into the coefficients for the first fattening.
  // The tadpole doesn't mean anything for the second fattening
  // since the input fields are unitarized.
  gaugeParam.tadpole_coeff = 1.0;
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam.scale = -1.0/24.0;
    if (eps_naik != 0) {
      gaugeParam.scale *= (1.0+eps_naik);
    }
  } else {
    gaugeParam.scale = 1.0;
  }
  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam.type = QUDA_WILSON_LINKS;

  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = prec;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dagger = dagger;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dslash_type = dslash_type;
  inv_param.mass = mass;
  inv_param.kappa = kappa = 1.0/(8.0+mass); // for laplace
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;
  inv_param.laplace3D = laplace3D; // for laplace

  // set verbosity prior to loadGaugeQuda
  setVerbosity(verbosity);
  inv_param.verbosity = verbosity;

  /*if (test_type < 2) {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
  } else {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  }*/

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  int tmpint = MAX(X[1]*X[2]*X[3], X[0]*X[2]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[2]);


  gaugeParam.ga_pad = tmpint;
  inv_param.sp_pad = tmpint;

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  // Allocate a lot of memory because I'm very confused
  void* milc_fatlink_cpu = malloc(4*V*gaugeSiteSize*gSize);
  void* milc_longlink_cpu = malloc(4*V*gaugeSiteSize*gSize);

  void* milc_fatlink_gpu = malloc(4*V*gaugeSiteSize*gSize);
  void* milc_longlink_gpu = malloc(4*V*gaugeSiteSize*gSize);

  void* qdp_fatlink_gpu[4];
  void* qdp_longlink_gpu[4];

  for (int dir = 0; dir < 4; dir++) {
    qdp_fatlink_gpu[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_longlink_gpu[dir] = malloc(V*gaugeSiteSize*gSize);

    qdp_fatlink_cpu[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_longlink_cpu[dir] = malloc(V*gaugeSiteSize*gSize);

    if (qdp_fatlink_gpu[dir] == NULL || qdp_longlink_gpu[dir] == NULL ||
          qdp_fatlink_cpu[dir] == NULL || qdp_longlink_cpu[dir] == NULL) {
      errorQuda("ERROR: malloc failed for fatlink/longlink");
    }
  }

  // create a base field
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_inlink[dir] == nullptr) {
      qdp_inlink[dir] = malloc(V*gaugeSiteSize*gSize);
    }
  }

  // load a field WITHOUT PHASES
  if (strcmp(latfile,"")) {
    read_gauge_field(latfile, qdp_inlink, gaugeParam.cpu_prec, gaugeParam.X, argc_copy, argv_copy);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gaugeParam, QUDA_STAGGERED_DSLASH, gaugeParam.cpu_prec);
    } // else it's already been loaded
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      construct_gauge_field(qdp_inlink, 1, gaugeParam.cpu_prec, &gaugeParam);
    } else {
      construct_fat_long_gauge_field(qdp_inlink, qdp_longlink_cpu, 1, gaugeParam.cpu_prec,&gaugeParam,compute_fatlong ? QUDA_STAGGERED_DSLASH : dslash_type);
    }
  }

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink_gpu[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
      memcpy(qdp_fatlink_cpu[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
      memset(qdp_longlink_gpu[dir],0,V*gaugeSiteSize*gSize);
      memset(qdp_longlink_cpu[dir],0,V*gaugeSiteSize*gSize);
    }
  } else { // QUDA_ASQTAD_DSLASH

    if (compute_fatlong) {

      ///////////////////////////
      // Set path coefficients //
      ///////////////////////////

      // Reference: "generic_ks/imp_actions/hisq/hisq_action.h",
      // in QHMC: https://github.com/jcosborn/qhmc/blob/master/lib/qopqdp/hisq.c

      double u1 = 1.0/tadpole_factor;
      double u2 = u1*u1;
      double u4 = u2*u2;
      double u6 = u4*u2;

      // First path: create V, W links
      double act_path_coeff_1[6] = {
           ( 1.0/8.0),                 /* one link */
        u2*( 0.0),                     /* Naik */
        u2*(-1.0/8.0)*0.5,             /* simple staple */
        u4*( 1.0/8.0)*0.25*0.5,        /* displace link in two directions */
        u6*(-1.0/8.0)*0.125*(1.0/6.0), /* displace link in three directions */
        u4*( 0.0)                      /* Lepage term */
      };

      // Second path: create X, long links
      double act_path_coeff_2[6] = {
          ((1.0 / 8.0) + (2.0 * 6.0 / 16.0) + (1.0 / 8.0)), // one link
                                                            // One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik
          (-1.0 / 24.0),                      // Naik
          (-1.0 / 8.0) * 0.5,                 // simple staple
          (1.0 / 8.0) * 0.25 * 0.5,           // displace link in two directions
          (-1.0 / 8.0) * 0.125 * (1.0 / 6.0), // displace link in three directions
          (-2.0 / 16.0)                       // Lepage term, correct O(a^2) 2x ASQTAD
      };

      // Paths for epsilon corrections. Not used if n_naiks = 1.
      double act_path_coeff_3[6] = {
          (1.0 / 8.0),   // one link b/c of Naik
          (-1.0 / 24.0), // Naik
          0.0,           // simple staple
          0.0,           // displace link in two directions
          0.0,           // displace link in three directions
          0.0            // Lepage term
      };

      ////////////////////////////////////
      // Set unitarization coefficients //
      ////////////////////////////////////

      setUnitarizeLinksConstants(unitarize_eps,
               max_allowed_error,
               reunit_allow_svd,
               reunit_svd_only,
               svd_rel_error,
               svd_abs_error);

      ///////////////////////////////////////////////////////////////////////
      // Create some temporary space if we want to test the epsilon fields //
      ///////////////////////////////////////////////////////////////////////

      void* qdp_fatlink_naik_temp[4];
      void* qdp_longlink_naik_temp[4];
      if (n_naiks == 2) {
        for (int dir = 0; dir < 4; dir++) {
          qdp_fatlink_naik_temp[dir] = malloc(V*gaugeSiteSize*gSize);
          qdp_longlink_naik_temp[dir] = malloc(V*gaugeSiteSize*gSize);
          memset(qdp_fatlink_naik_temp[dir],0,V*gaugeSiteSize*gSize);
          memset(qdp_longlink_naik_temp[dir],0,V*gaugeSiteSize*gSize);
        }
      }

      //////////////////////////
      // Create the CPU links //
      //////////////////////////

      double* act_paths[3] = { act_path_coeff_1, act_path_coeff_2, act_path_coeff_3 };

      // defined in "llfat_reference.cpp"
      computeHISQLinksCPU(qdp_fatlink_cpu, qdp_longlink_cpu, (n_naiks == 2) ? qdp_fatlink_naik_temp : nullptr,
          (n_naiks == 2) ? qdp_longlink_naik_temp : nullptr, qdp_inlink, &gaugeParam, act_paths, eps_naik);

      if (n_naiks == 2) {
        // Override the naik fields into the fat/long link fields
        for (int dir = 0; dir < 4; dir++) {
          memcpy(qdp_fatlink_cpu[dir],qdp_fatlink_naik_temp[dir], V*gaugeSiteSize*gSize);
          memcpy(qdp_longlink_cpu[dir],qdp_longlink_naik_temp[dir], V*gaugeSiteSize*gSize);
          memset(qdp_fatlink_naik_temp[dir],0,V*gaugeSiteSize*gSize);
          memset(qdp_longlink_naik_temp[dir],0,V*gaugeSiteSize*gSize);
        }
      }

      //////////////////////////
      // Create the GPU links //
      //////////////////////////

      // Builds don't support reconstruct
      gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;

      // Skip eps field for now
      // Note: GPU link creation only works for single and double precision
      computeHISQLinksGPU(qdp_fatlink_gpu, qdp_longlink_gpu,
                          (n_naiks == 2) ? qdp_fatlink_naik_temp : nullptr,
                          (n_naiks == 2) ? qdp_longlink_naik_temp : nullptr,
                          qdp_inlink, &gaugeParam, act_paths, eps_naik);

      if (n_naiks == 2) {
        // Override the naik fields into the fat/long link fields
        for (int dir = 0; dir < 4; dir++) {
          memcpy(qdp_fatlink_gpu[dir],qdp_fatlink_naik_temp[dir], V*gaugeSiteSize*gSize);
          memcpy(qdp_longlink_gpu[dir],qdp_longlink_naik_temp[dir], V*gaugeSiteSize*gSize);
          free(qdp_fatlink_naik_temp[dir]); qdp_fatlink_naik_temp[dir] = nullptr;
          free(qdp_longlink_naik_temp[dir]); qdp_longlink_naik_temp[dir] = nullptr;
        }
      }



    } else { //

      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink_gpu[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
        memcpy(qdp_fatlink_cpu[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
        memcpy(qdp_longlink_gpu[dir],qdp_longlink_cpu[dir],V*gaugeSiteSize*gSize);
      }
    }

  }

  // Alright, we've created all the void** links.
  // Create the void* pointers
  reorderQDPtoMILC(milc_fatlink_gpu,qdp_fatlink_gpu,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  reorderQDPtoMILC(milc_fatlink_cpu,qdp_fatlink_cpu,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  reorderQDPtoMILC(milc_longlink_gpu,qdp_longlink_gpu,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  reorderQDPtoMILC(milc_longlink_cpu,qdp_longlink_cpu,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  // Create ghost zones for CPU fields,
  // prepare and load the GPU fields

#ifdef MULTI_GPU

  gaugeParam.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(milc_fatlink_cpu, gaugeParam);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink_cpu = cpuFat->Ghost();

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink_cpu, gaugeParam);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink_cpu = cpuLong->Ghost();

  int x_face_size = X[1]*X[2]*X[3]/2;
  int y_face_size = X[0]*X[2]*X[3]/2;
  int z_face_size = X[0]*X[1]*X[3]/2;
  int t_face_size = X[0]*X[1]*X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;
#endif

  gaugeParam.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
  if (dslash_type == QUDA_STAGGERED_DSLASH) {
#ifdef USE_LEGACY_DSLASH
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = link_recon;
#else
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = (link_recon == QUDA_RECONSTRUCT_12) ?
        QUDA_RECONSTRUCT_13 :
        (link_recon == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_9 : link_recon;
#endif
  } else {
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }

  // printfQuda("Fat links sending...");
  loadGaugeQuda(milc_fatlink_gpu, &gaugeParam);
  // printfQuda("Fat links sent\n");

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;

#ifdef MULTI_GPU
  gaugeParam.ga_pad = 3*pad_size;
#endif

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = (link_recon == QUDA_RECONSTRUCT_12) ?
        QUDA_RECONSTRUCT_13 :
        (link_recon == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_9 : link_recon;

    // printfQuda("Long links sending...");
    loadGaugeQuda(milc_longlink_gpu, &gaugeParam);
    // printfQuda("Long links sent...\n");
  }

  ColorSpinorParam csParam;
  csParam.nColor = 3;
  csParam.nSpin = 1;
  csParam.nDim = 5;
  for (int d = 0; d < 4; d++) { csParam.x[d] = gaugeParam.X[d]; }
  csParam.x[4] = Nsrc; // number of sources becomes the fifth dimension

  csParam.setPrecision(inv_param.cpu_prec);
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  csParam.pad = 0;
  if (test_type < 2 && dslash_type != QUDA_LAPLACE_DSLASH) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }

  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis; // this parameter is meaningless for staggered
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);
  tmpCpu = new cpuColorSpinorField(csParam);

  // printfQuda("Randomizing fields ...\n");

  spinor->Source(QUDA_RANDOM_SOURCE);

  csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  csParam.pad = inv_param.sp_pad;
  csParam.setPrecision(inv_param.cuda_prec);

  // printfQuda("Creating cudaSpinor\n");
  cudaSpinor = new cudaColorSpinorField(csParam);

  // printfQuda("Creating cudaSpinorOut\n");
  cudaSpinorOut = new cudaColorSpinorField(csParam);

  // printfQuda("Sending spinor field to GPU\n");
  *cudaSpinor = *spinor;

  cudaDeviceSynchronize();
  checkCudaError();

  // double spinor_norm2 = blas::norm2(*spinor);
  // double cuda_spinor_norm2=  blas::norm2(*cudaSpinor);
  // printfQuda("Source CPU = %f, CUDA=%f\n", spinor_norm2, cuda_spinor_norm2);

  // if(test_type == 2) csParam.x[0] /=2;

  // csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
  tmp = new cudaColorSpinorField(csParam);

  bool pc = (test_type == 1); // For test_type 0, can use either pc or not pc
                              // because both call the same "Dslash" directly.
  DiracParam diracParam;
  setDiracParam(diracParam, &inv_param, pc);

  diracParam.tmp1 = tmp;

  dirac = Dirac::create(diracParam);

  for (int dir = 0; dir < 4; dir++) {
    free(qdp_fatlink_gpu[dir]); qdp_fatlink_gpu[dir] = nullptr;
    free(qdp_longlink_gpu[dir]); qdp_longlink_gpu[dir] = nullptr;
  }
  free(milc_fatlink_gpu); milc_fatlink_gpu = nullptr;
  free(milc_longlink_gpu); milc_longlink_gpu = nullptr;
  free(milc_fatlink_cpu); milc_fatlink_cpu = nullptr;
  free(milc_longlink_cpu); milc_longlink_cpu = nullptr;

  gaugeParam.reconstruct = link_recon;

  return;
}

void end(void)
{
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_fatlink_cpu[dir] != nullptr) { free(qdp_fatlink_cpu[dir]); qdp_fatlink_cpu[dir] = nullptr; }
    if (qdp_longlink_cpu[dir] != nullptr) { free(qdp_longlink_cpu[dir]); qdp_longlink_cpu[dir] = nullptr; }
  }

  if (dirac != nullptr) {
    delete dirac;
    dirac = nullptr;
  }
  if (cudaSpinor != nullptr) {
    delete cudaSpinor;
    cudaSpinor = nullptr;
  }
  if (cudaSpinorOut != nullptr) {
    delete cudaSpinorOut;
    cudaSpinorOut = nullptr;
  }
  if (tmp != nullptr) {
    delete tmp;
    tmp = nullptr;
  }

  if (spinor != nullptr) { delete spinor; spinor = nullptr; }
  if (spinorOut != nullptr) { delete spinorOut; spinorOut = nullptr; }
  if (spinorRef != nullptr) { delete spinorRef; spinorRef = nullptr; }
  if (tmpCpu != nullptr) { delete tmpCpu; tmpCpu = nullptr; }

  freeGaugeQuda();

  if (cpuFat) { delete cpuFat; cpuFat = nullptr; }
  if (cpuLong) { delete cpuLong; cpuLong = nullptr; }
  commDimPartitionedReset();

  endQuda();
}

struct DslashTime {
  double event_time;
  double cpu_time;
  double cpu_min;
  double cpu_max;

  DslashTime() : event_time(0.0), cpu_time(0.0), cpu_min(DBL_MAX), cpu_max(0.0) {}
};

DslashTime dslashCUDA(int niter) {

  DslashTime dslash_time;
  timeval tstart, tstop;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  comm_barrier();
  cudaEventRecord(start, 0);

  for (int i = 0; i < niter; i++) {

    gettimeofday(&tstart, NULL);

    switch (test_type) {
    case 0: dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity); break;
    case 1: dirac->M(*cudaSpinorOut, *cudaSpinor); break;
    case 2: dirac->M(*cudaSpinorOut, *cudaSpinor); break;
    }

    gettimeofday(&tstop, NULL);
    long ds = tstop.tv_sec - tstart.tv_sec;
    long dus = tstop.tv_usec - tstart.tv_usec;
    double elapsed = ds + 0.000001*dus;

    dslash_time.cpu_time += elapsed;
    // skip first and last iterations since they may skew these metrics if comms are not synchronous
    if (i>0 && i<niter) {
      if (elapsed < dslash_time.cpu_min) dslash_time.cpu_min = elapsed;
      if (elapsed > dslash_time.cpu_max) dslash_time.cpu_max = elapsed;
    }
  }

  cudaEventCreate(&end);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  dslash_time.event_time = runTime / 1000;

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    errorQuda("with ERROR: %s\n", cudaGetErrorString(stat));

  return dslash_time;
}

void staggeredDslashRef()
{

  // compare to dslash reference implementation
  // printfQuda("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
    case 0:
      staggered_dslash(spinorRef, qdp_fatlink_cpu, qdp_longlink_cpu, ghost_fatlink_cpu, ghost_longlink_cpu, spinor,
          parity, dagger, inv_param.cpu_prec, gaugeParam.cpu_prec, dslash_type);
      break;
    case 1:
      matdagmat(spinorRef, qdp_fatlink_cpu, qdp_longlink_cpu, ghost_fatlink_cpu, ghost_longlink_cpu, spinor, mass, 0,
          inv_param.cpu_prec, gaugeParam.cpu_prec, tmpCpu, parity, dslash_type);
      break;
    case 2:
      // Not sure about the !dagger...
      staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&spinorRef->Even()), qdp_fatlink_cpu, qdp_longlink_cpu,
          ghost_fatlink_cpu, ghost_longlink_cpu, reinterpret_cast<cpuColorSpinorField *>(&spinor->Odd()),
          QUDA_EVEN_PARITY, !dagger, inv_param.cpu_prec, gaugeParam.cpu_prec, dslash_type);
      staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&spinorRef->Odd()), qdp_fatlink_cpu, qdp_longlink_cpu,
          ghost_fatlink_cpu, ghost_longlink_cpu, reinterpret_cast<cpuColorSpinorField *>(&spinor->Even()),
          QUDA_ODD_PARITY, !dagger, inv_param.cpu_prec, gaugeParam.cpu_prec, dslash_type);
      if (dslash_type == QUDA_LAPLACE_DSLASH) {
        xpay(spinor->V(), kappa, spinorRef->V(), spinor->Length(), gaugeParam.cpu_prec);
      } else {
        axpy(2*mass, spinor->V(), spinorRef->V(), spinor->Length(), gaugeParam.cpu_prec);
      }
      break;
    default:
      errorQuda("Test type not defined");
  }

}

TEST(dslash, verify) {
  double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
  double tol
      = (inv_param.cuda_prec == QUDA_DOUBLE_PRECISION ? 1e-11 :
                                                        (inv_param.cuda_prec == QUDA_SINGLE_PRECISION ? 1e-4 : 1e-2));
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

static int dslashTest()
{

  for (int dir = 0; dir < 4; dir++) {
    qdp_fatlink_cpu[dir] = nullptr;
    qdp_longlink_cpu[dir] = nullptr;
  }

  dirac = nullptr;
  cudaSpinor = nullptr;
  cudaSpinorOut = nullptr;
  tmp = nullptr;

  spinor = nullptr;
  spinorOut = nullptr;
  spinorRef = nullptr;
  tmpCpu = nullptr;

  bool failed = false;

  // return code for google test
  int test_rc = 0;
  init();

  int attempts = 1;

  for (int i=0; i<attempts; i++) {

    { // warm-up run
      printfQuda("Tuning...\n");
      dslashCUDA(1);
    }
    printfQuda("Executing %d kernel loops...", niter);

    // reset flop counter
    dirac->Flops();

    DslashTime dslash_time = dslashCUDA(niter);

    *spinorOut = *cudaSpinorOut;

    printfQuda("%fus per kernel call\n", 1e6*dslash_time.event_time / niter);
    staggeredDslashRef();

    double spinor_ref_norm2 = blas::norm2(*spinorRef);
    double spinor_out_norm2 = blas::norm2(*spinorOut);

    // Catching nans is weird.
    if (std::isnan(spinor_ref_norm2)) { failed = true; }
    if (std::isnan(spinor_out_norm2)) { failed = true; }

    unsigned long long flops = dirac->Flops();
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/dslash_time.event_time);

    if (niter > 2) { // only print this if valid
      printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for aggregate message size %lu bytes\n",
                 1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.event_time, 1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.cpu_time,
                 1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_max, 1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_min,
                 2*cudaSpinor->GhostBytes());
    }

    double cuda_spinor_out_norm2 = blas::norm2(*cudaSpinorOut);
    printfQuda("Results: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", spinor_ref_norm2, cuda_spinor_out_norm2, spinor_out_norm2);

    if (verify_results) {
      test_rc = RUN_ALL_TESTS();
      if (test_rc != 0 || failed) warningQuda("Tests failed");
    }
  }
  end();

  return test_rc;
}


void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %d           %d       %d/%d/%d        %d \n", get_prec_str(prec), get_recon_str(link_recon),
      test_type, dagger, xdim, ydim, zdim, tdim);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
      dimPartitioned(3));

  return ;

}


void usage_extra(char** argv )
{
  printfQuda("Extra options:\n");
  printfQuda("    --test <0/1/2>                              # Test method\n");
  printfQuda("                                                0: Even destination spinor\n");
  printfQuda("                                                1: Odd destination spinor\n");
  printfQuda("                                                2: Full spinor\n");
  return ;
}

int main(int argc, char **argv)
{
  // hack for loading gauge fields
  argc_copy = argc;
  argv_copy = argv;


  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  for (int i=1 ;i < argc; i++){

    if (process_command_line_option(argc, argv, &i) == 0) { continue; }

    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure a reasonable default
    // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH &&
      dslash_type != QUDA_ASQTAD_DSLASH &&
      dslash_type != QUDA_LAPLACE_DSLASH) {
    warningQuda("The dslash_type %d isn't staggered, asqtad, or laplace. Defaulting to asqtad.\n", dslash_type);
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  // Sanity checkL: if you pass in a gauge field, want to test the asqtad/hisq dslash, and don't
  // ask to build the fat/long links... it doesn't make sense.
  if (strcmp(latfile,"") && !compute_fatlong && dslash_type == QUDA_ASQTAD_DSLASH) {
    errorQuda("Cannot load a gauge field and test the ASQTAD/HISQ operator without setting \"--compute-fat-long true\".\n");
  }

  // Set n_naiks to 2 if eps_naik != 0.0
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (eps_naik != 0.0) {
      if (compute_fatlong) {
        n_naiks = 2;
        printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
      } else {
        eps_naik = 0.0;
        printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
      }
    } else {
      printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
    }
  }

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    if (test_type != 2) {
      errorQuda("Test type %d is not supported for the Laplace operator.\n", test_type);
    }
  }

  // If we're building fat/long links, there are some
  // tests we have to skip.
  if (dslash_type == QUDA_ASQTAD_DSLASH && compute_fatlong) {
    if (prec == QUDA_HALF_PRECISION /* half */) {
      errorQuda("Half precision unsupported in fat/long compute");
    }
  }
  if (dslash_type == QUDA_LAPLACE_DSLASH && prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision unsupported for Laplace operator.\n");
  }

  display_test_info();

  // return result of RUN_ALL_TESTS
  int test_rc = dslashTest();

  finalizeComms();

  return test_rc;
}
