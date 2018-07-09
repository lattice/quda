#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "quda.h"
#include "gauge_field.h"
#include "test_util.h"
#include "llfat_reference.h"
#include "misc.h"
#include "util_quda.h"
#include "malloc_quda.h"
#include <unitarization_links.h>
#include "dslash_quda.h"
#include "ks_improved_force.h"

#ifdef MULTI_GPU
#include "comm_quda.h"
#endif

#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

using namespace quda;

extern void usage(char** argv);
extern bool verify_results;

extern int device;
extern int xdim, ydim, zdim, tdim;
extern int gridsize_from_cmdline[];

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern int niter;

// relativistic correction for naik term
extern double eps_naik;
// Number of naiks. If eps_naik is 0.0, we only need
// to construct one naik.
static int n_naiks = 1;

static QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
static QudaGaugeFieldOrder gauge_order = QUDA_MILC_GAUGE_ORDER;

static size_t gSize;

// The file "generic_ks/fermion_links_hisq_load_milc.c" 
// within MILC is the ultimate reference for what's going on here.

// Unitarization coefficients
static double unitarize_eps  = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only  = false;
static double svd_rel_error  = 1e-4;
static double svd_abs_error  = 1e-4;
static double max_allowed_error = 1e-11;



// CPU-style BLAS routines
void cpu_axy(QudaPrecision prec, double a, void* x, void* y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double* dst = (double*)y;
    double* src = (double*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] = a*src[i];
    }
  } else { // QUDA_SINGLE_PRECISION
    float* dst = (float*)y;
    float* src = (float*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] = a*src[i];
    }
  }
}

void cpu_xpy(QudaPrecision prec, void* x, void* y, int size)
{
  if (prec == QUDA_DOUBLE_PRECISION) {
    double* dst = (double*)y;
    double* src = (double*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] += src[i];
    }
  } else { // QUDA_SINGLE_PRECISION
    float* dst = (float*)y;
    float* src = (float*)x;
    for (int i = 0; i < size; i++)
    {
      dst[i] += src[i];
    }
  }
}


/*--------------------------------------------------------------------*/
// Some notation:
// U -- original link, SU(3), copied to "field" from "site"
// V -- after 1st level of smearing, non-SU(3)
// W -- unitarized, SU(3)
// X -- after 2nd level of smearing, non-SU(3)
/*--------------------------------------------------------------------*/

static void hisq_test()
{

  QudaGaugeParam qudaGaugeParam;

  initQuda(device);

  cpu_prec = prec;
  gSize = cpu_prec;  
  qudaGaugeParam = newQudaGaugeParam();

  qudaGaugeParam.anisotropy = 1.0;

  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;

  setDims(qudaGaugeParam.X);

  qudaGaugeParam.cpu_prec = cpu_prec;
  qudaGaugeParam.cuda_prec = qudaGaugeParam.cuda_prec_sloppy = prec;

  if (gauge_order != QUDA_MILC_GAUGE_ORDER)
    errorQuda("Unsupported gauge order %d", gauge_order);

  qudaGaugeParam.gauge_order = gauge_order;
  qudaGaugeParam.type = QUDA_WILSON_LINKS;
  qudaGaugeParam.reconstruct = qudaGaugeParam.reconstruct_sloppy = link_recon;
  qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_MILC;
  qudaGaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  qudaGaugeParam.ga_pad = 0;

  // Needed for unitarization, following "unitarize_link_test.cpp"
  GaugeFieldParam gParam(0, qudaGaugeParam);
  gParam.pad = 0;
  gParam.link_type   = QUDA_GENERAL_LINKS;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.order = gauge_order;

  ///////////////////////////////////////////////////////////////
  // Set up the coefficients for each part of the HISQ stencil //
  ///////////////////////////////////////////////////////////////
  
  // Reference: "generic_ks/imp_actions/hisq/hisq_action.h"

  // First path: create V, W links 
  double act_path_coeff_1[6] = {
    ( 1.0/8.0),                 /* one link */
      0.0,                      /* Naik */
    (-1.0/8.0)*0.5,             /* simple staple */
    ( 1.0/8.0)*0.25*0.5,        /* displace link in two directions */
    (-1.0/8.0)*0.125*(1.0/6.0), /* displace link in three directions */
      0.0                       /* Lepage term */
  };

  // Second path: create X, long links
  double act_path_coeff_2[6] = {
    (( 1.0/8.0)+(2.0*6.0/16.0)+(1.0/8.0)),   /* one link */
        /* One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik */
    (-1.0/24.0),                             /* Naik */
    (-1.0/8.0)*0.5,                          /* simple staple */
    ( 1.0/8.0)*0.25*0.5,                     /* displace link in two directions */
    (-1.0/8.0)*0.125*(1.0/6.0),              /* displace link in three directions */
    (-2.0/16.0)                              /* Lepage term, correct O(a^2) 2x ASQTAD */
  };

  // Paths for epsilon corrections. Not used if n_naiks = 1.
  double act_path_coeff_3[6] = {
    ( 1.0/8.0),    /* one link b/c of Naik */
    (-1.0/24.0),   /* Naik */
      0.0,         /* simple staple */
      0.0,         /* displace link in two directions */
      0.0,         /* displace link in three directions */
      0.0          /* Lepage term */
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

  /////////////////
  // Input links //
  /////////////////

  void* sitelink[4];
  for(int i=0;i < 4;i++) sitelink[i] = pinned_malloc(V*gaugeSiteSize*gSize);

  void* sitelink_ex[4];
  for(int i=0;i < 4;i++) sitelink_ex[i] = pinned_malloc(V_ex*gaugeSiteSize*gSize);

  void* milc_sitelink;
  milc_sitelink = (void*)safe_malloc(4*V*gaugeSiteSize*gSize);

#ifdef MULTI_GPU
  void* ghost_sitelink[4];
  void* ghost_sitelink_diag[16];
  void* ghost_wlink[4];
  void* ghost_wlink_diag[16];
#endif

  // Note: this could be replaced with loading a gauge field
  createSiteLinkCPU(sitelink, qudaGaugeParam.cpu_prec, 0); // 0 -> no phases
  for(int i=0; i<V; ++i){
    for(int dir=0; dir<4; ++dir){
      char* src = (char*)sitelink[dir];
      memcpy((char*)milc_sitelink + (i*4 + dir)*gaugeSiteSize*gSize, src+i*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
    }	
  }

  //////////////////////
  // Perform GPU test //
  //////////////////////

  // Paths for step 1:
  void* vlink  = pinned_malloc(4*V*gaugeSiteSize*gSize); // V links
  void* wlink  = pinned_malloc(4*V*gaugeSiteSize*gSize); // W links
  
  // Paths for step 2:
  void* fatlink = pinned_malloc(4*V*gaugeSiteSize*gSize); // final fat ("X") links
  void* longlink = pinned_malloc(4*V*gaugeSiteSize*gSize); // final long links

  // Place to accumulate Naiks
  void* fatlink_eps = nullptr;
  void* longlink_eps = nullptr;
  if (n_naiks > 1) {
    fatlink_eps = pinned_malloc(4*V*gaugeSiteSize*gSize); // epsilon fat links
    longlink_eps = pinned_malloc(4*V*gaugeSiteSize*gSize); // epsilon long naiks
  }
  
  // Tuning run...
  {
    printfQuda("Tuning...\n");
    computeKSLinkQuda(vlink , longlink, wlink, milc_sitelink, act_path_coeff_2, &qudaGaugeParam);
  }

  struct timeval t0, t1;
  printfQuda("Running %d iterations of computation\n", niter);
  gettimeofday(&t0, NULL);
  for (int n = 0; n < niter; n++) {

    // If we create cudaGaugeField objs, we can do this 100% on the GPU, no copying!

    // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
    computeKSLinkQuda(vlink, nullptr, wlink, milc_sitelink, act_path_coeff_1, &qudaGaugeParam);

    if (n_naiks > 1) {
      // Create Naiks, 3rd path table set
      computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_path_coeff_3, &qudaGaugeParam);

      // Rescale+copy Naiks into Naik field
      cpu_axy(prec, eps_naik, fatlink, fatlink_eps, V*4*gaugeSiteSize);
      cpu_axy(prec, eps_naik, longlink, longlink_eps, V*4*gaugeSiteSize);
    } else {
      memset(fatlink, 0, V*4*gaugeSiteSize*gSize);
      memset(longlink, 0, V*4*gaugeSiteSize*gSize);
    }

    // Create X and long links, 2nd path table set
    computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_path_coeff_2, &qudaGaugeParam);

    if (n_naiks > 1) {
      // Add into Naik field
      cpu_xpy(prec, fatlink, fatlink_eps, V*4*gaugeSiteSize);
      cpu_xpy(prec, longlink, longlink_eps, V*4*gaugeSiteSize);
    }
  }
  gettimeofday(&t1, NULL);

  double secs = TDIFF(t0,t1);

  ////////////////////
  // Begin CPU test //
  ////////////////////


  ///////////////////////////////
  // Create extended CPU field //
  ///////////////////////////////
  int X1=Z[0];
  int X2=Z[1];
  int X3=Z[2];
  int X4=Z[3];

  for(int i=0; i < V_ex; i++){
    int sid = i;
    int oddBit=0;
    if(i >= Vh_ex){
      sid = i - Vh_ex;
      oddBit = 1;
    }

    int za = sid/E1h;
    int x1h = sid - za*E1h;
    int zb = za/E2;
    int x2 = za - zb*E2;
    int x4 = zb/E3;
    int x3 = zb - x4*E3;
    int x1odd = (x2 + x3 + x4 + oddBit) & 1;
    int x1 = 2*x1h + x1odd;


    if( x1< 2 || x1 >= X1 +2
        || x2< 2 || x2 >= X2 +2
        || x3< 2 || x3 >= X3 +2
        || x4< 2 || x4 >= X4 +2){
#ifdef MULTI_GPU
      continue;
#endif
    }



    x1 = (x1 - 2 + X1) % X1;
    x2 = (x2 - 2 + X2) % X2;
    x3 = (x3 - 2 + X3) % X3;
    x4 = (x4 - 2 + X4) % X4;

    int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
    if(oddBit){
      idx += Vh;
    }
    for(int dir= 0; dir < 4; dir++){
      char* src = (char*)sitelink[dir];
      char* dst = (char*)sitelink_ex[dir];
      memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
    }//dir
  }//i

  /////////////////////////////////////
  // Allocate all CPU intermediaries //
  /////////////////////////////////////

  void* v_reflink[4];         // V link -- fat7 smeared link
  void* w_reflink[4];         // unitarized V link
  void* w_reflink_ex[4];      // extended W link
  void* long_reflink[4];      // Final long link
  void* fat_reflink[4];       // Final fat link
  void* long_reflink_eps[4];  // Long link for fermion with non-zero epsilon
  void* fat_reflink_eps[4];   // Fat link for fermion with non-zero epsilon
  for(int i=0;i < 4;i++){
    v_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    w_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    w_reflink_ex[i] = safe_malloc(V_ex*gaugeSiteSize*gSize);
    long_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    fat_reflink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    if (n_naiks > 1) {
      long_reflink_eps[i] = safe_malloc(V*gaugeSiteSize*gSize);
      fat_reflink_eps[i] = safe_malloc(V*gaugeSiteSize*gSize);
    }
  }

  // Copy of V link needed for CPU unitarization routines
  void* v_sitelink = pinned_malloc(4*V*gaugeSiteSize*gSize);


  if (verify_results){

    //FIXME: we have this complication because references takes coeff as float/double
    //        depending on the precision while the GPU code aways take coeff as double
    void* coeff;
    double coeff_dp[6];
    float  coeff_sp[6];

    /////////////////////////////////////////////////////
    // Create V links (fat7 links), 1st path table set //
    /////////////////////////////////////////////////////

    for (int i=0; i < 6;i++) coeff_sp[i] = coeff_dp[i] = act_path_coeff_1[i];
    coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void*)coeff_dp : (void*)coeff_sp;

    // Only need fat links.
#ifdef MULTI_GPU
    int optflag = 0;
    //we need x,y,z site links in the back and forward T slice
    // so it is 3*2*Vs_t
    int Vs[4] = {Vs_x, Vs_y, Vs_z, Vs_t};
    for (int i=0; i < 4; i++) ghost_sitelink[i] = safe_malloc(8*Vs[i]*gaugeSiteSize*gSize);

    /*
       nu |     |
          |_____|
            mu
       */

    for(int nu=0;nu < 4;nu++){
      for(int mu=0; mu < 4;mu++){
        if(nu == mu){
          ghost_sitelink_diag[nu*4+mu] = NULL;
        }else{
          //the other directions
          int dir1, dir2;
          for(dir1= 0; dir1 < 4; dir1++){
            if(dir1 !=nu && dir1 != mu){
              break;
            }
          }
          for(dir2=0; dir2 < 4; dir2++){
            if(dir2 != nu && dir2 != mu && dir2 != dir1){
              break;
            }
          }
          ghost_sitelink_diag[nu*4+mu] = safe_malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
          memset(ghost_sitelink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
        }

      }
    }
    exchange_cpu_sitelink(qudaGaugeParam.X, sitelink, ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, &qudaGaugeParam, optflag);
    llfat_reference_mg(v_reflink, sitelink, ghost_sitelink, ghost_sitelink_diag, qudaGaugeParam.cpu_prec, coeff);
#else
    llfat_reference(v_reflink, sitelink, qudaGaugeParam.cpu_prec, coeff);
#endif

    /////////////////////////////////////////
    // Create W links (unitarized V links) //
    /////////////////////////////////////////

    // This is based on "unitarize_link_test.cpp"

    // Format change
    if (prec == QUDA_DOUBLE_PRECISION){
      double* link = reinterpret_cast<double*>(v_sitelink);
      for(int dir=0; dir<4; ++dir){
        double* slink = reinterpret_cast<double*>(v_reflink[dir]);
        for(int i=0; i<V; ++i){
          for(int j=0; j<gaugeSiteSize; j++){
            link[(i*4 + dir)*gaugeSiteSize + j] = slink[i*gaugeSiteSize + j];
          }
        }
      }
    } else if(prec == QUDA_SINGLE_PRECISION){
      float* link = reinterpret_cast<float*>(v_sitelink);
      for(int dir=0; dir<4; ++dir){
        float* slink = reinterpret_cast<float*>(v_reflink[dir]);
        for(int i=0; i<V; ++i){
          for(int j=0; j<gaugeSiteSize; j++){
            link[(i*4 + dir)*gaugeSiteSize + j] = slink[i*gaugeSiteSize + j];
          }
        }
      }
    }

    // Prepare cpuGaugeFields for unitarization
    gParam.create = QUDA_REFERENCE_FIELD_CREATE;
    gParam.gauge = v_sitelink;
    cpuGaugeField* cpuVLink = new cpuGaugeField(gParam);

    gParam.create = QUDA_ZERO_FIELD_CREATE;
    cpuGaugeField* cpuWLink = new cpuGaugeField(gParam);

    // unitarize
    unitarizeLinksCPU(*cpuWLink, *cpuVLink);

    // Copy back into "w_reflink"
    if (prec == QUDA_DOUBLE_PRECISION){
      double* link = reinterpret_cast<double*>(cpuWLink->Gauge_p());
      for(int dir=0; dir<4; ++dir){
        double* slink = reinterpret_cast<double*>(w_reflink[dir]);
        for(int i=0; i<V; ++i){
          for(int j=0; j<gaugeSiteSize; j++){
            slink[i*gaugeSiteSize + j] = link[(i*4 + dir)*gaugeSiteSize + j];
          }
        }
      }
    } else if(prec == QUDA_SINGLE_PRECISION){
      float* link = reinterpret_cast<float*>(cpuWLink->Gauge_p());
      for(int dir=0; dir<4; ++dir){
        float* slink = reinterpret_cast<float*>(w_reflink[dir]);
        for(int i=0; i<V; ++i){
          for(int j=0; j<gaugeSiteSize; j++){
            slink[i*gaugeSiteSize + j] = link[(i*4 + dir)*gaugeSiteSize + j];
          }
        }
      }
    }


    // Clean up cpuGaugeFields, we don't need them anymore.

    delete cpuVLink;
    delete cpuWLink;

    ///////////////////////////////////
    // Prepare for extended W fields //
    ///////////////////////////////////

    for(int i=0; i < V_ex; i++) {
      int sid = i;
      int oddBit=0;
      if(i >= Vh_ex){
        sid = i - Vh_ex;
        oddBit = 1;
      }

      int za = sid/E1h;
      int x1h = sid - za*E1h;
      int zb = za/E2;
      int x2 = za - zb*E2;
      int x4 = zb/E3;
      int x3 = zb - x4*E3;
      int x1odd = (x2 + x3 + x4 + oddBit) & 1;
      int x1 = 2*x1h + x1odd;


      if( x1< 2 || x1 >= X1 +2
          || x2< 2 || x2 >= X2 +2
          || x3< 2 || x3 >= X3 +2
          || x4< 2 || x4 >= X4 +2){
  #ifdef MULTI_GPU
        continue;
  #endif
      }



      x1 = (x1 - 2 + X1) % X1;
      x2 = (x2 - 2 + X2) % X2;
      x3 = (x3 - 2 + X3) % X3;
      x4 = (x4 - 2 + X4) % X4;

      int idx = (x4*X3*X2*X1+x3*X2*X1+x2*X1+x1)>>1;
      if(oddBit){
        idx += Vh;
      }
      for(int dir= 0; dir < 4; dir++){
        char* src = (char*)w_reflink[dir];
        char* dst = (char*)w_reflink_ex[dir];
        memcpy(dst+i*gaugeSiteSize*gSize, src+idx*gaugeSiteSize*gSize, gaugeSiteSize*gSize);
      }//dir
    }//i

    //////////////////////////////
    // Create extended W fields //
    //////////////////////////////

#ifdef MULTI_GPU
    optflag = 0;
    //we need x,y,z site links in the back and forward T slice
    // so it is 3*2*Vs_t
    for (int i=0; i < 4; i++) ghost_wlink[i] = safe_malloc(8*Vs[i]*gaugeSiteSize*gSize);

    /*
       nu |     |
          |_____|
            mu
       */

    for(int nu=0;nu < 4;nu++){
      for(int mu=0; mu < 4;mu++){
        if(nu == mu){
          ghost_wlink_diag[nu*4+mu] = NULL;
        }else{
          //the other directions
          int dir1, dir2;
          for(dir1= 0; dir1 < 4; dir1++){
            if(dir1 !=nu && dir1 != mu){
              break;
            }
          }
          for(dir2=0; dir2 < 4; dir2++){
            if(dir2 != nu && dir2 != mu && dir2 != dir1){
              break;
            }
          }
          ghost_wlink_diag[nu*4+mu] = safe_malloc(Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
          memset(ghost_wlink_diag[nu*4+mu], 0, Z[dir1]*Z[dir2]*gaugeSiteSize*gSize);
        }

      }
    }
#endif

    ////////////////////////////////////////////
    // Prepare to create Naiks, 3rd table set //
    ////////////////////////////////////////////

    if (n_naiks > 1) {

      for (int i=0; i < 6;i++) coeff_sp[i] = coeff_dp[i] = act_path_coeff_3[i];
      coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void*)coeff_dp : (void*)coeff_sp;

  #ifdef MULTI_GPU

      exchange_cpu_sitelink(qudaGaugeParam.X, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, &qudaGaugeParam, optflag);
      llfat_reference_mg(fat_reflink, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, coeff);
    
      {
        int R[4] = {2,2,2,2};
        exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, w_reflink_ex, QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0, 4);
        computeLongLinkCPU(long_reflink, w_reflink_ex, qudaGaugeParam.cpu_prec, coeff);
      }
  #else
      llfat_reference(fat_reflink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
      computeLongLinkCPU(long_reflink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
  #endif

      // Rescale fat and long links into eps links
      for (int i = 0; i < 4; i++) {
        cpu_axy(prec, eps_naik, fat_reflink[i], fat_reflink_eps[i], V*gaugeSiteSize);
        cpu_axy(prec, eps_naik, long_reflink[i], long_reflink_eps[i], V*gaugeSiteSize);
      }
    }

    /////////////////////////////////////////////////////////////
    // Prepare to create X links and long links, 2nd table set //
    /////////////////////////////////////////////////////////////

    for (int i=0; i < 6;i++) coeff_sp[i] = coeff_dp[i] = act_path_coeff_2[i];
    coeff = (prec == QUDA_DOUBLE_PRECISION) ? (void*)coeff_dp : (void*)coeff_sp;

#ifdef MULTI_GPU
    optflag = 0;

    // We've already built the extended W fields.

    exchange_cpu_sitelink(qudaGaugeParam.X, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, &qudaGaugeParam, optflag);
    llfat_reference_mg(fat_reflink, w_reflink, ghost_wlink, ghost_wlink_diag, qudaGaugeParam.cpu_prec, coeff);
  
    {
      int R[4] = {2,2,2,2};
      exchange_cpu_sitelink_ex(qudaGaugeParam.X, R, w_reflink_ex, QUDA_QDP_GAUGE_ORDER, qudaGaugeParam.cpu_prec, 0, 4);
      computeLongLinkCPU(long_reflink, w_reflink_ex, qudaGaugeParam.cpu_prec, coeff);
    }
#else
    llfat_reference(fat_reflink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
    computeLongLinkCPU(long_reflink, w_reflink, qudaGaugeParam.cpu_prec, coeff);
#endif

    if (n_naiks > 1) {
      // Accumulate into eps links.
      for (int i = 0; i < 4; i++) {
        cpu_xpy(prec, fat_reflink[i], fat_reflink_eps[i], V*gaugeSiteSize);
        cpu_xpy(prec, long_reflink[i], long_reflink_eps[i], V*gaugeSiteSize);
      }
    }

  }//verify_results

  ////////////////////////////////////////////////////////////////////
  // Layout change for fatlink, fatlink_eps, longlink, longlink_eps //
  ////////////////////////////////////////////////////////////////////

  void* myfatlink [4];
  void* mylonglink [4];
  void* myfatlink_eps [4];
  void* mylonglink_eps [4];
  for(int i=0; i < 4; i++) {

    myfatlink [i] = safe_malloc(V*gaugeSiteSize*gSize);
    mylonglink[i] = safe_malloc(V*gaugeSiteSize*gSize);
    memset(myfatlink [i], 0, V*gaugeSiteSize*gSize);
    memset(mylonglink[i], 0, V*gaugeSiteSize*gSize);
    
    if (n_naiks > 1) {
      myfatlink_eps [i] = safe_malloc(V*gaugeSiteSize*gSize);
      mylonglink_eps[i] = safe_malloc(V*gaugeSiteSize*gSize);
      memset(myfatlink_eps [i], 0, V*gaugeSiteSize*gSize);
      memset(mylonglink_eps[i], 0, V*gaugeSiteSize*gSize);
    }
  }

  for(int i=0; i < V; i++){
    for(int dir=0; dir< 4; dir++){
      char* src = ((char*)fatlink )+ (4*i+dir)*gaugeSiteSize*gSize;
      char* dst = ((char*)myfatlink [dir]) + i*gaugeSiteSize*gSize;
      memcpy(dst, src, gaugeSiteSize*gSize);

      src = ((char*)longlink)+ (4*i+dir)*gaugeSiteSize*gSize;
      dst = ((char*)mylonglink[dir]) + i*gaugeSiteSize*gSize;
      memcpy(dst, src, gaugeSiteSize*gSize);

      if (n_naiks > 1) {
        src = ((char*)fatlink_eps )+ (4*i+dir)*gaugeSiteSize*gSize;
        dst = ((char*)myfatlink_eps [dir]) + i*gaugeSiteSize*gSize;
        memcpy(dst, src, gaugeSiteSize*gSize);

        src = ((char*)longlink_eps)+ (4*i+dir)*gaugeSiteSize*gSize;
        dst = ((char*)mylonglink_eps[dir]) + i*gaugeSiteSize*gSize;
        memcpy(dst, src, gaugeSiteSize*gSize);
      }
    }
  }

  //////////////////////////////
  // Perform the verification //
  //////////////////////////////

  if (verify_results) {
    printfQuda("Checking fat links...\n");
    int res=1;
    for(int dir=0; dir<4; dir++){
      res &= compare_floats(fat_reflink[dir], myfatlink [dir], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
    }
    
    strong_check_link(myfatlink , "GPU results: ",
		      fat_reflink, "CPU reference results:",
		      V, qudaGaugeParam.cpu_prec);
    
    printfQuda("Fat-link test %s\n\n",(1 == res) ? "PASSED" : "FAILED");



    printfQuda("Checking long links...\n");
    res = 1;
    for(int dir=0; dir<4; ++dir){
      res &= compare_floats(long_reflink[dir], mylonglink[dir], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
    }
      
    strong_check_link(mylonglink, "GPU results: ",
		      long_reflink, "CPU reference results:",
		      V, qudaGaugeParam.cpu_prec);
      
    printfQuda("Long-link test %s\n\n",(1 == res) ? "PASSED" : "FAILED");

    if (n_naiks > 1) {

      printfQuda("Checking fat eps_naik links...\n");
      res=1;
      for(int dir=0; dir<4; dir++){
        res &= compare_floats(fat_reflink_eps[dir], myfatlink_eps [dir], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
      }
      
      strong_check_link(myfatlink_eps , "GPU results: ",
            fat_reflink_eps, "CPU reference results:",
            V, qudaGaugeParam.cpu_prec);
      
      printfQuda("Fat-link eps_naik test %s\n\n",(1 == res) ? "PASSED" : "FAILED");


      printfQuda("Checking long eps_naik links...\n");
      res = 1;
      for(int dir=0; dir<4; ++dir){
        res &= compare_floats(long_reflink_eps[dir], mylonglink_eps[dir], V*gaugeSiteSize, 1e-3, qudaGaugeParam.cpu_prec);
      }
        
      strong_check_link(mylonglink_eps, "GPU results: ",
            long_reflink_eps, "CPU reference results:",
            V, qudaGaugeParam.cpu_prec);
        
      printfQuda("Long-link eps_naik test %s\n\n",(1 == res) ? "PASSED" : "FAILED");
    }
  }

  // FIXME: does not include unitarization, extra naiks
  int volume = qudaGaugeParam.X[0]*qudaGaugeParam.X[1]*qudaGaugeParam.X[2]*qudaGaugeParam.X[3];
  long long flops = 61632 * (long long)niter; // Constructing V field
  // Constructing W field?
  // Constructing separate Naiks
  flops += 61632 * (long long)niter; // Constructing X field
  flops += (252*4)*(long long)niter; // long-link contribution

  double perf = flops*volume/(secs*1024*1024*1024);
  printfQuda("link computation time =%.2f ms, flops= %.2f Gflops\n", (secs*1000)/niter, perf);

  for (int i=0; i < 4; i++) {
    host_free(myfatlink [i]);
    host_free(mylonglink[i]);
    if (n_naiks > 1) {
      host_free(myfatlink_eps [i]);
      host_free(mylonglink_eps[i]);
    }
  }

#ifdef MULTI_GPU
  if (verify_results){
    for(int i=0; i<4; i++){
      host_free(ghost_sitelink[i]);
      host_free(ghost_wlink[i]);
      for(int j=0;j <4; j++){
        if (i==j) continue;
        host_free(ghost_sitelink_diag[i*4+j]);
        host_free(ghost_wlink_diag[i*4+j]);
      }
    }
  }
#endif


  for(int i=0; i < 4; i++){
    host_free(sitelink[i]);
    host_free(sitelink_ex[i]);
    host_free(v_reflink[i]);
    host_free(w_reflink[i]);
    host_free(w_reflink_ex[i]);
    host_free(fat_reflink[i]);
    host_free(long_reflink[i]);

    if (n_naiks > 1) {
      host_free(fat_reflink_eps[i]);
      host_free(long_reflink_eps[i]);
    }
  }

  // Clean up GPU compute links
  host_free(vlink);
  host_free(v_sitelink);
  host_free(wlink);
  host_free(fatlink);
  host_free(longlink);
  
  if (n_naiks > 1) {
    host_free(fatlink_eps);
    host_free(longlink_eps);
  }

  if(milc_sitelink) host_free(milc_sitelink);
#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif
  endQuda();
}

static void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("link_precision           link_reconstruct           space_dimension        T_dimension       Ordering\n");
  printfQuda("%s                       %s                         %d/%d/%d/                  %d             %s \n", 
      get_prec_str(prec),
      get_recon_str(link_recon), 
      xdim, ydim, zdim, tdim,
      get_gauge_order_str(gauge_order));

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n",
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3));

  return ;

}


int main(int argc, char **argv)
{
  // for speed
  xdim=ydim=zdim=tdim=8;

  //default to 18 reconstruct
  link_recon = QUDA_RECONSTRUCT_NO;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  // For now:
  niter = 100;

  for (int i = 1; i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (eps_naik != 0.0) { n_naiks = 2; }

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  hisq_test();
  finalizeComms();
}


