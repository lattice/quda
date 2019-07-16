#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <test_util.h>
#include <quda_internal.h>
#include <quda.h>
#include <util_quda.h>
#include <staggered_gauge_utils.h>
#include <llfat_reference.h>
#include <unitarization_links.h>
#include "misc.h"

extern double tadpole_factor;
// Unitarization coefficients
static double unitarize_eps  = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only  = false;
static double svd_rel_error  = 1e-4;
static double svd_abs_error  = 1e-4;
static double max_allowed_error = 1e-11;

// Wrap everything for the GPU construction of fat/long links here
void computeHISQLinksGPU(void** qdp_fatlink, void** qdp_longlink,
			 void** qdp_fatlink_eps, void** qdp_longlink_eps,
			 void** qdp_inlink, QudaGaugeParam &gauge_param,
			 double** act_path_coeffs, double eps_naik,
			 size_t gSize, int n_naiks) {
  
  // inlink in different format
  void *milc_inlink = pinned_malloc(4*V*gaugeSiteSize*gSize);
  reorderQDPtoMILC(milc_inlink,qdp_inlink,V,gaugeSiteSize,gauge_param.cpu_prec,gauge_param.cpu_prec);

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
  computeKSLinkQuda(milc_vlink, nullptr, milc_wlink, milc_inlink, act_path_coeffs[0], &gauge_param);

  if (n_naiks > 1) {
    // Create Naiks, 3rd path table set
    computeKSLinkQuda(milc_fatlink, milc_longlink, nullptr, milc_wlink, act_path_coeffs[2], &gauge_param);

    // Rescale+copy Naiks into Naik field
    cpu_axy(gauge_param.cpu_prec, eps_naik, milc_fatlink, milc_fatlink_eps, V*4*gaugeSiteSize);
    cpu_axy(gauge_param.cpu_prec, eps_naik, milc_longlink, milc_longlink_eps, V*4*gaugeSiteSize);
  } else {
    memset(milc_fatlink, 0, V*4*gaugeSiteSize*gSize);
    memset(milc_longlink, 0, V*4*gaugeSiteSize*gSize);
  }

  // Create X and long links, 2nd path table set
  computeKSLinkQuda(milc_fatlink, milc_longlink, nullptr, milc_wlink, act_path_coeffs[1], &gauge_param);

  if (n_naiks > 1) {
    // Add into Naik field
    cpu_xpy(gauge_param.cpu_prec, milc_fatlink, milc_fatlink_eps, V*4*gaugeSiteSize);
    cpu_xpy(gauge_param.cpu_prec, milc_longlink, milc_longlink_eps, V*4*gaugeSiteSize);
  }

  // Copy back
  reorderMILCtoQDP(qdp_fatlink,milc_fatlink,V,gaugeSiteSize,gauge_param.cpu_prec,gauge_param.cpu_prec);
  reorderMILCtoQDP(qdp_longlink,milc_longlink,V,gaugeSiteSize,gauge_param.cpu_prec,gauge_param.cpu_prec);

  if (n_naiks > 1) {
    // Add into Naik field
    reorderMILCtoQDP(qdp_fatlink_eps,milc_fatlink_eps,V,gaugeSiteSize,gauge_param.cpu_prec,gauge_param.cpu_prec);
    reorderMILCtoQDP(qdp_longlink_eps,milc_longlink_eps,V,gaugeSiteSize,gauge_param.cpu_prec,gauge_param.cpu_prec);
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

void computeFatLong(void** qdp_fatlink, void** qdp_longlink,
		    void** qdp_inlink, QudaGaugeParam &gauge_param,
		    size_t gSize, int n_naiks, double eps_naik){
  
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
    (1.0/8.0),                     /* one link */
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
  
  double* act_paths[3] = { act_path_coeff_1, act_path_coeff_2, act_path_coeff_3 };
  
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
    }
  }
  
  //////////////////////////
  // Create the GPU links //
  //////////////////////////

  // Skip eps field for now  
  // Note: GPU link creation only works for single and double precision
  computeHISQLinksGPU(qdp_fatlink, qdp_longlink,
		      (n_naiks == 2) ? qdp_fatlink_naik_temp : nullptr,
		      (n_naiks == 2) ? qdp_longlink_naik_temp : nullptr,
		      qdp_inlink, gauge_param, act_paths, eps_naik, gSize, n_naiks);
  
  if (n_naiks == 2) {
    // Override the naik fields into the fat/long link fields
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir],qdp_fatlink_naik_temp[dir], V*gaugeSiteSize*gSize);
      memcpy(qdp_longlink[dir],qdp_longlink_naik_temp[dir], V*gaugeSiteSize*gSize);
      free(qdp_fatlink_naik_temp[dir]); qdp_fatlink_naik_temp[dir] = nullptr;
      free(qdp_longlink_naik_temp[dir]); qdp_longlink_naik_temp[dir] = nullptr;
    }
  }  
}



