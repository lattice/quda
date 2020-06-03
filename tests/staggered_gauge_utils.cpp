#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <host_utils.h>
#include <quda_internal.h>
#include <quda.h>
#include <util_quda.h>
#include <staggered_gauge_utils.h>
#include <llfat_reference.h>
#include <unitarization_links.h>
#include "misc.h"

extern double tadpole_factor;
// Unitarization coefficients
static double unitarize_eps = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only = false;
static double svd_rel_error = 1e-4;
static double svd_abs_error = 1e-4;
static double max_allowed_error = 1e-11;

// Wrap everything for the GPU construction of fat/long links here
void computeHISQLinksGPU(void **qdp_fatlink, void **qdp_longlink, void **qdp_fatlink_eps, void **qdp_longlink_eps,
                         void **qdp_inlink, QudaGaugeParam &gauge_param_in, double **act_path_coeffs, double eps_naik,
                         size_t gauge_data_type_size, int n_naiks)
{
  // since a lot of intermediaries can be general matrices, override the recon in `gauge_param_in`
  auto gauge_param = gauge_param_in;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO; // probably irrelevant

  // inlink in different format
  void *milc_inlink = pinned_malloc(4 * V * gauge_site_size * gauge_data_type_size);
  reorderQDPtoMILC(milc_inlink, qdp_inlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  // Paths for step 1:
  void *milc_vlink = pinned_malloc(4 * V * gauge_site_size * gauge_data_type_size); // V links
  void *milc_wlink = pinned_malloc(4 * V * gauge_site_size * gauge_data_type_size); // W links

  // Paths for step 2:
  void *milc_fatlink = pinned_malloc(4 * V * gauge_site_size * gauge_data_type_size);  // final fat ("X") links
  void *milc_longlink = pinned_malloc(4 * V * gauge_site_size * gauge_data_type_size); // final long links

  // Place to accumulate Naiks, step 3:
  void *milc_fatlink_eps = nullptr;
  void *milc_longlink_eps = nullptr;
  if (n_naiks > 1) {
    milc_fatlink_eps = pinned_malloc(4 * V * gauge_site_size * gauge_data_type_size);  // epsilon fat links
    milc_longlink_eps = pinned_malloc(4 * V * gauge_site_size * gauge_data_type_size); // epsilon long naiks
  }

  // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
  computeKSLinkQuda(milc_vlink, nullptr, milc_wlink, milc_inlink, act_path_coeffs[0], &gauge_param);

  if (n_naiks > 1) {
    // Create Naiks, 3rd path table set
    computeKSLinkQuda(milc_fatlink, milc_longlink, nullptr, milc_wlink, act_path_coeffs[2], &gauge_param);

    // Rescale+copy Naiks into Naik field
    cpu_axy(gauge_param.cpu_prec, eps_naik, milc_fatlink, milc_fatlink_eps, V * 4 * gauge_site_size);
    cpu_axy(gauge_param.cpu_prec, eps_naik, milc_longlink, milc_longlink_eps, V * 4 * gauge_site_size);
  } else {
    memset(milc_fatlink, 0, V * 4 * gauge_site_size * gauge_data_type_size);
    memset(milc_longlink, 0, V * 4 * gauge_site_size * gauge_data_type_size);
  }

  // Create X and long links, 2nd path table set
  computeKSLinkQuda(milc_fatlink, milc_longlink, nullptr, milc_wlink, act_path_coeffs[1], &gauge_param);

  if (n_naiks > 1) {
    // Add into Naik field
    cpu_xpy(gauge_param.cpu_prec, milc_fatlink, milc_fatlink_eps, V * 4 * gauge_site_size);
    cpu_xpy(gauge_param.cpu_prec, milc_longlink, milc_longlink_eps, V * 4 * gauge_site_size);
  }

  // Copy back
  reorderMILCtoQDP(qdp_fatlink, milc_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderMILCtoQDP(qdp_longlink, milc_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  if (n_naiks > 1) {
    // Add into Naik field
    reorderMILCtoQDP(qdp_fatlink_eps, milc_fatlink_eps, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
    reorderMILCtoQDP(qdp_longlink_eps, milc_longlink_eps, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
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

void setActionPaths(double **act_paths)
{
  ///////////////////////////
  // Set path coefficients //
  ///////////////////////////

  // Reference: "generic_ks/imp_actions/hisq/hisq_action.h",
  // in QHMC: https://github.com/jcosborn/qhmc/blob/master/lib/qopqdp/hisq.c

  double u1 = 1.0 / tadpole_factor;
  double u2 = u1 * u1;
  double u4 = u2 * u2;
  double u6 = u4 * u2;

  // First path: create V, W links
  double act_path_coeff_1[6] = {
    (1.0 / 8.0),                             // one link
    u2 * (0.0),                              // Naik
    u2 * (-1.0 / 8.0) * 0.5,                 // simple staple
    u4 * (1.0 / 8.0) * 0.25 * 0.5,           // displace link in two directions
    u6 * (-1.0 / 8.0) * 0.125 * (1.0 / 6.0), // displace link in three directions
    u4 * (0.0)                               // Lepage term
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

  for (int i = 0; i < 6; i++) {
    act_paths[0][i] = act_path_coeff_1[i];
    act_paths[1][i] = act_path_coeff_2[i];
    act_paths[2][i] = act_path_coeff_3[i];
  }

  ////////////////////////////////////
  // Set unitarization coefficients //
  ////////////////////////////////////

  setUnitarizeLinksConstants(unitarize_eps, max_allowed_error, reunit_allow_svd, reunit_svd_only, svd_rel_error,
                             svd_abs_error);
}

void computeFatLongGPU(void **qdp_fatlink, void **qdp_longlink, void **qdp_inlink, QudaGaugeParam &gauge_param,
                       size_t gauge_data_type_size, int n_naiks, double eps_naik)
{
  double **act_paths = new double *[3];
  for (int i = 0; i < 3; i++) act_paths[i] = new double[6];
  setActionPaths(act_paths);

  ///////////////////////////////////////////////////////////////////////
  // Create some temporary space if we want to test the epsilon fields //
  ///////////////////////////////////////////////////////////////////////

  void *qdp_fatlink_naik_temp[4];
  void *qdp_longlink_naik_temp[4];
  if (n_naiks == 2) {
    for (int dir = 0; dir < 4; dir++) {
      qdp_fatlink_naik_temp[dir] = malloc(V * gauge_site_size * gauge_data_type_size);
      qdp_longlink_naik_temp[dir] = malloc(V * gauge_site_size * gauge_data_type_size);
    }
  }

  //////////////////////////
  // Create the GPU links //
  //////////////////////////

  // Skip eps field for now
  // Note: GPU link creation only works for single and double precision
  computeHISQLinksGPU(qdp_fatlink, qdp_longlink, (n_naiks == 2) ? qdp_fatlink_naik_temp : nullptr,
                      (n_naiks == 2) ? qdp_longlink_naik_temp : nullptr, qdp_inlink, gauge_param, act_paths, eps_naik,
                      gauge_data_type_size, n_naiks);

  if (n_naiks == 2) {
    // Override the naik fields into the fat/long link fields
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], qdp_fatlink_naik_temp[dir], V * gauge_site_size * gauge_data_type_size);
      memcpy(qdp_longlink[dir], qdp_longlink_naik_temp[dir], V * gauge_site_size * gauge_data_type_size);
      free(qdp_fatlink_naik_temp[dir]);
      qdp_fatlink_naik_temp[dir] = nullptr;
      free(qdp_longlink_naik_temp[dir]);
      qdp_longlink_naik_temp[dir] = nullptr;
    }
  }

  for (int i = 0; i < 3; i++) delete[] act_paths[i];
  delete[] act_paths;
}

void computeFatLongGPUandCPU(void **qdp_fatlink_gpu, void **qdp_longlink_gpu, void **qdp_fatlink_cpu,
                             void **qdp_longlink_cpu, void **qdp_inlink, QudaGaugeParam &gauge_param,
                             size_t gauge_data_type_size, int n_naiks, double eps_naik)
{
  double **act_paths = new double *[3];
  for (int i = 0; i < 3; i++) act_paths[i] = new double[6];
  setActionPaths(act_paths);

  ///////////////////////////////////////////////////////////////////////
  // Create some temporary space if we want to test the epsilon fields //
  ///////////////////////////////////////////////////////////////////////

  void *qdp_fatlink_naik_temp[4];
  void *qdp_longlink_naik_temp[4];
  if (n_naiks == 2) {
    for (int dir = 0; dir < 4; dir++) {
      qdp_fatlink_naik_temp[dir] = malloc(V * gauge_site_size * gauge_data_type_size);
      qdp_longlink_naik_temp[dir] = malloc(V * gauge_site_size * gauge_data_type_size);
      memset(qdp_fatlink_naik_temp[dir], 0, V * gauge_site_size * gauge_data_type_size);
      memset(qdp_longlink_naik_temp[dir], 0, V * gauge_site_size * gauge_data_type_size);
    }
  }

  //////////////////////////
  // Create the CPU links //
  //////////////////////////

  // defined in "llfat_reference.cpp"
  computeHISQLinksCPU(qdp_fatlink_cpu, qdp_longlink_cpu, (n_naiks == 2) ? qdp_fatlink_naik_temp : nullptr,
                      (n_naiks == 2) ? qdp_longlink_naik_temp : nullptr, qdp_inlink, &gauge_param, act_paths, eps_naik);

  if (n_naiks == 2) {
    // Override the naik fields into the fat/long link fields
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink_cpu[dir], qdp_fatlink_naik_temp[dir], V * gauge_site_size * gauge_data_type_size);
      memcpy(qdp_longlink_cpu[dir], qdp_longlink_naik_temp[dir], V * gauge_site_size * gauge_data_type_size);
      memset(qdp_fatlink_naik_temp[dir], 0, V * gauge_site_size * gauge_data_type_size);
      memset(qdp_longlink_naik_temp[dir], 0, V * gauge_site_size * gauge_data_type_size);
    }
  }

  //////////////////////////
  // Create the GPU links //
  //////////////////////////

  // Skip eps field for now
  // Note: GPU link creation only works for single and double precision
  computeHISQLinksGPU(qdp_fatlink_gpu, qdp_longlink_gpu, (n_naiks == 2) ? qdp_fatlink_naik_temp : nullptr,
                      (n_naiks == 2) ? qdp_longlink_naik_temp : nullptr, qdp_inlink, gauge_param, act_paths, eps_naik,
                      gauge_data_type_size, n_naiks);

  if (n_naiks == 2) {
    // Override the naik fields into the fat/long link fields
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink_gpu[dir], qdp_fatlink_naik_temp[dir], V * gauge_site_size * gauge_data_type_size);
      memcpy(qdp_longlink_gpu[dir], qdp_longlink_naik_temp[dir], V * gauge_site_size * gauge_data_type_size);
      free(qdp_fatlink_naik_temp[dir]);
      qdp_fatlink_naik_temp[dir] = nullptr;
      free(qdp_longlink_naik_temp[dir]);
      qdp_longlink_naik_temp[dir] = nullptr;
    }
  }

  for (int i = 0; i < 3; i++) delete[] act_paths[i];
  delete[] act_paths;
}


// Routine that takes in a QDP-ordered field and outputs the plaquette.
// Assumes the gauge fields already have phases on them (unless it's the Laplace op),
// so it corrects the sign as appropriate.
void computeStaggeredPlaquetteQDPOrder(void** qdp_link, double plaq[3], const QudaGaugeParam& gauge_param_in,
                                       const QudaDslashType dslash_type)
{
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    errorQuda("computeStaggeredPlaquetteQDPOrder does not support dslash type %d\n", dslash_type);
  }

  // Make no assumptions about any part of gauge_param_in beyond what we need to grab.
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  for (int d = 0; d < 4; d++) {
    gauge_param.X[d] = gauge_param_in.X[d];
  }

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = gauge_param_in.cpu_prec;

  gauge_param.cuda_prec = gauge_param_in.cuda_prec;
  gauge_param.reconstruct = gauge_param_in.reconstruct;

  gauge_param.cuda_prec_sloppy = gauge_param_in.cuda_prec; // for ease of use
  gauge_param.reconstruct_sloppy = gauge_param_in.reconstruct_sloppy;

  gauge_param.anisotropy = 1;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size = x_face_size > y_face_size ? x_face_size : y_face_size;
  pad_size = pad_size > z_face_size ? pad_size : z_face_size; 
  pad_size = pad_size > t_face_size ? pad_size : t_face_size;
  gauge_param.ga_pad = pad_size;
#endif

  loadGaugeQuda(qdp_link, &gauge_param);
  plaqQuda(plaq);

  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_ASQTAD_DSLASH) {
    plaq[0] = -plaq[0];
    plaq[1] = -plaq[1];
    plaq[2] = -plaq[2];
  }

}

