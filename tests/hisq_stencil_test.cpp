#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "quda.h"
#include "gauge_field.h"
#include "host_utils.h"
#include <command_line_params.h>
#include "misc.h"
#include "util_quda.h"
#include "malloc_quda.h"
#include <unitarization_links.h>
#include "ks_improved_force.h"

#ifdef MULTI_GPU
#include "comm_quda.h"
#endif

#define TDIFF(a, b) (b.tv_sec - a.tv_sec + 0.000001 * (b.tv_usec - a.tv_usec))

using namespace quda;

// Number of naiks. If eps_naik is 0.0, we only need
// to construct one naik.
static QudaGaugeFieldOrder gauge_order = QUDA_MILC_GAUGE_ORDER;

// The file "generic_ks/fermion_links_hisq_load_milc.c"
// within MILC is the ultimate reference for what's going on here.

// Unitarization coefficients
static double unitarize_eps = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only = false;
static double svd_rel_error = 1e-4;
static double svd_abs_error = 1e-4;
static double max_allowed_error = 1e-11;

/*--------------------------------------------------------------------*/
// Some notation:
// U -- original link, SU(3), copied to "field" from "site"
// V -- after 1st level of smearing, non-SU(3)
// W -- unitarized, SU(3)
// X -- after 2nd level of smearing, non-SU(3)
/*--------------------------------------------------------------------*/

static void hisq_test()
{
  QudaGaugeParam gauge_param;

  initQuda(device_ordinal);

  if (prec == QUDA_HALF_PRECISION || prec == QUDA_QUARTER_PRECISION) {
    errorQuda("Precision %d is unsupported in some link fattening routines\n", prec);
  }

  if (gauge_order != QUDA_MILC_GAUGE_ORDER) errorQuda("Unsupported gauge order %d", gauge_order);

  cpu_prec = prec;
  host_gauge_data_type_size = cpu_prec;

  gauge_param = newQudaGaugeParam();

  setStaggeredGaugeParam(gauge_param);

  setDims(gauge_param.X);

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec_sloppy = cuda_prec;
  gauge_param.reconstruct_sloppy = link_recon;

  ///////////////////////////////////////////////////////////////
  // Set up the coefficients for each part of the HISQ stencil //
  ///////////////////////////////////////////////////////////////

  // Reference: "generic_ks/imp_actions/hisq/hisq_action.h",
  // in QHMC: https://github.com/jcosborn/qhmc/blob/master/lib/qopqdp/hisq.c

  double u1 = 1.0 / tadpole_factor;
  double u2 = u1 * u1;
  double u4 = u2 * u2;
  double u6 = u4 * u2;

  std::array<std::array<double, 6>, 3> act_paths;

  // First path: create V, W links
  act_paths[0] = {
    (1.0 / 8.0),                             /* one link */
    u2 * (0.0),                              /* Naik */
    u2 * (-1.0 / 8.0) * 0.5,                 /* simple staple */
    u4 * (1.0 / 8.0) * 0.25 * 0.5,           /* displace link in two directions */
    u6 * (-1.0 / 8.0) * 0.125 * (1.0 / 6.0), /* displace link in three directions */
    u4 * (0.0)                               /* Lepage term */
  };

  // Second path: create X, long links
  act_paths[1] = {
    ((1.0 / 8.0) + (2.0 * 6.0 / 16.0) + (1.0 / 8.0)), /* one link */
                                                      /* One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik */
    (-1.0 / 24.0),                                    /* Naik */
    (-1.0 / 8.0) * 0.5,                               /* simple staple */
    (1.0 / 8.0) * 0.25 * 0.5,                         /* displace link in two directions */
    (-1.0 / 8.0) * 0.125 * (1.0 / 6.0),               /* displace link in three directions */
    (-2.0 / 16.0)                                     /* Lepage term, correct O(a^2) 2x ASQTAD */
  };

  // Paths for epsilon corrections. Not used if n_naiks = 1.
  act_paths[2] = {
    (1.0 / 8.0),   /* one link b/c of Naik */
    (-1.0 / 24.0), /* Naik */
    0.0,           /* simple staple */
    0.0,           /* displace link in two directions */
    0.0,           /* displace link in three directions */
    0.0            /* Lepage term */
  };

  ////////////////////////////////////
  // Set unitarization coefficients //
  ////////////////////////////////////

  setUnitarizeLinksConstants(unitarize_eps, max_allowed_error, reunit_allow_svd, reunit_svd_only, svd_rel_error,
                             svd_abs_error);

  /////////////////
  // Input links //
  /////////////////

  void *qdp_sitelink[4] = {nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < 4; i++) qdp_sitelink[i] = pinned_malloc(V * gauge_site_size * host_gauge_data_type_size);

  void *milc_sitelink = (void *)safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  // Note: this could be replaced with loading a gauge field
  createSiteLinkCPU(qdp_sitelink, gauge_param.cpu_prec, 0); // 0 -> no phases
  reorderQDPtoMILC(milc_sitelink, qdp_sitelink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  //////////////////////
  // Perform GPU test //
  //////////////////////

  // Paths for step 1:
  void *vlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size); // V links
  void *wlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size); // W links

  // Paths for step 2:
  void *fatlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);  // final fat ("X") links
  void *longlink = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size); // final long links

  // Place to accumulate Naiks
  void *fatlink_eps = nullptr;
  void *longlink_eps = nullptr;
  if (n_naiks > 1) {
    fatlink_eps = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);  // epsilon fat links
    longlink_eps = pinned_malloc(4 * V * gauge_site_size * host_gauge_data_type_size); // epsilon long naiks
  }

  // Tuning run...
  {
    printfQuda("Tuning...\n");
    computeKSLinkQuda(vlink, longlink, wlink, milc_sitelink, act_paths[1].data(), &gauge_param);
  }

  struct timeval t0, t1;
  printfQuda("Running %d iterations of computation\n", niter);
  gettimeofday(&t0, NULL);
  for (int n = 0; n < niter; n++) {

    // If we create cudaGaugeField objs, we can do this 100% on the GPU, no copying!

    // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
    computeKSLinkQuda(vlink, nullptr, wlink, milc_sitelink, act_paths[0].data(), &gauge_param);

    if (n_naiks > 1) {
      // Create Naiks, 3rd path table set
      computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_paths[2].data(), &gauge_param);

      // Rescale+copy Naiks into Naik field
      cpu_axy(prec, eps_naik, fatlink, fatlink_eps, V * 4 * gauge_site_size);
      cpu_axy(prec, eps_naik, longlink, longlink_eps, V * 4 * gauge_site_size);
    } else {
      memset(fatlink, 0, V * 4 * gauge_site_size * host_gauge_data_type_size);
      memset(longlink, 0, V * 4 * gauge_site_size * host_gauge_data_type_size);
    }

    // Create X and long links, 2nd path table set
    computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_paths[1].data(), &gauge_param);

    if (n_naiks > 1) {
      // Add into Naik field
      cpu_xpy(prec, fatlink, fatlink_eps, V * 4 * gauge_site_size);
      cpu_xpy(prec, longlink, longlink_eps, V * 4 * gauge_site_size);
    }
  }
  gettimeofday(&t1, NULL);

  double secs = TDIFF(t0, t1);

  ///////////////////////
  // Perform CPU Build //
  ///////////////////////

  // fat and long links for fermions with zero epsilon
  void *fat_reflink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *long_reflink[4] = {nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < 4; i++) {
    fat_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    long_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  }

  // fat and long links for fermions with non-zero epsilon
  void *fat_reflink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
  void *long_reflink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
  if (n_naiks > 1) {
    for (int i = 0; i < 4; i++) {
      fat_reflink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      long_reflink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    }
  }

  if (verify_results) {
    computeHISQLinksCPU(fat_reflink, long_reflink, fat_reflink_eps, long_reflink_eps, qdp_sitelink, &gauge_param,
                        act_paths, eps_naik);
  }

  ////////////////////////////////////////////////////////////////////
  // Layout change for fatlink, fatlink_eps, longlink, longlink_eps //
  ////////////////////////////////////////////////////////////////////

  void *qdp_fatlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_longlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_fatlink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_longlink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < 4; i++) {
    qdp_fatlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_longlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    if (n_naiks > 1) {
      qdp_fatlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    }
  }

  reorderMILCtoQDP(qdp_fatlink, fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderMILCtoQDP(qdp_longlink, longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  if (n_naiks > 1) {
    reorderMILCtoQDP(qdp_fatlink_eps, fatlink_eps, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
    reorderMILCtoQDP(qdp_longlink_eps, longlink_eps, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  }

  //////////////////////////////
  // Perform the verification //
  //////////////////////////////

  if (verify_results) {
    printfQuda("Checking fat links...\n");
    int res = 1;
    for (int dir = 0; dir < 4; dir++) {
      res &= compare_floats(fat_reflink[dir], qdp_fatlink[dir], V * gauge_site_size, 1e-3, gauge_param.cpu_prec);
    }

    strong_check_link(qdp_fatlink, "GPU results: ", fat_reflink, "CPU reference results:", V, gauge_param.cpu_prec);

    printfQuda("Fat-link test %s\n\n", (1 == res) ? "PASSED" : "FAILED");

    printfQuda("Checking long links...\n");
    res = 1;
    for (int dir = 0; dir < 4; ++dir) {
      res &= compare_floats(long_reflink[dir], qdp_longlink[dir], V * gauge_site_size, 1e-3, gauge_param.cpu_prec);
    }

    strong_check_link(qdp_longlink, "GPU results: ", long_reflink, "CPU reference results:", V, gauge_param.cpu_prec);

    printfQuda("Long-link test %s\n\n", (1 == res) ? "PASSED" : "FAILED");

    if (n_naiks > 1) {

      printfQuda("Checking fat eps_naik links...\n");
      res = 1;
      for (int dir = 0; dir < 4; dir++) {
        res &= compare_floats(fat_reflink_eps[dir], qdp_fatlink_eps[dir], V * gauge_site_size, 1e-3,
                              gauge_param.cpu_prec);
      }

      strong_check_link(qdp_fatlink_eps, "GPU results: ", fat_reflink_eps, "CPU reference results:", V,
                        gauge_param.cpu_prec);

      printfQuda("Fat-link eps_naik test %s\n\n", (1 == res) ? "PASSED" : "FAILED");

      printfQuda("Checking long eps_naik links...\n");
      res = 1;
      for (int dir = 0; dir < 4; ++dir) {
        res &= compare_floats(long_reflink_eps[dir], qdp_longlink_eps[dir], V * gauge_site_size, 1e-3,
                              gauge_param.cpu_prec);
      }

      strong_check_link(qdp_longlink_eps, "GPU results: ", long_reflink_eps, "CPU reference results:", V,
                        gauge_param.cpu_prec);

      printfQuda("Long-link eps_naik test %s\n\n", (1 == res) ? "PASSED" : "FAILED");
    }
  }

  // FIXME: does not include unitarization, extra naiks
  int volume = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3];
  long long flops = 61632 * (long long)niter; // Constructing V field
  // Constructing W field?
  // Constructing separate Naiks
  flops += 61632 * (long long)niter;     // Constructing X field
  flops += (252 * 4) * (long long)niter; // long-link contribution

  double perf = flops * volume / (secs * 1024 * 1024 * 1024);
  printfQuda("link computation time =%.2f ms, flops= %.2f Gflops\n", (secs * 1000) / niter, perf);

  for (int i = 0; i < 4; i++) {
    host_free(qdp_fatlink[i]);
    host_free(qdp_longlink[i]);
    if (n_naiks > 1) {
      host_free(qdp_fatlink_eps[i]);
      host_free(qdp_longlink_eps[i]);
    }
  }

  for (int i = 0; i < 4; i++) {
    host_free(qdp_sitelink[i]);
    host_free(fat_reflink[i]);
    host_free(long_reflink[i]);
    if (n_naiks > 1) {
      host_free(fat_reflink_eps[i]);
      host_free(long_reflink_eps[i]);
    }
  }

  // Clean up GPU compute links
  host_free(vlink);
  host_free(wlink);
  host_free(fatlink);
  host_free(longlink);

  if (n_naiks > 1) {
    host_free(fatlink_eps);
    host_free(longlink_eps);
  }

  if (milc_sitelink) host_free(milc_sitelink);
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
             get_prec_str(prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim, get_gauge_order_str(gauge_order));

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));

  printfQuda("Number of Naiks: %d\n", n_naiks);
}

int main(int argc, char **argv)
{
  // for speed
  xdim = ydim = zdim = tdim = 8;

  // default to 18 reconstruct
  link_recon = QUDA_RECONSTRUCT_NO;
  cpu_prec = prec = QUDA_DOUBLE_PRECISION;

  auto app = make_app();
  // app->get_formatter()->column_width(40);
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (eps_naik != 0.0) { n_naiks = 2; }

  setVerbosity(verbosity);

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  hisq_test();
  finalizeComms();
}
