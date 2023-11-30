#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <gauge_field.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <misc.h>
#include <unitarization_links.h>
#include <ks_improved_force.h>

#include <assert.h>
#include <gtest/gtest.h>
#include <tune_quda.h>

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

struct HisqStencilTestWrapper {

  static inline QudaGaugeParam gauge_param;

  // staple coefficients for different portions of the HISQ stencil build
  static inline std::array<std::array<double, 6>, 3> act_paths;

  // initial links in MILC order
  static inline void* milc_sitelink = nullptr;

  // storage for CPU reference fat and long links w/zero Naik
  static inline void *fat_reflink[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *long_reflink[4] = {nullptr, nullptr, nullptr, nullptr};

  // storage for CPU reference fat and long links w/non-zero Naik
  static inline void *fat_reflink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *long_reflink_eps[4] = {nullptr, nullptr, nullptr, nullptr};

  // Paths for step 1:
  static inline void *vlink = nullptr;
  static inline void *wlink = nullptr;

  // Paths for step 2:
  static inline void *fatlink = nullptr;
  static inline void *longlink = nullptr;

  // Place to accumulate Naiks
  static inline void *fatlink_eps = nullptr;
  static inline void *longlink_eps = nullptr;

  static inline void *qdp_sitelink[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *qdp_fatlink[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *qdp_longlink[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *qdp_fatlink_eps[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *qdp_longlink_eps[4] = {nullptr, nullptr, nullptr, nullptr};

  void set_naik(bool has_naik) {
    if (has_naik) {
      eps_naik = -0.03; // semi-arbitrary
      n_naiks = 2;
    } else {
      eps_naik = 0.0;
      n_naiks = 1;
    }
  }

  void init_ctest(QudaPrecision prec_, QudaReconstructType link_recon_, bool has_naik) {
    prec = prec_;
    link_recon = link_recon_;

    set_naik(has_naik);

    gauge_param = newQudaGaugeParam();
    setStaggeredGaugeParam(gauge_param);

    gauge_param.cuda_prec = prec;

    static bool first_time = true;
    if (first_time) {
      // force the Naik build up front, it doesn't effect the non-naik fields
      set_naik(true);
      init_host();
      set_naik(has_naik);
      first_time = false;
    }
    init();
  }

  void init_test() {
    gauge_param = newQudaGaugeParam();
    setStaggeredGaugeParam(gauge_param);

    static bool first_time = true;
    if (first_time) {
      init_host();
      first_time = false;
    }
    init();
  }

  void init_host() {
    setDims(gauge_param.X);
    dw_setDims(gauge_param.X, 1);

    ///////////////////////////////////////////////////////////////
    // Set up the coefficients for each part of the HISQ stencil //
    ///////////////////////////////////////////////////////////////

    // Reference: "generic_ks/imp_actions/hisq/hisq_action.h",
    // in QHMC: https://github.com/jcosborn/qhmc/blob/master/lib/qopqdp/hisq.c

    double u1 = 1.0 / tadpole_factor;
    double u2 = u1 * u1;
    double u4 = u2 * u2;
    double u6 = u4 * u2;

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

    for (int i = 0; i < 4; i++) qdp_sitelink[i] = pinned_malloc(V * gauge_site_size * host_gauge_data_type_size);

    // Note: this could be replaced with loading a gauge field
    createSiteLinkCPU(qdp_sitelink, gauge_param.cpu_prec, 0); // 0 -> no phases

    ///////////////////////
    // Perform CPU Build //
    ///////////////////////

    for (int i = 0; i < 4; i++) {
      // fat and long links for fermions with zero epsilon
      fat_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      long_reflink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);

      // fat and long links for fermions with non-zero epsilon
      if (n_naiks > 1) {
        fat_reflink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
        long_reflink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      }
    }

    computeHISQLinksCPU(fat_reflink, long_reflink, fat_reflink_eps, long_reflink_eps, qdp_sitelink, &gauge_param,
                        act_paths, eps_naik);

    /////////////////////////////////////////////////////////////////////
    // Allocate CPU-precision host storage for fields built on the GPU //
    /////////////////////////////////////////////////////////////////////

    // QDP order fields
    for (int i = 0; i < 4; i++) {
      qdp_fatlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      if (n_naiks > 1) {
        qdp_fatlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
        qdp_longlink_eps[i] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      }
    }

#ifdef MULTI_GPU
    exchange_llfat_cleanup();
#endif
  }

  void init() {

    // reset the reconstruct in gauge param
    gauge_param.reconstruct = link_recon;

    /////////////////////////////////////////////////////////////////
    // Create a CPU copy of the initial field in the GPU precision //
    /////////////////////////////////////////////////////////////////

    milc_sitelink = (void *)safe_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec);
    reorderQDPtoMILC(milc_sitelink, qdp_sitelink, V, gauge_site_size, gauge_param.cuda_prec, gauge_param.cpu_prec);

    ///////////////////////////////////////////////////////
    // Allocate host storage for fields built on the GPU //
    ///////////////////////////////////////////////////////

    // Paths for step 1:
    vlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // V links
    wlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // W links

    // Paths for step 2:
    fatlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec);  // final fat ("X") links
    longlink = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // final long links

    // Place to accumulate Naiks
    if (n_naiks > 1) {
      fatlink_eps = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec);  // epsilon fat links
      longlink_eps = pinned_malloc(4 * V * gauge_site_size * gauge_param.cuda_prec); // epsilon long naiks
    }
  }

  static void end() {
    if (milc_sitelink) host_free(milc_sitelink);

    // Clean up GPU compute links
    if (vlink) host_free(vlink);
    if (wlink) host_free(wlink);
    if (fatlink) host_free(fatlink);
    if (longlink) host_free(longlink);

    if (n_naiks > 1) {
      if (fatlink_eps) host_free(fatlink_eps);
      if (longlink_eps) host_free(longlink_eps);
    }

    freeGaugeQuda();
  }

  static void destroy() {

    for (int i = 0; i < 4; i++) {
      host_free(fat_reflink[i]);
      host_free(long_reflink[i]);
      if (n_naiks > 1) {
        host_free(fat_reflink_eps[i]);
        host_free(long_reflink_eps[i]);
      }
    }

    for (int i = 0; i < 4; i++) {
      host_free(qdp_sitelink[i]);
      host_free(qdp_fatlink[i]);
      host_free(qdp_longlink[i]);
      if (n_naiks > 1) {
        host_free(qdp_fatlink_eps[i]);
        host_free(qdp_longlink_eps[i]);
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

  double llfatCUDA(int niter) {
    host_timer_t host_timer;

    comm_barrier();
    host_timer.start();

    // manually override precision of input fields
    auto cpu_param_backup = gauge_param.cpu_prec;
    gauge_param.cpu_prec = gauge_param.cuda_prec;

    for (int i = 0; i < niter; i++) {
      // If we create cudaGaugeField objs, we can do this 100% on the GPU, no copying!

      // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
      computeKSLinkQuda(vlink, nullptr, wlink, milc_sitelink, act_paths[0].data(), &gauge_param);

      if (n_naiks > 1) {
        // Create Naiks, 3rd path table set
        computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_paths[2].data(), &gauge_param);

        // Rescale+copy Naiks into Naik field
        cpu_axy(gauge_param.cuda_prec, eps_naik, fatlink, fatlink_eps, V * 4 * gauge_site_size);
        cpu_axy(gauge_param.cuda_prec, eps_naik, longlink, longlink_eps, V * 4 * gauge_site_size);
      } else {
        memset(fatlink, 0, V * 4 * gauge_site_size * gauge_param.cuda_prec);
        memset(longlink, 0, V * 4 * gauge_site_size * gauge_param.cuda_prec);
      }

      // Create X and long links, 2nd path table set
      computeKSLinkQuda(fatlink, longlink, nullptr, wlink, act_paths[1].data(), &gauge_param);

      if (n_naiks > 1) {
        // Add into Naik field
        cpu_xpy(gauge_param.cuda_prec, fatlink, fatlink_eps, V * 4 * gauge_site_size);
        cpu_xpy(gauge_param.cuda_prec, longlink, longlink_eps, V * 4 * gauge_site_size);
      }
    }

    gauge_param.cpu_prec = cpu_param_backup;

    host_timer.stop();

    return host_timer.last();
  }

  void run_test(int niter, bool print_metrics = false) {
    //////////////////////
    // Perform GPU test //
    //////////////////////

    printfQuda("Tuning...\n");
    llfatCUDA(1);

    auto flops0 = quda::Tunable::flops_global();
    auto bytes0 = quda::Tunable::bytes_global();

    printfQuda("Running %d iterations of computation\n", niter);
    double secs = llfatCUDA(niter);

    unsigned long long flops = (quda::Tunable::flops_global() - flops0);
    unsigned long long bytes = (quda::Tunable::bytes_global() - bytes0);

    if (print_metrics) {
      // FIXME: does not include unitarization, extra naiks
      int volume = gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3];
      //long long flops = 61632 * (long long)niter; // Constructing V field
      // Constructing W field?
      // Constructing separate Naiks
      //flops += 61632 * (long long)niter;     // Constructing X field
      //flops += (252 * 4) * (long long)niter; // long-link contribution

      printfQuda("%fus per HISQ link build\n", 1e6 * secs / niter);

      printfQuda("%llu flops per HISQ link build, %llu flops per site %llu bytes per site\n", flops / niter,
                    (flops / niter) / volume, (bytes / niter) / volume);

      double gflops = 1.0e-9 * flops / secs;
        printfQuda("GFLOPS = %f\n", gflops);

      double gbytes = 1.0e-9 * bytes / secs;
        printfQuda("GBYTES = %f\n", gbytes);

      // Old metric
      //double perf = flops / (secs * 1024 * 1024 * 1024);
      //printfQuda("link computation time =%.2f ms, flops= %.2f Gflops\n", (secs * 1000) / niter, perf);
    }
  }

  std::array<double, 2> verify()
  {
    ////////////////////////////////////////////////////////////////////
    // Layout change for fatlink, fatlink_eps, longlink, longlink_eps //
    ////////////////////////////////////////////////////////////////////

    reorderMILCtoQDP(qdp_fatlink, fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cuda_prec);
    reorderMILCtoQDP(qdp_longlink, longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cuda_prec);

    if (n_naiks > 1) {
      reorderMILCtoQDP(qdp_fatlink_eps, fatlink_eps, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cuda_prec);
      reorderMILCtoQDP(qdp_longlink_eps, longlink_eps, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cuda_prec);
    }

    //////////////////////////////
    // Perform the verification //
    //////////////////////////////

    std::array<double, 2> res = {0., 0.};

    // extra factor of 10 b/c the norm isn't normalized
    double max_dev = 10. * getTolerance(prec);

    // Non-zero epsilon check
    if (n_naiks > 1) {
      for (int dir = 0; dir < 4; dir++) {
        res[0] = std::max(res[0],
          compare_floats_v2(fat_reflink_eps[dir], qdp_fatlink_eps[dir], V * gauge_site_size, max_dev,
                            gauge_param.cpu_prec));
      }

      strong_check_link(qdp_fatlink_eps, "Fat link GPU results: ", fat_reflink_eps, "CPU reference results:", V,
                        gauge_param.cpu_prec);

      for (int dir = 0; dir < 4; ++dir) {
        res[1] = std::max(res[1],
          compare_floats_v2(long_reflink_eps[dir], qdp_longlink_eps[dir], V * gauge_site_size, max_dev,
                            gauge_param.cpu_prec));
      }

      strong_check_link(qdp_longlink_eps, "Long link GPU results: ", long_reflink_eps, "CPU reference results:", V,
                        gauge_param.cpu_prec);
    } else {
      for (int dir = 0; dir < 4; dir++) {
        res[0] = std::max(res[0],
          compare_floats_v2(fat_reflink[dir], qdp_fatlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec));
      }

      strong_check_link(qdp_fatlink, "Fat link GPU results: ", fat_reflink, "CPU reference results:", V, gauge_param.cpu_prec);

      for (int dir = 0; dir < 4; ++dir) {
        res[1] = std::max(res[1],
          compare_floats_v2(long_reflink[dir], qdp_longlink[dir], V * gauge_site_size, max_dev, gauge_param.cpu_prec));
      }

      strong_check_link(qdp_longlink, "Long link GPU results: ", long_reflink, "CPU reference results:", V, gauge_param.cpu_prec);
    }

    printfQuda("Fat link test %s\n", (res[0] < max_dev) ? "PASSED" : "FAILED");
    printfQuda("Long link test %s\n", (res[1] < max_dev) ? "PASSED" : "FAILED");

    return res;

  }
};
