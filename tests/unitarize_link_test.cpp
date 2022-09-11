#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "quda.h"
#include "timer.h"
#include "gauge_field.h"
#include "host_utils.h"
#include <command_line_params.h>
#include "misc.h"
#include "util_quda.h"
#include "llfat_quda.h"
#include <unitarization_links.h>
#include "ks_improved_force.h"

#ifdef MULTI_GPU
#include "comm_quda.h"
#endif

// google test frame work
#include <gtest/gtest.h>

#define TDIFF(a, b) (b.tv_sec - a.tv_sec + 0.000001 * (b.tv_usec - a.tv_usec))

static double unitarize_eps = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only = false;
static double svd_rel_error = 1e-4;
static double svd_abs_error = 1e-4;
static double max_allowed_error = 1e-11;

static QudaGaugeFieldOrder gauge_order = QUDA_MILC_GAUGE_ORDER;

quda::cpuGaugeField *cpuFatLink, *cpuULink, *cudaResult;
quda::cudaGaugeField *cudaFatLink, *cudaULink;

const double unittol = (prec == QUDA_DOUBLE_PRECISION) ? 1e-10 : 1e-6;

TEST(unitarization, verify)
{
  unitarizeLinksCPU(*cpuULink, *cpuFatLink);
  cudaULink->saveCPUField(*cudaResult);

  int res = compare_floats(cudaResult->Gauge_p(), cpuULink->Gauge_p(), 4 * cudaResult->Volume() * gauge_site_size,
                           unittol, cpu_prec);

#ifdef MULTI_GPU
  quda::comm_allreduce_int(res);
  res /= quda::comm_size();
#endif

  ASSERT_EQ(res, 1) << "CPU and CUDA implementations do not agree";
}

static int unitarize_link_test(int &test_rc)
{
  setVerbosity(verbosity);
  QudaGaugeParam qudaGaugeParam = newQudaGaugeParam();

  qudaGaugeParam.anisotropy = 1.0;

  qudaGaugeParam.X[0] = xdim;
  qudaGaugeParam.X[1] = ydim;
  qudaGaugeParam.X[2] = zdim;
  qudaGaugeParam.X[3] = tdim;

  setDims(qudaGaugeParam.X);

  qudaGaugeParam.type = QUDA_WILSON_LINKS;

  qudaGaugeParam.t_boundary = QUDA_PERIODIC_T;
  qudaGaugeParam.anisotropy = 1.0;
  qudaGaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  qudaGaugeParam.ga_pad = 0;
  qudaGaugeParam.cpu_prec = cpu_prec;
  qudaGaugeParam.cuda_prec = prec;
  qudaGaugeParam.cuda_prec_sloppy = prec;

  if (gauge_order != QUDA_MILC_GAUGE_ORDER) errorQuda("Unsupported gauge order %d", gauge_order);

  qudaGaugeParam.gauge_order = gauge_order;
  qudaGaugeParam.type = QUDA_WILSON_LINKS;
  qudaGaugeParam.reconstruct = link_recon;
  qudaGaugeParam.reconstruct_sloppy = qudaGaugeParam.reconstruct;

  qudaGaugeParam.llfat_ga_pad = qudaGaugeParam.site_ga_pad = qudaGaugeParam.ga_pad = qudaGaugeParam.staple_pad = 0;

  quda::GaugeFieldParam gParam(qudaGaugeParam);
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.order = gauge_order;

  void *inlink = (void *)safe_malloc(4 * V * gauge_site_size * cpu_prec);
  void *fatlink = (void *)safe_malloc(4 * V * gauge_site_size * cpu_prec);

  void *sitelink[4];
  for (int i = 0; i < 4; i++) sitelink[i] = pinned_malloc(V * gauge_site_size * cpu_prec);

  createSiteLinkCPU(sitelink, qudaGaugeParam.cpu_prec, 1);

  if (cpu_prec == QUDA_DOUBLE_PRECISION) {
    double *link = reinterpret_cast<double *>(inlink);
    for (int dir = 0; dir < 4; ++dir) {
      double *slink = reinterpret_cast<double *>(sitelink[dir]);
      for (int i = 0; i < V; ++i) {
        for (auto j = 0lu; j < gauge_site_size; j++) {
          link[(i * 4 + dir) * gauge_site_size + j] = slink[i * gauge_site_size + j];
        }
      }
    }
  } else if (cpu_prec == QUDA_SINGLE_PRECISION) {
    float *link = reinterpret_cast<float *>(inlink);
    for (int dir = 0; dir < 4; ++dir) {
      float *slink = reinterpret_cast<float *>(sitelink[dir]);
      for (int i = 0; i < V; ++i) {
        for (auto j = 0lu; j < gauge_site_size; j++) {
          link[(i * 4 + dir) * gauge_site_size + j] = slink[i * gauge_site_size + j];
        }
      }
    }
  }

  gParam.create = QUDA_REFERENCE_FIELD_CREATE;
  gParam.gauge = fatlink;
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  cpuFatLink = new quda::cpuGaugeField(gParam);

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  cpuULink = new quda::cpuGaugeField(gParam);

  gParam.create = QUDA_ZERO_FIELD_CREATE;
  cudaResult = new quda::cpuGaugeField(gParam);

  gParam.pad = 0;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.setPrecision(prec, true);
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  cudaFatLink = new quda::cudaGaugeField(gParam);
  cudaULink = new quda::cudaGaugeField(gParam);

  { // create fat links
    double act_path_coeff[6];
    act_path_coeff[0] = 0.625000;
    act_path_coeff[1] = -0.058479;
    act_path_coeff[2] = -0.087719;
    act_path_coeff[3] = 0.030778;
    act_path_coeff[4] = -0.007200;
    act_path_coeff[5] = -0.123113;

    computeKSLinkQuda(fatlink, NULL, NULL, inlink, act_path_coeff, &qudaGaugeParam);

    cudaFatLink->loadCPUField(*cpuFatLink);
  }

  quda::setUnitarizeLinksConstants(unitarize_eps, max_allowed_error, reunit_allow_svd, reunit_svd_only, svd_rel_error,
                                   svd_abs_error);

  int *num_failures_h = static_cast<int *>(mapped_malloc(sizeof(int)));
  int *num_failures_d = static_cast<int *>(get_mapped_device_pointer(num_failures_h));
  *num_failures_h = 0;

  struct timeval t0, t1;

  gettimeofday(&t0, NULL);
  unitarizeLinks(*cudaULink, *cudaFatLink, num_failures_d);
  gettimeofday(&t1, NULL);

  if (verify_results) {
    test_rc = RUN_ALL_TESTS();
    if (test_rc != 0) warningQuda("Tests failed");
  }

  delete cudaResult;
  delete cpuULink;
  delete cpuFatLink;
  delete cudaFatLink;
  delete cudaULink;
  for (int dir = 0; dir < 4; ++dir) host_free(sitelink[dir]);

  host_free(fatlink);

  int num_failures = *num_failures_h;
  host_free(num_failures_h);

  host_free(inlink);
#ifdef MULTI_GPU
  exchange_llfat_cleanup();
#endif

  printfQuda("Unitarization time: %g ms\n", TDIFF(t0, t1) * 1000);
  return num_failures;
}

static void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("link_precision      link_reconstruct           space_dimension        T_dimension    algorithm           "
             "max allowed error  deviation tolerance\n");
  printfQuda("%8s              %s                         %d/%d/%d/                 %d            %s         %g        "
             "     %g\n",
             get_prec_str(prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim,
             get_unitarization_str(reunit_svd_only), max_allowed_error, unittol);

#ifdef MULTI_GPU
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
#endif
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);
  int test_rc = 0;

  // default to 18 reconstruct, 8^3 x 8
  link_recon = QUDA_RECONSTRUCT_NO;
  xdim = ydim = zdim = tdim = 8;

  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  display_test_info();
  int num_failures = unitarize_link_test(test_rc);
  int num_procs = 1;
#ifdef MULTI_GPU
  quda::comm_allreduce_int(num_failures);
  num_procs = quda::comm_size();
#endif

  printfQuda("Number of failures = %d\n", num_failures);
  if (num_failures > 0) {
    printfQuda("Failure rate = %lf\n", num_failures / (4.0 * V * num_procs));
    printfQuda("You may want to increase the error tolerance or vary the unitarization parameters\n");
  } else {
    printfQuda("Unitarization successfull!\n");
  }

  endQuda();
  finalizeComms();

  return test_rc;
}
