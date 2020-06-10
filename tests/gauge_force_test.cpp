#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <gauge_field.h>
#include "misc.h"
#include "gauge_force_reference.h"
#include "gauge_force_quda.h"
#include <sys/time.h>
#include <dslash_quda.h>
#include <gtest/gtest.h>

static QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

int length[] = {
  3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
};

float loop_coeff_f[]={
  1.1,
  1.2,
  1.3,
  1.4,
  1.5,
  1.6,
  2.5,
  2.6,
  2.7,
  2.8,
  2.9,
  3.0,
  3.1,
  3.2,
  3.3,
  3.4,
  3.5,
  3.6,
  3.7,
  3.8,
  3.9,
  4.0,
  4.1,
  4.2,
  4.3,
  4.4,
  4.5,
  4.6,
  4.7,
  4.8,
  4.9,
  5.0,
  5.1,
  5.2,
  5.3,
  5.4,
  5.5,
  5.6,
  5.7,
  5.8,
  5.9,
  5.0,
  6.1,
  6.2,
  6.3,
  6.4,
  6.5,
  6.6,
};

int path_dir_x[][5] = {
  {1, 7, 6 },
  {6, 7, 1 },
  {2, 7, 5 },
  {5, 7, 2 },
  {3, 7, 4 },
  {4, 7, 3 },
  {0, 1, 7, 7, 6 },
  {1, 7, 7, 6, 0 },
  {6, 7, 7, 1, 0 },
  {0, 6, 7, 7, 1 },
  {0, 2, 7, 7, 5 },
  {2, 7, 7, 5, 0 },
  {5, 7, 7, 2, 0 },
  {0, 5, 7, 7, 2 },
  {0, 3, 7, 7, 4 },
  {3, 7, 7, 4, 0 },
  {4, 7, 7, 3, 0 },
  {0, 4, 7, 7, 3 },
  {6, 6, 7, 1, 1 },
  {1, 1, 7, 6, 6 },
  {5, 5, 7, 2, 2 },
  {2, 2, 7, 5, 5 },
  {4, 4, 7, 3, 3 },
  {3, 3, 7, 4, 4 },
  {1, 2, 7, 6, 5 },
  {5, 6, 7, 2, 1 },
  {1, 5, 7, 6, 2 },
  {2, 6, 7, 5, 1 },
  {6, 2, 7, 1, 5 },
  {5, 1, 7, 2, 6 },
  {6, 5, 7, 1, 2 },
  {2, 1, 7, 5, 6 },
  {1, 3, 7, 6, 4 },
  {4, 6, 7, 3, 1 },
  {1, 4, 7, 6, 3 },
  {3, 6, 7, 4, 1 },
  {6, 3, 7, 1, 4 },
  {4, 1, 7, 3, 6 },
  {6, 4, 7, 1, 3 },
  {3, 1, 7, 4, 6 },
  {2, 3, 7, 5, 4 },
  {4, 5, 7, 3, 2 },
  {2, 4, 7, 5, 3 },
  {3, 5, 7, 4, 2 },
  {5, 3, 7, 2, 4 },
  {4, 2, 7, 3, 5 },
  {5, 4, 7, 2, 3 },
  {3, 2, 7, 4, 5 },
};


int path_dir_y[][5] = {
  { 2 ,6 ,5 },
  { 5 ,6 ,2 },
  { 3 ,6 ,4 },
  { 4 ,6 ,3 },
  { 0 ,6 ,7 },
  { 7 ,6 ,0 },
  { 1 ,2 ,6 ,6 ,5 },
  { 2 ,6 ,6 ,5 ,1 },
  { 5 ,6 ,6 ,2 ,1 },
  { 1 ,5 ,6 ,6 ,2 },
  { 1 ,3 ,6 ,6 ,4 },
  { 3 ,6 ,6 ,4 ,1 },
  { 4 ,6 ,6 ,3 ,1 },
  { 1 ,4 ,6 ,6 ,3 },
  { 1 ,0 ,6 ,6 ,7 },
  { 0 ,6 ,6 ,7 ,1 },
  { 7 ,6 ,6 ,0 ,1 },
  { 1 ,7 ,6 ,6 ,0 },
  { 5 ,5 ,6 ,2 ,2 },
  { 2 ,2 ,6 ,5 ,5 },
  { 4 ,4 ,6 ,3 ,3 },
  { 3 ,3 ,6 ,4 ,4 },
  { 7 ,7 ,6 ,0 ,0 },
  { 0 ,0 ,6 ,7 ,7 },
  { 2 ,3 ,6 ,5 ,4 },
  { 4 ,5 ,6 ,3 ,2 },
  { 2 ,4 ,6 ,5 ,3 },
  { 3 ,5 ,6 ,4 ,2 },
  { 5 ,3 ,6 ,2 ,4 },
  { 4 ,2 ,6 ,3 ,5 },
  { 5 ,4 ,6 ,2 ,3 },
  { 3 ,2 ,6 ,4 ,5 },
  { 2 ,0 ,6 ,5 ,7 },
  { 7 ,5 ,6 ,0 ,2 },
  { 2 ,7 ,6 ,5 ,0 },
  { 0 ,5 ,6 ,7 ,2 },
  { 5 ,0 ,6 ,2 ,7 },
  { 7 ,2 ,6 ,0 ,5 },
  { 5 ,7 ,6 ,2 ,0 },
  { 0 ,2 ,6 ,7 ,5 },
  { 3 ,0 ,6 ,4 ,7 },
  { 7 ,4 ,6 ,0 ,3 },
  { 3 ,7 ,6 ,4 ,0 },
  { 0 ,4 ,6 ,7 ,3 },
  { 4 ,0 ,6 ,3 ,7 },
  { 7 ,3 ,6 ,0 ,4 },
  { 4 ,7 ,6 ,3 ,0 },
  { 0 ,3 ,6 ,7 ,4 }
};

int path_dir_z[][5] = {
  {3, 5, 4},       {4, 5, 3},       {0, 5, 7},       {7, 5, 0},       {1, 5, 6},       {6, 5, 1},       {2, 3, 5, 5, 4},
  {3, 5, 5, 4, 2}, {4, 5, 5, 3, 2}, {2, 4, 5, 5, 3}, {2, 0, 5, 5, 7}, {0, 5, 5, 7, 2}, {7, 5, 5, 0, 2}, {2, 7, 5, 5, 0},
  {2, 1, 5, 5, 6}, {1, 5, 5, 6, 2}, {6, 5, 5, 1, 2}, {2, 6, 5, 5, 1}, {4, 4, 5, 3, 3}, {3, 3, 5, 4, 4}, {7, 7, 5, 0, 0},
  {0, 0, 5, 7, 7}, {6, 6, 5, 1, 1}, {1, 1, 5, 6, 6}, {3, 0, 5, 4, 7}, {7, 4, 5, 0, 3}, {3, 7, 5, 4, 0}, {0, 4, 5, 7, 3},
  {4, 0, 5, 3, 7}, {7, 3, 5, 0, 4}, {4, 7, 5, 3, 0}, {0, 3, 5, 7, 4}, {3, 1, 5, 4, 6}, {6, 4, 5, 1, 3}, {3, 6, 5, 4, 1},
  {1, 4, 5, 6, 3}, {4, 1, 5, 3, 6}, {6, 3, 5, 1, 4}, {4, 6, 5, 3, 1}, {1, 3, 5, 6, 4}, {0, 1, 5, 7, 6}, {6, 7, 5, 1, 0},
  {0, 6, 5, 7, 1}, {1, 7, 5, 6, 0}, {7, 1, 5, 0, 6}, {6, 0, 5, 1, 7}, {7, 6, 5, 0, 1}, {1, 0, 5, 6, 7}};

int path_dir_t[][5] = {
  { 0 ,4 ,7 },
  { 7 ,4 ,0 },
  { 1 ,4 ,6 },
  { 6 ,4 ,1 },
  { 2 ,4 ,5 },
  { 5 ,4 ,2 },
  { 3 ,0 ,4 ,4 ,7 },
  { 0 ,4 ,4 ,7 ,3 },
  { 7 ,4 ,4 ,0 ,3 },
  { 3 ,7 ,4 ,4 ,0 },
  { 3 ,1 ,4 ,4 ,6 },
  { 1 ,4 ,4 ,6 ,3 },
  { 6 ,4 ,4 ,1 ,3 },
  { 3 ,6 ,4 ,4 ,1 },
  { 3 ,2 ,4 ,4 ,5 },
  { 2 ,4 ,4 ,5 ,3 },
  { 5 ,4 ,4 ,2 ,3 },
  { 3 ,5 ,4 ,4 ,2 },
  { 7 ,7 ,4 ,0 ,0 },
  { 0 ,0 ,4 ,7 ,7 },
  { 6 ,6 ,4 ,1 ,1 },
  { 1 ,1 ,4 ,6 ,6 },
  { 5 ,5 ,4 ,2 ,2 },
  { 2 ,2 ,4 ,5 ,5 },
  { 0 ,1 ,4 ,7 ,6 },
  { 6 ,7 ,4 ,1 ,0 },
  { 0 ,6 ,4 ,7 ,1 },
  { 1 ,7 ,4 ,6 ,0 },
  { 7 ,1 ,4 ,0 ,6 },
  { 6 ,0 ,4 ,1 ,7 },
  { 7 ,6 ,4 ,0 ,1 },
  { 1 ,0 ,4 ,6 ,7 },
  { 0 ,2 ,4 ,7 ,5 },
  { 5 ,7 ,4 ,2 ,0 },
  { 0 ,5 ,4 ,7 ,2 },
  { 2 ,7 ,4 ,5 ,0 },
  { 7 ,2 ,4 ,0 ,5 },
  { 5 ,0 ,4 ,2 ,7 },
  { 7 ,5 ,4 ,0 ,2 },
  { 2 ,0 ,4 ,5 ,7 },
  { 1 ,2 ,4 ,6 ,5 },
  { 5 ,6 ,4 ,2 ,1 },
  { 1 ,5 ,4 ,6 ,2 },
  { 2 ,6 ,4 ,5 ,1 },
  { 6 ,2 ,4 ,1 ,5 },
  { 5 ,1 ,4 ,2 ,6 },
  { 6 ,5 ,4 ,1 ,2 },
  { 2 ,1 ,4 ,5 ,6 }
};

static double force_check;
static double deviation;

void gauge_force_test(void)
{
  int max_length = 6;

  initQuda(device);
  setVerbosityQuda(QUDA_VERBOSE,"",stdout);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  gauge_param.gauge_order = gauge_order;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  setDims(gauge_param.X);

  double loop_coeff_d[sizeof(loop_coeff_f)/sizeof(float)];
  for(unsigned int i=0;i < sizeof(loop_coeff_f)/sizeof(float); i++){
    loop_coeff_d[i] = loop_coeff_f[i];
  }

  void* loop_coeff;
  if (gauge_param.cpu_prec == QUDA_SINGLE_PRECISION) {
    loop_coeff = (void*)&loop_coeff_f[0];
  } else {
    loop_coeff = loop_coeff_d;
  }
  double eb3 = 0.3;
  int num_paths = sizeof(path_dir_x)/sizeof(path_dir_x[0]);

  int** input_path_buf[4];
  for (int dir = 0; dir < 4; dir++) {
    input_path_buf[dir] = (int **)safe_malloc(num_paths * sizeof(int *));
    for (int i = 0; i < num_paths; i++) {
      input_path_buf[dir][i] = (int*)safe_malloc(length[i]*sizeof(int));
      if (dir == 0)
        memcpy(input_path_buf[dir][i], path_dir_x[i], length[i] * sizeof(int));
      else if (dir ==1) memcpy(input_path_buf[dir][i], path_dir_y[i], length[i]*sizeof(int));
      else if (dir ==2) memcpy(input_path_buf[dir][i], path_dir_z[i], length[i]*sizeof(int));
      else if (dir ==3) memcpy(input_path_buf[dir][i], path_dir_t[i], length[i]*sizeof(int));
    }
  }

  quda::GaugeFieldParam param(0, gauge_param);
  param.create = QUDA_NULL_FIELD_CREATE;
  param.order = QUDA_QDP_GAUGE_ORDER;
  auto U_qdp = new quda::cpuGaugeField(param);

  // fills the gauge field with random numbers
  createSiteLinkCPU((void **)U_qdp->Gauge_p(), gauge_param.cpu_prec, 0);

  param.order = QUDA_MILC_GAUGE_ORDER;
  auto U_milc = new quda::cpuGaugeField(param);
  if (gauge_order == QUDA_MILC_GAUGE_ORDER) U_milc->copy(*U_qdp);
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  auto Mom_milc = new quda::cpuGaugeField(param);
  auto Mom_ref_milc = new quda::cpuGaugeField(param);

  param.order = QUDA_QDP_GAUGE_ORDER;
  auto Mom_qdp = new quda::cpuGaugeField(param);

  // initialize some data in cpuMom
  createMomCPU(Mom_ref_milc->Gauge_p(), gauge_param.cpu_prec);
  if (gauge_order == QUDA_MILC_GAUGE_ORDER) Mom_milc->copy(*Mom_ref_milc);
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) Mom_qdp->copy(*Mom_ref_milc);
  void *mom = nullptr;
  void *sitelink = nullptr;

  if (gauge_order == QUDA_MILC_GAUGE_ORDER) {
    sitelink = U_milc->Gauge_p();
    mom = Mom_milc->Gauge_p();
  } else if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    sitelink = U_qdp->Gauge_p();
    mom = Mom_qdp->Gauge_p();
  } else {
    errorQuda("Unsupported gauge order %d", gauge_order);
  }

  if (getTuning() == QUDA_TUNE_YES)
    computeGaugeForceQuda(mom, sitelink, input_path_buf, length, loop_coeff_d, num_paths, max_length, eb3, &gauge_param);

  struct timeval t0, t1;
  double total_time = 0.0;
  // Multiple execution to exclude warmup time in the first run

  auto &Mom_ = gauge_order == QUDA_MILC_GAUGE_ORDER ? Mom_milc : Mom_qdp;
  for (int i = 0; i < niter; i++) {
    Mom_->copy(*Mom_ref_milc); // restore initial momentum for correctness
    gettimeofday(&t0, NULL);
    computeGaugeForceQuda(mom, sitelink, input_path_buf, length, loop_coeff_d, num_paths, max_length, eb3, &gauge_param);
    gettimeofday(&t1, NULL);
    total_time += t1.tv_sec - t0.tv_sec + 0.000001*(t1.tv_usec - t0.tv_usec);
  }
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) Mom_milc->copy(*Mom_qdp);

  //The number comes from CPU implementation in MILC, gauge_force_imp.c
  int flops = 153004;

  void *refmom = Mom_ref_milc->Gauge_p();
  if (verify_results) {
    gauge_force_reference(refmom, eb3, (void **)U_qdp->Gauge_p(), gauge_param.cpu_prec, input_path_buf, length,
                          loop_coeff, num_paths);
    force_check = compare_floats(Mom_milc->Gauge_p(), refmom, 4 * V * mom_site_size, getTolerance(cuda_prec), gauge_param.cpu_prec);
    strong_check_mom(Mom_milc->Gauge_p(), refmom, 4 * V, gauge_param.cpu_prec);
  }

  printfQuda("\nComputing momentum action\n");
  auto action_quda = momActionQuda(mom, &gauge_param);
  auto action_ref = mom_action(refmom, gauge_param.cpu_prec, 4 * V);
  deviation = std::abs(action_quda - action_ref) / std::abs(action_ref);
  printfQuda("QUDA action = %e, reference = %e relative deviation = %e\n", action_quda, action_ref, deviation);

  double perf = 1.0*niter*flops*V/(total_time*1e+9);
  printfQuda("total time = %.2f ms\n", total_time*1e+3);
  printfQuda("overall performance : %.2f GFLOPS\n",perf);

  for(int dir = 0; dir < 4; dir++){
    for(int i=0;i < num_paths; i++) host_free(input_path_buf[dir][i]);
    host_free(input_path_buf[dir]);
  }

  delete U_qdp;
  delete U_milc;
  delete Mom_qdp;
  delete Mom_milc;
  delete Mom_ref_milc;

  endQuda();
}

TEST(force, verify)
{
  ASSERT_EQ(force_check, 1) << "CPU and QUDA implementations do not agree";
}

TEST(action, verify)
{
  ASSERT_LE(deviation, getTolerance(cuda_prec)) << "CPU and QUDA implementations do not agree";
}

static void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("link_precision           link_reconstruct           space_dim(x/y/z)              T_dimension        Gauge_order    niter\n");
  printfQuda("%s                       %s                         %d/%d/%d                       %d                  "
             "%s           %d\n",
             get_prec_str(prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim, get_gauge_order_str(gauge_order),
             niter);
}

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;

  // command line options
  auto app = make_app();
  CLI::TransformPairs<QudaGaugeFieldOrder> gauge_order_map {{"milc", QUDA_MILC_GAUGE_ORDER},
                                                            {"qdp", QUDA_QDP_GAUGE_ORDER}};
  app->add_option("--gauge-order", gauge_order, "")->transform(CLI::QUDACheckedTransformer(gauge_order_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  gauge_force_test();

  if (verify_results) {
    // Ensure gtest prints only from rank 0
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

    test_rc = RUN_ALL_TESTS();
    if (test_rc != 0) warningQuda("Tests failed");
  }

  finalizeComms();
  return test_rc;
}
