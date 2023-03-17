#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <gauge_field.h>
#include "misc.h"
#include "gauge_force_reference.h"
#include <gauge_path_quda.h>
#include <timer.h>
#include <gtest/gtest.h>

static QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

int length[] = {
  3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
};

float loop_coeff_f[] = {
  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4,
  3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
  5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6,
};

int path_dir_x[][5] = {
  {1, 7, 6},       {6, 7, 1},       {2, 7, 5},       {5, 7, 2},       {3, 7, 4},       {4, 7, 3},       {0, 1, 7, 7, 6},
  {1, 7, 7, 6, 0}, {6, 7, 7, 1, 0}, {0, 6, 7, 7, 1}, {0, 2, 7, 7, 5}, {2, 7, 7, 5, 0}, {5, 7, 7, 2, 0}, {0, 5, 7, 7, 2},
  {0, 3, 7, 7, 4}, {3, 7, 7, 4, 0}, {4, 7, 7, 3, 0}, {0, 4, 7, 7, 3}, {6, 6, 7, 1, 1}, {1, 1, 7, 6, 6}, {5, 5, 7, 2, 2},
  {2, 2, 7, 5, 5}, {4, 4, 7, 3, 3}, {3, 3, 7, 4, 4}, {1, 2, 7, 6, 5}, {5, 6, 7, 2, 1}, {1, 5, 7, 6, 2}, {2, 6, 7, 5, 1},
  {6, 2, 7, 1, 5}, {5, 1, 7, 2, 6}, {6, 5, 7, 1, 2}, {2, 1, 7, 5, 6}, {1, 3, 7, 6, 4}, {4, 6, 7, 3, 1}, {1, 4, 7, 6, 3},
  {3, 6, 7, 4, 1}, {6, 3, 7, 1, 4}, {4, 1, 7, 3, 6}, {6, 4, 7, 1, 3}, {3, 1, 7, 4, 6}, {2, 3, 7, 5, 4}, {4, 5, 7, 3, 2},
  {2, 4, 7, 5, 3}, {3, 5, 7, 4, 2}, {5, 3, 7, 2, 4}, {4, 2, 7, 3, 5}, {5, 4, 7, 2, 3}, {3, 2, 7, 4, 5},
};

int path_dir_y[][5] = {
  {2, 6, 5},       {5, 6, 2},       {3, 6, 4},       {4, 6, 3},       {0, 6, 7},       {7, 6, 0},       {1, 2, 6, 6, 5},
  {2, 6, 6, 5, 1}, {5, 6, 6, 2, 1}, {1, 5, 6, 6, 2}, {1, 3, 6, 6, 4}, {3, 6, 6, 4, 1}, {4, 6, 6, 3, 1}, {1, 4, 6, 6, 3},
  {1, 0, 6, 6, 7}, {0, 6, 6, 7, 1}, {7, 6, 6, 0, 1}, {1, 7, 6, 6, 0}, {5, 5, 6, 2, 2}, {2, 2, 6, 5, 5}, {4, 4, 6, 3, 3},
  {3, 3, 6, 4, 4}, {7, 7, 6, 0, 0}, {0, 0, 6, 7, 7}, {2, 3, 6, 5, 4}, {4, 5, 6, 3, 2}, {2, 4, 6, 5, 3}, {3, 5, 6, 4, 2},
  {5, 3, 6, 2, 4}, {4, 2, 6, 3, 5}, {5, 4, 6, 2, 3}, {3, 2, 6, 4, 5}, {2, 0, 6, 5, 7}, {7, 5, 6, 0, 2}, {2, 7, 6, 5, 0},
  {0, 5, 6, 7, 2}, {5, 0, 6, 2, 7}, {7, 2, 6, 0, 5}, {5, 7, 6, 2, 0}, {0, 2, 6, 7, 5}, {3, 0, 6, 4, 7}, {7, 4, 6, 0, 3},
  {3, 7, 6, 4, 0}, {0, 4, 6, 7, 3}, {4, 0, 6, 3, 7}, {7, 3, 6, 0, 4}, {4, 7, 6, 3, 0}, {0, 3, 6, 7, 4}};

int path_dir_z[][5] = {
  {3, 5, 4},       {4, 5, 3},       {0, 5, 7},       {7, 5, 0},       {1, 5, 6},       {6, 5, 1},       {2, 3, 5, 5, 4},
  {3, 5, 5, 4, 2}, {4, 5, 5, 3, 2}, {2, 4, 5, 5, 3}, {2, 0, 5, 5, 7}, {0, 5, 5, 7, 2}, {7, 5, 5, 0, 2}, {2, 7, 5, 5, 0},
  {2, 1, 5, 5, 6}, {1, 5, 5, 6, 2}, {6, 5, 5, 1, 2}, {2, 6, 5, 5, 1}, {4, 4, 5, 3, 3}, {3, 3, 5, 4, 4}, {7, 7, 5, 0, 0},
  {0, 0, 5, 7, 7}, {6, 6, 5, 1, 1}, {1, 1, 5, 6, 6}, {3, 0, 5, 4, 7}, {7, 4, 5, 0, 3}, {3, 7, 5, 4, 0}, {0, 4, 5, 7, 3},
  {4, 0, 5, 3, 7}, {7, 3, 5, 0, 4}, {4, 7, 5, 3, 0}, {0, 3, 5, 7, 4}, {3, 1, 5, 4, 6}, {6, 4, 5, 1, 3}, {3, 6, 5, 4, 1},
  {1, 4, 5, 6, 3}, {4, 1, 5, 3, 6}, {6, 3, 5, 1, 4}, {4, 6, 5, 3, 1}, {1, 3, 5, 6, 4}, {0, 1, 5, 7, 6}, {6, 7, 5, 1, 0},
  {0, 6, 5, 7, 1}, {1, 7, 5, 6, 0}, {7, 1, 5, 0, 6}, {6, 0, 5, 1, 7}, {7, 6, 5, 0, 1}, {1, 0, 5, 6, 7}};

int path_dir_t[][5] = {
  {0, 4, 7},       {7, 4, 0},       {1, 4, 6},       {6, 4, 1},       {2, 4, 5},       {5, 4, 2},       {3, 0, 4, 4, 7},
  {0, 4, 4, 7, 3}, {7, 4, 4, 0, 3}, {3, 7, 4, 4, 0}, {3, 1, 4, 4, 6}, {1, 4, 4, 6, 3}, {6, 4, 4, 1, 3}, {3, 6, 4, 4, 1},
  {3, 2, 4, 4, 5}, {2, 4, 4, 5, 3}, {5, 4, 4, 2, 3}, {3, 5, 4, 4, 2}, {7, 7, 4, 0, 0}, {0, 0, 4, 7, 7}, {6, 6, 4, 1, 1},
  {1, 1, 4, 6, 6}, {5, 5, 4, 2, 2}, {2, 2, 4, 5, 5}, {0, 1, 4, 7, 6}, {6, 7, 4, 1, 0}, {0, 6, 4, 7, 1}, {1, 7, 4, 6, 0},
  {7, 1, 4, 0, 6}, {6, 0, 4, 1, 7}, {7, 6, 4, 0, 1}, {1, 0, 4, 6, 7}, {0, 2, 4, 7, 5}, {5, 7, 4, 2, 0}, {0, 5, 4, 7, 2},
  {2, 7, 4, 5, 0}, {7, 2, 4, 0, 5}, {5, 0, 4, 2, 7}, {7, 5, 4, 0, 2}, {2, 0, 4, 5, 7}, {1, 2, 4, 6, 5}, {5, 6, 4, 2, 1},
  {1, 5, 4, 6, 2}, {2, 6, 4, 5, 1}, {6, 2, 4, 1, 5}, {5, 1, 4, 2, 6}, {6, 5, 4, 1, 2}, {2, 1, 4, 5, 6}};

// for gauge loop trace tests; plaquette + rectangle only
// do not change---these are used for a verification relative
// to the plaquette as well
int trace_loop_length[] = {4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};

double trace_loop_coeff_d[]
  = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6};

int trace_path[][6] {{0, 1, 7, 6},       {0, 2, 7, 5},       {0, 3, 7, 4},       {1, 2, 6, 5},       {1, 3, 6, 4},
                     {2, 3, 5, 4},       {0, 0, 1, 7, 7, 6}, {0, 1, 1, 7, 6, 6}, {0, 0, 2, 7, 7, 5}, {0, 2, 2, 7, 5, 5},
                     {0, 0, 3, 7, 7, 4}, {0, 3, 3, 7, 4, 4}, {1, 1, 2, 6, 6, 5}, {1, 2, 2, 6, 5, 5}, {1, 1, 3, 6, 6, 4},
                     {1, 3, 3, 6, 4, 4}, {2, 2, 3, 5, 5, 4}, {2, 3, 3, 5, 4, 4}};

static int force_check;
static int path_check;
static double force_deviation;
static double loop_deviation;
static double plaq_deviation;

// The same function is used to test computePath.
// If compute_force is false then a path is computed
void gauge_force_test(bool compute_force = true)
{
  int max_length = 6;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  gauge_param.gauge_order = gauge_order;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  setDims(gauge_param.X);

  double loop_coeff_d[sizeof(loop_coeff_f) / sizeof(float)];
  for (unsigned int i = 0; i < sizeof(loop_coeff_f) / sizeof(float); i++) { loop_coeff_d[i] = loop_coeff_f[i]; }

  void *loop_coeff;
  if (gauge_param.cpu_prec == QUDA_SINGLE_PRECISION) {
    loop_coeff = (void *)&loop_coeff_f[0];
  } else {
    loop_coeff = loop_coeff_d;
  }
  double eb3 = 0.3;
  int num_paths = sizeof(path_dir_x) / sizeof(path_dir_x[0]);

  int **input_path_buf[4];
  for (int dir = 0; dir < 4; dir++) {
    input_path_buf[dir] = (int **)safe_malloc(num_paths * sizeof(int *));
    for (int i = 0; i < num_paths; i++) {
      input_path_buf[dir][i] = (int *)safe_malloc(length[i] * sizeof(int));
      if (dir == 0)
        memcpy(input_path_buf[dir][i], path_dir_x[i], length[i] * sizeof(int));
      else if (dir == 1)
        memcpy(input_path_buf[dir][i], path_dir_y[i], length[i] * sizeof(int));
      else if (dir == 2)
        memcpy(input_path_buf[dir][i], path_dir_z[i], length[i] * sizeof(int));
      else if (dir == 3)
        memcpy(input_path_buf[dir][i], path_dir_t[i], length[i] * sizeof(int));
    }
  }

  quda::GaugeFieldParam param(gauge_param);
  param.create = QUDA_NULL_FIELD_CREATE;
  param.order = QUDA_QDP_GAUGE_ORDER;
  param.location = QUDA_CPU_FIELD_LOCATION;
  quda::cpuGaugeField U_qdp(param);

  // fills the gauge field with random numbers
  createSiteLinkCPU((void **)U_qdp.Gauge_p(), gauge_param.cpu_prec, 0);

  param.order = QUDA_MILC_GAUGE_ORDER;
  quda::cpuGaugeField U_milc(param);
  if (gauge_order == QUDA_MILC_GAUGE_ORDER) U_milc.copy(U_qdp);
  if (compute_force) {
    param.reconstruct = QUDA_RECONSTRUCT_10;
    param.link_type = QUDA_ASQTAD_MOM_LINKS;
  } else {
    param.reconstruct = QUDA_RECONSTRUCT_NO;
  }
  param.create = QUDA_ZERO_FIELD_CREATE;
  quda::cpuGaugeField Mom_milc(param);
  quda::cpuGaugeField Mom_ref_milc(param);

  param.order = QUDA_QDP_GAUGE_ORDER;
  quda::cpuGaugeField Mom_qdp(param);

  // initialize some data in cpuMom
  if (compute_force) {
    createMomCPU(Mom_ref_milc.Gauge_p(), gauge_param.cpu_prec);
    if (gauge_order == QUDA_MILC_GAUGE_ORDER) Mom_milc.copy(Mom_ref_milc);
    if (gauge_order == QUDA_QDP_GAUGE_ORDER) Mom_qdp.copy(Mom_ref_milc);
  }
  void *mom = nullptr;
  void *sitelink = nullptr;

  if (gauge_order == QUDA_MILC_GAUGE_ORDER) {
    sitelink = U_milc.Gauge_p();
    mom = Mom_milc.Gauge_p();
  } else if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    sitelink = U_qdp.Gauge_p();
    mom = Mom_qdp.Gauge_p();
  } else {
    errorQuda("Unsupported gauge order %d", gauge_order);
  }

  if (getTuning() == QUDA_TUNE_YES) {
    if (compute_force)
      computeGaugeForceQuda(mom, sitelink, input_path_buf, length, loop_coeff_d, num_paths, max_length, eb3,
                            &gauge_param);
    else
      computeGaugePathQuda(mom, sitelink, input_path_buf, length, loop_coeff_d, num_paths, max_length, eb3, &gauge_param);
  }

  quda::host_timer_t host_timer;
  double time_sec = 0.0;
  // Multiple execution to exclude warmup time in the first run

  auto &Mom_ = gauge_order == QUDA_MILC_GAUGE_ORDER ? Mom_milc : Mom_qdp;
  for (int i = 0; i < niter; i++) {
    Mom_.copy(Mom_ref_milc); // restore initial momentum for correctness
    host_timer.start();
    if (compute_force)
      computeGaugeForceQuda(mom, sitelink, input_path_buf, length, loop_coeff_d, num_paths, max_length, eb3,
                            &gauge_param);
    else
      computeGaugePathQuda(mom, sitelink, input_path_buf, length, loop_coeff_d, num_paths, max_length, eb3, &gauge_param);
    host_timer.stop();
    time_sec += host_timer.last();
  }
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) Mom_milc.copy(Mom_qdp);

  // The number comes from CPU implementation in MILC, gauge_force_imp.c
  int flops = 153004;

  void *refmom = Mom_ref_milc.Gauge_p();
  int *check_out = compute_force ? &force_check : &path_check;
  if (verify_results) {
    gauge_force_reference(refmom, eb3, (void **)U_qdp.Gauge_p(), gauge_param.cpu_prec, input_path_buf, length,
                          loop_coeff, num_paths, compute_force);
    *check_out
      = compare_floats(Mom_milc.Gauge_p(), refmom, 4 * V * mom_site_size, getTolerance(cuda_prec), gauge_param.cpu_prec);
    if (compute_force) strong_check_mom(Mom_milc.Gauge_p(), refmom, 4 * V, gauge_param.cpu_prec);
  }

  if (compute_force) {
    logQuda(QUDA_VERBOSE, "\nComputing momentum action\n");
    auto action_quda = momActionQuda(mom, &gauge_param);
    auto action_ref = mom_action(refmom, gauge_param.cpu_prec, 4 * V);
    force_deviation = std::abs(action_quda - action_ref) / std::abs(action_ref);
    logQuda(QUDA_VERBOSE, "QUDA action = %e, reference = %e relative deviation = %e\n", action_quda, action_ref,
            force_deviation);
  }

  double perf = 1.0 * niter * flops * V / (time_sec * 1e+9);
  if (compute_force) {
    printfQuda("Force calculation total time = %.2f ms ; overall performance : %.2f GFLOPS\n", time_sec * 1e+3, perf);
  } else {
    printfQuda("Gauge path calculation total time = %.2f ms ; overall performance : %.2f GFLOPS\n", time_sec * 1e+3,
               perf);
  }
  for (int dir = 0; dir < 4; dir++) {
    for (int i = 0; i < num_paths; i++) host_free(input_path_buf[dir][i]);
    host_free(input_path_buf[dir]);
  }
}

void gauge_loop_test()
{
  int max_length = 6;

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  gauge_param.gauge_order = gauge_order;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  setDims(gauge_param.X);

  double *trace_loop_coeff_p = &trace_loop_coeff_d[0];
  int num_paths = sizeof(trace_path) / sizeof(trace_path[0]);
  int *trace_loop_length_p = &trace_loop_length[0];

  // b/c we can't cast int*[6] -> int**
  int **trace_path_p = new int *[num_paths];
  for (int i = 0; i < num_paths; i++) {
    trace_path_p[i] = new int[max_length];
    for (int j = 0; j < trace_loop_length[i]; j++) trace_path_p[i][j] = trace_path[i][j];
  }

  quda::GaugeFieldParam param(gauge_param);
  param.create = QUDA_NULL_FIELD_CREATE;
  param.order = QUDA_QDP_GAUGE_ORDER;
  param.location = QUDA_CPU_FIELD_LOCATION;
  quda::cpuGaugeField U_qdp(param);

  // fills the gauge field with random numbers
  createSiteLinkCPU((void **)U_qdp.Gauge_p(), gauge_param.cpu_prec, 0);

  param.order = QUDA_MILC_GAUGE_ORDER;
  quda::cpuGaugeField U_milc(param);
  if (gauge_order == QUDA_MILC_GAUGE_ORDER) U_milc.copy(U_qdp);

  void *sitelink = nullptr;

  if (gauge_order == QUDA_MILC_GAUGE_ORDER) {
    sitelink = U_milc.Gauge_p();
  } else if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    sitelink = U_qdp.Gauge_p();
  } else {
    errorQuda("Unsupported gauge order %d", gauge_order);
  }

  // upload gauge field
  loadGaugeQuda(sitelink, &gauge_param);

  // storage for traces
  using double_complex = double _Complex;
  std::vector<double_complex> traces(num_paths);
  double scale_factor = 2.0;

  // compute various observables
  QudaGaugeObservableParam obsParam = newQudaGaugeObservableParam();
  obsParam.compute_gauge_loop_trace = QUDA_BOOLEAN_TRUE;
  obsParam.traces = traces.data();
  obsParam.input_path_buff = trace_path_p;
  obsParam.path_length = trace_loop_length_p;
  obsParam.loop_coeff = trace_loop_coeff_p;
  obsParam.num_paths = num_paths;
  obsParam.max_length = max_length;
  obsParam.factor = scale_factor;

  // compute the plaquette as part of validation
  obsParam.compute_plaquette = QUDA_BOOLEAN_TRUE;

  if (getTuning() == QUDA_TUNE_YES) { gaugeObservablesQuda(&obsParam); }

  quda::host_timer_t host_timer;
  // Multiple execution to exclude warmup time in the first run

  host_timer.start();
  for (int i = 0; i < niter; i++) { gaugeObservablesQuda(&obsParam); }
  host_timer.stop();

  // 6 loops of length 4, 12 loops of length 6 + 18 paths worth of traces and rescales
  int flops = (4 * 6 + 6 * 12) * 198 + 18 * 8;

  std::vector<quda::Complex> traces_ref(num_paths);

  if (verify_results) {
    gauge_loop_trace_reference((void **)U_qdp.Gauge_p(), gauge_param.cpu_prec, traces_ref, scale_factor, trace_path_p,
                               trace_loop_length_p, trace_loop_coeff_p, num_paths);

    loop_deviation = 0;
    for (int i = 0; i < num_paths; i++) {
      double *t_ptr = (double *)(&traces[i]);
      std::complex<double> traces_(t_ptr[0], t_ptr[1]);
      auto diff = std::abs(traces_ref[i] - traces_);
      auto norm = std::abs(traces_ref[i]);
      loop_deviation += diff / norm;
      logQuda(QUDA_VERBOSE, "Loop %d QUDA trace %e + I %e Reference trace %e + I %e Deviation %e\n", i, traces_.real(),
              traces_.imag(), traces_ref[i].real(), traces_ref[i].imag(), diff / norm);
    }

    // Second check: we can reconstruct the plaquette from the first six loops we calculated
    double plaq_factor = 1. / (V * U_qdp.Ncolor() * quda::comm_size());
    std::vector<quda::Complex> plaq_components(6);
    for (int i = 0; i < 6; i++) plaq_components[i] = traces_ref[i] / trace_loop_coeff_d[i] / scale_factor * plaq_factor;

    double plaq_loop[3];
    // spatial: xy, xz, yz
    plaq_loop[1] = ((plaq_components[0] + plaq_components[1] + plaq_components[3]) / 3.).real();
    // temporal: xt, yt, zt
    plaq_loop[2] = ((plaq_components[2] + plaq_components[4] + plaq_components[5]) / 3.).real();
    plaq_loop[0] = 0.5 * (plaq_loop[1] + plaq_loop[2]);

    plaq_deviation = std::abs(obsParam.plaquette[0] - plaq_loop[0]) / std::abs(obsParam.plaquette[0]);
    logQuda(QUDA_VERBOSE,
            "Plaquette loop space %e time %e total %e ; plaqQuda space %e time %e total %e ; deviation %e\n",
            plaq_loop[0], plaq_loop[1], plaq_loop[2], obsParam.plaquette[0], obsParam.plaquette[1],
            obsParam.plaquette[2], plaq_deviation);
  }

  double perf = 1.0 * niter * flops * V / (host_timer.last() * 1e+9);
  printfQuda("Gauge loop trace total time = %.2f ms ; overall performance : %.2f GFLOPS\n", host_timer.last() * 1e+3,
             perf);

  freeGaugeQuda();

  for (int i = 0; i < num_paths; i++) delete[] trace_path_p[i];
  delete[] trace_path_p;
}

TEST(force, verify) { ASSERT_EQ(force_check, 1) << "CPU and QUDA force implementations do not agree"; }

TEST(action, verify)
{
  ASSERT_LE(force_deviation, getTolerance(cuda_prec)) << "CPU and QUDA momentum action implementations do not agree";
}

TEST(path, verify) { ASSERT_EQ(path_check, 1) << "CPU and QUDA path implementations do not agree"; }

TEST(loop_traces, verify)
{
  ASSERT_LE(loop_deviation, getTolerance(cuda_prec)) << "CPU and QUDA loop trace implementations do not agree";
}

TEST(plaquette, verify)
{
  ASSERT_LE(plaq_deviation, getTolerance(cuda_prec))
    << "Plaquette from QUDA loop trace and QUDA dedicated plaquette function do not agree";
}

static void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("link_precision           link_reconstruct           space_dim(x/y/z)              T_dimension        "
             "Gauge_order    niter\n");
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

  initQuda(device_ordinal);

  setVerbosity(verbosity);

  display_test_info();

  gauge_force_test();

  // The same test is also used for gauge path (compute_force=false)
  gauge_force_test(false);

  gauge_loop_test();

  if (verify_results) {
    // Ensure gtest prints only from rank 0
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

    test_rc = RUN_ALL_TESTS();
    if (test_rc != 0) warningQuda("Tests failed");
  }

  endQuda();
  finalizeComms();
  return test_rc;
}
