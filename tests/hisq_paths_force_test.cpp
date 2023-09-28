#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <quda.h>
#include "host_utils.h"
#include <command_line_params.h>
#include "gauge_field.h"
#include "misc.h"
#include "hisq_force_reference.h"
#include "ks_improved_force.h"
#include "momentum.h"
#include <timer.h>
#include <gtest/gtest.h>

using namespace quda;

cpuGaugeField *cpuGauge = NULL;
cudaGaugeField *cudaForce = NULL;
cpuGaugeField *cpuForce = NULL;
cpuGaugeField *hostVerifyForce = NULL;

cudaGaugeField *cudaMom = NULL;
cpuGaugeField *cpuMom = NULL;
cpuGaugeField *refMom = NULL;

QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

cpuGaugeField *cpuOprod = NULL;
cudaGaugeField *cudaOprod = NULL;
cpuGaugeField *cpuLongLinkOprod = NULL;
cudaGaugeField *cudaLongLinkOprod = NULL;

int ODD_BIT = 1;

QudaPrecision force_prec = QUDA_DOUBLE_PRECISION;

cudaGaugeField *cudaGauge_ex = NULL;
cpuGaugeField *cpuGauge_ex = NULL;
cudaGaugeField *cudaForce_ex = NULL;
cpuGaugeField *cpuForce_ex = NULL;
cpuGaugeField *cpuOprod_ex = NULL;
cudaGaugeField *cudaOprod_ex = NULL;
cpuGaugeField *cpuLongLinkOprod_ex = NULL;
cudaGaugeField *cudaLongLinkOprod_ex = NULL;

static void setPrecision(QudaPrecision precision)
{
  force_prec = precision;
  return;
}

/**
  @brief Compute the aggregate bytes and flops for various components within the HISQ force

  @param[in] prec Input precision
  @param[in] recon Gauge link reconstruct
  @param[in] has_lepage Whether or not there is a Lepage contribution
  @param[out] staples_io Bytes streamed for the force contribution from HISQ staples
  @param[out] staples_flops FLOPS for the force contribution from HISQ staples
  @param[out] long_io Bytes streamed for the force contribution from the HISQ long link
  @param[out] long_flops FLOPS for the force contribution from the HISQ long link
  @param[out] complete_io Bytes streamed for the force contribution from the force completion
  @param[out] complete_flops FLOPS for the force contribution from the force completion
*/
void total_staple_io_flops(QudaPrecision prec, QudaReconstructType recon, bool has_lepage, long long &staples_io,
                           long long &staples_flops, long long &long_io, long long &long_flops, long long &complete_io,
                           long long &complete_flops)
{
  // TO DO: update me with new counts in hisq kernel core file

  // total IO counting for the middle/side/all link kernels
  // Explanation about these numbers can be founed in the corresponding kernel functions in
  // the hisq kernel core file
  long long linksize = prec * recon;
  long long cmsize = prec * 18ll;

  // FLOPS for matrix multiply, matrix add, matrix rescale, and anti-Hermitian projection
  long long matrix_flops[4] = {198ll, 18ll, 18ll, 23ll};

  // fully fused all directions
  int one_link_io[2] = {0, 12}; // no links, 12 color matrices
  int one_link_flops[4] = {0, 4, 4, 0};

  // The outer (left-most) [2] is for forwards/backwards

  // The inner (right-most) [2] is for link vs color matrix
  int three_link_io[2][2] = {{72, 192}, {48, 144}};
  int three_lepage_link_io[2][2] = {{220, 288}, {172, 192}};
  int five_seven_link_io[2][2] = {{1176, 1872}, {1176, 1440}};

  // The inner (right-most) [4] corresponds to multiply/add/rescale/projection
  int three_link_flops[2][4] = {{96, 48, 48, 0}, {48, 24, 24, 0}};
  int three_lepage_link_flops[2][4] = {{288, 120, 120, 0}, {192, 72, 72, 0}};
  int five_seven_link_flops[2][4] = {{1920, 864, 864, 0}, {1440, 576, 576, 0}};

  // Compute the bytes loaded/stored
  long long staples_io_parts[2] = {0ll, 0ll};
  for (int m = 0; m < 2; m++) {
    staples_io_parts[m] = one_link_io[m] + five_seven_link_io[0][m] + five_seven_link_io[1][m];
    if (has_lepage)
      staples_io_parts[m] += three_lepage_link_io[0][m] + three_lepage_link_io[1][m];
    else
      staples_io_parts[m] += three_link_io[0][m] + three_link_io[1][m];
  }
  staples_io = (staples_io_parts[0] * linksize + staples_io_parts[1] * cmsize) * V;

  // Compute the total number of bytes
  staples_flops = 0;
  // multiply/add/rescale/antiherm-proj
  for (int i = 0; i < 4; i++) {
    staples_flops += one_link_flops[i] * matrix_flops[i];
    // forwards vs backwards
    for (int d = 0; d < 2; d++) {
      staples_flops += five_seven_link_flops[d][i] * matrix_flops[i];
      if (has_lepage)
        staples_flops += three_lepage_link_flops[d][i] * matrix_flops[i];
      else
        staples_flops += three_link_flops[d][i] * matrix_flops[i];
    }
  }
  staples_flops *= V;

  printfQuda("staples flop/byte = %.4f lepage %c\n",
             static_cast<double>(staples_flops) / static_cast<double>(staples_io), has_lepage ? 'y' : 'n');

  // the long link is only computed during the first part of the chain rule,
  // which is also when the Lepage term is included
  long_io = 0;
  long_flops = 0;
  if (has_lepage) {
    long_io = (16 * linksize + 20 * cmsize) * V;

    int long_link_flops[4] = {24, 12, 4, 0};
    for (int i = 0; i < 4; i++) long_flops += long_link_flops[i] * matrix_flops[i];
    long_flops *= V;

    printfQuda("long flop/byte = %.4f\n", static_cast<double>(long_flops) / static_cast<double>(long_io));
  }

  // In both cases, the force complete routine is called
  complete_io = (4 * linksize + 8 * cmsize) * V;
  int complete_link_flops[4] = {4, 0, 4, 4};
  complete_flops = 0;
  for (int i = 0; i < 4; i++) complete_flops += complete_link_flops[i] * matrix_flops[i];
  printfQuda("complete flop/byte = %.4f\n", static_cast<double>(complete_flops) / static_cast<double>(complete_io));
  return;
}

#ifdef MULTI_GPU
static lat_dim_t R = {2, 2, 2, 2};
#else
static lat_dim_t R = {0, 0, 0, 0};
#endif

// one-time initializations at start of tests
static void hisq_force_startup()
{
  int X[4] = {xdim, ydim, zdim, tdim};
  setDims(X);

  setVerbosity(verbosity);

  quda::RNG *rng;

  // initialize RNG
  {
    // Create a dummy field with which to initialize the RNG
    quda::ColorSpinorParam param;
    param.v = nullptr;
    param.nColor = 3;
    param.nSpin = 1;
    param.setPrecision(force_prec);
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.nDim = 4;
    param.pc_type = QUDA_4D_PC;
    param.siteSubset = QUDA_FULL_SITE_SUBSET;
    param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    param.location = QUDA_CPU_FIELD_LOCATION;
    for (int d = 0; d < 4; d++) param.x[d] = X[d];
    quda::ColorSpinorField spinor_in(param);
    rng = new quda::RNG(spinor_in, 1234);
  }

  QudaGaugeParam qudaGaugeParam = newQudaGaugeParam();
  QudaGaugeParam qudaGaugeParam_ex;

  for (int d = 0; d < 4; d++) qudaGaugeParam.X[d] = X[d];

  // need to do some thinking for recon
  qudaGaugeParam.cpu_prec = force_prec;
  qudaGaugeParam.cuda_prec = force_prec;
  qudaGaugeParam.reconstruct = (link_recon == QUDA_RECONSTRUCT_12 ? QUDA_RECONSTRUCT_13 : link_recon);
  qudaGaugeParam.type = QUDA_GENERAL_LINKS;
  qudaGaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  qudaGaugeParam.staggered_phase_type
    = (link_recon == QUDA_RECONSTRUCT_12 ? QUDA_STAGGERED_PHASE_MILC : QUDA_STAGGERED_PHASE_NO);
  qudaGaugeParam.staggered_phase_applied = true;
  qudaGaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  qudaGaugeParam.anisotropy = 1.0;
  qudaGaugeParam.tadpole_coeff = 1.0;

  memcpy(&qudaGaugeParam_ex, &qudaGaugeParam, sizeof(QudaGaugeParam));

  int pad_size = 0;
#ifdef MULTI_GPU
  int x_face_size = qudaGaugeParam_ex.X[1] * qudaGaugeParam_ex.X[2] * qudaGaugeParam_ex.X[3] / 2;
  int y_face_size = qudaGaugeParam_ex.X[0] * qudaGaugeParam_ex.X[2] * qudaGaugeParam_ex.X[3] / 2;
  int z_face_size = qudaGaugeParam_ex.X[0] * qudaGaugeParam_ex.X[1] * qudaGaugeParam_ex.X[3] / 2;
  int t_face_size = qudaGaugeParam_ex.X[0] * qudaGaugeParam_ex.X[1] * qudaGaugeParam_ex.X[2] / 2;
  pad_size = std::max({x_face_size, y_face_size, z_face_size, t_face_size});
#endif
  qudaGaugeParam_ex.ga_pad = 3 * pad_size; // long links

  GaugeFieldParam gParam_ex;
  GaugeFieldParam gParam;

  // create device gauge field
  gParam_ex = GaugeFieldParam(qudaGaugeParam_ex);
  gParam_ex.location = QUDA_CUDA_FIELD_LOCATION;
  gParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.reconstruct = (link_recon == QUDA_RECONSTRUCT_12 ? QUDA_RECONSTRUCT_13 : link_recon);
  gParam_ex.setPrecision(force_prec, true);
  for (int d = 0; d < 4; d++) {
    gParam_ex.r[d] = (comm_dim_partitioned(d)) ? 2 : 0;
    gParam_ex.x[d] = X[d] + 2 * gParam_ex.r[d];
  } // set halo region for GPU
  cudaGauge_ex = new cudaGaugeField(gParam_ex);

  // Create the host gauge field
  memcpy(&qudaGaugeParam_ex, &qudaGaugeParam, sizeof(QudaGaugeParam));

  gParam = GaugeFieldParam(qudaGaugeParam);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.order = gauge_order;
  cpuGauge = new cpuGaugeField(gParam);

  gParam_ex = GaugeFieldParam(qudaGaugeParam_ex);
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.link_type = QUDA_GENERAL_LINKS;
  gParam_ex.order = QUDA_QDP_GAUGE_ORDER;
  for (int d = 0; d < 4; d++) {
    gParam_ex.r[d] = R[d];
    gParam_ex.x[d] = gParam.x[d] + 2 * gParam_ex.r[d];
  } // set halo region for CPU
  cpuGauge_ex = new cpuGaugeField(gParam_ex);

  auto generated_link_type = (link_recon == QUDA_RECONSTRUCT_NO ?
                                SITELINK_PHASE_NO :
                                (link_recon == QUDA_RECONSTRUCT_13 ? SITELINK_PHASE_U1 : SITELINK_PHASE_MILC));
  createSiteLinkCPU((void **)cpuGauge->Gauge_p(), qudaGaugeParam.cpu_prec, generated_link_type);
  copyExtendedGauge(*cpuGauge_ex, *cpuGauge, QUDA_CPU_FIELD_LOCATION);

  qudaGaugeParam.type = QUDA_GENERAL_LINKS;
  qudaGaugeParam.t_boundary = QUDA_PERIODIC_T;
  qudaGaugeParam.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
  qudaGaugeParam.staggered_phase_applied = false;
  memcpy(&qudaGaugeParam_ex, &qudaGaugeParam, sizeof(QudaGaugeParam));

  gParam = GaugeFieldParam(qudaGaugeParam);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;

  gParam_ex = GaugeFieldParam(qudaGaugeParam_ex);
  gParam_ex.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;

  /**************************
   * Create the force fields *
   **************************/
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.setPrecision(prec);
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.order = gauge_order;
  cpuForce = new cpuGaugeField(gParam);
  hostVerifyForce = new cpuGaugeField(gParam);

  gParam_ex.location = QUDA_CPU_FIELD_LOCATION;
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam_ex.setPrecision(prec);
  gParam_ex.create = QUDA_NULL_FIELD_CREATE;
  gParam_ex.link_type = QUDA_GENERAL_LINKS;
  gParam_ex.order = gauge_order;
  for (int d = 0; d < 4; d++) {
    gParam_ex.r[d] = R[d];
    gParam_ex.x[d] = gParam.x[d] + 2 * gParam_ex.r[d];
  }
  cpuForce_ex = new cpuGaugeField(gParam_ex);

  // create the momentum matrix
  gParam.location = QUDA_CPU_FIELD_LOCATION;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.setPrecision(prec);
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.order = QUDA_MILC_GAUGE_ORDER;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  cpuMom = new cpuGaugeField(gParam);
  refMom = new cpuGaugeField(gParam);

  /**********************************
   * Create the outer product fields *
   **********************************/

  // Create four full-volume random spinor fields
  void *stag_for_oprod = safe_malloc(4 * cpuGauge->Volume() * stag_spinor_site_size * force_prec);

  // Allocate the outer product fields and populate them with the random spinor fields
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.order = gauge_order;
  cpuOprod = new cpuGaugeField(gParam);
  cpuLongLinkOprod = new cpuGaugeField(gParam);

  // Create extended outer product fields
  gParam_ex.location = QUDA_CPU_FIELD_LOCATION;
  gParam_ex.link_type = QUDA_GENERAL_LINKS;
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam_ex.order = QUDA_QDP_GAUGE_ORDER;
  for (int d = 0; d < 4; d++) {
    gParam_ex.r[d] = R[d];
    gParam_ex.x[d] = gParam.x[d] + 2 * gParam_ex.r[d];
  } // set halo region for CPU
  cpuOprod_ex = new cpuGaugeField(gParam_ex);
  cpuLongLinkOprod_ex = new cpuGaugeField(gParam_ex);

  // initialize the CPU outer product fields and exchange once
  createStagForOprodCPU(stag_for_oprod, force_prec, qudaGaugeParam.X, *rng);
  computeLinkOrderedOuterProduct(stag_for_oprod, cpuOprod->Gauge_p(), force_prec, 1);
  computeLinkOrderedOuterProduct(stag_for_oprod, cpuLongLinkOprod->Gauge_p(), force_prec, 3);

  copyExtendedGauge(*cpuOprod_ex, *cpuOprod, QUDA_CPU_FIELD_LOCATION);
  copyExtendedGauge(*cpuLongLinkOprod_ex, *cpuLongLinkOprod, QUDA_CPU_FIELD_LOCATION);

  // free the initial spinor field
  host_free(stag_for_oprod);

  /**************************
   * Create remaining fields *
   ***************************/
  gParam_ex.location = QUDA_CUDA_FIELD_LOCATION;
  gParam_ex.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam_ex.setPrecision(prec, true);
  for (int d = 0; d < 4; d++) {
    gParam_ex.r[d] = (comm_dim_partitioned(d)) ? 2 : 0;
    gParam_ex.x[d] = gParam.x[d] + 2 * gParam_ex.r[d];
  } // set halo region
  cudaForce_ex = new cudaGaugeField(gParam_ex);
  cudaOprod_ex = new cudaGaugeField(gParam_ex);
  cudaLongLinkOprod_ex = new cudaGaugeField(gParam_ex);

  // create a device force for verify
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.link_type = QUDA_GENERAL_LINKS;
  gParam.setPrecision(prec, true);
  cudaForce = new cudaGaugeField(gParam);

  // create the device momentum field
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.create = QUDA_NULL_FIELD_CREATE;
  gParam.reconstruct = QUDA_RECONSTRUCT_10;
  gParam.link_type = QUDA_ASQTAD_MOM_LINKS;
  gParam.setPrecision(prec, true);
  cudaMom = new cudaGaugeField(gParam);

  /********************************************************************
   * Copy to and exchange gauge and outer product fields on the device *
   ********************************************************************/
  cpuGauge_ex->exchangeExtendedGhost(R, true);
  cudaGauge_ex->loadCPUField(*cpuGauge);
  cudaGauge_ex->exchangeExtendedGhost(cudaGauge_ex->R());

  cpuOprod_ex->exchangeExtendedGhost(R, true);
  cudaOprod_ex->loadCPUField(*cpuOprod);
  cudaOprod_ex->exchangeExtendedGhost(cudaOprod_ex->R());

  cpuLongLinkOprod_ex->exchangeExtendedGhost(R, true);
  cudaLongLinkOprod_ex->loadCPUField(*cpuLongLinkOprod);
  cudaLongLinkOprod_ex->exchangeExtendedGhost(cudaLongLinkOprod_ex->R());

  /**********************
   * Do a little cleanup *
   **********************/
  delete rng;
}

// one-time teardown at end of tests
static void hisq_force_teardown()
{
  delete cudaMom;
  delete cudaForce;
  delete cudaForce_ex;
  delete cudaGauge_ex;
  delete cudaOprod_ex;
  delete cudaLongLinkOprod_ex;

  delete cpuGauge;
  delete cpuForce;
  delete hostVerifyForce;
  delete cpuMom;
  delete refMom;
  delete cpuOprod;
  delete cpuLongLinkOprod;

  delete cpuGauge_ex;
  delete cpuForce_ex;
  delete cpuOprod_ex;
  delete cpuLongLinkOprod_ex;
}

static int hisq_force_test(bool lepage)
{
  // float d_weight = 1.0;
  // { one, naik, three, five, seven, lepage }
  // double d_act_path_coeff[6] = { 1., 0., 0., 0., 0., 0. };
  double d_act_path_coeff[6] = {0.625000, -0.058479, -0.087719, 0.030778, -0.007200, lepage ? -0.123113 : 0.};

  quda::host_timer_t host_timer;

  double host_time_sec = 0.0;
  double staple_time_sec = 0.0;
  double long_time_sec = 0.0;
  double complete_time_sec = 0.0;

  /********************************
   * Zero momenta and force fields *
   ********************************/
  cpuForce->zero();
  cpuForce_ex->zero();
  cpuMom->zero();
  refMom->zero();

  cudaForce->zero();
  cudaForce_ex->zero();
  cudaMom->zero();

  /**************************************
   * Force contribution from the staples *
   **************************************/
  host_timer.start();
  fermion_force::hisqStaplesForce(*cudaForce_ex, *cudaOprod_ex, *cudaGauge_ex, d_act_path_coeff);
  qudaDeviceSynchronize();
  host_timer.stop();
  staple_time_sec = host_timer.last();

  if (verify_results) {
    host_timer.start();
    hisqStaplesForceCPU(d_act_path_coeff, *cpuOprod_ex, *cpuGauge_ex, cpuForce_ex);
    host_timer.stop();
    host_time_sec = host_timer.last();

    copyExtendedGauge(*cpuForce, *cpuForce_ex, QUDA_CPU_FIELD_LOCATION);
    copyExtendedGauge(*cudaForce, *cudaForce_ex, QUDA_CUDA_FIELD_LOCATION);
    cudaForce->saveCPUField(*hostVerifyForce);

    int res = 1;
    for (int dir = 0; dir < 4; dir++) {
      res &= compare_floats(reinterpret_cast<void **>(cpuForce->Gauge_p())[dir],
                            reinterpret_cast<void **>(hostVerifyForce->Gauge_p())[dir], V * gauge_site_size,
                            getTolerance(force_prec), force_prec);
    }

    strong_check_link(reinterpret_cast<void **>(hostVerifyForce->Gauge_p()),
                      "GPU results: ", reinterpret_cast<void **>(cpuForce->Gauge_p()), "CPU reference results:", V,
                      force_prec);
    logQuda(QUDA_SUMMARIZE, "Lepage %s staples force test %s\n\n", lepage ? "enabled" : "disabled",
            (1 == res) ? "PASSED" : "FAILED");
  }

  /*****************************************
   * Force contribution from the long links *
   ******************************************/

  // Only compute the long link when also computing the Lepage term
  // This is consistent with the chain rule for HISQ
  if (lepage && d_act_path_coeff[1] != 0.) {
    host_timer.start();
    fermion_force::hisqLongLinkForce(*cudaForce_ex, *cudaLongLinkOprod_ex, *cudaGauge_ex, d_act_path_coeff[1]);
    qudaDeviceSynchronize();
    host_timer.stop();
    long_time_sec = host_timer.last();

    if (verify_results) {
      host_timer.start();
      hisqLongLinkForceCPU(d_act_path_coeff[1], *cpuLongLinkOprod_ex, *cpuGauge_ex, cpuForce_ex);
      host_timer.stop();
      host_time_sec += host_timer.last();

      copyExtendedGauge(*cpuForce, *cpuForce_ex, QUDA_CPU_FIELD_LOCATION);
      copyExtendedGauge(*cudaForce, *cudaForce_ex, QUDA_CUDA_FIELD_LOCATION);
      cudaForce->saveCPUField(*hostVerifyForce);

      int res = 1;
      for (int dir = 0; dir < 4; dir++) {
        res &= compare_floats(reinterpret_cast<void **>(cpuForce->Gauge_p())[dir],
                              reinterpret_cast<void **>(hostVerifyForce->Gauge_p())[dir], V * gauge_site_size,
                              getTolerance(force_prec), force_prec);
      }

      strong_check_link(reinterpret_cast<void **>(hostVerifyForce->Gauge_p()),
                        "GPU results: ", reinterpret_cast<void **>(cpuForce->Gauge_p()), "CPU reference results:", V,
                        force_prec);
      logQuda(QUDA_SUMMARIZE, "Long link force test %s\n\n", (1 == res) ? "PASSED" : "FAILED");
    }
  }

  host_timer.start();
  fermion_force::hisqCompleteForce(*cudaForce_ex, *cudaGauge_ex);
  updateMomentum(*cudaMom, 1.0, *cudaForce_ex, __func__);
  qudaDeviceSynchronize();
  host_timer.stop();
  complete_time_sec = host_timer.last();

  if (verify_results) {
    host_timer.start();
    hisqCompleteForceCPU(*cpuForce_ex, *cpuGauge_ex, refMom);
    host_timer.stop();
    host_time_sec += host_timer.last();

    cudaMom->saveCPUField(*cpuMom);
  }

  int accuracy_level = 3;
  if (verify_results) {
    int res = compare_floats(cpuMom->Gauge_p(), refMom->Gauge_p(), 4 * cpuMom->Volume() * mom_site_size,
                             getTolerance(force_prec), force_prec);
    accuracy_level = strong_check_mom(cpuMom->Gauge_p(), refMom->Gauge_p(), 4 * cpuMom->Volume(), force_prec);
    logQuda(QUDA_SUMMARIZE, "Test (lepage coeff %e) %s\n", d_act_path_coeff[5], (1 == res) ? "PASSED" : "FAILED");
  }
  long long staple_io, staple_flops, long_io, long_flops, complete_io, complete_flops;
  total_staple_io_flops(force_prec, link_recon, lepage, staple_io, staple_flops, long_io, long_flops, complete_io,
                        complete_flops);

  {
    double perf_flops = static_cast<double>(staple_flops) / (staple_time_sec)*1e-9;
    double perf = static_cast<double>(staple_io) / (staple_time_sec)*1e-9;
    printfQuda("Staples time: %.6f ms, perf = %.6f GFLOPS, achieved bandwidth= %.6f GB/s, Lepage %c\n",
               staple_time_sec * 1e3, perf_flops, perf, lepage ? 'y' : 'n');
  }

  if (lepage) {
    double perf_flops = static_cast<double>(long_flops) / (long_time_sec)*1e-9;
    double perf = static_cast<double>(long_io) / (long_time_sec)*1e-9;
    printfQuda("LongLink time: %.6f ms, perf = %.6f GFLOPS, achieved bandwidth= %.6f GB/s\n", long_time_sec * 1e3,
               perf_flops, perf);
  }

  {
    double perf_flops = static_cast<double>(complete_flops) / (complete_time_sec)*1e-9;
    double perf = static_cast<double>(complete_io) / (complete_time_sec)*1e-9;
    printfQuda("Completion time: %.6f ms, perf = %.6f GFLOPS, achieved bandwidth= %.6f GB/s\n", complete_time_sec * 1e3,
               perf_flops, perf);
  }

  printfQuda("Host time : %g ms, Lepage %c\n", host_time_sec * 1e3, lepage ? 'y' : 'n');

  return accuracy_level;
}

static void display_test_info()
{
  printfQuda("running the following fermion force computation test:\n");

  printfQuda(
    "force_precision           link_reconstruct           space_dim(x/y/z)         T_dimension       Gauge_order\n");
  printfQuda("%s                       %s                         %d/%d/%d                  %d                %s\n",
             get_prec_str(force_prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim,
             get_gauge_order_str(gauge_order));
}

TEST(paths, verify)
{
  int level = hisq_force_test(true);
  // prevent tests from failing when verify is set to false
  if (verify_results) {
    int tolerance = getNegLog10Tolerance(force_prec);
    ASSERT_GE(level, tolerance) << "CPU and GPU implementations do not agree";
  }
}

TEST(paths_no_lepage, verify)
{
  int level = hisq_force_test(false);
  // prevent tests from failing when verify is set to false
  if (verify_results) {
    int tolerance = getNegLog10Tolerance(force_prec);
    ASSERT_GE(level, tolerance) << "CPU and GPU implementations do not agree";
  }
}

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  auto app = make_app();

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);
  initRand();
  initQuda(device_ordinal);

  setPrecision(prec);

  display_test_info();

  if (prec != QUDA_DOUBLE_PRECISION && prec != QUDA_SINGLE_PRECISION) errorQuda("Invalid precision %d", prec);
  // FIXME: debugging recon 12
  if (link_recon != QUDA_RECONSTRUCT_NO && link_recon != QUDA_RECONSTRUCT_13 /* && link_recon != QUDA_RECONSTRUCT_12*/)
    errorQuda("Invalid reconstruct %d", link_recon);

  // one-time setup
  hisq_force_startup();

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  int test_rc = RUN_ALL_TESTS();
  if (test_rc != 0) warningQuda("Tests failed");

  hisq_force_teardown();

  endQuda();
  finalizeComms();

  return test_rc;
}
