#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "TMCloverForce_reference.h"
#include "misc.h"
#include <color_spinor_field.h> // convenient quark field container
#include <command_line_params.h>
#include <gauge_field.h>
#include <host_utils.h>
#include <quda.h>
// #include "gauge_force_reference.h"
// #include <gauge_path_quda.h>
#include <gtest/gtest.h>
#include <timer.h>

static QudaGaugeFieldOrder gauge_order = QUDA_QDP_GAUGE_ORDER;

static int force_check;
static int path_check;
static double force_deviation;
// static double loop_deviation;
// static double plaq_deviation;
QudaInvertParam inv_param;
std::vector<char> gauge_;
std::array<void *, 4> gauge;
std::vector<char> clover;
std::vector<char> clover_inv;

// same as invert_test.cpp
void init(int argc, char **argv)
{
  // Set QUDA's internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam eig_param = newQudaEigParam();
  QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];

  if (inv_multigrid) {
    setQudaMgSolveTypes();
    setMultigridInvertParam(inv_param);
    // Set sub structures
    mg_param.invert_param = &mg_inv_param;
    for (int i = 0; i < mg_levels; i++) {
      if (mg_eig[i]) {
        // mg_eig_param[i] = newQudaEigParam();
        setMultigridEigParam(mg_eig_param[i], i);
        mg_param.eig_param[i] = &mg_eig_param[i];
      } else {
        mg_param.eig_param[i] = nullptr;
      }
    }
    // Set MG
    setMultigridParam(mg_param);
  } else {
    setInvertParam(inv_param);
  }
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;

  if (inv_deflate) {
    setEigParam(eig_param);
    inv_param.eig_param = &eig_param;
  } else {
    inv_param.eig_param = nullptr;
  }

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  gauge_.resize(4 * V * gauge_site_size * host_gauge_data_type_size);
  for (int i = 0; i < 4; i++) gauge[i] = gauge_.data() + i * V * gauge_site_size * host_gauge_data_type_size;

  printfQuda("Randomizing gauge fields... ");
  constructHostGaugeField(gauge.data(), gauge_param, argc, argv);

  printfQuda("Sending gauge field to GPU\n");
  loadGaugeQuda(gauge.data(), &gauge_param);

  // Allocate host side memory for clover terms if needed.
  //----------------------------------------------------------------------------
  // Allocate space on the host (always best to allocate and free in the same scope)
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover.resize(V * clover_site_size * host_clover_data_type_size);
    clover_inv.resize(V * clover_site_size * host_spinor_data_type_size);
    // constructHostCloverField(clover.data(), clover_inv.data(), inv_param);
    if (compute_clover)
        printfQuda("Computing clover field on GPU\n");
      else {
        printfQuda("Sending clover field to GPU\n");
        constructHostCloverField(clover.data(), clover_inv.data(), inv_param);
      }
      inv_param.compute_clover = compute_clover;
      inv_param.return_clover = compute_clover;
      inv_param.compute_clover_inverse = true;
      inv_param.return_clover_inverse = true;
      inv_param.num_offset=1;
      inv_param.true_res_offset[0]=0; // not use but printed. So we initialize it to a meaningless value
      inv_param.iter_res_offset[0]=0; // same as above
      inv_param.action[0]=0;
      inv_param.action[1]=0;
    // Load the clover terms to the device
    loadCloverQuda(clover.data(), clover_inv.data(), &inv_param);
  } else {
    errorQuda("dslash type ( dslash_type = %d ) must have the clover", dslash_type);
  }
}
// The same function is used to test computePath.
// If compute_force is false then a path is computed
void TMCloverForce_test()
{
  QudaGaugeParam gauge_param = newQudaGaugeParam();

  setGaugeParam(gauge_param);
  // setWilsonGaugeParam(gauge_param);

  gauge_param.gauge_order = gauge_order;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  setDims(gauge_param.X);

  quda::GaugeFieldParam param(gauge_param);
  param.create = QUDA_NULL_FIELD_CREATE;
  param.order = QUDA_QDP_GAUGE_ORDER;
  param.location = QUDA_CPU_FIELD_LOCATION;
  // quda::cpuGaugeField U_qdp(param);

  // fills the gauge field with random numbers
  // createSiteLinkCPU((void **)U_qdp.Gauge_p(), gauge_param.cpu_prec, 0);

  param.order = QUDA_MILC_GAUGE_ORDER;
  // quda::cpuGaugeField U_milc(param);
  // if (gauge_order == QUDA_MILC_GAUGE_ORDER) U_milc.copy(U_qdp);
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  // param.reconstruct = QUDA_RECONSTRUCT_NO;

  param.create = QUDA_ZERO_FIELD_CREATE;
  quda::GaugeField Mom_milc(param);
  quda::GaugeField Mom_ref_milc(param);

  param.order = QUDA_QDP_GAUGE_ORDER;
  quda::GaugeField Mom_qdp(param);

  // initialize some data in cpuMom
  // we need to set the mom to zero because computeTMCloverForceQuda is overwriting the momentum
  createMomCPU(Mom_ref_milc.data(), gauge_param.cpu_prec, 0.0);
  if (gauge_order == QUDA_MILC_GAUGE_ORDER) Mom_milc.copy(Mom_ref_milc);
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) Mom_qdp.copy(Mom_ref_milc);

  void *mom = nullptr;
  void *mom_array[QUDA_MAX_DIM];

  if (gauge_order == QUDA_MILC_GAUGE_ORDER) {
    mom = Mom_milc.data();
  } else if (gauge_order == QUDA_QDP_GAUGE_ORDER) {
    for (int d = 0; d < 4; d++) mom_array[d] = Mom_qdp.data(d);
    mom = reinterpret_cast<void *>(mom_array);
  } else {
    errorQuda("Unsupported gauge order %d", gauge_order);
  }

  // inv_param = newQudaInvertParam();
  // setInvertParam(inv_param);

  // inv_param.compute_clover = 1;
  // inv_param.compute_clover_inverse = 1;
  // inv_param.clover_csw=0;
  // inv_param.kappa=1;// we can not set to zero
  // inv_param.mu=0;
  // std::vector<quda::ColorSpinorField> in(Nsrc);
  std::vector<quda::ColorSpinorField> out_nvector(nvector * Nsrc);
  std::vector<std::vector<void *>> in(Nsrc, std::vector<void *>(nvector));
  std::vector<quda::ColorSpinorField> out_nvector0(nvector * Nsrc);
  std::vector<std::vector<void *>> in0(Nsrc, std::vector<void *>(nvector));

  quda::ColorSpinorField check;
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  check = quda::ColorSpinorField(cs_param);

  quda::RNG rng(check, 1234);

  inv_param.num_offset = nvector;
  for (int i = 0; i < nvector; i++) {
    // Set masses and offsets
    // masses[i] = 0.06 + i * i * 0.01;
    // inv_param.offset[i] = 4 * masses[i] * masses[i];
    // Set tolerances for the heavy quarks, these can be set independently
    // (functions of i) if desired
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
    // Allocate memory and set pointers
    for (int n = 0; n < Nsrc; n++) {
      out_nvector[n * nvector + i] = quda::ColorSpinorField(cs_param);
      spinorNoise(out_nvector[n * nvector + i], rng, QUDA_NOISE_GAUSS);
      in[n][i] = out_nvector[n * nvector + i].data();

      out_nvector0[n * nvector + i] = quda::ColorSpinorField(cs_param);
      spinorNoise(out_nvector0[n * nvector + i], rng, QUDA_NOISE_GAUSS);
      in0[n][i] = out_nvector0[n * nvector + i].data();

    }
  }

  double *coeff=(double*) malloc(sizeof(double)*nvector);
  for(int i=0;i<nvector;i++){
    coeff[i]=4. * inv_param.kappa * inv_param.kappa;
    coeff[i] += coeff[i]* (i+1)/10.0;
  }
  if (getTuning() == QUDA_TUNE_YES)
    computeTMCloverForceQuda(mom, in[0].data(), in0[0].data(), coeff, nvector,  &gauge_param, &inv_param, detratio);

  printfQuda("Device function computed\n");
  quda::host_timer_t host_timer;
  double time_sec = 0.0;
  // Multiple execution to exclude warmup time in the first run

  auto &Mom_ = gauge_order == QUDA_MILC_GAUGE_ORDER ? Mom_milc : Mom_qdp;
  for (int i = 0; i < niter; i++) {
    Mom_.copy(Mom_ref_milc); // restore initial momentum for correctness
    host_timer.start();
    computeTMCloverForceQuda(mom, in[0].data(), in0[0].data(), coeff, nvector,  &gauge_param, &inv_param, detratio);

    host_timer.stop();
    time_sec += host_timer.last();
  }
  if (gauge_order == QUDA_QDP_GAUGE_ORDER) Mom_milc.copy(Mom_qdp);

  // The number comes from CPU implementation in MILC, gauge_force_imp.c
  int flops = 153004;

  void *refmom = Mom_ref_milc.data();
  // int *check_out = compute_force ? &force_check : &path_check;
  int *check_out = true ? &force_check : &path_check;
  if (verify_results) {
    
    TMCloverForce_reference(refmom, in[0].data(), in0[0].data(), coeff, nvector, gauge, clover, clover_inv, &gauge_param, &inv_param, detratio);
    *check_out
      = compare_floats(Mom_milc.data(), refmom, 4 * V * mom_site_size, getTolerance(cuda_prec), gauge_param.cpu_prec);
    // if (compute_force)
    strong_check_mom(Mom_milc.data(), refmom, 4 * V, gauge_param.cpu_prec);
  }

  logQuda(QUDA_VERBOSE, "\nComputing momentum action\n");
  auto action_quda = momActionQuda(mom, &gauge_param);
  auto action_ref = mom_action(refmom, gauge_param.cpu_prec, 4 * V);
  force_deviation = std::abs(action_quda - action_ref) / std::abs(action_ref);
  logQuda(QUDA_VERBOSE, "QUDA action = %e, reference = %e relative deviation = %e\n", action_quda, action_ref,
          force_deviation);
  printfQuda("QUDA action = %e, reference = %e relative deviation = %e\n", action_quda, action_ref, force_deviation);

  double perf = 1.0 * niter * flops * V / (time_sec * 1e+9);
  // if (compute_force) {
  printfQuda("Force calculation total time = %.2f ms ; overall performance : %.2f GFLOPS\n", time_sec * 1e+3, perf);
  free(coeff);
}

TEST(force, verify) { ASSERT_EQ(force_check, 1) << "CPU and QUDA force implementations do not agree"; }

TEST(action, verify)
{
  ASSERT_LE(force_deviation, getTolerance(cuda_prec)) << "CPU and QUDA momentum action implementations do not agree";
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
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
  printfQuda("nvector: %d\n",nvector);
  printfQuda("detratio: %d\n",detratio);
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
  add_clover_force_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  initQuda(device_ordinal);
  init(argc, argv);

  display_test_info();

  TMCloverForce_test();

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
