#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clover_force_reference.h"
#include "misc.h"
#include <color_spinor_field.h> // convenient quark field container
#include <command_line_params.h>
#include <gauge_field.h>
#include <host_utils.h>
#include <quda.h>
#include <gtest/gtest.h>

static int force_check;
static int path_check;
static double force_deviation;
QudaGaugeParam gauge_param;
QudaInvertParam inv_param;
quda::GaugeField gauge;
quda::GaugeField mom;
quda::GaugeField mom_ref;
std::vector<char> clover;
std::vector<char> clover_inv;

void init(int argc, char **argv)
{
  // Set QUDA's internal parameters
  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  setDims(gauge_param.X);

  // Allocate host gauge field objects
  quda::GaugeFieldParam param(gauge_param, nullptr, QUDA_SU3_LINKS);
  param.create = QUDA_NULL_FIELD_CREATE;
  param.order = QUDA_QDP_GAUGE_ORDER;
  gauge = quda::GaugeField(param);

  printfQuda("Randomizing gauge fields... ");
  constructHostGaugeField(gauge, gauge_param, argc, argv);

  printfQuda("Sending gauge field to GPU\n");
  loadGaugeQuda(gauge.raw_pointer(), &gauge_param);

  param.order = QUDA_MILC_GAUGE_ORDER;
  param.link_type = QUDA_ASQTAD_MOM_LINKS;
  param.reconstruct = QUDA_RECONSTRUCT_10;
  param.create = QUDA_ZERO_FIELD_CREATE;
  mom = quda::GaugeField(param);
  mom_ref = quda::GaugeField(param);

  // Allocate host side memory for clover terms if needed.
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover.resize(V * clover_site_size * host_clover_data_type_size);
    clover_inv.resize(V * clover_site_size * host_spinor_data_type_size);
    compute_clover = true;
    constructHostCloverField(clover.data(), clover_inv.data(), inv_param);
    // Load the clover terms to the device
    loadCloverQuda(clover.data(), clover_inv.data(), &inv_param);
  } else {
    errorQuda("dslash type ( dslash_type = %d ) must have the clover", dslash_type);
  }
}

void destroy()
{
  gauge = {};
  mom = {};
  mom_ref = {};
}

using test_t = ::testing::tuple<bool, int>;

std::tuple<int, double> clover_force_test(test_t param)
{
  bool detratio = ::testing::get<0>(param);
  int nvector = ::testing::get<1>(param);

  std::vector<quda::ColorSpinorField> out_nvector(nvector);
  std::vector<void *> in(nvector);
  std::vector<quda::ColorSpinorField> out_nvector0(nvector);
  std::vector<void *> in0(nvector);

  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);

  quda::RNG rng(mom, 1234);

  inv_param.num_offset = nvector;
  for (int i = 0; i < nvector; i++) {
    // Allocate memory and set pointers
    out_nvector[i] = quda::ColorSpinorField(cs_param);
    spinorNoise(out_nvector[i], rng, QUDA_NOISE_GAUSS);
    in[i] = out_nvector[i].data();

    out_nvector0[i] = quda::ColorSpinorField(cs_param);
    spinorNoise(out_nvector0[i], rng, QUDA_NOISE_GAUSS);
    in0[i] = out_nvector0[i].data();
  }

  std::vector<double> coeff(nvector);
  for(int i=0;i<nvector;i++){
    coeff[i] = 4. * inv_param.kappa * inv_param.kappa;
    coeff[i] += coeff[i] * (i + 1) / 10.0;
  }
  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  gauge_param.overwrite_mom = 1;
  if (getTuning() == QUDA_TUNE_YES)
    computeTMCloverForceQuda(mom.data(), in.data(), in0.data(), coeff.data(), nvector, &gauge_param, &inv_param,
                             detratio);

  // Multiple execution to exclude warmup time in the first run
  double time_sec = 0.0;
  double gflops = 0.0;
  for (int i = 0; i < niter; i++) {
    computeTMCloverForceQuda(mom.data(), in.data(), in0.data(), coeff.data(), nvector, &gauge_param, &inv_param,
                             detratio);
    time_sec += inv_param.secs;
    gflops += inv_param.gflops;
  }

  int *check_out = true ? &force_check : &path_check;
  std::array<void *, 4> u = {gauge.data(0), gauge.data(1), gauge.data(2), gauge.data(3)};
  if (verify_results) {
    gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
    mom_ref.zero();
    TMCloverForce_reference(mom_ref.data(), in.data(), in0.data(), coeff.data(), nvector, u, clover, clover_inv,
                            &gauge_param, &inv_param, detratio);
    *check_out
      = compare_floats(mom.data(), mom_ref.data(), 4 * V * mom_site_size, getTolerance(cuda_prec), gauge_param.cpu_prec);
    // if (compute_force)
    strong_check_mom(mom.data(), mom_ref.data(), 4 * V, gauge_param.cpu_prec);
  }

  logQuda(QUDA_VERBOSE, "\nComputing momentum action\n");
  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  auto action_quda = momActionQuda(mom.data(), &gauge_param);
  auto action_ref = mom_action(mom_ref.data(), gauge_param.cpu_prec, 4 * V);
  force_deviation = std::abs(action_quda - action_ref) / std::abs(action_ref);
  logQuda(QUDA_VERBOSE, "QUDA action = %e, reference = %e relative deviation = %e\n", action_quda, action_ref,
          force_deviation);
  printfQuda("QUDA action = %e, reference = %e relative deviation = %e\n", action_quda, action_ref, force_deviation);
  printfQuda("Force calculation total time = %.2f ms ; overall performance : %.2f GFLOPS\n", time_sec * 1e+3,
             gflops / time_sec);

  return {force_check, force_deviation};
}

class CloverForceTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  CloverForceTest() : param(GetParam()) { }
};

TEST_P(CloverForceTest, verify)
{
  auto deviation = clover_force_test(GetParam());
  ASSERT_EQ(std::get<0>(deviation), 1) << "CPU and QUDA force implementations do not agree";
  ASSERT_LE(std::get<1>(deviation), getTolerance(cuda_prec)) << "CPU and QUDA momentum action implementations do not agree";
}

static void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("link_precision           link_reconstruct           space_dim(x/y/z)              T_dimension        "
             "Gauge_order    niter\n");
  printfQuda("%s                       %s                         %d/%d/%d                       %d                  "
             "%s           %d\n",
             get_prec_str(prec), get_recon_str(link_recon), xdim, ydim, zdim, tdim,
             get_gauge_order_str(QUDA_MILC_GAUGE_ORDER), niter);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
               dimPartitioned(3));
}

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string name;
  if (::testing::get<0>(param.param)) name += std::string("ratio_");
  name += std::string("nvector_") + std::to_string(::testing::get<1>(param.param));
  return name;
}

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;

  // command line options
  auto app = make_app();
  add_clover_force_option_group(app);
  add_testing_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);
  initQuda(device_ordinal);
  init(argc, argv);

  display_test_info();

  if (enable_testing) {
    // Ensure gtest prints only from rank 0
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

    test_rc = RUN_ALL_TESTS();
  } else {
    clover_force_test({detratio, Nsrc});
  }

  destroy();
  endQuda();
  finalizeComms();
  return test_rc;
}

INSTANTIATE_TEST_SUITE_P(CloverForceTest, CloverForceTest,
                         ::testing::Combine(::testing::Values(false, true), ::testing::Values(1, 8)), gettestname);
