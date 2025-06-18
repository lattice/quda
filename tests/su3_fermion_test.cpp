// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <util_quda.h>
#include <instantiate.h>

#include "host_utils.h"
#include "gauge_utils.h"
#include "command_line_params.h"
#include "dslash_reference.h"
#include "misc.h"
#include "test.h"

using test_t = ::testing::tuple<QudaPrecision, QudaGaugeSmearType>;

class SU3FermionTest : public ::testing::TestWithParam<test_t>
{
protected:
  QudaPrecision precision;
  QudaGaugeSmearType smear_type;

public:
  SU3FermionTest() : precision(::testing::get<0>(GetParam())), smear_type(::testing::get<1>(GetParam())) { }
};

int argc_copy;
char **argv_copy;

void run(test_t param)
{
  prec = ::testing::get<0>(param);
  gauge_smear_type = ::testing::get<1>(param);

  using namespace quda;

  if (!is_enabled_spin(4)) errorQuda("Test requires Wilson-type fermion enablement");

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  setWilsonGaugeParam(gauge_param);
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  void *gauge[4], *new_gauge[4];
  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    new_gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  }

  constructHostGaugeField(gauge, gauge_param, argc_copy, argv_copy);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);
  saveGaugeQuda(new_gauge, &gauge_param);
  // start the timer
  quda::host_timer_t host_timer, host_safe_timer, host_hier_timer, host_fwd_timer;

  // The commented out section is all geared towards gauge observables, so unlikely to be needed for now
  // // Prepare various perf info
  // long long flops_plaquette = 6ll * 597 * V;
  // long long flops_ploop = 198ll * V + 6 * V / gauge_param.X[3];

  // // Prepare a gauge observable struct
  // QudaGaugeObservableParam param = newQudaGaugeObservableParam();

  // The user may specify which measurements they wish to perform/omit
  // using the QudaGaugeObservableParam struct, and whether or not to
  // perform suN projection at each measurement step. We recommend that
  // users perform suN projection.
  // A unique observable param struct is constructed for each measurement.

  // Gauge Smearing Routines
  //---------------------------------------------------------------------------
  // Stout smearing should be equivalent to APE smearing
  // on D dimensional lattices for rho = alpha/2*(D-1).
  // Typical values for
  // APE: alpha=0.6
  // Stout: rho=0.1
  // Over Improved Stout: rho=0.08, epsilon=-0.25
  //
  // Typically, the user will use smearing for Q charge data only, so
  // we hardcode to compute Q only and not the plaquette. Users may
  // of course set these as they wish.  SU(N) projection su_project=true is recommended.
  QudaGaugeObservableParam *obs_param = new QudaGaugeObservableParam[gauge_smear_steps / measurement_interval + 1];
  for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
    obs_param[i] = newQudaGaugeObservableParam();
    obs_param[i].compute_plaquette = QUDA_BOOLEAN_FALSE;
    obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
    obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  }
  QudaFermMeasurements ferm_meas = newQudaFermMeasurements();
  // We here set all the problem parameters for all possible smearing types.
  QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
  smear_param.smear_type = gauge_smear_type;
  smear_param.n_steps = gauge_smear_steps;
  smear_param.adj_n_save = gauge_n_save;
  smear_param.hier_threshold = hier_threshold;
  smear_param.meas_interval = measurement_interval;
  smear_param.alpha = gauge_smear_alpha;
  smear_param.rho = gauge_smear_rho;
  smear_param.epsilon = gauge_smear_epsilon;
  smear_param.alpha1 = gauge_smear_alpha1;
  smear_param.alpha2 = gauge_smear_alpha2;
  smear_param.alpha3 = gauge_smear_alpha3;
  smear_param.dir_ignore = gauge_smear_dir_ignore;

  QudaInvertParam invParam = newQudaInvertParam();
  invParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  invParam.cuda_prec = cuda_prec;
  invParam.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invParam.dirac_order = QUDA_DIRAC_ORDER;
  invParam.verbosity = verbosity;

  quda::ColorSpinorParam cs_param;

  constructWilsonTestSpinorParam(&cs_param, &invParam, &gauge_param);

  std::vector<quda::ColorSpinorField> check_safe(Nsrc, cs_param), check_hier(Nsrc, cs_param), check_fwd(Nsrc, cs_param),
    check_o(Nsrc, cs_param);

  std::vector<void *> check_safe_arr, check_hier_arr, check_fwd_arr, check_arr;
  for (int i = 0; i < Nsrc; i++) {
    spinorNoise(check_o[i], i, QUDA_NOISE_GAUSS);
    check_arr.push_back(check_o[i].data());
    check_safe_arr.push_back(check_safe[i].data());
    check_hier_arr.push_back(check_hier[i].data());
    check_fwd_arr.push_back(check_fwd[i].data());
  }

  printf("Inspecting the very first element of fermion #1 we will use:\n");
  check_o[0].PrintVector(0, 0, 0);
  printf("Inspecting the very first element of set #1 the 3 un-evolved fermions (should be zero):\n");
  printf("Hierarchical method:\n");
  check_hier[0].PrintVector(0, 0, 0);
  printf("Safe method:\n");
  check_safe[0].PrintVector(0, 0, 0);
  printf("Forward method:\n");
  check_fwd[0].PrintVector(0, 0, 0);

  host_timer.start(); // start the timer
  switch (smear_param.smear_type) {
  case QUDA_GAUGE_SMEAR_APE:
  case QUDA_GAUGE_SMEAR_STOUT:
  case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
  case QUDA_GAUGE_SMEAR_HYP: {
    performGaugeSmearQuda(&smear_param, obs_param);
    break;
  }

    // Here we use a typical use case which is different from simple smearing in that
    // the user will want to compute the plaquette values to compute the gauge energy.
  case QUDA_GAUGE_SMEAR_WILSON_FLOW:
  case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: {
    for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
      obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
    }

    // Perform two adjoint flow algorithms, these methods dont alter the final value for the gauge so we excecute them first
    host_hier_timer.start();
    performAdjGFlowHier(check_hier_arr.data(), check_arr.data(), &invParam, &smear_param, &ferm_meas, Nsrc);
    host_hier_timer.stop();
    host_safe_timer.start();
    performAdjGFlowSafe(check_safe_arr.data(), check_arr.data(), &invParam, &smear_param, Nsrc);
    host_safe_timer.stop();
    // Perform forward flow algorithm
    host_fwd_timer.start();
    performGFlowQuda(check_fwd_arr.data(), check_arr.data(), &invParam, &smear_param, obs_param, Nsrc);
    host_fwd_timer.stop();

    printfQuda("Time elapsed for adjoint hierarchical fermion/gauge smearing = %g secs\n", host_hier_timer.last());
    printfQuda("Time elapsed for adjoint safe fermion/gauge smearing = %g secs\n", host_safe_timer.last());
    printfQuda("Time elapsed for forward fermion/gauge smearing = %g secs\n", host_fwd_timer.last());

    break;
  }
  default: errorQuda("Undefined gauge smear type %d given", smear_param.smear_type);
  }

  host_timer.stop(); // stop the timer
  printfQuda("Total time for collective fermion/gauge smearing = %g secs\n", host_timer.last());

  for (int j = 0; j < Nsrc; j++) {
    printfQuda("\n\nCorrectness test for fermion #%i out of %i\n", j + 1, Nsrc);
    printf("Now, inspecting the very first element of the 3 evolved fermions:\n");
    printf("Hierarchical method:\n");
    check_hier[j].PrintVector(0, 0, 0);
    printf("Safe method:\n");
    check_safe[j].PrintVector(0, 0, 0);
    printf("Forward method:\n");
    check_fwd[j].PrintVector(0, 0, 0);

    double method_adj_diff = 0.;
    /* To access the ith complex entry in a raw vector, do, for example: check.data<std::complex<double>*>()[i]*/
    for (int i = 0; i < V * 24; i++) {
      method_adj_diff += pow(fabs(check_safe[j].data<double *>()[i] - check_hier[j].data<double *>()[i]), 2);
    }
    double method_adj_check = sqrt(method_adj_diff) / (V * 24.);
    printf(
      "Mean of mag errors between Safe and Hierarchical Adj methods (should be zero up to machine precision) = %1.5e\n",
      method_adj_check);

    std::complex<double> trace_fwd, trace_adj;
    trace_fwd
      = twoColorSpinorContract(check_o[j].data<std::complex<double> *>(), check_fwd[j].data<std::complex<double> *>());
    trace_adj
      = twoColorSpinorContract(check_o[j].data<std::complex<double> *>(), check_safe[j].data<std::complex<double> *>());

    auto trace_diff_err = 2. * std::fabs(trace_fwd - std::conj(trace_adj)) / std::fabs(trace_fwd + std::conj(trace_adj));

    printf("The two numbers below should be complex conjugates of one another\n");
    printf("<check,adj_check> is %1.5e, %1.5e \n", trace_adj.real(), trace_adj.imag());
    printf("<check,fwd_check> is %1.5e, %1.5e \n", trace_fwd.real(), trace_fwd.imag());
    printf("Fractional error of (<check,adj_check> - <check,fwd_check>.conj()) = %1.5e \n", trace_diff_err);

    auto eps = getTolerance(prec);

    printfQuda("Checking adjoint safe/hier match\n");
    EXPECT_LE(method_adj_check, gauge_smear_steps * gauge_smear_steps * eps);
    if (method_adj_check > gauge_smear_steps * gauge_smear_steps * eps)
      warningQuda("Adjoint safe/hier difference %e greater than tolerance %e\n", method_adj_check,
                  gauge_smear_steps * gauge_smear_steps * eps);

    printfQuda("Checking fractional error\n");
    EXPECT_LE(trace_diff_err, gauge_smear_steps * gauge_smear_steps * eps);
    if (trace_diff_err > gauge_smear_steps * gauge_smear_steps * eps)
      warningQuda("Fractional error %e greater than tolerance %e\n", trace_diff_err,
                  gauge_smear_steps * gauge_smear_steps * eps);
  }

  if (verify_results) check_gauge(gauge, new_gauge, 1e-3, gauge_param.cpu_prec);

  for (int dir = 0; dir < 4; dir++) {
    host_free(gauge[dir]);
    host_free(new_gauge[dir]);
  }

  check_o = {};
  check_hier = {};
  check_safe = {};
  check_fwd = {};

  freeGaugeQuda();
}

TEST_P(SU3FermionTest, verify)
{
  if (!quda::is_enabled(precision)) GTEST_SKIP();
  run(GetParam());
}

struct su3_fermion_test : quda_test {

  void display_info() const override
  {
    quda_test::display_info();
    printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
    printfQuda("%s   %s             %s            %s            %d/%d/%d          %d\n", get_prec_str(prec),
               get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
               tdim);

    // Specific test
    printfQuda("\n%s smearing\n", get_gauge_smear_str(gauge_smear_type));
    switch (gauge_smear_type) {
    case QUDA_GAUGE_SMEAR_APE: printfQuda(" - alpha %f\n", gauge_smear_alpha); break;
    case QUDA_GAUGE_SMEAR_STOUT: printfQuda(" - rho %f\n", gauge_smear_rho); break;
    case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
      printfQuda(" - rho %f\n", gauge_smear_rho);
      printfQuda(" - epsilon %f\n", gauge_smear_epsilon);
      break;
    case QUDA_GAUGE_SMEAR_HYP:
      printfQuda(" - alpha1 %f\n", gauge_smear_alpha1);
      printfQuda(" - alpha2 %f\n", gauge_smear_alpha2);
      printfQuda(" - alpha3 %f\n", gauge_smear_alpha3);
      break;
    case QUDA_GAUGE_SMEAR_WILSON_FLOW:
    case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: printfQuda(" - epsilon %f\n", gauge_smear_epsilon); break;
    default: errorQuda("Undefined test type %d given", test_type);
    }
    printfQuda(" - smearing steps %d\n", gauge_smear_steps);
    printfQuda(" - smearing ignore direction %d\n", gauge_smear_dir_ignore);
    printfQuda(" - Measurement interval %d\n", measurement_interval);
  }

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);
    add_su3_option_group(app);
  }

  su3_fermion_test(int argc, char **argv) : quda_test("SU3 Fermion Test", argc, argv) { }
};

auto test_str = [](testing::TestParamInfo<test_t> param) {
  return std::string(get_prec_str(::testing::get<0>(param.param))) + "_"
    + get_gauge_smear_str(::testing::get<1>(param.param));
};

INSTANTIATE_TEST_SUITE_P(Flow, SU3FermionTest,
                         ::testing::Combine(::testing::Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION),
                                            ::testing::Values(QUDA_GAUGE_SMEAR_WILSON_FLOW,
                                                              QUDA_GAUGE_SMEAR_SYMANZIK_FLOW)),
                         test_str);

int main(int argc, char **argv)
{
  argc_copy = argc;
  argv_copy = argv;

  su3_fermion_test test(argc, argv);
  test.init();
  if (enable_testing) {
    return test.execute();
  } else {
    if (gauge_smear_type != QUDA_GAUGE_SMEAR_WILSON_FLOW && gauge_smear_type != QUDA_GAUGE_SMEAR_SYMANZIK_FLOW) {
      warningQuda("Smear type %s not supported, setting to Wilson Flow", get_gauge_smear_str(gauge_smear_type));
      gauge_smear_type = QUDA_GAUGE_SMEAR_WILSON_FLOW;
    }
    run(test_t {prec, gauge_smear_type});
  };
}
