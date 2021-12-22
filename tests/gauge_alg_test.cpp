#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <gauge_field.h>

#include <comm_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <misc.h>
#include <timer.h>
#include <gauge_tools.h>
#include <tune_quda.h>

#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

#include <qio_field.h>

#include <gtest/gtest.h>

using namespace quda;

//***********************************************************//
// This boolean controls whether or not the full Google test //
// is done. If the user passes a value of 1 or 2 to --test   //
// then a single instance of OVR or FFT gauge fixing is done //
// and the value of this bool is set to false. Otherwise the //
// Google tests are performed.                               //
//***********************************************************//
bool execute = true;

// Gauge IO related
bool gauge_load;
bool gauge_store;
void *host_gauge[4];

// Define the command line options and option group for this test
int gf_gauge_dir = 4;
int gf_maxiter = 10000;
int gf_verbosity_interval = 100;
double gf_ovr_relaxation_boost = 1.5;
double gf_fft_alpha = 0.8;
bool gf_fft_autotune = true;
int gf_reunit_interval = 10;
double gf_tolerance = 1e-6;
bool gf_theta_condition = false;
QudaGaugeFixType gf_fix_type = QUDA_GAUGEFIX_TYPE_OVR;

void display_test_info()
{
  printfQuda("running the following test:\n");

  switch (test_type) {
  case 0: printfQuda("\n Google testing\n"); break;
  case 1:
    printfQuda("\n%s %s gauge fix\n", get_gaugefix_str(gf_fix_type), gf_gauge_dir == 4 ? "Landau" : "Coulomb");
    break;
  default: errorQuda("Undefined test type %d given", test_type);
  }

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

void add_gaugefix_option_group(std::shared_ptr<QUDAApp> quda_app)
{
  CLI::TransformPairs<QudaGaugeFixType> fix_type_map {{"ovr", QUDA_GAUGEFIX_TYPE_OVR}, {"fft", QUDA_GAUGEFIX_TYPE_FFT}};

  // Option group for gauge fixing related options
  auto opgroup = quda_app->add_option_group("gaugefix", "Options controlling gauge fixing tests");
  opgroup->add_option("--gf-dir", gf_gauge_dir,
                      "The orthogonal direction of the gauge fixing, 3=Coulomb, 4=Landau. (default 4)");
  opgroup->add_option("--gf-maxiter", gf_maxiter,
                      "The maximun number of gauge fixing iterations to be applied (default 10000)");
  opgroup->add_option("--gf-verbosity-interval", gf_verbosity_interval,
                      "Print the gauge fixing progress every N steps (default 100)");
  opgroup->add_option("--gf-ovr-relaxation-boost", gf_ovr_relaxation_boost,
                      "The overrelaxation boost parameter for the overrelaxation method (default 1.5)");
  opgroup->add_option("--gf-fft-alpha", gf_fft_alpha, "The Alpha parameter in the FFT method (default 0.8)");
  opgroup->add_option("--gf-fft-autotune", gf_fft_autotune,
                      "Autotune the Alpha parameter in the FFT method (default true)");
  opgroup->add_option("--gf-reunit-interval", gf_reunit_interval,
                      "Reunitarise the gauge field every N steps (default 10)");
  opgroup->add_option("--gf-tol", gf_tolerance, "The tolerance of the gauge fixing quality (default 1e-6)");
  opgroup->add_option(
    "--gf-theta-condition", gf_theta_condition,
    "Use the theta value to determine the gauge fixing if true. If false, use the delta value (default false)");
  opgroup->add_option("--gf-fix-type", gf_fix_type, "The type of algorithm to use for fixing (default ovr)")
    ->transform(CLI::QUDACheckedTransformer(fix_type_map));
}

void setGaugeFixParam(QudaGaugeFixParam &fix_param)
{
  fix_param.fix_type = gf_fix_type;
  fix_param.gauge_dir = gf_gauge_dir;
  fix_param.maxiter = gf_maxiter;
  fix_param.verbosity_interval = gf_verbosity_interval;
  fix_param.reunit_interval = gf_reunit_interval;
  fix_param.tolerance = gf_tolerance;
  fix_param.ovr_relaxation_boost = gf_ovr_relaxation_boost;
  fix_param.fft_alpha = gf_fft_alpha;
  fix_param.fft_autotune = gf_fft_alpha ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  fix_param.theta_condition = gf_theta_condition ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
  fix_param.precision = cuda_prec;
}

class GaugeAlgTest : public ::testing::Test
{

protected:
  QudaGaugeParam gauge_param;
  QudaGaugeFixParam fix_param;

  host_timer_t host_timer_1, host_timer_2;
  double2 det_u;
  double2 trace_u;
  double3 plaq_u;
  cudaGaugeField *U;
  int nsteps;
  int nhbsteps;
  int novrsteps;
  bool coldstart;
  double beta_value;
  RNG *randstates;

  void SetReunitarizationConsts()
  {
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    setUnitarizeLinksConstants(unitarize_eps, max_error, reunit_allow_svd, reunit_svd_only, svd_rel_error, svd_abs_error);
  }

  bool checkDimsPartitioned()
  {
    if (comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3))
      return true;
    return false;
  }

  bool comparePlaquette(double3 a, double3 b)
  {
    double a0, a1, a2;
    a0 = std::abs(a.x - b.x);
    a1 = std::abs(a.y - b.y);
    a2 = std::abs(a.z - b.z);
    double prec_val = 1.0e-5;
    if (prec == QUDA_DOUBLE_PRECISION) prec_val = 1e-10;
    return ((a0 < prec_val) && (a1 < prec_val) && (a2 < prec_val));
  }

  bool checkDeterminant(double2 det)
  {
    double prec_val = 1.0e-5;
    if (prec == QUDA_DOUBLE_PRECISION) prec_val = 1e-8;
    return (std::abs(1.0 - det.x) < prec_val && std::abs(det.y) < prec_val);
  }

  virtual void SetUp()
  {
    if (execute) {

      // Setup gauge container.
      gauge_param = newQudaGaugeParam();
      setWilsonGaugeParam(gauge_param);
      gauge_param.t_boundary = QUDA_PERIODIC_T;

      // Reunitarization setup
      int *num_failures_h = (int *)mapped_malloc(sizeof(int));
      int *num_failures_d = (int *)get_mapped_device_pointer(num_failures_h);
      SetReunitarizationConsts();

      host_timer_1.start();

      // If no field is loaded, create a physical quenched field on the device
      if (!gauge_load) {
        GaugeFieldParam device_gauge_param(gauge_param);
        device_gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
        device_gauge_param.create = QUDA_NULL_FIELD_CREATE;
        device_gauge_param.reconstruct = link_recon;
        device_gauge_param.setPrecision(cuda_prec, true);
        for (int d = 0; d < 4; d++) {
          if (comm_dim_partitioned(d)) device_gauge_param.r[d] = 2;
          device_gauge_param.x[d] += 2 * device_gauge_param.r[d];
        }

        U = new cudaGaugeField(device_gauge_param);

        RNG randstates(*U, quda_seed);

        nsteps = heatbath_num_steps;
        nhbsteps = heatbath_num_heatbath_per_step;
        novrsteps = heatbath_num_overrelax_per_step;
        coldstart = heatbath_coldstart;
        beta_value = heatbath_beta_value;
        host_timer_2.start();

        if (coldstart)
          InitGaugeField(*U);
        else
          InitGaugeField(*U, randstates);

        for (int step = 1; step <= nsteps; ++step) {
          printfQuda("Step %d\n", step);
          Monte(*U, randstates, beta_value, nhbsteps, novrsteps);

          // Reunitarization
          *num_failures_h = 0;
          unitarizeLinks(*U, num_failures_d);
          qudaDeviceSynchronize();
          if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)", *num_failures_h);
          plaq_u = plaquette(*U);
          printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq_u.x, plaq_u.y, plaq_u.z);
        }

        host_timer_2.stop();
        printfQuda("Time Monte -> %.6f s\n", host_timer_2.last());
      } else {

        // If a field is loaded, create a device field and copy
        printfQuda("Copying gauge field from host\n");
        gauge_param.location = QUDA_CPU_FIELD_LOCATION;
        GaugeFieldParam host_gauge_param(gauge_param, host_gauge);
        host_gauge_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
        GaugeField *host = GaugeField::Create(host_gauge_param);

        // switch the parameters for creating the mirror precise cuda gauge field
        host_gauge_param.create = QUDA_NULL_FIELD_CREATE;
        host_gauge_param.reconstruct = gauge_param.reconstruct;
        host_gauge_param.setPrecision(gauge_param.cuda_prec, true);

        if (comm_partitioned()) {
          int R[4] = {0, 0, 0, 0};
          for (int d = 0; d < 4; d++)
            if (comm_dim_partitioned(d)) R[d] = 2;
          static TimeProfile GaugeFix("GaugeFix");
          cudaGaugeField *tmp = new cudaGaugeField(host_gauge_param);
          tmp->copy(*host);
          U = createExtendedGauge(*tmp, R, GaugeFix);
          delete tmp;
        } else {
          U = new cudaGaugeField(host_gauge_param);
          U->copy(*host);
        }

        delete host;

        // Reunitarization
        *num_failures_h = 0;
        unitarizeLinks(*U, num_failures_d);
        qudaDeviceSynchronize();
        if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)", *num_failures_h);
      }

      // Unfixed Gauge data
      plaq_u = plaquette(*U);
      det_u = getLinkDeterminant(*U);
      trace_u = getLinkTrace(*U);
      printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq_u.x, plaq_u.y, plaq_u.z);
      printfQuda("Det: %.16e:%.16e\n", det_u.x, det_u.y);
      printfQuda("Tr: %.16e:%.16e\n", trace_u.x / 3.0, trace_u.y / 3.0);

      // If a specific test type is requested, perform it now and then
      // turn off all Google tests in the tear down.
      switch (test_type) {
      case 0: // Do the Google testing
        // Set gauge fixing params from the command line
        // and adjust for this test type
        fix_param = newQudaGaugeFixParam();
        setGaugeFixParam(fix_param);
        break;
      case 1: // Do a specific test
        run();
        break;
      default: errorQuda("Invalid test type %d ", test_type);
      }

      host_free(num_failures_h);
    }
  }

  virtual void TearDown()
  {
    if (execute) {

      // Compare gauge fixed data with original data
      auto plaq_gf = plaquette(*U);
      auto det_gf = getLinkDeterminant(*U);
      auto trace_gf = getLinkTrace(*U);
      printfQuda("Plaq:     %.16e, %.16e, %.16e\n", plaq_u.x, plaq_u.y, plaq_u.z);
      printfQuda("Plaq GF:  %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
      printfQuda("Det:      %.16e, %.16e\n", det_u.x, det_u.y);
      printfQuda("Det GF:   %.16e, %.16e\n", det_gf.x, det_gf.y);
      printfQuda("Trace:    %.16e, %.16e\n", trace_u.x / 3.0, trace_u.y / 3.0);
      printfQuda("Trace GF: %.16e, %.16e\n", trace_gf.x / 3.0, trace_gf.y / 3.0);

      // As an observable, the plaquette value must remain invariant after
      // gauge fixing.
      ASSERT_TRUE(comparePlaquette(plaq_u, plaq_gf));

      // The determinant of any SU(N) gauge field element must be (1.0,0.0) to
      // machine precision
      ASSERT_TRUE(checkDeterminant(det_gf));

      delete U;
      // Release all temporary memory used for data exchange between GPUs in multi-GPU mode
      PGaugeExchangeFree();

      host_timer_1.stop();
      printfQuda("Time -> %.6f s\n", host_timer_1.last());
    }
    // If we performed a specific instance, switch off the
    // Google testing.
    if (test_type != 0) execute = false;
    saveTuneCache();
  }

  virtual void run()
  {
    if (execute) {
      // Set gauge fixing params from the command line
      fix_param = newQudaGaugeFixParam();
      setGaugeFixParam(fix_param);

      printfQuda("%s gauge fixing with %s method\n", fix_param.gauge_dir == 4 ? "Landau" : "Coulomb",
                 get_gaugefix_str(fix_param.fix_type));

      // Setup CPU gauge container.
      gauge_param = newQudaGaugeParam();
      setWilsonGaugeParam(gauge_param);
      gauge_param.t_boundary = QUDA_PERIODIC_T;
      gauge_param.location = QUDA_CPU_FIELD_LOCATION;

      void *cpu_gauge[4];
      for (int dir = 0; dir < 4; dir++) { cpu_gauge[dir] = safe_malloc(V * gauge_site_size * cpu_prec); }

      GaugeFieldParam param(gauge_param);
      param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
      param.create = QUDA_NULL_FIELD_CREATE;
      param.link_type = gauge_param.type;
      param.reconstruct = gauge_param.reconstruct;
      param.setPrecision(cuda_prec, true);

      auto *gauge = new cudaGaugeField(param);

      // Copy the target U field (extended) into regular GPU field, then
      // save to a CPU field. This is done to test the CPU interface function
      // and instructs the user how to use void pointers for the gauge data,
      // and the gauge_param container.
      copyExtendedGauge(*gauge, *U, QUDA_CUDA_FIELD_LOCATION);
      saveGaugeFieldQuda((void *)cpu_gauge, (void *)gauge, &gauge_param);
      delete gauge;

      // Compute gauge fixing via interface
      computeGaugeFixingQuda(cpu_gauge, &gauge_param, &fix_param, nullptr);

      // cpu_gauge now contains the fixed gauge on the CPU. We now load that gauge
      // to the device for inspection in the TearDown.
      GaugeFieldParam fixed_param(gauge_param, cpu_gauge);
      auto *fixed_cpu_gauge = new cpuGaugeField(fixed_param);

      // Copy the CPU field to U.
      U->loadCPUField(*fixed_cpu_gauge);

      for (int dir = 0; dir < 4; dir++) host_free(cpu_gauge[dir]);
      delete fixed_cpu_gauge;

      // Save if output string is specified
      if (gauge_store) save_gauge();
      saveTuneCache();
    }
  }

  virtual void save_gauge()
  {
    printfQuda("Saving the gauge field to file %s\n", gauge_outfile);

    void *cpu_gauge[4];
    for (int dir = 0; dir < 4; dir++) { cpu_gauge[dir] = safe_malloc(V * gauge_site_size * gauge_param.cpu_prec); }

    GaugeFieldParam param(gauge_param);
    param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    param.create = QUDA_NULL_FIELD_CREATE;
    param.link_type = gauge_param.type;
    param.reconstruct = gauge_param.reconstruct;
    param.setPrecision(param.Precision(), true);

    cudaGaugeField *gauge;
    gauge = new cudaGaugeField(param);

    // copy into regular field
    copyExtendedGauge(*gauge, *U, QUDA_CUDA_FIELD_LOCATION);
    saveGaugeFieldQuda((void *)cpu_gauge, (void *)gauge, &gauge_param);

    // Write to disk
    write_gauge_field(gauge_outfile, cpu_gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char **)0);

    for (int dir = 0; dir < 4; dir++) host_free(cpu_gauge[dir]);
    delete gauge;
  }
};

TEST_F(GaugeAlgTest, Generation)
{
  if (execute && !gauge_load) {
    // Assert that the generated gauge is
    // on the SU(N) manifold
    det_u = getLinkDeterminant(*U);
    ASSERT_TRUE(checkDeterminant(det_u));
  }
}

TEST_F(GaugeAlgTest, Landau_Overrelaxation)
{
  if (execute) {
    printfQuda("Landau gauge fixing with overrelaxation method\n");

    fix_param.fix_type = QUDA_GAUGEFIX_TYPE_OVR;
    fix_param.gauge_dir = 4;

    gaugeFixingOVR(*U, fix_param);
    saveTuneCache();
  }
}

TEST_F(GaugeAlgTest, Coulomb_Overrelaxation)
{
  if (execute) {
    printfQuda("Coulomb gauge fixing with overrelaxation method\n");

    fix_param.fix_type = QUDA_GAUGEFIX_TYPE_OVR;
    fix_param.gauge_dir = 3;

    gaugeFixingOVR(*U, fix_param);
    saveTuneCache();
  }
}

TEST_F(GaugeAlgTest, Landau_FFT)
{
  if (execute) {
    if (!comm_partitioned()) {
      printfQuda("Landau gauge fixing with steepest descent method with FFT\n");

      fix_param.fix_type = QUDA_GAUGEFIX_TYPE_FFT;
      fix_param.gauge_dir = 4;

      gaugeFixingFFT(*U, fix_param);
      saveTuneCache();
    }
  }
}

TEST_F(GaugeAlgTest, Coulomb_FFT)
{
  if (execute) {
    if (!comm_partitioned()) {
      printfQuda("Coulomb gauge fixing with steepest descent method with FFT\n");

      fix_param.fix_type = QUDA_GAUGEFIX_TYPE_FFT;
      fix_param.gauge_dir = 3;

      gaugeFixingFFT(*U, fix_param);
      saveTuneCache();
    }
  }
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  // command line options
  auto app = make_app();
  add_gaugefix_option_group(app);
  add_heatbath_option_group(app);

  test_type = 0;
  CLI::TransformPairs<int> test_type_map {{"Google", 0}, {"OVR", 1}, {"FFT", 2}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setVerbosity(verbosity);
  setQudaPrecisions();
  setWilsonGaugeParam(gauge_param);
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  setDims(gauge_param.X);

  // call srand() with a rank-dependent seed
  initRand();
  // initialize the QUDA library
  initQuda(device_ordinal);

  display_test_info();

  // If we are passing a gauge field to the test, we must allocate host memory.
  // If no gauge is passed, we generate a quenched field on the device.
  gauge_load = strcmp(latfile, "");
  gauge_store = strcmp(gauge_outfile, "");

  if (gauge_load) {
    printfQuda("Loading gauge field from host\n");
    for (int dir = 0; dir < 4; dir++) {
      host_gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    }
    constructHostGaugeField(host_gauge, gauge_param, argc, argv);
  }

  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  // return code for google test
  int test_rc = RUN_ALL_TESTS();
  if (gauge_load) {
    // release memory
    for (int dir = 0; dir < 4; dir++) host_free(host_gauge[dir]);
  }

  endQuda();

  finalizeComms();

  return test_rc;
}
