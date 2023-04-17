#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <gauge_field.h>

#include <misc.h>
#include <timer.h>
#include <gauge_tools.h>
#include <tune_quda.h>

#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

#include <qio_field.h>
#include <test.h>

using namespace quda;

//***********************************************************//
// This boolean controls whether or not the full Google test //
// is done. If the user passes a value of 1 or 2 to --test   //
// then a single instance of OVR or FFT gauge fixing is done //
// and the value of this bool is set to false. Otherwise the //
// Google tests are performed.                               //
//***********************************************************//
bool execute = true;

bool gauge_load;
bool gauge_store;

std::array<std::vector<char>, 4> host_gauge;

class GaugeAlgTest : public ::testing::Test
{
protected:
  QudaGaugeParam param;
  Timer<false> a0, a1;
  GaugeField *U;
  double3 plaq;

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

  bool comparePlaquette(double3 a, double3 b)
  {
    auto a0 = std::abs(a.x - b.x);
    auto a1 = std::abs(a.y - b.y);
    auto a2 = std::abs(a.z - b.z);
    double prec_val = 1.0e-5;
    if (prec == QUDA_DOUBLE_PRECISION) prec_val = gf_tolerance * 1e2;
    return ((a0 < prec_val) && (a1 < prec_val) && (a2 < prec_val));
  }

  bool CheckDeterminant(double2 detu)
  {
    double prec_val = 5e-8;
    if (prec == QUDA_DOUBLE_PRECISION) prec_val = gf_tolerance * 1e2;
    return (std::abs(1.0 - detu.x) < prec_val && std::abs(detu.y) < prec_val);
  }

  virtual void SetUp()
  {
#ifndef QUDA_BUILD_NATIVE_FFT // skip FFT tests if FFT not available
    const ::testing::TestInfo *const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    const char *name = test_info->name();
    if (strcmp(name, "Landau_FFT") == 0 || strcmp(name, "Coulomb_FFT") == 0) {
      execute = false;
      GTEST_SKIP();
    }
#endif
    if (execute) {
      setVerbosity(verbosity);
      param = newQudaGaugeParam();

      // Setup gauge container.
      setWilsonGaugeParam(param);
      param.t_boundary = QUDA_PERIODIC_T;

      // Reunitarization setup
      int *num_failures_h = (int *)mapped_malloc(sizeof(int));
      int *num_failures_d = (int *)get_mapped_device_pointer(num_failures_h);
      SetReunitarizationConsts();

      a0.start();

      // If no field is loaded, create a physical quenched field on the device
      if (!gauge_load) {
        GaugeFieldParam gParam(param);
        gParam.location = QUDA_CUDA_FIELD_LOCATION;
        gParam.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
        gParam.create = QUDA_NULL_FIELD_CREATE;
        gParam.reconstruct = link_recon;
        gParam.setPrecision(prec, true);
        for (int d = 0; d < 4; d++) {
          if (comm_dim_partitioned(d)) gParam.r[d] = 2;
          gParam.x[d] += 2 * gParam.r[d];
        }

        U = new cudaGaugeField(gParam);

        RNG randstates(*U, 1234);

        int nsteps = heatbath_num_steps;
        int nhbsteps = heatbath_num_heatbath_per_step;
        int novrsteps = heatbath_num_overrelax_per_step;
        bool coldstart = heatbath_coldstart;
        double beta_value = heatbath_beta_value;
        a1.start();

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
          plaq = plaquette(*U);
          printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
        }

        a1.stop();
        printfQuda("Time Monte -> %.6f s\n", a1.last());
      } else {

        // If a field is loaded, create a device field and copy
        printfQuda("Copying gauge field from host\n");
        param.location = QUDA_CPU_FIELD_LOCATION;
        void *h_gauge[] = {host_gauge[0].data(), host_gauge[1].data(), host_gauge[2].data(), host_gauge[3].data()};
        GaugeFieldParam gauge_field_param(param, h_gauge);
        gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
        GaugeField *host = GaugeField::Create(gauge_field_param);

        // switch the parameters for creating the mirror precise cuda gauge field
        gauge_field_param.location = QUDA_CUDA_FIELD_LOCATION;
        gauge_field_param.create = QUDA_NULL_FIELD_CREATE;
        gauge_field_param.reconstruct = param.reconstruct;
        gauge_field_param.setPrecision(param.cuda_prec, true);

        if (comm_partitioned()) {
          lat_dim_t R = {0, 0, 0, 0};
          for (int d = 0; d < 4; d++)
            if (comm_dim_partitioned(d)) R[d] = 2;
          static TimeProfile GaugeFix("GaugeFix");
          cudaGaugeField *tmp = new cudaGaugeField(gauge_field_param);
          tmp->copy(*host);
          U = createExtendedGauge(*tmp, R, GaugeFix);
          delete tmp;
        } else {
          U = new cudaGaugeField(gauge_field_param);
          U->copy(*host);
        }

        delete host;

        // Reunitarization
        *num_failures_h = 0;
        unitarizeLinks(*U, num_failures_d);
        qudaDeviceSynchronize();
        if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)", *num_failures_h);

        plaq = plaquette(*U);
        printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
      }

      // If a specific test type is requested, perfrom it now and then
      // turn off all Google tests in the tear down.
      switch (test_type) {
      case 0:
        // Do the Google testing
        break;
      case 1: run_ovr(); break;
      case 2: run_fft(); break;
      default: errorQuda("Invalid test type %d", test_type);
      }

      host_free(num_failures_h);
    }
  }

  virtual void TearDown()
  {
    if (execute) {
      auto detu = getLinkDeterminant(*U);
      auto tru = getLinkTrace(*U);
      printfQuda("Det: %.16e:%.16e\n", detu.x, detu.y);
      printfQuda("Tr: %.16e:%.16e\n", tru.x / 3.0, tru.y / 3.0);

      delete U;
      // Release all temporary memory used for data exchange between GPUs in multi-GPU mode
      PGaugeExchangeFree();

      a0.stop();
      printfQuda("Time -> %.6f s\n", a0.last());
    }
    // Reset execute.  If we performed a specific instance, switch off the Google testing.
    execute = (test_type == 0);
  }

  virtual void run_ovr()
  {
    if (execute) {
      gaugeFixingOVR(*U, gf_gauge_dir, gf_maxiter, gf_verbosity_interval, gf_ovr_relaxation_boost, gf_tolerance,
                     gf_reunit_interval, gf_theta_condition);
      auto plaq_gf = plaquette(*U);
      printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
      printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
      ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
      // Save if output string is specified
      if (gauge_store) save_gauge();
    }
  }
  virtual void run_fft()
  {
    if (execute) {
      if (!comm_partitioned()) {
        printfQuda("Landau gauge fixing with steepest descent method with FFTs\n");
        gaugeFixingFFT(*U, gf_gauge_dir, gf_maxiter, gf_verbosity_interval, gf_fft_alpha, gf_fft_autotune, gf_tolerance,
                       gf_theta_condition);

        auto plaq_gf = plaquette(*U);
        printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
        printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
        ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
        // Save if output string is specified
        if (gauge_store) save_gauge();
      } else {
        errorQuda("Cannot perform FFT gauge fixing with MPI partitions.");
      }
    }
  }

  virtual void save_gauge()
  {
    printfQuda("Saving the gauge field to file %s\n", gauge_outfile.c_str());

    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);

    void *cpu_gauge[4];
    for (int dir = 0; dir < 4; dir++) { cpu_gauge[dir] = safe_malloc(V * gauge_site_size * gauge_param.cpu_prec); }

    GaugeFieldParam gParam(param);
    gParam.location = QUDA_CUDA_FIELD_LOCATION;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.link_type = param.type;
    gParam.reconstruct = param.reconstruct;
    gParam.setPrecision(gParam.Precision(), true);

    cudaGaugeField *gauge = new cudaGaugeField(gParam);

    // copy into regular field
    copyExtendedGauge(*gauge, *U, QUDA_CUDA_FIELD_LOCATION);
    saveGaugeFieldQuda((void *)cpu_gauge, (void *)gauge, &gauge_param);

    // Write to disk
    write_gauge_field(gauge_outfile.c_str(), cpu_gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char **)0);

    for (int dir = 0; dir < 4; dir++) host_free(cpu_gauge[dir]);
    delete gauge;
  }
};

TEST_F(GaugeAlgTest, Generation)
{
  if (execute && !gauge_load) {
    auto detu = getLinkDeterminant(*U);
    ASSERT_TRUE(CheckDeterminant(detu));
  }
}

TEST_F(GaugeAlgTest, Landau_Overrelaxation)
{
  if (execute) {
    printfQuda("Landau gauge fixing with overrelaxation\n");
    gaugeFixingOVR(*U, 4, gf_maxiter, gf_verbosity_interval, gf_ovr_relaxation_boost, gf_tolerance, gf_reunit_interval,
                   gf_theta_condition);
    auto plaq_gf = plaquette(*U);
    printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
    printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
    ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
  }
}

TEST_F(GaugeAlgTest, Coulomb_Overrelaxation)
{
  if (execute) {
    printfQuda("Coulomb gauge fixing with overrelaxation\n");
    gaugeFixingOVR(*U, 3, gf_maxiter, gf_verbosity_interval, gf_ovr_relaxation_boost, gf_tolerance, gf_reunit_interval,
                   gf_theta_condition);
    auto plaq_gf = plaquette(*U);
    printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
    printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
    ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
  }
}

TEST_F(GaugeAlgTest, Landau_FFT)
{
  if (execute) {
    if (!comm_partitioned()) {
      printfQuda("Landau gauge fixing with steepest descent method with FFTs\n");
      gaugeFixingFFT(*U, 4, gf_maxiter, gf_verbosity_interval, gf_fft_alpha, gf_fft_autotune, gf_tolerance,
                     gf_theta_condition);
      auto plaq_gf = plaquette(*U);
      printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
      printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
      ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
    }
  }
}

TEST_F(GaugeAlgTest, Coulomb_FFT)
{
  if (execute) {
    if (!comm_partitioned()) {
      printfQuda("Coulomb gauge fixing with steepest descent method with FFTs\n");
      gaugeFixingFFT(*U, 4, gf_maxiter, gf_verbosity_interval, gf_fft_alpha, gf_fft_autotune, gf_tolerance,
                     gf_theta_condition);
      auto plaq_gf = plaquette(*U);
      printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
      printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
      ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
    }
  }
}

struct gauge_alg_test : quda_test {

  void display_info() const override
  {
    quda_test::display_info();

    switch (test_type) {
    case 0: printfQuda("\n Google testing\n"); break;
    case 1: printfQuda("\nOVR gauge fix\n"); break;
    case 2: printfQuda("\nFFT gauge fix\n"); break;
    default: errorQuda("Undefined test type %d given", test_type);
    }

    printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
    printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
               get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
               tdim, Lsdim);
  }

  void add_command_line_group(std::shared_ptr<QUDAApp> app) const override
  {
    quda_test::add_command_line_group(app);
    add_gaugefix_option_group(app);
    add_heatbath_option_group(app);

    test_type = 0;
    CLI::TransformPairs<int> test_type_map {{"Google", 0}, {"OVR", 1}, {"FFT", 2}};
    app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  }

  gauge_alg_test(int argc, char **argv) : quda_test("Gauge Alg Test", argc, argv) { }
};

int main(int argc, char **argv)
{
  gauge_alg_test test(argc, argv);
  test.init();

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  gauge_load = (latfile.size() > 0);
  gauge_store = (gauge_outfile.size() > 0);

  // If we are passing a gauge field to the test, we must allocate host memory.
  // If no gauge is passed, we generate a quenched field on the device.
  if (gauge_load) {
    printfQuda("Loading gauge field from host\n");
    for (int dir = 0; dir < 4; dir++) { host_gauge[dir].resize(V * gauge_site_size * host_gauge_data_type_size); }
    void *h_gauge[] = {host_gauge[0].data(), host_gauge[1].data(), host_gauge[2].data(), host_gauge[3].data()};
    constructHostGaugeField(h_gauge, gauge_param, argc, argv);
  }

  return test.execute(); // run tests
}
