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

void *host_gauge[4];

void display_test_info()
{
  printfQuda("running the following test:\n");

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

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

class GaugeAlgTest : public ::testing::Test
{
protected:
  QudaGaugeParam param;

  Timer<false> a0, a1;
  double2 detu;
  double3 plaq;
  GaugeField *U;
  int nsteps;
  int nhbsteps;
  int novrsteps;
  bool coldstart;
  double beta_value;

  bool unit_test;

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
    if (prec == QUDA_DOUBLE_PRECISION) prec_val = gf_tolerance*1e2;
    return ((a0 < prec_val) && (a1 < prec_val) && (a2 < prec_val));
  }

  bool CheckDeterminant(double2 detu)
  {
    double prec_val = 5e-8;
    if (prec == QUDA_DOUBLE_PRECISION) prec_val =  gf_tolerance*1e2;
    return (std::abs(1.0 - detu.x) < prec_val && std::abs(detu.y) < prec_val);
  }

  virtual void SetUp()
  {
    if (execute) {
      setVerbosity(QUDA_VERBOSE);
      param = newQudaGaugeParam();

      // Setup gauge container.
      setWilsonGaugeParam(param);

      // Reunitarization setup
      int *num_failures_h = (int *)mapped_malloc(sizeof(int));
      int *num_failures_d = (int *)get_mapped_device_pointer(num_failures_h);
      SetReunitarizationConsts();	

      a0.start();
      
      // If no field is loaded, create a physical quenched field on the device
      if (!strcmp(latfile, "")) {

	GaugeFieldParam gParam(0, param);
	gParam.pad = 0;
	gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
	gParam.create      = QUDA_NULL_FIELD_CREATE;
	gParam.link_type   = param.type;
	gParam.reconstruct = param.reconstruct;
	gParam.setPrecision(gParam.Precision(), true);

	if(checkDimsPartitioned()) {
	  int y[4];
	  int R[4] = {0,0,0,0};
	  for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
	  for(int dir=0; dir<4; ++dir) y[dir] = param.X[dir] + 2 * R[dir];
	  int pad = 0;
	  GaugeFieldParam gParamEx(y, prec, link_recon,
				   pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
	  gParamEx.create = QUDA_ZERO_FIELD_CREATE;
	  gParamEx.order = gParam.order;
	  gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
	  gParamEx.t_boundary = gParam.t_boundary;
	  gParamEx.nFace = 1;
	  for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
	  U = new cudaGaugeField(gParamEx);
	} else {
	  U = new cudaGaugeField(gParam);
	}
	
	// CURAND random generator initialization
	randstates = new RNG(*U, 1234);
	
	nsteps = heatbath_num_steps;
	nhbsteps = heatbath_num_heatbath_per_step;
	novrsteps = heatbath_num_overrelax_per_step;
	coldstart = heatbath_coldstart;
	beta_value = heatbath_beta_value;
	a1.start();
		
	if (link_recon != QUDA_RECONSTRUCT_8 && coldstart)
	  InitGaugeField(*U);
	else
	  InitGaugeField(*U, *randstates);
	
	for (int step = 1; step <= nsteps; ++step) {
	  printfQuda("Step %d\n", step);
	  Monte(*U, *randstates, beta_value, nhbsteps, novrsteps);
	  
	  // Reunitarization
	  *num_failures_h = 0;
	  unitarizeLinks(*U, num_failures_d);
	  qudaDeviceSynchronize();
	  if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)\n", *num_failures_h);
	  plaq = plaquette(*U);
	  printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);	  
	}
	
	a1.stop();	
	printfQuda("Time Monte -> %.6f s\n", a1.last());	
      } else {

	// If a field is loaded, create a device feild and copy 
	GaugeField *host = nullptr;
	printfQuda("Copying gauge field from host\n");	  
	param.location = QUDA_CPU_FIELD_LOCATION;
	GaugeFieldParam gauge_field_param(host_gauge, param);	
	if (gauge_field_param.order <= 4) gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_NO;	
	host = static_cast<GaugeField*>(new cpuGaugeField(gauge_field_param));

	// switch the parameters for creating the mirror precise cuda gauge field
	gauge_field_param.create = QUDA_NULL_FIELD_CREATE;
	gauge_field_param.reconstruct = param.reconstruct;
	gauge_field_param.setPrecision(param.cuda_prec, true);
	gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
	gauge_field_param.pad = param.ga_pad;
	
	if(checkDimsPartitioned()) {	  
	  int R[4] = {0, 0, 0, 0};
	  for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
	  static TimeProfile GaugeFix("GaugeFix");
	  cudaGaugeField *tmp = new cudaGaugeField(gauge_field_param);
	  tmp->copy(*host);	
	  tmp->exchangeGhost();	  
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
	if (*num_failures_h > 0) errorQuda("Error in the unitarization (%d errors)\n", *num_failures_h);
	
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
      default: errorQuda("Invalid test type %d ", test_type);
      }

      host_free(num_failures_h);
    }
  }

  virtual void TearDown()
  {
    if (execute) {
      detu = getLinkDeterminant(*U);
      double2 tru = getLinkTrace(*U);
      printfQuda("Det: %.16e:%.16e\n", detu.x, detu.y);
      printfQuda("Tr: %.16e:%.16e\n", tru.x / 3.0, tru.y / 3.0);

      delete U;
      // Release all temporary memory used for data exchange between GPUs in multi-GPU mode
      PGaugeExchangeFree();

      a0.stop();
      printfQuda("Time -> %.6f s\n", a0.last());
      if (!strcmp(latfile, "")) delete randstates;
    }
    // If we performed a specific instance, switch off the
    // Google testing.
    if (test_type != 0) execute = false;
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
      saveTuneCache();
      // Save if output string is specified
      if (strcmp(gauge_outfile, "")) save_gauge();
    }
  }
  virtual void run_fft()
  {
    if (execute) {
      if (!checkDimsPartitioned()) {
        printfQuda("Landau gauge fixing with steepest descent method with FFTs\n");
        gaugeFixingFFT(*U, gf_gauge_dir, gf_maxiter, gf_verbosity_interval, gf_fft_alpha, gf_fft_autotune, gf_tolerance,
                       gf_theta_condition);

        auto plaq_gf = plaquette(*U);
	printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
	printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
        ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
	saveTuneCache();
        // Save if output string is specified
        if (strcmp(gauge_outfile, "")) save_gauge();
      } else {
        errorQuda("Cannot perform FFT gauge fixing with MPI partitions.");
      }
    }
  }

  virtual void save_gauge()
  {

    printfQuda("Saving the gauge field to file %s\n", gauge_outfile);

    QudaGaugeParam gauge_param = newQudaGaugeParam();
    setWilsonGaugeParam(gauge_param);

    void *cpu_gauge[4];
    for (int dir = 0; dir < 4; dir++) { cpu_gauge[dir] = malloc(V * gauge_site_size * gauge_param.cpu_prec); }

    GaugeFieldParam gParam(0, param);
    gParam.pad = 0;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.link_type = param.type;
    gParam.reconstruct = param.reconstruct;
    gParam.setPrecision(gParam.Precision(), true);

    cudaGaugeField *gauge;
    gauge = new cudaGaugeField(gParam);

    // copy into regular field
    copyExtendedGauge(*gauge, *U, QUDA_CUDA_FIELD_LOCATION);
    saveGaugeFieldQuda((void *)cpu_gauge, (void *)gauge, &gauge_param);

    // Write to disk
    write_gauge_field(gauge_outfile, cpu_gauge, gauge_param.cpu_prec, gauge_param.X, 0, (char **)0);

    for (int dir = 0; dir < 4; dir++) free(cpu_gauge[dir]);
    delete gauge;
  }
};


TEST_F(GaugeAlgTest, Generation)
{
  if (execute && !strcmp(latfile, "")) {
    detu = getLinkDeterminant(*U);
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
    saveTuneCache();
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
    saveTuneCache();	  
  }
}

TEST_F(GaugeAlgTest, Landau_FFT)
{
  if (execute) {
    if (!checkDimsPartitioned()) {
      printfQuda("Landau gauge fixing with steepest descent method with FFTs\n");
      gaugeFixingFFT(*U, 4, gf_maxiter, gf_verbosity_interval, gf_fft_alpha, gf_fft_autotune, gf_tolerance,
                     gf_theta_condition);
      auto plaq_gf = plaquette(*U);
      printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
      printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
      ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
      saveTuneCache();
    }
  }
}

TEST_F(GaugeAlgTest, Coulomb_FFT)
{
  if (execute) {
    if (!checkDimsPartitioned()) {
      printfQuda("Coulomb gauge fixing with steepest descent method with FFTs\n");
      gaugeFixingFFT(*U, 4, gf_maxiter, gf_verbosity_interval, gf_fft_alpha, gf_fft_autotune, gf_tolerance,
                     gf_theta_condition);
      auto plaq_gf = plaquette(*U);
      printfQuda("Plaq:    %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
      printfQuda("Plaq GF: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
      ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
      saveTuneCache();
    }
  }
}

int main(int argc, char **argv)
{
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
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;

  setWilsonGaugeParam(gauge_param);
  setDims(gauge_param.X);

  display_test_info();
  
  // If we are passing a gauge field to the test, we must allocate host memory.
  // If no gauge is passed, we generate a quenched field on the device.
  if (strcmp(latfile, "")) {
    printfQuda("Loading gauge field from host\n");
    for (int dir = 0; dir < 4; dir++) {
      host_gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    }
    constructHostGaugeField(host_gauge, gauge_param, argc, argv);    
  }

  // call srand() with a rank-dependent seed
  initRand();
  
  // initialize the QUDA library
  initQuda(device_ordinal);
    
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  // return code for google test
  int test_rc = RUN_ALL_TESTS();

  endQuda();

  // release memory
  for (int dir = 0; dir < 4; dir++) free(host_gauge[dir]);
  
  finalizeComms();

  return test_rc;
}
