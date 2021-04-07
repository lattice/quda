#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <timer.h>
#include <gauge_field.h>

#include <comm_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <gauge_tools.h>

#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>

#include <qio_field.h>

#include <gtest/gtest.h>

using namespace quda;

class GaugeAlgTest : public ::testing::Test {

  QudaGaugeParam param;
  host_timer_t a0;
  host_timer_t a1;
  RNG * randstates;

protected:
  cudaGaugeField *U;
  double3 plaq;

  void SetReunitarizationConsts(){
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only  = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    setUnitarizeLinksConstants(unitarize_eps, max_error,
                               reunit_allow_svd, reunit_svd_only,
                               svd_rel_error, svd_abs_error);

  }

  bool comparePlaquette(double3 a, double3 b){
    double a0,a1,a2;
    a0 = std::abs(a.x - b.x);
    a1 = std::abs(a.y - b.y);
    a2 = std::abs(a.z - b.z);
    double prec_val = 1.0e-5;
    if (prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
    if ((a0 < prec_val) && (a1 < prec_val) && (a2 < prec_val)) return true;
    return false;
  }

  bool CheckDeterminant(double2 detu){
    double prec_val = 5e-8;
    if (prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
    if (std::abs(1.0 - detu.x) < prec_val && std::abs(detu.y) < prec_val) return true;
    return false;
  }

  virtual void SetUp() {
    setVerbosity(verbosity);

    param = newQudaGaugeParam();

    // Setup gauge container.
    param.cpu_prec = prec;
    param.cpu_prec = prec;
    param.cuda_prec = prec;
    param.reconstruct = link_recon;
    param.cuda_prec_sloppy = prec;
    param.reconstruct_sloppy = link_recon;

    param.type = QUDA_WILSON_LINKS;
    param.gauge_order = QUDA_MILC_GAUGE_ORDER;

    param.X[0] = xdim;
    param.X[1] = ydim;
    param.X[2] = zdim;
    param.X[3] = tdim;
    setDims(param.X);

    param.anisotropy = 1.0;  //don't support anisotropy for now!!!!!!
    param.t_boundary = QUDA_PERIODIC_T;
    param.gauge_fix = QUDA_GAUGE_FIXED_NO;
    param.ga_pad = 0;

    GaugeFieldParam gParam(0, param);
    gParam.pad = 0;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.create      = QUDA_NULL_FIELD_CREATE;
    gParam.link_type   = param.type;
    gParam.reconstruct = param.reconstruct;
    gParam.setPrecision(gParam.Precision(), true);

#ifdef MULTI_GPU
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
#else
    U = new cudaGaugeField(gParam);
#endif
    // CURAND random generator initialization
    randstates = new RNG(*U, 1234);

    int nsteps = 10;
    int nhbsteps = 4;
    int novrsteps = 4;
    bool coldstart = false;
    double beta_value = 6.2;

    a0.start();
    a1.start();

    int *num_failures_h = (int *)mapped_malloc(sizeof(int));
    int *num_failures_d = (int *)get_mapped_device_pointer(num_failures_h);

    if (coldstart)
      InitGaugeField(*U);
    else
      InitGaugeField(*U, *randstates);

    // Reunitarization setup
    SetReunitarizationConsts();
    plaquette(*U);

    for (int step=1; step<=nsteps; ++step) {
      printfQuda("Step %d\n",step);
      Monte(*U, *randstates, beta_value, nhbsteps, novrsteps);

      //Reunitarize gauge links...
      *num_failures_h = 0;
      unitarizeLinks(*U, num_failures_d);
      if (*num_failures_h > 0) errorQuda("Error in the unitarization");

      plaquette(*U);
    }
    a1.stop();

    printfQuda("Time Monte -> %.6f s\n", a1.last());
    plaq = plaquette(*U);
    printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);

    host_free(num_failures_h);
  }

  virtual void TearDown() {
    auto detu = getLinkDeterminant(*U);
    double2 tru = getLinkTrace(*U);
    printfQuda("Det: %.16e:%.16e\n", detu.x, detu.y);
    printfQuda("Tr: %.16e:%.16e\n", tru.x/3.0, tru.y/3.0);

    delete U;
    //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
    PGaugeExchangeFree();

    a0.stop();
    printfQuda("Time -> %.6f s\n", a0.last());
    delete randstates;
  }

};

TEST_F(GaugeAlgTest, Generation)
{
  auto detu = getLinkDeterminant(*U);
  plaq = plaquette(*U);
  bool testgen = false;
  //check plaquette value for beta = 6.2
  if (plaq.x < 0.614 && plaq.x > 0.611 && plaq.y < 0.614 && plaq.y > 0.611) testgen = true;

  if (testgen) { ASSERT_TRUE(CheckDeterminant(detu)); }
}

TEST_F(GaugeAlgTest, Landau_Overrelaxation)
{
  const int reunit_interval = 10;
  printfQuda("Landau gauge fixing with overrelaxation\n");
  gaugeFixingOVR(*U, 4, 100, 10, 1.5, 0, reunit_interval, 1);
  auto plaq_gf = plaquette(*U);
  printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
  ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
}

TEST_F(GaugeAlgTest, Coulomb_Overrelaxation)
{
  const int reunit_interval = 10;
  printfQuda("Coulomb gauge fixing with overrelaxation\n");
  gaugeFixingOVR(*U, 3, 100, 10, 1.5, 0, reunit_interval, 1);
  auto plaq_gf = plaquette(*U);
  printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
  ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
}

TEST_F(GaugeAlgTest, Landau_FFT)
{
  if (!comm_partitioned()) {
    printfQuda("Landau gauge fixing with steepest descent method with FFTs\n");
    gaugeFixingFFT(*U, 4, 100, 10, 0.08, 0, 0, 1);
    auto plaq_gf = plaquette(*U);
    printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
    ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
  }
}

TEST_F(GaugeAlgTest, Coulomb_FFT)
{
  if (!comm_partitioned()) {
    printfQuda("Coulomb gauge fixing with steepest descent method with FFTs\n");
    gaugeFixingFFT(*U, 3, 100, 10, 0.08, 0, 0, 1);
    auto plaq_gf = plaquette(*U);
    printfQuda("Plaq: %.16e, %.16e, %.16e\n", plaq_gf.x, plaq_gf.y, plaq_gf.z);
    ASSERT_TRUE(comparePlaquette(plaq, plaq_gf));
  }
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;
  xdim=ydim=zdim=tdim=32;

  // command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  initQuda(device_ordinal);
  test_rc = RUN_ALL_TESTS();
  endQuda();

  finalizeComms();

  return test_rc;
}
