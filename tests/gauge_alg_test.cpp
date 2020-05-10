#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
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

using   namespace quda;

int num_failures=0;
int *num_failures_dev;

#define MAX(a,b) ((a)>(b)?(a):(b))
#define DABS(a) ((a)<(0.)?(-(a)):(a))

class GaugeAlgTest : public ::testing::Test {
 protected:
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

  bool checkDimsPartitioned(){
    if(comm_dim_partitioned(0) || comm_dim_partitioned(1) || comm_dim_partitioned(2) || comm_dim_partitioned(3)) return true;
    return false;
  }

  bool comparePlaquette(double3 a, double3 b){
    double a0,a1,a2;
    a0 = DABS(a.x - b.x);
    a1=DABS(a.y - b.y);
    a2=DABS(a.z - b.z);
    double prec_val = 1.0e-5;
    if(prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
    if( (a0 < prec_val) && (a1  < prec_val)  && (a2  < prec_val) ) return true;
    return false;
  }

  bool CheckDeterminant(double2 detu){
    double prec_val = 5e-8;
    if(prec == QUDA_DOUBLE_PRECISION) prec_val = 1.0e-15;
    if(DABS(1.0 - detu.x) < prec_val && DABS(detu.y) < prec_val) return true;
    return false;
  }


  void CallUnitarizeLinks(cudaGaugeField *cudaInGauge){
    unitarizeLinks(*cudaInGauge, num_failures_dev);
    cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if(num_failures>0){
      cudaFree(num_failures_dev);
      errorQuda("Error in the unitarization\n");
      exit(1);
    }
    cudaMemset(num_failures_dev, 0, sizeof(int));
  }

  virtual void SetUp() {
    setVerbosity(QUDA_VERBOSE);

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
    cudaInGauge = new cudaGaugeField(gParamEx);
#else
    cudaInGauge = new cudaGaugeField(gParam);
#endif
    // CURAND random generator initialization
    randstates = new RNG(gParam, 1234);
    randstates->Init();

    nsteps = 10;
    nhbsteps = 4;
    novrsteps = 4;
    coldstart = false;
    beta_value = 6.2;

    a0.Start(__func__, __FILE__, __LINE__);
    a1.Start(__func__, __FILE__, __LINE__);

    cudaMalloc((void**)&num_failures_dev, sizeof(int));
    cudaMemset(num_failures_dev, 0, sizeof(int));
    if(num_failures_dev == NULL) errorQuda("cudaMalloc failed for dev_pointer\n");
    if(link_recon != QUDA_RECONSTRUCT_8 && coldstart) InitGaugeField( *cudaInGauge);
     else{
       InitGaugeField( *cudaInGauge, *randstates );
     }
    // Reunitarization setup
    SetReunitarizationConsts();
    plaquette(*cudaInGauge);

    for(int step=1; step<=nsteps; ++step){
      printfQuda("Step %d\n",step);
      Monte( *cudaInGauge, *randstates, beta_value, nhbsteps, novrsteps);
      //Reunitarize gauge links...
      CallUnitarizeLinks(cudaInGauge);
      plaquette(*cudaInGauge);
    }
    a1.Stop(__func__, __FILE__, __LINE__);

    printfQuda("Time Monte -> %.6f s\n", a1.Last());
    plaq = plaquette(*cudaInGauge);
    printfQuda("Plaq: %.16e , %.16e, %.16e\n", plaq.x, plaq.y, plaq.z);
  }

  virtual void TearDown() {
    detu = getLinkDeterminant(*cudaInGauge);
    double2 tru = getLinkTrace(*cudaInGauge);
    printfQuda("Det: %.16e:%.16e\n", detu.x, detu.y);
    printfQuda("Tr: %.16e:%.16e\n", tru.x/3.0, tru.y/3.0);


    delete cudaInGauge;
    cudaFree(num_failures_dev);
    //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
    PGaugeExchangeFree();

    a0.Stop(__func__, __FILE__, __LINE__);
    printfQuda("Time -> %.6f s\n", a0.Last());
    randstates->Release();
    delete randstates;
  }


  QudaGaugeParam param;

  Timer a0,a1;
  double2 detu;// = getLinkDeterminant(*cudaInGauge);
  double3 plaq;// = plaquette( *cudaInGauge, QUDA_CUDA_FIELD_LOCATION) ;
  cudaGaugeField *cudaInGauge;
  int nsteps;
  int nhbsteps;
  int novrsteps;
  bool coldstart;
  double beta_value;
  RNG * randstates;

};


TEST_F(GaugeAlgTest,Generation){
  detu = getLinkDeterminant(*cudaInGauge);
  plaq = plaquette(*cudaInGauge);
  bool testgen = false;
  //check plaquette value for beta = 6.2
  if(plaq.x < 0.614 && plaq.x > 0.611 && plaq.y < 0.614 && plaq.y > 0.611) testgen = true;

  if(testgen){
    ASSERT_TRUE(CheckDeterminant(detu));
  }
}

TEST_F(GaugeAlgTest,Landau_Overrelaxation){
  const int reunit_interval = 10;
  printfQuda("Landau gauge fixing with overrelaxation\n");
  gaugefixingOVR(*cudaInGauge, 4, 100, 10, 1.5, 0, reunit_interval, 1);
  ASSERT_TRUE(comparePlaquette(plaq, plaquette(*cudaInGauge)));
}

TEST_F(GaugeAlgTest,Coulomb_Overrelaxation){
  const int reunit_interval = 10;
  printfQuda("Coulomb gauge fixing with overrelaxation\n");
  gaugefixingOVR(*cudaInGauge, 3, 100, 10, 1.5, 0, reunit_interval, 1);
  ASSERT_TRUE(comparePlaquette(plaq, plaquette(*cudaInGauge)));
}

TEST_F(GaugeAlgTest,Landau_FFT){
  if(!checkDimsPartitioned()){
    printfQuda("Landau gauge fixing with steepest descent method with FFTs\n");
    gaugefixingFFT(*cudaInGauge, 4, 100, 10, 0.08, 0, 0, 1);
    ASSERT_TRUE(comparePlaquette(plaq, plaquette(*cudaInGauge)));
  }
}

TEST_F(GaugeAlgTest,Coulomb_FFT){
  if(!checkDimsPartitioned()){
    printfQuda("Coulomb gauge fixing with steepest descent method with FFTs\n");
    gaugefixingFFT(*cudaInGauge, 3, 100, 10, 0.08, 0, 0, 1);
    ASSERT_TRUE(comparePlaquette(plaq, plaquette(*cudaInGauge)));
  }
}


int main(int argc, char **argv){
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

  initQuda(device);
  test_rc = RUN_ALL_TESTS();
  endQuda();

  finalizeComms();

  return test_rc;
}
