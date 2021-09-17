#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
//#include <contract_reference.h>
#include "misc.h"

// google test
#include <gtest/gtest.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>

// For googletest, names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore.
const char *names[] = {"StaggeredQSmearAllT","StaggeredQSmearFixedT"};

namespace quda
{
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %d/%d/%d          %d         %d\n", get_prec_str(prec), get_prec_str(prec_sloppy),
             xdim, ydim, zdim, tdim, Lsdim);

  printfQuda("contractFTQuda test");
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

int main(int argc, char **argv)
{
  // Start Google Test Suite
  //-----------------------------------------------------------------------------
  ::testing::InitGoogleTest(&argc, argv);

  // QUDA initialise
  //-----------------------------------------------------------------------------
  // command line options
  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  // call srand() with a rank-dependent seed
  initRand();
  display_test_info();

  // initialize the QUDA library
  initQuda(device_ordinal);
  int X[4] = {xdim, ydim, zdim, tdim}; // local dims
  setDims(X);
  // Hack: use the domain wall dimensions so we may use the 5th dim for multi indexing
  dw_setDims(X, 1);  
  //
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize,64*1024*1024); // DEBUG
  //-----------------------------------------------------------------------------

  prec = QUDA_INVALID_PRECISION;

  // Check for correctness
  int result = 0;
  if (verify_results) {
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    result = RUN_ALL_TESTS();
    if (result) warningQuda("Google tests for contractFTQuda failed!");
  }
  //-----------------------------------------------------------------------------

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return result;
}

// Functions used for Google testing
//-----------------------------------------------------------------------------

// Performs the CPU GPU comparison with the given parameters
int test(int qsmearType, QudaPrecision test_prec)//0 => all t-slices, 
{
  int X[4] = {xdim, ydim, zdim, tdim};

  int nSpin = 0;

  ColorSpinorParam cs_param;

  cs_param.nColor = 3;
  cs_param.nSpin  = 1;
  cs_param.nDim   = 4;
  for(int i = 0; i < 4; i++)
    cs_param.x[i] = X[i];
  cs_param.x[4] = 1;
  cs_param.siteSubset = QUDA_FULL_SITE_SUBSET;
  cs_param.setPrecision(test_prec);
  cs_param.pad = 0;
  cs_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  cs_param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  cs_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless for staggered, but required by the code.
  cs_param.create = QUDA_ZERO_FIELD_CREATE;
  cs_param.location = QUDA_CPU_FIELD_LOCATION;
  cs_param.pc_type = QUDA_4D_PC;

  int my_spinor_site_size = 3; //

  size_t data_size = (test_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t spinor_field_floats = V * my_spinor_site_size * 2; // 
  void *buff  = malloc(spinor_field_floats * data_size);

  // spinor set to random values
  if (test_prec == QUDA_SINGLE_PRECISION) {
    for (size_t i = 0; i < spinor_field_floats; i++) {
      ((float *)buff)[i]  = 2.*(rand() / (float)RAND_MAX) - 1.;
    }
  } else {
    for (size_t i = 0; i < spinor_field_floats; i++) {
      ((double *)buff)[i] = 2.*(rand() / (double)RAND_MAX) - 1.;
    }
  }

  // array of spinor field for each source spin and color
  void* spinor;
  size_t off=0; 
  const int nspinors = 1;
  for(int s=0; s<nsponors; ++s, off += spinor_field_floats * data_size) {
    spinor[s] = (void*)((uintptr_t)buff  + off);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  setStaggeredGaugeParam(gauge_param); 
  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  void* qdp_inlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_fatlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_longlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void *milc_fatlink = nullptr;
  void *milc_longlink = nullptr;
  GaugeField *cpuFat = nullptr;
  GaugeField *cpuLong = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_fatlink[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_longlink[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  }
  milc_fatlink = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  milc_longlink = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  // For load, etc
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, argc, argv);
  // Reorder gauge fields to MILC order
  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  // This needs to be called before `loadFatLongGaugeQuda` because this routine also loads the
  // gauge fields with different parameters.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);
    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  // Create ghost gauge fields in case of multi GPU builds.
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuFatParam(gauge_param, milc_fatlink);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = GaugeField::Create(cpuFatParam);

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(gauge_param, milc_longlink);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = GaugeField::Create(cpuLongParam);

  loadFatLongGaugeQuda(milc_fatlink, milc_longlink, gauge_param);

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------
    

  // Host side spinor data passed to QUDA.
  // QUDA will allocate GPU memory, transfer the data,
  // perform the requested operation, and return the
  // result

  // Perform GPU smearing.

  // smearing parameters
  double omega = 2.0;
  int n_steps  = 50;
  double smear_coeff = -1.0 * omega * omega / ( 4*Nsteps );
  
  const int compute_2link = 0;
  const int t0            = 0;// not used yet

  performTwoLinkGaussianSmearNStep(spinor, &inv_param, n_steps, smear_coeff, compute_2link, t0);  

  // Compare qsmearing from the host and device. Return the number of detected faults.
  int faults = 0;
  
  free(buffIn);
  free(buffOut);

  return faults;
}

// The following tests gets each contraction type and precision using google testing framework
using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

class QSmearTest : public ::testing::TestWithParam<::testing::tuple<int, int>>
{
  protected:
  ::testing::tuple<int, int> param;

  public:
  virtual ~QSmearTest() {}
  virtual void SetUp() { param = GetParam(); }
};

// Sets up the Google test
TEST_P(QSmearTest, verify)
{
  QudaPrecision prec = getPrecision(::testing::get<0>(GetParam()));
  int qsmearType = ::testing::get<1>(GetParam());
  if ((QUDA_PRECISION & prec) == 0) GTEST_SKIP();
  auto faults = test(qsmearType, prec);
  //EXPECT_EQ(faults, 0) << "CPU and GPU implementations do not agree";
}

// Helper function to construct the test name
std::string getQSmearName(testing::TestParamInfo<::testing::tuple<int, int>> param)
{
  int prec = ::testing::get<0>(param.param);
  int qsmearType = ::testing::get<1>(param.param);
  std::string str(names[contractType]);
  str += std::string("_");
  str += std::string(get_prec_str(getPrecision(prec)));
  return str; // names[contractType] + "_" + prec_str[prec];
}

// Instantiate all test cases: prec 3==double, 2==float; contractType 2==staggered_FT
INSTANTIATE_TEST_SUITE_P(QUDA, QSmearTest, Combine(Range(2, 3), Range(2, NcontractType)), getQSmearName);



