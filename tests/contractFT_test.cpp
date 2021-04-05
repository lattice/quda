#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <contract_reference.h>
#include "misc.h"

// google test
#include <gtest/gtest.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>

// If you add a new contraction type, this must be updated++
constexpr int NcontractType = 3;
// For googletest, names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore.
const char *names[] = {"DegrandRossi_FT_t","DegrandRossi_FT_z", "Staggered_FT_t"};

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
  int X[4] = {xdim, ydim, zdim, tdim};
  setDims(X);
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
int test(int contractionType, QudaPrecision test_prec)
{
  int X[4] = {xdim, ydim, zdim, tdim};

  QudaContractType cType = QUDA_CONTRACT_TYPE_INVALID;
  int nSpin = 0;
  int red_size = 0;
  switch (contractionType) {
  case 0:
    cType = QUDA_CONTRACT_TYPE_DR_FT_T;
    nSpin = 4;
    setSpinorSiteSize(spinor_site_size);
    red_size = X[3];
    break;
  case 1:
    cType = QUDA_CONTRACT_TYPE_DR_FT_Z;
    nSpin = 4;
    setSpinorSiteSize(spinor_site_size);
    red_size = X[2];
    break;
  case 2:
    cType = QUDA_CONTRACT_TYPE_STAGGERED_FT_T;
    nSpin = 1;
    setSpinorSiteSize(stag_spinor_site_size);
    red_size = X[3];
    break;
  default: errorQuda("Undefined contraction type %d\n", contractionType);
  }

  ColorSpinorParam cs_param;
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;
  QudaInvertParam inv_param = newQudaInvertParam();
  setContractInvertParam(inv_param);
  inv_param.cpu_prec = test_prec;
  inv_param.cuda_prec = test_prec;
  inv_param.cuda_prec_sloppy = test_prec;
  inv_param.cuda_prec_precondition = test_prec;
  if ( nSpin == 1 ) {
    inv_param.dslash_type = QUDA_STAGGERED_DSLASH;
    constructStaggeredSpinorParam(&cs_param, &inv_param, &gauge_param);
  } else {
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
    constructWilsonSpinorParam(&cs_param, &inv_param, &gauge_param);
  }


  size_t data_size = (test_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  int src_colors = 2; // source color (dilutions)
  int const nprops = nSpin * src_colors;
  size_t spinor_field_floats = V * my_spinor_site_size * 2;
  void *buffX = malloc(nprops * spinor_field_floats * data_size);
  void *buffY = malloc(nprops * spinor_field_floats * data_size);

  // propagators set to random values
  if (test_prec == QUDA_SINGLE_PRECISION) {
    for (size_t i = 0; i < spinor_field_floats * nprops; i++) {
      ((float *)buffX)[i] = 2.*(rand() / (float)RAND_MAX) - 1.;
      ((float *)buffY)[i] = 2.*(rand() / (float)RAND_MAX) - 1.;
    }
  } else {
    for (size_t i = 0; i < spinor_field_floats * nprops; i++) {
      ((double *)buffX)[i] = 2.*(rand() / (float)RAND_MAX) - 1.;
      ((double *)buffY)[i] = 2.*(rand() / (float)RAND_MAX) - 1.;
    }
  }

  // array of spinor field for each source spin and color
  void* spinorX[nprops];
  void* spinorY[nprops];
  { size_t off=0; for(int s=0; s<nprops; ++s, off += spinor_field_floats * data_size) {
      spinorX[s] = (void*)((uintptr_t)buffX + off);
      spinorY[s] = (void*)((uintptr_t)buffY + off);
    }}

  const int source_position[4]{0,0,0,0};
  const int n_mom = 1;
  const int mom[4]{0,0,0,0};

  int const n_contract_results = red_size * n_mom * nSpin*nSpin * 2;
  void *d_result = malloc(n_contract_results * sizeof(double)); // meson correlators are always double
  for(int c=0; c < n_contract_results; ++c) ((double*)d_result)[c] = 0.0;

  // Host side spinor data and result passed to QUDA.
  // QUDA will allocate GPU memory, transfer the data,
  // perform the requested contraction, and return the
  // result in the array 'result'

  // Perform GPU contraction.
  contractFTQuda(spinorX, spinorY, &d_result, cType, (void*)(&cs_param), src_colors, X, source_position, n_mom, mom);

  printfQuda("contraction:");
  for(int c=0; c < n_contract_results; c += 2) {
    if( c % 8 == 0 ) printfQuda("\n%3d ",c/2);
    printfQuda(" (%10.3e,%10.3e)",((double*)d_result)[c],((double*)d_result)[c+1]);
  }
  printfQuda("\n");
  // Compare contraction from the host and device. Return the number of detected faults.
  int faults = 0;
  /*  TODO: write CPU checks
  if (test_prec == QUDA_DOUBLE_PRECISION) {
    faults = contractionFT_reference((double *)spinorX, (double *)spinorY, (double *)d_result, cType,
                          src_colors, X, source_position, n_mom, mom);
  } else {
    faults = contractionFT_reference((float *)spinorX, (float *)spinorY, (float *)d_result, cType,
                          src_colors, X, source_position, n_mom, mom);
  }
  */
  printfQuda("Contraction comparison for contraction type %s complete with %d/%d faults\n", get_contract_str(cType),
             faults, n_contract_results);

  free(d_result);
  free(buffY);
  free(buffX);

  return faults;
}

// The following tests gets each contraction type and precision using google testing framework
using ::testing::Bool;
using ::testing::Combine;
using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

class ContractionTest : public ::testing::TestWithParam<::testing::tuple<int, int>>
{
  protected:
  ::testing::tuple<int, int> param;

  public:
  virtual ~ContractionTest() {}
  virtual void SetUp() { param = GetParam(); }
};

// Sets up the Google test
TEST_P(ContractionTest, verify)
{
  QudaPrecision prec = getPrecision(::testing::get<0>(GetParam()));
  int contractionType = ::testing::get<1>(GetParam());
  if ((QUDA_PRECISION & prec) == 0) GTEST_SKIP();
  auto faults = test(contractionType, prec);
  EXPECT_EQ(faults, 0) << "CPU and GPU implementations do not agree";
}

// Helper function to construct the test name
std::string getContractName(testing::TestParamInfo<::testing::tuple<int, int>> param)
{
  int prec = ::testing::get<0>(param.param);
  int contractType = ::testing::get<1>(param.param);
  std::string str(names[contractType]);
  str += std::string("_");
  str += std::string(get_prec_str(getPrecision(prec)));
  return str; // names[contractType] + "_" + prec_str[prec];
}

// Instantiate all test cases
INSTANTIATE_TEST_SUITE_P(QUDA, ContractionTest, Combine(Range(2, 4), Range(0, NcontractType)), getContractName);
