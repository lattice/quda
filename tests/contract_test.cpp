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
constexpr int NcontractType = 2;
// For googletest, names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore.
const char *names[] = {"OpenSpin", "DegrandRossi"};
const char *prec_str[] = {"quarter", "half", "single", "double"};

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

  printfQuda("Contraction test");
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
    if (result) warningQuda("Google tests for QUDA contraction failed!");
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

  QudaInvertParam inv_param = newQudaInvertParam();
  setContractInvertParam(inv_param);
  inv_param.cpu_prec = test_prec;
  inv_param.cuda_prec = test_prec;
  inv_param.cuda_prec_sloppy = test_prec;
  inv_param.cuda_prec_precondition = test_prec;

  size_t data_size = (test_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  void *spinorX = safe_malloc(V * spinor_site_size * data_size);
  void *spinorY = safe_malloc(V * spinor_site_size * data_size);
  void *d_result = safe_malloc(2 * V * 16 * data_size);

  if (test_prec == QUDA_SINGLE_PRECISION) {
    for (auto i = 0lu; i < V * spinor_site_size; i++) {
      ((float *)spinorX)[i] = rand() / (float)RAND_MAX;
      ((float *)spinorY)[i] = rand() / (float)RAND_MAX;
    }
  } else {
    for (auto i = 0lu; i < V * spinor_site_size; i++) {
      ((double *)spinorX)[i] = rand() / (double)RAND_MAX;
      ((double *)spinorY)[i] = rand() / (double)RAND_MAX;
    }
  }

  // Host side spinor data and result passed to QUDA.
  // QUDA will allocate GPU memory, transfer the data,
  // perform the requested contraction, and return the
  // result in the array 'result'
  // We then compare the GPU result with a CPU refernce code

  QudaContractType cType = QUDA_CONTRACT_TYPE_INVALID;
  switch (contractionType) {
  case 0: cType = QUDA_CONTRACT_TYPE_OPEN; break;
  case 1: cType = QUDA_CONTRACT_TYPE_DR; break;
  default: errorQuda("Undefined contraction type %d\n", contractionType);
  }

  // Perform GPU contraction.
  contractQuda(spinorX, spinorY, d_result, cType, &inv_param, X);

  // Compare each site contraction from the host and device.
  // It returns the number of faults it detects.
  int faults = 0;
  if (test_prec == QUDA_DOUBLE_PRECISION) {
    faults = contraction_reference((double *)spinorX, (double *)spinorY, (double *)d_result, cType);
  } else {
    faults = contraction_reference((float *)spinorX, (float *)spinorY, (float *)d_result, cType);
  }

  printfQuda("Contraction comparison for contraction type %s complete with %d/%d faults\n", get_contract_str(cType),
             faults, V * 16 * 2);

  host_free(spinorX);
  host_free(spinorY);
  host_free(d_result);

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
  virtual ~ContractionTest() { }
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
  str += std::string(prec_str[prec]);
  return str;
}

// Instantiate all test cases
INSTANTIATE_TEST_SUITE_P(QUDA, ContractionTest, Combine(Range(2, 4), Range(0, NcontractType)), getContractName);
