#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <contract_reference.h>
#include "misc.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

// google test
#include <gtest/gtest.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>


// If you add a new contraction type, this must be updated++
constexpr int NcontractType = 2;

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

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setInvertParam(QudaInvertParam &inv_param)
{

  inv_param.Ls = 1;
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_precondition = cuda_prec_precondition;

  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  // Quda performs contractions in Degrand-Rossi gamma basis,
  // but the user may suppy vectors in any supported order.
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;
}

const char *prec_str[] = {"single", "double"};

// For googletest, names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore.
const char *names[] = {"OpenSpin", "DegrandRossi"};

int main(int argc, char **argv)
{

  // QUDA initialise
  //-----------------------------------------------------------------------------
  // command line options
  auto app = make_app();
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);
  // call srand() with a rank-dependent seed
  initRand();
  display_test_info();

  // initialize the QUDA library
  initQuda(device);
  int X[4] = {xdim, ydim, zdim, tdim};
  setDims(X);
  setSpinorSiteSize(24);
  //-----------------------------------------------------------------------------

  // Start Google Test Suite
  //-----------------------------------------------------------------------------
  ::testing::InitGoogleTest(&argc, argv);

  prec = QUDA_INVALID_PRECISION;

  // Clear previous error state if it exists
  cudaGetLastError();

  // Check for correctness
  if (verify_results) {
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    int result = RUN_ALL_TESTS();
    if (result) warningQuda("Google tests for QUDA contraction failed!");
  }
  //-----------------------------------------------------------------------------

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}

// Functions used for Google testing
//-----------------------------------------------------------------------------

// Performs the CPU GPU comparison with the given parameters
void test(int contractionType, int Prec)
{

  QudaPrecision testPrec = QUDA_INVALID_PRECISION;
  switch (Prec) {
  case 0: testPrec = QUDA_SINGLE_PRECISION; break;
  case 1: testPrec = QUDA_DOUBLE_PRECISION; break;
  default: errorQuda("Undefined QUDA precision type %d\n", Prec);
  }

  int X[4] = {xdim, ydim, zdim, tdim};

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  inv_param.cpu_prec = testPrec;
  inv_param.cuda_prec = testPrec;
  inv_param.cuda_prec_sloppy = testPrec;
  inv_param.cuda_prec_precondition = testPrec;

  size_t sSize = (testPrec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  void *spinorX = malloc(V * spinorSiteSize * sSize);
  void *spinorY = malloc(V * spinorSiteSize * sSize);
  void *d_result = malloc(2 * V * 16 * sSize);

  if (testPrec == QUDA_SINGLE_PRECISION) {
    for (int i = 0; i < V * spinorSiteSize; i++) {
      ((float *)spinorX)[i] = rand() / (float)RAND_MAX;
      ((float *)spinorY)[i] = rand() / (float)RAND_MAX;
    }
  } else {
    for (int i = 0; i < V * spinorSiteSize; i++) {
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
  if (testPrec == QUDA_DOUBLE_PRECISION) {
    faults = contraction_reference((double *)spinorX, (double *)spinorY, (double *)d_result, cType, X);
  } else {
    faults = contraction_reference((float *)spinorX, (float *)spinorY, (float *)d_result, cType, X);
  }

  printfQuda("Contraction comparison for contraction type %s complete with %d/%d faults\n", get_contract_str(cType),
             faults, V * 16 * 2);

  EXPECT_LE(faults, 0) << "CPU and GPU implementations do not agree";

  free(spinorX);
  free(spinorY);
  free(d_result);
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
  int prec = ::testing::get<0>(GetParam());
  int contractionType = ::testing::get<1>(GetParam());
  test(contractionType, prec);
}

// Helper function to construct the test name
std::string getContractName(testing::TestParamInfo<::testing::tuple<int, int>> param)
{
  int prec = ::testing::get<0>(param.param);
  int contractType = ::testing::get<1>(param.param);
  std::string str(names[contractType]);
  str += std::string("_");
  str += std::string(prec_str[prec]);
  return str; // names[contractType] + "_" + prec_str[prec];
}

// Instantiate all test cases
INSTANTIATE_TEST_SUITE_P(QUDA, ContractionTest, Combine(Range(0, 2), Range(0, NcontractType)), getContractName);
