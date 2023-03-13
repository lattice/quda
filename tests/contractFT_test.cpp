#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <contractFT_reference.h>
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
  int X[4] = {xdim, ydim, zdim, tdim}; // local dims
  setDims(X);
  //cudaDeviceSetLimit(cudaLimitPrintfFifoSize,64*1024*1024); // DEBUG-JNS
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
    red_size = comm_dim(3)*X[3]; //DMH: total temporal dim
    break;
  case 1:
    cType = QUDA_CONTRACT_TYPE_DR_FT_Z;
    nSpin = 4;
    red_size = comm_dim(2)*X[2]; //DMH total Z dim
    break;
  case 2:
    cType = QUDA_CONTRACT_TYPE_STAGGERED_FT_T;
    nSpin = 1;
    red_size = comm_dim(3)*X[3]; //DMH total temporal dim
    break;
  default: errorQuda("Undefined contraction type %d\n", contractionType);
  }

  ColorSpinorParam cs_param;

  cs_param.nColor = 3;
  cs_param.nSpin = nSpin;
  cs_param.nDim = 4;
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

  int my_spinor_site_size = nSpin * 3; //DMH: nSpin X nColor 

  size_t data_size = (test_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  int src_colors = 1; // source color (dilutions)
  int const nprops = nSpin * src_colors;
  size_t spinor_field_floats = V * my_spinor_site_size * 2; // DMH: Vol * spinor elems * 2(re,im) 
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
      ((double *)buffX)[i] = 2.*(rand() / (double)RAND_MAX) - 1.;
      ((double *)buffY)[i] = 2.*(rand() / (double)RAND_MAX) - 1.;
    }
  }

  // array of spinor field for each source spin and color
  void* spinorX[nprops];
  void* spinorY[nprops];
  size_t off=0; 
  for(int s=0; s<nprops; ++s, off += spinor_field_floats * data_size) {
    spinorX[s] = (void*)((uintptr_t)buffX + off);
    spinorY[s] = (void*)((uintptr_t)buffY + off);
  }

  const QudaFFTSymmType eo = QUDA_FFT_SYMM_EO;
  const QudaFFTSymmType ev = QUDA_FFT_SYMM_EVEN;
  const QudaFFTSymmType od = QUDA_FFT_SYMM_ODD;
  const int source_position[4]{0,0,0,0};
  const int n_mom = 18;
  const int mom[n_mom*4]{
      0, 0, 0, 0,     0, 0, 0, 0,
      1, 0, 0, 0,    -1, 0, 0, 0,     1, 0, 0, 0,     1, 0, 0, 0,
      0, 1, 0, 0,     0,-1, 0, 0,     0, 1, 0, 0,     0, 1, 0, 0,     
      0, 0, 1, 0,     0, 0,-1, 0,     0, 0, 1, 0,     0, 0, 1, 0,
      0, 1, 1, 0,     0,-1,-1, 0,     0, 1, 1, 0,     0, 1, 1, 0
      };
  const QudaFFTSymmType fft_type[n_mom*4]{
    eo, eo, eo, eo, // (0,0,0)
    ev, ev, ev, eo,
    eo, eo, eo, eo, // (1,0,0)
    eo, eo, eo, eo,
    ev, ev, ev, eo,
    od, ev, ev, eo,
    eo, eo, eo, eo, // (0,1,0)
    eo, eo, eo, eo,
    ev, ev, ev, eo,
    ev, od, ev, eo,
    eo, eo, eo, eo, // (0,0,1)
    eo, eo, eo, eo,
    ev, ev, ev, eo,
    ev, ev, od, eo,
    eo, eo, eo, eo, // (0,1,1)
    eo, eo, eo, eo,
    ev, ev, ev, eo,
    ev, od, od, eo
      };

  int const n_contract_results = red_size * n_mom * nSpin*nSpin * 2;
  void *d_result = malloc(n_contract_results * sizeof(double)); // meson correlators are always double
  for(int c=0; c < n_contract_results; ++c) ((double*)d_result)[c] = 0.0;

  // Host side spinor data and result passed to QUDA.
  // QUDA will allocate GPU memory, transfer the data,
  // perform the requested contraction, and return the
  // result in the array 'result'

  // Perform GPU contraction.
  contractFTQuda(spinorX, spinorY, &d_result, cType, (void*)(&cs_param), src_colors, X, source_position, n_mom, mom, fft_type);

  #if 0
  const char* ftype[4]{"?","O","E","EO"};
  printfQuda("contractions:");
  for(int k=0; k<n_mom; ++k) {
    printfQuda("\np = %2d %2d %2d %2d;  sym = %2s %2s %2s %2s",
	       mom[4*k+0],mom[4*k+1],mom[4*k+2],mom[4*k+3],
	       ftype[fft_type[4*k+0]],ftype[fft_type[4*k+1]],ftype[fft_type[4*k+2]],ftype[fft_type[4*k+3]]);
    printfQuda("\n[");
    for(int c=0; c<red_size*nSpin*nSpin*2; c+= 2) {
      int indx = k*red_size*nSpin*nSpin*2 + c;
      if( c > 0 && (c % 8) == 0 ) printfQuda("\n");
      printfQuda(" (%10.3e+%10.3ej),",((double*)d_result)[indx],((double*)d_result)[indx+1]);
    }
    printfQuda("]\n");
  }
  printfQuda("\n");
  #endif
  // Compare contraction from the host and device. Return the number of detected faults.
  int faults = 0;
  if (test_prec == QUDA_DOUBLE_PRECISION) {
    faults = contractionFT_reference<double>((double **)spinorX, (double **)spinorY, (double *)d_result, cType,
                          src_colors, X, source_position, n_mom, mom, fft_type);
  } else {
    faults = contractionFT_reference<float>((float **)spinorX, (float **)spinorY, (double *)d_result, cType,
                          src_colors, X, source_position, n_mom, mom, fft_type);
  }
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

// Instantiate all test cases: prec 3==double, 2==float; contractType 2==staggered_FT
INSTANTIATE_TEST_SUITE_P(QUDA, ContractionTest, Combine(Range(2, 3), Range(2, NcontractType)), getContractName);
