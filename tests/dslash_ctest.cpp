#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"
#include "dslash_test_helpers.h"

#include "dslash_test_utils.h"

// google test frame work
#include <gtest/gtest.h>

using namespace quda;

DslashTestWrapper dslash_test_wrapper;

// For loading the gauge fields
int argc_copy;
char **argv_copy;

const char *prec_str[] = {"quarter", "half", "single", "double"};
const char *recon_str[] = {"r18", "r12", "r8"};

// For googletest names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore

void display_test_info(int precision, QudaReconstructType link_recon)
{
  auto prec = getPrecision(precision);
  // printfQuda("running the following test:\n");

  printfQuda("prec    recon   test_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension dslash_type    niter\n");
  printfQuda("%6s   %2s       %s           %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n",
             get_prec_str(prec), get_recon_str(link_recon), get_string(dtest_type_map, dslash_test_wrapper.dtest_type).c_str(),
             get_matpc_str(matpc_type), dagger, xdim, ydim, zdim, tdim, Lsdim, get_dslash_str(dslash_type), niter);
  // printfQuda("Grid partition info:     X  Y  Z  T\n");
  // printfQuda("                         %d  %d  %d  %d\n",
  //   dimPartitioned(0),
  //   dimPartitioned(1),
  //   dimPartitioned(2),
  //   dimPartitioned(3));

  if (dslash_test_wrapper.test_split_grid) {
    printfQuda("Testing with split grid: %d  %d  %d  %d\n", grid_partition[0], grid_partition[1], grid_partition[2],
               grid_partition[3]);
  }

  return ;

}

using ::testing::TestWithParam;
using ::testing::Bool;
using ::testing::Values;
using ::testing::Range;
using ::testing::Combine;

class DslashTest : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {
protected:
  ::testing::tuple<int, int, int> param;

  bool skip()
  {
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if ((QUDA_PRECISION & getPrecision(::testing::get<0>(GetParam()))) == 0
        || (QUDA_RECONSTRUCT & getReconstructNibble(recon)) == 0) {
      return true;
    }

    if (dslash_type == QUDA_MOBIUS_DWF_DSLASH && dslash_test_wrapper.dtest_type == dslash_test_type::MatPCDagMatPCLocal
        && (::testing::get<0>(GetParam()) == 2 || ::testing::get<0>(GetParam()) == 3)) {
      warningQuda("Only fixed precision supported for MatPCDagMatPCLocal operator, skipping...");
      return true;
    }

    if (::testing::get<2>(GetParam()) > 0 && dslash_test_wrapper.test_split_grid) { return true; }

    return false;
  }

public:
  virtual ~DslashTest() { }
  virtual void SetUp() {
    int prec = ::testing::get<0>(GetParam());
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if (skip()) GTEST_SKIP();

    int value = ::testing::get<2>(GetParam());
    for(int j=0; j < 4;j++){
      if (value &  (1 << j)){
        commDimPartitionedSet(j);
      }

    }
    updateR();

    dslash_test_wrapper.init(argc_copy, argv_copy, prec, recon);
    display_test_info(prec, recon);
  }

  virtual void TearDown()
  {
    if (skip()) GTEST_SKIP();
    dslash_test_wrapper.end();
    commDimPartitionedReset();
  }

  static void SetUpTestCase() { initQuda(device_ordinal); }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase() {
    endQuda();
  }

};

TEST_P(DslashTest, verify)
{
  dslash_test_wrapper.dslashRef();

  dslash_test_wrapper.dslashCUDA(1);
  dslash_test_wrapper.dslashCUDA(2);

  double deviation = 1e-14;
  if (dslash_test_wrapper.test_split_grid) {
    for (int n = 0; n < dslash_test_wrapper.num_src; n++) {
      double norm2_cpu = blas::norm2(*dslash_test_wrapper.spinorRef);
      double norm2_cpu_cuda = blas::norm2(*dslash_test_wrapper.vp_spinorOut[n]);
      printfQuda("Result: CPU = %f, CPU-QUDA = %f\n", norm2_cpu, norm2_cpu_cuda);
      deviation = std::max(deviation, pow(10, -(double)(cpuColorSpinorField::Compare(*dslash_test_wrapper.spinorRef, *dslash_test_wrapper.vp_spinorOut[n]))));
    }
  } else {

    if (!dslash_test_wrapper.transfer) *dslash_test_wrapper.spinorOut = *dslash_test_wrapper.cudaSpinorOut;

    double norm2_cpu = blas::norm2(*dslash_test_wrapper.spinorRef);
    double norm2_cpu_cuda = blas::norm2(*dslash_test_wrapper.spinorOut);
    if (!dslash_test_wrapper.transfer) {
      double norm2_cuda = blas::norm2(*dslash_test_wrapper.cudaSpinorOut);
      printfQuda("Results: CPU = %f, CUDA=%f, CPU-CUDA = %f\n", norm2_cpu, norm2_cuda, norm2_cpu_cuda);
    } else {
      printfQuda("Result: CPU = %f, CPU-QUDA = %f\n", norm2_cpu, norm2_cpu_cuda);
    }
    deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*dslash_test_wrapper.spinorRef, *dslash_test_wrapper.spinorOut)));
  }

  double tol = getTolerance(dslash_test_wrapper.inv_param.cuda_prec);
  // If we are using tensor core we tolerate a greater deviation
  if (dslash_type == QUDA_MOBIUS_DWF_DSLASH && dslash_test_wrapper.dtest_type == dslash_test_type::MatPCDagMatPCLocal) tol *= 10;
  if (dslash_test_wrapper.gauge_param.reconstruct == QUDA_RECONSTRUCT_8 && dslash_test_wrapper.inv_param.cuda_prec >= QUDA_HALF_PRECISION)
    tol *= 10; // if recon 8, we tolerate a greater deviation

  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

TEST_P(DslashTest, benchmark)
{  
  dslash_test_wrapper.dslashCUDA(1); // warm-up run
  if (!dslash_test_wrapper.transfer) dslash_test_wrapper.dirac->Flops();
  auto dslash_time = dslash_test_wrapper.dslashCUDA(niter);

  printfQuda("%fus per kernel call\n", 1e6 * dslash_time.event_time / niter);
  // FIXME No flops count for twisted-clover yet
  unsigned long long flops = 0;
  if (!dslash_test_wrapper.transfer) flops = dslash_test_wrapper.dirac->Flops();
  double gflops = 1.0e-9 * flops / dslash_time.event_time;
  printfQuda("GFLOPS = %f\n", gflops);
  RecordProperty("Gflops", std::to_string(gflops));
  RecordProperty("Halo_bidirectitonal_BW_GPU", 1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.event_time);
  RecordProperty("Halo_bidirectitonal_BW_CPU", 1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.cpu_time);
  RecordProperty("Halo_bidirectitonal_BW_CPU_min", 1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_max);
  RecordProperty("Halo_bidirectitonal_BW_CPU_max", 1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_min);
  RecordProperty("Halo_message_size_bytes", 2 * dslash_test_wrapper.cudaSpinor->GhostBytes());

  printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for aggregate "
             "message size %lu bytes\n",
             1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.event_time,
             1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.cpu_time,
             1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_max,
             1.0e-9 * 2 * dslash_test_wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_min, 2 * dslash_test_wrapper.cudaSpinor->GhostBytes());
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;
  // command line options
  auto app = make_app();
  app->add_option("--test", dslash_test_wrapper.dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
  add_split_grid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  dslash_test_wrapper.num_src = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
  dslash_test_wrapper.test_split_grid = dslash_test_wrapper.num_src > 1;

  // The 'SetUp()' method of the Google Test class from which DslashTest
  // in derived has no arguments, but QUDA's implementation requires the
  // use of argc and argv to set up the test via the function 'init'.
  // As a workaround, we declare argc_copy and argv_copy as global pointers
  // so that they are visible inside the 'init' function.
  argc_copy = argc;
  argv_copy = argv;

  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
  test_rc = RUN_ALL_TESTS();

  finalizeComms();
  return test_rc;
}

std::string getdslashtestname(testing::TestParamInfo<::testing::tuple<int, int, int>> param)
{
  const int prec = ::testing::get<0>(param.param);
  const int recon = ::testing::get<1>(param.param);
  const int part = ::testing::get<2>(param.param);
  std::stringstream ss;
  // std::cout << "getdslashtestname" << get_dslash_str(dslash_type) << "_" << prec_str[prec] << "_r" << recon <<
  // "_partition" << part << std::endl; ss << get_dslash_str(dslash_type) << "_";
  ss << prec_str[prec];
  ss << "_r" << recon;
  ss << "_partition" << part;
  return ss.str();
}

#ifdef MULTI_GPU
INSTANTIATE_TEST_SUITE_P(QUDA, DslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 Range(0, 16)),
                         getdslashtestname);
#else
INSTANTIATE_TEST_SUITE_P(QUDA, DslashTest,
                         Combine(Range(0, 4),
                                 ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                 ::testing::Values(0)),
                         getdslashtestname);
#endif
