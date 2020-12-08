#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include <quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>

#include "dslash_test_helpers.h"
#include <assert.h>
#include <gtest/gtest.h>

#include "staggered_dslash_test_utils.h"

using namespace quda;

StaggeredDslashTestWrapper wrapper;

bool gauge_loaded = false;

const char *prec_str[] = {"quarter", "half", "single", "double"};
const char *recon_str[] = {"r18", "r13", "r9"};

void init(int precision, QudaReconstructType link_recon, int partition)
{
  wrapper.init_ctest(precision, link_recon, partition);
}

void end()
{
  wrapper.end();
}

void display_test_info(int precision, QudaReconstructType link_recon)
{
  auto prec = precision == 2 ? QUDA_DOUBLE_PRECISION : precision  == 1 ? QUDA_SINGLE_PRECISION : QUDA_HALF_PRECISION;

  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %s           %d       %d/%d/%d        %d \n", get_prec_str(prec), get_recon_str(link_recon),
             get_string(dtest_type_map, dtest_type).c_str(), dagger, xdim, ydim, zdim, tdim);
}

using ::testing::TestWithParam;
using ::testing::Bool;
using ::testing::Values;
using ::testing::Range;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Bool;
using ::testing::Values;
using ::testing::Range;
using ::testing::Combine;

class StaggeredDslashTest : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {
protected:
  ::testing::tuple<int, int, int> param;

  bool skip()
  {
    QudaReconstructType recon = static_cast<QudaReconstructType>(::testing::get<1>(GetParam()));

    if ((QUDA_PRECISION & getPrecision(::testing::get<0>(GetParam()))) == 0
        || (QUDA_RECONSTRUCT & getReconstructNibble(recon)) == 0) {
      return true;
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH && compute_fatlong
        && (::testing::get<0>(GetParam()) == 0 || ::testing::get<0>(GetParam()) == 1)) {
      warningQuda("Fixed precision unsupported in fat/long compute, skipping...");
      return true;
    }

    if (dslash_type == QUDA_ASQTAD_DSLASH && compute_fatlong && (getReconstructNibble(recon) & 1)) {
      warningQuda("Reconstruct 9 unsupported in fat/long compute, skipping...");
      return true;
    }

    if (dslash_type == QUDA_LAPLACE_DSLASH && (::testing::get<0>(GetParam()) == 0 || ::testing::get<0>(GetParam()) == 1)) {
      warningQuda("Fixed precision unsupported for Laplace operator, skipping...");
      return true;
    }
    return false;
  }

public:
  virtual ~StaggeredDslashTest() { }
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

    init(prec, recon, value);
    display_test_info(prec, recon);
  }

  virtual void TearDown()
  {
    if (skip()) GTEST_SKIP();
    end();
  }

  static void SetUpTestCase() { initQuda(device_ordinal); }

  // Per-test-case tear-down.
  // Called after the last test in this test case.
  // Can be omitted if not needed.
  static void TearDownTestCase() { endQuda(); }
};

 TEST_P(StaggeredDslashTest, verify) {
   double deviation = 1.0;
   double tol = getTolerance(wrapper.inv_param.cuda_prec);

   bool failed = false; // for the nan catch

   // check for skip_kernel
   if (wrapper.spinorRef != nullptr) {

     { // warm-up run
       // printfQuda("Tuning...\n");
       wrapper.dslashCUDA(1);
     }

     wrapper.dslashCUDA(2);

     *wrapper.spinorOut = *wrapper.cudaSpinorOut;

     wrapper.staggeredDslashRef();

     double spinor_ref_norm2 = blas::norm2(*wrapper.spinorRef);
     double spinor_out_norm2 = blas::norm2(*wrapper.spinorOut);

     // for verification
     // printfQuda("\n\nCUDA: %f\n\n", ((double*)(spinorOut->V()))[0]);
     // printfQuda("\n\nCPU:  %f\n\n", ((double*)(spinorRef->V()))[0]);

     // Catching nans is weird.
     if (std::isnan(spinor_ref_norm2)) { failed = true; }
     if (std::isnan(spinor_out_norm2)) { failed = true; }

     double cuda_spinor_out_norm2 = blas::norm2(*wrapper.cudaSpinorOut);
     printfQuda("Results: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", spinor_ref_norm2, cuda_spinor_out_norm2, spinor_out_norm2);
     deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*wrapper.spinorRef, *wrapper.spinorOut)));
     if (failed) { deviation = 1.0; }
   }
    ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
  }

TEST_P(StaggeredDslashTest, benchmark) {

  { // warm-up run
    // printfQuda("Tuning...\n");
    wrapper.dslashCUDA(1);
  }

  // reset flop counter
  wrapper.dirac->Flops();

  DslashTime dslash_time = wrapper.dslashCUDA(niter);

  *wrapper.spinorOut = *wrapper.cudaSpinorOut;

  printfQuda("%fus per kernel call\n", 1e6 * dslash_time.event_time / niter);

  unsigned long long flops = wrapper.dirac->Flops();
  double gflops = 1.0e-9 * flops / dslash_time.event_time;
  printfQuda("GFLOPS = %f\n", gflops);
  RecordProperty("Gflops", std::to_string(gflops));

  RecordProperty("Halo_bidirectitonal_BW_GPU", 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.event_time);
  RecordProperty("Halo_bidirectitonal_BW_CPU", 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.cpu_time);
  RecordProperty("Halo_bidirectitonal_BW_CPU_min", 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_max);
  RecordProperty("Halo_bidirectitonal_BW_CPU_max", 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_min);
  RecordProperty("Halo_message_size_bytes", 2 * wrapper.cudaSpinor->GhostBytes());

  printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for aggregate "
             "message size %lu bytes\n",
      1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.event_time,
      1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.cpu_time,
      1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_max,
      1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_min, 2 * wrapper.cudaSpinor->GhostBytes());
}

  int main(int argc, char **argv)
  {
    // hack for loading gauge fields
    wrapper.argc_copy = argc;
    wrapper.argv_copy = argv;

    // initialize CPU field backup
    int pmax = 1;
#ifdef MULTI_GPU
    pmax = 16;
#endif
    for (int p = 0; p < pmax; p++) {
      for (int d = 0; d < 4; d++) {
        wrapper.qdp_fatlink_cpu_backup[p][d] = nullptr;
        wrapper.qdp_longlink_cpu_backup[p][d] = nullptr;
        wrapper.qdp_inlink_backup[p][d] = nullptr;
      }
    }

    // initalize google test
    ::testing::InitGoogleTest(&argc, argv);
    auto app = make_app();
    app->add_option("--test", dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
    try {
      app->parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      return app->exit(e);
    }
    initComms(argc, argv, gridsize_from_cmdline);

    // Ensure gtest prints only from rank 0
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

    // Only these fermions are supported in this file. Ensure a reasonable default,
    // ensure that the default is improved staggered
    if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
      printfQuda("dslash_type %s not supported, defaulting to %s\n", get_dslash_str(dslash_type),
                 get_dslash_str(QUDA_ASQTAD_DSLASH));
      dslash_type = QUDA_ASQTAD_DSLASH;
    }

    // Sanity check: if you pass in a gauge field, want to test the asqtad/hisq dslash, and don't
    // ask to build the fat/long links... it doesn't make sense.
    if (strcmp(latfile,"") && !compute_fatlong && dslash_type == QUDA_ASQTAD_DSLASH) {
      errorQuda("Cannot load a gauge field and test the ASQTAD/HISQ operator without setting \"--compute-fat-long true\".\n");
      compute_fatlong = true;
    }

    // Set n_naiks to 2 if eps_naik != 0.0
    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      if (eps_naik != 0.0) {
        if (compute_fatlong) {
          n_naiks = 2;
          printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
        } else {
          eps_naik = 0.0;
          printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
        }
      } else {
        printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
      }
    }

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    if (dtest_type != dslash_test_type::Mat) {
      errorQuda("Test type %s is not supported for the Laplace operator.\n", get_string(dtest_type_map, dtest_type).c_str());
    }
  }

    // return result of RUN_ALL_TESTS
    int test_rc = RUN_ALL_TESTS();

    // Clean up loaded gauge field
    for (int dir = 0; dir < 4; dir++) {
      if (wrapper.qdp_inlink[dir] != nullptr) { free(wrapper.qdp_inlink[dir]); wrapper.qdp_inlink[dir] = nullptr; }
    }

    // Clean up per-partition backup
    for (int p = 0; p < pmax; p++) {
      for (int d = 0; d < 4; d++) {
        if (wrapper.qdp_inlink_backup[p][d] != nullptr) { free(wrapper.qdp_inlink_backup[p][d]); wrapper.qdp_inlink_backup[p][d] = nullptr; }
        if (wrapper.qdp_fatlink_cpu_backup[p][d] != nullptr) {
          free(wrapper.qdp_fatlink_cpu_backup[p][d]);
          wrapper.qdp_fatlink_cpu_backup[p][d] = nullptr;
        }
        if (wrapper.qdp_longlink_cpu_backup[p][d] != nullptr) {
          free(wrapper.qdp_longlink_cpu_backup[p][d]);
          wrapper.qdp_longlink_cpu_backup[p][d] = nullptr;
        }
      }
    }

    finalizeComms();

    return test_rc;
  }

  std::string getstaggereddslashtestname(testing::TestParamInfo<::testing::tuple<int, int, int>> param){
   const int prec = ::testing::get<0>(param.param);
   const int recon = ::testing::get<1>(param.param);
   const int part = ::testing::get<2>(param.param);
   std::stringstream ss;
   // ss << get_dslash_str(dslash_type) << "_";
   ss << prec_str[prec];
   ss << "_r" << recon;
   ss << "_partition" << part;
   return ss.str();
  }

#ifdef MULTI_GPU
  INSTANTIATE_TEST_SUITE_P(QUDA, StaggeredDslashTest,
                           Combine(Range(0, 4),
                                   ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                   Range(0, 16)),
                           getstaggereddslashtestname);
#else
  INSTANTIATE_TEST_SUITE_P(QUDA, StaggeredDslashTest,
                           Combine(Range(0, 4),
                                   ::testing::Values(QUDA_RECONSTRUCT_NO, QUDA_RECONSTRUCT_12, QUDA_RECONSTRUCT_8),
                                   ::testing::Values(0)),
                           getstaggereddslashtestname);
#endif
