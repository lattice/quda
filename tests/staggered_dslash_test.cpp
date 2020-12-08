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

#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <gauge_field.h>
#include <unitarization_links.h>

#include "dslash_test_helpers.h"
#include <assert.h>
#include <gtest/gtest.h>

#include "staggered_dslash_test_utils.h"

using namespace quda;

StaggeredDslashTestWrapper wrapper;

static int dslashTest()
{

  bool failed = false;

  // return code for google test
  int test_rc = 0;
  wrapper.init_test();

  int attempts = 1;

  for (int i=0; i<attempts; i++) {

    { // warm-up run
      printfQuda("Tuning...\n");
      wrapper.dslashCUDA(1);
    }
    printfQuda("Executing %d kernel loops...", niter);

    // reset flop counter
    wrapper.dirac->Flops();

    DslashTime dslash_time = wrapper.dslashCUDA(niter);

    *wrapper.spinorOut = *wrapper.cudaSpinorOut;

    printfQuda("%fus per kernel call\n", 1e6*dslash_time.event_time / niter);
    wrapper.staggeredDslashRef();

    double spinor_ref_norm2 = blas::norm2(*wrapper.spinorRef);
    double spinor_out_norm2 = blas::norm2(*wrapper.spinorOut);

    // Catching nans is weird.
    if (std::isnan(spinor_ref_norm2)) { failed = true; }
    if (std::isnan(spinor_out_norm2)) { failed = true; }

    unsigned long long flops = wrapper.dirac->Flops();
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/dslash_time.event_time);

    if (niter > 2) { // only print this if valid
      printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for "
                 "aggregate message size %lu bytes\n",
                 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.event_time,
                 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() * niter / dslash_time.cpu_time,
                 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_max,
                 1.0e-9 * 2 * wrapper.cudaSpinor->GhostBytes() / dslash_time.cpu_min, 2 * wrapper.cudaSpinor->GhostBytes());
    }

    double cuda_spinor_out_norm2 = blas::norm2(*wrapper.cudaSpinorOut);
    printfQuda("Results: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", spinor_ref_norm2, cuda_spinor_out_norm2, spinor_out_norm2);

    if (verify_results) {
      test_rc = RUN_ALL_TESTS();
      if (test_rc != 0 || failed) warningQuda("Tests failed");
    }
  }
  wrapper.end();

  return test_rc;
}

TEST(dslash, verify) {
  double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*wrapper.spinorRef, *wrapper.spinorOut)));
  double tol = getTolerance(prec);
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %s           %d       %d/%d/%d        %d \n", get_prec_str(prec), get_recon_str(link_recon),
      get_string(dtest_type_map, dtest_type).c_str(), dagger, xdim, ydim, zdim, tdim);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
      dimPartitioned(3));
}

int main(int argc, char **argv)
{
  // hack for loading gauge fields
  wrapper.argc_copy = argc;
  wrapper.argv_copy = argv;

  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  // command line options
  auto app = make_app();
  app->add_option("--test", dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  for (int d = 0; d < 4; d++) {
    if (dim_partitioned[d]) { commDimPartitionedSet(d); }
  }
  updateR();

  initQuda(device_ordinal);

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

  // Sanity check: if you pass in a gauge field, want to test the asqtad/hisq dslash,
  // and don't ask to build the fat/long links... it doesn't make sense.
  if (strcmp(latfile,"") && !compute_fatlong && dslash_type == QUDA_ASQTAD_DSLASH) {
    errorQuda("Cannot load a gauge field and test the ASQTAD/HISQ operator without setting \"--compute-fat-long true\".\n");
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

  // If we're building fat/long links, there are some
  // tests we have to skip.
  if (dslash_type == QUDA_ASQTAD_DSLASH && compute_fatlong) {
    if (prec < QUDA_SINGLE_PRECISION /* half */) { errorQuda("Half precision unsupported in fat/long compute"); }
  }

  display_test_info();

  // return result of RUN_ALL_TESTS
  int test_rc = dslashTest();

  endQuda();

  finalizeComms();

  return test_rc;
}
