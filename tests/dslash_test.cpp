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

// google test frame work
#include <gtest/gtest.h>

#include "dslash_test_utils.h"

using namespace quda;

DslashTestWrapper wrapper;

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    recon   dtest_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension "
      "dslash_type    niter\n");
  printfQuda("%6s   %2s       %s           %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n",
      get_prec_str(prec), get_recon_str(link_recon), get_string(dtest_type_map, wrapper.dtest_type).c_str(),
      get_matpc_str(matpc_type), dagger, xdim, ydim, zdim, tdim, Lsdim, get_dslash_str(dslash_type), niter);
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3));

  if (wrapper.test_split_grid) {
    printfQuda("Testing with split grid: %d  %d  %d  %d\n", grid_partition[0], grid_partition[1], grid_partition[2],
        grid_partition[3]);
  }
}


TEST(dslash, verify) {
  double deviation = wrapper.verify();
  double tol = getTolerance(wrapper.inv_param.cuda_prec);
  // If we are using tensor core we tolerate a greater deviation
  if (dslash_type == QUDA_MOBIUS_DWF_DSLASH && wrapper.dtest_type == dslash_test_type::MatPCDagMatPCLocal) tol *= 10;
  if (wrapper.gauge_param.reconstruct == QUDA_RECONSTRUCT_8) tol *= 10; // if recon 8, we tolerate a greater deviation

  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  // return code for google test
  int test_rc = 0;
  // command line options
  auto app = make_app();
  app->add_option("--test", wrapper.dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
  add_eofa_option_group(app);
  add_split_grid_option_group(app);

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
  wrapper.init_test(argc, argv);

  display_test_info();

  int attempts = 1;
  wrapper.dslashRef();
  for (int i=0; i<attempts; i++) {
    wrapper.run_test(2);
    if (verify_results) {
      ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
      if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

      test_rc = RUN_ALL_TESTS();
      if (test_rc != 0) warningQuda("Tests failed");
    }
  }    
  wrapper.end();

  endQuda();

  finalizeComms();
  return test_rc;
}
