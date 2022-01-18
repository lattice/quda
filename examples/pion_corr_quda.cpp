// QUDA headers
#include <quda.h>
#include <host_utils.h>
#include <command_line_params.h>

// C++
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv)
{

  // ---- START INITIALIZATOIN ---- //
  //
  // helper functions, populate multgrip params with default params
  setQudaDefaultMgTestParams();
  auto app = make_app();   // Parameter class that reads commandline arguments. It modifies global variables.
  // functions in test/utils/command_line_params.cpp
  add_multigrid_option_group(app);
  // add_eigen_option_group(app); deflation
  add_contraction_option_group(app);
  add_su3_option_group(app);
  // add the option we want 
  add_laph_option_group(app);

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setQudaPrecisions();

  // init QUDA
  initComms(argc, argv, gridsize_from_cmdline);

  // Run-time parameter checks
  {
    if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
      if (comm_rank() == 0) printf("dslash_type %d not supported\n", dslash_type);
      exit(0);
    }
    if (inv_multigrid) {
      // Only these fermions are supported with MG
      if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
        if (comm_rank() == 0) printf("dslash_type %d not supported for MG\n", dslash_type);
        exit(0);
      }
      // Only these solve types are supported with MG
      if (solve_type != QUDA_DIRECT_SOLVE && solve_type != QUDA_DIRECT_PC_SOLVE) {
        if (comm_rank() == 0) printf("Solve_type %d not supported with MG. Please use QUDA_DIRECT_SOLVE or QUDA_DIRECT_PC_SOLVE\n\n",
                                     solve_type);
        exit(0);
      }
    }
  }

  initQuda(device_ordinal);
  
  // ---- FINISH INITIALIZATOIN ---- //

  if (comm_rank() == 0) printf("\n\n\n a fresh program begins! \n\n\n");
  
  
  
  // ---- START CLEAN UP ---- //
  
  endQuda();
  finalizeComms();
  
  // ---- FINISH CLEAN UP ---- //

  return 0;
}

