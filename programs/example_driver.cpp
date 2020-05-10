#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <command_line_params.h>
#include <host_utils.h>
#include <misc.h>

#include <stoch_laph_quark_smear.h>

using namespace quda;

int main(int argc, char **argv)
{
  // Set up QUDA
  //----------------------------------------------------------------------------------
  setQudaDefaultMgTestParams();

  // QUDA's command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  
  // Here we demonstrate how to add custom command line parameters.
  //---------------------------------------------------------------
  // This will govern what type of source we wish to invert.
  int source_type = 0; // Defaulted to point source.
  CLI::TransformPairs<int> source_type_map {{"point", 0}, {"random", 1}, {"diluted", 2}};
  app->add_option("--source-type", source_type, "Example programs Source type")->transform(CLI::CheckedTransformer(source_type_map));

  // This will govern how many sources we wish to invert.
  int n_sources = 12; // Defaulted to 12
  app->add_option("--n-sources", n_sources, "Example programs number of sources");

  // Parse the command line options
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }  

  // Sanity check for spin/color dilution
  if(n_sources%12 != 0 && source_type == 2) {
    errorQuda("Must have some multiple of nSpin * nColor = 12 to use dilution. % given.", n_sources);
  }

  // Sanity check for preconditioner
  if(inv_deflate && inv_multigrid) {
    printfQuda("Error: Cannot use both deflation and multigrid preconditioners on top level solve.\n");
    exit(0);
  }

  // Set some default values for precisions and solve types
  // if none are passed through the command line
  setQudaDefaultPrecs();
  if(inv_multigrid) {
    setQudaDefaultMgTestParams();    
    setQudaDefaultMgSolveTypes();
  }
  
  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);
  
  // Set QUDA's internal parameter structures
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];
  QudaEigParam eig_param = newQudaEigParam();
  if(inv_multigrid) {
    setMultigridInvertParam(inv_param);
    // Set sub structures
    mg_param.invert_param = &mg_inv_param;
    for (int i = 0; i < mg_levels; i++) {
      if (mg_eig[i]) {
	mg_eig_param[i] = newQudaEigParam();
	setMultigridEigParam(mg_eig_param[i], i);
	mg_param.eig_param[i] = &mg_eig_param[i];
      } else {
	mg_param.eig_param[i] = nullptr;
      }
    }
    // Set MG
    setMultigridParam(mg_param);  
  }
  else {
    setInvertParam(inv_param);
  }
  
  if(inv_deflate) {
    setEigParam(eig_param);
    inv_param.eig_param = &eig_param;
  } else {
    inv_param.eig_param = nullptr;
  }

  // Initialize the QUDA library
  initQuda(device);

  // Initialise QUDA's host RNG
  initRand();
  setSpinorSiteSize(24);
  setDims(gauge_param.X);
  display_driver_info();

  // Allocate host side memory for the gauge field.
  //-----------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Allocate host side memory for clover terms if needed.
  //------------------------------------------------------
  void *clover = nullptr;
  void *clover_inv = nullptr;
  // Allocate space on the host (always best to allocate and free in the same scope)
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover = malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, inv_param);
    if(inv_multigrid) {
      // This line ensures that if we need to construct the clover inverse (in either the smoother or the solver) we do so
      if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) {
	inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
      }
    }
    // Load the clover terms to the device
    loadCloverQuda(clover, clover_inv, &inv_param);
    if(inv_multigrid) {
      // Restore actual solve_type we want to do
      inv_param.solve_type = solve_type;
    }
  }

  // Now QUDA is initialised and the fields are loaded, we may setup the preconditioner
  void *mg_preconditioner = nullptr;
  if(inv_multigrid) {
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
  }
  //----------------------------------------------------------------------------------


  // We now emulate an external application interfacing with QUDA
  //----------------------------------------------------------------------------------  
  // Host side data for sources/quarks
  // V is a global variable set by QUDA in the setDims(gauge_param.X);
  // it is the MPI Local volume, not the Global volume.

  // query local MPI coordinates
  int t_rank = comm_coord(3);
  int z_rank = comm_coord(2);
  int y_rank = comm_coord(1);
  int x_rank = comm_coord(0);
  printfQuda("t_rank = %d, z_rank = %d, y_rank = %d, x_rank = %d\n",
	     t_rank, z_rank, y_rank, x_rank);
  
  // query MPI dimensions
  int t_dimMPI = comm_dim(3);
  int z_dimMPI = comm_dim(2);
  int y_dimMPI = comm_dim(1);
  int x_dimMPU = comm_dim(0);
  printfQuda("t_dimMPI = %d, z_dimMPI = %d, y_dimMPI = %d, x_dimMPI = %d\n",
	     t_dimMPI, z_dimMPI, y_dimMPI, x_dimMPI);
  
  // deduce spatial volume
  int spatial_vol = (z_dimMPI * zdim) * (y_dimMPI * ydim) * (x_dimMPI * xdim);
  printfQuda("Spatial volume = %d");
  
  // We will write our sources in lexicographical ordering. That is
  // Space - Spin - Color - ReIm with the fastest on the right.
  // The spacetime ordering is T - Z - Y - X with X running the fastest. 
 
  void **host_quarks = (void **)malloc(n_sources * sizeof(void *));
  for (int i = 0; i < n_sources; i++) {
    host_quarks[i] = (void *)malloc(V * 24 * sizeof(double));
    if(source_type == 0) {
      // make a point source at (0,0,0,0)
      for(int j=0; j<V*24; j++) {
	if(j==0 && t_rank == 0) ((double*)host_quarks[i])[j] = 1.0;
	else ((double*)host_quarks[i])[j] = 0.0;
      }
    } else if(j==1) {
      // make a random source on the t=0 time slice
      for(int j=0; j<V*24; j++) {
	if(j < ) ((double*)host_quarks[i])[j] = 1.0;
	else ((double*)host_quarks[i])[j] = 0.0;      
      

      
  // local lattice dims
  int X[4];
  for (int i=0; i<4; i++) X[i] = gauge_param.X[i];

  programExample(host_quarks, inv_param, X);
  //----------------------------------------------------------------------------------  
  
  // free the multigrid solver
  if(inv_multigrid) destroyMultigridQuda(mg_preconditioner);
  
  // Clean up memory allocations
  for (int i = 0; i < n_sources; i++) {
    free(host_quarks[i]);
  }
  free(host_quarks);
  for (int i = 0; i < n_evecs; i++) {
    free(host_evecs[i]);
  }
  free(host_evecs);
  free(host_sinks);
  free(host_noise);
  
  freeGaugeQuda();
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    freeCloverQuda();
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  // finalize the QUDA library
  endQuda();
  finalizeComms();
  
  return 0;
}
