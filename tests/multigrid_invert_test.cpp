#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits>

// QUDA headers
#include <quda.h>
#include <util_quda.h>

// External headers
#include <host_utils.h>
#include <misc.h>
#include <command_line_params.h>
#include <dslash_reference.h>

#include <staggered_gauge_utils.h>


namespace quda {
  extern void setTransferGPU(bool);
}

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("MG parameters\n");
  printfQuda(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) {
    printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
    printfQuda(" - level %d number of pre-smoother applications %d\n", i+1, nu_pre[i]);
    printfQuda(" - level %d number of post-smoother applications %d\n", i+1, nu_post[i]);
  }

  printfQuda("Outer solver paramers\n");
  printfQuda(" - pipeline = %d\n", pipeline);

  printfQuda("Eigensolver parameters\n");
  for (int i = 0; i < mg_levels; i++) {
    if (low_mode_check || mg_eig[i]) {
      printfQuda(" - level %d solver mode %s\n", i + 1, get_eig_type_str(mg_eig_type[i]));
      printfQuda(" - level %d spectrum requested %s\n", i + 1, get_eig_spectrum_str(mg_eig_spectrum[i]));
      printfQuda(" - level %d number of eigenvectors requested nConv %d\n", i + 1, nvec[i]);
      printfQuda(" - level %d size of eigenvector search space %d\n", i + 1, mg_eig_nEv[i]);
      printfQuda(" - level %d size of Krylov space %d\n", i + 1, mg_eig_nKr[i]);
      printfQuda(" - level %d solver tolerance %e\n", i + 1, mg_eig_tol[i]);
      printfQuda(" - level %d convergence required (%s)\n", i + 1, mg_eig_require_convergence[i] ? "true" : "false");
      printfQuda(" - level %d Operator: daggered (%s) , norm-op (%s)\n", i + 1, mg_eig_use_dagger[i] ? "true" : "false",
                 mg_eig_use_normop[i] ? "true" : "false");
      if (mg_eig_use_poly_acc[i]) {
        printfQuda(" - level %d Chebyshev polynomial degree %d\n", i + 1, mg_eig_poly_deg[i]);
        printfQuda(" - level %d Chebyshev polynomial minumum %e\n", i + 1, mg_eig_amin[i]);
        if (mg_eig_amax[i] <= 0)
          printfQuda(" - level %d Chebyshev polynomial maximum will be computed\n", i + 1);
        else
          printfQuda(" - level %d Chebyshev polynomial maximum %e\n", i + 1, mg_eig_amax[i]);
      }
      printfQuda("\n");
    }
  }
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return ;
}

int main(int argc, char **argv)
{
  // Parse command line options
  auto app = make_app();
  add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set some default values for precisions and solve types
  // if none are passed through the command line
  setQudaDefaultPrecs();
  if(inv_multigrid) {
    setQudaDefaultMgTestParams();    
    setQudaDefaultMgSolveTypes();
  }
  
  // Initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);
  
  // Only these fermions are supported in this file
  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  // Only these solve types are supported in this file
  if (inv_multigrid && solve_type != QUDA_DIRECT_SOLVE && solve_type != QUDA_DIRECT_PC_SOLVE) {
    printfQuda("\nsolve_type %d not supported. Please use QUDA_DIRECT_SOLVE or QUDA_DIRECT_PC_SOLVE\n\n", solve_type);
    exit(0);
  }
  
  // Set QUDA's internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];
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
  
  // All parameters have been set. Display the parameters via stdout
  display_test_info();

  // Initialize the QUDA library
  initQuda(device);

  // Set some dimension parameters for the host routines.
  setDims(gauge_param.X);
  setSpinorSiteSize(24);

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);
  
  // Allocate host side memory for clover terms if needed.
  //----------------------------------------------------------------------------
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
  
  // Compute plaquette as a sanity check
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();
  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];

  // Vector construct START
  //-----------------------------------------------------------------------------------
  ColorSpinorField *in;
  ColorSpinorField *out;
  ColorSpinorField *check;
  ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  cs_param.location = QUDA_CPU_FIELD_LOCATION;
  in = quda::ColorSpinorField::Create(cs_param);
  out = quda::ColorSpinorField::Create(cs_param);
  check = quda::ColorSpinorField::Create(cs_param);
  // Vector construct END
  //-----------------------------------------------------------------------------------

  // QUDA invert test BEGIN
  //----------------------------------------------------------------------------
  for (int i = 0; i < Nsrc; i++) {
    constructRandomSpinorSource(in->V(), 4, 3, inv_param.cpu_prec, gauge_param.X, *rng);
    invertQuda(out->V(), in->V(), &inv_param);
    time[i] = inv_param.secs;
    gflops[i] = inv_param.gflops / inv_param.secs;
    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
               inv_param.gflops / inv_param.secs);
  }
  // QUDA invert test COMPLETE
  //----------------------------------------------------------------------------

  rng->Release();
  delete rng;

  // free the multigrid solver
  if(inv_multigrid) destroyMultigridQuda(mg_preconditioner);

  // Compute performance statistics
  if (Nsrc > 1) performanceStats(time, gflops);
  delete[] time;
  delete[] gflops;

  // Perform host side verification of inversion if requested
  if (verify_results) {
    verifyInversion(out->V(), in->V(), check->V(), gauge_param, inv_param, gauge, clover, clover_inv);
  }
  
  // Clean up memory allocations
  delete in;
  delete out;
  delete check;
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
