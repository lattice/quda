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

// Experimental routine for LAPH quark smearing.
// This routine accepts the noise data from a single stochastic noise vector in noise array, the T 3D
// eigenvectors in eigen_vecs, and returns the smeared quarks in quark. Below is a description
// of the workflow.
//
// 1. The eigenvector array we use is a matrix with nColor * L^3 * T rows and 
// nEigenVec/dilution_scheme columns. This is right multiplied by a matrix of stochastic noise coefficents
// with nEigenVec/dilution_scheme rows and nSpin columns. The result is a matrix with nSpin columns and 
// nColor * L^3 * T rows. Each column contains the data for one source.
//
// 2. We copy the data in each column to a QUDA ColorSpinorField object of length nSpin * nColor * L^3 * T, 
// with the spin elements populated/zeroed out as dictated by the dilution scheme.
// 
// 3. We pass each of these sources to the inverter, and then populated the apprpriate quark 
// array with the solution.
// 
// 4. We then repopulate the eigenvector array and stochastic noise coefficents with the next set of 
// LapH dilution data and repeat steps 1,2,3.

int main(int argc, char **argv) {

  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }  

  if(inv_deflate && inv_multigrid) {
    printfQuda("Error: Cannot use both deflation and multigrid preconditioners on top level solve.\n");
    exit(0);
  }
  
  // Move these
  if (multishift > 1) {
    // set a correct default for the multi-shift solver
    solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
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
  
  // Set QUDA's internal parameters
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
  
  // Make up some junk data
  int dil_scheme = 16;
  int n_evecs = 400;
  // Deduce the number of eigenvectors to be used per dilution.
  if (n_evecs%dil_scheme != 0) {
    errorQuda("Number of eigen vectors passed %d is not a multiple of the dilution scheme %d", n_evecs, dil_scheme);
  }
  int n_dil_vecs = n_evecs/dil_scheme;
  int n_sources = 4 * dil_scheme;

  // Host side data for sources/quarks
  void **host_quarks = (void **)malloc(n_sources * sizeof(void *));
  for (int i = 0; i < n_sources; i++) {
    host_quarks[i] = (void *)malloc(V * 24 * inv_param.cpu_prec);
  }
  // Parameter object describing the sources and smeared quarks
  ColorSpinorParam cpu_quark_param(host_quarks[0], inv_param, gauge_param.X, false, QUDA_CPU_FIELD_LOCATION);
  cpu_quark_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
  // QUDA style wrappers around the host data
  std::vector<ColorSpinorField*> quarks;
  quarks.reserve(n_sources);
  for (int i = 0; i < n_sources; i++) {
    cpu_quark_param.v = host_quarks[i];
    quarks.push_back(ColorSpinorField::Create(cpu_quark_param));
  }  

  // Host side data for eigenvecs
  void **host_evecs = (void **)malloc(n_evecs * sizeof(void *));
  for (int i = 0; i < n_evecs; i++) {
    host_evecs[i] = (void *)malloc(V * 6 * inv_param.cpu_prec);
  }
  // Parameter object describing evecs
  ColorSpinorParam cpu_evec_param(host_evecs[0], inv_param, gauge_param.X, false, QUDA_CPU_FIELD_LOCATION);
  // Switch to spin 1
  cpu_evec_param.nSpin = 1;
  // QUDA style wrappers around the host data
  std::vector<ColorSpinorField*> evecs;
  evecs.reserve(n_evecs);
  for (int i = 0; i < n_evecs; i++) {
    cpu_evec_param.v = host_evecs[i];
    evecs.push_back(ColorSpinorField::Create(cpu_evec_param));
  }  

  // In the real application, the QUDA ColorSpinorField objects would alias real host data. In the
  // meantime, we fill the arrays with junk  
  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 7253);
  rng->Init();
  for (int i = 0; i < n_sources; i++) {
    constructRandomSpinorSource(host_quarks[i], 4, 3, inv_param.cpu_prec, gauge_param.X, *rng);
  }
  for (int i = 0; i < n_evecs; i++) {
    constructRandomSpinorSource(host_evecs[i], 1, 3, inv_param.cpu_prec, gauge_param.X, *rng);
  }
  
  // Create device vectors for quarks
  ColorSpinorParam cuda_quark_param(cpu_quark_param);
  cuda_quark_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_quark_param.create = QUDA_ZERO_FIELD_CREATE;
  cuda_quark_param.setPrecision(inv_param.cpu_prec, inv_param.cpu_prec, true);
  std::vector<ColorSpinorField *> quda_quarks;
  quda_quarks.reserve(n_sources);
  for (int i = 0; i < n_sources; i++) {
    quda_quarks.push_back(ColorSpinorField::Create(cuda_quark_param));
    // Copy data from host to device
    *quda_quarks[i] = *quarks[i];
  }

  // Create device vectors for evecs
  ColorSpinorParam cuda_evec_param(cpu_evec_param);
  cuda_evec_param.location = QUDA_CUDA_FIELD_LOCATION;
  cuda_evec_param.create = QUDA_ZERO_FIELD_CREATE;
  cuda_evec_param.setPrecision(inv_param.cpu_prec, inv_param.cpu_prec, true);
  cuda_evec_param.nSpin = 1;
  std::vector<ColorSpinorField *> quda_evecs;
  quda_evecs.reserve(n_evecs);
  for (int i = 0; i < n_evecs; i++) {
    quda_evecs.push_back(ColorSpinorField::Create(cuda_evec_param));
    // Copy data from host to device
    *quda_evecs[i] = *evecs[i];
  }

  // Finally, construct an array of stochastic noise (junk for now)
  Complex noise[4 * n_dil_vecs];
  for(int i=0; i < 4 * n_dil_vecs; i++) {
    //noise[i].real(rand() / (double)RAND_MAX);
    //noise[i].imag(rand() / (double)RAND_MAX);

    noise[i].real(1.0);
    noise[i].imag(0.0);
  }

  // Host side data for sinks
  void *host_sinks = (void *)malloc(n_evecs * dil_scheme * 4 * sizeof(double));

  // Use the dilution scheme and stochstic noise to construct quark sources
  double time_lsc = -clock();
  laphSourceConstruct(quda_quarks, quda_evecs, noise, dil_scheme);
  time_lsc += clock();

  // The quarks sources are located in quda_quarks. We invert using those
  // sources and place the propagator back into quda_quarks
  double time_lsi = -clock();
  laphSourceInvert(quda_quarks, &inv_param, gauge_param.X);
  time_lsi += clock();
  
  // We now perfrom the projection back onto the eigenspace. The data
  // is placed in host_sinks in i, X, Y, Z, T, spin order
  double time_lsp = -clock();
  laphSinkProject(quda_quarks, quda_evecs, host_sinks, dil_scheme);
  time_lsp += clock();

  printfQuda("LSC time = %e\n", time_lsc/CLOCKS_PER_SEC);
  printfQuda("LSI time = %e\n", time_lsi/CLOCKS_PER_SEC);
  printfQuda("LSP time = %e\n", time_lsp/CLOCKS_PER_SEC);

  rng->Release();
  delete rng;

  // free the multigrid solver
  if(inv_multigrid) destroyMultigridQuda(mg_preconditioner);
  
  // Clean up memory allocations
  for (int i = 0; i < n_sources; i++) {
    free(host_quarks[i]);
    delete quarks[i];
    delete quda_quarks[i];
  }
  free(host_quarks);
  for (int i = 0; i < n_evecs; i++) {
    free(host_evecs[i]);
    delete evecs[i];
    delete quda_evecs[i];
  }
  
  free(host_sinks);
  
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
