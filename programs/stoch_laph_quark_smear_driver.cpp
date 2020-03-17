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
  add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }  

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);
  
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setInvertParam(inv_param);
  setWilsonGaugeParam(gauge_param);

  // Initialize the QUDA library
  initQuda(device);

  // Initialise QUDA's host RNG
  initRand();
  setSpinorSiteSize(24);
  setDims(gauge_param.X);
  display_driver_info();
  
  // Make up some junk data
  int dil_scheme = 16;
  int n_evecs = 400;
  // Deduce the number of eigenvectors to be used per dilution.
  if (n_evecs%dil_scheme != 0) {
    errorQuda("Number of eigen vectors passed %d is not a multiple of the dilution scheme %d", n_evecs, dil_scheme);
  }
  int n_dil_vecs = n_evecs/dil_scheme;
  int n_sources = 4 * dil_scheme;

  printfQuda("Size = V * 24 * inv_param.cpu_prec = %d\n", V * 24 * inv_param.cpu_prec);
  
  // Host side data for sources/quarks
  void **host_quarks = (void **)malloc(n_sources * sizeof(void *));
  for (int i = 0; i < n_sources; i++) {
    host_quarks[i] = (void *)malloc(V * 24 * inv_param.cpu_prec);
  }  
  // Parameter object describing the sources and smeared quarks
  ColorSpinorParam cpu_quark_param(host_quarks[0], inv_param, gauge_param.X, false, QUDA_CPU_FIELD_LOCATION);  
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
  ColorSpinorParam cuda_evec_param(cuda_quark_param);
  cuda_evec_param.nSpin = 1;
  std::vector<ColorSpinorField *> quda_evecs;
  quda_evecs.reserve(n_evecs);
  for (int i = 0; i < n_evecs; i++) {
    quda_evecs.push_back(ColorSpinorField::Create(cuda_evec_param));
    // Copy data from host to device
    *quda_evecs[i] = *evecs[i];
  }
  
  // Finally, construct an array of stochastic noise (junk for now)
  Complex *noise = (Complex *)safe_malloc((4 * n_dil_vecs) * sizeof(Complex));
  for(int i=0; i < 4 * n_dil_vecs; i++) {
    noise[i].real(rand() / (double)RAND_MAX);
    noise[i].imag(rand() / (double)RAND_MAX);
  }

  // Use the dilution scheme and stochstic noise to construct quark sources
  laphSourceConstruct(quda_quarks, quda_evecs, noise, dil_scheme);

  // The quarks sources are located in quda_quarks. We invert using those
  // sources and place the propagator back into quda_quarks
  laphSourceInvert(quda_quarks, &inv_param, gauge_param.X);
  
  // Host side data for sinks
  void **host_sinks = (void **)malloc(n_evecs * dil_scheme * sizeof(void *));
  for (int i = 0; i < n_evecs * dil_scheme; i++) {
    host_sinks[i] = (void *)malloc(V * 4 * inv_param.cpu_prec);
  }
  
  // We now perfrom the projection back onto the eigenspace. The data
  // is placed in host_sinks in i, X, Y, Z, T, spin order 
  laphSinkProject(quda_quarks, quda_evecs, host_sinks, dil_scheme);
  
  return 0;
}
