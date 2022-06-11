#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
//#include <color_spinor_field.h> // convenient quark field container

#include <misc.h>
#include <timer.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>

// Place params above "eigensolve_test_gtest.hpp" so they
// are visible therein.
QudaGaugeParam gauge_param;
QudaInvertParam eig_inv_param;
QudaEigParam eig_param;

// if "--enable-testing true" is passed, we run the tests defined in here
#include <eigensolve_test_gtest.hpp>

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
  if (eig_type == QUDA_EIG_BLK_TR_LANCZOS) printfQuda(" - eigenvector block size %d\n", eig_block_size);
  printfQuda(" - number of eigenvectors requested %d\n", eig_n_conv);
  printfQuda(" - size of eigenvector search space %d\n", eig_n_ev);
  printfQuda(" - size of Krylov space %d\n", eig_n_kr);
  printfQuda(" - solver tolerance %e\n", eig_tol);
  printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
  if (eig_compute_svd) {
    printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
    printfQuda(" - ***********************************************************\n");
    printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
    printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
    printfQuda(" - ***********************************************************\n");
  } else {
    printfQuda(" - Operator: daggered (%s) , norm-op (%s), even-odd pc (%s)\n",
	       eig_use_dagger ? "true" : "false",
               eig_use_normop ? "true" : "false",
	       eig_use_pc ? "true" : "false");
  }
  if (eig_use_poly_acc) {
    printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
    printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
    if (eig_amax <= 0)
      printfQuda(" - Chebyshev polynomial maximum will be computed\n");
    else
      printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
  }
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

std::vector<char> gauge_;
std::array<void *, 4> gauge;
std::vector<char> clover;
std::vector<char> clover_inv;

void init(int argc, char **argv)
{
  // Construct QUDA param structures to define the problem
  //------------------------------------------------------
  gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  // Though no inversions are performed, the inv_param
  // structure contains all the information we need to
  // construct the dirac operator.
  eig_inv_param = newQudaInvertParam();
  setInvertParam(eig_inv_param);
  
  eig_param = newQudaEigParam();
  // We encapsualte the inv_param structure inside the
  // eig_param structure
  eig_param.invert_param = &eig_inv_param;
  setEigParam(eig_param);
  display_test_info();
  //------------------------------------------------------

  // Set lattice dimensions for Dslash reference
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    dw_setDims(gauge_param.X, eig_inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  gauge_.resize(4 * V * gauge_site_size * host_gauge_data_type_size);
  for (int i = 0; i < 4; i++) gauge[i] = gauge_.data() + i * V * gauge_site_size * host_gauge_data_type_size;
  constructHostGaugeField(gauge.data(), gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda(gauge.data(), &gauge_param);

  // Allocate host side memory for clover terms if needed.
  //----------------------------------------------------------------------------
  // Allocate space on the host (always best to allocate and free in the same scope)
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover.resize(V * clover_site_size * host_clover_data_type_size);
    clover_inv.resize(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover.data(), clover_inv.data(), eig_inv_param);
    // Load the clover terms to the device
    loadCloverQuda(clover.data(), clover_inv.data(), &eig_inv_param);
  }
}

std::vector<double> eigensolve(test_t test_param)
{
  eig_param.eig_type = ::testing::get<0>(test_param);
  eig_param.use_norm_op = ::testing::get<1>(test_param);
  eig_param.use_pc = ::testing::get<2>(test_param);
  eig_param.compute_svd = ::testing::get<3>(test_param);
  eig_param.spectrum = ::testing::get<4>(test_param);

  logQuda(QUDA_SUMMARIZE, "Solver = %s, norm-op = %s, even-odd = %s, with SVD = %s, spectrum = %s\n",
	  get_eig_type_str(eig_param.eig_type),
	  eig_param.use_norm_op == QUDA_BOOLEAN_TRUE ? "true" : "false",
	  eig_param.use_pc == QUDA_BOOLEAN_TRUE ? "true" : "false",
	  eig_param.compute_svd == QUDA_BOOLEAN_TRUE ? "true" : "false",
	  get_eig_spectrum_str(eig_param.spectrum));
  
  // Vector construct START
  //----------------------------------------------------------------------------
  // Host side arrays to store the eigenpairs computed by QUDA
  std::vector<quda::ColorSpinorField> evecs(eig_n_conv);
  quda::ColorSpinorField check;
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &eig_inv_param, &gauge_param);
  check = quda::ColorSpinorField(cs_param);
  
  // Void pointers to host side arrays, compatible with the QUDA interface.
  std::vector<void *> host_evecs_ptr(eig_n_conv);

  // Allocate host side memory and pointers
  for (int i = 0; i < eig_n_conv; i++) {
    evecs[i] = quda::ColorSpinorField(cs_param);
    host_evecs_ptr[i] = evecs[i].V();
  }

  // Complex eigenvalues
  std::vector<__complex__ double> evals(eig_n_conv);
  // Vector construct END
  //----------------------------------------------------------------------------

  // QUDA eigensolver test BEGIN
  //----------------------------------------------------------------------------
  // This function returns the host_evecs and host_evals pointers, populated with the
  // requested data, at the requested prec. All the information needed to perfom the
  // solve is in the eig_param container. If eig_param.arpack_check == true and
  // precision is double, the routine will use ARPACK rather than the GPU.
  quda::host_timer_t host_timer;
  host_timer.start();
  if (eig_param.arpack_check && !(eig_inv_param.cpu_prec == QUDA_DOUBLE_PRECISION)) {
    errorQuda("ARPACK check only available in double precision");
  }
  eigensolveQuda(host_evecs_ptr.data(), evals.data(), &eig_param);
  host_timer.stop();
  printfQuda("Time for %s solution = %f\n", eig_param.arpack_check ? "ARPACK" : "QUDA", host_timer.last());

  std::vector<double> residua(eig_n_conv, 0.0);
  // Perform host side verification of inversion if requested
  if (verify_results) {
    for (int i = 0; i < eig_n_conv; i++) {
      residua[i] = verifyEigenvector(evecs[i].V(), evecs[i].V(), check.V(), gauge_param, eig_inv_param,
				     gauge.data(), clover.data(), clover_inv.data(), *(double *)&evals[i]);
    }
  }
  return residua;
  // QUDA eigensolver test COMPLETE
  //----------------------------------------------------------------------------
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_madwf_option_group(app);
  add_comms_option_group(app);
  add_testing_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();
  
  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // Only these fermions are supported in this file
  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH
      && dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }
  
  // Initialize the QUDA library
  initQuda(device_ordinal);

  // Initialise this test (parameters, gauge, clover)
  init(argc, argv);
  
  // Compute plaquette as a sanity check
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  int result = 0;
  if (enable_testing) { // tests are defined in invert_test_gtest.hpp
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    result = RUN_ALL_TESTS();
  } else {
    eigensolve(test_t {eig_param.eig_type, eig_param.use_norm_op, eig_param.use_pc, eig_param.compute_svd, eig_param.spectrum});
  }
  
  // Memory clean-up
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    freeCloverQuda();
  }

  // Finalize the QUDA library
  endQuda();
  finalizeComms();

  return result;
}
