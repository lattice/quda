#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// QUDA headers
#include <quda.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>
#include <qio_field.h>
#include "test.h"

QudaGaugeParam gauge_param;
QudaInvertParam eig_inv_param;
QudaEigParam eig_param;

// if "--enable-testing true" is passed, we run the tests defined in here
#include <staggered_eigensolve_test_gtest.hpp>

void display_test_info(QudaEigParam &param)
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy), xdim, ydim, zdim,
             tdim, Lsdim);

  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(param.eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(param.spectrum));
  if (param.eig_type == QUDA_EIG_BLK_TR_LANCZOS) printfQuda(" - eigenvector block size %d\n", param.block_size);
  printfQuda(" - number of eigenvectors requested %d\n", param.n_conv);
  printfQuda(" - size of eigenvector search space %d\n", param.n_ev);
  printfQuda(" - size of Krylov space %d\n", param.n_kr);
  printfQuda(" - solver tolerance %e\n", param.tol);
  printfQuda(" - convergence required (%s)\n", param.require_convergence ? "true" : "false");
  if (param.compute_svd) {
    printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
    printfQuda(" - ***********************************************************\n");
    printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
    printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
    printfQuda(" - ***********************************************************\n");
  } else {
    printfQuda(" - Operator: daggered (%s) , norm-op (%s), even-odd pc (%s)\n", param.use_dagger ? "true" : "false",
               param.use_norm_op ? "true" : "false", param.use_pc ? "true" : "false");
  }
  if (param.use_poly_acc) {
    printfQuda(" - Chebyshev polynomial degree %d\n", param.poly_deg);
    printfQuda(" - Chebyshev polynomial minumum %e\n", param.a_min);
    if (param.a_max <= 0)
      printfQuda(" - Chebyshev polynomial maximum will be computed\n");
    else
      printfQuda(" - Chebyshev polynomial maximum %e\n\n", param.a_max);
  }
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

GaugeField cpuFatQDP = {};
GaugeField cpuLongQDP = {};
GaugeField cpuFatMILC = {};
GaugeField cpuLongMILC = {};

void init()
{
  // Set QUDA internal parameters
  gauge_param = newQudaGaugeParam();
  setStaggeredGaugeParam(gauge_param);

  // Though no inversions are performed, the inv_param
  // structure contains all the information we need to
  // construct the dirac operator.
  eig_inv_param = newQudaInvertParam();
  setStaggeredInvertParam(eig_inv_param);

  eig_param = newQudaEigParam();
  // We encapsualte the inv_param structure inside the eig_param structure
  eig_param.invert_param = &eig_inv_param;
  setEigParam(eig_param);

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, 1);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuParam(gauge_param);
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuParam.create = QUDA_NULL_FIELD_CREATE;
  GaugeField cpuIn = GaugeField(cpuParam);
  cpuFatQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuFatMILC = GaugeField(cpuParam);

  cpuParam.link_type = QUDA_ASQTAD_LONG_LINKS;
  cpuParam.nFace = 3;
  cpuParam.order = QUDA_QDP_GAUGE_ORDER;
  cpuLongQDP = GaugeField(cpuParam);
  cpuParam.order = QUDA_MILC_GAUGE_ORDER;
  cpuLongMILC = GaugeField(cpuParam);

  void *qdp_inlink[4] = {cpuIn.data(0), cpuIn.data(1), cpuIn.data(2), cpuIn.data(3)};
  void *qdp_fatlink[4] = {cpuFatQDP.data(0), cpuFatQDP.data(1), cpuFatQDP.data(2), cpuFatQDP.data(3)};
  void *qdp_longlink[4] = {cpuLongQDP.data(0), cpuLongQDP.data(1), cpuLongQDP.data(2), cpuLongQDP.data(3)};
  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, 0, nullptr, true);

  // Reorder gauge fields to MILC order
  cpuFatMILC = cpuFatQDP;
  cpuLongMILC = cpuLongQDP;

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  // This needs to be called before `loadFatLongGaugeQuda` because this routine also loads the
  // gauge fields with different parameters.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);
    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  freeGaugeQuda();

  loadFatLongGaugeQuda(cpuFatMILC.data(), cpuLongMILC.data(), gauge_param);

  // now copy back to QDP aliases, since these are used for the reference dslash
  cpuFatQDP = cpuFatMILC;
  cpuLongQDP = cpuLongMILC;
  // ensure QDP alias has exchanged ghosts
  cpuFatQDP.exchangeGhost();
  cpuLongQDP.exchangeGhost();

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------
}

std::vector<double> eigensolve(test_t test_param)
{
  // Collect testing parameters from gtest
  eig_param.eig_type = ::testing::get<0>(test_param);
  eig_param.use_norm_op = ::testing::get<1>(test_param);
  eig_param.use_pc = ::testing::get<2>(test_param);
  eig_param.compute_svd = ::testing::get<3>(test_param);
  eig_param.spectrum = ::testing::get<4>(test_param);

  eig_inv_param.solution_type = eig_param.use_pc ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

  // whether we are using the resident smeared gauge or not
  eig_param.use_smeared_gauge = gauge_smear;

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    int dimension = laplace3D < 4 ? 3 : 4;
    eig_inv_param.kappa = 1.0 / (2 * dimension + mass);
    eig_inv_param.laplace3D = laplace3D;
    if (dimension == 3) {
      eig_param.ortho_dim = laplace3D;
      eig_param.ortho_dim_size_local = tdim;
    }
  }

  // For gtest testing, we prohibit the use of polynomial acceleration as
  // the fine tuning required can inhibit convergence of an otherwise
  // perfectly good algorithm. We also have a default value of 4
  // for the block size in Block TRLM, and 4 for the batched rotation.
  // The user may change these values via the command line:
  // --eig-block-size
  // --eig-batched-rotate
  if (enable_testing) {
    eig_use_poly_acc = false;
    eig_param.use_poly_acc = QUDA_BOOLEAN_FALSE;
    eig_batched_rotate != 0 ? eig_param.batched_rotate = eig_batched_rotate : eig_param.batched_rotate = 0;
  }

  logQuda(QUDA_SUMMARIZE, "Action = %s, Solver = %s, norm-op = %s, even-odd = %s, with SVD = %s, spectrum = %s\n",
          get_dslash_str(dslash_type), get_eig_type_str(eig_param.eig_type),
          eig_param.use_norm_op == QUDA_BOOLEAN_TRUE ? "true" : "false",
          eig_param.use_pc == QUDA_BOOLEAN_TRUE ? "true" : "false",
          eig_param.compute_svd == QUDA_BOOLEAN_TRUE ? "true" : "false", get_eig_spectrum_str(eig_param.spectrum));

  if (!enable_testing || (enable_testing && getVerbosity() >= QUDA_VERBOSE)) display_test_info(eig_param);

  // Gauge Smearing Routines
  if (gauge_smear) {
    quda::host_timer_t host_timer;
    host_timer.start(); // start the timer

    std::vector<QudaGaugeObservableParam> obs_param(gauge_smear_steps / measurement_interval + 1);
    for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
      obs_param[i] = newQudaGaugeObservableParam();
      obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
      obs_param[i].compute_qcharge = QUDA_BOOLEAN_TRUE;
      obs_param[i].su_project = su_project ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
    }

    // We here set all the problem parameters for all possible smearing types.
    QudaGaugeSmearParam smear_param = newQudaGaugeSmearParam();
    setGaugeSmearParam(smear_param);

    switch (smear_param.smear_type) {
    case QUDA_GAUGE_SMEAR_APE:
    case QUDA_GAUGE_SMEAR_STOUT:
    case QUDA_GAUGE_SMEAR_OVRIMP_STOUT:
    case QUDA_GAUGE_SMEAR_HYP: {
      performGaugeSmearQuda(&smear_param, obs_param.data());
      break;
    }

      // Here we use a typical use case which is different from simple smearing in that
      // the user will want to compute the plaquette values to compute the gauge energy.
    case QUDA_GAUGE_SMEAR_WILSON_FLOW:
    case QUDA_GAUGE_SMEAR_SYMANZIK_FLOW: {
      for (int i = 0; i < gauge_smear_steps / measurement_interval + 1; i++) {
        obs_param[i].compute_plaquette = QUDA_BOOLEAN_TRUE;
      }
      performWFlowQuda(&smear_param, obs_param.data());
      break;
    }
    default: errorQuda("Undefined gauge smear type %d given", smear_param.smear_type);
    }

    host_timer.stop();
    printfQuda("Time for gauge smearing = %f\n", host_timer.last());
  }

  // Vector construct START
  //----------------------------------------------------------------------------
  // Host side arrays to store the eigenpairs computed by QUDA
  int n_eig = eig_n_conv;
  if (eig_param.compute_svd == QUDA_BOOLEAN_TRUE) n_eig *= 2;
  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &eig_inv_param, &gauge_param);
  // Void pointers to host side arrays, compatible with the QUDA interface.
  std::vector<void *> host_evecs_ptr(n_eig);
  // Allocate host side memory and pointers
  std::vector<quda::ColorSpinorField> evecs(n_eig, cs_param);
  for (int i = 0; i < n_eig; i++) host_evecs_ptr[i] = evecs[i].data();

  // Complex eigenvalues
  int n_batch = laplace3D == 3 ? eig_param.ortho_dim_size_local * comm_dim(3) : 1;
  std::vector<__complex__ double> evals(eig_n_conv * n_batch);
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
  eigensolveQuda(host_evecs_ptr.data(), evals.data(), &eig_param);
  host_timer.stop();
  printfQuda("Time for %s solution = %f\n", eig_param.arpack_check ? "ARPACK" : "QUDA", host_timer.last());

  // Perform host side verification of eigenvector if requested.

  std::vector<double> residua(eig_n_conv, 0.0);
  // Perform host side verification of eigenvector if requested.
  if (verify_results) {
    for (int i = 0; i < eig_n_conv; i++) {
      if (eig_param.compute_svd == QUDA_BOOLEAN_TRUE) {
        std::vector<double _Complex> sigma(n_batch);
        for (auto b = 0; b < n_batch; b++) sigma[b] = evals[b * eig_n_conv + i];
        residua[i] = verifyStaggeredTypeSingularVector(evecs[i], evecs[i + eig_n_conv], sigma, i, eig_param, cpuFatQDP,
                                                       cpuLongQDP, laplace3D);
      } else {
        std::vector<double _Complex> lambda(n_batch);
        for (auto b = 0; b < n_batch; b++) lambda[b] = evals[b * eig_n_conv + i];
        residua[i] = verifyStaggeredTypeEigenvector(evecs[i], lambda, i, eig_param, cpuFatQDP, cpuLongQDP, laplace3D);
      }
    }
  }
  return residua;
  // QUDA eigensolver test COMPLETE
  //----------------------------------------------------------------------------
}

void cleanup()
{
  cpuFatQDP = {};
  cpuLongQDP = {};
  cpuFatMILC = {};
  cpuLongMILC = {};
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  // Set defaults
  setQudaStaggeredDefaultInvTestParams();

  auto app = make_app();
  add_eigen_option_group(app);
  add_su3_option_group(app);
  add_testing_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  setVerbosity(verbosity);

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  initRand();

  // Only these fermions are supported in this file
  if constexpr (is_enabled_laplace()) {
    if (!is_staggered(dslash_type) && !is_laplace(dslash_type))
      errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  } else {
    if (is_laplace(dslash_type))
      errorQuda("The Laplace dslash is not enabled, cmake configure with -DQUDA_DIRAC_LAPLACE=ON");
    if (!is_staggered(dslash_type)) errorQuda("dslash_type %s not supported", get_dslash_str(dslash_type));
  }

  if (eig_param.arpack_check && !(prec == QUDA_DOUBLE_PRECISION)) {
    errorQuda("ARPACK check only available in double precision");
  }

  // Sanity check combinations of solve type and solution type
  if ((solve_type == QUDA_DIRECT_SOLVE && solution_type != QUDA_MAT_SOLUTION)
      || (solve_type == QUDA_DIRECT_PC_SOLVE && solution_type != QUDA_MATPC_SOLUTION)
      || (solve_type == QUDA_NORMOP_SOLVE && solution_type != QUDA_MATDAG_MAT_SOLUTION)) {
    errorQuda("Invalid combination of solve_type %s and solution_type %s", get_solve_str(solve_type),
              get_solution_str(solution_type));
  }

  initQuda(device_ordinal);

  if (enable_testing) {
    // We need to force a well-behaved operator + reasonable convergence, otherwise
    // the staggered tests will fail. These checks are designed to be consistent
    // with what's in [src]/tests/CMakeFiles.txt, which have been "sanity checked"
    bool changes = false;
    if (!compute_fatlong) {
      compute_fatlong = true;
      changes = true;
    }

    double expected_tol = (prec == QUDA_SINGLE_PRECISION) ? 1e-4 : 1e-5;
    if (eig_tol != expected_tol) {
      eig_tol = expected_tol;
      changes = true;
    }
    if (niter != 1000) {
      niter = 1000;
      changes = true;
    }
    if (eig_n_kr != 256) {
      eig_n_kr = 256;
      changes = true;
    }
    if (eig_block_size != 4) { eig_block_size = 4; }

    if (changes) {
      printfQuda("For gtest, various defaults are changed:\n");
      printfQuda("  --compute-fat-long true\n");
      printfQuda("  --eig-tol (1e-5 for double, 1e-4 for single)\n");
      printfQuda("  --niter 1000\n");
      printfQuda("  --eig-n-kr 256\n");
    }
  }

  init();

  int result = 0;
  if (enable_testing) { // tests are defined in invert_test_gtest.hpp
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    result = RUN_ALL_TESTS();
  } else {
    eigensolve(
      test_t {eig_param.eig_type, eig_param.use_norm_op, eig_param.use_pc, eig_param.compute_svd, eig_param.spectrum});
  }

  cleanup();

  // Memory clean-up
  freeGaugeQuda();

  // Finalize the QUDA library
  endQuda();
  finalizeComms();

  return result;
}
