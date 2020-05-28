#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h>
#include <gauge_field.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec    prec_sloppy   multishift  matpc_type  recon  recon_sloppy solve_type S_dimension T_dimension "
             "Ls_dimension   dslash_type  normalization\n");
  printfQuda(
    "%6s   %6s          %d     %12s     %2s     %2s         %10s %3d/%3d/%3d     %3d         %2d       %14s  %8s\n",
    get_prec_str(prec), get_prec_str(prec_sloppy), multishift, get_matpc_str(matpc_type), get_recon_str(link_recon),
    get_recon_str(link_recon_sloppy), get_solve_str(solve_type), xdim, ydim, zdim, tdim, Lsdim,
    get_dslash_str(dslash_type), get_mass_normalization_str(normalization));

  if (inv_multigrid) {
    printfQuda("MG parameters\n");
    printfQuda(" - number of levels %d\n", mg_levels);
    for (int i = 0; i < mg_levels - 1; i++) {
      printfQuda(" - level %d number of null-space vectors %d\n", i + 1, nvec[i]);
      printfQuda(" - level %d number of pre-smoother applications %d\n", i + 1, nu_pre[i]);
      printfQuda(" - level %d number of post-smoother applications %d\n", i + 1, nu_post[i]);
    }

    printfQuda("MG Eigensolver parameters\n");
    for (int i = 0; i < mg_levels; i++) {
      if (low_mode_check || mg_eig[i]) {
        printfQuda(" - level %d solver mode %s\n", i + 1, get_eig_type_str(mg_eig_type[i]));
        printfQuda(" - level %d spectrum requested %s\n", i + 1, get_eig_spectrum_str(mg_eig_spectrum[i]));
        if (mg_eig_type[i] == QUDA_EIG_BLK_TR_LANCZOS)
          printfQuda(" - eigenvector block size %d\n", mg_eig_block_size[i]);
        printfQuda(" - level %d number of eigenvectors requested nConv %d\n", i + 1, nvec[i]);
        printfQuda(" - level %d size of eigenvector search space %d\n", i + 1, mg_eig_nEv[i]);
        printfQuda(" - level %d size of Krylov space %d\n", i + 1, mg_eig_nKr[i]);
        printfQuda(" - level %d solver tolerance %e\n", i + 1, mg_eig_tol[i]);
        printfQuda(" - level %d convergence required (%s)\n", i + 1, mg_eig_require_convergence[i] ? "true" : "false");
        printfQuda(" - level %d Operator: daggered (%s) , norm-op (%s)\n", i + 1,
                   mg_eig_use_dagger[i] ? "true" : "false", mg_eig_use_normop[i] ? "true" : "false");
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
  }

  if (inv_deflate) {
    printfQuda("\n   Eigensolver parameters\n");
    printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
    printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
    if (eig_type == QUDA_EIG_BLK_TR_LANCZOS) printfQuda(" - eigenvector block size %d\n", eig_block_size);
    printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
    printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
    printfQuda(" - size of Krylov space %d\n", eig_nKr);
    printfQuda(" - solver tolerance %e\n", eig_tol);
    printfQuda(" - convergence required (%s)\n", eig_require_convergence ? "true" : "false");
    if (eig_compute_svd) {
      printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
      printfQuda(" - ***********************************************************\n");
      printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
      printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
      printfQuda(" - ***********************************************************\n");
    } else {
      printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n", eig_use_dagger ? "true" : "false",
                 eig_use_normop ? "true" : "false");
    }
    if (eig_use_poly_acc) {
      printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
      printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
      if (eig_amax <= 0)
        printfQuda(" - Chebyshev polynomial maximum will be computed\n");
      else
        printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
    }
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

int main(int argc, char **argv)
{
  setQudaDefaultMgTestParams();
  // Parse command line options
  auto app = make_app();
  add_eigen_option_group(app);
  add_deflation_option_group(app);
  add_multigrid_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"full_ee_prec", 1}, {"full_oo_prec", 2}, {"even", 3},
                                          {"odd", 4},  {"mcg_even", 5},     {"mcg_odd", 6}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (!inv_multigrid) solve_type = QUDA_INVALID_SOLVE;

  if (inv_deflate && inv_multigrid) {
    printfQuda("Error: Cannot use both deflation and multigrid preconditioners on top level solve.\n");
    exit(0);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  initRand();

  // Only these fermions are supported in this file. Ensure a reasonable default,
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    printfQuda("dslash_type %s not supported, defaulting to %s\n", get_dslash_str(dslash_type),
               get_dslash_str(QUDA_ASQTAD_DSLASH));
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  // Need to add support for LAPLACE MG?
  if (inv_multigrid) {
    if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH) {
      printfQuda("dslash_type %s not supported for multigrid preconditioner\n", get_dslash_str(dslash_type));
      exit(0);
    }
  }

  // Deduce operator, solution, and operator preconditioning types
  if (!inv_multigrid) setQudaStaggeredInvTestParams();

  display_test_info();

  // Set QUDA internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  setStaggeredGaugeParam(gauge_param);
  if (!inv_multigrid) setStaggeredInvertParam(inv_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaEigParam mg_eig_param[mg_levels];

  if (inv_multigrid) {

    // Set some default values for MG solve types
    setQudaMgSolveTypes();

    setStaggeredMGInvertParam(inv_param);
    // Set sub structures
    mg_param.invert_param = &mg_inv_param;
    // Since the top level is just a unitary rotation, we manually
    // set mg_eig[0] to false (and throw a warning if a user set it to true)
    if (mg_eig[0]) {
      printfQuda("Warning: Cannot specify near-null vectors for top level.\n");
      mg_eig[0] = false;
    }
    for (int i = 1; i < mg_levels; i++) {
      if (mg_eig[i]) {
        mg_eig_param[i] = newQudaEigParam();
        setMultigridEigParam(mg_eig_param[i], i);
        mg_param.eig_param[i] = &mg_eig_param[i];
      } else {
        mg_param.eig_param[i] = nullptr;
      }
    }
    setStaggeredMultigridParam(mg_param);
  }

  QudaEigParam eig_param = newQudaEigParam();
  if (inv_deflate) {
    setEigParam(eig_param);
    inv_param.eig_param = &eig_param;
  } else {
    inv_param.eig_param = nullptr;
  }

  // This must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gauge_param.X);
  // Hack: use the domain wall dimensions so we may use the 5th dim for multi indexing
  dw_setDims(gauge_param.X, 1);
  setSpinorSiteSize(6);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  // Allocate host staggered gauge fields
  void* qdp_inlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_fatlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_longlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void *milc_fatlink = nullptr;
  void *milc_longlink = nullptr;
  GaugeField *cpuFat = nullptr;
  GaugeField *cpuLong = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_fatlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_longlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  }
  milc_fatlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  milc_longlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  // For load, etc
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, argc, argv);
  // Reorder gauge fields to MILC order
  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

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

  // Create ghost gauge fields in case of multi GPU builds.
#ifdef MULTI_GPU

  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.location = QUDA_CPU_FIELD_LOCATION;

  GaugeFieldParam cpuFatParam(milc_fatlink, gauge_param);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = GaugeField::Create(cpuFatParam);

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gauge_param);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = GaugeField::Create(cpuLongParam);
  // constructStaggeredHostGhostGaugeField(cpuFat, cpuLong, milc_fatlink, milc_longlink, gauge_param);
#endif

  loadFatLongGaugeQuda(milc_fatlink, milc_longlink, gauge_param);

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------

  // Setup the multigrid preconditioner
  void *mg_preconditioner = nullptr;
  if (inv_multigrid) {
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
  }

  // Staggered vector construct START
  //-----------------------------------------------------------------------------------
  quda::ColorSpinorField *in;
  quda::ColorSpinorField *out;
  quda::ColorSpinorField *ref;
  quda::ColorSpinorField *tmp;
  quda::ColorSpinorParam cs_param;
  constructStaggeredTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  in = quda::ColorSpinorField::Create(cs_param);
  out = quda::ColorSpinorField::Create(cs_param);
  ref = quda::ColorSpinorField::Create(cs_param);
  tmp = quda::ColorSpinorField::Create(cs_param);
  // Staggered vector construct END
  //-----------------------------------------------------------------------------------

  // Prepare rng
  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();

  // Performance measuring
  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];

  // Pointers for tests 5 and 6
  // Quark masses
  std::vector<double> masses(multishift);
  // Host array for solutions
  void **outArray = (void **)malloc(multishift * sizeof(void *));
  // QUDA host array for internal checks and malloc
  std::vector<ColorSpinorField *> qudaOutArray(multishift);

  // QUDA invert test
  //----------------------------------------------------------------------------
  switch (test_type) {
  case 0: // full parity solution, full parity system
  case 1: // full parity solution, solving EVEN EVEN prec system
  case 2: // full parity solution, solving ODD ODD prec system
  case 3: // even parity solution, solving EVEN system
  case 4: // odd parity solution, solving ODD system

    if (multishift != 1) {
      printfQuda("Multishift not supported for test %d\n", test_type);
      exit(0);
    }

    for (int k = 0; k < Nsrc; k++) {
      quda::spinorNoise(*in, *rng, QUDA_NOISE_UNIFORM);
      if (inv_deflate) eig_param.preserve_deflation = k < Nsrc - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      invertQuda(out->V(), in->V(), &inv_param);

      time[k] = inv_param.secs;
      gflops[k] = inv_param.gflops / inv_param.secs;
      printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
                 inv_param.gflops / inv_param.secs);
      if (verify_results)
        verifyStaggeredInversion(tmp, ref, in, out, mass, qdp_fatlink, qdp_longlink, (void **)cpuFat->Ghost(),
                                 (void **)cpuLong->Ghost(), gauge_param, inv_param, 0);
    }

    // Compute timings
    if (Nsrc > 1) performanceStats(time, gflops);
    break;

  case 5: // multi mass CG, even parity solution, solving EVEN system
  case 6: // multi mass CG, odd parity solution, solving ODD system

    if (multishift < 2) {
      printfQuda("Multishift inverter requires more than one shift, multishift = %d\n", multishift);
      exit(0);
    }

    inv_param.num_offset = multishift;
    for (int i = 0; i < multishift; i++) {
      // Set masses and offsets
      masses[i] = 0.06 + i * i * 0.01;
      inv_param.offset[i] = 4 * masses[i] * masses[i];
      // Set tolerances for the heavy quarks, these can be set independently
      // (functions of i) if desired
      inv_param.tol_offset[i] = inv_param.tol;
      inv_param.tol_hq_offset[i] = inv_param.tol_hq;
      // Allocate memory and set pointers
      qudaOutArray[i] = ColorSpinorField::Create(cs_param);
      outArray[i] = qudaOutArray[i]->V();
    }
    quda::spinorNoise(*in, *rng, QUDA_NOISE_UNIFORM);
    invertMultiShiftQuda((void **)outArray, in->V(), &inv_param);
    cudaDeviceSynchronize();

    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
               inv_param.gflops / inv_param.secs);

    for (int i = 0; i < multishift; i++) {
      printfQuda("%dth solution: mass=%f, ", i, masses[i]);
      verifyStaggeredInversion(tmp, ref, in, qudaOutArray[i], masses[i], qdp_fatlink, qdp_longlink,
                               (void **)cpuFat->Ghost(), (void **)cpuLong->Ghost(), gauge_param, inv_param, i);
    }

    for (int i = 0; i < multishift; i++) delete qudaOutArray[i];
    break;

  default: errorQuda("Unsupported test type");

  } // switch

  delete[] time;
  delete[] gflops;

  // Free RNG
  rng->Release();
  delete rng;

  // Free the multigrid solver
  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);

  // Clean up gauge fields
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_inlink[dir] != nullptr) {
      free(qdp_inlink[dir]);
      qdp_inlink[dir] = nullptr;
    }
    if (qdp_fatlink[dir] != nullptr) {
      free(qdp_fatlink[dir]);
      qdp_fatlink[dir] = nullptr;
    }
    if (qdp_longlink[dir] != nullptr) {
      free(qdp_longlink[dir]);
      qdp_longlink[dir] = nullptr;
    }
  }
  if (milc_fatlink != nullptr) { free(milc_fatlink); milc_fatlink = nullptr; }
  if (milc_longlink != nullptr) { free(milc_longlink); milc_longlink = nullptr; }

#ifdef MULTI_GPU
  if (cpuFat != nullptr) { delete cpuFat; cpuFat = nullptr; }
  if (cpuLong != nullptr) { delete cpuLong; cpuLong = nullptr; }
#endif

  delete in;
  delete out;
  delete ref;
  delete tmp;
  free(outArray);

  // Finalize the QUDA library
  endQuda();

  // Finalize the communications layer
  finalizeComms();

  return 0;
}
