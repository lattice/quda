#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <misc.h>
#include <test_util.h>
#include <test_params.h>
#include <dslash_util.h>
#include <staggered_gauge_utils.h>
#include <unitarization_links.h>
#include <llfat_reference.h>
#include <gauge_field.h>
#include <unitarization_links.h>
#include <blas_reference.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

#define my_spinor_site_size 6

void **ghost_fatlink, **ghost_longlink;

size_t gSize = sizeof(double);

static int n_naiks = 1;

// For loading the gauge fields
int argc_copy;
char **argv_copy;

int X[4];

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d/%d/%d          %d \n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy),
             get_staggered_test_type(test_type), xdim, ydim, zdim, tdim);

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));

  return;
}

void eigensolve_test()
{
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setStaggeredGaugeParam(gauge_param);

  QudaEigParam eig_param = newQudaEigParam();
  // Though no inversions are performed, the inv_param
  // structure contains all the information we need to
  // construct the dirac operator. We encapsualte the
  // inv_param structure inside the eig_param structure
  // to avoid any confusion
  QudaInvertParam eig_inv_param = newQudaInvertParam();
  setStaggeredInvertParam(eig_inv_param);
  eig_param.invert_param = &eig_inv_param;
  setEigParam(eig_param);

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, Nsrc); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  void *qdp_inlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_fatlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_longlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *milc_fatlink = nullptr;
  void *milc_longlink = nullptr;

  gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V * gauge_site_size * gSize);
    qdp_fatlink[dir] = malloc(V * gauge_site_size * gSize);
    qdp_longlink[dir] = malloc(V * gauge_site_size * gSize);
  }
  milc_fatlink = malloc(4 * V * gauge_site_size * gSize);
  milc_longlink = malloc(4 * V * gauge_site_size * gSize);

  // for load, etc
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  // load a field WITHOUT PHASES
  if (strcmp(latfile, "")) {
    read_gauge_field(latfile, qdp_inlink, gauge_param.cpu_prec, gauge_param.X, argc_copy, argv_copy);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gauge_param, QUDA_STAGGERED_DSLASH, gauge_param.cpu_prec);
    }
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      construct_gauge_field(qdp_inlink, 1, gauge_param.cpu_prec, &gauge_param);
    } else {
      construct_fat_long_gauge_field(qdp_inlink, qdp_longlink, 1, gauge_param.cpu_prec, &gauge_param,
                                     compute_fatlong ? QUDA_STAGGERED_DSLASH : dslash_type);
    }
  }

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);

  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], qdp_inlink[dir], V * gauge_site_size * gSize);
      memset(qdp_longlink[dir], 0, V * gauge_site_size * gSize);
    }
  } else { // QUDA_ASQTAD_DSLASH

    if (compute_fatlong) {
      computeFatLongGPU(qdp_fatlink, qdp_longlink, qdp_inlink, gauge_param, gSize, n_naiks, eps_naik);
    } else {
      for (int dir = 0; dir < 4; dir++) { memcpy(qdp_fatlink[dir], qdp_inlink[dir], V * gauge_site_size * gSize); }
    }

    // Compute fat link plaquette.
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);

    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

#ifdef MULTI_GPU
  int tmp_value = MAX(ydim * zdim * tdim / 2, xdim * zdim * tdim / 2);
  tmp_value = MAX(tmp_value, xdim * ydim * tdim / 2);
  tmp_value = MAX(tmp_value, xdim * ydim * zdim / 2);

  int fat_pad = tmp_value;
  int link_pad = 3 * tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(milc_fatlink, gauge_param);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void **)cpuFat->Ghost();

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gauge_param);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void **)cpuLong->Ghost();

#else
  int fat_pad = 0;
  int link_pad = 0;
#endif

  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS :
    QUDA_ASQTAD_FAT_LINKS;
  gauge_param.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
  } else {
    gauge_param.reconstruct = gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gauge_param.cuda_prec_precondition = gauge_param.cuda_prec_sloppy;
  gauge_param.reconstruct_precondition = gauge_param.reconstruct_sloppy;
  loadGaugeQuda(milc_fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.cuda_prec_precondition = gauge_param.cuda_prec_sloppy;
    gauge_param.reconstruct_precondition = gauge_param.reconstruct_sloppy;
    loadGaugeQuda(milc_longlink, &gauge_param);
  }

  switch (test_type) {
  case 0: // full parity solution
  case 3: // even
  case 4: {
    // QUDA eigensolver test
    //----------------------------------------------------------------------------

    // Host side arrays to store the eigenpairs computed by QUDA
    void **host_evecs = (void **)malloc(eig_nConv * sizeof(void *));
    for (int i = 0; i < eig_nConv; i++) {
      host_evecs[i] = (void *)malloc(V * my_spinor_site_size * eig_inv_param.cpu_prec);
    }
    double _Complex *host_evals = (double _Complex *)malloc(eig_param.nEv * sizeof(double _Complex));

    // This function returns the host_evecs and host_evals pointers, populated with
    // the requested data, at the requested prec. All the information needed to
    // perfom the solve is in the eig_param container.
    // If eig_param.arpack_check == true and precision is double, the routine will
    // use ARPACK rather than the GPU.
    double time = -((double)clock());
    if (eig_param.arpack_check && !(prec == QUDA_DOUBLE_PRECISION)) {
      errorQuda("ARPACK check only available in double precision");
    }

    eigensolveQuda(host_evecs, host_evals, &eig_param);
    time += (double)clock();
    printfQuda("Time for %s solution = %f\n", eig_param.arpack_check ? "ARPACK" : "QUDA", time / CLOCKS_PER_SEC);

    // Deallocate host memory
    for (int i = 0; i < eig_nConv; i++) free(host_evecs[i]);
    free(host_evecs);
    free(host_evals);

  } break;
  default: errorQuda("Unsupported test type");
  } // switch

  // Clean up gauge fields.
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
  if (milc_fatlink != nullptr) {
    free(milc_fatlink);
    milc_fatlink = nullptr;
  }
  if (milc_longlink != nullptr) {
    free(milc_longlink);
    milc_longlink = nullptr;
  }

#ifdef MULTI_GPU
  if (cpuFat != nullptr) {
    delete cpuFat;
    cpuFat = nullptr;
  }
  if (cpuLong != nullptr) {
    delete cpuLong;
    cpuLong = nullptr;
  }
#endif
}

int main(int argc, char **argv)
{

  // Set a default
  solve_type = QUDA_INVALID_SOLVE;

  auto app = make_app();
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"even", 3}, {"odd", 4}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));
  add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  if (test_type != 0 && test_type != 3 && test_type != 4) { errorQuda("Test type %d is outside the valid range.\n", test_type); }

  // Ensure a reasonable default
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    warningQuda("The dslash_type %d isn't staggered, asqtad, or laplace. Defaulting to asqtad.\n", dslash_type);
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    // LAPLACE operator path
    if (test_type != 0) { errorQuda("Test type %d is not supported for the Laplace operator.\n", test_type); }
    solve_type = QUDA_DIRECT_SOLVE;
    solution_type = QUDA_MAT_SOLUTION;
    matpc_type = QUDA_MATPC_EVEN_EVEN; // doesn't matter
  } else {
    // STAGGERED operator path
    if (solve_type == QUDA_INVALID_SOLVE) {
      if (test_type == 0) {
        solve_type = QUDA_DIRECT_SOLVE;
      } else {
        solve_type = QUDA_DIRECT_PC_SOLVE;
      }
    }

    if (test_type == 3) {
      matpc_type = QUDA_MATPC_EVEN_EVEN;
    } else if (test_type == 4) {
      matpc_type = QUDA_MATPC_ODD_ODD;
    } else if (test_type == 0) {
      matpc_type = QUDA_MATPC_EVEN_EVEN; // it doesn't matter
    }

    if (test_type == 0) {
      solution_type = QUDA_MAT_SOLUTION;
    } else {
      solution_type = QUDA_MATPC_SOLUTION;
    }
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) { prec_sloppy = prec; }

  if (prec_refinement_sloppy == QUDA_INVALID_PRECISION) { prec_refinement_sloppy = prec_sloppy; }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) { link_recon_sloppy = link_recon; }

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

  display_test_info();

  printfQuda("dslash_type = %d\n", dslash_type);

  argc_copy = argc;
  argv_copy = argv;

  eigensolve_test();

  endQuda();

  // finalize the communications layer
  finalizeComms();
}
