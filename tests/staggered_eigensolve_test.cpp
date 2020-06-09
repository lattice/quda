#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>
#include <unitarization_links.h>
#include <gauge_field.h>

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_utils.h>
#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d/%d/%d          %d \n", get_prec_str(prec),
             get_prec_str(prec_sloppy), get_recon_str(link_recon), get_recon_str(link_recon_sloppy),
             get_staggered_test_type(test_type), xdim, ydim, zdim, tdim);

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
    if (eig_amax < 0)
      printfQuda(" - Chebyshev polynomial maximum will be computed\n");
    else
      printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
  }

  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));

  return;
}

int main(int argc, char **argv)
{
  // Set a default
  solve_type = QUDA_INVALID_SOLVE;

  auto app = make_app();
  add_eigen_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"full", 0}, {"even", 3}, {"odd", 4}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // Only these fermions are supported in this file. Ensure a reasonable default,
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    printfQuda("dslash_type %s not supported, defaulting to %s\n", get_dslash_str(dslash_type),
               get_dslash_str(QUDA_ASQTAD_DSLASH));
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  setQudaStaggeredEigTestParams();

  display_test_info();

  // Set QUDA internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setStaggeredGaugeParam(gauge_param);
  // Though no inversions are performed, the inv_param
  // structure contains all the information we need to
  // construct the dirac operator. We encapsualte the
  // inv_param structure inside the eig_param structure
  // to avoid any confusion
  QudaInvertParam eig_inv_param = newQudaInvertParam();
  setStaggeredInvertParam(eig_inv_param);
  QudaEigParam eig_param = newQudaEigParam();
  setEigParam(eig_param);
  // We encapsulate the eigensolver parameters inside the invert parameter structure
  eig_param.invert_param = &eig_inv_param;

  if (eig_param.arpack_check && !(prec == QUDA_DOUBLE_PRECISION)) {
    errorQuda("ARPACK check only available in double precision");
  }

  // This must be before the FaceBuffer is created
  // (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, 1); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  // Staggered Gauge construct START
  //-----------------------------------------------------------------------------------
  void *qdp_inlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_fatlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_longlink[4] = {nullptr, nullptr, nullptr, nullptr};
  void *milc_fatlink = nullptr;
  void *milc_longlink = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_fatlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
    qdp_longlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
  }
  milc_fatlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
  milc_longlink = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

  constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, argc, argv);

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    // Compute fat link plaquette
    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);
    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  // Reorder gauge fields to MILC order
  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  loadFatLongGaugeQuda(milc_fatlink, milc_longlink, gauge_param);

  // Staggered Gauge construct END
  //-----------------------------------------------------------------------------------

  // Host side arrays to store the eigenpairs computed by QUDA
  void **host_evecs = (void **)malloc(eig_nConv * sizeof(void *));
  for (int i = 0; i < eig_nConv; i++) {
    host_evecs[i] = (void *)malloc(V * my_spinor_site_size * eig_inv_param.cpu_prec);
  }
  double _Complex *host_evals = (double _Complex *)malloc(eig_param.nEv * sizeof(double _Complex));

  double time = 0.0;

  // QUDA eigensolver test
  //----------------------------------------------------------------------------
  switch (test_type) {
  case 0: // full parity solution
  case 3: // even
  case 4: // odd
    // This function returns the host_evecs and host_evals pointers, populated with
    // the requested data, at the requested prec. All the information needed to
    // perfom the solve is in the eig_param container.
    // If eig_param.arpack_check == true and precision is double, the routine will
    // use ARPACK rather than the GPU.

    time = -((double)clock());
    eigensolveQuda(host_evecs, host_evals, &eig_param);
    time += (double)clock();

    printfQuda("Time for %s solution = %f\n", eig_param.arpack_check ? "ARPACK" : "QUDA", time / CLOCKS_PER_SEC);
    break;

  default: errorQuda("Unsupported test type");

  } // switch

  // Deallocate host memory
  for (int i = 0; i < eig_nConv; i++) free(host_evecs[i]);
  free(host_evecs);
  free(host_evals);

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

  endQuda();
  finalizeComms();
}
