#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <llfat_reference.h>
#include "misc.h"
#include <gauge_field.h>
#include <covdev_reference.h>
#include <unitarization_links.h>
#include <random_quda.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <blas_quda.h>

// Various CPU fields lifted from
// staggered_invert_test.cpp

#define my_spinor_site_size 6

static int n_naiks = 1;

void **ghost_fatlink, **ghost_longlink;

cpuColorSpinorField *in;
cpuColorSpinorField *out;
cpuColorSpinorField *ref;
cpuColorSpinorField *tmp;

namespace quda
{
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
  for (int i = 0; i < mg_levels - 1; i++) {
    printfQuda(" - level %d number of null-space vectors %d\n", i + 1, nvec[i]);
    printfQuda(" - level %d number of pre-smoother applications %d\n", i + 1, nu_pre[i]);
    printfQuda(" - level %d number of post-smoother applications %d\n", i + 1, nu_post[i]);
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
  return;
}

int main(int argc, char **argv)
{
  // We give here the default values to some of the array
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_SUMMARIZE;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    setup_tol[i] = 5e-6;
    setup_maxiter[i] = 500;
    mu_factor[i] = 1.;
    coarse_solve_type[i] = QUDA_INVALID_SOLVE;
    smoother_solve_type[i] = QUDA_INVALID_SOLVE;
    schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_GCR_INVERTER;
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 100;
    solver_location[i] = QUDA_CUDA_FIELD_LOCATION;
    setup_location[i] = QUDA_CUDA_FIELD_LOCATION;
    nu_pre[i] = 2;
    nu_post[i] = 2;
    n_block_ortho[i] = 1;

    // Default eigensolver params
    mg_eig[i] = false;
    mg_eig_tol[i] = 1e-3;
    mg_eig_require_convergence[i] = QUDA_BOOLEAN_TRUE;
    mg_eig_type[i] = QUDA_EIG_TR_LANCZOS;
    mg_eig_spectrum[i] = QUDA_SPECTRUM_SR_EIG;
    mg_eig_check_interval[i] = 5;
    mg_eig_max_restarts[i] = 100;
    mg_eig_use_normop[i] = QUDA_BOOLEAN_FALSE;
    mg_eig_use_dagger[i] = QUDA_BOOLEAN_FALSE;
    mg_eig_use_poly_acc[i] = QUDA_BOOLEAN_TRUE;
    mg_eig_poly_deg[i] = 100;
    mg_eig_amin[i] = 1.0;
    mg_eig_amax[i] = -1.0; // use power iterations

    setup_ca_basis[i] = QUDA_POWER_BASIS;
    setup_ca_basis_size[i] = 4;
    setup_ca_lambda_min[i] = 0.0;
    setup_ca_lambda_max[i] = -1.0; // use power iterations

    coarse_solver_ca_basis[i] = QUDA_POWER_BASIS;
    coarse_solver_ca_basis_size[i] = 4;
    coarse_solver_ca_lambda_min[i] = 0.0;
    coarse_solver_ca_lambda_max[i] = -1.0;

    strcpy(mg_vec_infile[i], "");
    strcpy(mg_vec_outfile[i], "");
  }
  reliable_delta = 1e-4;

  // Give the dslash type a reasonable default.
  dslash_type = QUDA_STAGGERED_DSLASH;

  // command line options
  auto app = make_app();
  add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (smoother_halo_prec == QUDA_INVALID_PRECISION) smoother_halo_prec = prec_null;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;
  for (int i = 0; i < QUDA_MAX_MG_LEVEL; i++) {
    if (coarse_solve_type[i] == QUDA_INVALID_SOLVE) coarse_solve_type[i] = solve_type;
    if (smoother_solve_type[i] == QUDA_INVALID_SOLVE) smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE;
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  // Need to add support for LAPLACE
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  // ESW HACK: needs to be addressed
  /*if (solve_type == QUDA_DIRECT_PC_SOLVE || coarse_solve_type[0] == QUDA_DIRECT_PC_SOLVE || smoother_solve_type[0] ==
  QUDA_DIRECT_PC_SOLVE) { printfQuda("staggered_multigtid_invert_test doesn't support preconditioned outer solve
  yet.\n"); exit(0);
  }*/

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];

  // Since the top level is just a unitary rotation, we manually
  // set mg_eig[0] to false (and throw a warning if a user set it to true)
  if (mg_eig[0]) {
    printfQuda("Warning: Cannot specify near-null vectors for top level.\n");
    mg_eig[0] = false;
  }
  for (int i = 0; i < mg_levels; i++) {
    mg_eig_param[i] = newQudaEigParam();
    setMultigridEigParam(mg_eig_param[i], i);
  }

  setStaggeredGaugeParam(gauge_param);
  setStaggeredInvertParam(inv_param);
  // Change some default params for staggered MG
  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.sp_pad = 0;

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();

  mg_param.invert_param = &mg_inv_param;
  for (int i = 0; i < mg_levels; i++) { mg_param.eig_param[i] = &mg_eig_param[i]; }

  setStaggeredMultigridParam(mg_param);

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, 1); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  /* Taken from staggered_invert_test to load gauge fields */
  size_t host_gauge_data_type_size = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

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

  // for load, etc
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

  // load a field WITHOUT PHASES
  if (strcmp(latfile, "")) {
    read_gauge_field(latfile, qdp_inlink, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gauge_param, QUDA_STAGGERED_DSLASH, gauge_param.cpu_prec);
    }
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      construct_gauge_field(qdp_fatlink, 1, gauge_param.cpu_prec, &gauge_param);
    } else {
      construct_fat_long_gauge_field(qdp_inlink, qdp_longlink, 1, gauge_param.cpu_prec, &gauge_param,
                                     compute_fatlong ? QUDA_STAGGERED_DSLASH : dslash_type);
    }
    // createSiteLinkCPU(inlink, gauge_param.cpu_prec, 0); // 0 for no phases
  }

  // Compute plaquette. Routine is aware that the gauge fields already have the phases on them.
  double plaq[3];
  computeStaggeredPlaquetteQDPOrder(qdp_inlink, plaq, gauge_param, dslash_type);

  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
      memset(qdp_longlink[dir], 0, V * gauge_site_size * host_gauge_data_type_size);
    }
  } else { // QUDA_ASQTAD_DSLASH

    if (compute_fatlong) {
      computeFatLongGPU(qdp_fatlink, qdp_longlink, qdp_inlink, gauge_param, host_gauge_data_type_size, n_naiks, eps_naik);
    } else { //

      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink[dir], qdp_inlink[dir], V * gauge_site_size * host_gauge_data_type_size);
      }
    }

    computeStaggeredPlaquetteQDPOrder(qdp_fatlink, plaq, gauge_param, dslash_type);

    printfQuda("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }

  // Alright, we've created all the void** links.
  // Create the void* pointers
  reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
  reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  csParam.nColor = 3;
  csParam.nSpin = 1;
  csParam.nDim = 5;
  for (int d = 0; d < 4; d++) csParam.x[d] = gauge_param.X[d];
  bool pc = (inv_param.solution_type == QUDA_MATPC_SOLUTION || inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) csParam.x[0] /= 2;
  csParam.x[4] = 1;

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  csParam.siteSubset = pc ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  in = new cpuColorSpinorField(csParam);
  out = new cpuColorSpinorField(csParam);
  ref = new cpuColorSpinorField(csParam);
  tmp = new cpuColorSpinorField(csParam);

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
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
  } else {
    gauge_param.reconstruct = gauge_param.reconstruct_sloppy = gauge_param.reconstruct_refinement_sloppy
      = QUDA_RECONSTRUCT_NO;
  }
  gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  loadGaugeQuda(milc_fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_refinement_sloppy = link_recon_sloppy;
    gauge_param.reconstruct_precondition = link_recon_precondition;
    loadGaugeQuda(milc_longlink, &gauge_param);
  }

  /* end stuff stolen from staggered_invert_test */

  // if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE)
  // inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  inv_param.solve_type = solve_type; // restore actual solve_type we want to do

  /* ESW HACK: comment this out to do a non-MG solve. */

  // setup the multigrid solver
  void *mg_preconditioner = newMultigridQuda(&mg_param);
  inv_param.preconditioner = mg_preconditioner;

  // Test: create a dummy invert param just to make sure
  // we're setting up gauge fields and such correctly.

  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();
  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];

  for (int k = 0; k < Nsrc; k++) {

    construct_spinor_source(in->V(), 1, 3, inv_param.cpu_prec, csParam.x, *rng);
    invertQuda(out->V(), in->V(), &inv_param);

    time[k] = inv_param.secs;
    gflops[k] = inv_param.gflops / inv_param.secs;
    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
               inv_param.gflops / inv_param.secs);
  }

  rng->Release();
  delete rng;

  double nrm2 = 0;
  double src2 = 0;

  int len = 0;
  if (inv_param.solution_type == QUDA_MAT_SOLUTION || inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION) {
    len = V;
  } else {
    len = Vh;
  }

  // Check solution
  if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

    // In QUDA, the full staggered operator has the sign convention
    //{{m, -D_eo},{-D_oe,m}}, while the CPU verify function does not
    // have the minus sign. Passing in QUDA_DAG_YES solves this
    // discrepancy
    staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&ref->Even()), qdp_fatlink, qdp_longlink, ghost_fatlink,
                     ghost_longlink, reinterpret_cast<cpuColorSpinorField *>(&out->Odd()), QUDA_EVEN_PARITY,
                     QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    staggered_dslash(reinterpret_cast<cpuColorSpinorField *>(&ref->Odd()), qdp_fatlink, qdp_longlink, ghost_fatlink,
                     ghost_longlink, reinterpret_cast<cpuColorSpinorField *>(&out->Even()), QUDA_ODD_PARITY,
                     QUDA_DAG_YES, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
    // if (dslash_type == QUDA_LAPLACE_DSLASH) {
    //  xpay(out->V(), kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
    //  ax(0.5/kappa, ref->V(), ref->Length(), gauge_param.cpu_prec);
    //} else {
    axpy(2 * mass, out->V(), ref->V(), ref->Length(), gauge_param.cpu_prec);
    //}

    // Reference debugging code: print the first component
    // of the even and odd partities within a solution vector.
    /*
    printfQuda("\nLength: %lu\n", ref->Length());

    // for verification
    printfQuda("\n\nEven:\n");
    printfQuda("CUDA: %f\n", ((double*)(in->Even().V()))[0]);
    printfQuda("Soln: %f\n", ((double*)(out->Even().V()))[0]);
    printfQuda("CPU:  %f\n", ((double*)(ref->Even().V()))[0]);

    printfQuda("\n\nOdd:\n");
    printfQuda("CUDA: %f\n", ((double*)(in->Odd().V()))[0]);
    printfQuda("Soln: %f\n", ((double*)(out->Odd().V()))[0]);
    printfQuda("CPU:  %f\n", ((double*)(ref->Odd().V()))[0]);
    printfQuda("\n\n");
    */

    mxpy(in->V(), ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
    src2 = norm_2(in->V(), len * my_spinor_site_size, inv_param.cpu_prec);

  } else if (inv_param.solution_type == QUDA_MATPC_SOLUTION) {

    matdagmat(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, out, mass, 0, inv_param.cpu_prec,
              gauge_param.cpu_prec, tmp, QUDA_EVEN_PARITY, dslash_type);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      printfQuda("%f %f\n", ((float *)in->V())[12], ((float *)ref->V())[12]);
    } else {
      printfQuda("%f %f\n", ((double *)in->V())[12], ((double *)ref->V())[12]);
    }

    mxpy(in->V(), ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
    nrm2 = norm_2(ref->V(), len * my_spinor_site_size, inv_param.cpu_prec);
    src2 = norm_2(in->V(), len * my_spinor_site_size, inv_param.cpu_prec);
  }

  double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
  double l2r = sqrt(nrm2 / src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
             inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);

  // Compute timings
  if (Nsrc > 1) {
    auto mean_time = 0.0;
    auto mean_time2 = 0.0;
    auto mean_gflops = 0.0;
    auto mean_gflops2 = 0.0;
    // skip first solve due to allocations, potential UVM swapping overhead
    for (int i = 1; i < Nsrc; i++) {
      mean_time += time[i];
      mean_time2 += time[i] * time[i];
      mean_gflops += gflops[i];
      mean_gflops2 += gflops[i] * gflops[i];
    }

    auto NsrcM1 = Nsrc - 1;

    mean_time /= NsrcM1;
    mean_time2 /= NsrcM1;
    auto stddev_time = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_time2 - mean_time * mean_time)) :
                                    std::numeric_limits<double>::infinity();
    mean_gflops /= NsrcM1;
    mean_gflops2 /= NsrcM1;
    auto stddev_gflops = NsrcM1 > 1 ?
      sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_gflops2 - mean_gflops * mean_gflops)) :
      std::numeric_limits<double>::infinity();
    printfQuda(
      "%d solves, with mean solve time %g (stddev = %g), mean GFLOPS %g (stddev = %g) [excluding first solve]\n", Nsrc,
      mean_time, stddev_time, mean_gflops, stddev_gflops);
  }

  delete[] time;
  delete[] gflops;

  // Clean up gauge fields, at least
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

  delete in;
  delete out;
  delete ref;
  delete tmp;

  // free the multigrid solver
  destroyMultigridQuda(mg_preconditioner);

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
