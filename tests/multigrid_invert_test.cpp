#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <limits>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"
#include <test_params.h>

#include <qio_field.h>
#include <color_spinor_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda {
  extern void setTransferGPU(bool);
}

void
display_test_info()
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
  // We give here the default values to some of the array
  solve_type = QUDA_DIRECT_PC_SOLVE;
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
    smoother_type[i] = QUDA_MR_INVERTER;
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

  // command line options
  auto app = make_app();
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
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
  for (int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    if (coarse_solve_type[i] == QUDA_INVALID_SOLVE) coarse_solve_type[i] = solve_type;
    if (smoother_solve_type[i] == QUDA_INVALID_SOLVE) smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE;
  }

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setMultigridInvertParam(inv_param);

  QudaMultigridParam mg_param = newQudaMultigridParam();
  // Set sub structures
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];
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
  mg_param.invert_param = &mg_inv_param;
  setMultigridParam(mg_param);

  display_test_info();

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  setDims(gauge_param.X);

  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover=0, *clover_inv=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gauge_site_size*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate an SU(3) field
    if (unit_gauge) {
      // unit SU(3) field
      construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    } else {
      // random SU(3) field
      construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    }
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = inv_param.clover_cpu_prec;
    clover = malloc(V*clover_site_size*cSize);
    clover_inv = malloc(V*clover_site_size*cSize);
    if (!compute_clover) construct_clover_field(clover, norm, diag, inv_param.clover_cpu_prec);

    inv_param.compute_clover = compute_clover;
    if (compute_clover) inv_param.return_clover = 1;
    inv_param.compute_clover_inverse = 1;
    inv_param.return_clover_inverse = 1;
  }

  void *spinorIn = malloc(V*spinor_site_size*sSize*inv_param.Ls);
  void *spinorCheck = malloc(V*spinor_site_size*sSize*inv_param.Ls);
  void *spinorOut = malloc(V * spinor_site_size * sSize * inv_param.Ls);

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // this line ensure that if we need to construct the clover inverse (in either the smoother or the solver) we do so
  if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);

  inv_param.solve_type = solve_type; // restore actual solve_type we want to do

  // setup the multigrid solver
  void *mg_preconditioner = newMultigridQuda(&mg_param);
  inv_param.preconditioner = mg_preconditioner;

  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();
  double *time = new double[Nsrc];
  double *gflops = new double[Nsrc];

  for (int i = 0; i < Nsrc; i++) {
    construct_spinor_source(spinorIn, 4, 3, inv_param.cpu_prec, gauge_param.X, *rng);
    invertQuda(spinorOut, spinorIn, &inv_param);

    time[i] = inv_param.secs;
    gflops[i] = inv_param.gflops / inv_param.secs;
    printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
               inv_param.gflops / inv_param.secs);
  }

  rng->Release();
  delete rng;

  // free the multigrid solver
  destroyMultigridQuda(mg_preconditioner);

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

  if (inv_param.solution_type == QUDA_MAT_SOLUTION) {

    if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
      if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
        tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
               inv_param.cpu_prec, gauge_param);
      } else {
        int tm_offset = V * spinor_site_size;
        void *evenOut = spinorCheck;
        void *oddOut = (char *)evenOut + tm_offset * cpu_prec;

        void *evenIn = spinorOut;
        void *oddIn = (char *)evenIn + tm_offset * cpu_prec;

        tm_ndeg_mat(evenOut, oddOut, gauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, 0,
                    inv_param.cpu_prec, gauge_param);
      }
    } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      tmc_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0,
              inv_param.cpu_prec, gauge_param);
    } else if (dslash_type == QUDA_WILSON_DSLASH) {
      wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
    } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      clover_mat(spinorCheck, gauge, clover, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);
    } else {
      errorQuda("Unsupported dslash_type");
    }
    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH && twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
        ax(0.5 / inv_param.kappa, spinorCheck, 2 * V * spinor_site_size, inv_param.cpu_prec);
      } else {
        ax(0.5 / inv_param.kappa, spinorCheck, V * spinor_site_size, inv_param.cpu_prec);
      }
    }

  } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {

    if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
      if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) {
        int tm_offset = Vh * spinor_site_size;
        void *out0 = spinorCheck;
        void *out1 = (char *)out0 + tm_offset * cpu_prec;

        void *in0 = spinorOut;
        void *in1 = (char *)in0 + tm_offset * cpu_prec;

        tm_ndeg_matpc(out0, out1, gauge, in0, in1, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                      inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
      } else {
        tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
      }
    } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      if (inv_param.twist_flavor != QUDA_TWIST_SINGLET) errorQuda("Twisted mass solution type not supported");
      tmc_matpc(spinorCheck, gauge, spinorOut, clover, clover_inv, inv_param.kappa, inv_param.mu,
                inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
    } else if (dslash_type == QUDA_WILSON_DSLASH) {
      wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
    } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      clover_matpc(spinorCheck, gauge, clover, clover_inv, spinorOut, inv_param.kappa, inv_param.matpc_type, 0,
                   inv_param.cpu_prec, gauge_param);
    } else {
      errorQuda("Unsupported dslash_type");
    }

    if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
      ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, Vh*spinor_site_size, inv_param.cpu_prec);
    }

  }

  int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
  mxpy(spinorIn, spinorCheck, vol*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
  double nrm2 = norm_2(spinorCheck, vol*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
  double src2 = norm_2(spinorIn, vol*spinor_site_size*inv_param.Ls, inv_param.cpu_prec);
  double l2r = sqrt(nrm2 / src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
	     inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);


  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  free(spinorIn);
  free(spinorCheck);
  free(spinorOut);

  // finalize the communications layer
  finalizeComms();

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  for (int dir = 0; dir<4; dir++) free(gauge[dir]);

  return 0;
}
