#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
// This extended test utilizes some QUDA internals
#include <qio_field.h>
#include <gauge_field.h>
#include <pgauge_monte.h>
#include <unitarization_links.h>

#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

namespace quda {
  extern void setTransferGPU(bool);
}

void setReunitarizationConsts()
{
  using namespace quda;
  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;
  const int reunit_allow_svd = 1;
  const int reunit_svd_only = 0;
  const double svd_rel_error = 1e-6;
  const double svd_abs_error = 1e-6;
  setUnitarizeLinksConstants(unitarize_eps, max_error, reunit_allow_svd, reunit_svd_only, svd_rel_error, svd_abs_error);
}

void CallUnitarizeLinks(quda::cudaGaugeField *cudaInGauge)
{
  using namespace quda;
  int *num_failures_dev = (int *)device_malloc(sizeof(int));
  int num_failures;
  cudaMemset(num_failures_dev, 0, sizeof(int));
  unitarizeLinks(*cudaInGauge, num_failures_dev);

  cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
  if (num_failures > 0) errorQuda("Error in the unitarization\n");
  device_free(num_failures_dev);
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
      if (mg_eig_type[i] == QUDA_EIG_BLK_TR_LANCZOS) printfQuda(" - eigenvector block size %d\n", mg_eig_block_size[i]);
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
}

int main(int argc, char **argv)
{
  setQudaDefaultMgTestParams();
  // command line options
  auto app = make_app();
  add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  // Only these fermions are supported in this file
  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  if (inv_multigrid) {
    // Only these fermions are supported with MG
    if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
        && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
      printfQuda("dslash_type %d not supported for MG\n", dslash_type);
      exit(0);
    }

    // Only these solve types are supported with MG
    if (solve_type != QUDA_DIRECT_SOLVE && solve_type != QUDA_DIRECT_PC_SOLVE) {
      printfQuda("Solve_type %d not supported with MG. Please use QUDA_DIRECT_SOLVE or QUDA_DIRECT_PC_SOLVE\n\n",
                 solve_type);
      exit(0);
    }
  }

  // Set QUDA's internal parameters
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaEigParam mg_eig_param[mg_levels];

  if (inv_multigrid) {
    setQudaMgSolveTypes();
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
  } else {
    setInvertParam(inv_param);
  }

  // All parameters have been set. Display the parameters via stdout
  display_test_info();

  // initialize the QUDA library
  initQuda(device);

  // *** Everything between here and the timer is application specific
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
    // This line ensures that if we need to construct the clover inverse (in either the smoother or the solver) we do so
    if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) {
      inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    }
    // Load the clover terms to the device
    loadCloverQuda(clover, clover_inv, &inv_param);
    // Restore actual solve_type we want to do
    inv_param.solve_type = solve_type;
  }

  void *spinorIn = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
  void *spinorCheck = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);
  void *spinorOut = malloc(V * spinor_site_size * host_spinor_data_type_size * inv_param.Ls);

  // start the timer
  double time0 = -((double)clock());
  {
    using namespace quda;
    GaugeFieldParam gParam(0, gauge_param);
    gParam.pad = 0;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.create      = QUDA_NULL_FIELD_CREATE;
    gParam.link_type   = gauge_param.type;
    gParam.reconstruct = gauge_param.reconstruct;
    gParam.setPrecision(gParam.Precision(), true);
    cudaGaugeField *gauge = new cudaGaugeField(gParam);

    int pad = 0;
    int y[4];
    int R[4] = {0,0,0,0};
    for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
    for(int dir=0; dir<4; ++dir) y[dir] = gauge_param.X[dir] + 2 * R[dir];
    GaugeFieldParam gParamEx(y, prec, link_recon,
			     pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = gParam.order;
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = gParam.t_boundary;
    gParamEx.nFace = 1;
    for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
    cudaGaugeField *gaugeEx = new cudaGaugeField(gParamEx);

    QudaGaugeObservableParam obs_param = newQudaGaugeObservableParam();
    obs_param.compute_plaquette = QUDA_BOOLEAN_TRUE;
    obs_param.compute_qcharge = QUDA_BOOLEAN_TRUE;

    // CURAND random generator initialization
    RNG *randstates = new RNG(*gauge, 1234);
    randstates->Init();
    int nsteps = 10;
    int nhbsteps = 1;
    int novrsteps = 1;
    bool coldstart = false;
    double beta_value = 6.2;

    if(link_recon != QUDA_RECONSTRUCT_8 && coldstart) InitGaugeField( *gaugeEx);
    else
      InitGaugeField(*gaugeEx, *randstates);
    // Reunitarization setup
    setReunitarizationConsts();

    // Do a series of Heatbath updates
    Monte(*gaugeEx, *randstates, beta_value, 100 * nhbsteps, 100 * novrsteps);

    // Copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

    // load the gauge field from gauge
    gauge_param.gauge_order = gauge->Order();
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;
    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
    gaugeObservablesQuda(&obs_param);

    // Demonstrate MG evolution on an evolving gauge field
    //----------------------------------------------------
    printfQuda("\n======================================================\n");
    printfQuda("Running pure gauge evolution test at constant quark mass\n");
    printfQuda("======================================================\n");
    printfQuda("step=%d plaquette = %g topological charge = %g, mass = %g kappa = %g, mu = %g\n", 0,
               obs_param.plaquette[0], obs_param.qcharge, inv_param.mass, inv_param.kappa, inv_param.mu);

    // this line ensure that if we need to construct the clover inverse (in either the smoother or the solver) we do so
    if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE)
      inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH)
      loadCloverQuda(clover, clover_inv, &inv_param);
    inv_param.solve_type = solve_type; // restore actual solve_type we want to do

    // Create a point source at 0 (in each subvolume...  FIXME)
    memset(spinorIn, 0, inv_param.Ls * V * spinor_site_size * host_spinor_data_type_size);
    memset(spinorCheck, 0, inv_param.Ls * V * spinor_site_size * host_spinor_data_type_size);
    memset(spinorOut, 0, inv_param.Ls * V * spinor_site_size * host_spinor_data_type_size);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      for (int i = 0; i < inv_param.Ls * V * spinor_site_size; i++) ((float *)spinorIn)[i] = rand() / (float)RAND_MAX;
    } else {
      for (int i = 0; i < inv_param.Ls * V * spinor_site_size; i++) ((double *)spinorIn)[i] = rand() / (double)RAND_MAX;
    }

    // Setup the multigrid solver
    void *mg_preconditioner = nullptr;
    if (inv_multigrid) {
      mg_preconditioner = newMultigridQuda(&mg_param);
      inv_param.preconditioner = mg_preconditioner;
    }
    invertQuda(spinorOut, spinorIn, &inv_param);

    for (int step = 1; step < nsteps; ++step) {
      freeGaugeQuda();
      Monte( *gaugeEx, *randstates, beta_value, nhbsteps, novrsteps);

      // Reunitarize gauge links
      CallUnitarizeLinks(gaugeEx);

      // Copy into regular field
      copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);
      loadGaugeQuda(gauge->Gauge_p(), &gauge_param);

      // Recompute Gauge Observables
      gaugeObservablesQuda(&obs_param);
      printfQuda("step=%d plaquette = %g topological charge = %g, mass = %g kappa = %g, mu = %g\n", step,
                 obs_param.plaquette[0], obs_param.qcharge, inv_param.mass, inv_param.kappa, inv_param.mu);

      // Update the multigrid operator for new gauge and clover fields
      if (inv_multigrid) updateMultigridQuda(mg_preconditioner, &mg_param);
      invertQuda(spinorOut, spinorIn, &inv_param);

      if (inv_multigrid && inv_param.iter == inv_param.maxiter) {
        char vec_outfile[QUDA_MAX_MG_LEVEL][256];
        for (int i=0; i<mg_param.n_level; i++) {
          strcpy(vec_outfile[i], mg_param.vec_outfile[i]);
          sprintf(mg_param.vec_outfile[i], "dump_step_evolve_%d", step);
        }
        warningQuda("Solver failed to converge within max iteration count - dumping null vectors to %s",
                    mg_param.vec_outfile[0]);

        dumpMultigridQuda(mg_preconditioner, &mg_param);
        for (int i=0; i<mg_param.n_level; i++) {
          strcpy(mg_param.vec_outfile[i], vec_outfile[i]); // restore output file name
        }
      }
    }

    // free the multigrid solver
    if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);

    // Demonstrate MG evolution on a fixed gauge field and different masses
    //---------------------------------------------------------------------
    // setup the multigrid solver
    printfQuda("\n====================================================\n");
    printfQuda("Running MG mass scaling test at constant gauge field\n");
    printfQuda("====================================================\n");

    if (inv_multigrid) {
      mg_param.preserve_deflation = mg_eig_preserve_deflation ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      for (int i = 0; i < mg_param.n_level; i++) mg_param.setup_maxiter_refresh[i] = 0;
      mg_preconditioner = newMultigridQuda(&mg_param);
      inv_param.preconditioner = mg_preconditioner;
    }

    invertQuda(spinorOut, spinorIn, &inv_param);

    freeGaugeQuda();

    // Reunitarize gauge links...
    CallUnitarizeLinks(gaugeEx);

    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
    // Recompute Gauge Observables
    gaugeObservablesQuda(&obs_param);

    for (int step = 1; step < nsteps; ++step) {

      // Increment the mass/kappa and mu values to emulate heavy/light flavour updates
      if (kappa == -1.0) {
        inv_param.mass = mass + 0.01 * step;
        inv_param.kappa = 1.0 / (2.0 * (1 + 3 / anisotropy + mass + 0.01 * step));
      } else {
        inv_param.kappa = kappa - 0.001 * step;
        inv_param.mass = 0.5 / (kappa - 0.001 * step) - (1 + 3 / anisotropy);
      }
      if (inv_multigrid) {
        mg_param.invert_param->mass = inv_param.mass;
        mg_param.invert_param->kappa = inv_param.kappa;
      }

      if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        // Multiply by -1.0 to emulate twist switch
        inv_param.mu = -1.0 * mu + 0.01 * step;
        if (inv_multigrid) mg_param.invert_param->mu = inv_param.mu;
      }

      printfQuda("step=%d plaquette = %g topological charge = %g, mass = %g kappa = %g, mu = %g\n", step,
                 obs_param.plaquette[0], obs_param.qcharge, inv_param.mass, inv_param.kappa, inv_param.mu);

      if (inv_multigrid)
        updateMultigridQuda(mg_preconditioner, &mg_param); // update the multigrid operator for new mass and mu values
      invertQuda(spinorOut, spinorIn, &inv_param);

      if (inv_multigrid && inv_param.iter == inv_param.maxiter) {
        char vec_outfile[QUDA_MAX_MG_LEVEL][256];
        for (int i = 0; i < mg_param.n_level; i++) {
          strcpy(vec_outfile[i], mg_param.vec_outfile[i]);
          sprintf(mg_param.vec_outfile[i], "dump_step_shift_%d", step);
        }
        warningQuda("Solver failed to converge within max iteration count - dumping null vectors to %s",
                    mg_param.vec_outfile[0]);

        dumpMultigridQuda(mg_preconditioner, &mg_param);
        for (int i = 0; i < mg_param.n_level; i++) {
          strcpy(mg_param.vec_outfile[i], vec_outfile[i]); // restore output file name
        }
      }
    }

    // free the multigrid solver
    if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);

    delete gauge;
    delete gaugeEx;
    //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
    PGaugeExchangeFree();

    randstates->Release();
    delete randstates;
  }

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", inv_param.iter, inv_param.secs,
             inv_param.gflops / inv_param.secs, time0);

  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  for (int dir = 0; dir<4; dir++) free(gauge[dir]);

  free(spinorIn);
  free(spinorCheck);
  free(spinorOut);

  return 0;
}
