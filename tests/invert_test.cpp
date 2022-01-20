#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h> // convenient quark field container

// External headers
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>

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
        printfQuda(" - level %d number of eigenvectors requested n_conv %d\n", i + 1, nvec[i]);
        printfQuda(" - level %d size of eigenvector search space %d\n", i + 1, mg_eig_n_ev[i]);
        printfQuda(" - level %d size of Krylov space %d\n", i + 1, mg_eig_n_kr[i]);
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
  add_eofa_option_group(app);
  add_multigrid_option_group(app);
  add_comms_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (inv_deflate && inv_multigrid) {
    printfQuda("Error: Cannot use both deflation and multigrid preconditioners on top level solve.\n");
    exit(0);
  }

  // If a value greater than 1 is passed, heavier masses will be constructed
  // and the multi-shift solver will be called
  if (multishift > 1) {
    // set a correct default for the multi-shift solver
    solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  if (dslash_type != QUDA_WILSON_DSLASH && dslash_type != QUDA_CLOVER_WILSON_DSLASH
      && dslash_type != QUDA_TWISTED_MASS_DSLASH && dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH
      && dslash_type != QUDA_MOBIUS_DWF_DSLASH && dslash_type != QUDA_MOBIUS_DWF_EOFA_DSLASH
      && dslash_type != QUDA_TWISTED_CLOVER_DSLASH && dslash_type != QUDA_DOMAIN_WALL_DSLASH) {
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
  QudaEigParam mg_eig_param[QUDA_MAX_MG_LEVEL];
  QudaEigParam eig_param = newQudaEigParam();

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

  if (inv_deflate) {
    setEigParam(eig_param);
    inv_param.eig_param = &eig_param;
  } else inv_param.eig_param = nullptr;

  if (inv_split_grid_deflate) {
    // Though no inversions are performed, the inv_param
    // structure contains all the information we need to
    // construct the dirac operator. We encapsualte the
    // inv_param structure inside the eig_param structure
    // to avoid any confusion
    QudaInvertParam eig_inv_param = inv_param;    
    // Specific changes to the invert param for the eigensolver
    // QUDA's device routines require UKQCD gamma basis. QUDA will
    // automatically rotate from this basis on the host, to UKQCD
    // on the device, and back to this basis upon completion.
    //eig_inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
    
    eig_inv_param.solve_type
      = (eig_inv_param.solution_type == QUDA_MAT_SOLUTION ? QUDA_DIRECT_SOLVE : QUDA_DIRECT_PC_SOLVE);
    // Place a copy of the modified Invert param inside Eig param
    eig_param.invert_param = &eig_inv_param;
    setEigParam(eig_param);
    
    inv_param.split_grid_eig_param = &eig_param;
  } else inv_param.split_grid_eig_param = nullptr;

  if (inv_split_grid_deflate && inv_deflate) errorQuda("inv_split_grid_deflate and inv_deflate are mutually exclusive choices");
  
  // All parameters have been set. Display the parameters via stdout
  display_test_info();

  // Initialize the QUDA library
  initQuda(device_ordinal);

  // params corresponds to split grid
  inv_param.split_grid[0] = grid_partition[0];
  inv_param.split_grid[1] = grid_partition[1];
  inv_param.split_grid[2] = grid_partition[2];
  inv_param.split_grid[3] = grid_partition[3];

  int num_sub_partition = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
  bool use_split_grid = num_sub_partition > 1;

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }

  // Allocate host side memory for the gauge field.
  //----------------------------------------------------------------------------
  void *gauge[4];
  // Allocate space on the host (always best to allocate and free in the same scope)
  for (int dir = 0; dir < 4; dir++) gauge[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
  constructHostGaugeField(gauge, gauge_param, argc, argv);
  // Load the gauge field to the device
  loadGaugeQuda((void *)gauge, &gauge_param);

  // Allocate host side memory for clover terms if needed.
  //----------------------------------------------------------------------------
  void *clover = nullptr;
  void *clover_inv = nullptr;
  // Allocate space on the host (always best to allocate and free in the same scope)
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    clover = safe_malloc(V * clover_site_size * host_clover_data_type_size);
    clover_inv = safe_malloc(V * clover_site_size * host_spinor_data_type_size);
    constructHostCloverField(clover, clover_inv, inv_param);
    if (inv_multigrid) {
      // This line ensures that if we need to construct the clover inverse (in either the smoother or the solver) we do so
      if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) {
        inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
      }
    }
    // Load the clover terms to the device
    loadCloverQuda(clover, clover_inv, &inv_param);
    if (inv_multigrid) {
      // Restore actual solve_type we want to do
      inv_param.solve_type = solve_type;
    }
  }

  // Now QUDA is initialised and the fields are loaded, we may setup the preconditioner
  void *mg_preconditioner = nullptr;
  if (inv_multigrid) {
    if (use_split_grid) { errorQuda("Split grid does not work with MG yet."); }
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
  }

  // Compute plaquette as a sanity check
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // Vector construct START
  //-----------------------------------------------------------------------------------
  std::vector<quda::ColorSpinorField *> in(Nsrc);
  std::vector<quda::ColorSpinorField *> out(Nsrc);
  std::vector<quda::ColorSpinorField *> out_multishift(multishift * Nsrc);
  quda::ColorSpinorField *check;
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  check = quda::ColorSpinorField::Create(cs_param);
  std::vector<std::vector<void *>> _hp_multi_x(Nsrc, std::vector<void *>(multishift));

  // QUDA host array for internal checks and malloc
  // Vector construct END
  //-----------------------------------------------------------------------------------

  // Quark masses
  std::vector<double> masses(multishift);

  // QUDA invert test BEGIN
  //----------------------------------------------------------------------------
  if (multishift > 1) {
    if (use_split_grid) { errorQuda("Split grid does not work with multishift yet."); }
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
      for (int n = 0; n < Nsrc; n++) {
        out_multishift[n * multishift + i] = quda::ColorSpinorField::Create(cs_param);
        _hp_multi_x[n][i] = out_multishift[n * multishift + i]->V();
      }
    }
  }

  std::vector<double> time(Nsrc);
  std::vector<double> gflops(Nsrc);
  std::vector<int> iter(Nsrc);

  auto *rng = new quda::RNG(*check, 1234);

  for (int i = 0; i < Nsrc; i++) {
    // Populate the host spinor with random numbers.
    in[i] = quda::ColorSpinorField::Create(cs_param);
    in[i]->Source(QUDA_RANDOM_SOURCE);
    out[i] = quda::ColorSpinorField::Create(cs_param);
    if(inv_test_init_guess) {
      out[i]->Source(QUDA_CONSTANT_SOURCE, 1.0);
      inv_param.use_init_guess = QUDA_USE_INIT_GUESS_YES;
      if(inv_split_grid_deflate) errorQuda("initial guess with split grid deflation not supported");
    }
  }
  
  if (!use_split_grid) {

    for (int i = 0; i < Nsrc; i++) {
      // If deflating, preserve the deflation space between solves
      if (inv_deflate) eig_param.preserve_deflation = i < Nsrc - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      // Perform QUDA inversions
      if (multishift > 1) {
        invertMultiShiftQuda(_hp_multi_x[i].data(), in[i]->V(), &inv_param);
      } else {
        invertQuda(out[i]->V(), in[i]->V(), &inv_param);
      }

      time[i] = inv_param.secs;
      gflops[i] = inv_param.gflops / inv_param.secs;
      iter[i] = inv_param.iter;
      printfQuda("Done: %i iter / %g secs = %g Gflops\n\n", inv_param.iter, inv_param.secs,
                 inv_param.gflops / inv_param.secs);
    }
  } else {
    inv_param.num_src = Nsrc;
    inv_param.num_src_per_sub_partition = Nsrc / num_sub_partition;
    // Raw pointers to solutions and sources
    std::vector<void *> _hp_x(Nsrc);
    std::vector<void *> _hp_b(Nsrc);
    for (int i = 0; i < Nsrc; i++) {
      _hp_x[i] = out[i]->V();
      _hp_b[i] = in[i]->V();
    }
    
    // Run split grid
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
        || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      invertMultiSrcCloverQuda(_hp_x.data(), _hp_b.data(), &inv_param, (void *)gauge, &gauge_param, clover, clover_inv);
    } else {
      if (inv_split_grid_deflate) {

	// Make host side copies of the source (temp WAR)
	std::vector<quda::ColorSpinorField *> in_orig(Nsrc);
	for (int i = 0; i < Nsrc; i++) {
	  in_orig[i] = quda::ColorSpinorField::Create(cs_param);
	  *in_orig[i] = *in[i];
	}
	
	
	inv_param.use_init_guess = QUDA_USE_INIT_GUESS_NO;
	inv_param.true_res = 1e10;
	double gflops = 0.0;
	double secs = 0.0;
	int iter = 0;
	for(int defl_restart = 0; inv_param.true_res > inv_param.tol && defl_restart < inv_split_grid_deflate_maxiter; defl_restart++) {
	  invertMultiSrcQuda(_hp_x.data(), _hp_b.data(), &inv_param, (void *)gauge, &gauge_param);
	  gflops += inv_param.gflops;
	  secs += inv_param.secs;
	  iter += inv_param.iter;
	}
	inv_param.gflops = gflops;
	inv_param.secs = secs;
	inv_param.iter = iter;

	// Copy back original sources
	for (int i = 0; i < Nsrc; i++) {
	  *in[i] = *in_orig[i];
	  delete in_orig[i];
	}
	in_orig.resize(0);
	
      } else {
	invertMultiSrcQuda(_hp_x.data(), _hp_b.data(), &inv_param, (void *)gauge, &gauge_param);	
      } 
    }
   
    comm_allreduce_int(&inv_param.iter);
    inv_param.iter /= comm_size() / num_sub_partition;
    comm_allreduce(&inv_param.gflops);
    inv_param.gflops /= comm_size() / num_sub_partition;
    comm_allreduce_max(&inv_param.secs);
    printfQuda("Done: %d sub-partitions - %i iter / %g secs = %g Gflops\n\n", num_sub_partition, inv_param.iter,
               inv_param.secs, inv_param.gflops / inv_param.secs);
  }

  // QUDA invert test COMPLETE
  //----------------------------------------------------------------------------

  delete rng;

  // free the multigrid solver
  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);

  // Compute performance statistics
  if (Nsrc > 1 && !use_split_grid) performanceStats(time, gflops, iter);

  // Perform host side verification of inversion if requested
  if (verify_results) {
    for (int i = 0; i < Nsrc; i++) {
      verifyInversion(out[i]->V(), _hp_multi_x[i].data(), in[i]->V(), check->V(), gauge_param, inv_param, gauge, clover,
                      clover_inv);
    }
  }

  // Clean up memory allocations
  delete check;
  if (multishift > 1) {
    for (auto p : out_multishift) { delete p; }
  }

  for (auto p : in) { delete p; }  
  for (auto p : out) { delete p; }

  freeGaugeQuda();
  for (int dir = 0; dir < 4; dir++) host_free(gauge[dir]);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    freeCloverQuda();
    if (clover) host_free(clover);
    if (clover_inv) host_free(clover_inv);
  }

  // finalize the QUDA library
  endQuda();
  finalizeComms();

  return 0;
}
