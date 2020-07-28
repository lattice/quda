#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

// QUDA headers
#include <quda.h>
#include <color_spinor_field.h> // convenient quark field container
#include <vector_io.h>
#include <blas_quda.h>
#include <dslash_quda.h>

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
  add_propagator_option_group(app);
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

  // Initialize the QUDA library
  initQuda(device);

  setVerbosity(verbosity);
  
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
  QudaEigParam mg_eig_param[mg_levels];
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
  } else {
    inv_param.eig_param = nullptr;
  }

  // All parameters have been set. Display the parameters via stdout
  display_test_info();

  // set parameters for the reference Dslash, and prepare fields to be loaded
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
      || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
    dw_setDims(gauge_param.X, inv_param.Ls);
  } else {
    setDims(gauge_param.X);
  }
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
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
  }

  // Compute plaquette as a sanity check
  double plaq[3];
  plaqQuda(plaq);
  printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);

  // Vector construct START
  //-----------------------------------------------------------------------------------
  // Source/Sink params
  quda::ColorSpinorField *in;
  quda::ColorSpinorField *out;
  quda::ColorSpinorField *check;
  quda::ColorSpinorParam cs_param;
  constructWilsonTestSpinorParam(&cs_param, &inv_param, &gauge_param);
  in = quda::ColorSpinorField::Create(cs_param);
  out = quda::ColorSpinorField::Create(cs_param);
  check = quda::ColorSpinorField::Create(cs_param);
  
  // Make Propagator params
  QudaDslashType dslash_type_orig = dslash_type;
  dslash_type = QUDA_WILSON_DSLASH;  
  QudaInvertParam inv_param4D;
  setInvertParam(inv_param4D);  
  quda::ColorSpinorParam cs_param4D;
  constructWilsonTestSpinorParam(&cs_param4D, &inv_param4D, &gauge_param);  

  // Host array for Propagators and 4D sources
  std::vector<quda::ColorSpinorField *> qudaQuark4D(12);
  std::vector<quda::ColorSpinorField *> qudaIn4D(12);
  // Allocate contiguous host memory
  void *out_array = (void*)malloc(12*V*3*4*2*sizeof(double));
  void *in_array = (void*)malloc(12*V*3*4*2*sizeof(double));

  // This flag instructs QUDA to use pre-allocated memory.
  cs_param4D.create = QUDA_REFERENCE_FIELD_CREATE;
  for (int dil = 0; dil < 12; dil++) {
    cs_param4D.v = (double*)in_array + dil*V*3*4*2;
    qudaIn4D[dil] = quda::ColorSpinorField::Create(cs_param4D);
    
    cs_param4D.v = (double*)out_array + dil*V*3*4*2;
    qudaQuark4D[dil] = quda::ColorSpinorField::Create(cs_param4D);
  }  

  // Construct 4D smearing parameters.
  // Borrow problem 4D parameters, then adjust
  QudaInvertParam inv_param_smear = newQudaInvertParam();
  dslash_type = QUDA_LAPLACE_DSLASH;  
  double coeff = -(prop_source_smear_coeff * prop_source_smear_coeff) / (4 * prop_source_smear_steps);
  double kappa_orig = kappa;
  double mass_orig = mass;
  mass = 1.0/coeff;
  kappa = -1.0;
  setInvertParam(inv_param_smear);
  //inv_param_smear.kappa = coeff;
  inv_param_smear.mass_normalization = QUDA_MASS_NORMALIZATION;  
  inv_param_smear.laplace3D = 3; // Omit t-dim
  inv_param_smear.solution_type = QUDA_MAT_SOLUTION;
  inv_param_smear.solve_type = QUDA_DIRECT_SOLVE;
  
  // Restore dslash type
  dslash_type = dslash_type_orig;  
  mass = mass_orig;
  kappa = kappa_orig;
  // Vector construct END
  //-----------------------------------------------------------------------------------

  // Contraction construct START
  //-----------------------------------------------------------------------------------
  size_t data_size = (prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t array_size = (contract_type == QUDA_CONTRACT_TYPE_OPEN || contract_type == QUDA_CONTRACT_TYPE_DR) ? V : tdim; 
  void *correlation_function = (double*)pinned_malloc(2 * 16 * comm_dim(3) * array_size * data_size);
  void *correlation_function_sum = (double*)pinned_malloc(2 * 16 * comm_dim(3) * array_size * data_size);
  memset(correlation_function_sum, 0, 2 * 16 * comm_dim(3) * array_size * data_size);
  int gamma_mat = 0;
  // Contraction construct END
  //-----------------------------------------------------------------------------------

  // QUDA propagator test BEGIN
  //-----------------------------------------------------------------------------------
  auto *rng = new quda::RNG(quda::LatticeFieldParam(gauge_param), 1234);
  rng->Init();

  double *time = new double[12];
  double *gflops = new double[12];

  for(int n=0; n<prop_n_sources; n++) {
    
    if (strcmp(prop_source_infile[n], "") != 0) {
      
      std::string infile(prop_source_infile[n]);
      std::vector<quda::ColorSpinorField *> src_ptr;
      src_ptr.reserve(12);
      for (int dil = 0; dil < 12; dil++) src_ptr.push_back(qudaIn4D[dil]);
      
      // load the source
      quda::VectorIO io(infile, QUDA_BOOLEAN_TRUE);
      io.load(src_ptr);

    } else {
      printfQuda("Source position: %d %d %d %d\n", prop_source_position[n][0], prop_source_position[n][1], prop_source_position[n][2], prop_source_position[n][3]);
    }

    for (int dil = 0; dil < 12; dil++) {
      
      size_t vol_bytes = V * my_spinor_site_size * host_spinor_data_type_size;

      memset(in->V(), Ls * vol_bytes, 0.0);
      memset(out->V(), Ls * vol_bytes, 0.0);
      
      if (strcmp(prop_source_infile[n], "") != 0) {
	memcpy(in->V(), qudaIn4D[dil]->V(), vol_bytes);
      } else {
	const int source[4] = {prop_source_position[n][0], 
			       prop_source_position[n][1], 
			       prop_source_position[n][2],
			       prop_source_position[n][3]};
	
	constructPointSpinorSource(qudaIn4D[dil]->V(), 4, 3, inv_param.cpu_prec, gauge_param.X, dil, source);
	//constructWallSpinorSource(qudaIn4D[dil]->V(), inv_param.cpu_prec, dil);
	
	// Gaussian smear the point source.
	performGaussianSmearNStep(qudaIn4D[dil]->V(), &inv_param_smear, prop_source_smear_steps);
	
	for(int i=0; i<5; i++) {
	  //qudaIn4D[dil]->PrintVector(i);
	}

	memcpy(in->V(), qudaIn4D[dil]->V(), vol_bytes);
      }
      
      // If deflating, preserve the deflation space between solves
      if (inv_deflate) eig_param.preserve_deflation = dil < 12 - 1 ? QUDA_BOOLEAN_TRUE : QUDA_BOOLEAN_FALSE;
      // Perform QUDA inversions
      inv_param.solver_normalization = QUDA_SOURCE_NORMALIZATION;
      invertQuda(out->V(), in->V(), &inv_param);
      
      time[dil] = inv_param.secs;
      gflops[dil] = inv_param.gflops / inv_param.secs;
      printfQuda("Prop %d done: %d iter / %g secs = %g Gflops\n\n", dil, inv_param.iter, inv_param.secs,
		 inv_param.gflops / inv_param.secs);

      if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || 
	  dslash_type == QUDA_MOBIUS_DWF_DSLASH || 
	  dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
	printfQuda("Constructing 4D prop from DWF prop\n");
	make4DQuarkProp(qudaQuark4D[dil]->V(), out->V(), &inv_param, &inv_param4D, gauge_param.X);
      } else {
	memcpy(qudaQuark4D[dil]->V(), out->V(), vol_bytes);
      }
      
      // Perform host side verification of inversion if requested
      if (verify_results) {
	verifyInversion(out->V(), in->V(), check->V(), gauge_param, inv_param, gauge, clover, clover_inv);
      }

      // Gaussian smear the sink.
      performGaussianSmearNStep(qudaQuark4D[dil]->V(), &inv_param_smear, prop_sink_smear_steps);

      for(int i=0; i<5; i++) {
	//qudaQuark4D[dil]->PrintVector(i);
      }
      
      // Perform GPU contraction.
      // Host side spinor data and contraction_result passed to QUDA.
      // QUDA will allocate GPU memory, transfer the data,
      // perform the requested contraction, and return the
      // result in the array contraction_result     
      contractQuda(qudaQuark4D[dil]->V(), qudaQuark4D[dil]->V(), 
		   ((double*)correlation_function) + 2*16*tdim*comm_coord(3), contract_type, &inv_param, gauge_param.X);

      comm_gather_array((double*)correlation_function, 2*16*tdim);
      for(int t=0; t<comm_dim(3) * tdim; t++) {
	//printfQuda("t=%d %e %e\n", t, ((double*)correlation_function)[2*(16*t + gamma_mat)], ((double*)correlation_function)[2*(16*t + gamma_mat) + 1]);
	((double*)correlation_function_sum)[2*(16*t + gamma_mat)] += ((double*)correlation_function)[2*(16*t + gamma_mat)];
	((double*)correlation_function_sum)[2*(16*t + gamma_mat)+1] += ((double*)correlation_function)[2*(16*t + gamma_mat)+1];
      }
    }

    for(int t=0; t<comm_dim(3) * tdim; t++) {
      printfQuda("sum t=%d %e %e\n", t, ((double*)correlation_function_sum)[2*(16*t + gamma_mat)], ((double*)correlation_function_sum)[2*(16*t + gamma_mat) + 1]);
    }
    
    for(int t=0; t<comm_dim(3) * tdim; t++) {
      printfQuda("ratio t=%d %e\n", t, (((double*)correlation_function_sum)[2*(16*t + gamma_mat)])/(((double*)correlation_function_sum)[2*(16*(t+1) + gamma_mat)]));
    }


    // Compute performance statistics
    Nsrc = 12;
    performanceStats(time, gflops);

    if (strcmp(prop_sink_outfile[n], "") != 0) {

      quda::VectorIO io(prop_sink_outfile[n], QUDA_BOOLEAN_TRUE);
      
      // Make an array of size qudaQuark4D.size()
      std::vector<quda::ColorSpinorField *> prop_ptr;
      prop_ptr.reserve(qudaQuark4D.size());
      
      // Down prec the props if requested, or just save
      if (prop_save_prec < prec) {
	io.downPrec(qudaQuark4D, prop_ptr, prop_save_prec);
	// save the vectors      
	io.save(prop_ptr);
	for (unsigned int i = 0; i < qudaQuark4D.size() && prop_save_prec < prec; i++) delete prop_ptr[i];	
      } else {
	for (unsigned int i = 0; i < qudaQuark4D.size(); i++) prop_ptr.push_back(qudaQuark4D[i]);
	// save the vectors
	io.save(prop_ptr);
      }
    }
  }    
    
  // QUDA propagator test COMPLETE
  //----------------------------------------------------------------------------

  // Free the multigrid solver
  if (inv_multigrid) destroyMultigridQuda(mg_preconditioner);
  
  // Clean up memory allocations
  rng->Release();
  delete rng;

  delete[] time;
  delete[] gflops;    

  delete in;  
  delete out;
  delete check;
  
  for (int i = 0; i < 12; i++) delete qudaQuark4D[i];

  freeGaugeQuda();
  for (int dir = 0; dir < 4; dir++) free(gauge[dir]);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    freeCloverQuda();
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  // finalize the QUDA library
  endQuda();
  finalizeComms();

  return 0;
}
