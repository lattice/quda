#include <multigrid.h>
#include <qio_field.h>
#include <string.h>

#include <eigensolve_quda.h>

namespace quda {  

  using namespace blas;

  static bool debug = false;

  MG::MG(MGParam &param, TimeProfile &profile_global)
    : Solver(param, profile), param(param), transfer(0), resetTransfer(false), presmoother(nullptr), postsmoother(nullptr),
      profile_global(profile_global),
      profile( "MG level " + std::to_string(param.level+1), false ),
      coarse(nullptr), fine(param.fine), coarse_solver(nullptr),
      param_coarse(nullptr), param_presmooth(nullptr), param_postsmooth(nullptr), param_coarse_solver(nullptr),
      r(nullptr), r_coarse(nullptr), x_coarse(nullptr), tmp_coarse(nullptr),
      diracResidual(param.matResidual->Expose()), diracSmoother(param.matSmooth->Expose()), diracSmootherSloppy(param.matSmoothSloppy->Expose()),
      diracCoarseResidual(nullptr), diracCoarseSmoother(nullptr), diracCoarseSmootherSloppy(nullptr),
      matCoarseResidual(nullptr), matCoarseSmoother(nullptr), matCoarseSmootherSloppy(nullptr),
      rng(nullptr)
  {
    postTrace();

    // for reporting level 1 is the fine level but internally use level 0 for indexing
    sprintf(prefix,"MG level %d (%s): ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
    setVerbosity(param.mg_global.verbosity[param.level]);
    setOutputPrefix(prefix);

    if (param.level >= QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level+1);

    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE)
      errorQuda("Cannot use preconditioned coarse grid solution without preconditioned smoother solve");

    // allocating vectors
    {
      // create residual vectors
      ColorSpinorParam csParam(*(param.B[0]));
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.location = param.location;
      csParam.setPrecision(param.mg_global.invert_param->cuda_prec_sloppy, QUDA_INVALID_PRECISION,
                           csParam.location == QUDA_CUDA_FIELD_LOCATION ? true : false);
      if (csParam.location==QUDA_CUDA_FIELD_LOCATION) {
        csParam.gammaBasis = param.level > 0 ? QUDA_DEGRAND_ROSSI_GAMMA_BASIS: QUDA_UKQCD_GAMMA_BASIS;
      }
      if (param.B[0]->Nspin() == 1) csParam.gammaBasis = param.B[0]->GammaBasis(); // hack for staggered to avoid unnecessary basis checks
      r = ColorSpinorField::Create(csParam);

      // if we're using preconditioning then allocate storage for the preconditioned source vector
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
      	csParam.x[0] /= 2;
      	csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      	b_tilde = ColorSpinorField::Create(csParam);
      }
    }

    if (param.level < param.Nlevel-1) {      
      if (param.mg_global.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
        if (param.mg_global.generate_all_levels == QUDA_BOOLEAN_YES || param.level == 0) {
	  
          if (param.B[0]->Location() == QUDA_CUDA_FIELD_LOCATION) {
            rng = new RNG(param.B[0]->Volume(), 1234, param.B[0]->X());
            rng->Init();
          }
	  
          // Initializing to random vectors, allocate eval memory
          for(int i=0; i<(int)param.B.size(); i++) {
            if (param.B[i]->Location() == QUDA_CPU_FIELD_LOCATION) r->Source(QUDA_RANDOM_SOURCE);
            else spinorNoise(*r, *rng, QUDA_NOISE_UNIFORM);
            *param.B[i] = *r;
	    param.evals.push_back(0.0);
          }
        }
	if ( param.mg_global.num_setup_iter[param.level] > 0 ) {
	  if (strcmp(param.mg_global.vec_infile,"") !=0 ) { // only load if infile is defined and not computing
	    loadVectors(param.B);
	  } else if (param.mg_global.use_low_modes) {
	    generateEigenVectors(); //Run the eigensolver
	  } else {
	    generateNullVectors(param.B);
	  }
	}
      } else if (strcmp(param.mg_global.vec_infile,"")!=0) { // only load if infile is defined and not computing
	loadVectors(param.B);
      } else { // generate free field vectors
	buildFreeVectors(param.B);
      }
    }
    
  
    // in case of iterative setup with MG the coarse level may be already built
    if (!transfer) reset();
    
    setOutputPrefix("");
    postTrace();
  }

  void MG::reset(bool refresh) {
    
    postTrace();
    setVerbosity(param.mg_global.verbosity[param.level]);
    setOutputPrefix(prefix);

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("%s level %d of %d levels\n", transfer ? "Resetting":"Creating", param.level+1, param.Nlevel);

    destroySmoother();
    destroyCoarseSolver();

    // reset the Dirac operator pointers since these may have changed
    diracResidual = param.matResidual->Expose();
    diracSmoother = param.matSmooth->Expose();
    diracSmootherSloppy = param.matSmoothSloppy->Expose();

    // Refresh the null-space vectors if we need to
    if (refresh && param.level < param.Nlevel-1) {
      if (param.mg_global.setup_maxiter_refresh[param.level]) generateNullVectors(param.B, refresh);
    }

    // if not on the coarsest level, update next
    if (param.level < param.Nlevel-1) {

      if (transfer) {
        // restoring FULL parity in Transfer changed at the end of this procedure
        transfer->setSiteSubset(QUDA_FULL_SITE_SUBSET, QUDA_INVALID_PARITY);
        if (resetTransfer || refresh) {
          transfer->reset();
          resetTransfer = false;
        }
      } else {
        // create transfer operator
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating transfer operator\n");
        transfer = new Transfer(param.B, param.Nvec, param.geoBlockSize, param.spinBlockSize,
                                param.mg_global.precision_null[param.level], profile);
        for (int i=0; i<QUDA_MAX_MG_LEVEL; i++) param.mg_global.geo_block_size[param.level][i] = param.geoBlockSize[i];

        // create coarse temporary vector
        tmp_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, r->Precision(), param.mg_global.location[param.level+1]);

        // create coarse residual vector
        r_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, r->Precision(), param.mg_global.location[param.level+1]);

        // create coarse solution vector
        x_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, r->Precision(), param.mg_global.location[param.level+1]);

        B_coarse = new std::vector<ColorSpinorField*>();
        int nVec_coarse = (param.level == param.Nlevel-2 ?  param.mg_global.n_vec[param.level+1] : std::max(param.Nvec, param.mg_global.n_vec[param.level+1]));
        B_coarse->resize(nVec_coarse);
	
        // only have single precision B vectors on the coarse grid
        QudaPrecision B_coarse_precision = std::max(param.mg_global.precision_null[param.level+1], QUDA_SINGLE_PRECISION);
        for (int i=0; i<nVec_coarse; i++)
          (*B_coarse)[i] = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, B_coarse_precision, param.mg_global.setup_location[param.level+1]);

        // if we're not generating on all levels then we need to propagate the vectors down
        if (param.mg_global.generate_all_levels == QUDA_BOOLEAN_NO) {
          if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Restricting null space vectors\n");
          for (int i=0; i<param.Nvec; i++) {
            zero(*(*B_coarse)[i]);
            transfer->R(*(*B_coarse)[i], *(param.B[i]));
          }
        }
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Transfer operator done\n");
      }
      
      createCoarseDirac();
    }
    
    // delay allocating smoother until after coarse-links have been created
    createSmoother();

    if (param.level < param.Nlevel-1) {
      // creating or resetting the coarse level
      if (coarse) {
        coarse->param.updateInvertParam(*param.mg_global.invert_param);
        coarse->param.delta = 1e-20;
        coarse->param.precision = param.mg_global.invert_param->cuda_prec_precondition;
        coarse->param.matResidual = matCoarseResidual;
        coarse->param.matSmooth = matCoarseSmoother;
        coarse->param.matSmoothSloppy = matCoarseSmootherSloppy;	
        coarse->reset(refresh);	
      } else {
        // create the next multigrid level
        param_coarse = new MGParam(param, *B_coarse, param.evals, matCoarseResidual, matCoarseSmoother, matCoarseSmootherSloppy, param.level+1);
        param_coarse->fine = this;
        param_coarse->delta = 1e-20;
        param_coarse->precision = param.mg_global.invert_param->cuda_prec_precondition;

        coarse = new MG(*param_coarse, profile_global);
      }
      setOutputPrefix(prefix); // restore since we just popped back from coarse grid

      createCoarseSolver();

      // now we can run the verification if requested
      if (param.mg_global.run_verify) verify();

      // resize the on-GPU null-space components to single-parity if we're doing a
      // single-parity solve (memory saving technique).
      {
        QudaSiteSubset site_subset = param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
        QudaMatPCType matpc_type = param.mg_global.invert_param->matpc_type;
        QudaParity parity = (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
        transfer->setSiteSubset(site_subset, parity); // use this to force location of transfer
      }
    }

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Setup of level %d of %d done\n", param.level+1, param.Nlevel);

    // print out profiling information for the adaptive setup
    if (getVerbosity() >= QUDA_VERBOSE) profile.Print();
    // Reset the profile for accurate solver timing
    profile.TPRESET();

    postTrace();
  }

  void MG::destroySmoother() {
    postTrace();
    if (presmoother) {
      delete presmoother;
      presmoother = nullptr;
    }

    if (param_presmooth) {
      delete param_presmooth;
      param_presmooth = nullptr;
    }

    if (postsmoother) {
      delete postsmoother;
      postsmoother = nullptr;
    }

    if (param_postsmooth) {
      delete param_postsmooth;
      param_postsmooth = nullptr;
    }
    postTrace();
  }

  void MG::createSmoother() {
    postTrace();

    // create the smoother for this level
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating smoother\n");
    destroySmoother();
    param_presmooth = new SolverParam(param);

    param_presmooth->is_preconditioner = false;
    param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_NO;
    param_presmooth->return_residual = true; // pre-smoother returns the residual vector for subsequent coarsening
    param_presmooth->use_init_guess = QUDA_USE_INIT_GUESS_NO;

    param_presmooth->precision = param.mg_global.invert_param->cuda_prec_sloppy;
    param_presmooth->precision_sloppy = (param.level == 0) ? param.mg_global.invert_param->cuda_prec_precondition : param.mg_global.invert_param->cuda_prec_sloppy;
    param_presmooth->precision_precondition = (param.level == 0) ? param.mg_global.invert_param->cuda_prec_precondition : param.mg_global.invert_param->cuda_prec_sloppy;

    param_presmooth->inv_type = param.smoother;
    param_presmooth->inv_type_precondition = QUDA_INVALID_INVERTER;
    param_presmooth->residual_type = (param_presmooth->inv_type == QUDA_MR_INVERTER) ? QUDA_INVALID_RESIDUAL : QUDA_L2_RELATIVE_RESIDUAL;
    param_presmooth->Nsteps = param.mg_global.smoother_schwarz_cycle[param.level];
    param_presmooth->maxiter = (param.level < param.Nlevel-1) ? param.nu_pre : param.nu_pre + param.nu_post;

    param_presmooth->Nkrylov = param_presmooth->maxiter;
    param_presmooth->pipeline = param_presmooth->maxiter;
    param_presmooth->tol = param.smoother_tol;
    param_presmooth->global_reduction = param.global_reduction;

    param_presmooth->sloppy_converge = true; // this means we don't check the true residual before declaring convergence

    param_presmooth->schwarz_type = param.mg_global.smoother_schwarz_type[param.level];
    // inner solver should recompute the true residual after each cycle if using Schwarz preconditioning
    param_presmooth->compute_true_res = (param_presmooth->schwarz_type != QUDA_INVALID_SCHWARZ) ? true : false;

    presmoother = ( (param.level < param.Nlevel-1 || param_presmooth->schwarz_type != QUDA_INVALID_SCHWARZ) &&
                    param_presmooth->inv_type != QUDA_INVALID_INVERTER && param_presmooth->maxiter > 0) ?
      Solver::create(*param_presmooth, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy, profile) : nullptr;

    if (param.level < param.Nlevel-1) { //Create the post smoother
      param_postsmooth = new SolverParam(*param_presmooth);
      param_postsmooth->return_residual = false;  // post smoother does not need to return the residual vector
      param_postsmooth->use_init_guess = QUDA_USE_INIT_GUESS_YES;

      param_postsmooth->maxiter = param.nu_post;
      param_postsmooth->Nkrylov = param_postsmooth->maxiter;
      param_postsmooth->pipeline = param_postsmooth->maxiter;

      // we never need to compute the true residual for a post smoother
      param_postsmooth->compute_true_res = false;

      postsmoother = (param_postsmooth->inv_type != QUDA_INVALID_INVERTER && param_postsmooth->maxiter > 0) ?
	Solver::create(*param_postsmooth, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy, profile) : nullptr;
    }
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Smoother done\n");
    postTrace();
  }

  void MG::createCoarseDirac() {
    postTrace();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating coarse Dirac operator\n");
    // check if we are coarsening the preconditioned system then
    bool preconditioned_coarsen = (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE);
    QudaMatPCType matpc_type = param.mg_global.invert_param->matpc_type;

    // create coarse grid operator
    DiracParam diracParam;
    diracParam.transfer = transfer;

    diracParam.dirac = preconditioned_coarsen ? const_cast<Dirac*>(diracSmoother) : const_cast<Dirac*>(diracResidual);
    diracParam.kappa = diracParam.dirac->Kappa();
    diracParam.mu = diracParam.dirac->Mu();
    diracParam.mu_factor = param.mg_global.mu_factor[param.level+1]-param.mg_global.mu_factor[param.level];

    diracParam.dagger = QUDA_DAG_NO;
    diracParam.matpcType = matpc_type;
    diracParam.type = QUDA_COARSE_DIRAC;
    diracParam.tmp1 = tmp_coarse;
    diracParam.halo_precision = param.mg_global.precision_null[param.level];
    constexpr int MAX_BLOCK_FLOAT_NC=32; // FIXME this is the maximum number of colors for which we support block-float format
    if (param.Nvec > MAX_BLOCK_FLOAT_NC) diracParam.halo_precision = QUDA_SINGLE_PRECISION;

    // use even-odd preconditioning for the coarse grid solver
    if (diracCoarseResidual) delete diracCoarseResidual;
    diracCoarseResidual = new DiracCoarse(diracParam, param.setup_location == QUDA_CUDA_FIELD_LOCATION ? true : false,
                                          param.mg_global.setup_minimize_memory == QUDA_BOOLEAN_YES ? true : false);

    // create smoothing operators
    diracParam.dirac = const_cast<Dirac*>(param.matSmooth->Expose());
    diracParam.halo_precision = param.mg_global.smoother_halo_precision[param.level+1];

    if (diracCoarseSmoother) delete diracCoarseSmoother;
    if (diracCoarseSmootherSloppy) delete diracCoarseSmootherSloppy;
    if (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE) {
      diracParam.type = QUDA_COARSEPC_DIRAC;
      diracParam.tmp1 = &(tmp_coarse->Even());
      diracCoarseSmoother = new DiracCoarsePC(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam);
      {
        bool schwarz = param.mg_global.smoother_schwarz_type[param.level+1] != QUDA_INVALID_SCHWARZ;
        for (int i=0; i<4; i++) diracParam.commDim[i] = schwarz ? 0 : 1;
      }
      diracCoarseSmootherSloppy = new DiracCoarsePC(static_cast<DiracCoarse&>(*diracCoarseSmoother),diracParam);
    } else {
      diracParam.type = QUDA_COARSE_DIRAC;
      diracParam.tmp1 = tmp_coarse;
      diracCoarseSmoother = new DiracCoarse(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam);
      {
        bool schwarz = param.mg_global.smoother_schwarz_type[param.level+1] != QUDA_INVALID_SCHWARZ;
        for (int i=0; i<4; i++) diracParam.commDim[i] = schwarz ? 0 : 1;
      }
      diracCoarseSmootherSloppy = new DiracCoarse(static_cast<DiracCoarse&>(*diracCoarseSmoother),diracParam);
    }

    if (matCoarseResidual) delete matCoarseResidual;
    if (matCoarseSmoother) delete matCoarseSmoother;
    if (matCoarseSmootherSloppy) delete matCoarseSmootherSloppy;
    matCoarseResidual = new DiracM(*diracCoarseResidual);
    matCoarseSmoother = new DiracM(*diracCoarseSmoother);
    matCoarseSmootherSloppy = new DiracM(*diracCoarseSmootherSloppy);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Coarse Dirac operator done\n");
    postTrace();
  }

  void MG::destroyCoarseSolver() {
    postTrace();
    if (param.cycle_type == QUDA_MG_CYCLE_VCYCLE && param.level < param.Nlevel-2) {
      // nothing to do
    } else if (param.cycle_type == QUDA_MG_CYCLE_RECURSIVE || param.level == param.Nlevel-2) {
      if (coarse_solver) {
        delete coarse_solver;
        coarse_solver = nullptr;
      }
      if (param_coarse_solver) {
        delete param_coarse_solver;
        param_coarse_solver = nullptr;
      }
    } else {
      errorQuda("Multigrid cycle type %d not supported", param.cycle_type);
    }
    postTrace();
  }

  void MG::createCoarseSolver() {
    postTrace();
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Creating coarse solver wrapper\n");
    destroyCoarseSolver();
    if (param.cycle_type == QUDA_MG_CYCLE_VCYCLE && param.level < param.Nlevel-2) {
      // if coarse solver is not a bottom solver and on the second to bottom level then we can just use the coarse solver as is
      coarse_solver = coarse;
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Assigned coarse solver to coarse MG operator\n");
    } else if (param.cycle_type == QUDA_MG_CYCLE_RECURSIVE || param.level == param.Nlevel-2) {
      
      param_coarse_solver = new SolverParam(param);
      param_coarse_solver->inv_type = param.mg_global.coarse_solver[param.level+1];      
      param_coarse_solver->is_preconditioner = false;
      param_coarse_solver->sloppy_converge = true; // this means we don't check the true residual before declaring convergence

      param_coarse_solver->preserve_source = QUDA_PRESERVE_SOURCE_NO;  // or can this be no
      param_coarse_solver->return_residual = false; // coarse solver does need to return residual vector

      param_coarse_solver->use_init_guess = QUDA_USE_INIT_GUESS_NO;  
      if(param.mg_global.deflate_coarsest) {
	param_coarse_solver->eig_param = *param.mg_global.eig_param[param.Nlevel-1];
	param_coarse_solver->deflate = QUDA_BOOLEAN_YES;
	param_coarse_solver->use_init_guess = QUDA_USE_INIT_GUESS_YES;
	printfQuda("deflate coarsest true\n");
      }
      param_coarse_solver->tol = param.mg_global.coarse_solver_tol[param.level+1];
      param_coarse_solver->global_reduction = true;
      param_coarse_solver->compute_true_res = false;
      param_coarse_solver->delta = 1e-8;
      param_coarse_solver->pipeline = 8;

      param_coarse_solver->maxiter = param.mg_global.coarse_solver_maxiter[param.level+1];
      param_coarse_solver->Nkrylov = param_coarse_solver->maxiter < 20 ? param_coarse_solver->maxiter : 20;
      if (param_coarse_solver->inv_type == QUDA_CA_CG_INVERTER ||
          param_coarse_solver->inv_type == QUDA_CA_CGNE_INVERTER ||
          param_coarse_solver->inv_type == QUDA_CA_CGNR_INVERTER ||
          param_coarse_solver->inv_type == QUDA_CA_GCR_INVERTER) {
        param_coarse_solver->ca_basis = param.mg_global.coarse_solver_ca_basis[param.level+1];
        param_coarse_solver->ca_lambda_min = param.mg_global.coarse_solver_ca_lambda_min[param.level+1];
        param_coarse_solver->ca_lambda_max = param.mg_global.coarse_solver_ca_lambda_max[param.level+1];
        param_coarse_solver->Nkrylov = param.mg_global.coarse_solver_ca_basis_size[param.level+1];
      }
      param_coarse_solver->inv_type_precondition = (param.level<param.Nlevel-2 || coarse->presmoother) ? QUDA_MG_INVERTER : QUDA_INVALID_INVERTER;
      param_coarse_solver->preconditioner = (param.level<param.Nlevel-2 || coarse->presmoother) ? coarse : nullptr;
      param_coarse_solver->mg_instance = true;
      param_coarse_solver->verbosity_precondition = param.mg_global.verbosity[param.level+1];

      // preconditioned solver wrapper is uniform precision
      param_coarse_solver->precision = r_coarse->Precision();
      param_coarse_solver->precision_sloppy = param_coarse_solver->precision;
      param_coarse_solver->precision_precondition = param_coarse_solver->precision_sloppy;

      if (param.mg_global.coarse_grid_solution_type[param.level+1] == QUDA_MATPC_SOLUTION) {
	Solver *solver = Solver::create(*param_coarse_solver, *matCoarseSmoother, *matCoarseSmoother, *matCoarseSmoother, profile);
	sprintf(coarse_prefix,"MG level %d (%s): ", param.level+2, param.mg_global.location[param.level+1] == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
	coarse_solver = new PreconditionedSolver(*solver, *matCoarseSmoother->Expose(), *param_coarse_solver, profile, coarse_prefix);
      } else {
	Solver *solver = Solver::create(*param_coarse_solver, *matCoarseResidual, *matCoarseResidual, *matCoarseResidual, profile);
	sprintf(coarse_prefix,"MG level %d (%s): ", param.level+2, param.mg_global.location[param.level+1] == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
	coarse_solver = new PreconditionedSolver(*solver, *matCoarseResidual->Expose(), *param_coarse_solver, profile, coarse_prefix);
      }

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Assigned coarse solver to preconditioned GCR solver\n");
    } else {
      errorQuda("Multigrid cycle type %d not supported", param.cycle_type);
    }
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Coarse solver wrapper done\n");
    postTrace();
  }
  
  MG::~MG() {
    if (param.level < param.Nlevel-1) {
      if (rng) rng->Release();
      delete rng;
      
      if (param.level == param.Nlevel-1 || param.cycle_type == QUDA_MG_CYCLE_RECURSIVE) {
	if (coarse_solver) delete coarse_solver;
	if (param_coarse_solver) delete param_coarse_solver;
      }

      if (B_coarse) {
	int nVec_coarse = (param.level == param.Nlevel-2 ?  param.mg_global.n_vec[param.level+1] : std::max(param.Nvec, param.mg_global.n_vec[param.level+1]));
	for (int i=0; i<nVec_coarse; i++) if ((*B_coarse)[i]) delete (*B_coarse)[i];
	delete B_coarse;
      }
      if (coarse) delete coarse;
      if (transfer) delete transfer;
      if (matCoarseSmootherSloppy) delete matCoarseSmootherSloppy;
      if (diracCoarseSmootherSloppy) delete diracCoarseSmootherSloppy;
      if (matCoarseSmoother) delete matCoarseSmoother;
      if (diracCoarseSmoother) delete diracCoarseSmoother;
      if (matCoarseResidual) delete matCoarseResidual;
      if (diracCoarseResidual) delete diracCoarseResidual;
      if (postsmoother) delete postsmoother;
      if (param_postsmooth) delete param_postsmooth;
    }

    if (presmoother) delete presmoother;
    if (param_presmooth) delete param_presmooth;

    if (b_tilde && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) delete b_tilde;
    if (r) delete r;
    if (r_coarse) delete r_coarse;
    if (x_coarse) delete x_coarse;
    if (tmp_coarse) delete tmp_coarse;

    if (param_coarse) delete param_coarse;

    if (getVerbosity() >= QUDA_VERBOSE) profile.Print();
  }

  // FIXME need to make this more robust (implement Solver::flops() for all solvers)
  double MG::flops() const {
    double flops = 0;

    if (param_coarse_solver) {
      flops += param_coarse_solver->gflops * 1e9;
      param_coarse_solver->gflops = 0;
    } else if (param.level < param.Nlevel-1) {
      flops += coarse->flops();
    }

    if (param_presmooth) {
      flops += param_presmooth->gflops * 1e9;
      param_presmooth->gflops = 0;
    }

    if (param_postsmooth) {
      flops += param_postsmooth->gflops * 1e9;
      param_postsmooth->gflops = 0;
    }

    if (transfer) {
      flops += transfer->flops();
    }

    return flops;
  }

  /**
     Verification that the constructed multigrid operator is valid
   */
  void MG::verify() {
    setOutputPrefix(prefix);

    // temporary fields used for verification
    ColorSpinorParam csParam(*r);
    csParam.create = QUDA_NULL_FIELD_CREATE;
    ColorSpinorField *tmp1 = ColorSpinorField::Create(csParam);
    ColorSpinorField *tmp2 = ColorSpinorField::Create(csParam);
    double deviation;

    QudaPrecision prec = (param.mg_global.precision_null[param.level] < csParam.Precision())
      ? param.mg_global.precision_null[param.level]  : csParam.Precision();
    
    // may want to revisit this---these were relaxed for cases where ghost_precision < precision
    // these were set while hacking in tests of quarter precision ghosts
    double tol = (prec == QUDA_QUARTER_PRECISION || prec == QUDA_HALF_PRECISION) ? 5e-2 : prec == QUDA_SINGLE_PRECISION ? 1e-3 : 1e-8;

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Checking 0 = (1 - P P^\\dagger) v_k for %d vectors\n", param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      // as well as copying to the correct location this also changes basis if necessary
      *tmp1 = *param.B[i];

      transfer->R(*r_coarse, *tmp1);
      transfer->P(*tmp2, *r_coarse);

      if (getVerbosity() >= QUDA_VERBOSE)
	printfQuda("Vector %d: norms v_k = %e P^\\dag v_k = %e P P^\\dag v_k = %e\n",
		   i, norm2(*tmp1), norm2(*r_coarse), norm2(*tmp2));

      deviation = sqrt( xmyNorm(*tmp1, *tmp2) / norm2(*tmp1) );
      if (getVerbosity() >= QUDA_VERBOSE)
	printfQuda("L2 relative deviation = %e\n", deviation);
      if (deviation > tol)
	errorQuda("L2 relative deviation for k=%d failed, %e > %e", i, deviation, tol);
    }

    if (param.mg_global.run_oblique_proj_check) {

      sprintf(prefix,"MG level %d (%s): Null vector Oblique Projections : ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
      setOutputPrefix(prefix);
      
      //Oblique projections
      if (getVerbosity() >= QUDA_SUMMARIZE)
	printfQuda("Checking 1 > || (1 - DP(P^dagDP)P^dag) v_k || / || v_k || for %d vectors\n",
		   param.Nvec);

      for (int i=0; i<param.Nvec; i++) {
	transfer->R(*r_coarse, *(param.B[i]));
	(*coarse_solver)(*x_coarse, *r_coarse); // this needs to be an exact solve to pass
	setOutputPrefix(prefix);                // restore prefix after return from coarse grid
	transfer->P(*tmp2, *x_coarse);
	(*param.matResidual)(*tmp1,*tmp2);
	*tmp2 = *(param.B[i]);
	if (getVerbosity() >= QUDA_SUMMARIZE) {
	  printfQuda("Vector %d: norms %e %e\n", i, norm2(*param.B[i]), norm2(*tmp1));
	  printfQuda("relative residual = %e\n", sqrt(xmyNorm(*tmp2, *tmp1) / norm2(*param.B[i])) );
	}
      }
      sprintf(prefix,"MG level %d (%s): ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
      setOutputPrefix(prefix);
    }
#if 0
    
    if (getVerbosity() >= QUDA_SUMMARIZE)
      printfQuda("Checking 1 > || (1 - D P (P^\\dagger D P) P^\\dagger v_k || / || v_k || for %d vectors\n",
		 param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      transfer->R(*r_coarse, *(param.B[i]));
      (*coarse)(*x_coarse, *r_coarse); // this needs to be an exact solve to pass
      setOutputPrefix(prefix); // restore output prefix
      transfer->P(*tmp2, *x_coarse);
      param.matResidual(*tmp1,*tmp2);
      *tmp2 = *(param.B[i]);
      if (getVerbosity() >= QUDA_VERBOSE) {
	printfQuda("Vector %d: norms %e %e ", i, norm2(*param.B[i]), norm2(*tmp1));
	printfQuda("relative residual = %e\n", sqrt(xmyNorm(*tmp2, *tmp1) / norm2(*param.B[i])) );
      }
    }
#endif

    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Checking 0 = (1 - P^\\dagger P) eta_c\n");
    x_coarse->Source(QUDA_RANDOM_SOURCE);
    transfer->P(*tmp2, *x_coarse);
    transfer->R(*r_coarse, *tmp2);
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Vector norms %e %e (fine tmp %e) ", norm2(*x_coarse), norm2(*r_coarse), norm2(*tmp2));

    deviation = sqrt( xmyNorm(*x_coarse, *r_coarse) / norm2(*x_coarse) );
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("L2 relative deviation = %e\n", deviation);
    if (deviation > tol ) errorQuda("L2 relative deviation = %e > %e failed", deviation, tol);
    if (getVerbosity() >= QUDA_SUMMARIZE) printfQuda("Checking 0 = (D_c - P^\\dagger D P) (native coarse operator to emulated operator)\n");

    ColorSpinorField *tmp_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, r->Precision(), param.mg_global.location[param.level+1]);
    zero(*tmp_coarse);
    zero(*r_coarse);

    tmp_coarse->Source(QUDA_RANDOM_SOURCE);
    transfer->P(*tmp1, *tmp_coarse);

    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
      double kappa = diracResidual->Kappa();
      if (param.level==0) {
	diracSmoother->DslashXpay(tmp2->Even(), tmp1->Odd(), QUDA_EVEN_PARITY, tmp1->Even(), -kappa);
	diracSmoother->DslashXpay(tmp2->Odd(), tmp1->Even(), QUDA_ODD_PARITY, tmp1->Odd(), -kappa);
      } else { // this is a hack since the coarse Dslash doesn't properly use the same xpay conventions yet
	diracSmoother->DslashXpay(tmp2->Even(), tmp1->Odd(), QUDA_EVEN_PARITY, tmp1->Even(), 1.0);
	diracSmoother->DslashXpay(tmp2->Odd(), tmp1->Even(), QUDA_ODD_PARITY, tmp1->Odd(), 1.0);
      }
    } else {
      (*param.matResidual)(*tmp2,*tmp1);
    }

    transfer->R(*x_coarse, *tmp2);
    (*param_coarse->matResidual)(*r_coarse, *tmp_coarse);


#if 0 // enable to print out emulated and actual coarse-grid operator vectors for debugging
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("emulated\n");
    for (int x=0; x<x_coarse->Volume(); x++) tmp1->PrintVector(x);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("actual\n");
    for (int x=0; x<r_coarse->Volume(); x++) tmp2->PrintVector(x);
#endif

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Vector norms Emulated=%e Native=%e ", norm2(*x_coarse), norm2(*r_coarse));

    deviation = sqrt( xmyNorm(*x_coarse, *r_coarse) / norm2(*x_coarse) );

    // When the mu is shifted on the coarse level; we can compute exxactly the error we introduce in the check:
    //  it is given by 2*kappa*delta_mu || tmp_coarse ||; where tmp_coarse is the random vector generated for the test
    if(diracResidual->Mu() != 0) {
      double delta_factor = param.mg_global.mu_factor[param.level+1] - param.mg_global.mu_factor[param.level];
      if(fabs(delta_factor) > tol ) {
	double delta_a = delta_factor * 2.0 * diracResidual->Kappa() *
	  diracResidual->Mu() * transfer->Vectors().TwistFlavor();
	deviation -= fabs(delta_a) * sqrt( norm2(*tmp_coarse) / norm2(*x_coarse) );
	deviation = fabs(deviation);
      }
    }
    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("L2 relative deviation = %e\n\n", deviation);
    if (deviation > tol) errorQuda("failed, deviation = %e (tol=%e)", deviation, tol);
    
    // here we check that the Hermitian conjugate operator is working
    // as expected for both the smoother and residual Dirac operators
    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
      diracSmoother->MdagM(tmp2->Even(), tmp1->Odd());
      Complex dot = cDotProduct(tmp2->Even(),tmp1->Odd());
      double deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Smoother normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
						     real(dot), imag(dot), deviation);
      if (deviation > tol) errorQuda("failed, deviation = %e (tol=%e)", deviation, tol);

      diracResidual->MdagM(*tmp2, *tmp1);
      dot = cDotProduct(*tmp2,*tmp1);

      deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Residual normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
						     real(dot), imag(dot), deviation);
      if (deviation > tol) errorQuda("failed, deviation = %e (tol=%e)", deviation, tol);
    } else {
      diracResidual->MdagM(*tmp2, *tmp1);
      Complex dot = cDotProduct(*tmp1,*tmp2);

      double deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
						     real(dot), imag(dot), deviation);
      if (deviation > tol) errorQuda("failed, deviation = %e (tol=%e)", deviation, tol);
    }


    if (param.mg_global.run_low_mode_check) {
      
      sprintf(prefix,"MG level %d (%s): eigenvector overlap : ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
      setOutputPrefix(prefix);

      //Reuse the space for the Null vectors. By this point,
      //the coarse grid has already been constructed.
      //generateEigenVectors(param.B);
      generateEigenVectors();
      
      for (int i=0; i<param.Nvec; i++) {

	//Restrict Evec, place result in r_coarse
	transfer->R(*r_coarse, *param.B[i]);
	//Prolong r_coarse, place result in tmp2
	transfer->P(*tmp2, *r_coarse);
	
	printfQuda("Vector %d: norms v_k = %e P^dag v_k = %e PP^dag v_k = %e\n",
		   i, norm2(*param.B[i]), norm2(*r_coarse), norm2(*tmp2) );

	//Compare v_k and PP^dag v_k.
	deviation = sqrt( xmyNorm(*param.B[i], *tmp2) / norm2(*param.B[i]) );
	printfQuda("L2 relative deviation = %e\n", deviation);

	if (param.mg_global.run_oblique_proj_check) {
	  
	  sprintf(prefix,"MG level %d (%s): eigenvector Oblique Projections : ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
	  setOutputPrefix(prefix);
	  
	  //Oblique projections
	  if (getVerbosity() >= QUDA_SUMMARIZE)
	    printfQuda("Checking 1 > || (1 - DP(P^dagDP)P^dag) v_k || / || v_k || for vector %d\n", i);
	  
	  transfer->R(*r_coarse, *param.B[i]);
	  (*coarse_solver)(*x_coarse, *r_coarse); // this needs to be an exact solve to pass
	  setOutputPrefix(prefix);                // restore prefix after return from coarse grid
	  transfer->P(*tmp2, *x_coarse);
	  (*param.matResidual)(*tmp1,*tmp2);

	  if (getVerbosity() >= QUDA_SUMMARIZE) {
	    printfQuda("Vector %d: norms v_k %e DP(P^dagDP)P^dag v_k %e\n", i, norm2(*param.B[i]), norm2(*tmp1));
	    printfQuda("L2 relative deviation = %e\n", sqrt(xmyNorm(*param.B[i], *tmp1) / norm2(*param.B[i])) );
	  }
	}
	
	sprintf(prefix,"MG level %d (%s): ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
	setOutputPrefix(prefix);
	
      }
    }
    
    delete tmp1;
    delete tmp2;
    delete tmp_coarse;
  }
  
  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    char prefix_bkup[100];  strncpy(prefix_bkup, prefix, 100);  setOutputPrefix(prefix);

    // if input vector is single parity then we must be solving the
    // preconditioned system in general this can only happen on the
    // top level
    QudaSolutionType outer_solution_type = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_MAT_SOLUTION : QUDA_MATPC_SOLUTION;
    QudaSolutionType inner_solution_type = param.coarse_grid_solution_type;

    if (debug) printfQuda("outer_solution_type = %d, inner_solution_type = %d\n", outer_solution_type, inner_solution_type);

    if ( outer_solution_type == QUDA_MATPC_SOLUTION && inner_solution_type == QUDA_MAT_SOLUTION)
      errorQuda("Unsupported solution type combination");

    if ( inner_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE)
      errorQuda("For this coarse grid solution type, a preconditioned smoother is required");

    if ( debug ) printfQuda("entering V-cycle with x2=%e, r2=%e\n", norm2(x), norm2(b));

    if (param.level < param.Nlevel-1) {
      //transfer->setTransferGPU(false); // use this to force location of transfer (need to check if still works for multi-level)
      
      // do the pre smoothing
      if ( debug ) printfQuda("pre-smoothing b2=%e\n", norm2(b));

      ColorSpinorField *out=nullptr, *in=nullptr;

      ColorSpinorField &residual = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? *r : r->Even();

      // FIXME only need to make a copy if not preconditioning
      residual = b; // copy source vector since we will overwrite source with iterated residual

      diracSmoother->prepare(in, out, x, residual, outer_solution_type);

      // b_tilde holds either a copy of preconditioned source or a pointer to original source
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) *b_tilde = *in;
      else b_tilde = &b;
      if (presmoother) (*presmoother)(*out, *in); else zero(*out);
      ColorSpinorField &solution = inner_solution_type == outer_solution_type ? x : x.Even();
      diracSmoother->reconstruct(solution, b, inner_solution_type);
      
      // if using preconditioned smoother then need to reconstruct full residual
      // FIXME extend this check for precision, Schwarz, etc.
      bool use_solver_residual =
	( (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE && inner_solution_type == QUDA_MATPC_SOLUTION) ||
	  (param.smoother_solve_type == QUDA_DIRECT_SOLVE && inner_solution_type == QUDA_MAT_SOLUTION) )
	? true : false;
      // FIXME this is currently borked if inner solver is preconditioned
      double r2 = 0.0;
      if (use_solver_residual) {
	if (debug) r2 = norm2(*r);
      } else {
	(*param.matResidual)(*r, x);
	if (debug) r2 = xmyNorm(b, *r);
	else axpby(1.0, b, -1.0, *r);
      }

      // We need this to ensure that the coarse level has been created.
      // e.g. in case of iterative setup with MG we use just pre- and post-smoothing at the first iteration.
      if (transfer) {
        // restrict to the coarse grid
        transfer->R(*r_coarse, residual);
        if ( debug ) printfQuda("after pre-smoothing x2 = %e, r2 = %e, r_coarse2 = %e\n", norm2(x), r2, norm2(*r_coarse));

        // recurse to the next lower level
	(*coarse_solver)(*x_coarse, *r_coarse); 
	setOutputPrefix(prefix); // restore prefix after return from coarse grid
	if ( debug ) printfQuda("after coarse solve x_coarse2 = %e r_coarse2 = %e\n", norm2(*x_coarse), norm2(*r_coarse));
	
        // prolongate back to this grid
        ColorSpinorField &x_coarse_2_fine = inner_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even(); // define according to inner solution type
        transfer->P(x_coarse_2_fine, *x_coarse); // repurpose residual storage
        xpy(x_coarse_2_fine, solution); // sum to solution FIXME - sum should be done inside the transfer operator
        if ( debug ) {
          printfQuda("Prolongated coarse solution y2 = %e\n", norm2(*r));
          printfQuda("after coarse-grid correction x2 = %e, r2 = %e\n", 
                     norm2(x), norm2(*r));
        }
      }

      // do the post smoothing
      //residual = outer_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even(); // refine for outer solution type
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
	in = b_tilde;
      } else { // this incurs unecessary copying
	*r = b;
	in = r;
      }

      // we should keep a copy of the prepared right hand side as we've already destroyed it
      //dirac.prepare(in, out, solution, residual, inner_solution_type);

      if (postsmoother) (*postsmoother)(*out, *in); // for inner solve preconditioned, in the should be the original prepared rhs

      diracSmoother->reconstruct(x, b, outer_solution_type);

    } else { // do the coarse grid solve
      
      ColorSpinorField *out=nullptr, *in=nullptr;
      //Deflate EVEN ODD here
      diracSmoother->prepare(in, out, x, b, outer_solution_type);
      if (presmoother) (*presmoother)(*out, *in);
      diracSmoother->reconstruct(x, b, outer_solution_type);
    }

    if ( debug ) {
      (*param.matResidual)(*r, x);
      double r2 = xmyNorm(b, *r);
      printfQuda("leaving V-cycle with x2=%e, r2=%e\n", norm2(x), r2);
    }

    setOutputPrefix(param.level == 0 ? "" : prefix_bkup);
  }

  //supports separate reading or single file read
  void MG::loadVectors(std::vector<ColorSpinorField*> &B) {
    
    std::string vec_infile(param.mg_global.vec_infile);
    vec_infile += "_level_";
    vec_infile += std::to_string(param.level);
    eig_solver->loadVectors(B, vec_infile);
    
  }
  
  void MG::saveVectors(std::vector<ColorSpinorField*> &B) {

    std::string vec_outfile(param.mg_global.vec_outfile);
    vec_outfile += "_level_";
    vec_outfile += std::to_string(param.level);
    eig_solver->saveVectors(B, vec_outfile);
    
  }

  void MG::dumpNullVectors()
  {
    saveVectors(param.B);
    if (param.level < param.Nlevel-2) coarse->dumpNullVectors();
  }
  
  void MG::generateNullVectors(std::vector<ColorSpinorField*> &B, bool refresh) {
    setOutputPrefix(prefix);

    SolverParam solverParam(param);  // Set solver field parameters:
    // set null-space generation options - need to expose these
    solverParam.maxiter = refresh ? param.mg_global.setup_maxiter_refresh[param.level] : param.mg_global.setup_maxiter[param.level];
    solverParam.tol = param.mg_global.setup_tol[param.level];
    solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
    solverParam.delta = 1e-1;
    solverParam.inv_type = param.mg_global.setup_inv_type[param.level];
    // Hard coded for now...
    if (solverParam.inv_type == QUDA_CA_CG_INVERTER ||
        solverParam.inv_type == QUDA_CA_CGNE_INVERTER ||
        solverParam.inv_type == QUDA_CA_CGNR_INVERTER ||
        solverParam.inv_type == QUDA_CA_GCR_INVERTER) {
      solverParam.ca_basis = param.mg_global.setup_ca_basis[param.level];
      solverParam.ca_lambda_min = param.mg_global.setup_ca_lambda_min[param.level];
      solverParam.ca_lambda_max = param.mg_global.setup_ca_lambda_max[param.level];
      solverParam.Nkrylov = param.mg_global.setup_ca_basis_size[param.level];
    } else {
      solverParam.Nkrylov = 4;
    }
    solverParam.pipeline = (solverParam.inv_type == QUDA_BICGSTAB_INVERTER ? 0 : 4); // FIXME: pipeline != 0 breaks BICGSTAB
    solverParam.precision = r->Precision();

    if (param.level == 0) { // this enables half precision on the fine grid only if set
      solverParam.precision_sloppy = param.mg_global.invert_param->cuda_prec_precondition;
      solverParam.precision_precondition = param.mg_global.invert_param->cuda_prec_precondition;
    } else {
      solverParam.precision_precondition = solverParam.precision;
    }
    solverParam.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
    solverParam.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;
    ColorSpinorParam csParam(*B[0]);  // Create spinor field parameters:
    csParam.setPrecision(r->Precision(), r->Precision(), true); // ensure native ordering
    csParam.location = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField *b = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
    ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

    csParam.create = QUDA_NULL_FIELD_CREATE;

    // if we not using GCR/MG smoother then we need to switch off Schwarz since regular Krylov solvers do not support it
    bool schwarz_reset = solverParam.inv_type != QUDA_MG_INVERTER && param.mg_global.smoother_schwarz_type[param.level] != QUDA_INVALID_SCHWARZ;
    if (schwarz_reset) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Disabling Schwarz for null-space finding");
      int commDim[QUDA_MAX_DIM];
      for (int i=0; i<QUDA_MAX_DIM; i++) commDim[i] = 1;
        diracSmootherSloppy->setCommDim(commDim);
    }

    // if quarter precision halo, promote for null-space finding to half precision
    QudaPrecision halo_precision = diracSmootherSloppy->HaloPrecision();
    if (halo_precision == QUDA_QUARTER_PRECISION) diracSmootherSloppy->setHaloPrecision(QUDA_HALF_PRECISION);

    Solver *solve;
    DiracMdagM *mdagm = (solverParam.inv_type == QUDA_CG_INVERTER || solverParam.inv_type == QUDA_CA_CG_INVERTER) ? new DiracMdagM(*diracSmoother) : nullptr;
    DiracMdagM *mdagmSloppy = (solverParam.inv_type == QUDA_CG_INVERTER || solverParam.inv_type == QUDA_CA_CG_INVERTER) ? new DiracMdagM(*diracSmootherSloppy) : nullptr;
    if (solverParam.inv_type == QUDA_CG_INVERTER || solverParam.inv_type == QUDA_CA_CG_INVERTER) {
      solve = Solver::create(solverParam, *mdagm, *mdagmSloppy, *mdagmSloppy, profile);
    } else if(solverParam.inv_type == QUDA_MG_INVERTER) {
      // in case MG has not been created, we create the Smoother
      if (!transfer) createSmoother();

      // run GCR with the MG as a preconditioner
      solverParam.inv_type_precondition = QUDA_MG_INVERTER;
      solverParam.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
      solverParam.precondition_cycle = 1;
      solverParam.tol_precondition = 1e-1;
      solverParam.maxiter_precondition = 1;
      solverParam.omega = 1.0;
      solverParam.verbosity_precondition = param.mg_global.verbosity[param.level+1];
      solverParam.precision_sloppy = solverParam.precision;
      solverParam.compute_true_res = 0;
      solverParam.preconditioner = this;

      solverParam.inv_type = QUDA_GCR_INVERTER;
      solve = Solver::create(solverParam, *param.matSmooth, *param.matSmooth, *param.matSmoothSloppy, profile);
      solverParam.inv_type = QUDA_MG_INVERTER;
    } else {
      solve = Solver::create(solverParam, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy, profile);
    }

    for (int si=0; si<param.mg_global.num_setup_iter[param.level]; si++ ) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Running vectors setup on level %d iter %d of %d\n", param.level+1, si+1, param.mg_global.num_setup_iter[param.level]);

      // global orthonormalization of the initial null-space vectors
      if(param.mg_global.pre_orthonormalize) {
        for(int i=0; i<(int)B.size(); i++) {
          for (int j=0; j<i; j++) {
            Complex alpha = cDotProduct(*B[j], *B[i]);// <j,i>
            caxpy(-alpha, *B[j], *B[i]); // i-<j,i>j
          }
          double nrm2 = norm2(*B[i]);
          if (nrm2 > 1e-16) ax(1.0 /sqrt(nrm2), *B[i]);// i/<i,i>
          else errorQuda("\nCannot normalize %u vector\n", i);
        }
      }

      // launch solver for each source
      for (int i=0; i<(int)B.size(); i++) {
        if (param.mg_global.setup_type == QUDA_TEST_VECTOR_SETUP) { // DDalphaAMG test vector idea
          *b = *B[i];  // inverting against the vector
          zero(*x);    // with zero initial guess
        } else {
          *x = *B[i];
          zero(*b);
        }
	
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Initial guess = %g\n", norm2(*x));
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Initial rhs = %g\n", norm2(*b));

        ColorSpinorField *out=nullptr, *in=nullptr;
        diracSmoother->prepare(in, out, *x, *b, QUDA_MAT_SOLUTION);
        (*solve)(*out, *in);
        diracSmoother->reconstruct(*x, *b, QUDA_MAT_SOLUTION);

        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Solution = %g\n", norm2(*x));
        *B[i] = *x;
      }

      // global orthonormalization of the generated null-space vectors
      if (param.mg_global.post_orthonormalize) {
        for(int i=0; i<(int)B.size(); i++) {
          for (int j=0; j<i; j++) {
            Complex alpha = cDotProduct(*B[j], *B[i]);// <j,i>
            caxpy(-alpha, *B[j], *B[i]); // i-<j,i>j
          }
          double nrm2 = norm2(*B[i]);
          if (sqrt(nrm2) > 1e-16) ax(1.0/sqrt(nrm2), *B[i]);// i/<i,i>
          else errorQuda("\nCannot normalize %u vector (nrm=%e)\n", i, sqrt(nrm2));
        }
      }

      if (solverParam.inv_type == QUDA_MG_INVERTER) {

        if (transfer) {
          resetTransfer = true;
          reset();
          if ( param.level < param.Nlevel-2 ) {
            if ( param.mg_global.generate_all_levels == QUDA_BOOLEAN_YES ) {
              coarse->generateNullVectors(*B_coarse, refresh);
            } else {
              if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Restricting null space vectors\n");
              for (int i=0; i<param.Nvec; i++) {
                zero(*(*B_coarse)[i]);
                transfer->R(*(*B_coarse)[i], *(param.B[i]));
              }
              // rebuild the transfer operator in the coarse level
              coarse->resetTransfer = true;
              coarse->reset();
            }
          }
        } else {
          reset();
        }
      }
    }

    delete solve;
    if (mdagm) delete mdagm;
    if (mdagmSloppy) delete mdagmSloppy;

    diracSmootherSloppy->setHaloPrecision(halo_precision); // restore halo precision

    delete x;
    delete b;

    // reenable Schwarz
    if (schwarz_reset) {
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Reenabling Schwarz for null-space finding");
      int commDim[QUDA_MAX_DIM];
      for (int i=0; i<QUDA_MAX_DIM; i++) commDim[i] = 0;
      diracSmootherSloppy->setCommDim(commDim);
    }

    if (param.mg_global.vec_store == QUDA_BOOLEAN_YES) { // conditional store of null vectors
      saveVectors(B);
    }

    return;
  }

  // generate a full span of free vectors.
  // FIXME: Assumes fine level is SU(3).
  void MG::buildFreeVectors(std::vector<ColorSpinorField*> &B) {

    const int Nvec = B.size();

    // Given the number of colors and spins, figure out if the number
    // of vectors in 'B' makes sense.
    const int Ncolor = B[0]->Ncolor();
    const int Nspin = B[0]->Nspin();

    if (Ncolor == 3) // fine level
    {
      if (Nspin == 4) // Wilson or Twisted Mass (singlet)
      {
        // There needs to be 6 null vectors -> 12 after chirality.
        if (Nvec != 6)
          errorQuda("\nError in MG::buildFreeVectors: Wilson-type fermions require Nvec = 6");
        
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Building %d free field vectors for Wilson-type fermions\n", Nvec);

        // Zero the null vectors.
        for (int i = 0; i < Nvec ;i++) zero(*B[i]);
        
        // Create a temporary vector.
        ColorSpinorParam csParam(*B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField *tmp = ColorSpinorField::Create(csParam);

        int counter = 0;
        for (int c = 0; c < Ncolor; c++)
        {
          for (int s = 0; s < 2; s++)
          {
            tmp->Source(QUDA_CONSTANT_SOURCE, 1, s, c);
            xpy(*tmp, *B[counter]);
            tmp->Source(QUDA_CONSTANT_SOURCE, 1, s+2, c);
            xpy(*tmp, *B[counter]);
            counter++;
          }
        }

        delete tmp;
      }
      else if (Nspin == 1) // Staggered
      {
        // There needs to be 24 null vectors -> 48 after chirality.
        if (Nvec != 24)
          errorQuda("\nError in MG::buildFreeVectors: Staggered-type fermions require Nvec = 24\n");
        
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Building %d free field vectors for Staggered-type fermions\n", Nvec);

        // Zero the null vectors.
        for (int i = 0; i < Nvec ;i++)
          zero(*B[i]);
        
        // Create a temporary vector.
        ColorSpinorParam csParam(*B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField *tmp = ColorSpinorField::Create(csParam);

        // Build free null vectors.
        for (int c = 0; c < B[0]->Ncolor(); c++)
        {
          // Need to pair an even+odd corner together
          // since they'll get split up.

          // 0000, 0001
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x0, c);
          xpy(*tmp,*B[8*c+0]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x1, c);
          xpy(*tmp,*B[8*c+0]);

          // 0010, 0011
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x2, c);
          xpy(*tmp,*B[8*c+1]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x3, c);
          xpy(*tmp,*B[8*c+1]);

          // 0100, 0101
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x4, c);
          xpy(*tmp,*B[8*c+2]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x5, c);
          xpy(*tmp,*B[8*c+2]);

          // 0110, 0111
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x6, c);
          xpy(*tmp,*B[8*c+3]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x7, c);
          xpy(*tmp,*B[8*c+3]);

          // 1000, 1001
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x8, c);
          xpy(*tmp,*B[8*c+4]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0x9, c);
          xpy(*tmp,*B[8*c+4]);

          // 1010, 1011
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0xA, c);
          xpy(*tmp,*B[8*c+5]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0xB, c);
          xpy(*tmp,*B[8*c+5]);

          // 1100, 1101
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0xC, c);
          xpy(*tmp,*B[8*c+6]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0xD, c);
          xpy(*tmp,*B[8*c+6]);

          // 1110, 1111
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0xE, c);
          xpy(*tmp,*B[8*c+7]);
          tmp->Source(QUDA_CORNER_SOURCE, 1, 0xF, c);
          xpy(*tmp,*B[8*c+7]);
        }

        delete tmp;
      }
      else
      {
        errorQuda("\nError in MG::buildFreeVectors: Unsupported combo of Nc %d, Nspin %d", Ncolor, Nspin);
      }
    }
    else // coarse level
    {
      if (Nspin == 2)
      {
        // There needs to be Ncolor null vectors.
        if (Nvec != Ncolor)
          errorQuda("\nError in MG::buildFreeVectors: Coarse fermions require Nvec = Ncolor");
        
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Building %d free field vectors for Coarse fermions\n", Ncolor);

        // Zero the null vectors.
        for (int i = 0; i < Nvec; i++) zero(*B[i]);
        
        // Create a temporary vector.
        ColorSpinorParam csParam(*B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField *tmp = ColorSpinorField::Create(csParam);

        for (int c = 0; c < Ncolor; c++)
        {
          tmp->Source(QUDA_CONSTANT_SOURCE, 1, 0, c);
          xpy(*tmp, *B[c]);
          tmp->Source(QUDA_CONSTANT_SOURCE, 1, 1, c);
          xpy(*tmp, *B[c]);
        }

        delete tmp;
      }
      else if (Nspin == 1)
      {
        // There needs to be Ncolor null vectors.
        if (Nvec != Ncolor)
          errorQuda("\nError in MG::buildFreeVectors: Coarse fermions require Nvec = Ncolor");
        
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Building %d free field vectors for Coarse fermions\n", Ncolor);

        // Zero the null vectors.
        for (int i = 0; i < Nvec; i++) zero(*B[i]);
        
        // Create a temporary vector.
        ColorSpinorParam csParam(*B[0]);
        csParam.create = QUDA_ZERO_FIELD_CREATE;
        ColorSpinorField *tmp = ColorSpinorField::Create(csParam);

        for (int c = 0; c < Ncolor; c++)
        {
          tmp->Source(QUDA_CONSTANT_SOURCE, 1, 0, c);
          xpy(*tmp, *B[c]);
        }

        delete tmp;
      }
      else
      {
        errorQuda("\nError in MG::buildFreeVectors: Unexpected Nspin = %d for coarse fermions", Nspin);
      }
    }

    // global orthonormalization of the generated null-space vectors
    if(param.mg_global.post_orthonormalize) {
      for(int i=0; i<(int)B.size(); i++) {
        double nrm2 = norm2(*B[i]);
        if (nrm2 > 1e-16) ax(1.0 /sqrt(nrm2), *B[i]);// i/<i,i>
        else errorQuda("\nCannot normalize %u vector\n", i);
      }
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Done building free vectors\n");
  }
  
  void MG::generateEigenVectors() {

      sprintf(prefix,"MG level %d (%s): Eigensolver: ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
      setOutputPrefix(prefix);

      //Extract eigensolver params from the size of the null space.
      int nConv = param.B.size(); param.mg_global.eig_param[param.level]->nConv = nConv;
      int nEv   = param.B.size(); param.mg_global.eig_param[param.level]->nEv = nEv;
      int nKr   = nEv+nEv/2; param.mg_global.eig_param[param.level]->nKr = nKr;

      //Dummy array to keep the eigensolver happy.
      std::vector<Complex> evals(nEv, 0.0);
      
      std::vector<ColorSpinorField*> B_evecs;
      ColorSpinorParam csParam(*param.B[0]);
      csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      //This is the vector precision used by matResidual
      csParam.setPrecision(param.mg_global.invert_param->cuda_prec_sloppy, QUDA_INVALID_PRECISION, true);

      for (int i=0; i<nKr; i++) B_evecs.push_back(ColorSpinorField::Create(csParam));

      EigenSolver *eig_solve = EigenSolver::create(param.mg_global.eig_param[param.level], *param.matResidual->Expose(), profile);
      (*eig_solve)(B_evecs, evals);
      
      //Copy evecs back to MG
      for (unsigned int i=0; i<param.B.size(); i++) {
	*param.B[i] = *B_evecs[i];
      }

      // only save if outfile is defined
      if (strcmp(param.mg_global.vec_outfile,"")!=0) {
	saveVectors(B_evecs);
      }

      //Local clean-up
      for (unsigned int i=0; i<B_evecs.size(); i++) {
	delete B_evecs[i];
      }

      sprintf(prefix,"MG level %d (%s): ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
      setOutputPrefix(prefix);      
  }

  void MG::deflateEigenvectors(std::vector<ColorSpinorField*> vec_defl,
			       std::vector<ColorSpinorField*> vec,
			       std::vector<ColorSpinorField*> eig_vecs,
			       std::vector<Complex> evals) {
    
    //number of evecs
    int Nvecs = eig_vecs.size();

    printfQuda("Deflating %d vectors from coarsest grid\n", Nvecs);

    //Perform Sum_i V_i * (L_i)^{-1} * (V_i)^dag * x_coarse = r_coarse_DEFL
    //for all i computed eigenvectors and values.

    //1. Take block inner product: (V_i)^dag * xcoarse = A_i
    Complex *s = new Complex[Nvecs];     
    blas::cDotProduct(s, eig_vecs, vec);

    //2. Perform block caxpy: x_coarse_DEFL = Sum_i V_i * (L_i)^{-1} * A_i
    for(int i=0; i<Nvecs; i++) {
      s[i] /= evals[i].real();
    }
    blas::zero(*vec_defl[0]);
    blas::caxpy(s, eig_vecs, vec_defl);
    
    printfQuda("Deflation complete\n");
    
  }

  void MG::deflateSVD(std::vector<ColorSpinorField*> vec_defl,
		      std::vector<ColorSpinorField*> vec,
		      std::vector<ColorSpinorField*> svd_vecs,
		      std::vector<Complex> svals) {
    
    //number of svecs
    int Svecs = svd_vecs.size()/2;

    //Perform Sum_i L_i * (S_i)^{-1} * (R_i)^dag * x_coarse = r_coarse_DEFL
    //for all i computed singular vectors and values.

    //1. Take block inner product: (R_i)^dag * xcoarse = A_i
    //   N.B. All right singular vectors are in even part of array
    //Pointer wrapper for Right SV
    std::vector<ColorSpinorField*> r_svd_ptr;    
    for(int i=0; i<Svecs; i++) r_svd_ptr.push_back(svd_vecs[2*i]);
        
    Complex *s = new Complex[Svecs];     
    blas::cDotProduct(s, r_svd_ptr, vec);
    
    //2. Perform block caxpy: x_coarse_DEFL = Sum_i L_i * (S_i)^{-1} * A_i
    std::vector<ColorSpinorField*> l_svd_ptr;    
    for(int i=0; i<Svecs; i++) {
      l_svd_ptr.push_back(svd_vecs[2*i+1]);
      s[i] /= svals[i].real();
    }
    blas::zero(*vec_defl[0]);
    blas::caxpy(s, l_svd_ptr, vec_defl);
    
    //Normalise the guess relative to the source
    double norma = sqrt(blas::norm2(*vec_defl[0]));
    double normb = sqrt(blas::norm2(*vec[0]));
    blas::ax(normb/norma, *vec_defl[0]);
    
  }
}
