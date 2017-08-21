#include <multigrid.h>
#include <qio_field.h>
#include <string.h>

#include <quda_arpack_interface.h>

namespace quda {  

  using namespace blas;

  static bool debug = false;

  MG::MG(MGParam &param, TimeProfile &profile_global)
    : Solver(param, profile), param(param), transfer(0), presmoother(0), postsmoother(0),
      profile_global(profile_global),
      profile( "MG level " + std::to_string(param.level+1), false ),
      coarse(nullptr), fine(param.fine), coarse_solver(nullptr),
      param_coarse(nullptr), param_presmooth(nullptr), param_postsmooth(nullptr),
      r(nullptr), r_coarse(nullptr), x_coarse(nullptr), tmp_coarse(nullptr),
      diracCoarseResidual(nullptr), diracCoarseSmoother(nullptr), matCoarseResidual(nullptr), matCoarseSmoother(nullptr) {

    // for reporting level 1 is the fine level but internally use level 0 for indexing
    sprintf(prefix,"MG level %d (%s): ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
    setOutputPrefix(prefix);

    printfQuda("Creating level %d of %d levels\n", param.level+1, param.Nlevel);

    if (param.level < param.Nlevel-1) {
      if (param.mg_global.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
	if (param.mg_global.generate_all_levels == QUDA_BOOLEAN_YES || param.level == 0) generateNullVectors(param.B);
      } else if (strcmp(param.mg_global.vec_infile,"")!=0) { // only load if infile is defined and not computing
	loadVectors(param.B);
      }
    }

    if (param.level >= QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level+1);

    createSmoother();

    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE)
      errorQuda("Cannot use preconditioned coarse grid solution without preconditioned smoother solve");

    // create residual vectors
    {
      ColorSpinorParam csParam(*(param.B[0]));
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.location = param.location;
      if (csParam.location==QUDA_CUDA_FIELD_LOCATION) {
	// all coarse GPU vectors use FLOAT2 ordering
	csParam.fieldOrder = (csParam.precision == QUDA_DOUBLE_PRECISION || param.level > 0) ? 
	  QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
	csParam.setPrecision(csParam.precision);
	csParam.gammaBasis = param.level > 0 ? QUDA_DEGRAND_ROSSI_GAMMA_BASIS: QUDA_UKQCD_GAMMA_BASIS;
      }
      r = ColorSpinorField::Create(csParam);

      // if we're using preconditioning then allocate storate for the preconditioned source vector
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
	csParam.x[0] /= 2;
	csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
	b_tilde = ColorSpinorField::Create(csParam);
      }
    }

    // if not on the coarsest level, construct it
    if (param.level < param.Nlevel-1) {
      QudaMatPCType matpc_type = param.mg_global.invert_param->matpc_type;

      // create transfer operator
      printfQuda("start creating transfer operator\n");
      transfer = new Transfer(param.B, param.Nvec, param.geoBlockSize, param.spinBlockSize,
			      param.location == QUDA_CUDA_FIELD_LOCATION ? true : false, profile);
      for (int i=0; i<QUDA_MAX_MG_LEVEL; i++) param.mg_global.geo_block_size[param.level][i] = param.geoBlockSize[i];

      //transfer->setTransferGPU(false); // use this to force location of transfer
      printfQuda("end creating transfer operator\n");

      // create coarse residual vector
      r_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, param.mg_global.location[param.level+1]);

      // create coarse solution vector
      x_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, param.mg_global.location[param.level+1]);

      // create coarse temporary vector
      tmp_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, param.mg_global.location[param.level+1]);

      // check if we are coarsening the preconditioned system then
      bool preconditioned_coarsen = (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE);

      // create coarse grid operator
      DiracParam diracParam;
      diracParam.transfer = transfer;

      diracParam.dirac = preconditioned_coarsen ? const_cast<Dirac*>(param.matSmooth->Expose()) : const_cast<Dirac*>(param.matResidual->Expose());
      diracParam.kappa = param.matResidual->Expose()->Kappa();
      diracParam.mu = param.matResidual->Expose()->Mu();
      diracParam.mu_factor = param.mg_global.mu_factor[param.level+1]-param.mg_global.mu_factor[param.level];

      diracParam.dagger = QUDA_DAG_NO;
      diracParam.matpcType = matpc_type;
      diracParam.tmp1 = tmp_coarse;
      // use even-odd preconditioning for the coarse grid solver
      diracCoarseResidual = new DiracCoarse(diracParam);
      matCoarseResidual = new DiracM(*diracCoarseResidual);

      // create smoothing operators
      diracParam.dirac = const_cast<Dirac*>(param.matSmooth->Expose());
      diracParam.type = (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE) ? QUDA_COARSEPC_DIRAC : QUDA_COARSE_DIRAC;
      diracParam.tmp1 = (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE) ? &(tmp_coarse->Even()) : tmp_coarse;
      diracCoarseSmoother = (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE) ?
	new DiracCoarsePC(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam) :
	new DiracCoarse(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam);
      diracCoarseSmootherSloppy = diracCoarseSmoother;  // for coarse grids these always alias for now (FIXME half precision support for coarse op)

      matCoarseSmoother = new DiracM(*diracCoarseSmoother);
      matCoarseSmootherSloppy = new DiracM(*diracCoarseSmootherSloppy);

      printfQuda("Creating coarse null-space vectors\n");
      B_coarse = new std::vector<ColorSpinorField*>();
      int nVec_coarse = std::max(param.Nvec, param.mg_global.n_vec[param.level+1]);
      B_coarse->resize(nVec_coarse);

      for (int i=0; i<nVec_coarse; i++)
	(*B_coarse)[i] = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec);

      // if we're not generating on all levels then we need to propagate the vectors down
      if (param.mg_global.generate_all_levels == QUDA_BOOLEAN_NO) {
	for (int i=0; i<param.Nvec; i++) {
	  zero(*(*B_coarse)[i]);
	  transfer->R(*(*B_coarse)[i], *(param.B[i]));
	}
      }

      // create the next multigrid level
      printfQuda("Creating next multigrid level\n");
      param_coarse = new MGParam(param, *B_coarse, matCoarseResidual, matCoarseSmoother, matCoarseSmootherSloppy, param.level+1);
      param_coarse->fine = this;
      param_coarse->delta = 1e-20;

      coarse = new MG(*param_coarse, profile_global);

      setOutputPrefix(prefix); // restore since we just popped back from coarse grid

      // if on the second to bottom level then we can just use the coarse solver as is
      if (param.cycle_type == QUDA_MG_CYCLE_VCYCLE || param.level == param.Nlevel-2) {
	coarse_solver = coarse;
	printfQuda("Assigned coarse solver to coarse MG operator\n");
      } else if (param.cycle_type == QUDA_MG_CYCLE_RECURSIVE) {
	param_coarse_solver = new SolverParam(param);

	param_coarse_solver->inv_type = QUDA_GCR_INVERTER;
	param_coarse_solver->inv_type_precondition = QUDA_MG_INVERTER;
	param_coarse_solver->preconditioner = coarse;

	param_coarse_solver->is_preconditioner = false;
	param_coarse_solver->preserve_source = QUDA_PRESERVE_SOURCE_YES;
	param_coarse_solver->use_init_guess = QUDA_USE_INIT_GUESS_NO;
	param_coarse_solver->maxiter = 11; // FIXME - dirty hack
	param_coarse_solver->Nkrylov = 10;
	param_coarse_solver->tol = param.mg_global.smoother_tol[param.level+1];
	param_coarse_solver->global_reduction = true;
	param_coarse_solver->compute_true_res = false;
	param_coarse_solver->delta = 1e-8;
	param_coarse_solver->verbosity_precondition = param.mg_global.verbosity[param.level+1];
	param_coarse_solver->pipeline = 5;

	// need this to ensure we don't use half precision on the preconditioner in GCR
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

	printfQuda("Assigned coarse solver to preconditioned GCR solver\n");
      } else {
	errorQuda("Multigrid cycle type %d not supported", param.cycle_type);
      }

    }

    printfQuda("setup completed\n");

    // now we can run through the verification if requested
    if (param.level == 0 && param.mg_global.run_verify) verify();

    if (param.level == 0) reset();

    // print out profiling information for the adaptive setup
    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
    // Reset the profile for accurate solver timing
    profile.TPRESET();

    setOutputPrefix("");
  }

  void MG::reset() {
    QudaSiteSubset site_subset = param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
    QudaMatPCType matpc_type = param.mg_global.invert_param->matpc_type;
    QudaParity parity = (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
    transfer->setSiteSubset(site_subset, parity); // use this to force location of transfer

    if (param.level < param.Nlevel-2) coarse->reset();
  }


  void MG::createSmoother() {
    // create the smoother for this level
    printfQuda("smoother has operator %s\n", typeid(param.matSmooth).name());

    param_presmooth = new SolverParam(param);

    param_presmooth->inv_type = param.smoother;
    param_presmooth->inv_type_precondition = QUDA_INVALID_INVERTER;
    param_presmooth->is_preconditioner = false;
    param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_NO;
    param_presmooth->use_init_guess = QUDA_USE_INIT_GUESS_NO;
    param_presmooth->maxiter = param.nu_pre;
    param_presmooth->Nkrylov = 4;
    param_presmooth->tol = param.smoother_tol;
    param_presmooth->global_reduction = param.global_reduction;
    if (param.level == 0) {
       param_presmooth->precision_sloppy = param.mg_global.invert_param->cuda_prec_precondition;
       param_presmooth->precision_precondition = param.mg_global.invert_param->cuda_prec_precondition;
    }

    if (param.level==param.Nlevel-1) {
      param_presmooth->Nkrylov = 20;
      param_presmooth->maxiter = 1000;
      param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_NO;
      param_presmooth->delta = 1e-8;
      param_presmooth->compute_true_res = false;
      param_presmooth->pipeline = 8;
    }

    presmoother = Solver::create(*param_presmooth, *param.matSmooth,
				 *param.matSmoothSloppy, *param.matSmoothSloppy, profile);

    if (param.level < param.Nlevel-1) { //Create the post smoother
      param_postsmooth = new SolverParam(*param_presmooth);
      param_postsmooth->use_init_guess = QUDA_USE_INIT_GUESS_YES;
      param_postsmooth->maxiter = param.nu_post;
      postsmoother = Solver::create(*param_postsmooth, *param.matSmooth,
				    *param.matSmoothSloppy, *param.matSmoothSloppy, profile);
    }
  }

  void MG::destroySmoother() {
    if (param.level < param.Nlevel-1) {
      if (postsmoother) delete postsmoother;
      postsmoother = nullptr;
    }
    if (presmoother) delete presmoother;
    presmoother = nullptr;

    if (param_presmooth) delete param_presmooth;
    param_presmooth = nullptr;
    if (param_postsmooth) delete param_postsmooth;
    param_postsmooth = nullptr;
  }

  MG::~MG() {
    if (param.level < param.Nlevel-1) {
      if (param.level < param.Nlevel-2 && param.cycle_type == QUDA_MG_CYCLE_RECURSIVE) {
	delete coarse_solver;
	delete param_coarse_solver;
      }

      if (B_coarse) {
	int nVec_coarse = std::max(param.Nvec, param.mg_global.n_vec[param.level+1]);
	for (int i=0; i<nVec_coarse; i++) if ((*B_coarse)[i]) delete (*B_coarse)[i];
	delete B_coarse;
      }
      if (coarse) delete coarse;
      if (transfer) delete transfer;
      if (matCoarseSmootherSloppy) delete matCoarseSmootherSloppy;
      if (diracCoarseSmootherSloppy && diracCoarseSmootherSloppy != diracCoarseSmoother) delete diracCoarseSmootherSloppy;
      if (matCoarseSmoother) delete matCoarseSmoother;
      if (diracCoarseSmoother) delete diracCoarseSmoother;
      if (matCoarseResidual) delete matCoarseResidual;
      if (diracCoarseResidual) delete diracCoarseResidual;
    }

    destroySmoother();

    if (b_tilde && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) delete b_tilde;
    if (r) delete r;
    if (r_coarse) delete r_coarse;
    if (x_coarse) delete x_coarse;
    if (tmp_coarse) delete tmp_coarse;

    if (param_coarse) delete param_coarse;

    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
  }

  double MG::flops() const {
    double flops = 0;
    if (param.level < param.Nlevel-1) flops += coarse->flops();

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
    double tol = std::pow(10.0, 4-2*csParam.precision);

    printfQuda("\n");
    printfQuda("Checking 0 = (1 - P P^\\dagger) v_k for %d vectors\n", param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      // as well as copying to the correct location this also changes basis if necessary
      *tmp1 = *param.B[i]; 

      transfer->R(*r_coarse, *tmp1);
      transfer->P(*tmp2, *r_coarse);

      printfQuda("Vector %d: norms v_k = %e P^\\dagger v_k = %e P P^\\dagger v_k = %e\n",
		 i, norm2(*tmp1), norm2(*r_coarse), norm2(*tmp2));

      deviation = sqrt( xmyNorm(*tmp1, *tmp2) / norm2(*tmp1) );
      printfQuda("L2 relative deviation = %e\n", deviation);
      if (deviation > tol) errorQuda("failed");
    }

#if 0
    printfQuda("Checking 1 > || (1 - D P (P^\\dagger D P) P^\\dagger v_k || / || v_k || for %d vectors\n", 
	       param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      transfer->R(*r_coarse, *(param.B[i]));
      (*coarse)(*x_coarse, *r_coarse); // this needs to be an exact solve to pass
      setOutputPrefix(prefix); // restore output prefix
      transfer->P(*tmp2, *x_coarse);
      param.matResidual(*tmp1,*tmp2);
      *tmp2 = *(param.B[i]);
      printfQuda("Vector %d: norms %e %e ", i, norm2(*param.B[i]), norm2(*tmp1));
      printfQuda("relative residual = %e\n", sqrt(xmyNorm(*tmp2, *tmp1) / norm2(*param.B[i])) );
    }
#endif

    printfQuda("\n");
    printfQuda("Checking 0 = (1 - P^\\dagger P) eta_c\n");
    x_coarse->Source(QUDA_RANDOM_SOURCE);
    transfer->P(*tmp2, *x_coarse);
    transfer->R(*r_coarse, *tmp2);
    printfQuda("Vector norms %e %e (fine tmp %e) ", norm2(*x_coarse), norm2(*r_coarse), norm2(*tmp2));

    deviation = sqrt( xmyNorm(*x_coarse, *r_coarse) / norm2(*x_coarse) );
    printfQuda("L2 relative deviation = %e\n", deviation);
    if (deviation > tol ) errorQuda("failed");

    printfQuda("\n");
    printfQuda("Comparing native coarse operator to emulated operator\n");
    ColorSpinorField *tmp_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, param.mg_global.location[param.level+1]);
    zero(*tmp_coarse);
    zero(*r_coarse);

    tmp_coarse->Source(QUDA_RANDOM_SOURCE);
    transfer->P(*tmp1, *tmp_coarse);

    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
      const Dirac &dirac = *(param.matSmooth->Expose());
      double kappa = param.matResidual->Expose()->Kappa();
      if (param.level==0) {
	dirac.DslashXpay(tmp2->Even(), tmp1->Odd(), QUDA_EVEN_PARITY, tmp1->Even(), -kappa);
	dirac.DslashXpay(tmp2->Odd(), tmp1->Even(), QUDA_ODD_PARITY, tmp1->Odd(), -kappa);
      } else { // this is a hack since the coarse Dslash doesn't properly use the same xpay conventions yet
	dirac.DslashXpay(tmp2->Even(), tmp1->Odd(), QUDA_EVEN_PARITY, tmp1->Even(), 1.0);
	dirac.DslashXpay(tmp2->Odd(), tmp1->Even(), QUDA_ODD_PARITY, tmp1->Odd(), 1.0);
      }
    } else {
      (*param.matResidual)(*tmp2,*tmp1);
    }

    transfer->R(*x_coarse, *tmp2);
    (*param_coarse->matResidual)(*r_coarse, *tmp_coarse);

#if 0 // enable to print out emulated and actual coarse-grid operator vectors for debugging
    printfQuda("emulated\n");
    for (int x=0; x<x_coarse->Volume(); x++) tmp1->PrintVector(x);

    printfQuda("actual\n");
    for (int x=0; x<r_coarse->Volume(); x++) tmp2->PrintVector(x);
#endif

    printfQuda("Vector norms Emulated=%e Native=%e ", norm2(*x_coarse), norm2(*r_coarse));

    deviation = sqrt( xmyNorm(*x_coarse, *r_coarse) / norm2(*x_coarse) );

    // When the mu is shifted on the coarse level; we can compute exxactly the error we introduce in the check:
    //  it is given by 2*kappa*delta_mu || tmp_coarse ||; where tmp_coarse is the random vector generated for the test
    if(param.matResidual->Expose()->Mu() != 0) {
      double delta_factor = param.mg_global.mu_factor[param.level+1] - param.mg_global.mu_factor[param.level];
      if(fabs(delta_factor) > tol ) {
	double delta_a = delta_factor * 2.0 * param.matResidual->Expose()->Kappa() *
	  param.matResidual->Expose()->Mu() * transfer->Vectors().TwistFlavor();
	deviation -= fabs(delta_a) * sqrt( norm2(*tmp_coarse) / norm2(*x_coarse) );
	deviation = fabs(deviation);
      }
    }
    printfQuda("L2 relative deviation = %e\n\n", deviation);
    if (deviation > tol) errorQuda("failed");
    
    // here we check that the Hermitian conjugate operator is working
    // as expected for both the smoother and residual Dirac operators
    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
      const Dirac &diracS = *(param.matSmooth->Expose());
      diracS.MdagM(tmp2->Even(), tmp1->Odd());
      Complex dot = cDotProduct(tmp2->Even(),tmp1->Odd());
      double deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      printfQuda("Smoother normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
		 real(dot), imag(dot), deviation);
      if (deviation > tol) errorQuda("failed");

      const Dirac &diracR = *(param.matResidual->Expose());
      diracR.MdagM(*tmp2, *tmp1);
      dot = cDotProduct(*tmp2,*tmp1);

      deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      printfQuda("Residual normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
		 real(dot), imag(dot), deviation);
      if (deviation > tol) errorQuda("failed");
    } else {
      const Dirac &dirac = *(param.matResidual->Expose());

      dirac.MdagM(*tmp2, *tmp1);
      Complex dot = cDotProduct(*tmp1,*tmp2);

      double deviation = std::fabs(dot.imag()) / std::fabs(dot.real());
      printfQuda("Normal operator test (eta^dag M^dag M eta): real=%e imag=%e, relative imaginary deviation=%e\n",
		 real(dot), imag(dot), deviation);
      if (deviation > tol) errorQuda("failed");
    }

#ifdef ARPACK_LIB
    printfQuda("\n");
    printfQuda("Check eigenvector overlap for level %d\n", param.level );

    int nmodes = 128;
    int ncv    = 256;
    double arpack_tol = 1e-7;
    char *which = (char*)malloc(256*sizeof(char));
    sprintf(which, "SM");/* ARPACK which="{S,L}{R,I,M}" */

    ColorSpinorParam cpuParam(*param.B[0]);
    cpuParam.create = QUDA_ZERO_FIELD_CREATE;

    cpuParam.location = QUDA_CPU_FIELD_LOCATION;
    cpuParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    if(param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) { 
      cpuParam.x[0] /= 2; 
      cpuParam.siteSubset = QUDA_PARITY_SITE_SUBSET; 
    }

    std::vector<ColorSpinorField*> evecsBuffer;
    evecsBuffer.reserve(nmodes);

    for (int i = 0; i < nmodes; i++) evecsBuffer.push_back( new cpuColorSpinorField(cpuParam) );

    QudaPrecision matPrecision = QUDA_SINGLE_PRECISION;//manually ajusted?
    QudaPrecision arpPrecision = QUDA_DOUBLE_PRECISION;//precision used in ARPACK routines, may not coincide with matvec precision
    
    void *evalsBuffer =  arpPrecision == QUDA_DOUBLE_PRECISION ? static_cast<void*>(new std::complex<double>[nmodes+1]) : static_cast<void*>( new std::complex<float>[nmodes+1]);
    //
    arpackSolve( evecsBuffer, evalsBuffer, *param.matSmooth,  matPrecision,  arpPrecision, arpack_tol, nmodes, ncv,  which);

    for (int i=0; i<nmodes; i++) {
      // as well as copying to the correct location this also changes basis if necessary
      *tmp1 = *evecsBuffer[i]; 

      transfer->R(*r_coarse, *tmp1);
      transfer->P(*tmp2, *r_coarse);

      printfQuda("Vector %d: norms v_k = %e P^\\dagger v_k = %e P P^\\dagger v_k = %e\n",
		 i, norm2(*tmp1), norm2(*r_coarse), norm2(*tmp2));

      deviation = sqrt( xmyNorm(*tmp1, *tmp2) / norm2(*tmp1) );
      printfQuda("L2 relative deviation = %e\n", deviation);
    }

    for (unsigned int i = 0; i < evecsBuffer.size(); i++) delete evecsBuffer[i];

    if( arpPrecision == QUDA_DOUBLE_PRECISION )  delete static_cast<std::complex<double>* >(evalsBuffer);
    else                                         delete static_cast<std::complex<float>* > (evalsBuffer);
 
    free(which);
#else
    warningQuda("\nThis test requires ARPACK.\n");
#endif

    delete tmp1;
    delete tmp2;
    delete tmp_coarse;

    if (param.level < param.Nlevel-2) coarse->verify();
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

      const Dirac &dirac = *(param.matSmooth->Expose());
      dirac.prepare(in, out, x, residual, outer_solution_type);

      // b_tilde holds either a copy of preconditioned source or a pointer to original source
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) *b_tilde = *in;
      else b_tilde = &b;

      (*presmoother)(*out, *in);

      ColorSpinorField &solution = inner_solution_type == outer_solution_type ? x : x.Even();
      dirac.reconstruct(solution, b, inner_solution_type);

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

      // do the post smoothing
      //residual = outer_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even(); // refine for outer solution type
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
	in = b_tilde;
      } else { // this incurs unecessary copying
	*r = b;
	in = r;
      }

      //dirac.prepare(in, out, solution, residual, inner_solution_type);
      // we should keep a copy of the prepared right hand side as we've already destroyed it
      (*postsmoother)(*out, *in); // for inner solve preconditioned, in the should be the original prepared rhs

      dirac.reconstruct(x, b, outer_solution_type);

    } else { // do the coarse grid solve

      const Dirac &dirac = *(param.matSmooth->Expose());
      ColorSpinorField *out=nullptr, *in=nullptr;

      dirac.prepare(in, out, x, b, outer_solution_type);
      (*presmoother)(*out, *in);
      dirac.reconstruct(x, b, outer_solution_type);
    }

    if ( debug ) {
      (*param.matResidual)(*r, x);
      double r2 = xmyNorm(b, *r);
      printfQuda("leaving V-cycle with x2=%e, r2=%e\n", norm2(x), r2);
    }

    setOutputPrefix(param.level == 0 ? "" : prefix_bkup);
  }

  //supports seperate reading or single file read
  void MG::loadVectors(std::vector<ColorSpinorField*> &B) {
    profile_global.TPSTOP(QUDA_PROFILE_INIT);
    profile_global.TPSTART(QUDA_PROFILE_IO);

    std::string vec_infile(param.mg_global.vec_infile);
    vec_infile += "_level_";
    vec_infile += std::to_string(param.level);

    const int Nvec = B.size();
    printfQuda("Start loading %d vectors from %s\n", Nvec, vec_infile.c_str());

    void **V = new void*[Nvec];
    for (int i=0; i<Nvec; i++) { 
      V[i] = B[i]->V();
      if (V[i] == NULL) {
	printfQuda("Could not allocate V[%d]\n", i);
      }
    }

    if (strcmp(vec_infile.c_str(),"")!=0) {
#ifdef HAVE_QIO
      read_spinor_field(vec_infile.c_str(), &V[0], B[0]->Precision(), B[0]->X(),
			B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0,  (char**)0);
#else
      errorQuda("\nQIO library was not built.\n");      
#endif
    } else {
      printfQuda("Using %d constant nullvectors\n", Nvec);
      //errorQuda("No nullspace file defined");

      for (int i = 0; i < (Nvec < 2 ? Nvec : 2); i++) {
	zero(*B[i]);
#if 1
	ColorSpinorParam csParam(*B[i]);
	csParam.create = QUDA_ZERO_FIELD_CREATE;
	ColorSpinorField *tmp = ColorSpinorField::Create(csParam);
	for (int s=i; s<4; s+=2) {
	  for (int c=0; c<B[i]->Ncolor(); c++) {
            tmp->Source(QUDA_CONSTANT_SOURCE, 1, s, c);
	    //tmp->Source(QUDA_SINUSOIDAL_SOURCE, 3, s, 2); // sin in dim 3, mode s, offset = 2
	    xpy(*tmp,*B[i]);
	  }
	}
	delete tmp;
#else
	printfQuda("Using random source for nullvector = %d\n",i);
	B[i]->Source(QUDA_RANDOM_SOURCE);
#endif
	//printfQuda("B[%d]\n",i);
	//for (int x=0; x<B[i]->Volume(); x++) static_cast<cpuColorSpinorField*>(B[i])->PrintVector(x);
      }

      for (int i=2; i<Nvec; i++) B[i] -> Source(QUDA_RANDOM_SOURCE);
    }

    printfQuda("Done loading vectors\n");
    profile_global.TPSTOP(QUDA_PROFILE_IO);
    profile_global.TPSTART(QUDA_PROFILE_INIT);
  }

  void MG::saveVectors(std::vector<ColorSpinorField*> &B) {
#ifdef HAVE_QIO
    profile_global.TPSTOP(QUDA_PROFILE_INIT);
    profile_global.TPSTART(QUDA_PROFILE_IO);
    std::string vec_outfile(param.mg_global.vec_outfile);
    vec_outfile += "_level_";
    vec_outfile += std::to_string(param.level);

    if (strcmp(param.mg_global.vec_outfile,"")!=0) {
      const int Nvec = B.size();
      printfQuda("Start saving %d vectors to %s\n", Nvec, vec_outfile.c_str());

      void **V = static_cast<void**>(safe_malloc(Nvec*sizeof(void*)));
      for (int i=0; i<Nvec; i++) {
	V[i] = B[i]->V();
	if (V[i] == NULL) {
	  printfQuda("Could not allocate V[%d]\n", i);
	}
      }

      write_spinor_field(vec_outfile.c_str(), &V[0], B[0]->Precision(), B[0]->X(),
			 B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0,  (char**)0);

      host_free(V);
      printfQuda("Done saving vectors\n");
    }

    profile_global.TPSTOP(QUDA_PROFILE_IO);
    profile_global.TPSTART(QUDA_PROFILE_INIT);
#else
    if (strcmp(param.mg_global.vec_outfile,"")!=0) {
      errorQuda("\nQIO library was not built.\n");
    }
#endif
  }

  void MG::generateNullVectors(std::vector<ColorSpinorField*> B) {
    printfQuda("\nGenerate null vectors\n");

    SolverParam solverParam(param);  // Set solver field parameters:

    // set null-space generation options - need to expose these
    solverParam.maxiter = 500;
    solverParam.tol = param.mg_global.setup_tol[param.level];
    solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
    solverParam.delta = 1e-7; 
    solverParam.inv_type = param.mg_global.setup_inv_type[param.level];
    solverParam.Nkrylov = 4;
    solverParam.pipeline = (solverParam.inv_type == QUDA_BICGSTABL_INVERTER ? 4 : 0); // pipeline != 0 breaks BICGSTAB
    
    if (param.level == 0) { // this enables half precision on the fine grid only if set
      solverParam.precision_sloppy = param.mg_global.invert_param->cuda_prec_precondition;
      solverParam.precision_precondition = param.mg_global.invert_param->cuda_prec_precondition;
      if (solverParam.precision_sloppy == QUDA_HALF_PRECISION) solverParam.delta = 1e-1;
    }
    solverParam.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
    solverParam.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;

    ColorSpinorParam csParam(*B[0]);  // Create spinor field parameters:
    // to force setting the field to be native first set to double-precision native order
    // then use the setPrecision method to set to native order
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.precision = QUDA_DOUBLE_PRECISION;
    csParam.setPrecision(B[0]->Precision());

    csParam.location = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    ColorSpinorField *b = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
    csParam.create = QUDA_NULL_FIELD_CREATE;

    std::vector<ColorSpinorField*> B_gpu;

    const Dirac &dirac = *(param.matSmooth->Expose());
    Solver *solve;
    DiracMdagM mdagm(dirac);
    const Dirac &diracSloppy = *(param.matSmoothSloppy->Expose());
    DiracMdagM mdagmSloppy(diracSloppy);
    if(solverParam.inv_type == QUDA_CG_INVERTER) {
      solverParam.maxiter = 2000;
      solve = Solver::create(solverParam, mdagm, mdagmSloppy, mdagmSloppy, profile);
    } else if(solverParam.inv_type == QUDA_GCR_INVERTER) {
      solverParam.inv_type_precondition = param.mg_global.smoother[param.level];
      solverParam.tol_precondition = param.mg_global.smoother_tol[param.level];
      solverParam.maxiter_precondition = param.mg_global.nu_pre[param.level]+param.mg_global.nu_post[param.level];
      solverParam.precondition_cycle = 1;
      solve = Solver::create(solverParam, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy, profile);
    } else {
      solve = Solver::create(solverParam, *param.matSmooth, *param.matSmoothSloppy, *param.matSmoothSloppy, profile);
    }

    // Generate sources and launch solver for each source:
    for(unsigned int i=0; i<B.size(); i++) {
      B[i]->Source(QUDA_RANDOM_SOURCE); //random initial guess

      B_gpu.push_back(ColorSpinorField::Create(csParam));
      ColorSpinorField *x = B_gpu[i];
      *x = *B[i]; // copy initial guess to GPU:

      zero(*b); // need zero rhs

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Initial guess = %g\n", norm2(*x));

      ColorSpinorField *out=nullptr, *in=nullptr;
      dirac.prepare(in, out, *x, *b, QUDA_MAT_SOLUTION);
      (*solve)(*out, *in);
      dirac.reconstruct(*x, *b, QUDA_MAT_SOLUTION);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Solution = %g\n", norm2(*x));

      // global orthonormalization of the generated null-space vectors
      for (int i=0; B_gpu[i] != x; ++i) {
	Complex alpha = cDotProduct(*B_gpu[i], *x);//<j,i>
	caxpy(-alpha, *B_gpu[i], *x); //i-<j,i>j
      }

      double nrm2 = norm2(*x);
      if (nrm2 > 1e-16) ax(1.0 /sqrt(nrm2), *x);
      else errorQuda("\nCannot orthogonalize %u vector\n", i);

    }

    delete solve;
    delete b;

    for (int i=0; i<(int)B.size(); i++) {
      *B[i] = *B_gpu[i];
      delete B_gpu[i];
    }

    if (strcmp(param.mg_global.vec_outfile,"")!=0) { // only save if outfile is defined
      saveVectors(B);
    }

    return;
  }

}
