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
    
    if( param.mg_global.generate_all_levels == QUDA_BOOLEAN_YES ) {
 
       if (param.level < param.Nlevel-1 ) { // null space generation only on level 1 currently
           if (param.mg_global.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
	      generateNullVectors(param.B);
           } else if(param.mg_global.compute_null_vector == QUDA_COMPUTE_LOW_MODE_VECTOR){
              computeLowModeVectors(param.B);
              warningQuda("\nUse eigensolver to generate nullspace..\n");
      	   } else {
		loadVectors(param.B);
      	   }
        }

    } else if ( param.mg_global.generate_all_levels == QUDA_BOOLEAN_NO ) {

       if (param.level == 0 ) { // null space generation only on level 1 currently
           if (param.mg_global.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
              generateNullVectors(param.B);
           } else if(param.mg_global.compute_null_vector == QUDA_COMPUTE_LOW_MODE_VECTOR){
              computeLowModeVectors(param.B);
              warningQuda("\nUse eigensolver to generate nullspace..\n");
           } else {
                loadVectors(param.B);
           }
       }
    }
    else {
	errorQuda("In MG Global. generate_all_levels is neither QUDA_BOOLEAN_YES, nor QUDA_BOOLEAN_NO");
    }

    if (param.level >= QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level+1);

    // create the smoother for this level
    printfQuda("smoother has operator %s\n", typeid(param.matSmooth).name());

    param_presmooth = new SolverParam(param);

    param_presmooth->inv_type = param.smoother;
    param_presmooth->inv_type_precondition = QUDA_INVALID_INVERTER;
    param_presmooth->is_preconditioner = false;
    param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_NO;
    param_presmooth->use_init_guess = QUDA_USE_INIT_GUESS_NO;
    param_presmooth->maxiter = param.nu_pre;
    param_presmooth->Nkrylov = 6;
    param_presmooth->tol = param.smoother_tol;
    param_presmooth->global_reduction = param.global_reduction;
    if (param.level == 0) {
       param_presmooth->precision_sloppy = param.mg_global.invert_param->cuda_prec_precondition;
       param_presmooth->precision_precondition = param.mg_global.invert_param->cuda_prec_precondition;
    }

    if (param.level==param.Nlevel-1) {
      param_presmooth->Nkrylov = 64;
      param_presmooth->maxiter = 1000;
      param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_NO;
      param_presmooth->delta = 1e-8;
      param_presmooth->compute_true_res = false;
      param_presmooth->pipeline = 5;
    }

    presmoother = Solver::create(*param_presmooth, param.matSmooth,
				 param.matSmoothSloppy, param.matSmoothSloppy, profile);

    if (param.level < param.Nlevel-1) { //Create the post smoother
      param_postsmooth = new SolverParam(*param_presmooth);
      param_postsmooth->use_init_guess = QUDA_USE_INIT_GUESS_YES;
      param_postsmooth->maxiter = param.nu_post;
      postsmoother = Solver::create(*param_postsmooth, param.matSmooth,
				    param.matSmoothSloppy, param.matSmoothSloppy, profile);
    }

    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && (param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE && param.smoother_solve_type != QUDA_NORMOP_PC_SOLVE))
      errorQuda("Cannot use preconditioned coarse grid solution without preconditioned smoother solve");

    // create residual vectors
    {
      ColorSpinorParam csParam(*(param.B[0]));
      csParam.create = QUDA_NULL_FIELD_CREATE;
      csParam.location = param.location;
      if (csParam.location==QUDA_CUDA_FIELD_LOCATION) {
	// all coarse GPU vectors use FLOAT2 ordering (also fine level for staggered field)
	csParam.fieldOrder = (csParam.precision == QUDA_DOUBLE_PRECISION || param.level > 0 || (param.level == 0 && param.B[0]->Nspin() == 1)) ?
	  QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
	csParam.setPrecision(csParam.precision);
	csParam.gammaBasis = param.level > 0 ? QUDA_DEGRAND_ROSSI_GAMMA_BASIS: QUDA_UKQCD_GAMMA_BASIS;
      }
      if(param.B[0]->Nspin() == 1)  csParam.gammaBasis = param.B[0]->GammaBasis();//We need this hack for staggered.
      r = ColorSpinorField::Create(csParam);

      // if we're using preconditioning then allocate storate for the preconditioned source vector
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE || param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE) {
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

      //Complex alpha = Complex(0.00889, 12.1);//if zero: no smoothing
//WARNING:
      Complex alpha = Complex(0.0, 0.0);//if zero: no smoothing
      printfQuda("\nSmoothing params: %le %le\n", alpha.real(), alpha.imag());
      transfer = new Transfer(param.B, &param.matResidual, alpha,  param.Nvec, param.geoBlockSize, param.spinBlockSize,
			      param.location == QUDA_CUDA_FIELD_LOCATION ? true : false, profile);
      for (int i=0; i<QUDA_MAX_MG_LEVEL; i++) param.mg_global.geo_block_size[param.level][i] = param.geoBlockSize[i];

      //transfer->setTransferGPU(false); // use this to force location of transfer
      //transfer->setSiteSubset(QUDA_FULL_SITE_SUBSET , (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY);
      printfQuda("end creating transfer operator\n");

      // create coarse residual vector
      r_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, param.mg_global.location[param.level+1]);
      // create coarse solution vector
      x_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, param.mg_global.location[param.level+1]);

      // create coarse temporary vector
      tmp_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec, param.mg_global.location[param.level+1]);

      // check if we are coarsening the preconditioned system then
      bool preconditioned_coarsen = (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE || param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE));

      // create coarse grid operator
      DiracParam diracParam;
      diracParam.transfer = transfer;
      diracParam.dirac = preconditioned_coarsen ? const_cast<Dirac*>(param.matSmooth.Expose()) : const_cast<Dirac*>(param.matResidual.Expose());
      if (param.B[0]->Nspin() != 1)
      {
        diracParam.kappa = param.matResidual.Expose()->Kappa();
        printfQuda("Kappa = %e\n", diracParam.kappa);
      }else {
        printfQuda("Mass = %e\n", diracParam.mass);
      }
      diracParam.dagger = QUDA_DAG_NO;
      diracParam.matpcType = matpc_type;
      diracParam.tmp1 = &(tmp_coarse->Even());
      // use even-odd preconditioning for the coarse grid solver
      diracCoarseResidual = new DiracCoarse(diracParam);
      matCoarseResidual = new DiracM(*diracCoarseResidual);

      // create smoothing operators
      //diracParam.dirac = const_cast<Dirac*>(param.matSmooth.Expose());
      diracParam.dirac =  (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_SOLVE ) ? const_cast<Dirac*>(param.matResidual.Expose()) : const_cast<Dirac*>(param.matSmooth.Expose());
      diracCoarseSmoother = (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE || param.mg_global.smoother_solve_type[param.level+1] == QUDA_NORMOP_PC_SOLVE) ?
      diracParam.type = (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE) ? QUDA_COARSEPC_DIRAC : QUDA_COARSE_DIRAC;
      diracCoarseSmoother = (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE) ?
	new DiracCoarsePC(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam) :
	new DiracCoarse(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam);
      diracCoarseSmootherSloppy = diracCoarseSmoother;  // for coarse grids these always alias for now (FIXME half precision support for coarse op)

      matCoarseSmoother = new DiracM(*diracCoarseSmoother);
      matCoarseSmootherSloppy = new DiracM(*diracCoarseSmootherSloppy);

      // coarse null space vectors (dummy for now)
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
      param_coarse = new MGParam(param, *B_coarse, *matCoarseResidual, *matCoarseSmoother, *matCoarseSmootherSloppy, param.level+1);
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
	param_coarse_solver->verbosity_precondition = QUDA_SILENT;
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

    // print out profiling information for the adaptive setup
    if (getVerbosity() >= QUDA_SUMMARIZE) profile.Print();
    // Reset the profile for accurate solver timing
    profile.TPRESET();

    setOutputPrefix("");
  }

  MG::~MG() {
    if (param.level < param.Nlevel-1) {
      if (param.level < param.Nlevel-2 && param.cycle_type == QUDA_MG_CYCLE_RECURSIVE) {
	delete coarse_solver;
	delete param_coarse_solver;
      }

      if (B_coarse) {
	for (int i=0; i<param.Nvec; i++) if ((*B_coarse)[i]) delete (*B_coarse)[i];
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
      if (postsmoother) delete postsmoother;
    }
    if (presmoother) delete presmoother;

    if (b_tilde && (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE || param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE)) delete b_tilde;
    if (r) delete r;
    if (r_coarse) delete r_coarse;
    if (x_coarse) delete x_coarse;
    if (tmp_coarse) delete tmp_coarse;

    if (param_coarse) delete param_coarse;
    if (param_presmooth) delete param_presmooth;
    if (param_postsmooth) delete param_postsmooth;

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

    Complex alpha_tmp = transfer->Alpha();
    transfer->SetAlpha(Complex(0.0, 0.0));

    // temporary fields used for verification
    ColorSpinorParam csParam(*r);
    csParam.create = !param.mg_global._2d_u1_emulation ? QUDA_NULL_FIELD_CREATE : QUDA_ZERO_FIELD_CREATE;

    if(param.B[0]->Nspin() == 1)  csParam.gammaBasis = param.B[0]->GammaBasis();

    ColorSpinorField *tmp1 = ColorSpinorField::Create(csParam);
    ColorSpinorField *tmp2 = ColorSpinorField::Create(csParam);

    ColorSpinorField *tmp5 = nullptr; 
    ColorSpinorField *tmp6 = nullptr; 
    if(param.B[0]->Nspin() == 1)
    {
      csParam.extendDimensionality();
      tmp5 = ColorSpinorField::Create(csParam);
      tmp6 = ColorSpinorField::Create(csParam);
      csParam.reduceDimensionality();
    }
    else
    {
      tmp5 = tmp1;
      tmp6 = tmp2;
    }

    double deviation;
    double tol = std::pow(10.0, 4-2*csParam.precision);
    //double tol = 1e-3;
#if 1
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
#endif 
#if 0 //in emulation mode this does not work
    printfQuda("Checking 1 > || (1 - P (P^\\dagger D P) P^\\dagger D v_k || / || v_k || for %d vectors\n", 
	       param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      *tmp1 = *param.B[i];
      param.matResidual(*tmp2,*tmp1);
      transfer->R(*r_coarse, *tmp2);
      (*coarse)(*x_coarse, *r_coarse); // this needs to be an exact solve to pass
      setOutputPrefix(prefix); // restore output prefix
      transfer->P(*tmp2, *x_coarse);
      printfQuda("Vector %d: norms %e %e ", i, norm2(*param.B[i]), norm2(*tmp2));
      printfQuda("relative residual = %e\n", sqrt(xmyNorm(*tmp1, *tmp2) / norm2(*param.B[i])) );
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
    transfer->P(*tmp1, *tmp_coarse);//convert 4d to 5d
    if(param.B[0]->Nspin() == 1){
      //if(csParam.location == QUDA_CPU_LOCATION) (static_cast<cpuColorSpinorField*>(tmp5))->copy(*tmp1);
      //else (static_cast<cudaColorSpinorField*>(tmp5))->copy(*tmp1);
      *tmp5 = *tmp1;
    }

    if (param.coarse_grid_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE) {
      const Dirac &dirac = *(param.matSmooth.Expose());
      double kappa = param.matResidual.Expose()->Kappa();
      if (param.level==0) {
	dirac.DslashXpay(tmp6->Even(), tmp5->Odd(), QUDA_EVEN_PARITY, tmp5->Even(), -kappa);
	dirac.DslashXpay(tmp6->Odd(), tmp5->Even(), QUDA_ODD_PARITY, tmp5->Odd(), -kappa);
      } else { // this is a hack since the coarse Dslash doesn't proerly use the same xpay conventions yet
	dirac.DslashXpay(tmp6->Even(), tmp5->Odd(), QUDA_EVEN_PARITY, tmp5->Even(), 1.0);
	dirac.DslashXpay(tmp6->Odd(), tmp5->Even(), QUDA_ODD_PARITY, tmp5->Odd(), 1.0);
      }
    } else {
      param.matResidual(*tmp6,*tmp5);
    }

    if(param.B[0]->Nspin() == 1) *tmp2 = *tmp6; //tmp2->copy(*tmp6);
    transfer->R(*x_coarse, *tmp2);
    param_coarse->matResidual(*r_coarse, *tmp_coarse);

#if 0 // enable to print out emulated and actual coarse-grid operator vectors for debugging
    {
      printfQuda("emulated\n");
      ColorSpinorParam param(*x_coarse);
      param.location = QUDA_CPU_FIELD_LOCATION;
      param.create = QUDA_NULL_FIELD_CREATE;
      param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

      cpuColorSpinorField *v1 = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(param));
      *v1 = *x_coarse;
      for (int x=0; x<x_coarse->Volume(); x++) v1->PrintVector(x);

      printfQuda("actual\n");
      cpuColorSpinorField *v2 = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(param));
      *v2 = *r_coarse;
      for (int x=0; x<r_coarse->Volume(); x++) v2->PrintVector(x);

      delete v1;
      delete v2;
    }
#endif


    printfQuda("Vector norms Emulated=%e Native=%e ", norm2(*x_coarse), norm2(*r_coarse));
    deviation = sqrt( xmyNorm(*x_coarse, *r_coarse) / norm2(*x_coarse) );
    printfQuda("L2 relative deviation = %e\n\n", deviation);
    if (deviation > tol) errorQuda("failed");
#if 1
    printfQuda("\n");
    printfQuda("Check eigenvector overlap for level %d\n", param.level );

    int nmodes = 128;
    int ncv    = 256;
    char *which = (char*)malloc(256*sizeof(char));
    sprintf(which, "SM");/* ARPACK which="{S,L}{R,I,M}" */

    ColorSpinorParam cpuParam(*param.B[0]);
    cpuParam.create = QUDA_ZERO_FIELD_CREATE;
    cpuParam.extendDimensionality();//make 5d field

    cpuParam.location = QUDA_CPU_FIELD_LOCATION;
    cpuParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    std::vector<ColorSpinorField*> evecsBuffer;
    evecsBuffer.reserve(nmodes);

    for (int i = 0; i < nmodes; i++) evecsBuffer.push_back( new cpuColorSpinorField(cpuParam) );
    
    Complex *evalsBuffer = new Complex[nmodes+1];
    //
    QudaPrecision matPrecision = QUDA_SINGLE_PRECISION;//manually ajusted?
    ArpackArgs args(param.matResidual, matPrecision, nmodes, ncv, which);    
    if(param.mg_global._2d_u1_emulation) 
    {
      args.Set2D();
      args.SetReducedColors(1);
      args.SetTol(1e-7);
    }
    //
    args(evecsBuffer, evalsBuffer);

    for (int i=0; i<nmodes; i++) {
      // as well as copying to the correct location this also changes basis if necessary
      printfQuda("\nNorm : %le\n", blas::norm2(*evecsBuffer[i]));
      *tmp5 = *evecsBuffer[i]; 
      *tmp1 = *tmp5;

      transfer->R(*r_coarse, *tmp1);
      transfer->P(*tmp2, *r_coarse);

      printfQuda("Vector %d: norms v_k = %e P^\\dagger v_k = %e P P^\\dagger v_k = %e\n",
		 i, norm2(*tmp1), norm2(*r_coarse), norm2(*tmp2));

      deviation = sqrt( xmyNorm(*tmp1, *tmp2) / norm2(*tmp1) );
      printfQuda("L2 relative deviation = %e\n", deviation);
    }

    for (unsigned int i = 0; i < evecsBuffer.size(); i++) delete evecsBuffer[i];
    delete [] evalsBuffer;

    free(which);
#endif

    delete tmp1;
    delete tmp2;
    delete tmp_coarse;

    if(param.B[0]->Nspin() == 1) {delete tmp5; delete tmp6;}

    transfer->SetAlpha(alpha_tmp);

    if (param.level < param.Nlevel-2) coarse->verify();
  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    char prefix_bkup[100];  strncpy(prefix_bkup, prefix, 100);  setOutputPrefix(prefix);

    // if input vector is single parity then we must be solving the
    // preconditioned system in general this can only happen on the
    // top level
    QudaSolutionType outer_solution_type = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_MAT_SOLUTION : QUDA_MATPC_SOLUTION;
    QudaSolutionType inner_solution_type = param.coarse_grid_solution_type;

    if ( debug ) printfQuda("outer_solution_type = %d, inner_solution_type = %d\n", outer_solution_type, inner_solution_type);

    if ( outer_solution_type == QUDA_MATPC_SOLUTION && inner_solution_type == QUDA_MAT_SOLUTION)
      errorQuda("Unsupported solution type combination");

    if ( inner_solution_type == QUDA_MATPC_SOLUTION && (param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE && param.smoother_solve_type != QUDA_NORMOP_PC_SOLVE ))
      errorQuda("For this coarse grid solution type, a preconditioned smoother is required");

    if ( debug ) printfQuda("entering V-cycle with x2=%e, r2=%e\n", norm2(x), norm2(b));

    if (param.level < param.Nlevel-1) {
      //transfer->setTransferGPU(false); // use this to force location of transfer (need to check if still works for multi-level)
      
      // do the pre smoothing
      if ( debug ) printfQuda("pre-smoothing b2=%e\n", norm2(b));

      ColorSpinorField *out=nullptr, *in=nullptr;

      ColorSpinorField *y = nullptr;

      if(r->Location() == QUDA_CPU_FIELD_LOCATION)
      {
         ColorSpinorParam csParam(x);
         csParam.create = QUDA_ZERO_FIELD_CREATE;

         csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
         if (x.SiteSubset() == QUDA_PARITY_SITE_SUBSET) csParam.x[0] *= 2;
         y = static_cast<ColorSpinorField*> (new cudaColorSpinorField(csParam));
      }
      else
      {
         y = r;
      }

      //ColorSpinorField &residual = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? *r : r->Even();
      ColorSpinorField &residual = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? *y : y->Even();

      // FIXME only need to make a copy if not preconditioning
      residual = b; // copy source vector since we will overwrite source with iterated residual

      const Dirac &dirac = *(param.matSmooth.Expose());
      dirac.prepare(in, out, x, residual, outer_solution_type);

      // b_tilde holds either a copy of preconditioned source or a pointer to original source
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE || param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE) *b_tilde = *in;
      else b_tilde = &b;

      (*presmoother)(*out, *in);

      ColorSpinorField &solution = inner_solution_type == outer_solution_type ? x : x.Even();
      dirac.reconstruct(solution, b, inner_solution_type);

      // if using preconditioned smoother then need to reconstruct full residual
      // FIXME extend this check for precision, Schwarz, etc.
      bool use_solver_residual =
	( ((param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE || param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE) && inner_solution_type == QUDA_MATPC_SOLUTION) ||
	  (param.smoother_solve_type == QUDA_DIRECT_SOLVE && inner_solution_type == QUDA_MAT_SOLUTION) )
	? true : false;

      // FIXME this is currently borked if inner solver is preconditioned
      double r2 = 0.0;
      if (use_solver_residual) {
	//if (debug) r2 = norm2(*r);
	if (debug) r2 = norm2(*y);
      } else {
	//param.matResidual(*r, x);
	//if (debug) r2 = xmyNorm(b, *r);
	//else axpby(1.0, b, -1.0, *r);
	param.matResidual(*y, x);
	if (debug) r2 = xmyNorm(b, *y);
	else axpby(1.0, b, -1.0, *y);
      }

      // restrict to the coarse grid
      transfer->R(*r_coarse, residual);
      if ( debug ) printfQuda("after pre-smoothing x2 = %e, r2 = %e, r_coarse2 = %e\n", norm2(x), r2, norm2(*r_coarse));

      // recurse to the next lower level
      (*coarse_solver)(*x_coarse, *r_coarse);

      setOutputPrefix(prefix); // restore prefix after return from coarse grid

      if ( debug ) printfQuda("after coarse solve x_coarse2 = %e r_coarse2 = %e\n", norm2(*x_coarse), norm2(*r_coarse));

      // prolongate back to this grid
      //ColorSpinorField &x_coarse_2_fine = inner_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even(); // define according to inner solution type
      ColorSpinorField &x_coarse_2_fine = inner_solution_type == QUDA_MAT_SOLUTION ? *y : y->Even(); // define according to inner solution type
      transfer->P(x_coarse_2_fine, *x_coarse); // repurpose residual storage

      xpy(x_coarse_2_fine, solution); // sum to solution FIXME - sum should be done inside the transfer operator
      if ( debug ) {
	//printfQuda("Prolongated coarse solution y2 = %e\n", norm2(*r));
	//printfQuda("after coarse-grid correction x2 = %e, r2 = %e\n", 
	//	   norm2(x), norm2(*r));
	printfQuda("Prolongated coarse solution y2 = %e\n", norm2(*y));
	printfQuda("after coarse-grid correction x2 = %e, r2 = %e\n", 
		   norm2(x), norm2(*y));
      }

      // do the post smoothing
      //residual = outer_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even(); // refine for outer solution type
      if (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE || param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE) {
	in = b_tilde;
      } else { // this incurs unecessary copying
	//*r = b;
	//in = r;
	*y = b;
	in = y;
      }
      //dirac.prepare(in, out, solution, residual, inner_solution_type);
      // we should keep a copy of the prepared right hand side as we've already destroyed it
      (*postsmoother)(*out, *in); // for inner solve preconditioned, in the should be the original prepared rhs
      dirac.reconstruct(x, b, outer_solution_type);

      if(r->Location() == QUDA_CPU_FIELD_LOCATION) { delete y;}

    } else { // do the coarse grid solve
      const Dirac &dirac = *(param.matSmooth.Expose());
      ColorSpinorField *out=nullptr, *in=nullptr;

      dirac.prepare(in, out, x, b, outer_solution_type);
      (*presmoother)(*out, *in);
      dirac.reconstruct(x, b, outer_solution_type);
    }

    if ( debug ) {
      param.matResidual(*r, x);
      double r2 = xmyNorm(b, *r);
      printfQuda("leaving V-cycle with x2=%e, r2=%e\n", norm2(x), r2);
    }

    setOutputPrefix(param.level == 0 ? "" : prefix_bkup);
  }

  //supports seperate reading or single file read
  void MG::loadVectors(std::vector<ColorSpinorField*> &B) {
    profile_global.TPSTOP(QUDA_PROFILE_INIT);
    profile_global.TPSTART(QUDA_PROFILE_IO);
    const char *vec_infile = param.mg_global.vec_infile;

    const int Nvec = B.size();
    printfQuda("Start loading %d vectors from %s\n", Nvec, vec_infile);

    void **V = new void*[Nvec];
    for (int i=0; i<Nvec; i++) { 
      V[i] = B[i]->V();
      if (V[i] == NULL) {
	printfQuda("Could not allocate V[%d]\n", i);
      }
    }
    
    if (strcmp(vec_infile,"")!=0) {
#if 1
      read_spinor_field(vec_infile, &V[0], B[0]->Precision(), B[0]->X(), 
			B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0,  (char**)0);
#else 
      for (int i=0; i<Nvec; i++) {
	char filename[256];
	sprintf(filename, "%s.%d", vec_infile, i);
	printfQuda("Reading vector %d from file %s\n", i, filename);
	read_spinor_field(filename, &V[i], B[i]->Precision(), B[i]->X(), 
			  B[i]->Ncolor(), B[i]->Nspin(), 1, 0,  (char**)0);
      }
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
    profile_global.TPSTOP(QUDA_PROFILE_INIT);
    profile_global.TPSTART(QUDA_PROFILE_IO);
    const char *vec_outfile = param.mg_global.vec_outfile;

    if (strcmp(vec_outfile,"")!=0) {
      const int Nvec = B.size();
      printfQuda("Start saving %d vectors from %s\n", Nvec, vec_outfile);

      void **V = static_cast<void**>(safe_malloc(Nvec*sizeof(void*)));
      for (int i=0; i<Nvec; i++) {
	V[i] = B[i]->V();
	if (V[i] == NULL) {
	  printfQuda("Could not allocate V[%d]\n", i);
	}
      }

#if 1
      write_spinor_field(vec_outfile, &V[0], B[0]->Precision(), B[0]->X(),
			 B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0,  (char**)0);
#else
      for (int i=0; i<Nvec; i++) {
	char filename[256];
	sprintf(filename, "%s.%d", vec_outfile, i);
	printf("Saving vector %d to file %s\n", i, filename);
	write_spinor_field(filename, &V[i], B[i]->Precision(), B[i]->X(),
			   B[i]->Ncolor(), B[i]->Nspin(), 1, 0,  (char**)0);
      }
#endif

      host_free(V);
      printfQuda("Done saving vectors\n");
    }

    profile_global.TPSTOP(QUDA_PROFILE_IO);
    profile_global.TPSTART(QUDA_PROFILE_INIT);
  }

  void MG::generateNullVectors(std::vector<ColorSpinorField*> B) {
    printfQuda("\nGenerate null vectors\n");
    //Create spinor field parameters:
    const Dirac &dirac = *(param.matSmooth.Expose());

    SolverParam solverParam(param);

    // set null-space generation options - need to expose these
    solverParam.maxiter = 500;
    solverParam.tol = param.level == 0 ? 5e-5 : 1e-7;
    solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
    solverParam.delta = 1e-7;
    solverParam.inv_type = (param.level == 0 && param.matSmooth.isStaggered() && param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE) ? QUDA_CG_INVERTER : QUDA_BICGSTAB_INVERTER;
    //solverParam.inv_type = QUDA_BICGSTABL_INVERTER;
    solverParam.Nkrylov = 4;
    solverParam.pipeline = (solverParam.inv_type == QUDA_BICGSTABL_INVERTER ? 4 : 0); // pipeline != 0 breaks BICGSTAB

    if (param.level == 0 && !param.matSmooth.isStaggered() ) { // this enables half precision on the fine grid only if set
      solverParam.precision_sloppy = param.mg_global.invert_param->cuda_prec_precondition;
      solverParam.precision_precondition = param.mg_global.invert_param->cuda_prec_precondition;
      if (solverParam.precision_sloppy == QUDA_HALF_PRECISION) solverParam.delta = 1e-1;
    }
    // end setting null-space generation options

    solverParam.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
    solverParam.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;

    ColorSpinorParam csParam(*B[0]);

    // to force setting the field to be native first set to double-precision native order
    // then use the setPrecision method to set to native order
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.precision = QUDA_DOUBLE_PRECISION;
    csParam.setPrecision(B[0]->Precision());

    csParam.location = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now

     if(param.B[0]->Nspin() == 1)  
     {
       csParam.extendDimensionality(); //convert 4d into 5d object
       csParam.gammaBasis = param.B[0]->GammaBasis();//hack for staggered
     }
     else csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    cpuColorSpinorField *tmp_cpu_field = nullptr;

    // Generate sources and launch solver for each source:
    for(std::vector<ColorSpinorField*>::iterator nullvec = B.begin() ; nullvec != B.end(); ++nullvec) {
	cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*nullvec);
        if(!param.mg_global._2d_u1_emulation)
        {
	  curr_nullvec->Source(QUDA_RANDOM_SOURCE);//random initial guess
        }
        else
        {
          blas::zero(*curr_nullvec);//need this
          generic2DSource(*curr_nullvec);
        }
        if(param.B[0]->Nspin() != 1) tmp_cpu_field = curr_nullvec;
        else
        {
          csParam.create = QUDA_REFERENCE_FIELD_CREATE;
          csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
          csParam.v      = curr_nullvec->V();//take a pointer
          tmp_cpu_field  = new cpuColorSpinorField(csParam);
          csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
        }

	csParam.create = QUDA_ZERO_FIELD_CREATE;
	ColorSpinorField *b = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
	//copy fields:
	csParam.create = QUDA_COPY_FIELD_CREATE;
	ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(*tmp_cpu_field, csParam));

	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Initial guess = %g\n", norm2(*x));
#if 1 //for debug only!
        Solver *solve = nullptr;
        ColorSpinorField *out=nullptr, *in=nullptr;
        if(param.matSmooth.isStaggered())
        {
          Dirac &dirac_tmp = const_cast<Dirac&>(dirac);
          double init_mass = dirac_tmp.Mass();
          dirac_tmp.setMass((0.0*init_mass));//0.001=>0.0001
	  solve = Solver::create(solverParam, param.matSmooth, param.matSmooth, param.matSmooth, profile);

          dirac.prepare(in, out, *x, *b, QUDA_MAT_SOLUTION);
          (*solve)(*out, *in);
          dirac.reconstruct(*x, *b, QUDA_MAT_SOLUTION);
/*
          if(param.mg_global._2d_u1_emulation)
          {
            *b = *tmp_cpu_field;
            xpy(*b, *x);
          }
*/
          dirac_tmp.setMass(init_mass);
        }
        else
        {
	  solve = Solver::create(solverParam, param.matSmooth, param.matSmoothSloppy, param.matSmoothSloppy, profile);
	
          dirac.prepare(in, out, *x, *b, QUDA_MAT_SOLUTION);
          (*solve)(*out, *in);
          dirac.reconstruct(*x, *b, QUDA_MAT_SOLUTION);
        }
	delete solve;
#endif
	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Solution norm squared = %g\n", norm2(*x));

	*tmp_cpu_field = *x;

        if (getVerbosity() >= QUDA_VERBOSE)
        {
        //
          csParam.create = QUDA_ZERO_FIELD_CREATE;

  	  ColorSpinorField *t = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
          double nrm_1 = blas::norm2(*x);
          printfQuda("Solution = %g\n", nrm_1);

          //check null vector quality:
          param.matResidual(*b, *x, *t);
          double nrm_2 = blas::norm2(*b);
          printfQuda("Null vector check = %g\n", nrm_2 / nrm_1);

          delete t;
        }

	// global orthoonormalization of the generated null-space vectors
	for (std::vector<ColorSpinorField*>::iterator prevvec = B.begin(); prevvec != nullvec; ++prevvec)//row id
	  {
	    cpuColorSpinorField *prev_nullvec = static_cast<cpuColorSpinorField*> (*prevvec);

	    Complex alpha = cDotProduct(*prev_nullvec, *curr_nullvec);//<j,i>
	    caxpy(-alpha, *prev_nullvec, *curr_nullvec); //i-<j,i>j
	  }

	double nrm2 = norm2(*curr_nullvec);
	if (nrm2 > 1e-16) ax(1.0 /sqrt(nrm2), *curr_nullvec);
	else errorQuda("\nCannot orthogonalize %ld vector\n", nullvec-B.begin());

	delete b;
	delete x;

        if(param.B[0]->Nspin() == 1) delete tmp_cpu_field;

      }//stop for-loop:

    saveVectors(B);

    return;
  }

  const bool use_mixed_precision = false;

  void MG::computeLowModeVectors(std::vector<ColorSpinorField*> B) {
    printfQuda("\nCompute low mode vectors\n");
    //Create spinor field parameters:
    const Dirac &dirac  = *(param.matEigen->Expose());
    //const Dirac &dirac  = *(param.matSmooth.Expose());//if use_mixed_precision 

    SolverParam solverParam(param);

    // set low-mode-space generation options - need to expose these
    if(!(param.matSmooth.isStaggered() && param.level == 0 /*&& param.smoother_solve_type == QUDA_NORMOP_PC_SOLVE*/)) errorQuda("\nEigensolver is cuurently not implemented for this operator\n");
    //set incremetal (solo-precision) eigcg with 12 RHS:
    const int rhs_num = 24;//48, 24 for real fld, 80 could be fine (128)
    const int Nvec = B.size();

    solverParam.inv_type = QUDA_INC_EIGCG_INVERTER;
    //general setup:
    solverParam.maxiter = 30000;
    solverParam.tol = 1e-16;//1e-14
    solverParam.use_init_guess = QUDA_USE_INIT_GUESS_NO;
    solverParam.delta = 1e-7;

    solverParam.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
    solverParam.compute_null_vector = QUDA_COMPUTE_LOW_MODE_VECTOR;
    solverParam.precision = param.mg_global.eigensolver_precision;
    //solverParam.precision = solverParam.precision_sloppy;
    solverParam.precision_sloppy = (param.mg_global.eigensolver_precision != solverParam.precision_sloppy && use_mixed_precision) ? solverParam.precision_sloppy : param.mg_global.eigensolver_precision;

    //eigcg specific setup:
    solverParam.nev = 8; 
    solverParam.m   = 128;//144
    solverParam.deflation_grid = rhs_num;//to generate 48 eigenvectors but we will compute 96 vectors
    solverParam.precision_ritz = (param.mg_global.eigensolver_precision != solverParam.precision_sloppy && use_mixed_precision) ? solverParam.precision_sloppy : param.mg_global.eigensolver_precision;// B[0]->Precision(); 
    solverParam.tol_restart = 5e+3*solverParam.tol;//think about this...
    solverParam.use_reduced_vector_set = true;
    solverParam.use_cg_updates = false;
    solverParam.cg_iterref_tol = 5e-2;
    solverParam.eigcg_max_restarts = 1;//3
    solverParam.max_restart_num = 3;
    solverParam.inc_tol = 1e-2;
    solverParam.eigenval_tol = 1e+1;//eigenvalue selection (default 1e-2) 
    solverParam.rhs_idx = 0;

    if( (solverParam.deflation_grid*solverParam.nev) < Nvec) errorQuda("\nIncorrect eigCG parameters.\n");

    ColorSpinorParam csParam(*B[0]);
    // to force setting the field to be native first set to double-precision native order
    // then use the setPrecision method to set to native order
    csParam.gammaBasis = param.B[0]->GammaBasis();//hack for staggered
    csParam.precision = QUDA_DOUBLE_PRECISION;
    if(solverParam.precision != QUDA_DOUBLE_PRECISION)
    {
      csParam.setPrecision(B[0]->Precision());//ok
    }
    else
    {
      printfQuda("\nRunning full precision eigensolver.\n");
    }

    printfQuda("\nPrecisions: (%d, %d, %d)\n", solverParam.precision, solverParam.precision_sloppy, solverParam.precision_ritz);
      
    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.create = QUDA_NULL_FIELD_CREATE;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    cpuColorSpinorField *v1 = static_cast<cpuColorSpinorField*>(ColorSpinorField::Create(csParam));

    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.location   = QUDA_CUDA_FIELD_LOCATION; // hard code to GPU location for null-space generation for now

    csParam.create = QUDA_ZERO_FIELD_CREATE;

//    std::vector<ColorSpinorField*>::iterator ref = B.begin();//reference field
//    cpuColorSpinorField *cpu_ref     = static_cast<cpuColorSpinorField*> (*ref);
    cpuColorSpinorField &cpu_even_ref = static_cast<cpuColorSpinorField&>(v1->Even());
    cpuColorSpinorField &cpu_odd_ref = static_cast<cpuColorSpinorField&>(v1->Odd());

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;

    //Create eigensolver:
    //
    Dirac &dirac_tmp = const_cast<Dirac&>(dirac);
    double init_mass = dirac_tmp.Mass();
    dirac_tmp.setMass((1.0*init_mass));//invert on zero mass, 0.001, 1.0 for l32t48

    //DeflatedSolver *solve = DeflatedSolver::create(solverParam, &param.matSmooth, &param.matSmooth, &param.matSmooth, &param.matSmooth, &profile);
    DeflatedSolver *solve = (param.mg_global.eigensolver_precision != solverParam.precision_sloppy && use_mixed_precision) ? DeflatedSolver::create(solverParam, param.matEigen, &param.matSmooth, &param.matSmooth, &param.matSmooth, &profile) : DeflatedSolver::create(solverParam, param.matEigen, param.matEigen, param.matEigen, param.matEigen, &profile);

    for(int rhs = 0; rhs < solverParam.deflation_grid; rhs++ )
    {
       if(!param.mg_global._2d_u1_emulation)
       {
         v1->Source(QUDA_RANDOM_SOURCE);//random initial guess
       }
       else
       {
         blas::zero(*v1);//need this
         generic2DSource(*v1);
       }

       csParam.create      = QUDA_COPY_FIELD_CREATE;
       ColorSpinorField *b = static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpu_even_ref, csParam));
       //
       csParam.create = QUDA_ZERO_FIELD_CREATE;
       ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
       //
       (*solve)(static_cast<cudaColorSpinorField*>(x), static_cast<cudaColorSpinorField*>(b));//launch eigCG on parity fields
       //
       printfQuda("\nWorked with RHS %d\n", rhs);
       delete x;
       delete b;
    }
    //save ritz vectors:
    solve->ExtractRitzVecs(B, Nvec, true);
    printfQuda("\nDelete incremental EigCG solver resources...\n");
    //clean resources:
    solve->CleanResources();
    //
    delete solve;

    bool solve_other_parity = false;//true

    if(solve_other_parity)
    {
       solverParam.rhs_idx = 0;

       QudaMatPCType curr_matpctype  = dirac_tmp.getMatPCType();
       QudaMatPCType other_matpctype = curr_matpctype == QUDA_MATPC_EVEN_EVEN ? QUDA_MATPC_ODD_ODD : QUDA_MATPC_EVEN_EVEN ;

       dirac_tmp.setMatPCType(other_matpctype);

       DeflatedSolver *solve = DeflatedSolver::create(solverParam, param.matEigen, param.matEigen, param.matEigen, param.matEigen, &profile);

       for(int rhs = 0; rhs < solverParam.deflation_grid; rhs++ )
       {
         if(!param.mg_global._2d_u1_emulation)
         {
           v1->Source(QUDA_RANDOM_SOURCE);//random initial guess
         }
         else
         {
           blas::zero(*v1);//need this
           generic2DSource(*v1);
         }
         csParam.create      = QUDA_COPY_FIELD_CREATE;
         ColorSpinorField *b = static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpu_odd_ref, csParam));

         csParam.create = QUDA_ZERO_FIELD_CREATE;
         ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

         (*solve)(static_cast<cudaColorSpinorField*>(x), static_cast<cudaColorSpinorField*>(b));

         delete x;
         delete b;
       }

       csParam.create = QUDA_ZERO_FIELD_CREATE;
       csParam.setPrecision(B[0]->Precision());//just to be safe   

       csParam.x[0] *= 2;
       csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
       //full field
       ColorSpinorField *t = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

       int id = 0;//current Ritz vector index

       for(std::vector<ColorSpinorField*>::iterator vec = B.begin() ; vec != B.end() ; ++vec){       

          cpuColorSpinorField *curr_vec = static_cast<cpuColorSpinorField*> (*vec);
          copy(*t, *curr_vec);

          if( other_matpctype == QUDA_MATPC_ODD_ODD )
          {
             solve->ExtractSingleRitzVec(&t->Odd(), id, QUDA_ODD_PARITY);
          }
          else
          {
             solve->ExtractSingleRitzVec(&t->Even(), id, QUDA_EVEN_PARITY);
          }
          copy(*curr_vec, *t);
          id += 1;
       }

       solve->CleanResources();

       delete t;
       delete solve;

       dirac_tmp.setMatPCType(curr_matpctype);
       dirac_tmp.setMass(init_mass);
    }
    else
    {
       dirac_tmp.setMass(init_mass);

       //create auxiliary gpu fields:
       csParam.create = QUDA_ZERO_FIELD_CREATE;
       //csParam.setPrecision(B[0]->Precision());//just to be safe   

       csParam.setPrecision(QUDA_DOUBLE_PRECISION);

       csParam.x[0] *= 2;
       csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
       //full field
       ColorSpinorField *t = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

       QudaMatPCType curr_matpctype  = dirac.getMatPCType();

       double _mass = dirac.Mass();

       printfQuda("\nReconstruct parity components with mass %le\n", _mass);

       for(std::vector<ColorSpinorField*>::iterator vec = B.begin() ; vec != B.end() ; ++vec){
         //restore the other parity
         cpuColorSpinorField *curr_vec = static_cast<cpuColorSpinorField*> (*vec);

         copy(*t, *curr_vec);

         if(curr_matpctype == QUDA_MATPC_EVEN_EVEN)
         {
            dirac.Dslash(t->Odd(), t->Even(), QUDA_ODD_PARITY);
            ax(+ 1.0 / (2*_mass), t->Odd());//hopping term has opposite sign
         }
         else
         {
            dirac.Dslash(t->Even(), t->Odd(), QUDA_EVEN_PARITY);
            ax(+ 1.0 / (2*_mass), t->Even());//hopping term has opposite sign
         }

         copy(*curr_vec, *t);

         for (std::vector<ColorSpinorField*>::iterator prevvec = B.begin(); prevvec != vec; ++prevvec)
         {
            cpuColorSpinorField *prev_vec = static_cast<cpuColorSpinorField*> (*prevvec);

            Complex alpha = cDotProduct(*prev_vec, *curr_vec);
            caxpy(-alpha, *prev_vec, *curr_vec); 
         }

         double nrm2 = norm2(*curr_vec);
         if (nrm2 > 1e-16) ax(1.0 /sqrt(nrm2), *curr_vec);
         else errorQuda("\nCannot orthogonalize %ld vector\n", vec-B.begin());


         if (getVerbosity() >= QUDA_VERBOSE)
         {
            copy(*t, *curr_vec);
            double nrm_1   = blas::norm2(*t);
            double nrm_1_e = blas::norm2(t->Even());
            double nrm_1_o = blas::norm2(t->Odd());
            printfQuda("Solution = %g (%g, %g) \n", nrm_1, nrm_1_e, nrm_1_o);

            csParam.create = QUDA_ZERO_FIELD_CREATE;

            csParam.setPrecision(B[0]->Precision());
            ColorSpinorField *t2 = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
            ColorSpinorField *t3 = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));

            copy(*t3, *t);

            param.matResidual(*t2, *t3);
            if(param.matResidual.isStaggered())
            {
              double nrm_2 = blas::norm2(*t2);
              blas::ax(2.0*dirac.Mass(), *t3);
              double nrm_3 = xmyNorm(*t2, *t3);
              printfQuda("NULL CHECK = %g (nrm_3 = %g)\n", nrm_2 / nrm_1, sqrt(nrm_3));
            }

            delete t3;
            delete t2;
         }//end QUDA_VERBOSE
       }//stop for-loop:

       delete t;
    }

    printfQuda("\n...done.\n");

    delete v1;
    //set original mass:
    //dirac_tmp.setMass(init_mass);

    saveVectors(B);

    return;
  }

}
