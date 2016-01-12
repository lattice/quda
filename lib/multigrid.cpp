#include <multigrid.h>
#include <qio_field.h>
#include <string.h>

namespace quda {  

  using namespace blas;

  static bool debug = false;

  MG::MG(MGParam &param, TimeProfile &profile_global) 
    : Solver(param, profile), param(param), transfer(0), presmoother(0), postsmoother(0),
      profile_global(profile_global),
      profile( "MG level " + std::to_string(param.level+1), false ),
      coarse(0), fine(param.fine), param_coarse(0), param_presmooth(0), param_postsmooth(0), r(0), r_coarse(0), x_coarse(0), 
      diracCoarseResidual(0), diracCoarseSmoother(0), matCoarseResidual(0), matCoarseSmoother(0) {

    // for reporting level 1 is the fine level but internally use level 0 for indexing
    sprintf(prefix,"MG level %d (%s): ", param.level+1, param.location == QUDA_CUDA_FIELD_LOCATION ? "GPU" : "CPU" );
    setOutputPrefix(prefix);

    printfQuda("Creating level %d of %d levels\n", param.level+1, param.Nlevel);

    if (param.level == 0) { // null space generation only on level 1 currently
      if (param.mg_global.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES) {
	generateNullVectors(param.B);
      } else {
	loadVectors(param.B);
      }
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
    param_presmooth->Nkrylov = 4;
    if (param.level==param.Nlevel-1) {
      param_presmooth->Nkrylov = 20;
      param_presmooth->maxiter = 1000;
      param_presmooth->tol = 2e-1;
      param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_NO;
      param_presmooth->delta = 1e-2;
      param_presmooth->compute_true_res = false;
    }

    presmoother = Solver::create(*param_presmooth, param.matSmooth,
				 param.matSmooth, param.matSmooth, profile);

    if (param.level < param.Nlevel-1) { //Create the post smoother
      param_postsmooth = new SolverParam(*param_presmooth);
      param_postsmooth->use_init_guess = QUDA_USE_INIT_GUESS_YES;
      param_postsmooth->maxiter = param.nu_post;
      postsmoother = Solver::create(*param_postsmooth, param.matSmooth,
				    param.matSmooth, param.matSmooth, profile);
    }

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
    }

    // if not on the coarsest level, construct it
    if (param.level < param.Nlevel-1) {
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

      // create coarse grid operator

      DiracParam diracParam;
      diracParam.transfer = transfer;
      diracParam.dirac = const_cast<Dirac*>(param.matResidual.Expose());
      diracParam.kappa = param.matResidual.Expose()->Kappa();
      diracParam.dagger = QUDA_DAG_NO;
      diracParam.matpcType = QUDA_MATPC_EVEN_EVEN; // FIXME should probably expose this
      printfQuda("Kappa = %e\n", diracParam.kappa);
      // use even-odd preconditioning for the coarse grid solver
      diracCoarseResidual = new DiracCoarse(diracParam);
      matCoarseResidual = new DiracM(*diracCoarseResidual);

      // create smoothing operators
      diracParam.dirac = const_cast<Dirac*>(param.matSmooth.Expose());
      diracCoarseSmoother = (param.mg_global.smoother_solve_type[param.level+1] == QUDA_DIRECT_PC_SOLVE) ?
	new DiracCoarsePC(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam) :
	new DiracCoarse(static_cast<DiracCoarse&>(*diracCoarseResidual), diracParam);
      matCoarseSmoother = new DiracM(*diracCoarseSmoother);


      // coarse null space vectors (dummy for now)
      printfQuda("Creating coarse null-space vectors\n");
      B_coarse = new std::vector<ColorSpinorField*>();
      B_coarse->resize(param.Nvec);

      for (int i=0; i<param.Nvec; i++) {
	(*B_coarse)[i] = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec);
	zero(*(*B_coarse)[i]);
	transfer->R(*(*B_coarse)[i], *(param.B[i]));
#if 0
	if (param.level != 99) {
	  ColorSpinorParam csParam2(*(*B_coarse)[i]);
	  csParam2.create = QUDA_ZERO_FIELD_CREATE;
	  ColorSpinorField *tmp = ColorSpinorField::Create(csParam2);
	  for (int s=i; s<(*B_coarse)[i]->Nspin(); s+=2) {
	    for (int c=0; c<(*B_coarse)[i]->Ncolor(); c++) {
	      tmp->Source(QUDA_CONSTANT_SOURCE, 1, s, c);
	      //tmp->Source(QUDA_SINUSOIDAL_SOURCE, 3, 2);                                                                                       
	      //xpy(*tmp,*(*B_coarse)[i]);
	    }
	  }
	  delete tmp;
          (*B_coarse)[i]->Source(QUDA_RANDOM_SOURCE);                                                                                      
          printfQuda("B_coarse[%d]\n", i);
	}
#endif

      }

      // create the next multigrid level
      printfQuda("Creating next multigrid level\n");
      param_coarse = new MGParam(param, *B_coarse, *matCoarseResidual, *matCoarseSmoother, param.level+1);
      param_coarse->fine = this;
      param_coarse->delta = 1e-20;

      coarse = new MG(*param_coarse, profile_global);
      setOutputPrefix(prefix); // restore since we just popped back from coarse grid
    }

    printfQuda("setup completed\n");

    // now we can run through the verification if requested
    if (param.level == 0 && param.mg_global.run_verify) verify();

    setOutputPrefix("");
  }

  MG::~MG() {
    if (param.level < param.Nlevel-1) {
      if (B_coarse) for (int i=0; i<param.Nvec; i++) delete (*B_coarse)[i];
      if (coarse) delete coarse;
      if (transfer) delete transfer;
      if (matCoarseSmoother) delete matCoarseSmoother;
      if (diracCoarseSmoother) delete diracCoarseSmoother;
      if (matCoarseResidual) delete matCoarseResidual;
      if (diracCoarseResidual) delete diracCoarseResidual;
      if (postsmoother) delete postsmoother;
    }
    if (presmoother) delete presmoother;

    if (r) delete r;
    if (r_coarse) delete r_coarse;
    if (x_coarse) delete x_coarse;

    if (param_coarse) delete param_coarse;
    if (param_presmooth) delete param_presmooth;
    if (param_postsmooth) delete param_postsmooth;

    if (getVerbosity() >= QUDA_VERBOSE) profile.Print();
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
    param.matResidual(*tmp2,*tmp1);
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
    
    delete tmp1;
    delete tmp2;
    delete tmp_coarse;

    if (param.level < param.Nlevel-2) coarse->verify();
  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {
    setOutputPrefix(prefix);

    // if input vector is single parity then we must be solving the
    // preconditioned system in general this can only happen on the
    // top level
    QudaSolutionType outer_solution_type = b.SiteSubset() == QUDA_FULL_SITE_SUBSET ? QUDA_MAT_SOLUTION : QUDA_MATPC_SOLUTION;
    QudaSolutionType inner_solution_type = param.coarse_grid_solution_type;

    if ( outer_solution_type == QUDA_MATPC_SOLUTION && inner_solution_type == QUDA_MAT_SOLUTION)
      errorQuda("Unsupported solution type combination");

    if ( inner_solution_type == QUDA_MATPC_SOLUTION && param.smoother_solve_type != QUDA_DIRECT_PC_SOLVE)
      errorQuda("For this coarse grid solution type, a preconditioned smoother is required");

    if ( debug ) {
      printfQuda("entering V-cycle with x2=%e, r2=%e\n", norm2(x), norm2(b));
    }

    if (param.level < param.Nlevel-1) {
      //transfer->setTransferGPU(false); // use this to force location of transfer
      
      // do the pre smoothing
      if ( debug ) printfQuda("pre-smoothing b2=%e\n", norm2(b));

      ColorSpinorField *out=nullptr, *in=nullptr;

      ColorSpinorField &residual = outer_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even();
      residual = b; // copy source vector since we will overwrite source with iterated residual

      const Dirac &dirac = *(param.matSmooth.Expose());
      dirac.prepare(in, out, x, residual, outer_solution_type);
      (*presmoother)(*out, *in);

      ColorSpinorField &solution = inner_solution_type == outer_solution_type ? x : x.Even();
      dirac.reconstruct(solution, b, inner_solution_type);

      double r2 = 0.0;

      // if using preconditioned smoother then need to reconstruct full residual
      // FIXME extend this check for precision, etc.
      // also need to extend this when residual operator is itself preconditioned
      bool use_solver_residual =
	( (param.smoother_solve_type == QUDA_DIRECT_PC_SOLVE && inner_solution_type == QUDA_MATPC_SOLUTION) ||
	  (param.smoother_solve_type == QUDA_DIRECT_SOLVE && inner_solution_type == QUDA_MAT_SOLUTION) )
	? true : false;

      // FIXME this is currently borked if inner solver is preconditioned
      if (use_solver_residual) {
	if (debug) r2 = norm2(*r);
      } else {
	param.matResidual(*r, x);
	if (debug) r2 = xmyNorm(b, *r);
	else axpby(1.0, b, -1.0, *r);
      }

      // restrict to the coarse grid
      transfer->R(*r_coarse, residual);
      if ( debug ) {
	printfQuda("after pre-smoothing x2 = %e, r2 = %e, r_coarse2 = %e\n", 
		   norm2(x), r2, norm2(*r_coarse));
      }

      // recurse to the next lower level
      (*coarse)(*x_coarse, *r_coarse);

      setOutputPrefix(prefix); // restore prefix after return from coarse grid

      if ( debug ) {
	printfQuda("after coarse solve x_coarse2 = %e r_coarse2 = %e\n", 
		   norm2(*x_coarse), norm2(*r_coarse));
      }

      // prolongate back to this grid
      residual = inner_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even(); // define according to inner solution type
      transfer->P(residual, *x_coarse); // repurpose residual storage
      // FIXME - sum should be done inside the transfer operator
      xpy(*r, solution); // sum to solution

      if ( debug ) {
	printfQuda("Prolongated coarse solution y2 = %e\n", norm2(*r));
	printfQuda("after coarse-grid correction x2 = %e, r2 = %e\n", 
		   norm2(x), norm2(*r));
      }

      // do the post smoothing
      residual = outer_solution_type == QUDA_MAT_SOLUTION ? *r : r->Even(); // refine for outer solution type
      residual = b;

      dirac.prepare(in, out, solution, residual, inner_solution_type);
      (*postsmoother)(*out, *in);
      dirac.reconstruct(x, b, outer_solution_type);

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

    setOutputPrefix("");
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

    SolverParam solverParam(param);

    // set null-space generation options - need to expose these
    solverParam.maxiter = 500;
    solverParam.tol = 5e-5;
    solverParam.use_init_guess = QUDA_USE_INIT_GUESS_YES;
    solverParam.delta = 1e-7;
    solverParam.inv_type = QUDA_BICGSTAB_INVERTER;
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
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;

    // Generate sources and launch solver for each source:
    for(std::vector<ColorSpinorField*>::iterator nullvec = B.begin() ; nullvec != B.end(); ++nullvec) {
	cpuColorSpinorField *curr_nullvec = static_cast<cpuColorSpinorField*> (*nullvec);
	curr_nullvec->Source(QUDA_RANDOM_SOURCE);//random initial guess

	csParam.create = QUDA_ZERO_FIELD_CREATE;
	ColorSpinorField *b = static_cast<ColorSpinorField*>(new cudaColorSpinorField(csParam));
	//copy fields:
	csParam.create = QUDA_COPY_FIELD_CREATE;
	ColorSpinorField *x = static_cast<ColorSpinorField*>(new cudaColorSpinorField(*curr_nullvec, csParam));

	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Initial guess = %g\n", norm2(*x));

	Solver *solve = Solver::create(solverParam, param.matSmooth, param.matSmooth, param.matSmooth, profile);

	const Dirac &dirac = *(param.matSmooth.Expose());
	ColorSpinorField *out=nullptr, *in=nullptr;
	dirac.prepare(in, out, *x, *b, QUDA_MAT_SOLUTION);
	(*solve)(*out, *in);
	dirac.reconstruct(*x, *b, QUDA_MAT_SOLUTION);

	delete solve;

	if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Solution = %g\n", norm2(*x));

	*curr_nullvec = *x;

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

      }//stop for-loop:

    saveVectors(B);

    return;
  }

}
