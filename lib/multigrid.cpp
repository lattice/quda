#include <multigrid.h>
#include <qio_field.h>
#include <string.h>

namespace quda {
  
  // FIXME - do basis check

  MG::MG(MGParam &param, TimeProfile &profile) 
    : Solver(param, profile), param(param), presmoother(0), postsmoother(0), coarse(0), fine(param.fine), 
      param_coarse(0), param_presmooth(0), param_postsmooth(0), r(0), r_coarse(0), matCoarse(0), 
      hack1(0), hack2(0), hack3(0), hack4(0) {

    printfQuda("MG: Creating level %d of %d levels\n", param.level, param.Nlevel);

    if (param.level > QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level);

    // create the smoother for this level
    std::cout << "MG: level " << param.level << " smoother has operator " << typeid(param.matSmooth).name() << std::endl;

    param_presmooth = new MGParam(param, param.B, param.matResidual, param.matSmooth);

    param_presmooth->inv_type = param.smoother;
    if (param_presmooth->level == 1) param_presmooth->inv_type_precondition = QUDA_GCR_INVERTER;
    param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_YES;
    param_presmooth->use_init_guess = QUDA_USE_INIT_GUESS_NO;
    param_presmooth->maxiter = param.nu_pre;
    param_presmooth->Nkrylov = 4;
    param_presmooth->inv_type_precondition = QUDA_INVALID_INVERTER;
    if (param.level==2) {
      param_presmooth->maxiter = 1000;
      param_presmooth->tol = 1e-10;
      param_presmooth->preserve_source = QUDA_PRESERVE_SOURCE_NO;
    }
    presmoother = Solver::create(*param_presmooth, param_presmooth->matResidual,
				 param_presmooth->matSmooth, param_presmooth->matSmooth, profile);

    if (param.level < param.Nlevel) {

      //Create the post smoother
      param_postsmooth = new MGParam(param, param.B, param.matResidual, param.matSmooth);
      
      param_postsmooth->inv_type = param.smoother;
      if (param_postsmooth->level == 1) param_postsmooth->inv_type_precondition = QUDA_GCR_INVERTER;
      param_postsmooth->preserve_source = QUDA_PRESERVE_SOURCE_YES;
      param_postsmooth->use_init_guess = QUDA_USE_INIT_GUESS_YES;
      param_postsmooth->maxiter = param.nu_post;
      param_postsmooth->Nkrylov = 4;
      param_postsmooth->inv_type_precondition = QUDA_INVALID_INVERTER;
      postsmoother = Solver::create(*param_postsmooth, param_postsmooth->matResidual, 
				    param_postsmooth->matSmooth, param_postsmooth->matSmooth, profile);
    }

    // if not on the coarsest level, construct it
    if (param.level < param.Nlevel) {
      // create transfer operator
      printfQuda("MG: start creating transfer operator\n");
      transfer = new Transfer(param.B, param.Nvec, param.geoBlockSize, param.spinBlockSize);
      printfQuda("MG: end creating transfer operator\n");

      // create residual vectors
      {
	ColorSpinorParam csParam(*(param.B[0]));
	csParam.create = QUDA_NULL_FIELD_CREATE;
	if (param.level==1) {
	  csParam.fieldOrder = (csParam.precision == QUDA_DOUBLE_PRECISION) ? QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
	  csParam.setPrecision(csParam.precision);
	  csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
	  r = new cudaColorSpinorField(csParam);
	} else {
	  r = new cpuColorSpinorField(csParam);	
	}
      }

      // create coarse residual vector
      r_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec);

      // create coarse solution vector
      x_coarse = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec);

      // create coarse grid operator
      // these need to be cpu fields
      ColorSpinorParam csParam(*(param.B[0]));
      csParam.create = QUDA_NULL_FIELD_CREATE;
      hack1 = new cpuColorSpinorField(csParam);
      hack2 = new cpuColorSpinorField(csParam);

      // these need to be gpu fields with native ordering basis
      csParam.fieldOrder = (csParam.precision == QUDA_DOUBLE_PRECISION) ? QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
      csParam.setPrecision(csParam.precision);
      csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      hack3 = new cudaColorSpinorField(csParam);  // FIXME allocate cudaSpinorFields
      hack4 = new cudaColorSpinorField(csParam);   // FIXME allocate cudaSpinorFields

      csParam.create = QUDA_ZERO_FIELD_CREATE;
      y = new cudaColorSpinorField(csParam);

      // note last two fields are cpu fields!
      DiracCoarse *matCoarse = new DiracCoarse(param.matResidual.Expose(), transfer, *hack1, *hack2, *hack3, *hack4);
      std::cout << "MG: level " << param.level << " creating coarse operator of type " << typeid(matCoarse).name() << std::endl;

      // coarse null space vectors (dummy for now)
      std::vector<ColorSpinorField*> B_coarse;

      // create the next multigrid level
      param_coarse = new MGParam(param, B_coarse, *matCoarse, *matCoarse);
      param_coarse->level++;

      param_coarse->fine = this;
      param_coarse->smoother = QUDA_BICGSTAB_INVERTER;
      param_coarse->delta = 1e-1;

      coarse = new MG(*param_coarse, profile);
    }

    printfQuda("MG: Setup of level %d completed\n", param.level);

    // now we can run through the verificaion
    //if (param.level == 1) verify();

  }

  MG::~MG() {
    if (param.level < param.Nlevel) {
      if (coarse) delete coarse;
      if (transfer) delete transfer;
    }
    if (presmoother) delete presmoother;
    if (postsmoother) delete postsmoother;

    if (r) delete r;
    if (r_coarse) delete r_coarse;
    if (x_coarse) delete x_coarse;

    if (param_coarse) delete param_coarse;
    if (param_presmooth) delete param_presmooth;
    if (param_postsmooth) delete param_postsmooth;

    if (matCoarse) delete matCoarse;

    if (hack1) delete hack1;
    if (hack2) delete hack2;
    if (hack3) delete hack3;
    if (hack4) delete hack4;
    if (y) delete y;
  }

  /**
     Verification that the constructed multigrid operator is valid
   */
  void MG::verify() {

    printfQuda("\nChecking 0 = (1 - P^\\dagger P) v_k for %d vectors\n", param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      *hack3 = *(param.B[i]);
      transfer->R(*r_coarse, *hack3);
      transfer->P(*hack4, *r_coarse);
      *hack2 = *hack4;

      /*for (int k=0; k<r_coarse->Volume(); k++) {
	static_cast<cpuColorSpinorField*>(param.B[i])->PrintVector(k);
	static_cast<cpuColorSpinorField*>(hack2)->PrintVector(k);
	printf("\n");
	}*/

      printfQuda("Vector %d: norms %e %e ", i, blas::norm2(*param.B[i]), blas::norm2(*hack2));
      printfQuda("deviation = %e\n", blas::xmyNorm(*(param.B[i]), *hack2));
    }

    /* printfQuda("\nChecking 1 > || (1 - D P (P^\\dagger D P) P^\\dagger v_k || / || v_k || for %d vectors\n", param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      transfer->R(*r_coarse, *(param.B[i]));
      (*coarse)(*x_coarse, *r_coarse);
      transfer->P(*hack2, *x_coarse);
      *hack3 = *hack2;
      param.matResidual(*hack4,*hack3);
      *hack1 = *hack4;

      printfQuda("Vector %d: norms %e %e ", i, blas::norm2(*param.B[i]), blas::norm2(*hack1));
      printfQuda("relative residual = %e\n", sqrt(blas::xmyNorm(*(param.B[i]), *hack1) / blas::norm2(*param.B[i])) );
    }*/

    printfQuda("\nChecking 0 = (1 - P^\\dagger P) eta_c \n");
    static_cast<cpuColorSpinorField*>(x_coarse)->Source(QUDA_RANDOM_SOURCE);
    transfer->P(*hack2, *x_coarse);
    transfer->R(*r_coarse, *hack2);
    printfQuda("Vector norms %e %e ", blas::norm2(*x_coarse), blas::norm2(*r_coarse));
    printfQuda("deviation = %e\n", blas::xmyNorm(*x_coarse, *r_coarse));

    printfQuda("\nChecking sinusoidal waves\n");
    int dim = 3;
    //for (int n=1; n<hack2->X(dim); n++) {
    for (int n=hack2->X(dim)-1; n<hack2->X(dim); n++) {
      ColorSpinorParam csParam(*hack2);
      csParam.create = QUDA_ZERO_FIELD_CREATE;
      ColorSpinorField *tmp = ColorSpinorField::Create(csParam);
      ColorSpinorField *tmp2 = ColorSpinorField::Create(csParam);

      static_cast<cpuColorSpinorField*>(tmp)->Source(QUDA_SINUSOIDAL_SOURCE, dim, n);
      
      transfer->R(*r_coarse, *tmp);
      transfer->P(*tmp2, *r_coarse);

      /*for (int k=0; k<r_coarse->Volume(); k++) {
	static_cast<cpuColorSpinorField*>(hack2)->PrintVector(k);
	static_cast<cpuColorSpinorField*>(hack1)->PrintVector(k);
	printf("\n");
	}*/

      double deviation = sqrt(blas::xmyNorm(*tmp, *tmp2) / blas::norm2(*tmp));

      blas::zero(*x_coarse);
      transfer->R(*r_coarse, *tmp);
      (*coarse)(*x_coarse, *r_coarse); setVerbosity(QUDA_VERBOSE);
      transfer->P(*tmp2, *x_coarse);
      param.matResidual(*tmp2,*tmp2);

      printfQuda("Dimension %d, mode %d: relative dev = %e, relative residual = %e\n", 
		 dim, n, deviation, sqrt(blas::xmyNorm(*tmp, *tmp2) / blas::norm2(*tmp)) );

      *hack3 = *tmp;
      blas::zero(*hack4);
      operator()(*hack4, *hack3);
      param.matResidual(*tmp2,*hack4);

      printfQuda("Dimension %d, mode %d: relative dev = %e, relative residual = %e\n", 
		 dim, n, deviation, sqrt(blas::xmyNorm(*tmp, *tmp2) / blas::norm2(*tmp)) );
      
    }


    exit(0);

  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    if (getVerbosity() >= QUDA_VERBOSE) {
      printfQuda("MG: level %d, x.order = %d b.order = %d\n", param.level, x.FieldOrder(), b.FieldOrder());
      
      printfQuda("MG: level %d, entering V-cycle with x2=%e, r2=%e\n", 
		 param.level, blas::norm2(x), blas::norm2(b));
    }

    if (param.level < param.Nlevel) {
      
      // do the pre smoothing
      printfQuda("MG: level %d, pre smoothing b2=%e\n", param.level, blas::norm2(b));
      //param.use_init_guess = QUDA_USE_INIT_GUESS_NO;
      //param.maxiter = param.nu_pre;
      
      (*presmoother)(x, b);

      // FIXME - residual computation should be in the previous smoother
      param.matResidual(*r, x);

      double r2 = blas::xmyNorm(b, *r);
      
      // restrict to the coarse grid
      printfQuda("MG: level %d, restriction\n", param.level);

      transfer->R(*r_coarse, *r);

      //*hack1 = *r;
      //for (int i=0; i<hack1->Volume(); i++) static_cast<cpuColorSpinorField*>(hack1)->PrintVector(i);
      //for (int i=0; i<r_coarse->Volume(); i++) static_cast<cpuColorSpinorField*>(r_coarse)->PrintVector(i);

      printfQuda("MG: after pre-smoothing r2 %e r_coarse2 = %e\n", r2, blas::norm2(*r_coarse));

      // recurse to the next lower level
      printfQuda("MG: level %d solving coarse operator\n", param.level);
      blas::zero(*x_coarse);
      (*coarse)(*x_coarse, *r_coarse);
      //blas::zero(*x_coarse);      
      printfQuda("MG: after coarse solve x_coarse2 %e r_coarse2 = %e\n", 
		 blas::norm2(*x_coarse), blas::norm2(*r_coarse)); 

      // prolongate back to this grid
      printfQuda("MG: level %d, prolongation\n", param.level);
      transfer->P(*r, *x_coarse); // repurpose residual storage
      printfQuda("MG: Prolongated coarse solution y2 = %e\n", blas::norm2(*r)); 
      printfQuda("MG: Old fine solution x2 = %e\n", blas::norm2(x));
      blas::xpy(*r, x); // sum to solution
      printfQuda("MG: New fine solution x2 = %e\n", blas::norm2(x));       
      // FIXME - residual computation should be in the previous smoother
      param.matResidual(*r, x);
      r2 = blas::xmyNorm(b, *r);
      printfQuda("MG: coarse-grid corrected  r2 = %e x_coarse2 = %e\n", r2, blas::norm2(*x_coarse));

      // do the post smoothing
      printfQuda("MG: level %d, post smoothing\n", param.level);
      //param.maxiter = param.nu_post;
      //param.use_init_guess = QUDA_USE_INIT_GUESS_NO;
      
      printfQuda("MG: norm check x2 = %e r2 = %e\n", blas::norm2(x),blas::norm2(*r));
      if(param_postsmooth->use_init_guess == QUDA_USE_INIT_GUESS_NO) {
        blas::copy(*y,x);
        (*postsmoother)(x, *r);
        blas::xpy(*y, x);
      } else {
	(*postsmoother)(x,b);
      }
      printfQuda("MG: Post smoothing fine solution x2 = %e\n", blas::norm2(x));
      param.matResidual(*r, x);
      r2 = blas::xmyNorm(b, *r);

      printfQuda("MG: level %d, leaving V-cycle with x2=%e, r2=%e\n", 
		 param.level, blas::norm2(x), blas::norm2(*r));

      //exit(0);
    } else { // do the coarse grid solve
      if (getVerbosity() >= QUDA_VERBOSE) 
	printfQuda("MG: level %d starting coarsest solve\n", param.level);
      //param.maxiter = 10;

      setVerbosity(QUDA_VERBOSE); 
      (*presmoother)(x, b);
      setVerbosity(QUDA_VERBOSE); 

      if (getVerbosity() >= QUDA_VERBOSE) 
	printfQuda("MG: level %d finished coarsest solve\n", param.level);
    }

  }

  //supports seperate reading or single file read
  void loadVectors(std::vector<ColorSpinorField*> &B) {
    printfQuda("Start loading %d vectors from %s\n", nvec, vecfile);

    if (nvec < 1 || nvec > 50) errorQuda("nvec not set");

    const int Nvec = nvec;

    void **V = new void*[Nvec];
    for (int i=0; i<Nvec; i++) { 
      V[i] = B[i]->V();
      if (V[i] == NULL) {
	printf("Could not allocate V[%d]\n", i);      
      }
    }
    
    if (strcmp(vecfile,"")!=0) {
#if 0
      read_spinor_field(vecfile, &V[0], B[0]->Precision(), B[0]->X(), 
			B[0]->Ncolor(), B[0]->Nspin(), Nvec, 0,  (char**)0);
#else 
      for (int i=0; i<Nvec; i++) {
	char filename[256];
	sprintf(filename, "%s.%d", vecfile, i);
	printf("Reading vector %d from file %s\n", i, filename);
	read_spinor_field(filename, &V[i], B[i]->Precision(), B[i]->X(), 
			  B[i]->Ncolor(), B[i]->Nspin(), 1, 0,  (char**)0);
      }
#endif
    } else {
      printfQuda("Using %d constant nullvectors\n", Nvec);
      //errorQuda("No nullspace file defined");
      for (int i = 0; i < 2; i++) {
	if(B[i]->Precision() == QUDA_SINGLE_PRECISION) {
	  printfQuda("Single precision\n");
	  for(int x = 0; x<B[i]->Volume(); x++) {
	    int s = i; {//for (int s=0; s<B[i]->Nspin(); s++) { 
	      for (int c=0; c<B[i]->Ncolor(); c++) {
		static_cast<float *>(V[i])[((x*4+s)*B[i]->Ncolor()+c)*2+0] = 1.0;
		static_cast<float *>(V[i])[((x*4+s+2)*B[i]->Ncolor()+c)*2+0] = 1.0;
	      }
	    }
	  }
	} else if(B[i]->Precision() == QUDA_DOUBLE_PRECISION) {
          printfQuda("Double precision\n");
	  for(int x = 0; x<B[i]->Volume(); x++) {
	    int s = i; {//for (int s=0; s<B[i]->Nspin(); s++) { 
	      for (int c=0; c<B[i]->Ncolor(); c++) {
		static_cast<double *>(V[i])[((x*4+s)*B[i]->Ncolor()+c)*2+0] = 1.0;
		static_cast<double *>(V[i])[((x*4+s+2)*B[i]->Ncolor()+c)*2+0] = 1.0;
	      }
	    }
	  }
	} else {
	  errorQuda("Constant null vectors not supported for precision = %d\n", B[i]->Precision());
	}
      }

      for (int i = 2; i < Nvec; i++) {
	if(B[i]->Precision() == QUDA_SINGLE_PRECISION) {
	  printfQuda("Single precision\n");
	  for(int x = 0; x<B[i]->Volume(); x++) {
	    for (int s=0; s<B[i]->Nspin(); s++) { 
	      for (int c=0; c<B[i]->Ncolor(); c++) {
		static_cast<float *>(V[i])[((x*4+s)*B[i]->Ncolor()+c)*2+0] = (float)rand()/RAND_MAX;
	      }
	    }
	  }
	} else if(B[i]->Precision() == QUDA_DOUBLE_PRECISION) {
          printfQuda("Double precision\n");
	  for(int x = 0; x<B[i]->Volume(); x++) {
	    for (int s=0; s<B[i]->Nspin(); s++) { 
	      for (int c=0; c<B[i]->Ncolor(); c++) {
		static_cast<double *>(V[i])[((x*4+s)*B[i]->Ncolor()+c)*2+0] = (double)rand()/RAND_MAX;
	      }
	    }
	  }
	} else {
	  errorQuda("Constant null vectors not supported for precision = %d\n", B[i]->Precision());
	}
      }

    }

    printfQuda("Done loading vectors\n");

  }

}
