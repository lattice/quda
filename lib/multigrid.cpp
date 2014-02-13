#include <multigrid.h>
#include <qio_field.h>
#include <string.h>

namespace quda {
  
  // FIXME - do basis check

  MG::MG(MGParam &param, TimeProfile &profile) 
    : Solver(param, profile), param(param), smoother(0), coarse(0), fine(param.fine), 
      param_coarse(0), r(0), r_coarse(0), matCoarse(0), hack1(0), hack2(0), hack3(0), hack4(0) {

    printfQuda("MG: Creating level %d of %d levels\n", param.level, param.Nlevel);

    if (param.level > QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level);

    // create the smoother for this level
    std::cout << "MG: level " << param.level << " smoother has operator " << typeid(param.matSmooth).name() << std::endl;
    param.inv_type = param.smoother;
    if (param.level == 1) param.inv_type_precondition = QUDA_GCR_INVERTER;
    param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
    //param.use_init_guess = QUDA_USE_INIT_GUESS_YES;
    smoother = Solver::create(param, param.matResidual, param.matSmooth, param.matSmooth, profile);

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
      MGParam *param_coarse = new MGParam(param, B_coarse, *matCoarse, *matCoarse);
      param_coarse->level++;

      param_coarse->fine = this;
      param_coarse->smoother = QUDA_BICGSTAB_INVERTER;

      coarse = new MG(*param_coarse, profile);
    }

    printfQuda("MG: Setup of level %d completed\n", param.level);
  }

  MG::~MG() {
    if (param.level < param.Nlevel) {
      if (coarse) delete coarse;
      if (transfer) delete transfer;
    }
    if (smoother) delete smoother;

    if (r) delete r;
    if (r_coarse) delete r_coarse;
    if (x_coarse) delete x_coarse;

    if (param_coarse) delete param_coarse;

    if (matCoarse) delete matCoarse;

    if (hack1) delete hack1;
    if (hack2) delete hack2;
    if (hack3) delete hack3;
    if (hack4) delete hack4;
    if (y) delete y;
  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    printfQuda("MG: level %d, x.order = %d b.order = %d\n", param.level, x.FieldOrder(), b.FieldOrder());

    printfQuda("MG: level %d, entering V-cycle with x2=%e, r2=%e\n", 
	       param.level, blas::norm2(x), blas::norm2(b));

    if (param.level < param.Nlevel) {
      
      // do the pre smoothing
      printfQuda("MG: level %d, pre smoothing\n", param.level);
      param.use_init_guess = QUDA_USE_INIT_GUESS_NO;
      param.maxiter = param.nu_pre;
      (*smoother)(x, b);

      // FIXME - residual computation should be in the previous smoother
      param.matResidual(*r, x);
      double r2 = blas::xmyNorm(b, *r);

      // restrict to the coarse grid
      printfQuda("MG: level %d, restriction\n", param.level);
      transfer->R(*r_coarse, *r);
      printfQuda("MG: after pre-smoothing r2 %e r_coarse2 = %e\n", r2, blas::norm2(*r_coarse));

      // recurse to the next lower level
      printfQuda("MG: level %d solving coarse operator\n", param.level);
      blas::zero(*x_coarse);
      (*coarse)(*x_coarse, *r_coarse);
      printfQuda("MG: after coarse solve x_coarse2 %e r_coarse2 = %e\n", blas::norm2(*x_coarse), blas::norm2(*r_coarse)); 

      // prolongate back to this grid
      printfQuda("MG: level %d, prolongation\n", param.level);
      transfer->P(*r, *x_coarse); // repurpose residual storage
      printfQuda("MG: Prolongated coarse solution r2 = %e\n", blas::norm2(*r)); 
      printfQuda("MG: Old fine solution x2 = %e\n", blas::norm2(x));
      blas::xpy(*r, x); // sum to solution
      printfQuda("MG: New fine solution x2 = %e\n", blas::norm2(x));       
      // FIXME - residual computation should be in the previous smoother
      param.matResidual(*r, x);
      r2 = blas::xmyNorm(b, *r);
      printfQuda("MG: coarse-grid corrected  r2 = %e x_coarse2 = %e\n", r2, blas::norm2(*x_coarse));

      // do the post smoothing
      printfQuda("MG: level %d, post smoothing\n", param.level);
      param.maxiter = param.nu_post;

      param.use_init_guess = QUDA_USE_INIT_GUESS_NO;

      printfQuda("MG: norm check x2 = %e r2 = %e\n", blas::norm2(x),blas::norm2(*r));
      if(param.use_init_guess == QUDA_USE_INIT_GUESS_NO) {
        //FIXME: MC - Use hack3 dummy field to store x because MR does not support initial guess
        blas::copy(*y,x);
        (*smoother)(x, *r);
        blas::xpy(*y, x);
      }
      else {
	(*smoother)(x,b);
      }
      printfQuda("MG: Post smoothing fine solution x2 = %e\n", blas::norm2(x));
      param.matResidual(*r, x);
      r2 = blas::xmyNorm(b, *r);

      printfQuda("MG: level %d, leaving V-cycle with x2=%e, r2=%e\n", 
		 param.level, blas::norm2(x), blas::norm2(*r));

    } else { // do the coarse grid solve

      printfQuda("MG: level %d starting coarsest solve\n", param.level);
      param.maxiter = 10;
      (*smoother)(x, b);
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
      for (int i = 0; i < Nvec; i++) {
	int length = B[i]->Length();
	if(B[i]->Precision() == QUDA_SINGLE_PRECISION) {
	  printfQuda("Single precision\n");
	  for(int index = 0; index < length; index+=2) {
	    static_cast<float *>(V[i])[index] = 1.0;
	    static_cast<float *>(V[i])[index+1] = 0.0;
	  }

	}
	else if(B[i]->Precision() == QUDA_DOUBLE_PRECISION) {
	  printfQuda("Double precision\n");
	  for(int index = 0; index < length; index+=2) {
	    static_cast<double *>(V[i])[index] = 1.0;
	    static_cast<double *>(V[i])[index+1] = 0.0;
	  }
	}
	else {
	  errorQuda("Constant null vectors not supported for precision = %d\n", B[i]->Precision());
	}

      }
    }

    printfQuda("Done loading vectors\n");
  }


}
