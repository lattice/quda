#include <multigrid.h>
#include <qio_field.h>

namespace quda {

  MG::MG(MGParam &param, TimeProfile &profile) 
    : Solver(param, profile), param(param), smoother(0), coarse(0), fine(param.fine), 
      r(0), r_coarse(0), hack1(0), hack2(0), hack3(0), hack4(0) {

    printfQuda("MG: Creating level %d of %d levels\n", param.level, param.Nlevel);

    if (param.level > QUDA_MAX_MG_LEVEL)
      errorQuda("Level=%d is greater than limit of multigrid recursion depth", param.level);

    // create the smoother for this level
    param.inv_type = param.smoother;
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
      csParam.setPrecision(csParam.precision);
      csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      hack3 = new cudaColorSpinorField(csParam);  // FIXME allocate cudaSpinorFields
      hack4 = new cudaColorSpinorField(csParam);   // FIXME allocate cudaSpinorFields

      DiracCoarse matCoarse(param.matResidual.Expose(), transfer, *hack1, *hack2, *hack3, *hack4);

      // create the next multigrid level
      MGParam coarse_param = param;
      coarse_param.level++;

      coarse_param.matResidual = matCoarse;
      coarse_param.matSmooth = matCoarse;

      coarse_param.fine = this;

      coarse = new MG(coarse_param, profile);
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

    if (hack1) delete hack1;
    if (hack2) delete hack2;
    if (hack3) delete hack3;
    if (hack4) delete hack4;
  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    printf("MG: level %d, entering V-cycle with x2=%e, r2=%e\n", 
	   param.level, blas::norm2(x), blas::norm2(b));

    if (param.level < param.Nlevel) {
      
      // do the pre smoothing
      printfQuda("MG: level %d, pre smoothing\n", param.level);
      param.maxiter = param.nu_pre;
      std::cout << x << b;
      (*smoother)(x, b);

      // restrict to the coarse grid
      printfQuda("MG: level %d, restriction\n", param.level);
      transfer->R(*r_coarse, *r);
      printfQuda("MG: r2 = %e r_coarse2 = %e\n", blas::norm2(*r), blas::norm2(*r_coarse));

      // recurse to the next lower level
      printfQuda("MG: solving coarse operator\n");
      //(*coarse)(*x_coarse, *r_coarse); 

      // prolongate back to this grid
      printfQuda("MG: level %d, prolongation\n", param.level);
      transfer->P(x, *x_coarse); // FIXME: need to ensure the prolongator sums to x here

      // do the post smoothing
      printfQuda("MG: level %d, post smoothing\n", param.level);
      param.maxiter = param.nu_post;
      (*smoother)(x, b);

    } else { // do the coarse grid solve

      (*smoother)(x, b);

    }

    printf("MG: level %d, leaving V-cycle with x2=%e, r2=%e\n", 
	   param.level, blas::norm2(x), blas::norm2(*r));

  }

  //supports seperate reading or single file read
  void loadVectors(std::vector<ColorSpinorField*> &B) {
    printfQuda("Start loading %d vectors from %s\n", nvec, vecfile);

    if (nvec < 1 || nvec > 20) errorQuda("nvec not set");

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
      errorQuda("No nullspace file defined");
    }

    printfQuda("Done loading vectors\n");
  }


}
