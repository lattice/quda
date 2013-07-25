#include <multigrid.h>
#include <qio_field.h>

namespace quda {

  MG::MG(MGParam &param, TimeProfile &profile) 
    : Solver(param, profile), param(param), smoother(0), coarse(0), fine(param.fine) {

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

      // create coarse grid operator
      // these need to be cpu fields
      ColorSpinorParam csParam(*(param.B[0]));  // FIXME this is crashing for some reason
      ColorSpinorField *tmp1 = new cpuColorSpinorField(csParam);
      ColorSpinorField *tmp2 = new cpuColorSpinorField(csParam);

      // these need to be gpu fields with native ordering basis
      ColorSpinorField *tmp3;  // FIXME allocate cudaSpinorFields
      ColorSpinorField *tmp4;   // FIXME allocate cudaSpinorFields
      DiracCoarse matCoarse(param.matResidual.Expose(), transfer, *tmp1, *tmp2, *tmp3, *tmp4);

      // create the next multigrid level
      MGParam coarse_param = param;
      coarse_param.level++;

      coarse_param.matResidual = matCoarse;
      coarse_param.matSmooth = matCoarse;

      coarse_param.fine = this;

      coarse = new MG(coarse_param, profile);
    }
  }

  MG::~MG() {
    if (param.level < param.Nlevel) {
      if (coarse) delete coarse;
      if (transfer) delete transfer;
    }
    if (smoother) delete smoother;

    if (tmp1) delete tmp1;
    if (tmp2) delete tmp2;
    if (tmp3) delete tmp3;
    if (tmp4) delete tmp4;
  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    if (param.level < param.Nlevel) {
      
      // do the pre smoothing
      param.maxiter = param.nu_pre;
      (*smoother)(x, b);

      // restrict to the coarse grid
      transfer->R(*r, *r_coarse);

      // recurse to the next lower level
      (*coarse)(*x_coarse, *r_coarse); 

      // prolongate back to this grid
      transfer->P(x, *x_coarse); // FIXME: need to ensure the prolongator sums to x here

      // do the post smoothing
      param.maxiter = param.nu_post;
      (*smoother)(x, b);

    } else { // do the coarse grid solve

      (*smoother)(x, b);

    }

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
