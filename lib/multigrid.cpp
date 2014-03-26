#include <multigrid.h>
#include <qio_field.h>
#include <string.h>

namespace quda {
  
  // FIXME - do basis check

  MG::MG(MGParam &param, TimeProfile &profile) 
    : Solver(param, profile), param(param), presmoother(0), postsmoother(0), coarse(0), fine(param.fine), 
      param_coarse(0), param_presmooth(0), param_postsmooth(0), r(0), r_coarse(0), x_coarse(0), matCoarse(0), 
      hack1(0), hack2(0), hack3(0), hack4(0), y(0) {

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

    // if not on the coarsest level, construct it
    if (param.level < param.Nlevel) {
      // create transfer operator
      printfQuda("MG: start creating transfer operator\n");
      transfer = new Transfer(param.B, param.Nvec, param.geoBlockSize, param.spinBlockSize);
      printfQuda("MG: end creating transfer operator\n");

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
      printfQuda("Creating coarse null-space vectors\n");
      B_coarse = new std::vector<ColorSpinorField*>();
      B_coarse->resize(param.Nvec);
      for (int i=0; i<param.Nvec; i++) {
	(*B_coarse)[i] = param.B[0]->CreateCoarse(param.geoBlockSize, param.spinBlockSize, param.Nvec);
	transfer->R(*(*B_coarse)[i], *(param.B[i]));
      }

      // create the next multigrid level
      printfQuda("Creating next multigrid level\n");
      param_coarse = new MGParam(param, *B_coarse, *matCoarse, *matCoarse);
      param_coarse->level++;
      param_coarse->fine = this;
      param_coarse->smoother = QUDA_BICGSTAB_INVERTER;
      param_coarse->delta = 1e-1;

      coarse = new MG(*param_coarse, profile);
    }

    printfQuda("MG: Setup of level %d completed\n", param.level);

    // now we can run through the verificaion
    if (param.level == 1) verify();
  }

  MG::~MG() {
    if (param.level < param.Nlevel) {
      if (B_coarse) for (int i=0; i<param.Nvec; i++) delete (*B_coarse)[i];
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
      transfer->R(*r_coarse, *param.B[i]);
      transfer->P(*hack2, *r_coarse);
      printfQuda("Vector %d: norms %e %e ", i, blas::norm2(*param.B[i]), blas::norm2(*hack2));
      printfQuda("deviation = %e\n", blas::xmyNorm(*(param.B[i]), *hack2));
    }

    printfQuda("\nChecking 1 > || (1 - D P (P^\\dagger D P) P^\\dagger v_k || / || v_k || for %d vectors\n", 
	       param.Nvec);

    for (int i=0; i<param.Nvec; i++) {
      transfer->R(*r_coarse, *(param.B[i]));
      (*coarse)(*x_coarse, *r_coarse);
      transfer->P(*hack2, *x_coarse);
      param.matResidual(*hack1,*hack2);
      printfQuda("Vector %d: norms %e %e ", i, blas::norm2(*param.B[i]), blas::norm2(*hack1));
      printfQuda("relative residual = %e\n", sqrt(blas::xmyNorm(*(param.B[i]), *hack1) / blas::norm2(*param.B[i])) );
    }

    printfQuda("\nChecking 0 = (1 - P^\\dagger P) eta_c \n");
    x_coarse->Source(QUDA_RANDOM_SOURCE);
    transfer->P(*hack2, *x_coarse);
    transfer->R(*r_coarse, *hack2);
    printfQuda("Vector norms %e %e ", blas::norm2(*x_coarse), blas::norm2(*r_coarse));
    printfQuda("deviation = %e\n", blas::xmyNorm(*x_coarse, *r_coarse));
  }

  void MG::operator()(ColorSpinorField &x, ColorSpinorField &b) {

    if (getVerbosity() >= QUDA_VERBOSE)
      printfQuda("MG: level %d, entering V-cycle with x2=%e, r2=%e\n", 
		 param.level, blas::norm2(x), blas::norm2(b));

    if (param.level < param.Nlevel) {
      
      // do the pre smoothing
      printfQuda("MG: level %d, pre smoothing b2=%e\n", param.level, blas::norm2(b));
      (*presmoother)(x, b);

      // FIXME - residual computation should be in the previous smoother
      param.matResidual(*r, x);
      double r2 = blas::xmyNorm(b, *r);

      // restrict to the coarse grid
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("MG: level %d, restriction\n", param.level);
      transfer->R(*r_coarse, *r);
      if (getVerbosity() >= QUDA_VERBOSE) 
	printfQuda("MG: level %d after pre-smoothing x2 = %e, r2 = %e, r_coarse2 = %e\n", 
		   param.level, blas::norm2(x), r2, blas::norm2(*r_coarse));

      // recurse to the next lower level
      (*coarse)(*x_coarse, *r_coarse);
      if (getVerbosity() >= QUDA_VERBOSE) 
	printfQuda("MG: level %d after coarse solve x_coarse2 = %e r_coarse2 = %e\n", 
		   param.level, blas::norm2(*x_coarse), blas::norm2(*r_coarse)); 

      // prolongate back to this grid
      transfer->P(*r, *x_coarse); // repurpose residual storage
      // FIXME - sum should be done inside the transfer operator
      blas::xpy(*r, x); // sum to solution

      if (getVerbosity() >= QUDA_VERBOSE) {
	printfQuda("MG: Prolongated coarse solution y2 = %e\n", blas::norm2(*r)); 
	printfQuda("MG: level %d, after coarse-grid correction x2 = %e, r2 = %e\n", 
		   param.level, blas::norm2(x), blas::norm2(*r));
      }

      // do the post smoothing
      (*postsmoother)(x,b);

    } else { // do the coarse grid solve
      (*presmoother)(x, b);
    }

    if (getVerbosity() >= QUDA_VERBOSE) {
      param.matResidual(*r, x);
      double r2 = blas::xmyNorm(b, *r);
      printfQuda("MG: level %d, leaving V-cycle with x2=%e, r2=%e\n", 
		 param.level, blas::norm2(x), r2);
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
	blas::zero(*B[i]);
	ColorSpinorParam csParam(*B[i]);
	csParam.create = QUDA_ZERO_FIELD_CREATE;
	ColorSpinorField *tmp = ColorSpinorField::Create(csParam);
	for (int s=i; s<4; s+=2) {
	  for (int c=0; c<B[i]->Ncolor(); c++) {
	    tmp->Source(QUDA_CONSTANT_SOURCE, 1, s, c);
	    blas::xpy(*tmp,*B[i]);
	  }
	}
	delete tmp;
      }

      for (int i=2; i<nvec; i++) B[i] -> Source(QUDA_RANDOM_SOURCE);
    }

    printfQuda("Done loading vectors\n");
  }

  void DiracCoarse::initializeCoarse() {

    QudaPrecision prec = t->Vectors().Precision();
    int ndim = t->Vectors().Ndim();
    int x[QUDA_MAX_DIM];
    //Number of coarse sites.
    const int *geo_bs = t->Geo_bs();
    for(int i = 0; i < ndim; i++) {
      x[i] = t->Vectors().X(i)/geo_bs[i];
    }

    //Coarse Color
    int Nc_c = t->nvec();

    //Coarse Spin
    int Ns_c = t->Vectors().Nspin()/t->Spin_bs();

    GaugeFieldParam gParam = new GaugeFieldParam();
    memcpy(gParam.x, x, QUDA_MAX_DIM*sizeof(int));
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.precision = prec;
    gParam.nDim = ndim;
    gParam.siteDim= 2*ndim+1;
    gParam.geometry = QUDA_COARSE_GEOMETRY;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    cpuGaugeField *Y = new cpuGaugeField(gParam);
    
    dirac->createCoarseOp(*t,*Y);
  }

}
