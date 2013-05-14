#include	<contractQuda.h>

void	setDiracPreParamCG	(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
{
	setDiracParam(diracParam, inv_param, pc);

	diracParam.gauge = gaugePrecondition;
	diracParam.fatGauge = gaugeFatPrecondition;
	diracParam.longGauge = gaugeLongPrecondition;    
	diracParam.clover = cloverPrecondition;

	for (int i=0; i<4; i++)
		diracParam.commDim[i] = 1;
}

void	createDiracCG		(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve)
{
	DiracParam diracParam;
	DiracParam diracSloppyParam;
	DiracParam diracPreParam;

	setDiracParam(diracParam, &param, pc_solve);
	setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
	setDiracPreParamCG(diracPreParam, &param, pc_solve);

	d = Dirac::create(diracParam);
	dSloppy = Dirac::create(diracSloppyParam);
	dPre = Dirac::create(diracPreParam);
}

/*	TODO Arregla la corriente conservada!!!	*/

void	loopPlainCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *cnRes, void **cnD, void **cnC)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	checkInvertParam(param);

	bool pc_solve	 = (param->solve_type == QUDA_DIRECT_PC_SOLVE ||
			    param->solve_type == QUDA_NORMEQ_PC_SOLVE);

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
			    param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if	(!pc_solve) param->spinorGiB *= 2;

	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if	(param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	else
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp2	 = NULL;

	const int *X = cudaGauge->X();

	void	*h_ctrn, *ctrnC, *ctrnS;

	printfQuda	("Allocating mem for contractions\n");

	if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
        	printfQuda	("Error allocating memory for contraction results in CPU.\n");
	        exit		(0);
	}

	cudaMemset(h_ctrn, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

	if	((cudaMalloc(&ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}

	cudaMemset(ctrnC, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

	if	((cudaMalloc(&ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}

	cudaMemset	(ctrnS, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

	printfQuda	("%ld bytes allocated in GPU for contractions\n", (long int) sizeof(double)*64*X[0]*X[1]*X[2]*X[3]);

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, param->input_location, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->inv_type == QUDA_BICGSTAB_INVERTER || param->inv_type == QUDA_GCR_INVERTER))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess*/
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nb	 = norm2(*b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	setDslashTuning	(param->tune, param->verbosity);
	setBlasTuning	(param->tune, param->verbosity);

	dirac.prepare	(in, out, *x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	switch	(param->inv_type)
	{
		case	QUDA_CG_INVERTER:
		// prepare source if we are doing CGNR
			if	(param->solution_type != QUDA_MATDAG_MAT_SOLUTION &&
        			 param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				cudaColorSpinorField tmp(*in);
				dirac.Mdag(*in, tmp);
			}

			{
				DiracMdagM	m(dirac), mSloppy(diracSloppy);
				CG		cg(m, mSloppy, *param, profileInvert);
				cg		(*out, *in);
			}

			break;

		case	QUDA_BICGSTAB_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION || 
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM		m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
			}

			break;

		case	QUDA_GCR_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION ||
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR		gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR	gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr	(*out, *in);
			}
			break;

		default:
			errorQuda	("Inverter type %d not implemented", param->inv_type);
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	tmp2		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	dim3		blockTwust(32, 1, 1);

	int	LX[4]	 = {X[0], X[1], X[2], X[3]};
	int	cDim[4]	 = {   1,    1,    1,    1};

	contractCuda	(b->Even(), x->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
	contractCuda	(b->Odd(),  x->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

	cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

	for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
		((double *) cnRes)[ix]	+= ((double*)h_ctrn)[ix];

	printfQuda	("Locals contracted\n");
	fflush		(stdout);

	for     (int mu=0; mu<4; mu++)
	{
		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();
		
		contractCuda	(b->Even(), tmp2->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(b->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		cudaMemcpy	(ctrnC, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToDevice);
		
		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(b->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(b->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(b->Even(), tmp2->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(b->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(b->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(b->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnD[mu])[ix]	-= ((double*)h_ctrn)[ix];

		cudaMemcpy	(h_ctrn, ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnC[mu])[ix]	-= ((double*)h_ctrn)[ix];
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_COMPUTE);
	profileContract.Stop(QUDA_PROFILE_TOTAL);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;       
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	x;
	delete	h_x;

	delete	b;
	delete	h_b;

	delete  tmp2;

	cudaFreeHost	(h_ctrn);
	cudaFree	(ctrnS);
	cudaFree	(ctrnC);

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	loopHPECG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *cnRes, void **cnD, void **cnC)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	checkInvertParam(param);

	bool pc_solve	 = (param->solve_type == QUDA_DIRECT_PC_SOLVE ||
			    param->solve_type == QUDA_NORMEQ_PC_SOLVE);

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
			    param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if	(!pc_solve) param->spinorGiB *= 2;

	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if	(param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	else
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp2	 = NULL;
	cudaColorSpinorField *tmp3	 = NULL;

	const int *X = cudaGauge->X();

	void	*h_ctrn, *ctrnC, *ctrnS;

	printfQuda	("Allocating mem for contractions\n");
	fflush	(stdout);

	if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
        	printfQuda	("Error allocating memory for contraction results in CPU.\n");
	        exit		(0);
	}

	cudaMemset(h_ctrn, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

	if	((cudaMalloc(&ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}

	cudaMemset(ctrnC, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);


	if	((cudaMalloc(&ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}

	cudaMemset	(ctrnS, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

	printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(double)*64*X[0]*X[1]*X[2]*X[3]);

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, param->input_location, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->inv_type == QUDA_BICGSTAB_INVERTER || param->inv_type == QUDA_GCR_INVERTER))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess*/
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nb	 = norm2(*b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	setDslashTuning	(param->tune, param->verbosity);
	setBlasTuning	(param->tune, param->verbosity);

	dirac.prepare	(in, out, *x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	switch	(param->inv_type)
	{
		case	QUDA_CG_INVERTER:
		// prepare source if we are doing CGNR
			if	(param->solution_type != QUDA_MATDAG_MAT_SOLUTION &&
        			 param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				cudaColorSpinorField tmp(*in);
				dirac.Mdag(*in, tmp);
			}

			{
				DiracMdagM	m(dirac), mSloppy(diracSloppy);
				CG		cg(m, mSloppy, *param, profileInvert);
				cg		(*out, *in);
			}

			break;

		case	QUDA_BICGSTAB_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION || 
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM		m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
			}

			break;

		case	QUDA_GCR_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION ||
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR		gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR	gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr	(*out, *in);
			}
			break;

		default:
			errorQuda	("Inverter type %d not implemented", param->inv_type);
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	tmp2		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);

	int	LX[4]	 = {X[0], X[1], X[2], X[3]};
	int	cDim[4]	 = {   1,    1,    1,    1};

	gamma5Cuda	(&(tmp2->Even()), &(b->Even()), blockTwost);
	gamma5Cuda	(&(tmp2->Odd()),  &(b->Odd()),  blockTwost);

	delete  h_b;
	delete  b;

	tmp3		 = new cudaColorSpinorField(cudaParam);

	printfQuda	("Synchronizing\n");

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	for	(int i = 0; i<4; i++)
	{
		wilsonDslashCuda	(&(tmp3->Even()), *gaugePrecise, &(tmp2->Odd()),  QUDA_EVEN_PARITY, 0, 0, param->kappa, cDim, profileContract);
		wilsonDslashCuda	(&(tmp3->Odd()),  *gaugePrecise, &(tmp2->Even()), QUDA_ODD_PARITY,  0, 0, param->kappa, cDim, profileContract);

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return;
		}

		double	mu_flavour	 = x->TwistFlavor()*param->mu;

		twistGamma5Cuda		(&(tmp2->Even()), &(tmp3->Even()), 1, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
		twistGamma5Cuda		(&(tmp2->Odd()),  &(tmp3->Odd()),  1, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	gamma5Cuda	(&(tmp3->Even()), &(tmp2->Even()), blockTwost);
	gamma5Cuda	(&(tmp3->Odd()),  &(tmp2->Odd()),  blockTwost);

	contractCuda	(tmp3->Even(), x->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
	contractCuda	(tmp3->Odd(),  x->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

	cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

	for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
		((double *) cnRes)[ix]	+= ((double*)h_ctrn)[ix];

	printfQuda	("Locals contracted\n");
	fflush		(stdout);

	for     (int mu=0; mu<4; mu++)
	{
		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();
		
		contractCuda	(tmp3->Even(), tmp2->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp3->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		cudaMemcpy	(ctrnC, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToDevice);
		
		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp3->Even(), tmp2->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp3->Odd(),  tmp2->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDevQuda	(&(tmp2->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu+4, cDim);
		covDevQuda	(&(tmp2->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu+4, cDim);
		cudaDeviceSynchronize	();

		contractCuda	(tmp2->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractCuda	(tmp2->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnD[mu])[ix]	-= ((double*)h_ctrn)[ix];

		cudaMemcpy	(h_ctrn, ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnC[mu])[ix]	-= ((double*)h_ctrn)[ix];
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_COMPUTE);
	profileContract.Stop(QUDA_PROFILE_TOTAL);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;        
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	x;
	delete	h_x;

	delete  tmp2;
	delete  tmp3;

	cudaFreeHost	(h_ctrn);
	cudaFree	(ctrnS);
	cudaFree	(ctrnC);

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	oneEndTrickCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void *cnRes_gv, void *cnRes_vv, void **cnD_gv, void **cnD_vv, void **cnC_gv, void **cnC_vv)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	checkInvertParam(param);

	bool pc_solve	 = (param->solve_type == QUDA_DIRECT_PC_SOLVE ||
			    param->solve_type == QUDA_NORMEQ_PC_SOLVE);

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
			    param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if	(!pc_solve) param->spinorGiB *= 2;

	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if	(param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	else
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp3	 = NULL;
	cudaColorSpinorField *tmp4	 = NULL;

	const int *X = cudaGauge->X();

	void	*h_ctrn, *ctrnC, *ctrnS;

	printfQuda	("Allocating mem for contractions\n");
	fflush	(stdout);

	if	((cudaMallocHost(&h_ctrn, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
        	printfQuda	("Error allocating memory for contraction results in CPU.\n");
	        exit		(0);
	}

	cudaMemset(h_ctrn, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

	if	((cudaMalloc(&ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}

	cudaMemset(ctrnC, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);


	if	((cudaMalloc(&ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3])) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}

	cudaMemset	(ctrnS, 0, sizeof(double)*32*X[0]*X[1]*X[2]*X[3]);

	printfQuda	("%ld bytes allocated in GPU for contractions\n", sizeof(double)*64*X[0]*X[1]*X[2]*X[3]);

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, param->input_location, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->inv_type == QUDA_BICGSTAB_INVERTER || param->inv_type == QUDA_GCR_INVERTER))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess*/
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nb	 = norm2(*b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	setDslashTuning	(param->tune, param->verbosity);
	setBlasTuning	(param->tune, param->verbosity);

	dirac.prepare	(in, out, *x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	switch	(param->inv_type)
	{
		case	QUDA_CG_INVERTER:
		// prepare source if we are doing CGNR
			if	(param->solution_type != QUDA_MATDAG_MAT_SOLUTION &&
        			 param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				cudaColorSpinorField tmp(*in);
				dirac.Mdag(*in, tmp);
			}

			{
				DiracMdagM	m(dirac), mSloppy(diracSloppy);
				CG		cg(m, mSloppy, *param, profileInvert);
				cg		(*out, *in);
			}

			break;

		case	QUDA_BICGSTAB_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION || 
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM		m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
			}

			break;

		case	QUDA_GCR_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION ||
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR		gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR	gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr	(*out, *in);
			}
			break;

		default:
			errorQuda	("Inverter type %d not implemented", param->inv_type);
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	delete  h_b;
	delete  b;

	tmp3		 = new cudaColorSpinorField(cudaParam);
	tmp4		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);

	DiracParam	dWParam;

	dWParam.type		 = QUDA_WILSON_DIRAC;
	dWParam.matpcType	 = QUDA_MATPC_EVEN_EVEN;
	dWParam.dagger		 = QUDA_DAG_NO;
	dWParam.gauge		 = gaugePrecise;
	dWParam.kappa		 = param->kappa;
	dWParam.mass		 = 1./(2.*param->kappa) - 4.;
	dWParam.m5		 = 0.;
	dWParam.mu		 = 0.;
	dWParam.verbose		 = param->verbosity;

	for	(int i=0; i<4; i++)
        	dWParam.commDim[i]	 = 1;   // comms are always on

	DiracWilson	*dW	 = new DiracWilson(dWParam);

	dW->M(*tmp4,*x);

	delete	dW;

	gamma5Cuda	(&(tmp3->Even()), &(tmp4->Even()), blockTwost);
	gamma5Cuda	(&(tmp3->Odd()),  &(tmp4->Odd()),  blockTwost);

	printfQuda	("Synchronizing\n");

	int	LX[4]	 = {X[0], X[1], X[2], X[3]};

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
		return;
	}

	contractGamma5Cuda	(x->Even(), tmp3->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
	contractGamma5Cuda	(x->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);

	cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

	for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
		((double *) cnRes_gv)[ix]	+= ((double*)h_ctrn)[ix];

	contractGamma5Cuda	(x->Even(), x->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
	contractGamma5Cuda	(x->Odd(),  x->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY); 

	cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

	for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
		((double *) cnRes_vv)[ix]       -= ((double*)h_ctrn)[ix];

	printfQuda	("Locals contracted\n");
	fflush		(stdout);

	for	(int mu=0; mu<4; mu++)	//Hasta 4
	{
		covDevQuda		(&(tmp4->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
		covDevQuda		(&(tmp4->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);

		contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);		//Term0
		contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
                cudaDeviceSynchronize	();

		cudaMemcpy		(ctrnC, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToDevice);

		covDevQuda		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
		covDevQuda		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);

		contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((double2*)ctrnC), 1, blockTwust, LX, QUDA_EVEN_PARITY);		//Term2 (C Sum)
		contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((double2*)ctrnC), 1, blockTwust, LX, QUDA_ODD_PARITY);
                cudaDeviceSynchronize	();

                contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);		//Term2 (D Dif)
		contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

		covDevQuda		(&(tmp4->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);
		covDevQuda		(&(tmp4->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);

		contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);		//Term1
		contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

		covDevQuda		(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);
		covDevQuda		(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);

		contractGamma5Cuda	(tmp4->Even(), tmp3->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);		//Term3
		contractGamma5Cuda	(tmp4->Odd(),  tmp3->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		cudaMemcpy		(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnD_gv[mu])[ix]	+= ((double*)h_ctrn)[ix];

		cudaMemcpy		(h_ctrn, ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnC_gv[mu])[ix]	+= ((double*)h_ctrn)[ix];
	}

	for     (int mu=0; mu<4; mu++)
	{
		covDevQuda	(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu, dWParam.commDim);
		covDevQuda	(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu, dWParam.commDim);
		
		contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 0, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 0, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();
		
		cudaMemcpy	(ctrnC, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToDevice);
		
		contractGamma5Cuda	(tmp4->Even(), x->Even(), ((double2*)ctrnC), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((double2*)ctrnC), 2, blockTwust, LX, QUDA_ODD_PARITY);
		
		contractGamma5Cuda	(tmp4->Even(), x->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		covDevQuda	(&(tmp4->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  mu+4, dWParam.commDim);
		covDevQuda	(&(tmp4->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, mu+4, dWParam.commDim);

		contractGamma5Cuda	(x->Even(), tmp4->Even(), ((double2*)ctrnS), 2, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(x->Odd(),  tmp4->Odd(),  ((double2*)ctrnS), 2, blockTwust, LX, QUDA_ODD_PARITY);

		contractGamma5Cuda	(tmp4->Even(), x->Even(), ((double2*)ctrnS), 1, blockTwust, LX, QUDA_EVEN_PARITY);
		contractGamma5Cuda	(tmp4->Odd(),  x->Odd(),  ((double2*)ctrnS), 1, blockTwust, LX, QUDA_ODD_PARITY);
		cudaDeviceSynchronize	();

		cudaMemcpy	(h_ctrn, ctrnS, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnD_vv[mu])[ix]	-= ((double*)h_ctrn)[ix];

		cudaMemcpy	(h_ctrn, ctrnC, sizeof(double)*32*X[0]*X[1]*X[2]*X[3], cudaMemcpyDeviceToHost);

		for	(int ix=0; ix<32*X[0]*X[1]*X[2]*X[3]; ix++)
			((double *) cnC_vv[mu])[ix]	-= ((double*)h_ctrn)[ix];
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_COMPUTE);
	profileContract.Stop(QUDA_PROFILE_TOTAL);

	profileInvert.Start(QUDA_PROFILE_D2H);
	*h_x	 = *x;                        
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printfQuda	("Error synchronizing!!!\n");
       		return;
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	x;
	delete	h_x;

	delete  tmp3;
	delete  tmp4;

	cudaFreeHost	(h_ctrn);
	cudaFree	(ctrnS);
	cudaFree	(ctrnC);

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	tDilHPECG	(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	//     checkInvertParam(param);

	bool pc_solve	 = (param->solve_type == QUDA_DIRECT_PC_SOLVE ||
			    param->solve_type == QUDA_NORMEQ_PC_SOLVE);

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
			    param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if	(!pc_solve) param->spinorGiB *= 2;

	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if	(param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	else
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;
	cudaColorSpinorField *tmp2	 = NULL;
	cudaColorSpinorField *tmp3	 = NULL;

	const int *X = cudaGauge->X();

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, param->input_location, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->inv_type == QUDA_BICGSTAB_INVERTER || param->inv_type == QUDA_GCR_INVERTER))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nb	 = norm2(*b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	setDslashTuning	(param->tune, param->verbosity);
	setBlasTuning	(param->tune, param->verbosity);

	dirac.prepare	(in, out, *x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	switch	(param->inv_type)
	{
		case	QUDA_CG_INVERTER:
		// prepare source if we are doing CGNR
			if	(param->solution_type != QUDA_MATDAG_MAT_SOLUTION &&
        			 param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				cudaColorSpinorField tmp(*in);
				dirac.Mdag(*in, tmp);
			}

			{
				DiracMdagM	m(dirac), mSloppy(diracSloppy);
				CG		cg(m, mSloppy, *param, profileInvert);
				cg		(*out, *in);
			}

			break;

		case	QUDA_BICGSTAB_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION || 
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM		m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
			}

			break;

		case	QUDA_GCR_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION ||
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR		gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR	gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr	(*out, *in);
			}
			break;

		default:
			errorQuda	("Inverter type %d not implemented", param->inv_type);
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	tmp2		 = new cudaColorSpinorField(cudaParam);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);
	int		LX[4]		 = { X[0], X[1], X[2], X[3] };
	int		commDim[4]	 = { 1, 1, 1, 1 };

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	tmp3		 = new cudaColorSpinorField(cudaParam);

	wilsonDslashCuda	(&(tmp2->Even()), *gaugePrecise, &(x->Odd()),  QUDA_EVEN_PARITY, 0, 0, param->kappa, commDim, profileContract);
	wilsonDslashCuda	(&(tmp2->Odd()),  *gaugePrecise, &(x->Even()), QUDA_ODD_PARITY,  0, 0, param->kappa, commDim, profileContract);

	double	mu_flavour	 = x->TwistFlavor()*param->mu;

	for	(int i = 0; i<3; i++)
	{
		twistGamma5Cuda		(&(tmp3->Even()), &(tmp2->Even()), 0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
		twistGamma5Cuda		(&(tmp3->Odd()),  &(tmp2->Odd()),  0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);

		wilsonDslashCuda	(&(tmp2->Even()), *gaugePrecise, &(tmp3->Odd()),  QUDA_EVEN_PARITY, 0, 0, param->kappa, commDim, profileContract);
		wilsonDslashCuda	(&(tmp2->Odd()),  *gaugePrecise, &(tmp3->Even()), QUDA_ODD_PARITY,  0, 0, param->kappa, commDim, profileContract);

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return;
		}
	}

	twistGamma5Cuda		(&(tmp3->Even()), &(tmp2->Even()), 0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);
	twistGamma5Cuda		(&(tmp3->Odd()),  &(tmp2->Odd()),  0, param->kappa, mu_flavour, 0., QUDA_TWIST_GAMMA5_INVERSE);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	delete	tmp2;

	const int	Lt	 = X[3]*comm_dim(3);
	int		cT	 = 0;

	for	(int time = tSlice; time < (tSlice+Lt); time += (Lt/nCoh))
	{
		int	tempT	 = time%Lt;

		if	((tempT/X[3]) == comm_coord(3))
		{
			int	tC	 = tempT - comm_coord(3)*X[3];

			contractTsliceCuda	(b->Even(), tmp3->Even(), ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_EVEN_PARITY);
			contractTsliceCuda	(b->Odd(),  tmp3->Odd(),  ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_ODD_PARITY);
		}

		cT++;
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_TOTAL);
	profileContract.Stop(QUDA_PROFILE_COMPUTE);

	profileInvert.Start(QUDA_PROFILE_D2H);
                                               
	*h_x	 = *x;        
                                               
	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	h_x;
	delete	x;
	delete	h_b;
	delete	b;

	delete  tmp3;

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

void	tDilutionCG	(void *hp_x, void *hp_b, QudaInvertParam *param, void **cnRes, const int tSlice, const int nCoh)
{
	profileInvert.Start(QUDA_PROFILE_TOTAL);

	int	lpFLAG	 = 0;

	if (!initialized) errorQuda("QUDA not initialized");
	if (getVerbosity() >= QUDA_DEBUG_VERBOSE) printQudaInvertParam(param);

	// check the gauge fields have been created
	cudaGaugeField *cudaGauge = checkGauge(param);

	if	(param->cuda_prec_precondition	!= QUDA_HALF_PRECISION)
	{
		printf	("\n\n\nUh oh, boy, something is very veeeery wrong...\n\n\n");
		param->cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	}

	if      (param->tol > 1E-5)
	{
		cudaGaugeField *temp	 = gaugeSloppy;
		gaugeSloppy		 = gaugePrecondition;
		gaugePrecondition	 = temp;
		lpFLAG			 = 1;

		QudaPrecision tempPrec	 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy	 = param->cuda_prec_precondition;
		param->cuda_prec_precondition = tempPrec;
	}

	//     checkInvertParam(param);

	bool pc_solve	 = (param->solve_type == QUDA_DIRECT_PC_SOLVE ||
			    param->solve_type == QUDA_NORMEQ_PC_SOLVE);

	bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
			    param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

	param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
	if	(!pc_solve) param->spinorGiB *= 2;

	param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
	if	(param->preserve_source == QUDA_PRESERVE_SOURCE_NO)
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
	else
		param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);

	param->secs = 0;
	param->gflops = 0;
	param->iter = 0;

	Dirac *d = NULL;
	Dirac *dSloppy = NULL;
	Dirac *dPre = NULL;

	// create the dirac operator
	createDiracCG	(d, dSloppy, dPre, *param, pc_solve);

	Dirac &dirac = *d;
	Dirac &diracSloppy = *dSloppy;
	Dirac &diracPre = *dPre;

	profileInvert.Start(QUDA_PROFILE_H2D);

	cudaColorSpinorField *b		 = NULL;
	cudaColorSpinorField *x		 = NULL;
	cudaColorSpinorField *in	 = NULL;
	cudaColorSpinorField *out	 = NULL;

	const int *X = cudaGauge->X();

	// wrap CPU host side pointers
	ColorSpinorParam cpuParam(hp_b, param->input_location, *param, X, pc_solution);
	ColorSpinorField *h_b	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	cpuParam.v = hp_x;
	ColorSpinorField *h_x	 = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
	static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

	// download source
	ColorSpinorParam	cudaParam(cpuParam, *param);
	cudaParam.create	 = QUDA_COPY_FIELD_CREATE;
	b			 = new cudaColorSpinorField(*h_b, cudaParam);

	if	(param->use_init_guess == QUDA_USE_INIT_GUESS_YES) // download initial guess
	{	// initial guess only supported for single-pass solvers
		if	((param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) && 
			(param->inv_type == QUDA_BICGSTAB_INVERTER || param->inv_type == QUDA_GCR_INVERTER))
			errorQuda("Initial guess not supported for two-pass solver");

		x	 = new cudaColorSpinorField(*h_x, cudaParam); // solution
	}
	else
	{ // zero initial guess
		cudaParam.create	 = QUDA_ZERO_FIELD_CREATE;
		x			 = new cudaColorSpinorField(cudaParam); // solution
	}

	profileInvert.Stop(QUDA_PROFILE_H2D);

	if (getVerbosity() >= QUDA_VERBOSE)
	{
		double nh_b	 = norm2(*h_b);
		double nb	 = norm2(*b);
		double nh_x	 = norm2(*h_x);
		double nx	 = norm2(*x);
		printfQuda	("Source:   CPU = %f, CUDA copy = %f\n", nh_b, nb);
		printfQuda	("Solution: CPU = %f, CUDA copy = %f\n", nh_x, nx);
	}

	setDslashTuning	(param->tune, param->verbosity);
	setBlasTuning	(param->tune, param->verbosity);

	dirac.prepare	(in, out, *x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		double nout	 = norm2(*out);
		printfQuda	("Prepared source   = %f\n", nin);
		printfQuda	("Prepared solution = %f\n", nout);
	}

	massRescale	(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *in);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nin	 = norm2(*in);
		printfQuda	("Prepared source post mass rescale = %f\n", nin);
	}

	switch	(param->inv_type)
	{
		case	QUDA_CG_INVERTER:
		// prepare source if we are doing CGNR
			if	(param->solution_type != QUDA_MATDAG_MAT_SOLUTION &&
        			 param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				cudaColorSpinorField tmp(*in);
				dirac.Mdag(*in, tmp);
			}

			{
				DiracMdagM	m(dirac), mSloppy(diracSloppy);
				CG		cg(m, mSloppy, *param, profileInvert);
				cg		(*out, *in);
			}

			break;

		case	QUDA_BICGSTAB_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION || 
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM		m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				BiCGstab	bicg(m, mSloppy, mPre, *param, profileInvert);
				bicg		(*out, *in);
			}

			break;

		case	QUDA_GCR_INVERTER:
			if	(param->solution_type == QUDA_MATDAG_MAT_SOLUTION ||
				 param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION)
			{
				DiracMdag	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR		gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr		(*out, *in);
				copyCuda	(*in, *out);
			}

			{
				DiracM	m(dirac), mSloppy(diracSloppy), mPre(diracPre);
				GCR	gcr(m, mSloppy, mPre, *param, profileInvert);
				gcr	(*out, *in);
			}
			break;

		default:
			errorQuda	("Inverter type %d not implemented", param->inv_type);
	}

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		printfQuda	("Solution = %f\n",nx);
	}

	dirac.reconstruct(*x, *b, param->solution_type);

	if	(getVerbosity() >= QUDA_VERBOSE)
        	printfQuda	("Contracting source\n");

	dim3		blockTwust(32, 1, 1);
	dim3		blockTwost(512, 1, 1);
	int		LX[4]		 = { X[0], X[1], X[2], X[3] };
	int		commDim[4]	 = { 1, 1, 1, 1 };

	printfQuda	("Synchronizing\n");

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return;
	}

	profileContract.Start(QUDA_PROFILE_TOTAL);
	profileContract.Start(QUDA_PROFILE_COMPUTE);

	const int	Lt	 = X[3]*comm_dim(3);
	int		cT	 = 0;

	for	(int time = tSlice; time < (tSlice+Lt); time += (Lt/nCoh))
	{
		int	tempT	 = time%Lt;

		if	((tempT/X[3]) == comm_coord(3))
		{
			int	tC	 = tempT - comm_coord(3)*X[3];

			contractTsliceCuda	(b->Even(), x->Even(), ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_EVEN_PARITY);
			contractTsliceCuda	(b->Odd(),  x->Odd(),  ((double2*)cnRes[cT]), 1, blockTwust, LX, tC, QUDA_ODD_PARITY);
		}

		cT++;
	}

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
       		return;
	}

	profileContract.Stop(QUDA_PROFILE_TOTAL);
	profileContract.Stop(QUDA_PROFILE_COMPUTE);

	profileInvert.Start(QUDA_PROFILE_D2H);

	*h_x	 = *x;

	profileInvert.Stop(QUDA_PROFILE_D2H);

	if	(getVerbosity() >= QUDA_VERBOSE)
	{
		double nx	 = norm2(*x);
		double nh_x	 = norm2(*h_x);
		printfQuda	("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
	}

	delete	h_x;
	delete	x;

	delete  h_b;
	delete  b;

	delete	d;
	delete	dSloppy;
	delete	dPre;

	if	(lpFLAG)
	{
		cudaGaugeField *temp		 = gaugeSloppy;
		gaugeSloppy			 = gaugePrecondition;
		gaugePrecondition		 = temp;

		QudaPrecision tempPrec		 = param->cuda_prec_sloppy;
		param->cuda_prec_sloppy		 = param->cuda_prec_precondition;
		param->cuda_prec_precondition	 = tempPrec;
	}

	// FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
	saveTuneCache	(getVerbosity());

	profileInvert.Stop(QUDA_PROFILE_TOTAL);
}

