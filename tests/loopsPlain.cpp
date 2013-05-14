#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include "misc.h"

#include "face_quda.h"

#ifdef QMP_COMMS
#include <qmp.h>
#endif

#include <gauge_qio.h>
#include <gsl/gsl_rng.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

//#define	TESTPOINT
//#define	RANDOM_CONF

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <contractQuda.h>
#include <cufft.h>

#include <mpi.h>

extern bool			tune;
extern int			device;
extern QudaDslashType		dslash_type;
extern int			xdim;
extern int			ydim;
extern int			zdim;
extern int			tdim;
extern int			numberHP;
extern int			nConf;
extern int			numberLP;
extern int			MaxP;
extern int			gridsize_from_cmdline[];
extern QudaReconstructType	link_recon;
extern QudaPrecision		prec;
extern QudaReconstructType	link_recon_sloppy;
extern QudaPrecision		prec_sloppy;

extern char			latfile[];

extern void			usage(char**);

int	genDataArray	(const int nSources, int *dataLP, int &flag)
{
	int	count = 0, power = 128, accum = nSources;

	if	(nSources < power)
	{
		dataLP[count]	= nSources;
		count		= 1;
		flag		= 0;

		return	count;
	}

	do
	{
		dataLP[count]	= power;

		accum	-= power;
		power	*= 2;
		count++;
	}	while	(accum > 0);

	if	(accum == 0)
	{
		flag	 = 0;
		return	count;
	}
	else
	{
		flag	 = 1;
		accum	+= power/2;

		dataLP[count-1]	 = accum;

		return	count;
	}
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d \n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     commDimPartitioned(0),
	     commDimPartitioned(1),
	     commDimPartitioned(2),
	     commDimPartitioned(3)); 
  
  return ;
  
}

void printCosa (int pt, int j, void **gauge, int id)
{
	double temp1, temp2, temp3, temp4, temp5, temp6;

	int k	 = pt*18;

	temp1	 = ((double**) gauge)[j][k+0];
	temp2	 = ((double**) gauge)[j][k+1];
	temp3	 = ((double**) gauge)[j][k+2];
	temp4	 = ((double**) gauge)[j][k+3];
	temp5	 = ((double**) gauge)[j][k+4];
	temp6	 = ((double**) gauge)[j][k+5];

	printf	("Punto %d %d (%d)\n\n", pt, j, id);
	printf	("id%d %+le +/- i %+le\t%+le +/- i %+le\t%+le +/- i %+le\n", id, temp1, temp2, temp3, temp4, temp5, temp6);

	temp1	 = ((double**) gauge)[j][k+6];
	temp2	 = ((double**) gauge)[j][k+7];
	temp3	 = ((double**) gauge)[j][k+8];
	temp4	 = ((double**) gauge)[j][k+9];
	temp5	 = ((double**) gauge)[j][k+10];
	temp6	 = ((double**) gauge)[j][k+11];

	printf	("id%d %+le +/- i %+le\t%+le +/- i %+le\t%+le +/- i %+le\n", id, temp1, temp2, temp3, temp4, temp5, temp6);

	temp1	 = ((double**) gauge)[j][k+12];
	temp2	 = ((double**) gauge)[j][k+13];
	temp3	 = ((double**) gauge)[j][k+14];
	temp4	 = ((double**) gauge)[j][k+15];
	temp5	 = ((double**) gauge)[j][k+16];
	temp6	 = ((double**) gauge)[j][k+17];

	printf	("id%d %+le +/- i %+le\t%+le +/- i %+le\t%+le +/- i %+le\n", id, temp1, temp2, temp3, temp4, temp5, temp6);
}

void	genRandomSource	(void *spinorIn, QudaInvertParam *inv_param, gsl_rng *rNum)
{
#ifdef	TESTPOINT
	if	(inv_param->cpu_prec == QUDA_SINGLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		if	(comm_rank() == 0)
			((float*) spinorIn)[7864338]		 = 1.;		//t-Component
	}
	else if	(inv_param->cpu_prec == QUDA_DOUBLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		if	(comm_rank() == 0)
//			((double*) spinorIn)[7864338]	 = 1.;
			((double*) spinorIn)[18]	 = 1.;
//			((double*) spinorIn)[0]	 = 1.;
/*
		for	(int i = 0; i<V; i++)
			for	(int j = 0; j<12; j++)
			{
				((double*) spinorIn)[i*24+j*2]		 = i;//TODO BORRAR
				((double*) spinorIn)[i*24+j*2+1]	 = j;//TODO BORRAR
			}
*/
	}
#else
	if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION) 
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		for	(int i = 0; i<V*12; i++)
		{
			int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);
	
			switch	(randomNumber)
			{
				case 0:
	
				((float*) spinorIn)[i*2]	= 1.;
				break;

				case 1:
	
				((float*) spinorIn)[i*2]	= -1.;
				break;
	
				case 2:
	
				((float*) spinorIn)[i*2+1]	= 1.;
				break;
	
				case 3:
	
				((float*) spinorIn)[i*2+1]	= -1.;
				break;
			}
		}
	}
	else
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		for	(int i = 0; i<V*12; i++)
		{
			int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);
	
			switch	(randomNumber)
			{
				case 0:
	
				((double*) spinorIn)[i*2]	= 1.;
				break;
	
				case 1:
	
				((double*) spinorIn)[i*2]	= -1.;
				break;
	
				case 2:
	
				((double*) spinorIn)[i*2+1]	= 1.;
				break;
	
				case 3:
	
				((double*) spinorIn)[i*2+1]	= -1.;
				break;
			}
		}
	}
#endif
}

int getFullLatIndex(int i, int oddBit, int commDir, int rank_id, int mpi_size){
  /*
    int boundaryCrossings = i/(Z[0]/2) + i/(Z[1]*Z[0]/2) + i/(Z[2]*Z[1]*Z[0]/2);
    return 2*i + (boundaryCrossings + oddBit) % 2;
  */
//(default sizes):

  int X1  = xdim;
  int X2  = ydim;
  int X3  = zdim;
  int X4  = tdim;

  int Y1 = X1, Y2 = X2, Y3 = X3, Y4 = X4;

  switch(commDir){
        case 0 : Y1 /= mpi_size;
        break;
        case 1 : Y2 /= mpi_size;
        break;
        case 2 : Y3 /= mpi_size;
        break;
        case 3 : Y4 /= mpi_size;
  }

  int Y1h = Y1/2;

  int sid = i;
  int za  = sid/Y1h;
  int x1h = sid - za*Y1h;
  int zb  = za/Y2;
  int x2  = za - zb*Y2;
  int zc  = zb/Y3; //zc<>x4
  int x3  = zb - zc*Y3;
  int xs  = zc/Y4;
  int x4  = zc - xs*Y4;
  int x1odd = (x2 + x3 + x4 + xs + oddBit) & 1;
  int x1 = 2*x1h + x1odd;//for even dims is ok.

  //do a shift in needed direction to get correct coordinate:
  switch(commDir){
        case 0 : x1 += rank_id*Y1;
        break;
        case 1 : x2 += rank_id*Y2;
        break;
        case 2 : x3 += rank_id*Y3;
        break;
        case 3 : x4 += rank_id*Y4;
  }

  int X = x1+x2*X1+x3*X1*X2+x4*X1*X2*X3+xs*X1*X2*X3*X4;

  return X;//this number is unique for each mpirank
}

template <typename Float>
void random(Float *v, const int length) {
  const int mpi_rank = comm_rank();
  const int mpi_size = comm_size();

  for(int i = 0; i < length; i++) {
    if(!(i % 24)) srand(getFullLatIndex(i / 24, 0, 3, mpi_rank, mpi_size));
      v[i] = rand() / (double)RAND_MAX;
  }
}

int	doCudaFFT	(const int keep, void *cnRes, void **cnD, void **cnC)
{
	static cufftHandle	fftPlan;
	static int		init = 0;
	int			nRank[3]	 = {xdim, ydim, zdim};
	const int		Vol		 = xdim*ydim*zdim;

	static cudaStream_t	streamCuFFT;

	if	(!keep)
	{
		if	(init)
		{
			cufftDestroy		(fftPlan);
			cudaStreamDestroy	(streamCuFFT);
		}

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}

		init	 = 0;

		return	0;
	}

	if	(!init)
	{
		cudaStreamCreate	(&streamCuFFT);

		if	(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_Z2Z, 16*tdim) != CUFFT_SUCCESS)
		{
			printf	("Error in the FFT!!!\n");
			return 1;
		}

		cufftSetCompatibilityMode	(fftPlan, CUFFT_COMPATIBILITY_NATIVE);
		cufftSetStream			(fftPlan, streamCuFFT);

		printfQuda	("Synchronizing\n");

		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}

		init	 = 1;
	}
	else
		printfQuda	("CuFFT plan already initialized\n");

	void	*ctrnS;

	if	((cudaMalloc(&ctrnS, sizeof(double)*32*Vol*tdim)) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}
	cudaMemcpy	(ctrnS, cnRes, sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);

	if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		printf  ("Error executing FFT!!!\n");
		return 1;
	}
	
	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return 1;
	}
	
	cudaMemcpy	(cnRes, ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

	for	(int mu=0; mu<4; mu++)
	{
		cudaMemcpy	(ctrnS, cnD[mu], sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnD[mu], ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);

		cudaMemcpy	(ctrnS, cnC[mu], sizeof(double)*32*Vol*tdim, cudaMemcpyHostToDevice);
	
		if	(cufftExecZ2Z(fftPlan, (double2 *) ctrnS, (double2 *) ctrnS, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			printf  ("Error executing FFT!!!\n");
			return 1;
		}
		
		if	(cudaDeviceSynchronize() != cudaSuccess)
		{
			printf	("Error synchronizing!!!\n");
			return 1;
		}
		
		cudaMemcpy	(cnC[mu], ctrnS, sizeof(double)*32*Vol*tdim, cudaMemcpyDeviceToHost);
	}

	cudaFree	(ctrnS);

	return	0;
}

void	dumpData	(int nSols, const char *Pref, int **mom, void *cnRes, void **cnD, void **cnC, const int iDiv)	//nSols -> nSol, iDiv desaparece
{
	FILE		*sfp;
	FILE		*sfpMu;

	char		file_name[256];
	int		nSol = 1;

	const int	Vol = xdim*ydim*zdim;

	sprintf(file_name, "Local.loop.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

	if	((sfp = fopen(file_name, "wb")) == NULL)
		printf("\nCannot open file %s\n", file_name),	exit(-1);

	sprintf(file_name, "covDev.loop.%s.%04d.%d_%d", Pref, nConf, comm_size(), comm_rank()); 

	if	((sfpMu = fopen(file_name, "wb")) == NULL)
		printf("\nCannot open file %s\n", file_name),	exit(-1);

	if	(iDiv)				//*	Y recuerda cambiar el nSols de abajo por nSol!!!
		nSol	 = nSols;		//*
	else					//*
		nSol	 = 1;			//*

	int	testMom	 = 0;

	for	(int ip=0; ip<Vol; ip++)
		for	(int wt=0; wt<tdim; wt++)
		{
			if	((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= MaxP)
			{
		        	int	rT	 = wt+comm_coord(3)*tdim;

				double	testR[4]	 = { 0., 0., 0., 0.};								// TEST
				double	testI[4]	 = { 0., 0., 0., 0.};								// TEST
				double	testRO		 = 0.;										// TEST
				double	testIO		 = 0.;										// TEST

				for	(int gm=0; gm<16; gm++)
				{										// TEST
					fprintf (sfp, "%03d %02d %02d %+d %+d %+d %+.10le %+.10le\n", nSols, rT, gm, mom[ip][0], mom[ip][1], mom[ip][2],
						((double2*)cnRes)[ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2*)cnRes)[ip+Vol*wt+Vol*tdim*gm].y/((double) nSol));

					for	(int mu = 0; mu < 4; mu++)
					{
						fprintf (sfpMu, "%03d %02d %d %02d %+d %+d %+d %+.10le %+.10le %+.10le %+.10le\n", nSols, rT, mu, gm, mom[ip][0], mom[ip][1], mom[ip][2],
							((double2**)cnD)[mu][ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2**)cnD)[mu][ip+Vol*wt+Vol*tdim*gm].y/((double) nSol),
							((double2**)cnC)[mu][ip+Vol*wt+Vol*tdim*gm].x/((double) nSol), ((double2**)cnC)[mu][ip+Vol*wt+Vol*tdim*gm].y/((double) nSol));

						if	((ip == testMom)&&((gm == 0)||(gm == 5)||(gm == 10)||(gm == 15)))			// TEST
						{
							testR[mu]	+= ((double2**)cnC)[mu][ip+Vol*wt+Vol*tdim*gm].x;
							testI[mu]	+= ((double2**)cnC)[mu][ip+Vol*wt+Vol*tdim*gm].y;

							if	(mu == 0)
							{
								testRO	+= ((double2*)cnRes)[ip+Vol*wt+Vol*tdim*gm].x;
								testIO	+= ((double2*)cnRes)[ip+Vol*wt+Vol*tdim*gm].y;
							}
						}
					}
				}

/*				if	(ip == testMom)
					for	(int mu = 0; mu < 4; mu++)
						printf	("%d %02d %+le %+le\n", mu, rT, testR[mu], testI[mu]);					// END TESTi

				if	(ip == testMom)
					printf	("D %02d %+le %+le\n", rT, testRO, testIO);					// END TESTi
*/
				fflush  (sfp);
				fflush  (sfpMu);
			}
		}

	fclose(sfp);
	fclose(sfpMu);

	return;
}

int	main	(int argc, char **argv)
{
	int	i, k;
	double	precisionHP	 = 4e-10;
	double	precisionLP	 = 2e-4;

	char	name[16];

	int	dataLP[16];
	int	maxSources, flag;

	for (i =1;i < argc; i++)
	{
		if	(process_command_line_option(argc, argv, &i) == 0)
			continue;
    
		printf	("ERROR: Invalid option:%s\n", argv[i]);
		usage	(argv);
	}

  	MPI_Init(&argc, &argv);
	initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);

	maxSources	 = genDataArray (numberLP, dataLP, flag);

	printfQuda	("Will dump %d files in ", maxSources);

	for	(i=0; i<maxSources; i++)
		printf	("%d ", dataLP[i]);


	printfQuda	("clusters.\n");

	//	Starts Quda initialization

	if	(prec_sloppy == QUDA_INVALID_PRECISION)
		prec_sloppy		 = prec;

	if	(link_recon_sloppy == QUDA_RECONSTRUCT_INVALID)
		link_recon_sloppy	 = link_recon;

  // *** QUDA parameters begin here.


	dslash_type				 = QUDA_TWISTED_MASS_DSLASH;
//	dslash_type				 = QUDA_WILSON_DSLASH;

	QudaPrecision cpu_prec			 = QUDA_DOUBLE_PRECISION;
	QudaPrecision cuda_prec			 = prec;
	QudaPrecision cuda_prec_sloppy		 = prec_sloppy;

	QudaGaugeParam gauge_param		 = newQudaGaugeParam();
	QudaInvertParam inv_param		 = newQudaInvertParam();

	gauge_param.X[0]			 = xdim;
	gauge_param.X[1]			 = ydim;
	gauge_param.X[2]			 = zdim;
	gauge_param.X[3]			 = tdim;

	gauge_param.anisotropy			 = 1.0;
	gauge_param.type			 = QUDA_WILSON_LINKS;
	gauge_param.gauge_order			 = QUDA_QDP_GAUGE_ORDER;
	gauge_param.t_boundary			 = QUDA_ANTI_PERIODIC_T;

	gauge_param.cpu_prec			 = cpu_prec;
	gauge_param.cuda_prec			 = cuda_prec;
	gauge_param.reconstruct			 = link_recon;
	gauge_param.cuda_prec_sloppy		 = cuda_prec_sloppy;
	gauge_param.reconstruct_sloppy		 = link_recon_sloppy;
	gauge_param.cuda_prec_precondition	 = QUDA_HALF_PRECISION;
	gauge_param.reconstruct_precondition	 = link_recon_sloppy;
	gauge_param.gauge_fix			 = QUDA_GAUGE_FIXED_NO;

	inv_param.dslash_type = dslash_type;

	double mass = -2.;
	inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

	if	(dslash_type == QUDA_TWISTED_MASS_DSLASH)
	{
		inv_param.mu = 0.003;
		inv_param.twist_flavor = QUDA_TWIST_MINUS;
	}

	inv_param.solution_type		 = QUDA_MAT_SOLUTION;
	inv_param.solve_type		 = QUDA_NORMEQ_PC_SOLVE;
//	inv_param.solve_type		 = QUDA_DIRECT_PC_SOLVE;
	inv_param.matpc_type		 = QUDA_MATPC_EVEN_EVEN;
	inv_param.dagger		 = QUDA_DAG_NO;
//	inv_param.mass_normalization	 = QUDA_MASS_NORMALIZATION;
	inv_param.mass_normalization	 = QUDA_KAPPA_NORMALIZATION;

//	inv_param.inv_type		 = QUDA_BICGSTAB_INVERTER;
	inv_param.inv_type		 = QUDA_CG_INVERTER;

	inv_param.gcrNkrylov		 = 30;
	inv_param.tol			 = precisionHP;
	inv_param.maxiter		 = 40000;
	inv_param.reliable_delta	 = 1e-2; // ignored by multi-shift solver

  // domain decomposition preconditioner parameters

	inv_param.inv_type_precondition	 = QUDA_INVALID_INVERTER;
	inv_param.schwarz_type		 = QUDA_ADDITIVE_SCHWARZ;
	inv_param.precondition_cycle	 = 1;
	inv_param.tol_precondition	 = 1e-1;
	inv_param.maxiter_precondition	 = 10;
	inv_param.verbosity_precondition = QUDA_SILENT;
	inv_param.omega			 = 1.0;


	inv_param.cpu_prec		 = cpu_prec;
	inv_param.cuda_prec		 = cuda_prec;
	inv_param.cuda_prec_sloppy	 = cuda_prec_sloppy;
	inv_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
	inv_param.preserve_source	 = QUDA_PRESERVE_SOURCE_NO;
	inv_param.gamma_basis		 = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
//	inv_param.gamma_basis		 = QUDA_UKQCD_GAMMA_BASIS;
	inv_param.dirac_order		 = QUDA_DIRAC_ORDER;

	inv_param.tune			 = QUDA_TUNE_YES;
//	inv_param.tune			 = QUDA_TUNE_NO;
//	inv_param.preserve_dirac	 = QUDA_PRESERVE_DIRAC_NO;

	inv_param.input_location	 = QUDA_CPU_FIELD_LOCATION;
	inv_param.output_location	 = QUDA_CPU_FIELD_LOCATION;

	gauge_param.ga_pad		 = 0; // 24*24*24/2;
	inv_param.sp_pad		 = 0; // 24*24*24/2;
	inv_param.cl_pad		 = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
	int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
	int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
	int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
	int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
	int pad_size =MAX(x_face_size, y_face_size);
	pad_size = MAX(pad_size, z_face_size);
	pad_size = MAX(pad_size, t_face_size);
	gauge_param.ga_pad = pad_size; 
//  inv_param.cl_pad = pad_size; 
//  inv_param.sp_pad = pad_size; 
#endif

	inv_param.verbosity = QUDA_VERBOSE;

	//set the T dimension partitioning flag
	//commDimPartitionedSet(3);

	// *** Everything between here and the call to initQuda() is
	// *** application-specific.

	// set parameters for the reference Dslash, and prepare fields to be loaded
	setDims			(gauge_param.X);

	setSpinorSiteSize	(24);

	size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
	size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

	void *gauge[4];

	for	(int dir = 0; dir < 4; dir++)
		if	((gauge[dir]	 = malloc(V*gaugeSiteSize*gSize)) == NULL)
		{
			printf	("Fatal Error; Couldn't allocate memory in host for gauge fields. Asked for %ld bytes.", V*gaugeSiteSize*gSize);
			exit	(1);
		}

//	totalMem	+= ((double) (V*gaugeSiteSize*gSize*4))/(1024.*1024.*1024.);

#ifdef  RANDOM_CONF
	construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
#else
	if	(read_custom_binary_gauge_field((double**)gauge, latfile, &gauge_param, &inv_param, gridsize_from_cmdline))
	{
		printf	("Fatal Error; Couldn't read gauge conf %s\n", latfile);
		exit	(1);
	}
//	construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
#endif

	comm_barrier	();

	const int	Vol	 = xdim*ydim*zdim;

  // initialize the QUDA library
	initQuda(device);

	void	*cnRes;

	if	((cudaHostAlloc(&cnRes, sizeof(double2)*16*tdim*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnRes_gv\n"), exit(1);

	cudaMemset	(cnRes, 0, sizeof(double2)*16*tdim*Vol);

	void	**cnD;
	void	**cnC;

	cnD	 = (void**) malloc(sizeof(double2*)*4);
	cnC	 = (void**) malloc(sizeof(double2*)*4);

	if	(cnD == NULL) printf("Error allocating memory cnD_HP\n"), exit(1);
	if	(cnC == NULL) printf("Error allocating memory cnC_HP\n"), exit(1);


	cudaDeviceSynchronize();

	for	(int mu = 0; mu < 4; mu++)
	{
		if	((cudaHostAlloc(&(cnD[mu]), sizeof(double2)*tdim*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnD_vv)[%d]\n", mu), exit(1);
		cudaMemset	(cnD[mu], 0, tdim*16*Vol*sizeof(double2));

		if	((cudaHostAlloc(&(cnC[mu]), sizeof(double2)*tdim*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnC_vv)[%d]\n", mu), exit(1);
		cudaMemset	(cnC[mu], 0, tdim*16*Vol*sizeof(double2));
	}

  // load the gauge field
	loadGaugeQuda	((void*)gauge, &gauge_param);


	void	*spinorIn	 = malloc(V*spinorSiteSize*sSize);
	void	*spinorCheck	 = malloc(V*spinorSiteSize*sSize);
	void	*spinorOut	 = malloc(V*spinorSiteSize*sSize);

	int	**mom;

	if	((mom = (int **) malloc(sizeof(int*)*Vol)) == NULL)
	{
		printfQuda	("Fatal Error: Couldn't allocate memory for momenta.");
		exit		(1);
	}

	for	(int ip=0; ip<Vol; ip++)
	{
		if	((mom[ip] = (int *) malloc(sizeof(int)*3)) == NULL)
		{
			printfQuda	("Fatal Error: Couldn't allocate memory for momenta.");
			exit		(1);
		}
		else
		{
			mom[ip][0]	 = 0;
			mom[ip][1]	 = 0;
			mom[ip][2]	 = 0;
		}
	}

//	totalMem	+= ((double) (Vol*sizeof(int)*3))/(1024.*1024.*1024.);

	int momIdx	 = 0;
	int totMom	 = 0;

	for	(int pz = 0; pz < zdim; pz++)
		for	(int py = 0; py < ydim; py++)
			for	(int px = 0; px < xdim; px++)
			{
				if	(px < xdim/2)
					mom[momIdx][0]	 = px;
				else
					mom[momIdx][0]	 = px - xdim;

				if	(py < ydim/2)
					mom[momIdx][1]	 = py;
				else
					mom[momIdx][1]	 = py - ydim;

				if	(pz < zdim/2)
					mom[momIdx][2]	 = pz;
				else
					mom[momIdx][2]	 = pz - zdim;

				if	((mom[momIdx][0]*mom[momIdx][0]+mom[momIdx][1]*mom[momIdx][1]+mom[momIdx][2]*mom[momIdx][2])<=MaxP)
					totMom++;

				momIdx++;
			}

	printfQuda	("\nTotal momenta %d\n\n", totMom);

	gsl_rng	*rNum	 = gsl_rng_alloc(gsl_rng_ranlux);
	gsl_rng_set	(rNum, (int) clock()*comm_rank());

	printfQuda	("Allocated memory for random number generator\n");

  // start the timer
	double time0 = -((double)clock());

  // perform the inversion

	printfQuda	("Starting inversions\n");

	for	(i=0; i<numberHP; i++)
	{
		genRandomSource	(spinorIn, &inv_param, rNum);
		inv_param.tol	 = precisionHP;
		loopPlainCG	(spinorOut, spinorIn, &inv_param, cnRes, cnD, cnC);

		doCudaFFT	(1, cnRes, cnD, cnC);

		sprintf		(name, "H%03d.S%03d", numberHP, i);
		dumpData	(1, name, mom, cnRes, cnD, cnC, 0);

		cudaMemset	(cnRes, 0, tdim*16*Vol*sizeof(double2));

		for	(int nu=0; nu<4; nu++)
		{
			cudaMemset	(cnD[nu], 0, tdim*16*Vol*sizeof(double2));
			cudaMemset	(cnC[nu], 0, tdim*16*Vol*sizeof(double2));
		}

		if	(numberLP > 0)
		{
			inv_param.tol	 = precisionLP;
			loopPlainCG	(spinorOut, spinorIn, &inv_param, cnRes, cnD, cnC);

			doCudaFFT	(1, cnRes, cnD, cnC);

			sprintf		(name, "M%03d.S%03d", numberHP, i);
			dumpData	(1, name, mom, cnRes, cnD, cnC, 0);

			cudaMemset	(cnRes, 0, tdim*16*Vol*sizeof(double2));

			for	(int nu=0; nu<4; nu++)
			{
				cudaMemset	(cnD[nu], 0, tdim*16*Vol*sizeof(double2));
				cudaMemset	(cnC[nu], 0, tdim*16*Vol*sizeof(double2));
			}
		}
	}

//	if	(numberLP > 0)
//	{
	for	(k=0; k<maxSources; k++)
	{
		inv_param.tol		 = precisionLP;

		for	(i=0; i<dataLP[k]; i++)
		{
//		for	(i=0; i<numberLP; i++)
//		{
			printfQuda	("\nSource LP %04d\n", i);
			genRandomSource	(spinorIn, &inv_param, rNum);
			loopPlainCG	(spinorOut, spinorIn, &inv_param, cnRes, cnD, cnC);
		}

		doCudaFFT	(1, cnRes, cnD, cnC);

		if	(flag && (k == (maxSources - 1)))
			sprintf	(name, "L9999");
		else
			sprintf	(name, "L%04d", dataLP[k]);
//		sprintf		(name, "L%04d", numberLP);
		dumpData	(dataLP[k], name, mom, cnRes, cnD, cnC, 0);	//dataLP[k] -> numberLP, el 1 se va

		cudaMemset	(cnRes, 0, tdim*16*Vol*sizeof(double2));

		for	(int mu=0; mu<4; mu++)							//*Todo esto se va al carajo
		{
			cudaMemset	(cnD[mu], 0, tdim*16*Vol*sizeof(double2));
			cudaMemset	(cnC[mu], 0, tdim*16*Vol*sizeof(double2));
		}										//*---Hasta aquí---
	}

//	doCudaFFT	(0, NULL, NULL, NULL, NULL, NULL, NULL);

  // stop the timer
	double timeIO	 = -((double)clock());
	time0		+= clock();
	time0		/= CLOCKS_PER_SEC;
    
	printfQuda	("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", inv_param.spinorGiB, gauge_param.gaugeGiB);
	printfQuda	("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

	gsl_rng_free(rNum);

	for	(int ip=0; ip<Vol; ip++)
		free	(mom[ip]);

	free		(mom);
	cudaFreeHost	(cnRes);

	for	(int mu=0; mu<4; mu++)
	{
		cudaFreeHost	(cnD[mu]);
		cudaFreeHost	(cnC[mu]);
	}

	free(cnD);
	free(cnC);

	if	(inv_param.solution_type == QUDA_MAT_SOLUTION)
	{
		if	(dslash_type == QUDA_TWISTED_MASS_DSLASH)
			tm_mat	(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
		else
			wil_mat	(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);

		if	(inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
			ax	(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	}
	else if	(inv_param.solution_type == QUDA_MATPC_SOLUTION)
	{   
		if	(dslash_type == QUDA_TWISTED_MASS_DSLASH)
			tm_matpc	(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
		else
			wil_matpc	(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);

		if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
			ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	}

	for	(int dir=0; dir<4; dir++)
		free	(gauge[dir]);

	mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);

	double nrm2	 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	double src2	 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);

	printfQuda	("Relative residual: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

	freeGaugeQuda	();

  // finalize the QUDA library
	endQuda		();

  // end if the communications layer
	MPI_Finalize	();

	free	(spinorIn);
	free	(spinorCheck);
	free	(spinorOut);

	timeIO		+= clock();
	timeIO		/= CLOCKS_PER_SEC;

	printf		("%g seconds spent on IO\n", timeIO);
	fflush		(stdout);

	return	0;
}

