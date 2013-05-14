#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
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

//#define	CONFIGFILE	"/home/avaquero/confs/L32T64/conf.1000"
//#define	CONFIGFILE	"/home/avaquero/confs/L48T96/conf.1000"

//#define	TESTPOINT
//#define	RANDOM_CONF

//#define		CROSSCHECK

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <contractQuda.h>
#include <cufft.h>

#include <mpi.h>

#define	tOffset	8		//Offset for time dilution

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

int	genDataArray	(const int nSources, int *dataHP, int &flag, int base)
{
	int	count = 0, power = 1, accum = nSources, base2 = base;

	if	(base > 1)
	{
		do
		{
			base2	>>= 1;
			power	<<= 1;
		}	while	(base2 != 1);
	}

	do
	{
		dataHP[count]	= power;

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

		dataHP[count-1]	 = accum;

		return	count;
	}
}

int	readPosition	(int N, int confNum)
{
	FILE	*rnPos;
	char	fileName[64];

	int	nConf, x, y, z, t;

	sprintf	(fileName, "randomPositions_%02d", N);

	if	((rnPos = fopen(fileName, "r")) == NULL)
	{
		printf	("Error opening file %s for reading. Exiting...\n", fileName);
		exit	(1);
	}

	do
	{
		fscanf	(rnPos, "%d %d %d %d %d\n", &nConf, &x, &y, &z, &t);

		if	(nConf == confNum)
		{
			fclose	(rnPos);
			return	t+tOffset;
		}

	}	while	(!feof(rnPos));

	fclose	(rnPos);

	printf	("Error: Configuration not found in randomPositions file %s\n", fileName);
	exit	(1);
}

int getFullLatIndex(int i, int oddBit, int commDir, int rank_id, int mpi_size)
{
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

  int sid = i/24;
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

  int cNu = ((i%24)&1);
  int col = (((i%24)/2)%3);
  int dIx = ((i%24)/6);

  char ccc[2];
  ccc[0] = 'R';
  ccc[1] = 'I';

  char ccl[3]; 
  ccl[0] = 'R';
  ccl[1] = 'G';
  ccl[2] = 'B';

  int j = i/24;

/*
	x4	 = j/(xdim*ydim*zdim/2);
	x3	 = ((j/(xdim*ydim/2))%zdim);
	x2	 = ((j/xdim/2)%ydim);
	x1	 = (j%xdim)/2 + oddBit;
*/
	printf	("R%d\tIdx %d/%d\tCoor (%02d,%02d,%02d,%02d) (%c,%c,%d)\n", rank_id, i, cNu+col*2+dIx*6+(((x4*zdim+x3)*ydim+x2)*xdim/2+x1h)*24, x1, x2, x3, x4, ccc[cNu], ccl[col], dIx);

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

void	genRandomSource	(void *spinorIn, QudaInvertParam *inv_param, gsl_rng *rNum, int tSlice)
{
	const int		Vol	 = xdim*ydim*zdim;
	const int		Lt	 = tdim*comm_dim(3);
	const int		Offset	 = V*12;


#ifdef	TESTPOINT
	if	(inv_param->cpu_prec == QUDA_SINGLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		for	(int time = tSlice; time<(Lt+tSlice); time+=(Lt/4))
		{
			int	tC	 = ((time%Lt) - tdim*comm_coord(3));	//%tdim*comm_dim(3)

			while	(tC < 0)	{	tC	+= tdim;	};

			if	(comm_coord(3) != (time%Lt)/tdim)
				continue;

			((float*) spinorIn)[12*Vol*tC]		 = 1.;		//t-Component	+18	1.
		}
	}
	else if	(inv_param->cpu_prec == QUDA_DOUBLE_PRECISION)
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		for	(int time = tSlice; time<(Lt+tSlice); time+=(Lt/4))
		{
			int	tC	 = ((time%Lt) - tdim*comm_coord(3));	//%tdim*comm_dim(3)

			while	(tC < 0)	{	tC	+= tdim;	};

			if	(comm_coord(3) != (time%Lt)/tdim)
				continue;

			((double*) spinorIn)[12*Vol*tC]		 = 1.;		//t-Component	+18
		}
	}
#else
	if (inv_param->cpu_prec == QUDA_SINGLE_PRECISION) 
	{
		for	(int i = 0; i<V*24; i++)
			((float*) spinorIn)[i]		 = 0.;

		for	(int time = tSlice; time<(Lt+tSlice); time+=(Lt/4))
		{
			int	tC	 = ((time%Lt) - tdim*comm_coord(3));

			while	(tC < 0)	{	tC	+= tdim;	};

			if	(comm_coord(3) != (time%Lt)/tdim)
				continue;

			for	(int i = 12*Vol*tC; i<12*Vol*(tC+1); i+=2)
			{
				int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);

				switch	(randomNumber)
				{
					case 0:
	
					((float*) spinorIn)[i]		 = 1.;
					break;

					case 1:
	
					((float*) spinorIn)[i]		 = -1.;
					break;
	
					case 2:
	
					((float*) spinorIn)[i+1]	 = 1.;
					break;
	
					case 3:
	
					((float*) spinorIn)[i+1]	 = -1.;
					break;
				}
			}

			for	(int i = 12*Vol*tC+Offset; i<(12*Vol*(tC+1)+Offset); i+=2)
			{
				int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);

				switch	(randomNumber)
				{
					case 0:
	
					((float*) spinorIn)[i]		 = 1.;
					break;

					case 1:

					((float*) spinorIn)[i]		 = -1.;
					break;

					case 2:

					((float*) spinorIn)[i+1]	 = 1.;
					break;

					case 3:

					((float*) spinorIn)[i+1]	 = -1.;
					break;
				}
			}
		}
	}
	else
	{
		for	(int i = 0; i<V*24; i++)
			((double*) spinorIn)[i]		 = 0.;

		for	(int time = tSlice; time<(Lt+tSlice); time+=(Lt/4))
		{
			int	tC	 = ((time%Lt) - tdim*comm_coord(3));

			while	(tC < 0)	{	tC	+= tdim;	};

			if	(comm_coord(3) != (time%Lt)/tdim)
				continue;

			for	(int i = 12*Vol*tC; i<12*Vol*(tC+1); i+=2)
			{
				int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);

				switch	(randomNumber)
				{
					case 0:
	
					((double*) spinorIn)[i]		 = 1.;
					break;
	
					case 1:
	
					((double*) spinorIn)[i]		 = -1.;
					break;
	
					case 2:
	
					((double*) spinorIn)[i+1]	 = 1.;
					break;
	
					case 3:
	
					((double*) spinorIn)[i+1]	 = -1.;
					break;
				}
			}

			for	(int i = 12*Vol*tC+Offset; i<(12*Vol*(tC+1)+Offset); i+=2)
			{
				int	randomNumber	=	gsl_rng_uniform_int	(rNum, 4);

				switch	(randomNumber)
				{
					case 0:
	
					((double*) spinorIn)[i]		= 1.;
					break;
	
					case 1:
	
					((double*) spinorIn)[i]		= -1.;
					break;
	
					case 2:
	
					((double*) spinorIn)[i+1]	= 1.;
					break;
	
					case 3:
	
					((double*) spinorIn)[i+1]	= -1.;
					break;
				}
			}

		}
	}
#endif
}

int	timeBlock	(int tSlice)
{
		if	(tSlice/tdim != comm_coord(3))
			return	0;
		else
			return	1;
}

int	doCudaFFT	(const int keep, void *cnRes)
{
	static cufftHandle	fftPlan;
	static int		init = 0;
	int			nRank[3]	 = {xdim, ydim, zdim};
	const int		Vol		 = xdim*ydim*zdim;

	if	(!keep)
	{
		if	(init)
		{
			cufftDestroy	(fftPlan);
			init	 = 0;
		}

		return	0;
	}

	if	(!init)
	{
		if	(cufftPlanMany(&fftPlan, 3, nRank, nRank, 1, Vol, nRank, 1, Vol, CUFFT_Z2Z, 16) != CUFFT_SUCCESS)
		{
			printf	("Error in the FFT!!!\n");
			return 1;
		}

		cufftSetCompatibilityMode	(fftPlan, CUFFT_COMPATIBILITY_NATIVE);

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

	if	((cudaMalloc(&ctrnS, sizeof(double)*32*Vol)) == cudaErrorMemoryAllocation)
	{
		printfQuda	("Error allocating memory for contraction results in GPU.\n");
		exit		(0);
	}

	cudaMemcpy	(ctrnS, cnRes, sizeof(double)*32*Vol, cudaMemcpyHostToDevice);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return 1;
	}
	
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
	
	cudaMemcpy	(cnRes, ctrnS, sizeof(double)*32*Vol, cudaMemcpyDeviceToHost);

	if	(cudaDeviceSynchronize() != cudaSuccess)
	{
		printf	("Error synchronizing!!!\n");
		return 1;
	}

	cudaFree	(ctrnS);

	return	0;
}

void	dumpData	(int nSols, int tSlice, const char *Pref, int **mom, void *cnRes, int iDiv)
{
	FILE		*sfp;
//	FILE		*sfpMu;

	char		file_name[256];

	const int	Vol = xdim*ydim*zdim;

	int		nSol;

//	if	(!timeBlock(tSlice))
//		return;

	if	(iDiv == 1)
		nSol	 = nSols;
	else
		nSol	 = 1;

	sprintf(file_name, "tDil.HPE.%s.%04d.%02d", Pref, nConf, tSlice); 

	if	((sfp = fopen(file_name, "wb")) == NULL)
		printf("\nCannot open file %s\n", file_name),	exit(-1);

	for	(int ip=0; ip<Vol; ip++)
	{
//		double	testRO	 = 0.;
//		double	testIO	 = 0.;

		if	((mom[ip][0]*mom[ip][0] + mom[ip][1]*mom[ip][1] + mom[ip][2]*mom[ip][2]) <= MaxP)
			for	(int gm=0; gm<16; gm++)
			{										// TEST
				fprintf (sfp, "%03d %02d %02d %+d %+d %+d %+.10le %+.10le\n", nSols, tSlice, gm, mom[ip][0], mom[ip][1], mom[ip][2],
					((double2*)cnRes)[ip+Vol*gm].x/((double) nSol), ((double2*)cnRes)[ip+Vol*gm].y/((double) nSol));

/*				if	((ip == 0)&&((gm == 0)||(gm == 5)||(gm == 10)||(gm == 15)))			// TEST
				{
			      		testRO	+= ((double2*)cnRes)[ip+Vol*gm].x;
					testIO	+= ((double2*)cnRes)[ip+Vol*gm].y;
				}*/
			}

//		if	(ip == 0)
//			printf	("R %02d(%d) %+le %+le\n", tSlice, comm_rank(), testRO, testIO);					// END TESTi
	}

	fclose(sfp);

	sfp	 = NULL;

	return;
}

void	reOrder	(double *array1, double *array2, const int arrayOffset)
{
	if	(array1 != array2)
	{
		for	(int i = 0; i<V*arrayOffset; i++)
			array2[i]	= 0.;
	}

	for	(int i = 0; i<V*arrayOffset; i++)
	{
		int	pointT		=	i/arrayOffset;
		int	offset		=	i%arrayOffset;
		int	oddBit		=	0;

		if	(pointT >= V/2)
		{
			pointT	-= V/2;
			oddBit	 = 1;
		}

		int za		 = pointT/(xdim/2);
		int x1h		 = pointT - za*(xdim/2);
		int zb		 = za/ydim;
		int x2		 = za - zb*ydim;
		int x4		 = zb/zdim;
		int x3		 = zb - x4*zdim;
		int x1odd	 = (x2 + x3 + x4 + oddBit) & 1;
		int x1		 = 2*x1h + x1odd;
		int X		 = x1 + xdim*(x2 + ydim*(x3 + zdim*x4));
		X		*= arrayOffset;
		X		+= offset;

		if	(array1 != array2)
			array2[X]	= array1[i];
		else
		{
			double	temp	 = array2[X];
			array2[X]	 = array1[i];
			array1[i]	 = temp;
		}
	}

	return;
}

extern void usage(char** );
int main(int argc, char **argv)
{
	int	i, j, k;
	double	precisionHP	 = 5e-10;
	double	precisionLP	 = 2e-3;

	char	name[128];

	int	tSlice		 = -1;

	int	dataHP[16];
	int	dataLP[16];
	int	maxSources, flag, maxSourcesLP, flagLP;

	for (i =1;i < argc; i++)
	{
		if	(process_command_line_option(argc, argv, &i) == 0)
			continue;
    
		printf	("ERROR: Invalid option:%s\n", argv[i]);
		usage	(argv);
	}

  	MPI_Init(&argc, &argv);
	initCommsGridQuda(4, gridsize_from_cmdline, NULL, NULL);

	if	((tSlice	 = readPosition	(0, nConf)) == -1)
		return	1;

	maxSources	 = genDataArray	(numberHP, dataHP, flag, 1);
	maxSourcesLP	 = genDataArray	(numberLP, dataLP, flagLP, 128);

	printfQuda	("Will dump %d files in ", maxSources);

	for	(i=0; i<maxSources; i++)
		printf	("%d ", dataHP[i]);

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
	inv_param.matpc_type		 = QUDA_MATPC_EVEN_EVEN;
	inv_param.dagger		 = QUDA_DAG_NO;
//	inv_param.mass_normalization	 = QUDA_MASS_NORMALIZATION;
	inv_param.mass_normalization	 = QUDA_KAPPA_NORMALIZATION;

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

	#ifdef	CROSSCHECK
		printf	("Using crosscheck!\n");
		inv_param.gamma_basis		 = QUDA_UKQCD_GAMMA_BASIS;
	#else
		inv_param.gamma_basis		 = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
	#endif

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

/*
	int device = comm_rank() % 2; // CUDA device number

	QudaDslashType dslash_type = QUDA_TWISTED_MASS_DSLASH;

	QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
	QudaPrecision cuda_prec = prec;
	QudaPrecision cuda_prec_sloppy = prec_sloppy;

	QudaGaugeParam gauge_param = newQudaGaugeParam();
	QudaInvertParam inv_param = newQudaInvertParam();
 
	gauge_param.X[0] = xdim;
	gauge_param.X[1] = ydim;
	gauge_param.X[2] = zdim;
	gauge_param.X[3] = tdim;

	gauge_param.anisotropy = 1.0;
	gauge_param.type = QUDA_WILSON_LINKS;
	gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
	gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  
	gauge_param.cpu_prec = cpu_prec;
	gauge_param.cuda_prec = cuda_prec;
	gauge_param.reconstruct = link_recon;
	gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
	gauge_param.reconstruct_sloppy = link_recon_sloppy;
	gauge_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
	gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

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
	inv_param.matpc_type		 = QUDA_MATPC_EVEN_EVEN;
	inv_param.dagger		 = QUDA_DAG_NO;
//	inv_param.mass_normalization	 = QUDA_MASS_NORMALIZATION;
	inv_param.mass_normalization	 = QUDA_KAPPA_NORMALIZATION;

	inv_param.inv_type		 = QUDA_CG_INVERTER;

	inv_param.gcrNkrylov		 = 30;
	inv_param.tol			 = precisionHP;
	inv_param.maxiter		 = 20000;
	inv_param.reliable_delta	 = 1e-2; // ignored by multi-shift solver

	// domain decomposition preconditioner parameters

	inv_param.inv_type_precondition = QUDA_INVALID_INVERTER;
	inv_param.tol_precondition = 1e-1;
	inv_param.maxiter_precondition = 10;
	inv_param.verbosity_precondition = QUDA_SILENT;
	inv_param.prec_precondition = QUDA_HALF_PRECISION;
	inv_param.omega = 1.0;

	inv_param.cpu_prec		 = cpu_prec;
	inv_param.cuda_prec		 = cuda_prec;
	inv_param.cuda_prec_sloppy	 = cuda_prec_sloppy;
	inv_param.preserve_source	 = QUDA_PRESERVE_SOURCE_NO;
	inv_param.gamma_basis		 = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
//	inv_param.gamma_basis		 = QUDA_UKQCD_GAMMA_BASIS;
	inv_param.dirac_order		 = QUDA_DIRAC_ORDER;

	inv_param.tune			 = QUDA_TUNE_YES;
//	inv_param.preserve_dirac	 = QUDA_PRESERVE_DIRAC_NO;

	gauge_param.ga_pad = 0; // 24*24*24/2;
	inv_param.sp_pad = 0; // 24*24*24/2;
	inv_param.cl_pad = 0; // 24*24*24/2;

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
//	inv_param.cl_pad = pad_size; 
//	inv_param.sp_pad = pad_size; 
#endif

	inv_param.verbosity = QUDA_VERBOSE;
*/


	//set the T dimension partitioning flag
	//commDimPartitionedSet(3);

	// *** Everything between here and the call to initQuda() is
	// *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
	setDims(gauge_param.X);

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

  // initialize the QUDA library
	initQuda(device);

  // load the gauge field
	loadGaugeQuda((void*)gauge, &gauge_param);

	const int	Vol	 = xdim*ydim*zdim;

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

	void	**cnRes, **cnRes2;

	if	((cnRes = (void **)malloc(sizeof(void*)*4)) == NULL) printf("Error allocating memory cnRes\n"), exit(1);
	if	((cnRes2 = (void **)malloc(sizeof(void*)*4)) == NULL) printf("Error allocating memory cnRes\n"), exit(1);

	for	(i=0; i<4; i++)	
	{
		if	((cudaHostAlloc(&(cnRes[i]), sizeof(double2)*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnRes\n"), exit(1);
		cudaMemset	(cnRes[i], 0, 16*Vol*sizeof(double2));
		if	((cudaHostAlloc(&(cnRes2[i]), sizeof(double2)*16*Vol, cudaHostAllocMapped)) != cudaSuccess) printf("Error allocating memory cnRes\n"), exit(1);
		cudaMemset	(cnRes2[i], 0, 16*Vol*sizeof(double2));
	}

	const int	Lt = tdim*comm_dim(3);

	comm_barrier();
  // start the timer
	double time0 = -((double)clock());

  // perform the inversion

	printfQuda	("Starting inversions\n");

	for	(k=0; k<maxSources; k++)
	{
		for	(i=0; i<dataHP[k]; i++)
		{
			genRandomSource(spinorIn, &inv_param, rNum, tSlice);


			#ifdef	CROSSCHECK
				reOrder	((double*)spinorIn, (double*)spinorCheck, 24);

				FILE	*out;

				if	((out = fopen("tDilSpHPE.In", "w+")) == NULL)
				{
					printf	("Error creating file.\n");
					return	-1;
				}

				for	(j=0; j<V*12; j++)
					fprintf	(out, "%+1.1lf %+1.1lf\n", ((double*)spinorCheck)[2*j], ((double*)spinorCheck)[2*j+1]);

				fclose	(out);
			#endif

			inv_param.tol	 = precisionHP;
			tDilHPECG	(spinorOut, spinorIn, &inv_param, cnRes, tSlice, 4);

			#ifdef	CROSSCHECK
				reOrder	((double*)spinorOut, (double*)spinorCheck, 24);

				if	((out = fopen("tDilSpHPE.Out", "w+")) == NULL)
				{
					printf	("Error creating file.\n");
					return	-1;
				}

				for	(j=0; j<V*12; j++)
					fprintf	(out, "%+2.8le %+2.8le\n", ((double*)spinorCheck)[2*j], ((double*)spinorCheck)[2*j+1]);

				fclose	(out);
			#endif

			if	(numberLP > 0)
			{
				inv_param.tol	 = precisionLP;
				tDilHPECG	(spinorOut, spinorIn, &inv_param, cnRes2, tSlice, 4);
			}
		}

		for	(j=0; j<4; j++)	
		{
			int	currentT		 = (tSlice+(j*Lt)/4)%Lt;

			if	(!timeBlock(currentT))
				continue;

			doCudaFFT	(1, cnRes[j]);

			if	(flag && (k == maxSources - 1))
				sprintf		(name, "H%03d.C999", numberHP);
			else
				sprintf		(name, "H%03d.C%03d", numberHP, dataHP[k]);

			dumpData	(dataHP[k], currentT, name, mom, cnRes[j], 0);
			cudaMemset	(cnRes[j], 0, 16*Vol*sizeof(double2));

			if	(numberLP > 0)
			{
				doCudaFFT	(1, cnRes2[j]);

				if	(flag && (k == maxSources - 1))
					sprintf		(name, "M%03d.C999", numberHP);
				else
					sprintf		(name, "M%03d.C%03d", numberHP, dataHP[k]);

				dumpData	(dataHP[k], currentT, name, mom, cnRes2[j], 0);

				cudaMemset	(cnRes2[j], 0, 16*Vol*sizeof(double2));
			}
		}
	}

//!NEW
//info:cnRes must be allocated via cudaHostAlloc (for zero copy with flag cudaHostAllocMapped in conjubction with
//cudaSetDeviceFlag() and cudaDeviceMapHost as an argument), device must support UVAS (all Fermi cards)

	for	(k=0; k<maxSourcesLP; k++)
        {
		if	((maxSourcesLP == 1) && (dataLP[k] == 0))
			continue;

		inv_param.tol		 = precisionLP;

		for	(i=0; i<dataLP[k]; i++)
		{
			printfQuda	("\nSource LP %04d\n", i);
			genRandomSource	(spinorIn, &inv_param, rNum, tSlice);
			tDilHPECG	(spinorOut, spinorIn, &inv_param, cnRes, tSlice, 4);		//REMOVE
		}

		for	(j=0; j<4; j++)
		{
			int	currentT		 = (tSlice+(j*Lt)/4)%Lt;

			if	(!timeBlock(currentT))
				continue;

			doCudaFFT	(1, cnRes[j]);

			if	(flagLP && (k == (maxSourcesLP - 1)))
				sprintf		(name, "L9999");
			else
				sprintf		(name, "L%04d", dataLP[k]);

			dumpData	(dataLP[k], currentT, name, mom, cnRes[j], 1);
			cudaMemset	(cnRes[j], 0, 16*Vol*sizeof(double2));
		}
	}

	doCudaFFT	(0, NULL);

	//	stop the timer             
	double timeIO	 = -((double)clock());
	time0		+= clock();
	time0		/= CLOCKS_PER_SEC;

	printfQuda	("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", inv_param.spinorGiB, gauge_param.gaugeGiB);
	printfQuda	("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

	if	(inv_param.solution_type == QUDA_MAT_SOLUTION)
	{
		if	(dslash_type == QUDA_TWISTED_MASS_DSLASH)
			tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 0, inv_param.cpu_prec, gauge_param);
		else
			wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, inv_param.cpu_prec, gauge_param);

		if	(inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
			ax(1./(2.*inv_param.kappa), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	}
	else if	(inv_param.solution_type == QUDA_MATPC_SOLUTION)
	{
		if	(dslash_type == QUDA_TWISTED_MASS_DSLASH)
			tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
		else
			wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);

		if	(inv_param.mass_normalization == QUDA_MASS_NORMALIZATION)
			ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	}


	mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);

	double nrm2	 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
	double src2	 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);

	printfQuda	("Relative residual: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

	for	(int dir = 0; dir < 4; dir++)
		free	(gauge[dir]);

	freeGaugeQuda();

	gsl_rng_free(rNum);

	for	(int ip=0; ip<Vol; ip++)
		free	(mom[ip]);

	free	(mom);

	for	(i=0; i<4; i++)
	{
		cudaFreeHost	(cnRes[i]);
		cudaFreeHost	(cnRes2[i]);
	}

	free	(cnRes);
	free	(cnRes2);

	free	(spinorIn);
	free	(spinorCheck);
	free	(spinorOut);

	timeIO		+= clock();
	timeIO		/= CLOCKS_PER_SEC;

	printf		("%g seconds spent on IO\n", timeIO);
	fflush		(stdout);

  // finalize the QUDA library
	endQuda		();

  // end if the communications layer
	MPI_Finalize	();

	return	0;
}

