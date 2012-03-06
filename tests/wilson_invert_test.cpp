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

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// Wilson, clover-improved Wilson, and twisted mass are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;

extern char latfile[];

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

extern void usage(char** );
int main(int argc, char **argv)
{
  /*
  int ndim=4, dims[4] = {1, 1, 1, 1};
  char dimchar[] = {'X', 'Y', 'Z', 'T'};
  char *gridsizeopt[] = {"--xgridsize", "--ygridsize", "--zgridsize", "--tgridsize"};

  for (int i=1; i<argc; i++) {
    for (int d=0; d<ndim; d++) {
      if (!strcmp(argv[i], gridsizeopt[d])) {
	if (i+1 >= argc) {
	  printf("Usage: %s <args>\n", argv[0]);
	  printf("%s\t Set %c comms grid size (default = 1)\n", gridsizeopt[d], dimchar[d]); 
	  exit(1);
	}     
	dims[d] = atoi(argv[i+1]);
	if (dims[d] <= 0 ) {
	  printf("Error: Invalid %c grid size\n", dimchar[d]);
	  exit(1);
	}
	i++;
	break;
      }
    }
  }
  */



  int i;
  for (i =1;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    } 
    
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
    

  }


  if (prec_sloppy == QUDA_INVALID_PRECISION){
    prec_sloppy = prec;
  }
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID){
    link_recon_sloppy = link_recon;
  }


  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);

  // *** QUDA parameters begin here.

  int multi_shift = 0; // whether to test multi-shift or standard solver

  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH) {
    printf("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = prec;
  QudaPrecision cuda_prec_sloppy = prec_sloppy;
  QudaPrecision cuda_prec_precondition = QUDA_HALF_PRECISION;

  // offsets used only by multi-shift solver
  int num_offsets = 4;
  double offsets[4] = {0.01, 0.02, 0.03, 0.04};

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
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.dslash_type = dslash_type;

  double mass = -0.2180;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/gauge_param.anisotropy + mass));

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.mu = 0.1;
    inv_param.twist_flavor = QUDA_TWIST_MINUS;
  }

  inv_param.solution_type = QUDA_MATPC_SOLUTION;
  inv_param.solve_type = QUDA_NORMEQ_PC_SOLVE;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.gcrNkrylov = 30;
  inv_param.tol = 5e-7;
  inv_param.maxiter = 30;
  inv_param.reliable_delta = 1e-1; // ignored by multi-shift solver

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = QUDA_MR_INVERTER;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  inv_param.prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.dirac_tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;
  inv_param.preserve_dirac = QUDA_PRESERVE_DIRAC_YES;

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
#endif

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.verbosity = QUDA_VERBOSE;

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  // set parameters for the reference Dslash, and prepare fields to be loaded
  setDims(gauge_param.X);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover_inv=0, *clover=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = (inv_param.clover_cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    construct_clover_field(clover_inv, norm, diag, inv_param.clover_cpu_prec);

    // The uninverted clover term is only needed when solving the unpreconditioned
    // system or when using "asymmetric" even/odd preconditioning.
    int preconditioned = (inv_param.solve_type == QUDA_DIRECT_PC_SOLVE ||
			  inv_param.solve_type == QUDA_NORMEQ_PC_SOLVE);
    int asymmetric = preconditioned &&
                         (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
                          inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);
    if (!preconditioned) {
      clover = clover_inv;
      clover_inv = NULL;
    } else if (asymmetric) { // fake it by using the same random matrix
      clover = clover_inv;   // for both clover and clover_inv
    } else {
      clover = NULL;
    }
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize);

  void *spinorOut = NULL, **spinorOutMulti = NULL;
  if (multi_shift) {
    spinorOutMulti = (void**)malloc(num_offsets*sizeof(void *));
    for (int i=0; i<num_offsets; i++) {
      spinorOutMulti[i] = malloc(V*spinorSiteSize*sSize);
    }
  } else {
    spinorOut = malloc(V*spinorSiteSize*sSize);
  }

  // create a point source at 0
  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) *((float*)spinorIn) = 1.0;
  else *((double*)spinorIn) = 1.0;

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // load the clover term, if desired
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);
  
  // perform the inversion
  if (multi_shift) {
    double resid_sq;
    invertMultiShiftQuda(spinorOutMulti, spinorIn, &inv_param,
			 offsets, num_offsets, &resid_sq);
  } else {
    invertQuda(spinorOut, spinorIn, &inv_param);
  }

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
    
  printf("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) printf("   Clover: %f GiB\n", inv_param.cloverGiB);
  printf("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);

  if (multi_shift) {

    void *spinorTmp = malloc(V*spinorSiteSize*sSize);

    printf("Host residuum checks: \n");
    for(int i=0; i < num_offsets; i++) {
      ax(0, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	tm_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
		 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
	tm_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
		 inv_param.matpc_type, 1, inv_param.cpu_prec, gauge_param);
      } else {
	wil_matpc(spinorTmp, gauge, spinorOutMulti[i], inv_param.kappa, inv_param.matpc_type, 0,
		  inv_param.cpu_prec, gauge_param);
	wil_matpc(spinorCheck, gauge, spinorTmp, inv_param.kappa, inv_param.matpc_type, 1,
		  inv_param.cpu_prec, gauge_param);
      }

      axpy(offsets[i], spinorOutMulti[i], spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      double src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);
      printf("Shift i=%d Relative residual: requested = %g, actual = %g\n", i, inv_param.tol, sqrt(nrm2/src2));
    }
    free(spinorTmp);

  } else {
    
    if (inv_param.solution_type == QUDA_MAT_SOLUTION) {
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	tm_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
	       0, inv_param.cpu_prec, gauge_param); 
      } else {
	wil_mat(spinorCheck, gauge, spinorOut, inv_param.kappa, 0, 
		inv_param.cpu_prec, gauge_param);
      }
      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	ax(0.5/inv_param.kappa, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      }
    } else if(inv_param.solution_type == QUDA_MATPC_SOLUTION) {   
      if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
	tm_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, 
		 inv_param.matpc_type, 0, inv_param.cpu_prec, gauge_param);
      } else {
	wil_matpc(spinorCheck, gauge, spinorOut, inv_param.kappa, inv_param.matpc_type, 0, 
		  inv_param.cpu_prec, gauge_param);
      }
      if (inv_param.mass_normalization == QUDA_MASS_NORMALIZATION) {
	ax(0.25/(inv_param.kappa*inv_param.kappa), spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
      }
    }

    mxpy(spinorIn, spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
    double nrm2 = norm_2(spinorCheck, V*spinorSiteSize, inv_param.cpu_prec);
    double src2 = norm_2(spinorIn, V*spinorSiteSize, inv_param.cpu_prec);
    printf("Relative residual: requested = %g, actual = %g\n", inv_param.tol, sqrt(nrm2/src2));

  }
    
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  // end if the communications layer
  endCommsQuda();

  return 0;
}
