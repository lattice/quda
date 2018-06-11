#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <algorithm>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
//#include <quda_arpack_interface.h>

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;

// Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaPrecision prec_sloppy;
extern QudaPrecision prec_precondition;
extern QudaPrecision prec_arpack;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass;
extern double kappa; // kappa of Dirac operator
extern double mu;
extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern char latfile[];

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern char vec_infile[];
extern char vec_outfile[];

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

extern int niter;
extern int gcrNkrylov; // number of inner iterations for GCR, or l for BiCGstab-l
extern int pipeline; // length of pipeline for fused operations in GCR or BiCGstab-l

extern int eig_nEv;
extern int eig_nKv;
extern double eig_tol;
extern int eig_maxiter;
extern bool eig_use_poly_acc;
extern int eig_poly_deg;
extern double eig_amin;
extern double eig_amax;
extern bool eig_use_normop;
extern bool eig_use_dagger;
extern bool eig_compute_svd;
extern QudaArpackSpectrumType arpack_spectrum;
extern int arpack_mode;
extern char arpack_logfile[512];

extern bool verify_results;

namespace quda {
  extern void setTransferGPU(bool);
}

void
display_test_info()
{
  printfQuda("running the following test:\n");
    
  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d         %d\n",
	     get_prec_str(prec),get_prec_str(prec_sloppy),
	     get_recon_str(link_recon), 
	     get_recon_str(link_recon_sloppy),  xdim, ydim, zdim, tdim, Lsdim);     

  printfQuda("   Arpack parameters\n");
  printfQuda(" - number of eigenvectors requested %d\n", eig_nEv);
  printfQuda(" - size of Krylov subspace %d\n", eig_nKv);
  printfQuda(" - solver tolerance %e\n", eig_tol);
  printfQuda(" - Arpack mode %d\n", arpack_mode);
  if(eig_compute_svd) {
    printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
  } else {    
    printfQuda(" - Operator: daggered (%s) , normal-op (%s)\n",
	       eig_use_dagger ? "true" : "false",
	       eig_use_normop ? "true" : "false");
  }
  if(eig_use_poly_acc) {
    printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
    printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
    printfQuda(" - Chebyshev polynomial maximum %e\n", eig_amax);
  }
  printfQuda(" - spectrum requested ");
  if (arpack_spectrum == QUDA_SR_SPECTRUM) printfQuda("Smallest Real\n");
  else if (arpack_spectrum == QUDA_LR_SPECTRUM) printfQuda("Largest Real\n");
  else if (arpack_spectrum == QUDA_SM_SPECTRUM) printfQuda("Smallest Modulus\n");
  else if (arpack_spectrum == QUDA_LM_SPECTRUM) printfQuda("Largest Modulus\n");
  else if (arpack_spectrum == QUDA_SI_SPECTRUM) printfQuda("Smallest Imaginary\n");
  else if (arpack_spectrum == QUDA_LI_SPECTRUM) printfQuda("Largest Imaginary\n");  
  
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3)); 
  return ;
}

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

void setGaugeParam(QudaGaugeParam &gauge_param) {
  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = anisotropy;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;
  
  gauge_param.cpu_prec = cpu_prec;

  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;

  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;

  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = link_recon_precondition;

  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  gauge_param.ga_pad = 0;
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
}

void setInvertParam(QudaInvertParam &inv_param) {

  if (kappa == -1.0) {
    inv_param.mass = mass;
    inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));
  } else {
    inv_param.kappa = kappa;
    inv_param.mass = 0.5/kappa - (1.0 + 3.0/anisotropy);
  }
  
  printfQuda("Kappa = %.8f Mass = %.8f\n", inv_param.kappa, inv_param.mass);

  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_UKQCD_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || 
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 
      2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  //No Even-Odd PC for the moment...
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  inv_param.solve_type = (inv_param.solution_type == QUDA_MAT_SOLUTION ?
			  QUDA_DIRECT_SOLVE : QUDA_DIRECT_PC_SOLVE);
  
  inv_param.matpc_type = matpc_type;
  
  inv_param.inv_type = QUDA_GCR_INVERTER;
  
  inv_param.verbosity = QUDA_VERBOSE;
  
  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.tol = tol;
  
  // require both L2 relative and heavy quark residual to determine 
  // convergence
  inv_param.residual_type = 
    static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  // specify a tolerance for the residual for heavy quark residual
  inv_param.tol_hq = tol_hq; 
  
  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.gcrNkrylov = 10;
  inv_param.pipeline = 10;
  inv_param.reliable_delta = 1e-4;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}


void setArpackParam(QudaArpackParam &arpack_param) {

  arpack_param.arpackMode    = arpack_mode; 
  arpack_param.nEv           = eig_nEv;
  arpack_param.nKv           = eig_nKv;
  arpack_param.spectrum      = arpack_spectrum;
  arpack_param.usePolyAcc    = eig_use_poly_acc ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  arpack_param.polyDeg       = eig_poly_deg;
  arpack_param.amin          = eig_amin;
  arpack_param.amax          = eig_amax;
  arpack_param.arpackTol     = eig_tol;
  arpack_param.arpackMaxiter = eig_maxiter;
  arpack_param.arpackPrec    = prec;
  arpack_param.useNormOp     = eig_use_normop ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  arpack_param.useDagger     = eig_use_dagger ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  arpack_param.SVD           = eig_compute_svd ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  if(eig_compute_svd) {
    warningQuda("Overriding any previous choices of operator type. SVD demands MdagM operator.\n");
    arpack_param.useDagger = QUDA_BOOLEAN_NO;
    arpack_param.useNormOp = QUDA_BOOLEAN_YES;
  }
  
  strcpy(arpack_param.arpackLogfile, arpack_logfile);

}


int main(int argc, char **argv)
{
  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  QudaArpackParam arpack_param = newQudaArpackParam();
  setArpackParam(arpack_param);
  
  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  setDims(gauge_param.X);
  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ?
    sizeof(double) : sizeof(float);

  void *gauge[4], *clover=0, *clover_inv=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    //generate a random SU(3) field
    construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    //generate a unit SU(3) field
    //construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are rands in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = inv_param.clover_cpu_prec;
    clover = malloc(V*cloverSiteSize*cSize);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    if (!compute_clover)
      construct_clover_field(clover, norm, diag, inv_param.clover_cpu_prec);
    
    inv_param.compute_clover = compute_clover;
    if (compute_clover) inv_param.return_clover = 1;
    inv_param.compute_clover_inverse = 1;
    inv_param.return_clover_inverse = 1;
  }

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // this line ensure that if we need to construct the clover inverse (in either the smoother or the solver) we do so
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) loadCloverQuda(clover, clover_inv, &inv_param);

  //Do ARPACK test here
  void *hostEvecs;
  void *hostEvals;
  //Do memory allocations here.
  int vol = xdim*ydim*zdim*tdim;
  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
    hostEvecs = (void*)malloc(vol*24*arpack_param.nKv*sizeof(float));
    hostEvals = (void*)malloc(     2*arpack_param.nKv*sizeof(float));
  } else {
    hostEvecs = (void*)malloc(vol*24*arpack_param.nKv*sizeof(double));
    hostEvals = (void*)malloc(     2*arpack_param.nKv*sizeof(double));
  }

  //This function returns the hostEvecs and hostEvals pointers, populated with the
  //requested data, at the requested prec.
  arpackEigensolveQuda(hostEvecs, hostEvals, &inv_param, &arpack_param, &gauge_param);

  //Access the eigenmode data
  printfQuda("First 10 elements of the eigenvectors and the eigenvalues:\n");    
  for(int i=0; i<arpack_param.nEv; i++) {
    printfQuda("eigenvector %d:\n",i);
    for(int j=0; j<10; j++) {
      if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
	printfQuda("(%e,%e)\n",		   
		   ((float*)hostEvecs)[i*vol*24 + j],
		   ((float*)hostEvecs)[i*vol*24 + j+1]);
      } else {
	printfQuda("(%e,%e)\n",		   
		   ((double*)hostEvecs)[i*vol*24 + j],
		   ((double*)hostEvecs)[i*vol*24 + j+1]);
      }
    }      
    printfQuda("eigenvalue %d = ", i);
    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      printfQuda("(%e,%e)\n",		   
		 ((float*)hostEvals)[i*2],
		 ((float*)hostEvals)[i*2+1]);
    } else {
      printfQuda("(%e,%e)\n",		   
		 ((double*)hostEvals)[i*2],
		 ((double*)hostEvals)[i*2+1]);
    }
  }
  
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

  free(hostEvecs);
  free(hostEvals);
  
  freeGaugeQuda();
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }

  for (int dir = 0; dir<4; dir++) free(gauge[dir]);

  return 0;
}
