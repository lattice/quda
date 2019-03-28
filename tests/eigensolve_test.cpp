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

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;
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
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass;
extern double kappa; // kappa of Dirac operator
extern double mu;
extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern char latfile[];
extern bool unit_gauge;
extern int Nsrc; // number of spinors to apply to simultaneously
extern int niter;
extern int gcrNkrylov; // number of inner iterations for GCR, or l for BiCGstab-l
extern int pipeline; // length of pipeline for fused operations in GCR or BiCGstab-l

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

//Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

extern int eig_nEv;
extern int eig_nKr;
extern int eig_nConv;
extern int eig_check_interval;
extern int eig_max_restarts;
extern double eig_tol;
extern int eig_maxiter;
extern bool eig_use_poly_acc;
extern int eig_poly_deg;
extern double eig_amin;
extern double eig_amax;
extern bool eig_use_normop;
extern bool eig_use_dagger;
extern bool eig_compute_svd;
extern QudaEigSpectrumType eig_spectrum;
extern QudaEigType eig_type;
extern bool eig_arpack_check;
extern char eig_arpack_logfile[];
extern char eig_QUDA_logfile[];
extern char eig_vec_infile[];
extern char eig_vec_outfile[];

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

  printfQuda("\n   Eigensolver parameters\n");
  printfQuda(" - solver mode %s\n", get_eig_type_str(eig_type));
  printfQuda(" - spectrum requested %s\n", get_eig_spectrum_str(eig_spectrum));
  printfQuda(" - number of eigenvectors requested %d\n", eig_nConv);
  printfQuda(" - size of eigenvector search space %d\n", eig_nEv);
  printfQuda(" - size of Krylov space %d\n", eig_nKr);
  printfQuda(" - solver tolerance %e\n", eig_tol);
  if(eig_compute_svd) {
    printfQuda(" - Operator: MdagM. Will compute SVD of M\n");
    printfQuda(" - ***********************************************************\n");
    printfQuda(" - **** Overriding any previous choices of operator type. ****\n");
    printfQuda(" - ****    SVD demands normal operator, will use MdagM    ****\n");
    printfQuda(" - ***********************************************************\n");
  } else {    
    printfQuda(" - Operator: daggered (%s) , norm-op (%s)\n",
	       eig_use_dagger ? "true" : "false",
	       eig_use_normop ? "true" : "false");
  }
  if(eig_use_poly_acc) {
    printfQuda(" - Chebyshev polynomial degree %d\n", eig_poly_deg);
    printfQuda(" - Chebyshev polynomial minumum %e\n", eig_amin);
    printfQuda(" - Chebyshev polynomial maximum %e\n\n", eig_amax);
  }
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

//Parameters defining the eigensolver
void setEigParam(QudaEigParam &eig_param) {

  eig_param.eig_type = eig_type;
  eig_param.spectrum = eig_spectrum;
  
  eig_param.nConv   = eig_nConv;
  eig_param.nEv     = eig_nEv;
  eig_param.nKr     = eig_nKr;
  eig_param.tol     = eig_tol;
  eig_param.check_interval = eig_check_interval;
  eig_param.max_restarts = eig_max_restarts;
  eig_param.cuda_prec_ritz = cuda_prec;

  eig_param.use_norm_op = eig_use_normop ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.use_dagger  = eig_use_dagger ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.compute_svd = eig_compute_svd ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  if(eig_compute_svd) {
    eig_param.use_dagger  = QUDA_BOOLEAN_NO;
    eig_param.use_norm_op = QUDA_BOOLEAN_YES;
  }
  
  eig_param.use_poly_acc = eig_use_poly_acc ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  eig_param.poly_deg = eig_poly_deg;
  eig_param.a_min    = eig_amin;
  eig_param.a_max    = eig_amax;

  eig_param.arpack_check = eig_arpack_check ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;
  strcpy(eig_param.arpack_logfile, eig_arpack_logfile);
  strcpy(eig_param.QUDA_logfile, eig_QUDA_logfile);

  strcpy(eig_param.vec_infile, eig_vec_infile);
  strcpy(eig_param.vec_outfile, eig_vec_outfile);
  
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

  QudaEigParam eig_param = newQudaEigParam();
  //Though no inversions are performed, the inv_param
  //structure contains all the information we need to
  //construct the dirac operator. We encapsualte the
  //inv_param structure inside the eig_param structure
  //to avoid any confusion
  QudaInvertParam eig_inv_param = newQudaInvertParam();
  setInvertParam(eig_inv_param);
  eig_param.invert_param = &eig_inv_param;
  setEigParam(eig_param);
  
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
  } else { // else generate an SU(3) field
    if(unit_gauge){
      //unit SU(3) field
      construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    } else {
      //random SU(3) field
      construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    }
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are rands in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = eig_inv_param.clover_cpu_prec;
    clover = malloc(V*cloverSiteSize*cSize);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    if (!compute_clover)
      construct_clover_field(clover, norm, diag, eig_inv_param.clover_cpu_prec);
    
    eig_inv_param.compute_clover = compute_clover;
    if (compute_clover) eig_inv_param.return_clover = 1;
    eig_inv_param.compute_clover_inverse = 1;
    eig_inv_param.return_clover_inverse = 1;
  }

  // initialize the QUDA library
  initQuda(device);

  // load the gauge field
  loadGaugeQuda((void*)gauge, &gauge_param);

  // this line ensure that if we need to construct the clover inverse
  // (in either the smoother or the solver) we do so
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
      dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    loadCloverQuda(clover, clover_inv, &eig_inv_param);
  }
  
  // QUDA eigensolver test 
  //----------------------------------------------------------------------------

  //Host side arrays to store the eigenvalues and vectors
  void *host_evecs;
  void *host_evals;

  //Memory allocation
  int vol = xdim*ydim*zdim*tdim;
  if (eig_inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
    host_evecs = (void*)malloc(vol*24*eig_param.nEv*sizeof(float));
    host_evals = (void*)malloc(     2*eig_param.nEv*sizeof(float));
  } else {
    host_evecs = (void*)malloc(vol*24*eig_param.nEv*sizeof(double));
    host_evals = (void*)malloc(     2*eig_param.nEv*sizeof(double));
  }
  
  //Demonstrate how to use inital guess. The vector will be
  //normalised in the eigensolve function.
  for(int i=0; i<vol*24; i++) {
    if(eig_inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) {
      
      ((double*)host_evecs)[i] = rand()/(double)RAND_MAX;
      //if(i<24) printfQuda("Host elem %d = %f\n", i, ((double*)host_evecs)[i]);
    } else {
      
      ((float*)host_evecs)[i] = rand()/(float)RAND_MAX;      
      //if(i<24) printfQuda("Host elem %d = %f\n", i, ((float*)host_evecs)[i]);
    }
  }
    
  //This function returns the host_evecs and host_evals pointers, populated with the
  //requested data, at the requested prec. All the information needed to perfom the
  //solve is in the eig_param container.
  double timeQUDA = -((double)clock());
  eigensolveQuda(host_evecs, host_evals, &eig_param);
  timeQUDA += (double)clock();
  printfQuda("Time for QUDA solution = %f\n", timeQUDA/CLOCKS_PER_SEC);

  //Deallocate host memory
  free(host_evecs);
  free(host_evals);
  
  if(eig_param.arpack_check) {
#ifdef ARPACK_LIB
    if(eig_inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) {
      //Perform a cross-check using the ARPACK interface
      //Use a different initial guess and reallocate memory
      
      host_evecs = (void*)malloc(vol*24*eig_param.nKr*sizeof(double));
      host_evals = (void*)malloc(     2*eig_param.nKr*sizeof(double));
      
      for(int i=0; i<vol*24; i++) {
	((double*)host_evecs)[i] = rand()/(double)RAND_MAX;
	if(i<24) printfQuda("Host elem %d = %f\n", i, ((double*)host_evecs)[i]);
      }
      
      double timeARPACK = -((double)clock());
      eigensolveARPACK(host_evecs, host_evals, &eig_param);
      timeARPACK += (double)clock();
      //Deallocate host memory
      free(host_evecs);
      free(host_evals);
      
      printfQuda("Time for ARPACK solution = %f\n\n", timeARPACK/CLOCKS_PER_SEC);
      printfQuda("************************************************\n");
      printfQuda("     Speed-up for QUDA Vs ARPACK is x%.1f       \n",
		 (timeARPACK/CLOCKS_PER_SEC)/(timeQUDA/CLOCKS_PER_SEC));
      printfQuda("************************************************\n\n");
    } else {
      errorQuda("Single prec ARPACK solves not supported. Rerun with double prec");
    }
#else
    errorQuda("ARPACK interface not built.");
#endif
  }
  
  
  freeGaugeQuda();  
  if(dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
     dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {    
    freeCloverQuda();    
  }
  
  // finalize the QUDA library
  endQuda();
  
  // finalize the communications layer
  finalizeComms();
  
  if(dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
     dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    if (clover) free(clover);
    if (clover_inv) free(clover_inv);
  }
  for (int dir = 0; dir<4; dir++) free(gauge[dir]);
  
  return 0;
}
