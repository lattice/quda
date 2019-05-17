#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <contract_reference.h>
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
#include <color_spinor_field.h> 

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
extern QudaPrecision prec_null;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass;
extern double kappa; // kappa of Dirac operator
extern double mu;
extern double epsilon;
extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern double reliable_delta;
extern char latfile[];
extern int Nsrc; // number of spinors to apply to simultaneously
extern int niter;
extern int gcrNkrylov; // number of inner iterations for GCR, or l for BiCGstab-l
extern int pipeline; // length of pipeline for fused operations in GCR or BiCGstab-l
extern int nvec[];
extern int mg_levels;

extern bool generate_nullspace;
extern bool generate_all_levels;
extern int nu_pre[QUDA_MAX_MG_LEVEL];
extern int nu_post[QUDA_MAX_MG_LEVEL];
extern QudaSolveType coarse_solve_type[QUDA_MAX_MG_LEVEL]; // type of solve to use in the smoothing on each level
extern QudaSolveType smoother_solve_type[QUDA_MAX_MG_LEVEL]; // type of solve to use in the smoothing on each level
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];
extern double mu_factor[QUDA_MAX_MG_LEVEL];

extern QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL];

extern QudaFieldLocation solver_location[QUDA_MAX_MG_LEVEL];
extern QudaFieldLocation setup_location[QUDA_MAX_MG_LEVEL];

extern QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL];
extern int num_setup_iter[QUDA_MAX_MG_LEVEL];
extern double setup_tol[QUDA_MAX_MG_LEVEL];
extern int setup_maxiter[QUDA_MAX_MG_LEVEL];
extern QudaCABasis setup_ca_basis[QUDA_MAX_MG_LEVEL];
extern int setup_ca_basis_size[QUDA_MAX_MG_LEVEL];
extern double setup_ca_lambda_min[QUDA_MAX_MG_LEVEL];
extern double setup_ca_lambda_max[QUDA_MAX_MG_LEVEL];

extern QudaSetupType setup_type;
extern bool pre_orthonormalize;
extern bool post_orthonormalize;
extern double omega;
extern QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL];
extern QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL];
extern double coarse_solver_tol[QUDA_MAX_MG_LEVEL];
extern QudaCABasis coarse_solver_ca_basis[QUDA_MAX_MG_LEVEL];
extern int coarse_solver_ca_basis_size[QUDA_MAX_MG_LEVEL];
extern double coarse_solver_ca_lambda_min[QUDA_MAX_MG_LEVEL];
extern double coarse_solver_ca_lambda_max[QUDA_MAX_MG_LEVEL];

extern double smoother_tol[QUDA_MAX_MG_LEVEL];
extern int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL];

extern QudaPrecision smoother_halo_prec;
extern QudaSchwarzType schwarz_type[QUDA_MAX_MG_LEVEL];
extern int schwarz_cycle[QUDA_MAX_MG_LEVEL];

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern char vec_infile[];
extern char vec_outfile[];

//Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

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

void setInvertParam(QudaInvertParam &inv_param) {  
  inv_param.Ls = 1;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  inv_param.dagger = QUDA_DAG_NO;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.verbosity_precondition = mg_verbosity[0];

  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.pipeline = pipeline;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.tol = tol;

  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType>(QUDA_L2_RELATIVE_RESIDUAL);
  inv_param.tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual

  // these can be set individually
  for (int i=0; i<inv_param.num_offset; i++) {
    inv_param.tol_offset[i] = inv_param.tol;
    inv_param.tol_hq_offset[i] = inv_param.tol_hq;
  }
  inv_param.maxiter = niter;
  inv_param.reliable_delta = reliable_delta;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
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
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  int X[4] = {xdim,ydim,zdim,tdim};  
  setDims(X);  
  setSpinorSiteSize(24);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param); 
  
  size_t sSize = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  void *spinorX = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorY = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *result  = malloc(2*V*16*sSize*inv_param.Ls);
  
  // start the timer
  double time0 = -((double)clock());
  
  // initialize the QUDA library
  initQuda(device);
  
  if (cpu_prec == QUDA_SINGLE_PRECISION) {
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) {
      ((float*)spinorX)[i] = rand() / (float)RAND_MAX;
      ((float*)spinorY)[i] = rand() / (float)RAND_MAX;
    }
  } else {
    for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) {
      ((double*)spinorX)[i] = rand() / (double)RAND_MAX;
      ((double*)spinorY)[i] = rand() / (double)RAND_MAX;
    }
  }

  //Host side spinor data and result passed to QUDA.
  //QUDA will allocate GPU memory, transfer the data,
  //perform the requested contraction, and return the
  //result in the array 'result'
  contractQuda(spinorX, spinorY, result, QUDA_CONTRACT_GAMMA5, QUDA_CONTRACT_GAMMA_I, &inv_param, X);

  //This function will compare each color contraction from the host and device.
  //It returns the number of faults it detects.
  int faults = contraction_reference(spinorX, spinorY, result, QUDA_CONTRACT_GAMMA_I,
				     cpu_prec, X);

  printfQuda("Contraction comparison complete with %d/%d faults\n", faults, V*16*2);
  
  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
  
  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
