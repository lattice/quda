#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include "misc.h"
#include <gauge_field.h>
#include <covdev_reference.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// Various CPU fields lifted from
// staggered_invert_test.cpp

#define mySpinorSiteSize 6
void *qdp_fatlink[4];
void *qdp_longlink[4];  

void *fatlink;
void *longlink;

#ifdef MULTI_GPU
void** ghost_fatlink, **ghost_longlink;
#endif

cpuColorSpinorField* in;
cpuColorSpinorField* out;
cpuColorSpinorField* ref;
cpuColorSpinorField* tmp;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

// For now, only staggered is supported. None of that improved staggered
// goodness yet.
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
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern char latfile[];
extern int Nsrc; // number of spinors to apply to simultaneously
extern int niter;
extern int gcrNkrylov; // number of inner iterations for GCR, or l for BiCGstab-l
extern int pipeline; // length of pipeline for fused operations in GCR or BiCGstab-l
extern int nvec[];
extern int mg_levels;

extern bool generate_nullspace;
extern bool generate_all_levels;
extern int nu_pre;
extern int nu_post;
extern QudaSolveType mg_solve_type[QUDA_MAX_MG_LEVEL];
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];
extern double mu_factor[QUDA_MAX_MG_LEVEL];

extern QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL];

extern QudaFieldLocation solver_location[QUDA_MAX_MG_LEVEL];
extern QudaFieldLocation setup_location[QUDA_MAX_MG_LEVEL];

extern QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL];
extern int num_setup_iter[QUDA_MAX_MG_LEVEL];
extern double setup_tol[QUDA_MAX_MG_LEVEL];
extern int setup_maxiter[QUDA_MAX_MG_LEVEL];
extern QudaSetupType setup_type;
extern bool pre_orthonormalize;
extern bool post_orthonormalize;
extern double omega;
extern QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL];
extern QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL];
extern double coarse_solver_tol[QUDA_MAX_MG_LEVEL];
extern double smoother_tol[QUDA_MAX_MG_LEVEL];
extern int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL];

extern QudaPrecision smoother_halo_prec;
extern QudaSchwarzType schwarz_type[QUDA_MAX_MG_LEVEL];
extern int schwarz_cycle[QUDA_MAX_MG_LEVEL];

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern char vec_infile[];
extern char vec_outfile[];

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

extern bool verify_results;

namespace quda {
  extern void setTransferGPU(bool);
}

template<typename Float>
void constructSpinorField(Float *res) {
  for(int src=0; src<Nsrc; src++) {
    for(int i = 0; i < Vh; i++) {
      for (int s = 0; s < 1; s++) {
        for (int m = 0; m < 3; m++) {
          res[(src*Vh + i)*(1*3*2) + s*(3*2) + m*(2) + 0] = rand() / (Float)RAND_MAX;
          res[(src*Vh + i)*(1*3*2) + s*(3*2) + m*(2) + 1] = rand() / (Float)RAND_MAX;
        }
      }
    }
  }
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

  printfQuda("MG parameters\n");
  printfQuda(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
  printfQuda(" - number of pre-smoother applications %d\n", nu_pre);
  printfQuda(" - number of post-smoother applications %d\n", nu_post);

  printfQuda("Outer solver paramers\n");
  printfQuda(" - pipeline = %d\n", pipeline);

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

  gauge_param.cpu_prec = cpu_prec;    
  gauge_param.cuda_prec = prec;
  gauge_param.reconstruct = link_recon;  
  gauge_param.cuda_prec_sloppy = prec_sloppy;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.anisotropy = 1.0;
  const double tadpole_coeff = 0.8; // hard coded in staggered_invert_test
  gauge_param.tadpole_coeff = tadpole_coeff; 

  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gauge_param.scale = dslash_type != QUDA_ASQTAD_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;
  gauge_param.type = QUDA_WILSON_LINKS;

  // QUDA_QDP_GAUGE_ORDER causes a segfault...
  gauge_param.gauge_order = QUDA_MILC_GAUGE_ORDER;
  
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

void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;//this will be used to setup SolverParam parent in MGParam class

  for (int i=0; i<mg_levels-1; i++) printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);

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

  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (4.0 + inv_param.mass));

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.is_staggered = QUDA_BOOLEAN_YES;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    mg_param.verbosity[i] = mg_verbosity[i];
    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];
    mg_param.setup_tol[i] = setup_tol[i];
    mg_param.setup_maxiter[i] = setup_maxiter[i];
    mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = nvec[i] == 0 ? 96 : nvec[i]; // default to 96 vectors if not set
    mg_param.precision_null[i] = prec_null; // precision to store the null-space basis
    mg_param.smoother_halo_precision[i] = smoother_halo_prec; // precision of the halo exchange in the smoother
    mg_param.nu_pre[i] = nu_pre;
    mg_param.nu_post[i] = nu_post;
    mg_param.mu_factor[i] = mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];

    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i];

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    mg_param.smoother_solve_type[i] = mg_solve_type[i]; // EVEN-ODD

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = schwarz_cycle[i];

    // Set set coarse_grid_solution_type: this defines which linear
    // system we are solving on a given level
    // * QUDA_MAT_SOLUTION - we are solving the full system and inject
    //   a full field into coarse grid
    // * QUDA_MATPC_SOLUTION - we are solving the e/o-preconditioned
    //   system, and only inject single parity field into coarse grid
    //
    // Multiple possible scenarios here
    //
    // 1. **Direct outer solver and direct smoother**: here we use
    // full-field residual coarsening, and everything involves the
    // full system so coarse_grid_solution_type = QUDA_MAT_SOLUTION
    //
    // 2. **Direct outer solver and preconditioned smoother**: here,
    // only the smoothing uses e/o preconditioning, so
    // coarse_grid_solution_type = QUDA_MAT_SOLUTION_TYPE.
    // We reconstruct the full residual prior to coarsening after the
    // pre-smoother, and then need to project the solution for post
    // smoothing.
    //
    // 3. **Preconditioned outer solver and preconditioned smoother**:
    // here we use single-parity residual coarsening throughout, so
    // coarse_grid_solution_type = QUDA_MATPC_SOLUTION.  This is a bit
    // questionable from a theoretical point of view, since we don't
    // coarsen the preconditioned operator directly, rather we coarsen
    // the full operator and preconditioned that, but it just works.
    // This is the optimal combination in general for Wilson-type
    // operators: although there is an occasional increase in
    // iteration or two), by working completely in the preconditioned
    // space, we save the cost of reconstructing the full residual
    // from the preconditioned smoother, and re-projecting for the
    // subsequent smoother, as well as reducing the cost of the
    // ancillary blas operations in the coarse-grid solve.
    //
    // Note, we cannot use preconditioned outer solve with direct
    // smoother
    //
    // Finally, we have to treat the top level carefully: for all
    // other levels the entry into and out of the grid will be a
    // full-field, which we can then work in Schur complement space or
    // not (e.g., freedom to choose coarse_grid_solution_type).  For
    // the top level, if the outer solver is for the preconditioned
    // system, then we must use preconditoning, e.g., option 3.) above.

    if (i == 0) { // top-level treatment
      if (coarse_solve_type[0] != solve_type)
        errorQuda("Mismatch between top-level MG solve type %d and outer solve type %d", coarse_solve_type[0], solve_type);

      if (solve_type == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (solve_type == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", solve_type);
      }

    } else {

      if (coarse_solve_type[i] == QUDA_DIRECT_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MAT_SOLUTION;
      } else if (coarse_solve_type[i] == QUDA_DIRECT_PC_SOLVE) {
        mg_param.coarse_grid_solution_type[i] = QUDA_MATPC_SOLUTION;
      } else {
        errorQuda("Unexpected solve_type = %d\n", coarse_solve_type[i]);
      }

    }
    
    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = solver_location[i];
    mg_param.setup_location[i] = setup_location[i];
  }

  // ESW hack
  /*mg_param.coarse_grid_solution_type[0] = QUDA_MAT_SOLUTION;
  mg_param.coarse_grid_solution_type[1] = QUDA_MATPC_SOLUTION;
  mg_param.coarse_grid_solution_type[2] = QUDA_MAT_SOLUTION;
  mg_param.coarse_grid_solution_type[3] = QUDA_MAT_SOLUTION;*/

  // coarsening the spin on the first restriction is undefined for staggered fields.
  mg_param.spin_block_size[0] = 0;

  mg_param.setup_type = setup_type;
  mg_param.pre_orthonormalize = pre_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;
  mg_param.post_orthonormalize = post_orthonormalize ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES
    : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = verify_results ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

  // set file i/o parameters
  strcpy(mg_param.vec_infile, vec_infile);
  strcpy(mg_param.vec_outfile, vec_outfile);

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 1000;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 10;

  //inv_param.verbosity = QUDA_SUMMARIZE;
  inv_param.verbosity = QUDA_VERBOSE;
  //inv_param.verbosity_precondition = QUDA_SUMMARIZE;
  inv_param.verbosity_precondition = QUDA_VERBOSE;
}

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

  inv_param.mass = mass;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.dagger = QUDA_DAG_NO;

  // no need to set clover params

  // do we want full solution or single-parity solution
  inv_param.solution_type = (solve_type == QUDA_DIRECT_PC_SOLVE) ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = QUDA_VERBOSE;
  //inv_param.verbosity_precondition = mg_verbosity[0];
  inv_param.verbosity_precondition = QUDA_SUMMARIZE; // ESW HACK

  /* ESW HACK: comment this out to do a non-MG solve. */
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
  inv_param.reliable_delta = 1e-4;

  // domain decomposition preconditioner parameters
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 1;
  inv_param.omega = 1.0;
}

int main(int argc, char **argv)
{
  // We give here the default values to some of the array
  for(int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_SUMMARIZE;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    setup_tol[i] = 5e-6;
    setup_maxiter[i] = 500;
    mu_factor[i] = 1.;
    mg_solve_type[i] = QUDA_INVALID_SOLVE;
    schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_GCR_INVERTER; 
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 100;
    solver_location[i] = QUDA_CUDA_FIELD_LOCATION;
    setup_location[i] = QUDA_CUDA_FIELD_LOCATION;
  }

  // Give the dslash type a reasonable default.
  dslash_type = QUDA_STAGGERED_DSLASH;

  for (int i = 1; i < argc; i++){
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }
    printf("ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (smoother_halo_prec == QUDA_INVALID_PRECISION) smoother_halo_prec = prec_null;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;
  for(int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    if (mg_solve_type[i] == QUDA_INVALID_SOLVE) mg_solve_type[i] = solve_type;
  }


  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  // Need to add support for ASQTAD, LAPLACE
  if (dslash_type != QUDA_STAGGERED_DSLASH) { // &&
      //dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      //dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      //dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  // ESW HACK: needs to be addressed
  if (solve_type == QUDA_DIRECT_PC_SOLVE || mg_solve_type[0] == QUDA_DIRECT_PC_SOLVE) {
    printfQuda("staggered_multigtid_invert_test doesn't support preconditioned outer solve yet.\n");
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  setGaugeParam(gauge_param);
  setInvertParam(inv_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;

  setMultigridParam(mg_param);

  // start the timer
  double time0 = -((double)clock());

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.


  setDims(gauge_param.X);

  setSpinorSiteSize(6);


  /* Taken from staggered_invert_test to load gauge fields */


  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for (int dir = 0; dir < 4; dir++) {
    qdp_fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  fatlink = malloc(4*V*gaugeSiteSize*gSize);
  longlink = malloc(4*V*gaugeSiteSize*gSize);


  void *inlinks=malloc(4*V*gaugeSiteSize*gSize);
  void *inlink[4];

  for (int dir = 0; dir < 4; dir++) {
     inlink[dir] = (void*)((char*)inlinks + dir*V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {
    read_gauge_field(latfile, inlink, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    applyGaugeFieldScaling_long(inlink, Vh, &gauge_param, dslash_type, gauge_param.cpu_prec);
    //construct_fat_long_gauge_field(inlink, nullptr, 3, gauge_param.cpu_prec, &gauge_param, dslash_type);
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], inlink[dir] , V*gaugeSiteSize*gSize);
    }

  } else {

    const int gen_type = 1; // free field.
    warningQuda("Gen type = %d", gen_type);

    construct_fat_long_gauge_field(qdp_fatlink, qdp_longlink, gen_type, gauge_param.cpu_prec, 
         &gauge_param, dslash_type);
  }

  free(inlinks);

  //if (dslash_type == QUDA_LAPLACE_DSLASH) {
  // Force unit gauge fields without phases just for convenience */
  //  construct_gauge_field(qdp_fatlink, 0, gauge_param.cpu_prec, &gauge_param);
  //} else {
    // hard coded plus hacked to force free field with staggered phases
    /*construct_fat_long_gauge_field(qdp_fatlink, qdp_longlink, 1, gauge_param.cpu_prec,
           &gauge_param, dslash_type);*/
  //}
  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<V; ++i){
      for(int j=0; j<gaugeSiteSize; ++j){
        if(gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION){
          ((double*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)qdp_fatlink[dir])[i*gaugeSiteSize + j];
          ((double*)longlink)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)qdp_longlink[dir])[i*gaugeSiteSize + j];
        }else{
          ((float*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)qdp_fatlink[dir])[i*gaugeSiteSize + j];
          ((float*)longlink)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)qdp_longlink[dir])[i*gaugeSiteSize + j];
        }
      }
    }
  }


  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  for (int d = 0; d < 4; d++) csParam.x[d] = gauge_param.X[d];
  bool pc = (inv_param.solution_type == QUDA_MATPC_SOLUTION || inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) csParam.x[0] /= 2;
  //csParam.x[4] = Nsrc;

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;
  csParam.siteSubset = pc ? QUDA_PARITY_SITE_SUBSET : QUDA_FULL_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;  
  in = new cpuColorSpinorField(csParam);  
  out = new cpuColorSpinorField(csParam);  
  ref = new cpuColorSpinorField(csParam);  
  tmp = new cpuColorSpinorField(csParam);  

  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
    constructSpinorField((float*)in->V());    
  }else{
    constructSpinorField((double*)in->V());
  }

#ifdef MULTI_GPU
  int tmp_value = MAX(ydim*zdim*tdim/2, xdim*zdim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*zdim/2);

  int fat_pad = tmp_value;
  int link_pad =  3*tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(fatlink, gauge_param);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(longlink, gauge_param);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();

#else
  int fat_pad = 0;
  int link_pad = 0;
#endif
  
  gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gauge_param.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
  } else {
    gauge_param.reconstruct= gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gauge_param.cuda_prec_precondition = gauge_param.cuda_prec_sloppy;
  gauge_param.reconstruct_precondition = gauge_param.reconstruct_sloppy;
  loadGaugeQuda(fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.reconstruct= link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.cuda_prec_precondition = gauge_param.cuda_prec_sloppy;
    gauge_param.reconstruct_precondition = gauge_param.reconstruct_sloppy;
    loadGaugeQuda(longlink, &gauge_param);
  }

  /* end stuff stolen from staggered_invert_test */

  //if (mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE || solve_type == QUDA_DIRECT_PC_SOLVE) inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
  inv_param.solve_type = solve_type; // restore actual solve_type we want to do

  /* ESW HACK: comment this out to do a non-MG solve. */

  // setup the multigrid solver
  void *mg_preconditioner = newMultigridQuda(&mg_param);
  inv_param.preconditioner = mg_preconditioner;

  // Test: create a dummy invert param just to make sure
  // we're setting up gauge fields and such correctly.



  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
    invertQuda((float*)out->V(), (float*)in->V(), &inv_param);
  }else{
    invertQuda((double*)out->V(), (double*)in->V(), &inv_param);
  }

  warningQuda("Got out of inverter, need to verify.\n");
  fflush(stdout);

  // free the multigrid solver
  destroyMultigridQuda(mg_preconditioner);

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;

    
  //printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
  //inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);
  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, 0.0);
  fflush(stdout);

  // I haven't been able to get this verify working,
  // and I'm not going to try for now. The issue is I need
  // to check both parities.
  /* 
  
  // Prepare for checks and such.
  double nrm2=0;
  double src2=0;
  double l2r=0;

  int len = V;//Vh*Nsrc;

  double factor = 1.0 / (2.0*mass);

#ifdef MULTI_GPU    
  //mat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, in, factor, QUDA_DAG_NO, inv_param.cpu_prec, gauge_param.cpu_prec, &inv_param);

  // Need to do both even and odd...
  staggered_dslash_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, in, 0, 0, inv_param.cpu_prec, gauge_param.cpu_prec);

#else
  mat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), factor, 0, inv_param.cpu_prec, gauge_param.cpu_prec);
#endif

  printfQuda("\nCompleted reference dslash.\n");
  fflush(stdout);

  ax( factor, in->V(), len*mySpinorSiteSize, inv_param.cpu_prec );

  int vol = inv_param.solution_type == QUDA_MAT_SOLUTION ? V : Vh;
  mxpy(in->V(), ref->V(), vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  nrm2 = norm_2(ref->V(), vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  src2 = norm_2(in->V(), vol*spinorSiteSize*inv_param.Ls, inv_param.cpu_prec);
  l2r = sqrt(nrm2 / src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g\n",
	     inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq);
  fflush(stdout);

  */ 
  // End I'm lazy.

  // Clean up cpu fields
  for(int i=0;i < 4;i++){
    free(qdp_fatlink[i]);
    free(qdp_longlink[i]);
  }

  printfQuda("free qdp links\n"); fflush(stdout);

  free(fatlink);
  printfQuda("free(fatlink)\n"); fflush(stdout);
  free(longlink);
  printfQuda("free(longlink)\n"); fflush(stdout);

  delete in;
  printfQuda("delete in\n"); fflush(stdout);
  delete out;
  printfQuda("delete out\n"); fflush(stdout);
  delete ref;
  printfQuda("delete ref\n"); fflush(stdout);
  delete tmp;
  printfQuda("delete tmp\n"); fflush(stdout);

  if (cpuFat) { delete cpuFat; printfQuda("delete cpuFat\n"); fflush(stdout); }
  if (cpuLong) { delete cpuLong; printfQuda("delete cpuLong\n"); fflush(stdout); }

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
