#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <random>

#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include <quda.h>
#include <string.h>
#include <face_quda.h>
#include "misc.h"
#include <gauge_field.h>
#include <blas_quda.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#ifdef MULTI_GPU
#include <face_quda.h>
#endif

#define MAX(a,b) ((a)>(b)?(a):(b))
#define mySpinorSiteSize 6

extern void usage(char** argv);
void *qdp_fatlink[4];
void *qdp_longlink[4];  

void *fatlink;
void *longlink;

#ifdef MULTI_GPU
void** ghost_fatlink, **ghost_longlink;
#endif

extern bool generate_nullspace;
extern char vec_infile[];
extern char vec_outfile[];

extern int device;
extern bool tune;

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;
extern QudaReconstructType link_recon_precondition;
extern QudaPrecision  prec_precondition;
extern double mass;
extern double mu;


//QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;

cpuColorSpinorField* in;
cpuColorSpinorField* out;
cpuColorSpinorField* ref;
cpuColorSpinorField* tmp;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern int test_type;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];

// Dirac operator type
extern QudaDslashType dslash_type;

extern QudaInverterType inv_type;
extern double mass; // the mass of the Dirac operator

extern char latfile[];
extern int niter;
extern int nvec[];
extern int mg_levels;

extern bool generate_nullspace;
extern bool generate_all_levels;
extern int nu_pre;
extern int nu_post;
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];

extern QudaInverterType precon_type;

 //Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

namespace quda {
  extern void setTransferGPU(bool);
}

static void end();

template<typename Float>
void constructSpinorField(Float *res, const int Vol) {
  for(int i = 0; i < Vol; i++) {
    for (int s = 0; s < 1; s++) {
      for (int m = 0; m < 3; m++) {
        res[i*(1*3*2) + s*(3*2) + m*(2) + 0] = rand() / (Float)RAND_MAX;
        res[i*(1*3*2) + s*(3*2) + m*(2) + 1] = rand() / (Float)RAND_MAX;
      }
    }
  }
}

#define USE_QDP_LINKS

void setGaugeParam(QudaGaugeParam &gaugeParam, double tadpole_coeff) {

  gaugeParam.X[0] = xdim;
  gaugeParam.X[1] = ydim;
  gaugeParam.X[2] = zdim;
  gaugeParam.X[3] = tdim;

  gaugeParam.cpu_prec = cpu_prec;    
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;  
  gaugeParam.cuda_prec_sloppy = prec_sloppy;
  gaugeParam.reconstruct_sloppy = link_recon_sloppy;
  gaugeParam.reconstruct_precondition = link_recon_precondition;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam.anisotropy = 1.0;
  gaugeParam.tadpole_coeff = tadpole_coeff;

  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gaugeParam.scale = dslash_type == QUDA_STAGGERED_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  gaugeParam.t_boundary = QUDA_PERIODIC_T;//QUDA_ANTI_PERIODIC_T;
#ifndef USE_QDP_LINKS
  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;
#else
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
#endif
  gaugeParam.ga_pad = 0;
  // For multi-GPU, ga_pad must be large enough to store a time-slice
#ifdef MULTI_GPU
  int x_face_size = gaugeParam.X[1]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int y_face_size = gaugeParam.X[0]*gaugeParam.X[2]*gaugeParam.X[3]/2;
  int z_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[3]/2;
  int t_face_size = gaugeParam.X[0]*gaugeParam.X[1]*gaugeParam.X[2]/2;
  int pad_size = std::max(x_face_size, y_face_size);
  pad_size = std::max(pad_size, z_face_size);
  pad_size = std::max(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif
}


void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;

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

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  //Free field!
  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (4 + mass));

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;//fixed

  inv_param.solve_type = QUDA_DIRECT_SOLVE;//fixed

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    mg_param.spin_block_size[i] = 1;// 1 or 0 (1 for parity blocking)
    mg_param.n_vec[i] =  nvec[i] == 0 ? 24 : nvec[i]; // default to 24 vectors if not set
    mg_param.nu_pre[i] = nu_pre;
    mg_param.nu_post[i] = nu_post;

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_VCYCLE;//QUDA_MG_CYCLE_RECURSIVE;

    mg_param.smoother[i] = precon_type;

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = tol_hq; // repurpose heavy-quark tolerance for now

    mg_param.global_reduction[i] = QUDA_BOOLEAN_YES;

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    //mg_param.smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE; // EVEN-ODD
    mg_param.smoother_solve_type[i] = QUDA_DIRECT_SOLVE; 

    // set to QUDA_MAT_SOLUTION to inject a full field into coarse grid
    // set to QUDA_MATPC_SOLUTION to inject single parity field into coarse grid

    // if we are using an outer even-odd preconditioned solve, then we
    // use single parity injection into the coarse grid
    mg_param.coarse_grid_solution_type[i] = (mg_param.smoother_solve_type[i] == QUDA_DIRECT_PC_SOLVE || mg_param.smoother_solve_type[i] == QUDA_NORMOP_PC_SOLVE) ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;//QUDA_MATPCDAG_MATPC_SOLUTION

    mg_param.omega[i] = 0.85; // over/under relaxation factor

    mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;
    //mg_param.location[i] = QUDA_CPU_FIELD_LOCATION;
  }

  mg_param.smoother_solve_type[0] = solve_type == QUDA_NORMOP_PC_SOLVE? QUDA_NORMOP_PC_SOLVE : mg_param.smoother_solve_type[0]; //or choose QUDA_DIRECT_SOLVE;
  //mg_param.smoother_solve_type[0] = QUDA_NORMOP_PC_SOLVE;//enforce PC solve
  // coarsen the spin on the first restriction is undefined for staggered fields
  mg_param.spin_block_size[0] = 0;

  if(mg_param.smoother_solve_type[0] == QUDA_NORMOP_PC_SOLVE || mg_param.smoother_solve_type[0] == QUDA_DIRECT_PC_SOLVE) 
  { 
     //mg_param.smoother[0] = QUDA_CG_INVERTER; //or choose QUDA_GCR_INVERTER
     mg_param.smoother[0] = QUDA_MR_INVERTER; //or choose QUDA_GCR_INVERTER (WARNING: QUDA_MR_INVERTER works better...)
     mg_param.coarse_grid_solution_type[0] = QUDA_MATPC_SOLUTION;//just to be safe.
     ////mg_param.coarse_grid_solution_type[0] = QUDA_MAT_SOLUTION;//remove
  } 
  //
  mg_param.null_solve_type = QUDA_DIRECT_SOLVE;
  //mg_param.null_solve_type = mg_param.smoother_solve_type[0] != QUDA_DIRECT_SOLVE ? mg_param.smoother_solve_type[0] : QUDA_NORMOP_PC_SOLVE;

  //number of the top-level smoothing:
  mg_param.nu_pre[0] = nu_pre;
  mg_param.nu_post[0] = nu_post;

  // coarse grid solver is GCR
  //mg_param.smoother[mg_levels-1] = QUDA_CG_INVERTER; //QUDA_GCR_INVERTER;
  mg_param.smoother[mg_levels-1] = QUDA_GCR_INVERTER;

//  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_NULL_VECTOR_NO;
  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES : QUDA_COMPUTE_LOW_MODE_VECTOR;
  //mg_param.compute_null_vector = QUDA_COMPUTE_LOW_MODE_VECTOR;

  if (mg_param.compute_null_vector == QUDA_COMPUTE_LOW_MODE_VECTOR) 
    mg_param.eigensolver_precision = cuda_prec;//??
  else
    mg_param.eigensolver_precision = cuda_prec;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_YES 
   :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = QUDA_BOOLEAN_YES;

  // set file i/o parameters
  strcpy(mg_param.vec_infile, vec_infile);
  strcpy(mg_param.vec_outfile, vec_outfile);

  // these need to tbe set for now but are actually ignored by the MG setup
  // needed to make it pass the initialization test
  inv_param.inv_type = QUDA_GCR_INVERTER;
  inv_param.tol = 1e-10;
  inv_param.maxiter = 100;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 16;//10

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

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  //Free field!
  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (4 + mass));

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  // do we want full solution or single-parity solution
// inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;  //QUDA_MAT_SOLUTION;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;//QUDA_NORMOP_PC_SOLVE?
  inv_param.matpc_type = matpc_type;//QUDA_MATPC_EVEN_EVEN

  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = QUDA_VERBOSE;
  //inv_param.verbosity_precondition = QUDA_SUMMARIZE;
  inv_param.verbosity_precondition = QUDA_VERBOSE;

  inv_param.inv_type_precondition = QUDA_MG_INVERTER;//MR->GCR
  inv_param.gcrNkrylov = 24;
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
  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;
//needed to test the top level error bahaviour
  //inv_param.use_init_guess = QUDA_USE_INIT_GUESS_YES;
  //inv_param.compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;

  return;
}

#define CHECK_PLAQUETTE

void mg_test(int argc, char** argv)
{

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param, 0.8);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;

  setMultigridParam(mg_param);


  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);
  //////////////////////////////////////////////////////////////
  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gauge_param.X);
  setSpinorSiteSize(6);

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

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, inlink, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    //construct_gauge_field(inlink, 2, gauge_param.cpu_prec, &gauge_param);
    construct_fat_long_gauge_field(inlink, nullptr, 3, gauge_param.cpu_prec, &gauge_param, dslash_type);
    QudaGaugeParam gParam = newQudaGaugeParam();
#ifdef CHECK_PLAQUETTE
    gParam.gauge_order = QUDA_QDP_GAUGE_ORDER ;

    gParam.X[0] = xdim;
    gParam.X[1] = ydim;
    gParam.X[2] = zdim;
    gParam.X[3] = tdim;

    gParam.anisotropy = 1.0;
    gParam.type = QUDA_WILSON_LINKS;
    gParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
    gParam.t_boundary = QUDA_PERIODIC_T;

    gParam.cpu_prec = cpu_prec;
    gParam.cuda_prec = cuda_prec;
    gParam.reconstruct = link_recon;
    gParam.cuda_prec_sloppy = cuda_prec_sloppy;
    gParam.reconstruct_sloppy = link_recon_sloppy;
    gParam.cuda_prec_precondition = cuda_prec_precondition;
    gParam.reconstruct_precondition = link_recon_precondition;
    gParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
    gParam.ga_pad = 0;

#ifdef MULTI_GPU
    int x_face_size = gParam.X[1]*gParam.X[2]*gParam.X[3]/2;
    int y_face_size = gParam.X[0]*gParam.X[2]*gParam.X[3]/2;
    int z_face_size = gParam.X[0]*gParam.X[1]*gParam.X[3]/2;
    int t_face_size = gParam.X[0]*gParam.X[1]*gParam.X[2]/2;
    int pad_size = std::max(x_face_size, y_face_size);
    pad_size = std::max(pad_size, z_face_size);
    pad_size = std::max(pad_size, t_face_size);
    gParam.ga_pad = pad_size;
#endif
    loadGaugeQuda( (void*)inlink,  &gParam );
    printfQuda("\nLoaded fields\n");

    double plq[3];
    plaqQuda (plq); 
    freeGaugeQuda();

    printfQuda("\nPlaq : %le, %le, %le\n", plq[0], plq[1], plq[2]);
#endif   
//see ./milc_qcd/generic_ks/fermion_links_fn_load_gpu.c
//see ./milc_qcd/generic_ks/fermion_links_fn_load_milc.c
// 'load_fn_links' ->  'load_imp_ferm_links' ->./milc_qcd/generic_ks/fermion_links_milc.c 
#ifdef MULTI_GPU
    QudaComputeFatMethod method = QUDA_COMPUTE_FAT_EXTENDED_VOLUME;
#else
    QudaComputeFatMethod method = QUDA_COMPUTE_FAT_STANDARD;
#endif

    for(int dir=0; dir<4; ++dir) gParam.X[dir] = gauge_param.X[dir];
    gParam.cuda_prec_sloppy = gParam.cpu_prec = gParam.cuda_prec = gauge_param.cpu_prec;
    gParam.type = QUDA_GENERAL_LINKS;

    gParam.reconstruct_sloppy = gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.gauge_order   = QUDA_MILC_GAUGE_ORDER;
    gParam.t_boundary    = QUDA_PERIODIC_T;
    gParam.gauge_fix     = QUDA_GAUGE_FIXED_NO;
    gParam.scale         = 1.0;
    gParam.anisotropy    = 1.0;
    gParam.tadpole_coeff = 1.0;
    gParam.scale         = 0;
    gParam.ga_pad        = 0;
    gParam.site_ga_pad   = 0;
    gParam.mom_ga_pad    = 0;
    gParam.llfat_ga_pad  = 0;

    double act_path_coeff[6];

    //fake parameters:
    for(int i = 0;i < 6;i++){
      act_path_coeff[i]= 0.1*i;
    }
//mass = 0.0102;l24t64
//mass = 0.002426
    act_path_coeff[0] = 1.0;
    act_path_coeff[1] = -4.166667e-02;
    act_path_coeff[2] = -6.250000e-02;
    act_path_coeff[3] = 1.562500e-02;
    act_path_coeff[4] = -2.604167e-03;
    act_path_coeff[5] = -1.250000e-01;


    //computeKSLinkQuda( fatlink, longlink, NULL, inlinks, const_cast<double*>(act_path_coeff), &gParam, method);
#ifdef USE_QDP_LINKS 
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], inlink[dir] , V*gaugeSiteSize*gSize);
    }
#else    
    for(int dir=0; dir<4; ++dir){
      double check_sum = 0.0;
      for(int i=0; i<V; ++i){
        for(int j=0; j<gaugeSiteSize; ++j){
          if(gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION){
            ((double*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)inlink[dir])[i*gaugeSiteSize + j];
            check_sum += ((double*)fatlink)[(i*4 + dir)*gaugeSiteSize + j];
          }else{
            ((float*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)inlink[dir])[i*gaugeSiteSize + j];
          }
        }
      }
    }
#endif
    freeGaugeQuda();

    //free(inlinks);
  } else { // else generate a random SU(3) field

//#define EXPERIMENTAL

#ifndef EXPERIMENTAL
//#define GAUGE_TRANSFORM
    const int gen_type = 1;

    construct_fat_long_gauge_field(qdp_fatlink, qdp_longlink, gen_type, gauge_param.cpu_prec, 
				 &gauge_param, dslash_type);
#ifdef GAUGE_TRANSFORM
    void *su3_buffer=malloc(4*V*gaugeSiteSize*gSize);
    void *su3_field[4];

    for (int dir = 0; dir < 4; dir++) {
       su3_field[dir] = (void*)((char*)su3_buffer + dir*V*gaugeSiteSize*gSize);
    }

    construct_gauge_field(su3_field, 1, gauge_param.cpu_prec, &gauge_param);

    const int dir = 3;
    transform_gauge_field(su3_field[dir], qdp_fatlink, qdp_longlink, gen_type, gauge_param.cpu_prec, 
				 &gauge_param, dslash_type);
    
    free(su3_buffer);
#endif //GAUGE_TRANSFORM

#ifdef CHECK_PLAQUETTE
    QudaGaugeParam gParam = newQudaGaugeParam();
    gParam.gauge_order = QUDA_QDP_GAUGE_ORDER ;

    gParam.X[0] = xdim;
    gParam.X[1] = ydim;
    gParam.X[2] = zdim;
    gParam.X[3] = tdim;

    gParam.anisotropy = 1.0;
    gParam.type = QUDA_WILSON_LINKS;
    gParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
    gParam.t_boundary = QUDA_PERIODIC_T;

    gParam.cpu_prec = cpu_prec;
    gParam.cuda_prec = cuda_prec;
    gParam.reconstruct = link_recon;
    gParam.cuda_prec_sloppy = cuda_prec_sloppy;
    gParam.reconstruct_sloppy = link_recon_sloppy;
    gParam.cuda_prec_precondition = cuda_prec_precondition;
    gParam.reconstruct_precondition = link_recon_precondition;
    gParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
    gParam.ga_pad = 0;

#ifdef MULTI_GPU
    int x_face_size = gParam.X[1]*gParam.X[2]*gParam.X[3]/2;
    int y_face_size = gParam.X[0]*gParam.X[2]*gParam.X[3]/2;
    int z_face_size = gParam.X[0]*gParam.X[1]*gParam.X[3]/2;
    int t_face_size = gParam.X[0]*gParam.X[1]*gParam.X[2]/2;
    int pad_size = std::max(x_face_size, y_face_size);
    pad_size = std::max(pad_size, z_face_size);
    pad_size = std::max(pad_size, t_face_size);
    gParam.ga_pad = pad_size;
#endif
    loadGaugeQuda( (void*)qdp_fatlink,  &gParam );
    printfQuda("\nLoaded fields\n");

    double plq[3];
    plaqQuda (plq);
    freeGaugeQuda();

    printfQuda("\nPlaq : %le, %le, %le\n", plq[0], plq[1], plq[2]);

#endif

#ifndef USE_QDP_LINKS
    for(int dir=0; dir<4; ++dir){
      double check_sum = 0.0;
      for(int i=0; i<V; ++i){
        for(int j=0; j<gaugeSiteSize; ++j){
          if(gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION){
            ((double*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)qdp_fatlink[dir])[i*gaugeSiteSize + j];
            ((double*)longlink)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)qdp_longlink[dir])[i*gaugeSiteSize + j];
            check_sum += ((double*)fatlink)[(i*4 + dir)*gaugeSiteSize + j];
          }else{
            ((float*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)qdp_fatlink[dir])[i*gaugeSiteSize + j];
            ((float*)longlink)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)qdp_longlink[dir])[i*gaugeSiteSize + j];
          }
        }
      }
      printfQuda("\nCheck sum dir %d : %1.18e\n", dir, check_sum); 
    }
#endif //USE_QDP_LINKS

#else //EXPERIMENTAL
/*
    void *inlinks=malloc(4*V*gaugeSiteSize*gSize);
    void *inlink[4];

    for (int dir = 0; dir < 4; dir++) {
       inlink[dir] = (void*)((char*)inlinks + dir*V*gaugeSiteSize*gSize);
    }
*/
    construct_gauge_field(inlink, 1, gauge_param.cpu_prec, &gauge_param);

#ifdef MULTI_GPU
    QudaComputeFatMethod method = QUDA_COMPUTE_FAT_EXTENDED_VOLUME;
#else
    QudaComputeFatMethod method = QUDA_COMPUTE_FAT_STANDARD;
#endif

    QudaGaugeParam gParam = newQudaGaugeParam();
    for(int dir=0; dir<4; ++dir) gParam.X[dir] = gauge_param.X[dir];
    gParam.cuda_prec_sloppy = gParam.cpu_prec = gParam.cuda_prec = gauge_param.cpu_prec;
    gParam.type = QUDA_GENERAL_LINKS;

    gParam.reconstruct_sloppy = gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.gauge_order   = QUDA_MILC_GAUGE_ORDER;
    gParam.t_boundary    = QUDA_PERIODIC_T;
    gParam.gauge_fix     = QUDA_GAUGE_FIXED_NO;
    gParam.scale         = 1.0;
    gParam.anisotropy    = 1.0;
    gParam.tadpole_coeff = 1.0;
    gParam.scale         = 0;
    gParam.ga_pad        = 0;
    gParam.site_ga_pad   = 0;
    gParam.mom_ga_pad    = 0;
    gParam.llfat_ga_pad  = 0;

    double act_path_coeff[6];

    act_path_coeff[0] = 1.0;
    act_path_coeff[1] = -1e-06;
    act_path_coeff[2] = -1e-06;
    act_path_coeff[3] = 1e-05;
    act_path_coeff[4] = -1e-06;
    act_path_coeff[5] = -1e-4;

    computeKSLinkQuda( fatlink, longlink, NULL, inlinks, const_cast<double*>(act_path_coeff), &gParam, method);
    freeGaugeQuda();

    //free(inlinks);
#endif
  }

  free(inlinks);

  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  for(int d = 0; d < 4; d++) {
    csParam.x[d] = gauge_param.X[d];
  }

  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  csParam.siteSubset =  QUDA_FULL_SITE_SUBSET;// QUDA_PARITY_SITE_SUBSET;//
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;  

  if(csParam.siteSubset == QUDA_PARITY_SITE_SUBSET) csParam.x[0] /= 2;
  in  = new cpuColorSpinorField(csParam);  
  out = new cpuColorSpinorField(csParam);  
  ref = new cpuColorSpinorField(csParam);  
  tmp = new cpuColorSpinorField(csParam);  

  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
    if(  inv_param.use_init_guess == QUDA_USE_INIT_GUESS_YES && inv_param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES)
      constructSpinorField((float*)out->V(), out->Volume());
    else
      constructSpinorField((float*)in->V(), in->Volume());    
  }else{
    if(  inv_param.use_init_guess == QUDA_USE_INIT_GUESS_YES && inv_param.compute_null_vector == QUDA_COMPUTE_NULL_VECTOR_YES)
      constructSpinorField((double*)out->V(), out->Volume());
    else
      constructSpinorField((double*)in->V(), in->Volume());
  }

#ifdef MULTI_GPU
  int tmp_value = MAX(ydim*zdim*tdim/2, xdim*zdim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*zdim/2);

  int fat_pad = tmp_value;
  int link_pad =  3*tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gauge_param.type = dslash_type == QUDA_STAGGERED_DSLASH ? 
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
#ifndef USE_QDP_LINKS
  GaugeFieldParam cpuFatParam(fatlink, gauge_param);
#else
  GaugeFieldParam cpuFatParam(qdp_fatlink, gauge_param);
#endif
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
#ifndef USE_QDP_LINKS
  GaugeFieldParam cpuLongParam(longlink, gauge_param);
#else
  GaugeFieldParam cpuLongParam(qdp_longlink, gauge_param);
#endif
  cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();


#else
  int fat_pad = 0;
  int link_pad = 0;
#endif
  
  gauge_param.type = dslash_type == QUDA_STAGGERED_DSLASH ? 
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gauge_param.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH) {
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
  } else {
    gauge_param.reconstruct= gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gauge_param.cuda_prec_precondition = QUDA_HALF_PRECISION;
#ifndef USE_QDP_LINKS
  loadGaugeQuda(fatlink, &gauge_param);
#else
  loadGaugeQuda(qdp_fatlink, &gauge_param);
#endif

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.reconstruct= link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
#ifndef USE_QDP_LINKS
    loadGaugeQuda(longlink, &gauge_param);
#else
    loadGaugeQuda(qdp_longlink, &gauge_param);
#endif
  }

  double time0 = -((double)clock()); // Start the timer

  double nrm2=0;
  double src2=0;

  if(inv_type != QUDA_GCR_INVERTER){
      inv_param.inv_type = QUDA_GCR_INVERTER;
      inv_param.gcrNkrylov = 24;
  }

  printfQuda("\nMass %le\n", mg_param.invert_param->mass);

  // setup the multigrid solver
  void *mg_preconditioner = newMultigridQuda(&mg_param);
  inv_param.preconditioner = mg_preconditioner;
  
  invertQuda(out->V(), in->V(), &inv_param);
  // free the multigrid solver
  destroyMultigridQuda(mg_preconditioner);
  //multigridQuda(out->V(), in->V(), &mg_param);

  time0 += clock(); 
  time0 /= CLOCKS_PER_SEC;

//MULTI_GPU code for staggered MG is not ready.
//#ifdef MULTI_GPU    
//  matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, 
//          out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVEN_PARITY);
//#else
  //matdagmat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_EVEN_PARITY);
  double kappa = 1.0 / (2.0*mass);
  mat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), kappa, 0, inv_param.cpu_prec, gauge_param.cpu_prec);
//#endif
  ax( kappa, in->V(), V*mySpinorSiteSize, inv_param.cpu_prec );
  mxpy(in->V(), ref->V(), V*mySpinorSiteSize, inv_param.cpu_prec);
  nrm2 = norm_2(ref->V(), V*mySpinorSiteSize, inv_param.cpu_prec);
  src2 = norm_2(in->V(), V*mySpinorSiteSize, inv_param.cpu_prec);
  double l2r = sqrt(nrm2/src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g;\n",
        inv_param.tol, inv_param.true_res, l2r);

  printfQuda("done: total time = %g secs, compute time = %g secs, %i iter / %g secs = %g gflops, \n", 
        time0, inv_param.secs, inv_param.iter, inv_param.secs,
        inv_param.gflops/inv_param.secs);

  end();
}



  static void end(void) 
{
  for(int i=0;i < 4;i++){
    free(qdp_fatlink[i]);
    free(qdp_longlink[i]);
  }

  free(fatlink);
  free(longlink);

  delete in;
  delete out;
  delete ref;
  delete tmp;

  if (cpuFat) delete cpuFat;
  if (cpuLong) delete cpuLong;

  endQuda();
}


  void
display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon test_type  S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %s         %d/%d/%d          %d \n",
      get_prec_str(prec),get_prec_str(prec_sloppy),
      get_recon_str(link_recon), 
      get_recon_str(link_recon_sloppy), get_test_type(test_type), xdim, ydim, zdim, tdim);     

  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
      dimPartitioned(0),
      dimPartitioned(1),
      dimPartitioned(2),
      dimPartitioned(3)); 

  return ;

}

  void
usage_extra(char** argv )
{
  printfQuda("Extra options:\n");
  printfQuda("    --test <0/1>                             # Test method\n");
  printfQuda("                                                0: Even even spinor CG inverter\n");
  printfQuda("                                                1: Odd odd spinor CG inverter\n");
  printfQuda("                                                3: Even even spinor multishift CG inverter\n");
  printfQuda("                                                4: Odd odd spinor multishift CG inverter\n");
  printfQuda("    --cpu_prec <double/single/half>          # Set CPU precision\n");

  return ;
}
int main(int argc, char** argv)
{
  for (int i = 1; i < argc; i++) {

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }   



    if( strcmp(argv[i], "--cpu_prec") == 0){
      if (i+1 >= argc){
        usage(argv);
      }
      cpu_prec= get_prec(argv[i+1]);
      i++;
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

  if(inv_type != QUDA_CG_INVERTER){
    if(test_type != 0 && test_type != 1) errorQuda("Preconditioning is currently not supported in multi-shift solver solvers");
  }


  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();
  
  printfQuda("dslash_type = %d\n", dslash_type);

  mg_test(argc, argv);

  // finalize the communications layer
  finalizeComms();

  return 0;
}
