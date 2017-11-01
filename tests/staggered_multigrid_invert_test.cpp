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


#include <dirac_quda.h>
#include <dslash_quda.h>

extern void usage(char** argv);
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
cpuColorSpinorField* tmp_fullsp;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

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
extern double anisotropy;
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
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];
extern double mu_factor[QUDA_MAX_MG_LEVEL];

extern QudaVerbosity mg_verbosity[QUDA_MAX_MG_LEVEL];

extern QudaInverterType setup_inv[QUDA_MAX_MG_LEVEL];
extern int num_setup_iter[QUDA_MAX_MG_LEVEL];
extern double setup_tol;
extern QudaSetupType setup_type;
extern bool pre_orthonormalize;
extern bool post_orthonormalize;
extern double omega;
extern QudaInverterType coarse_solver[QUDA_MAX_MG_LEVEL];
extern QudaInverterType smoother_type[QUDA_MAX_MG_LEVEL];
extern double coarse_solver_tol[QUDA_MAX_MG_LEVEL];
extern double smoother_tol[QUDA_MAX_MG_LEVEL];
extern int coarse_solver_maxiter[QUDA_MAX_MG_LEVEL];

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

QudaPrecision &cpu_prec = prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;

static void end();

void setGaugeParam(QudaGaugeParam &gauge_param) {

  const double tadpole_coeff = 0.8; // hardcoded, taken from staggered_invert_test.

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  gauge_param.anisotropy = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;//QUDA_ANTI_PERIODIC_T;
  gauge_param.tadpole_coeff = tadpole_coeff;
  
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


  // special to staggered tests.
  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gauge_param.scale = dslash_type == QUDA_STAGGERED_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);
}

void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;//this will be used to setup SolverParam parent in MGParam class

  for (int i=0; i<mg_levels-1; i++) printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);

  inv_param.Ls = Nsrc;

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

  inv_param.tune = QUDA_TUNE_NO;

  inv_param.dslash_type = dslash_type;

  //Free field!
  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (4 + inv_param.mass));

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.matpc_type = matpc_type;//!
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;//fixed

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;//WAS 4
      mg_param.geo_block_size[i][4] = 1;//to be safe with 5th dimension
    }
    mg_param.verbosity[i] = mg_verbosity[i];
    //mg_param.setup_inv_type[i] = setup_inv[i];

    mg_param.setup_inv_type[i] = setup_inv[i];
    mg_param.num_setup_iter[i] = num_setup_iter[i];//default 1
    mg_param.setup_tol[i] = setup_tol;

    mg_param.spin_block_size[i] = 1;// 1 or 0 (1 for parity blocking)
    mg_param.n_vec[i] =  nvec[i] == 0 ? 24 : nvec[i]; // default to 24 vectors if not set
    mg_param.nu_pre[i] = nu_pre;
    mg_param.nu_post[i] = nu_post;
    //mg_param.mu_factor[i] = mu_factor[i];

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;
    // set the coarse solver wrappers including bottom solver
    mg_param.coarse_solver[i] = coarse_solver[i];
    mg_param.coarse_solver_tol[i] = coarse_solver_tol[i];
    mg_param.coarse_solver_maxiter[i] = coarse_solver_maxiter[i];

    mg_param.smoother[i] = smoother_type[i];

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = smoother_tol[i]; // repurpose heavy-quark tolerance for now

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    // We'll eventually want the e-o preconditioning (since it's a nice normal op),
    // but for now let's not.
    mg_param.smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE; 

    // set to QUDA_ADDITIVE_SCHWARZ for Additive Schwarz precondioned smoother (presently only impelemented for MR)
    mg_param.smoother_schwarz_type[i] = schwarz_type[i];

    // if using Schwarz preconditioning then use local reductions only
    mg_param.global_reduction[i] = (schwarz_type[i] == QUDA_INVALID_SCHWARZ) ? QUDA_BOOLEAN_YES : QUDA_BOOLEAN_NO;

    // set number of Schwarz cycles to apply
    mg_param.smoother_schwarz_cycle[i] = schwarz_cycle[i];

    // set to QUDA_MAT_SOLUTION to inject a full field into coarse grid
    // set to QUDA_MATPC_SOLUTION to inject single parity field into coarse grid

    // if we are using an outer even-odd preconditioned solve, then we
    // use single parity injection into the coarse grid
    mg_param.coarse_grid_solution_type[i] = solve_type == QUDA_DIRECT_PC_SOLVE ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

    mg_param.omega[i] = omega; // over/under relaxation factor

    mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;
    //mg_param.location[i] = QUDA_CPU_FIELD_LOCATION;
  }

  // coarsen the spin on the first restriction is undefined for staggered fields
  mg_param.spin_block_size[0] = 1;
  warningQuda("Level 0 spin block size is set to: %d", mg_param.spin_block_size[0]);

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
  inv_param.inv_type = QUDA_GCR_INVERTER;//should this correlate with smoother? (it will be overwritten in mg setup, like param_presmooth->inv_type = param.smoother;, but be careful!)
  inv_param.tol = 1e-10;
  inv_param.maxiter = 100;
  inv_param.reliable_delta = 1e-10;
  inv_param.gcrNkrylov = 10;

  //inv_param.verbosity = QUDA_SUMMARIZE;
  inv_param.verbosity = QUDA_VERBOSE;
  //inv_param.verbosity_precondition = QUDA_SUMMARIZE;
  inv_param.verbosity_precondition = QUDA_VERBOSE;
}

void setInvertParam(QudaInvertParam &inv_param)
{
  inv_param.Ls = Nsrc;

  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec; 
  inv_param.cuda_prec_sloppy = prec_sloppy;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  // No need to set clover params...
  inv_param.clover_coeff = clover_coeff;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  inv_param.mass = mass;
  inv_param.mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param.dagger = QUDA_DAG_NO;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;//QUDA_DIRECT_SOLVE;
  inv_param.matpc_type = matpc_type;

  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.verbosity_precondition = QUDA_SUMMARIZE; // mg_verbosity[0];

  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.pipeline = pipeline;
  inv_param.gcrNkrylov = gcrNkrylov;
  inv_param.tol = tol;

  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param.residual_type = static_cast<QudaResidualType_s>(0);
  inv_param.residual_type = (tol != 0) ? static_cast<QudaResidualType_s> ( inv_param.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : inv_param.residual_type;
  inv_param.residual_type = (tol_hq != 0) ? static_cast<QudaResidualType_s> (inv_param.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : inv_param.residual_type;
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

int invert_test(int argc, char** argv)
{
  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;
  setMultigridParam(mg_param);

  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X,Nsrc); // so we can use 5-d indexing from dwf
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

  if (strcmp(latfile,"")) {
    read_gauge_field(latfile, inlink, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_fat_long_gauge_field(inlink, nullptr, 3, gauge_param.cpu_prec, &gauge_param, dslash_type);
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir], inlink[dir] , V*gaugeSiteSize*gSize);
    }

  } else {

    const int gen_type = 2; // free field.
    warningQuda("Gen type = %d", gen_type);

    construct_fat_long_gauge_field(qdp_fatlink, qdp_longlink, gen_type, gauge_param.cpu_prec, 
				 &gauge_param, dslash_type);
  }

  free(inlinks);

  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=5;
//  csParam.nDim=4;
  for (int d = 0; d < 4; d++) csParam.x[d] = gauge_param.X[d];
  csParam.x[4] = Nsrc;

  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;  

  if(  csParam.siteSubset == QUDA_PARITY_SITE_SUBSET) csParam.x[0] /= 2;

  in = new cpuColorSpinorField(csParam);  
  out = new cpuColorSpinorField(csParam);  
  ref = new cpuColorSpinorField(csParam);  
  tmp = new cpuColorSpinorField(csParam);

  in->Source(QUDA_RANDOM_SOURCE);

  printfQuda("\nSource norm : %1.15e\n", sqrt(blas::norm2(*in)));

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
  GaugeFieldParam cpuFatParam(qdp_fatlink, gauge_param);
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(qdp_longlink, gauge_param);
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
  gauge_param.cuda_prec_precondition = gauge_param.cuda_prec_sloppy;
  gauge_param.reconstruct_precondition = gauge_param.reconstruct_sloppy;
  loadGaugeQuda(qdp_fatlink, &gauge_param);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param.ga_pad = link_pad;
    gauge_param.reconstruct= link_recon;
    gauge_param.reconstruct_sloppy = link_recon_sloppy;
    gauge_param.cuda_prec_precondition = gauge_param.cuda_prec_sloppy;
    gauge_param.reconstruct_precondition = gauge_param.reconstruct_sloppy;
    loadGaugeQuda(qdp_fatlink, &gauge_param);
  }

  double time0 = -((double)clock()); // Start the timer

  double nrm2=0;
  double src2=0;
  int ret = 0;

  int len = V;//Vh*Nsrc;
  {//switch
      
      inv_param.matpc_type = matpc_type;

      printf("inv_param.pipeline = %d\n", inv_param.pipeline);
      void *mg_preconditioner = inv_param.inv_type_precondition == QUDA_MG_INVERTER ?  newMultigridQuda(&mg_param) : nullptr;
      inv_param.preconditioner = mg_preconditioner;

      invertQuda(out->V(), in->V(), &inv_param);

      if (inv_param.inv_type_precondition == QUDA_MG_INVERTER) destroyMultigridQuda(mg_preconditioner);

      time0 += clock(); 
      time0 /= CLOCKS_PER_SEC;

      double factor = 1.0 / (2.0*mass);
#ifdef MULTI_GPU    
      mat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, in, factor, QUDA_DAG_NO, inv_param.cpu_prec, gauge_param.cpu_prec, &inv_param);

#else
      mat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), factor, 0, inv_param.cpu_prec, gauge_param.cpu_prec);
#endif
      ax( factor, in->V(), len*mySpinorSiteSize, inv_param.cpu_prec );

      mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);

  }//switch

  double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
  double l2r = sqrt(nrm2/src2);

  printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
      inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);

  printfQuda("done: total time = %g secs, compute time = %g secs, %i iter / %g secs = %g gflops, \n", 
      time0, inv_param.secs, inv_param.iter, inv_param.secs,
      inv_param.gflops/inv_param.secs);

  end();
  return ret;
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

  printfQuda("prec    sloppy_prec    link_recon  sloppy_link_recon S_dimension T_dimension\n");
  printfQuda("%s   %s             %s            %s            %d/%d/%d          %d \n",
      get_prec_str(prec),get_prec_str(prec_sloppy),
      get_recon_str(link_recon), 
      get_recon_str(link_recon_sloppy), xdim, ydim, zdim, tdim);     

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
  // default
  dslash_type = QUDA_STAGGERED_DSLASH;

  // We give here the default values to some of the array
  for(int i =0; i<QUDA_MAX_MG_LEVEL; i++) {
    mg_verbosity[i] = QUDA_SILENT;
    setup_inv[i] = QUDA_BICGSTAB_INVERTER;
    num_setup_iter[i] = 1;
    mu_factor[i] = 1.;
    schwarz_type[i] = QUDA_INVALID_SCHWARZ;
    schwarz_cycle[i] = 1;
    smoother_type[i] = QUDA_MR_INVERTER;
    smoother_tol[i] = 0.25;
    coarse_solver[i] = QUDA_GCR_INVERTER;
    coarse_solver_tol[i] = 0.25;
    coarse_solver_maxiter[i] = 10;
  }

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

  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;


  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rand-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_ASQTAD_DSLASH &&
      dslash_type != QUDA_STAGGERED_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }
  
  printfQuda("dslash_type = %d\n", dslash_type);

  int ret = invert_test(argc, argv);

  // finalize the communications layer
  finalizeComms();

  return ret;
}
