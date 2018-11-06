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
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>


#include <gauge_field.h>
#include <gauge_tools.h>
#include <pgauge_monte.h>
#include <random_quda.h>
#include <unitarization_links.h>


#define MAX(a,b) ((a)>(b)?(a):(b))

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

// Wilson, clover-improved Wilson, twisted mass, and domain wall are supported.
extern QudaDslashType dslash_type;
extern bool tune;
extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaPrecision  prec_sloppy;
extern QudaPrecision  prec_precondition;
extern QudaReconstructType link_recon_sloppy;
extern QudaReconstructType link_recon_precondition;
extern double mass;
extern double mu;
extern double anisotropy;
extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern char latfile[];
extern int niter;
extern int nvec[];
extern int mg_levels;

extern bool generate_nullspace;
extern bool generate_all_levels;
extern int nu_pre;
extern int nu_post;
extern int geo_block_size[QUDA_MAX_MG_LEVEL][QUDA_MAX_DIM];

extern QudaInverterType smoother_type;

extern QudaMatPCType matpc_type;
extern QudaSolveType solve_type;

extern char vec_infile[];
extern char vec_outfile[];

//Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;

extern void usage(char** );

extern double clover_coeff;
extern bool compute_clover;

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

  printfQuda("MG parameters\n");
  printfQuda(" - number of levels %d\n", mg_levels);
  for (int i=0; i<mg_levels-1; i++) printfQuda(" - level %d number of null-space vectors %d\n", i+1, nvec[i]);
  printfQuda(" - number of pre-smoother applications %d\n", nu_pre);
  printfQuda(" - number of post-smoother applications %d\n", nu_post);

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

void setMultigridParam(QudaMultigridParam &mg_param) {
  QudaInvertParam &inv_param = *mg_param.invert_param;

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

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
    inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
  }

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.dslash_type = dslash_type;

  //Free field!
  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  inv_param.matpc_type = matpc_type;
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  inv_param.solve_type = QUDA_DIRECT_SOLVE;

  mg_param.invert_param = &inv_param;
  mg_param.n_level = mg_levels;
  for (int i=0; i<mg_param.n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
      // if not defined use 4
      mg_param.geo_block_size[i][j] = geo_block_size[i][j] ? geo_block_size[i][j] : 4;
    }
    mg_param.spin_block_size[i] = 1;
    mg_param.n_vec[i] = nvec[i] == 0 ? 24 : nvec[i]; // default to 24 vectors if not set
    mg_param.nu_pre[i] = nu_pre;
    mg_param.nu_post[i] = nu_post;

    mg_param.cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    mg_param.smoother[i] = smoother_type;

    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param.smoother_tol[i] = tol_hq; // repurpose heavy-quark tolerance for now

    mg_param.global_reduction[i] = QUDA_BOOLEAN_YES;

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    // set to QUDA_DIRECT_PC_SOLVE for to enable even/odd preconditioning on the smoother
    mg_param.smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE; // EVEN-ODD

    // set to QUDA_MAT_SOLUTION to inject a full field into coarse grid
    // set to QUDA_MATPC_SOLUTION to inject single parity field into coarse grid

    // if we are using an outer even-odd preconditioned solve, then we
    // use single parity injection into the coarse grid
    mg_param.coarse_grid_solution_type[i] = solve_type == QUDA_DIRECT_PC_SOLVE ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

    mg_param.omega[i] = 0.85; // over/under relaxation factor

    mg_param.location[i] = QUDA_CUDA_FIELD_LOCATION;
  }

  // only coarsen the spin on the first restriction
  mg_param.spin_block_size[0] = 2;

  // coarse grid solver is GCR
  mg_param.smoother[mg_levels-1] = QUDA_GCR_INVERTER;

  mg_param.compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES
    : QUDA_COMPUTE_NULL_VECTOR_NO;

  mg_param.generate_all_levels = generate_all_levels ? QUDA_BOOLEAN_YES :  QUDA_BOOLEAN_NO;

  mg_param.run_verify = QUDA_BOOLEAN_YES;

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

  inv_param.verbosity = QUDA_SUMMARIZE;
  inv_param.verbosity_precondition = QUDA_SUMMARIZE;
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

  //Free field!
  inv_param.mass = mass;
  inv_param.kappa = 1.0 / (2.0 * (1 + 3/anisotropy + mass));

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = mu;
    inv_param.twist_flavor = twist_flavor;
    inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) ? 2 : 1;

    if (twist_flavor == QUDA_TWIST_NONDEG_DOUBLET) {
      printfQuda("Twisted-mass doublet non supported (yet)\n");
      exit(0);
    }
  }

  inv_param.clover_coeff = clover_coeff;

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;

  // do we want full solution or single-parity solution
  inv_param.solution_type = QUDA_MAT_SOLUTION;

  // do we want to use an even-odd preconditioned solve or not
  inv_param.solve_type = solve_type;
  inv_param.matpc_type = matpc_type;

  inv_param.inv_type = QUDA_GCR_INVERTER;

  inv_param.verbosity = QUDA_VERBOSE;
  inv_param.verbosity_precondition = QUDA_SILENT;


  inv_param.inv_type_precondition = QUDA_MG_INVERTER;
  inv_param.gcrNkrylov = 20;
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

 void setReunitarizationConsts(){
   using namespace quda;
    const double unitarize_eps = 1e-14;
    const double max_error = 1e-10;
    const int reunit_allow_svd = 1;
    const int reunit_svd_only  = 0;
    const double svd_rel_error = 1e-6;
    const double svd_abs_error = 1e-6;
    setUnitarizeLinksConstants(unitarize_eps, max_error,
                               reunit_allow_svd, reunit_svd_only,
                               svd_rel_error, svd_abs_error);

  }

void CallUnitarizeLinks(quda::cudaGaugeField *cudaInGauge){
   using namespace quda;
   int *num_failures_dev = (int*)device_malloc(sizeof(int));
   int num_failures;
   cudaMemset(num_failures_dev, 0, sizeof(int));
   unitarizeLinks(*cudaInGauge, num_failures_dev);

   cudaMemcpy(&num_failures, num_failures_dev, sizeof(int), cudaMemcpyDeviceToHost);
   if(num_failures>0) errorQuda("Error in the unitarization\n");
   device_free(num_failures_dev);
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
  if (link_recon_sloppy == QUDA_RECONSTRUCT_INVALID) link_recon_sloppy = link_recon;
  if (link_recon_precondition == QUDA_RECONSTRUCT_INVALID) link_recon_precondition = link_recon_sloppy;

  // initialize QMP/MPI, QUDA comms grid and RNG (test_util.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // call srand() with a rank-dependent seed
  initRand();

  display_test_info();

  // *** QUDA parameters begin here.

  if (dslash_type != QUDA_WILSON_DSLASH &&
      dslash_type != QUDA_CLOVER_WILSON_DSLASH &&
      dslash_type != QUDA_TWISTED_MASS_DSLASH &&
      dslash_type != QUDA_TWISTED_CLOVER_DSLASH) {
    printfQuda("dslash_type %d not supported\n", dslash_type);
    exit(0);
  }

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setGaugeParam(gauge_param);

  QudaInvertParam mg_inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param = newQudaMultigridParam();
  mg_param.invert_param = &mg_inv_param;

  setMultigridParam(mg_param);


  QudaInvertParam inv_param = newQudaInvertParam();
  setInvertParam(inv_param);

  // *** Everything between here and the call to initQuda() is
  // *** application-specific.

  setDims(gauge_param.X);

  setSpinorSiteSize(24);

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  size_t sSize = (inv_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void *gauge[4], *clover=0, *clover_inv=0;

  for (int dir = 0; dir < 4; dir++) {
    gauge[dir] = malloc(V*gaugeSiteSize*gSize);
  }

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(gauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    //generate a random SU(3) field
    //construct_gauge_field(gauge, 1, gauge_param.cpu_prec, &gauge_param);
    //generate a unit SU(3) field
    construct_gauge_field(gauge, 0, gauge_param.cpu_prec, &gauge_param);
    
  }

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    double norm = 0.1; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    size_t cSize = inv_param.clover_cpu_prec;
    clover = malloc(V*cloverSiteSize*cSize);
    clover_inv = malloc(V*cloverSiteSize*cSize);
    if (!compute_clover) construct_clover_field(clover, norm, diag, inv_param.clover_cpu_prec);

    inv_param.compute_clover = compute_clover;
    if (compute_clover) inv_param.return_clover = 1;
    inv_param.compute_clover_inverse = 1;
    inv_param.return_clover_inverse = 1;
  }

  void *spinorIn = malloc(V*spinorSiteSize*sSize*inv_param.Ls);
  void *spinorCheck = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  void *spinorOut = NULL;
  spinorOut = malloc(V*spinorSiteSize*sSize*inv_param.Ls);

  // start the timer
  double time0 = -((double)clock());

  // initialize the QUDA library
  initQuda(device);

  {
    using namespace quda;
    GaugeFieldParam gParam(0, gauge_param);
    gParam.pad = 0;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.create      = QUDA_NULL_FIELD_CREATE;
    gParam.link_type   = gauge_param.type;
    gParam.reconstruct = gauge_param.reconstruct;
    gParam.order       = (gauge_param.cuda_prec == QUDA_DOUBLE_PRECISION || gauge_param.reconstruct == QUDA_RECONSTRUCT_NO )
      ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    cudaGaugeField *gauge = new cudaGaugeField(gParam);

    int pad = 0;
    int y[4];
    int R[4] = {0,0,0,0};
    for(int dir=0; dir<4; ++dir) if(comm_dim_partitioned(dir)) R[dir] = 2;
    for(int dir=0; dir<4; ++dir) y[dir] = gauge_param.X[dir] + 2 * R[dir];
    GaugeFieldParam gParamEx(y, prec, link_recon,
			     pad, QUDA_VECTOR_GEOMETRY, QUDA_GHOST_EXCHANGE_EXTENDED);
    gParamEx.create = QUDA_ZERO_FIELD_CREATE;
    gParamEx.order = gParam.order;
    gParamEx.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParamEx.t_boundary = gParam.t_boundary;
    gParamEx.nFace = 1;
    for(int dir=0; dir<4; ++dir) gParamEx.r[dir] = R[dir];
    cudaGaugeField *gaugeEx = new cudaGaugeField(gParamEx);
    int halfvolume = xdim*ydim*zdim*tdim >> 1;
    // CURAND random generator initialization
    RNG *randstates = new RNG(halfvolume, 1234, gauge_param.X);
    randstates->Init();

    int nsteps = 100;
    int nhbsteps = 1;
    int novrsteps = 1;
    bool  coldstart = false;
    double beta_value = 6.2;

    if(link_recon != QUDA_RECONSTRUCT_8 && coldstart) InitGaugeField( *gaugeEx);
    else{
      InitGaugeField( *gaugeEx, *randstates );
    }
    // Reunitarization setup
    setReunitarizationConsts();
    plaquette( *gaugeEx, QUDA_CUDA_FIELD_LOCATION) ;

    Monte( *gaugeEx, *randstates, beta_value, 100*nhbsteps, 100*novrsteps);

    // copy into regular field
    copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

    // load the gauge field from gauge
    gauge_param.gauge_order = (gauge_param.cuda_prec == QUDA_DOUBLE_PRECISION || gauge_param.reconstruct == QUDA_RECONSTRUCT_NO )
      ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
    gauge_param.location = QUDA_CUDA_FIELD_LOCATION;

    loadGaugeQuda(gauge->Gauge_p(), &gauge_param);

    // setup the multigrid solver
    void *mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;

    // create a point source at 0 (in each subvolume...  FIXME)
    memset(spinorIn, 0, inv_param.Ls*V*spinorSiteSize*sSize);
    memset(spinorCheck, 0, inv_param.Ls*V*spinorSiteSize*sSize);
    memset(spinorOut, 0, inv_param.Ls*V*spinorSiteSize*sSize);

    if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION) {
      for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((float*)spinorIn)[i] = rand() / (float)RAND_MAX;
    } else {
      for (int i=0; i<inv_param.Ls*V*spinorSiteSize; i++) ((double*)spinorIn)[i] = rand() / (double)RAND_MAX;
    }

    invertQuda(spinorOut, spinorIn, &inv_param);

    freeGaugeQuda();

    for(int step=1; step<=nsteps; ++step){
      Monte( *gaugeEx, *randstates, beta_value, nhbsteps, novrsteps);

      //Reunitarize gauge links...
      CallUnitarizeLinks(gaugeEx);
      double3 plaq = plaquette( *gaugeEx, QUDA_CUDA_FIELD_LOCATION) ;
      printfQuda("step=%d plaquette = %e\n", step, plaq.x);

      // copy into regular field
      copyExtendedGauge(*gauge, *gaugeEx, QUDA_CUDA_FIELD_LOCATION);

      loadGaugeQuda(gauge->Gauge_p(), &gauge_param);
      
      updateMultigridQuda(mg_preconditioner, &mg_param); // update the multigrid operator for and new gauge and clover fields

      invertQuda(spinorOut, spinorIn, &inv_param);

      freeGaugeQuda();
    }

    // free the multigrid solver
    destroyMultigridQuda(mg_preconditioner);

    delete gauge;
    delete gaugeEx;
    //Release all temporary memory used for data exchange between GPUs in multi-GPU mode
    PGaugeExchangeFree();
 
    randstates->Release();
    delete randstates;
  }

  // stop the timer
  time0 += clock();
  time0 /= CLOCKS_PER_SEC;
    
  printfQuda("Device memory used:\n   Spinor: %f GiB\n    Gauge: %f GiB\n", 
	 inv_param.spinorGiB, gauge_param.gaugeGiB);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) printfQuda("   Clover: %f GiB\n", inv_param.cloverGiB);
  //printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
  //inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, time0);
  printfQuda("\nDone: %i iter / %g secs = %g Gflops, total time = %g secs\n", 
	 inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs, 0.0);

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
