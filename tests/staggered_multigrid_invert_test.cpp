#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

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
QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;

extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;
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

//mg:
extern char latfile[];
extern int niter;
extern int nvec;
extern int mg_levels;

extern int nu_pre;
extern int nu_post;
extern int geo_block_size[];

extern QudaInverterType precon_type;
//mg

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


static void
set_params(QudaGaugeParam* gaugeParam, QudaInvertParam* inv_param, QudaMultigridParam *mg_param,
    int X1, int  X2, int X3, int X4,
    QudaPrecision cpu_prec, QudaPrecision prec, QudaPrecision prec_sloppy,
    QudaReconstructType link_recon, QudaReconstructType link_recon_sloppy,
    double mass, double tol, int maxiter, double reliable_delta,
    double tadpole_coeff
    )
{
  gaugeParam->X[0] = X1;
  gaugeParam->X[1] = X2;
  gaugeParam->X[2] = X3;
  gaugeParam->X[3] = X4;

  gaugeParam->cpu_prec = cpu_prec;    
  gaugeParam->cuda_prec = prec;
  gaugeParam->reconstruct = link_recon;  
  gaugeParam->cuda_prec_sloppy = prec_sloppy;
  gaugeParam->reconstruct_sloppy = link_recon_sloppy;
  gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = tadpole_coeff;

  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gaugeParam->scale = dslash_type == QUDA_STAGGERED_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  gaugeParam->t_boundary = QUDA_PERIODIC_T;//QUDA_ANTI_PERIODIC_T;
#ifndef USE_QDP_LINKS
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;
#else
  gaugeParam->gauge_order = QUDA_QDP_GAUGE_ORDER;
#endif
  gaugeParam->ga_pad = X1*X2*X3/2;

  inv_param->verbosity = QUDA_VERBOSE;
  inv_param->mass = mass;

  // outer solver parameters
  inv_param->inv_type = QUDA_GCR_INVERTER;
  inv_param->tol = tol;
  inv_param->tol_restart = 1e-3; 
  inv_param->maxiter = 5000;
  inv_param->reliable_delta = 1e-2;
  inv_param->use_sloppy_partial_accumulator = false;
  inv_param->pipeline = false;
  //mg:
  //Warning: Outer solver always has these parameters:
  inv_param->solution_type = QUDA_MAT_SOLUTION;//QUDA_MATPCDAG_MATPC_SOLUTION
  inv_param->solve_type = QUDA_DIRECT_SOLVE;//QUDA_NORMOP_PC_SOLVE
  //
  inv_param->matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param->dagger = QUDA_DAG_NO;
  inv_param->mass_normalization = QUDA_MASS_NORMALIZATION;  
  
#if __COMPUTE_CAPABILITY__ >= 200
  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param->residual_type = static_cast<QudaResidualType_s>(0);
  inv_param->residual_type = (tol != 0) ? static_cast<QudaResidualType_s> ( inv_param->residual_type | QUDA_L2_RELATIVE_RESIDUAL) : inv_param->residual_type;
  inv_param->residual_type = (tol_hq != 0) ? static_cast<QudaResidualType_s> (inv_param->residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : inv_param->residual_type;

  inv_param->tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual
#else
  if(tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // Pre Fermi architecture only supports L2 relative residual norm
  inv_param->residual_type = QUDA_L2_RELATIVE_RESIDUAL;
#endif
  inv_param->Nsteps = 2; 

  inv_param->gcrNkrylov = 20;

  // domain decomposition preconditioner parameters
  inv_param->inv_type_precondition = QUDA_MG_INVERTER;
  inv_param->schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param->precondition_cycle = 1;
  inv_param->tol_precondition = 1e-1;
  inv_param->maxiter_precondition = 10;
  inv_param->verbosity_precondition = QUDA_SILENT;
  inv_param->cuda_prec_precondition = prec_sloppy;
  inv_param->omega = 1.0; 

  inv_param->cpu_prec = cpu_prec;
  inv_param->cuda_prec = prec; 
  inv_param->cuda_prec_sloppy = prec_sloppy;
  inv_param->preserve_source = QUDA_PRESERVE_SOURCE_NO;
  inv_param->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param->dirac_order = QUDA_DIRAC_ORDER;

  inv_param->dslash_type = dslash_type;

  inv_param->tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;
  inv_param->sp_pad = 0;//X1*X2*X3/2;
  inv_param->use_init_guess = QUDA_USE_INIT_GUESS_YES;

  inv_param->input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param->output_location = QUDA_CPU_FIELD_LOCATION;
  
  mg_param->invert_param = inv_param;
  mg_param->n_level = mg_levels;
  for (int i=0; i<mg_param->n_level; i++) {
    for (int j=0; j<QUDA_MAX_DIM; j++) {
      mg_param->geo_block_size[i][j] = geo_block_size[j];
    }
    mg_param->spin_block_size[i] = 1;
    mg_param->n_vec[i] = nvec;
    mg_param->nu_pre[i] = nu_pre;
    mg_param->nu_post[i] = nu_post;

    mg_param->smoother[i] = precon_type;

    // set to QUDA_DIRECT_SOLVE for no even/odd preconditioning on the smoother
    mg_param->smoother_solve_type[i] = QUDA_DIRECT_SOLVE;
    mg_param->omega[i] = 1.0; // over/under relaxation factor

    mg_param->location[i] = QUDA_CUDA_FIELD_LOCATION;
  }
  // coarsen the spin on the first restriction is undefined for staggered fields
  mg_param->spin_block_size[0] = 0;

  // coarse grid solver is GCR
  mg_param->smoother[mg_levels-1] = QUDA_GCR_INVERTER;
  mg_param->compute_null_vector = generate_nullspace ? QUDA_COMPUTE_NULL_VECTOR_YES
    : QUDA_COMPUTE_NULL_VECTOR_NO;

  // set file i/o parameters
  strcpy(mg_param->vec_infile, vec_infile);
  strcpy(mg_param->vec_outfile, vec_outfile);

  return;
}


void mg_test(void)
{
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();
  QudaMultigridParam mg_param;

  set_params(&gaugeParam, &inv_param, &mg_param,
      xdim, ydim, zdim, tdim,
      cpu_prec, prec, prec_sloppy,
      link_recon, link_recon_sloppy, mass, tol, 500, 1e-3,
      0.8);

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gaugeParam.X);
  setSpinorSiteSize(6);

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  for (int dir = 0; dir < 4; dir++) {
    qdp_fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  fatlink = malloc(4*V*gaugeSiteSize*gSize);
  longlink = malloc(4*V*gaugeSiteSize*gSize);

  construct_fat_long_gauge_field(qdp_fatlink, qdp_longlink, 1, gaugeParam.cpu_prec, 
				 &gaugeParam, dslash_type);

#ifndef USE_QDP_LINKS
  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<V; ++i){
      for(int j=0; j<gaugeSiteSize; ++j){
        if(gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION){
          ((double*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)qdp_fatlink[dir])[i*gaugeSiteSize + j];
          ((double*)longlink)[(i*4 + dir)*gaugeSiteSize + j] = ((double*)qdp_longlink[dir])[i*gaugeSiteSize + j];
        }else{
          ((float*)fatlink)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)qdp_fatlink[dir])[i*gaugeSiteSize + j];
          ((float*)longlink)[(i*4 + dir)*gaugeSiteSize + j] = ((float*)qdp_longlink[dir])[i*gaugeSiteSize + j];
        }
      }
    }
  }
#endif

  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  for(int d = 0; d < 4; d++) {
    csParam.x[d] = gaugeParam.X[d];
  }
  
  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  csParam.siteSubset =  QUDA_FULL_SITE_SUBSET;// QUDA_PARITY_SITE_SUBSET;//
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;  
  in = new cpuColorSpinorField(csParam);  
  out = new cpuColorSpinorField(csParam);  
  ref = new cpuColorSpinorField(csParam);  
  tmp = new cpuColorSpinorField(csParam);  

  if(csParam.siteSubset == QUDA_PARITY_SITE_SUBSET) csParam.x[0] /= 2;

  if (inv_param.cpu_prec == QUDA_SINGLE_PRECISION){
    constructSpinorField((float*)in->V(), in->Volume());    
  }else{
    constructSpinorField((double*)in->V(), in->Volume());
  }

#ifdef MULTI_GPU
  int tmp_value = MAX(ydim*zdim*tdim/2, xdim*zdim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*tdim/2);
  tmp_value = MAX(tmp_value, xdim*ydim*zdim/2);

  int fat_pad = tmp_value;
  int link_pad =  3*tmp_value;

  // FIXME: currently assume staggered is SU(3)
  gaugeParam.type = dslash_type == QUDA_STAGGERED_DSLASH ? 
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
#ifndef USE_QDP_LINKS
  GaugeFieldParam cpuFatParam(fatlink, gaugeParam);
#else
  GaugeFieldParam cpuFatParam(qdp_fatlink, gaugeParam);
#endif
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
#ifndef USE_QDP_LINKS
  GaugeFieldParam cpuLongParam(longlink, gaugeParam);
#else
  GaugeFieldParam cpuLongParam(qdp_longlink, gaugeParam);
#endif
  cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();


#else
  int fat_pad = 0;
  int link_pad = 0;
#endif
  
  gaugeParam.type = dslash_type == QUDA_STAGGERED_DSLASH ? 
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH) {
    gaugeParam.reconstruct = link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
  } else {
    gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gaugeParam.cuda_prec_precondition = QUDA_HALF_PRECISION;
#ifndef USE_QDP_LINKS
  loadGaugeQuda(fatlink, &gaugeParam);
#else
  loadGaugeQuda(qdp_fatlink, &gaugeParam);
#endif

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.ga_pad = link_pad;
    gaugeParam.reconstruct= link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
#ifndef USE_QDP_LINKS
    loadGaugeQuda(longlink, &gaugeParam);
#else
    loadGaugeQuda(qdp_longlink, &gaugeParam);
#endif
  }

  double time0 = -((double)clock()); // Start the timer

  double nrm2=0;
  double src2=0;

  if(inv_type != QUDA_GCR_INVERTER){
      inv_param.inv_type = QUDA_GCR_INVERTER;
      inv_param.gcrNkrylov = 50;
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
/*
#ifdef MULTI_GPU    
  matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, 
          out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVEN_PARITY);
#else
  matdagmat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_EVEN_PARITY);
#endif

  mxpy(in->V(), ref->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);
  nrm2 = norm_2(ref->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);
  src2 = norm_2(in->V(), Vh*mySpinorSiteSize, inv_param.cpu_prec);

  if (test_type <=2){

    double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
    double l2r = sqrt(nrm2/src2);

    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
        inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);

    printfQuda("done: total time = %g secs, compute time = %g secs, %i iter / %g secs = %g gflops, \n", 
        time0, inv_param.secs, inv_param.iter, inv_param.secs,
        inv_param.gflops/inv_param.secs);
  }
*/
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

  mg_test();

  // finalize the communications layer
  finalizeComms();

  return 0;
}
