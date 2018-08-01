#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <test_util.h>
#include <dslash_util.h>
#include <blas_reference.h>
#include <staggered_dslash_reference.h>
#include "llfat_reference.h"
#include <quda.h>
#include <string.h>
#include "misc.h"
#include <gauge_field.h>
#include <blas_quda.h>
#include <unitarization_links.h>

#if defined(QMP_COMMS)
#include <qmp.h>
#elif defined(MPI_COMMS)
#include <mpi.h>
#endif

#include <qio_field.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define mySpinorSiteSize 6

extern void usage(char** argv);

#ifdef MULTI_GPU
void** ghost_fatlink, **ghost_longlink;
#endif

extern int device;

extern QudaReconstructType link_recon;
extern QudaPrecision prec;
QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;

extern QudaReconstructType link_recon_sloppy;
extern QudaPrecision  prec_sloppy;
cpuColorSpinorField* in;
cpuColorSpinorField* out;
cpuColorSpinorField* ref;
cpuColorSpinorField* tmp;


extern double tol; // tolerance for inverter
extern double tol_hq; // heavy-quark tolerance for inverter
extern double reliable_delta;
extern int test_type;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];

extern int Nsrc; // number of spinors to apply to simultaneously
extern int niter;

// Dirac operator type
extern QudaDslashType dslash_type;


extern QudaInverterType inv_type;
extern double mass; // the mass of the Dirac operator

// HISQ tadpole factor
extern double tadpole_factor;

extern bool compute_fatlong; // build the true fat/long links or use random numbers

// relativistic correction for naik term
extern double eps_naik;
// Number of naiks. If eps_naik is 0.0, we only need
// to construct one naik.
static int n_naiks = 1;

extern char latfile[];

extern int pipeline; // length of pipeline for fused operations in GCR or BiCGstab-l
extern int solution_accumulator_pipeline; // length of pipeline for fused solution update from the direction vectors

extern QudaSolveType solve_type;

// Unitarization coefficients
static double unitarize_eps  = 1e-6;
static bool reunit_allow_svd = true;
static bool reunit_svd_only  = false;
static double svd_rel_error  = 1e-4;
static double svd_abs_error  = 1e-4;
static double max_allowed_error = 1e-11;

// For loading the gauge fields
int argc_copy;
char** argv_copy;

static void end();

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

static void
set_params(QudaGaugeParam* gaugeParam, QudaInvertParam* inv_param,
    int X1, int  X2, int X3, int X4,
    QudaPrecision cpu_prec, QudaPrecision prec, QudaPrecision prec_sloppy,
    QudaReconstructType link_recon, QudaReconstructType link_recon_sloppy,
    double mass, double tol,
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
  gaugeParam->type = QUDA_WILSON_LINKS;
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = tadpole_coeff;

  if (dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gaugeParam->scale = dslash_type != QUDA_ASQTAD_DSLASH ? 1.0 : -1.0/(24.0*tadpole_coeff*tadpole_coeff);

  gaugeParam->t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER;
  gaugeParam->ga_pad = X1*X2*X3/2;

  inv_param->verbosity = QUDA_VERBOSE;
  inv_param->mass = mass;
  inv_param->kappa = 1.0/(8.0 + mass); // for Laplace operator

  // outer solver parameters
  inv_param->inv_type = inv_type;
  inv_param->tol = tol;
  inv_param->tol_restart = 1e-3; //now theoretical background for this parameter... 
  inv_param->maxiter = niter;
  inv_param->reliable_delta = reliable_delta;
  inv_param->use_sloppy_partial_accumulator = false;
  inv_param->solution_accumulator_pipeline = solution_accumulator_pipeline;
  inv_param->pipeline = pipeline;

  inv_param->Ls = Nsrc;
  
  if(tol_hq == 0 && tol == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }
  // require both L2 relative and heavy quark residual to determine convergence
  inv_param->residual_type = static_cast<QudaResidualType_s>(0);
  inv_param->residual_type = (tol != 0) ? static_cast<QudaResidualType_s> ( inv_param->residual_type | QUDA_L2_RELATIVE_RESIDUAL) : inv_param->residual_type;
  inv_param->residual_type = (tol_hq != 0) ? static_cast<QudaResidualType_s> (inv_param->residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : inv_param->residual_type;

  inv_param->tol_hq = tol_hq; // specify a tolerance for the residual for heavy quark residual
 
  inv_param->Nsteps = 2; 


  //inv_param->inv_type = QUDA_GCR_INVERTER;
  //inv_param->gcrNkrylov = 10;

  // domain decomposition preconditioner parameters
  inv_param->inv_type_precondition = QUDA_SD_INVERTER;
  inv_param->tol_precondition = 1e-1;
  inv_param->maxiter_precondition = 10;
  inv_param->verbosity_precondition = QUDA_SILENT;
  inv_param->cuda_prec_precondition = inv_param->cuda_prec_sloppy;

  inv_param->solution_type = dslash_type == QUDA_LAPLACE_DSLASH ? QUDA_MAT_SOLUTION : QUDA_MATPCDAG_MATPC_SOLUTION;
  inv_param->solve_type = dslash_type == QUDA_LAPLACE_DSLASH ? solve_type : QUDA_NORMOP_PC_SOLVE;
  inv_param->matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param->dagger = QUDA_DAG_NO;
  inv_param->mass_normalization = QUDA_MASS_NORMALIZATION;

  inv_param->cpu_prec = cpu_prec;
  inv_param->cuda_prec = prec; 
  inv_param->cuda_prec_sloppy = prec_sloppy;
  inv_param->preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // this is meaningless, but must be thus set
  inv_param->dirac_order = QUDA_DIRAC_ORDER;

  inv_param->dslash_type = dslash_type;

  inv_param->sp_pad = X1*X2*X3/2;
  inv_param->use_init_guess = QUDA_USE_INIT_GUESS_YES;

  inv_param->input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param->output_location = QUDA_CPU_FIELD_LOCATION;
}

void computeHISQLinksGPU(void** fatlink, void** longlink,
        void** fatlink_eps, void** longlink_eps,
        void** inlink, void* qudaGaugeParamPtr,
        double** act_path_coeffs, double eps_naik) {

  QudaGaugeParam gaugeParam = *(reinterpret_cast<QudaGaugeParam*>(qudaGaugeParamPtr));
  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  // inlink in different format
  void *inlink_milc = pinned_malloc(4*V*gaugeSiteSize*gSize);
  reorderQDPtoMILC(inlink_milc,inlink,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);

  // Paths for step 1:
  void* vlink_milc  = pinned_malloc(4*V*gaugeSiteSize*gSize); // V links
  void* wlink_milc  = pinned_malloc(4*V*gaugeSiteSize*gSize); // W links
  
  // Paths for step 2:
  void* fatlink_milc = pinned_malloc(4*V*gaugeSiteSize*gSize); // final fat ("X") links
  void* longlink_milc = pinned_malloc(4*V*gaugeSiteSize*gSize); // final long links
  
  // Create V links (fat7 links) and W links (unitarized V links), 1st path table set
  computeKSLinkQuda(vlink_milc, nullptr, wlink_milc, inlink_milc, act_path_coeffs[0], &gaugeParam);

  // Create X and long links, 2nd path table set
  computeKSLinkQuda(fatlink_milc, longlink_milc, nullptr, wlink_milc, act_path_coeffs[1], &gaugeParam);

  // Copy back
  reorderMILCtoQDP(fatlink,fatlink_milc,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  reorderMILCtoQDP(longlink,longlink_milc,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);

  // Clean up GPU compute links
  host_free(inlink_milc);
  host_free(vlink_milc);
  host_free(wlink_milc);
  host_free(fatlink_milc);
  host_free(longlink_milc);

}


  int
invert_test(void)
{
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  QudaInvertParam inv_param = newQudaInvertParam();

  set_params(&gaugeParam, &inv_param,
      xdim, ydim, zdim, tdim,
      cpu_prec, prec, prec_sloppy,
      link_recon, link_recon_sloppy, mass, tol, tadpole_factor);

  // this must be before the FaceBuffer is created (this is because it allocates pinned memory - FIXME)
  initQuda(device);

  setDims(gaugeParam.X);
  dw_setDims(gaugeParam.X,Nsrc); // so we can use 5-d indexing from dwf
  setSpinorSiteSize(6);

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  void* qdp_inlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_fatlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* qdp_longlink[4] = {nullptr,nullptr,nullptr,nullptr};
  void* milc_fatlink = nullptr;
  void* milc_longlink = nullptr;

  for (int dir = 0; dir < 4; dir++) {
    qdp_inlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    qdp_longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  milc_fatlink = malloc(4*V*gaugeSiteSize*gSize);
  milc_longlink = malloc(4*V*gaugeSiteSize*gSize);

  // load a field WITHOUT PHASES
  if (strcmp(latfile,"")) {
    read_gauge_field(latfile, qdp_inlink, gaugeParam.cpu_prec, gaugeParam.X, argc_copy, argv_copy);
    if (dslash_type != QUDA_LAPLACE_DSLASH) {
      applyGaugeFieldScaling_long(qdp_inlink, Vh, &gaugeParam, QUDA_STAGGERED_DSLASH, gaugeParam.cpu_prec);
    }
  } else {
    if (dslash_type == QUDA_LAPLACE_DSLASH) {
      construct_gauge_field(qdp_inlink, 0, gaugeParam.cpu_prec, &gaugeParam);
    } else {
      construct_fat_long_gauge_field(qdp_inlink, qdp_longlink, 1, gaugeParam.cpu_prec,&gaugeParam,dslash_type);
    }
    //createSiteLinkCPU(inlink, gaugeParam.cpu_prec, 0); // 0 for no phases
  }

#ifdef GPU_GAUGE_TOOLS
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  double plaq[3];
  loadGaugeQuda(qdp_inlink, &gaugeParam);
  plaqQuda(plaq);
  gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

  if (dslash_type != QUDA_LAPLACE_DSLASH) {
    plaq[0] = -plaq[0]; // correction because we've already put phases on the fields
    plaq[1] = -plaq[1];
    plaq[2] = -plaq[2];
  }

  printf("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
#endif

  // QUDA_STAGGERED_DSLASH follows the same codepath whether or not you 
  // "compute" the fat/long links or not.
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    for (int dir = 0; dir < 4; dir++) {
      memcpy(qdp_fatlink[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
      memset(qdp_longlink[dir],0,V*gaugeSiteSize*gSize);
    }
  } else { // QUDA_ASQTAD_DSLASH

    if (compute_fatlong) {

      ///////////////////////////
      // Set path coefficients //
      ///////////////////////////

      // Reference: "generic_ks/imp_actions/hisq/hisq_action.h",
      // in QHMC: https://github.com/jcosborn/qhmc/blob/master/lib/qopqdp/hisq.c

      double u1 = 1.0/tadpole_factor;
      double u2 = u1*u1;
      double u4 = u2*u2;
      double u6 = u4*u2;

      // First path: create V, W links 
      double act_path_coeff_1[6] = {
           ( 1.0/8.0),                 /* one link */
        u2*( 0.0),                     /* Naik */
        u2*(-1.0/8.0)*0.5,             /* simple staple */
        u4*( 1.0/8.0)*0.25*0.5,        /* displace link in two directions */
        u6*(-1.0/8.0)*0.125*(1.0/6.0), /* displace link in three directions */
        u4*( 0.0)                      /* Lepage term */
      };

      // Second path: create X, long links
      double act_path_coeff_2[6] = {
        (( 1.0/8.0)+(2.0*6.0/16.0)+(1.0/8.0)),   // one link 
            // One link is 1/8 as in fat7 + 2*3/8 for Lepage + 1/8 for Naik 
        (-1.0/24.0),                             // Naik 
        (-1.0/8.0)*0.5,                          // simple staple 
        ( 1.0/8.0)*0.25*0.5,                     // displace link in two directions 
        (-1.0/8.0)*0.125*(1.0/6.0),              // displace link in three directions 
        (-2.0/16.0)                              // Lepage term, correct O(a^2) 2x ASQTAD 
      };

      // Paths for epsilon corrections. Not used if n_naiks = 1.
      double act_path_coeff_3[6] = {
        ( 1.0/8.0),    // one link b/c of Naik 
        (-1.0/24.0),   // Naik 
          0.0,         // simple staple 
          0.0,         // displace link in two directions 
          0.0,         // displace link in three directions 
          0.0          // Lepage term 
      };

      double* act_paths[3] = { act_path_coeff_1, act_path_coeff_2, act_path_coeff_3 };

      // silence some Naik complaining
      (void)n_naiks;


      ////////////////////////////////////
      // Set unitarization coefficients //
      ////////////////////////////////////

      setUnitarizeLinksConstants(unitarize_eps,
               max_allowed_error,
               reunit_allow_svd,
               reunit_svd_only,
               svd_rel_error,
               svd_abs_error);

      //////////////////////////
      // Create the GPU links //
      //////////////////////////

      // Skip eps field for now

      // Note: GPU link creation only works for single and double precision
      computeHISQLinksGPU(qdp_fatlink, qdp_longlink,
                          nullptr, nullptr,
                          qdp_inlink, &gaugeParam, act_paths, 0.0 /* eps_naik */);

      


    } else { //

      for (int dir = 0; dir < 4; dir++) {
        memcpy(qdp_fatlink[dir],qdp_inlink[dir], V*gaugeSiteSize*gSize);
      }
    }

  }

#ifdef GPU_GAUGE_TOOLS
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
    double plaq[3];
    loadGaugeQuda(qdp_fatlink, &gaugeParam);
    plaqQuda(plaq);
    gaugeParam.gauge_order = QUDA_MILC_GAUGE_ORDER;

    plaq[0] = -plaq[0]; // correction because we've already put phases on the fields
    plaq[1] = -plaq[1];
    plaq[2] = -plaq[2];

    printf("Computed fat link plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }
#endif

  // Alright, we've created all the void** links.
  // Create the void* pointers
  reorderQDPtoMILC(milc_fatlink,qdp_fatlink,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);
  reorderQDPtoMILC(milc_longlink,qdp_longlink,V,gaugeSiteSize,gaugeParam.cpu_prec,gaugeParam.cpu_prec);


  ColorSpinorParam csParam;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=5;
  for (int d = 0; d < 4; d++) csParam.x[d] = gaugeParam.X[d];
  bool pc = (inv_param.solution_type == QUDA_MATPC_SOLUTION || inv_param.solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);
  if (pc) csParam.x[0] /= 2;
  csParam.x[4] = Nsrc;

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
  gaugeParam.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(milc_fatlink, gaugeParam);
  cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = (void**)cpuFat->Ghost();

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(milc_longlink, gaugeParam);
  cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
  cpuGaugeField *cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = (void**)cpuLong->Ghost();

#else
  int fat_pad = 0;
  int link_pad = 0;
#endif
  
  gaugeParam.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
    QUDA_SU3_LINKS : QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.ga_pad = fat_pad;
  if (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
    gaugeParam.reconstruct = link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
  } else {
    gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }
  gaugeParam.cuda_prec_precondition = gaugeParam.cuda_prec_sloppy;
  gaugeParam.reconstruct_precondition = gaugeParam.reconstruct_sloppy;
  loadGaugeQuda(milc_fatlink, &gaugeParam);

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
    gaugeParam.ga_pad = link_pad;
    gaugeParam.reconstruct= link_recon;
    gaugeParam.reconstruct_sloppy = link_recon_sloppy;
    gaugeParam.cuda_prec_precondition = gaugeParam.cuda_prec_sloppy;
    gaugeParam.reconstruct_precondition = gaugeParam.reconstruct_sloppy;
    loadGaugeQuda(milc_longlink, &gaugeParam);
  }

  double time0 = -((double)clock()); // Start the timer

  double nrm2=0;
  double src2=0;
  int ret = 0;

  int len = Vh*Nsrc;

  switch(test_type){
    case 0: //even
      if(inv_type == QUDA_GCR_INVERTER){
      	inv_param.inv_type = QUDA_GCR_INVERTER;
      	inv_param.gcrNkrylov = 50;
      }else if(inv_type == QUDA_PCG_INVERTER){
	inv_param.inv_type = QUDA_PCG_INVERTER;
      }
      inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;

      invertQuda(out->V(), in->V(), &inv_param);

      time0 += clock(); 
      time0 /= CLOCKS_PER_SEC;

#ifdef MULTI_GPU    
      matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, 
          out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_EVEN_PARITY);
#else
      matdagmat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_EVEN_PARITY);
#endif

      mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);

      break;

    case 1: //odd
      if(inv_type == QUDA_GCR_INVERTER){
      	inv_param.inv_type = QUDA_GCR_INVERTER;
      	inv_param.gcrNkrylov = 50;
      }else if(inv_type == QUDA_PCG_INVERTER){
	inv_param.inv_type = QUDA_PCG_INVERTER;
      }

      inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
      invertQuda(out->V(), in->V(), &inv_param);	
      time0 += clock(); // stop the timer
      time0 /= CLOCKS_PER_SEC;

#ifdef MULTI_GPU
      matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, 
          out, mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp, QUDA_ODD_PARITY);
#else
      matdagmat(ref->V(), qdp_fatlink, qdp_longlink, out->V(), mass, 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), QUDA_ODD_PARITY);	
#endif
      mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
      src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);

      break;

    case 2: //full spinor

      errorQuda("full spinor not supported\n");
      break;

    case 3: //multi mass CG, even
    case 4:

#define NUM_OFFSETS 12

      {    
        double masses[NUM_OFFSETS] ={0.06, 0.061, 0.064, 0.070, 0.077, 0.081, 0.1, 0.11, 0.12, 0.13, 0.14, 0.205};
        inv_param.num_offset = NUM_OFFSETS;
        // these can be set independently
        for (int i=0; i<inv_param.num_offset; i++) {
          inv_param.tol_offset[i] = inv_param.tol;
          inv_param.tol_hq_offset[i] = inv_param.tol_hq;
        }
        void* outArray[NUM_OFFSETS];

        cpuColorSpinorField* spinorOutArray[NUM_OFFSETS];
        spinorOutArray[0] = out;    
        for(int i=1;i < inv_param.num_offset; i++){
          spinorOutArray[i] = new cpuColorSpinorField(csParam);       
        }

        for(int i=0;i < inv_param.num_offset; i++){
          outArray[i] = spinorOutArray[i]->V();
          inv_param.offset[i] = 4*masses[i]*masses[i];
        }

        if (test_type == 3) {
          inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;      
        } else {
          inv_param.matpc_type = QUDA_MATPC_ODD_ODD;      
        }

        invertMultiShiftQuda(outArray, in->V(), &inv_param);	

        cudaDeviceSynchronize();
        time0 += clock(); // stop the timer
        time0 /= CLOCKS_PER_SEC;

        printfQuda("done: total time = %g secs, compute time = %g, %i iter / %g secs = %g gflops\n", 
            time0, inv_param.secs, inv_param.iter, inv_param.secs,
            inv_param.gflops/inv_param.secs);


        printfQuda("checking the solution\n");
        QudaParity parity = QUDA_INVALID_PARITY;
        if (inv_param.solve_type == QUDA_NORMOP_SOLVE){
          //parity = QUDA_EVENODD_PARITY;
          errorQuda("full parity not supported\n");
        }else if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN){
          parity = QUDA_EVEN_PARITY;
        }else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD){
          parity = QUDA_ODD_PARITY;
        }else{
          errorQuda("ERROR: invalid spinor parity \n");
          exit(1);
        }
        for(int i=0;i < inv_param.num_offset;i++){
          printfQuda("%dth solution: mass=%f, ", i, masses[i]);
#ifdef MULTI_GPU
          matdagmat_mg4dir(ref, qdp_fatlink, qdp_longlink, ghost_fatlink, ghost_longlink, 
              spinorOutArray[i], masses[i], 0, inv_param.cpu_prec, 
              gaugeParam.cpu_prec, tmp, parity);
#else
          matdagmat(ref->V(), qdp_fatlink, qdp_longlink, outArray[i], masses[i], 0, inv_param.cpu_prec, gaugeParam.cpu_prec, tmp->V(), parity);
#endif

	  mxpy(in->V(), ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
	  double nrm2 = norm_2(ref->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
	  double src2 = norm_2(in->V(), len*mySpinorSiteSize, inv_param.cpu_prec);
	  double hqr = sqrt(blas::HeavyQuarkResidualNorm(*spinorOutArray[i], *ref).z);
	  double l2r = sqrt(nrm2/src2);
	
	  printfQuda("Shift %d residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
		   i, inv_param.tol_offset[i], inv_param.true_res_offset[i], l2r, 
		   inv_param.tol_hq_offset[i], inv_param.true_res_hq_offset[i], hqr);

	  //emperical, if the cpu residue is more than 1 order the target accuracy, the it fails to converge
	  if (sqrt(nrm2/src2) > 10*inv_param.tol_offset[i]){
	    ret |=1;
	  }
	}

        for(int i=1; i < inv_param.num_offset;i++) delete spinorOutArray[i];
      }
      break;

    default:
      errorQuda("Unsupported test type");

  }//switch

  if (test_type <=2){

    double hqr = sqrt(blas::HeavyQuarkResidualNorm(*out, *ref).z);
    double l2r = sqrt(nrm2/src2);

    printfQuda("Residuals: (L2 relative) tol %g, QUDA = %g, host = %g; (heavy-quark) tol %g, QUDA = %g, host = %g\n",
        inv_param.tol, inv_param.true_res, l2r, inv_param.tol_hq, inv_param.true_res_hq, hqr);

    printfQuda("done: total time = %g secs, compute time = %g secs, %i iter / %g secs = %g gflops, \n", 
        time0, inv_param.secs, inv_param.iter, inv_param.secs,
        inv_param.gflops/inv_param.secs);
  }

  // Clean up gauge fields, at least
  for (int dir = 0; dir < 4; dir++) {
    if (qdp_inlink[dir] != nullptr) { free(qdp_inlink[dir]); qdp_inlink[dir] = nullptr; }
    if (qdp_fatlink[dir] != nullptr) { free(qdp_fatlink[dir]); qdp_fatlink[dir] = nullptr; }
    if (qdp_longlink[dir] != nullptr) { free(qdp_longlink[dir]); qdp_longlink[dir] = nullptr; }
  }
  if (milc_fatlink != nullptr) { free(milc_fatlink); milc_fatlink = nullptr; }
  if (milc_longlink != nullptr) { free(milc_longlink); milc_longlink = nullptr; }

  if (cpuFat != nullptr) { delete cpuFat; cpuFat = nullptr; }
  if (cpuLong != nullptr) { delete cpuLong; cpuLong = nullptr; }

  end();
  return ret;
}



  static void
end(void) 
{

  delete in;
  delete out;
  delete ref;
  delete tmp;

  

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

  argc_copy = argc;
  argv_copy = argv;

  int ret = invert_test();

  // finalize the communications layer
  finalizeComms();

  return ret;
}
