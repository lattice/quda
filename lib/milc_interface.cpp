#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <quda.h>
#include <quda_milc_interface.h>
#include <quda_internal.h>
#include <color_spinor_field.h>
#include <string.h>
#include <hisq_links_quda.h>
#include <ks_improved_force.h>
#include <dslash_quda.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

#ifdef BUILD_MILC_INTERFACE

// code for NVTX taken from Jiri Kraus' blog post:
// http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/

#ifdef MILC_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
  int color_id = cid; \
  color_id = color_id%num_colors;\
  nvtxEventAttributes_t eventAttrib = {0}; \
  eventAttrib.version = NVTX_VERSION; \
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
  eventAttrib.colorType = NVTX_COLOR_ARGB; \
  eventAttrib.color = colors[color_id]; \
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
  eventAttrib.message.ascii = name; \
  nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


static bool initialized = false;
static int gridDim[4];
static int localDim[4];
static int clover_alloc = 0;
static bool invalidate_quda_gauge = true;
static bool create_quda_gauge = false;


using namespace quda;
using namespace quda::fermion_force;



#define QUDAMILC_VERBOSE 1
template <bool start>
void  inline qudamilc_called(const char* func, QudaVerbosity verb){
#ifdef QUDAMILC_VERBOSE
if (verb >= QUDA_VERBOSE) {
     if(start){
       printf("QUDA_MILC_INTERFACE: %s (called) \n",func);
       PUSH_RANGE(func,1)
     }
     else {
      printf("QUDA_MILC_INTERFACE: %s (return) \n",func);
      POP_RANGE
     }
   }
#endif

}

template <bool start>
void inline qudamilc_called(const char * func){
  qudamilc_called<start>(func, getVerbosity());
}


void qudaInit(QudaInitArgs_t input)
{
  qudamilc_called<true>(__func__);
  if(initialized) return;
  setVerbosityQuda(input.verbosity, "", stdout);
  qudaSetLayout(input.layout);
  initialized = true;
    qudamilc_called<false>(__func__);

}

void qudaFinalize()
{
  qudamilc_called<true>(__func__);
  endQuda();
    qudamilc_called<false>(__func__);
}

/**
 *  Implements a lexicographical mapping of node coordinates to ranks,
 *  with t varying fastest.
 */ 
static int rankFromCoords(const int *coords, void *fdata)
{
  int *dims = static_cast<int *>(fdata);

  int rank = coords[3];
  for (int i = 2; i >= 0; i--) {
    rank = dims[i] * rank + coords[i];
  }
  return rank;
}


void qudaSetLayout(QudaLayout_t input)
{
  int local_dim[4];
  for(int dir=0; dir<4; ++dir){ local_dim[dir] = input.latsize[dir]; }
#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir){ local_dim[dir] /= input.machsize[dir]; }
#endif
  for(int dir=0; dir<4; ++dir){
    if(local_dim[dir]%2 != 0){
      printf("Error: Odd lattice dimensions are not supported\n");
      exit(1);
    }
  }

  for(int dir=0; dir<4; ++dir) localDim[dir] = local_dim[dir];

#ifdef GPU_COMMS
  setKernelPackT(true);
#endif

#ifdef MULTI_GPU
  for(int dir=0; dir<4; ++dir)  gridDim[dir] = input.machsize[dir];
  initCommsGridQuda(4, gridDim, rankFromCoords, (void *)(gridDim));
  static int device = -1;
#else
  for(int dir=0; dir<4; ++dir)  gridDim[dir] = 1;
  static int device = input.device;
#endif

  initQuda(device);
}


void qudaHisqParamsInit(QudaHisqParams_t params)
{

  static bool initialized = false;

  if(initialized) return;
  qudamilc_called<true>(__func__);

  const bool reunit_allow_svd = (params.reunit_allow_svd) ? true : false;
  const bool reunit_svd_only  = (params.reunit_svd_only) ? true : false;


  const double unitarize_eps = 1e-14;
  const double max_error = 1e-10;

#ifdef GPU_HISQ_FORCE
  quda::fermion_force::setUnitarizeForceConstants(unitarize_eps,
      params.force_filter,
      max_error,
      reunit_allow_svd,
      reunit_svd_only,
      params.reunit_svd_rel_error,
      params.reunit_svd_abs_error);
#endif

#ifdef GPU_UNITARIZE
  setUnitarizeLinksConstants(unitarize_eps,
      max_error,
      reunit_allow_svd,
      reunit_svd_only,
      params.reunit_svd_rel_error,
      params.reunit_svd_abs_error);
#endif // UNITARIZE_GPU                            

  initialized = true;
  qudamilc_called<false>(__func__);
  return;
}



static QudaGaugeParam newMILCGaugeParam(const int* dim, QudaPrecision prec, QudaLinkType link_type)
{
  QudaGaugeParam gParam = newQudaGaugeParam();
  for(int dir=0; dir<4; ++dir) gParam.X[dir] = dim[dir];
  gParam.cuda_prec_sloppy = gParam.cpu_prec = gParam.cuda_prec = prec;
  gParam.type = link_type;

  gParam.reconstruct_sloppy = gParam.reconstruct = ((link_type == QUDA_SU3_LINKS) ? QUDA_RECONSTRUCT_12 : QUDA_RECONSTRUCT_NO);
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
  return gParam;
}


static  void invalidateGaugeQuda() {
  freeGaugeQuda();
  invalidate_quda_gauge = true;
}



#ifdef GPU_FATLINK

void qudaLoadKSLink(int prec, QudaFatLinkArgs_t fatlink_args,
    const double act_path_coeff[6], void* inlink, void* fatlink, void* longlink)
{
  qudamilc_called<true>(__func__);
 // create_quda_gauge = true;

#ifdef MULTI_GPU  
  QudaComputeFatMethod method = QUDA_COMPUTE_FAT_EXTENDED_VOLUME;
#else
  QudaComputeFatMethod method = QUDA_COMPUTE_FAT_STANDARD;
#endif

  QudaGaugeParam param = newMILCGaugeParam(localDim,
      (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);

  computeKSLinkQuda(fatlink, longlink, NULL, inlink, const_cast<double*>(act_path_coeff), &param, method);
  qudamilc_called<false>(__func__);
  // require loadGaugeQuda to be called in subsequent solve
//  invalidateGaugeQuda(); 
}



void qudaLoadUnitarizedLink(int prec, QudaFatLinkArgs_t fatlink_args,
    const double act_path_coeff[6], void* inlink, void* fatlink, void* ulink)
{
  qudamilc_called<true>(__func__);
#ifdef MULTI_GPU  
  QudaComputeFatMethod method = QUDA_COMPUTE_FAT_EXTENDED_VOLUME;
#else
  QudaComputeFatMethod method = QUDA_COMPUTE_FAT_STANDARD;
#endif

  QudaGaugeParam param = newMILCGaugeParam(localDim,
      (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);

  computeKSLinkQuda(fatlink, NULL, ulink, inlink, const_cast<double*>(act_path_coeff), &param, method);
    qudamilc_called<false>(__func__);
  return;
}

#endif


#ifdef GPU_HISQ_FORCE

void qudaHisqForce(int prec, const double level2_coeff[6], const double fat7_coeff[6],
    const void* const staple_src[4], const void* const one_link_src[4], const void* const naik_src[4],
    const void* const w_link, const void* const v_link, const void* const u_link,
    void* const milc_momentum)
{
  qudamilc_called<true>(__func__);

  QudaGaugeParam gParam = newMILCGaugeParam(localDim,
      (prec==1) ?  QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);

  long long flops;
  computeHISQForceQuda(milc_momentum, &flops, level2_coeff, fat7_coeff,
      staple_src, one_link_src, naik_src,
      w_link, v_link, u_link, &gParam);
  qudamilc_called<false>(__func__);
  return;
}


void qudaAsqtadForce(int prec, const double act_path_coeff[6], 
    const void* const one_link_src[4], const void* const naik_src[4],
    const void* const link, void* const milc_momentum)
{
  qudamilc_called<true>(__func__);


  QudaGaugeParam gParam = newMILCGaugeParam(localDim,
      (prec==1) ?  QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);

  long long flops;
  computeAsqtadForceQuda(milc_momentum, &flops, act_path_coeff, one_link_src, naik_src, link, &gParam);

  qudamilc_called<false>(__func__);
  return;
}



void qudaComputeOprod(int prec, int num_terms, double** coeff,
    void** quark_field, void* oprod[2])
{
    qudamilc_called<true>(__func__);
  QudaGaugeParam oprodParam = newMILCGaugeParam(localDim,
      (prec==1) ?  QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);

  computeStaggeredOprodQuda(oprod, quark_field, num_terms, coeff, &oprodParam);
  qudamilc_called<false>(__func__);
  return;
}

#endif // GPU_HISQ_FORCE

#ifdef GPU_GAUGE_FORCE
void  qudaUpdateU(int prec, double eps, void* momentum, void* link)
{
  qudamilc_called<true>(__func__);
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim,
      (prec==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);


  updateGaugeFieldQuda(link, momentum, eps, 0, 0, &gaugeParam);
  qudamilc_called<false>(__func__);
  return;
}


// gauge force code
static int getVolume(const int dim[4])
{
  return dim[0]*dim[1]*dim[2]*dim[3];
}


  template<class Real>
static void setLoopCoeffs(const double milc_loop_coeff[3],
    Real loop_coeff[48])
{
  // 6 plaquette terms
  for(int i=0; i<6; ++i) loop_coeff[i] = milc_loop_coeff[0];      

  // 18 rectangle terms
  for(int i=0; i<18; ++i) loop_coeff[i+6] = milc_loop_coeff[1];

  for(int i=0; i<24; ++i) loop_coeff[i+24] = milc_loop_coeff[2];

  return;
}



static inline int opp(int dir){
  return 7-dir;
}


static void createGaugeForcePaths(int **paths, int dir){

  int index=0;
  // Plaquette paths
  for(int i=0; i<4; ++i){
    if(i==dir) continue;
    paths[index][0] = i;        paths[index][1] = opp(dir);   paths[index++][2] = opp(i);
    paths[index][0] = opp(i);   paths[index][1] = opp(dir);   paths[index++][2] = i;
  }

  // Rectangle Paths
  for(int i=0; i<4; ++i){
    if(i==dir) continue;
    paths[index][0] = paths[index][1] = i;       paths[index][2] = opp(dir); paths[index][3] = paths[index][4] = opp(i);
    index++;  
    paths[index][0] = paths[index][1] = opp(i);  paths[index][2] = opp(dir); paths[index][3] = paths[index][4] = i;
    index++;
    paths[index][0] = dir; paths[index][1] = i; paths[index][2] = paths[index][3] = opp(dir); paths[index][4] = opp(i);
    index++;
    paths[index][0] = dir; paths[index][1] = opp(i); paths[index][2] = paths[index][3] = opp(dir); paths[index][4] = i;
    index++;
    paths[index][0] = i;  paths[index][1] = paths[index][2] = opp(dir); paths[index][3] = opp(i); paths[index][4] = dir; 
    index++;
    paths[index][0] = opp(i);  paths[index][1] = paths[index][2] = opp(dir); paths[index][3] = i; paths[index][4] = dir; 
    index++;
  }


  // Staple paths
  for(int i=0; i<4; ++i){
    for(int j=0; j<4; ++j){
      if(i==dir || j==dir || i==j) continue;

      paths[index][0] = i; paths[index][1] = j; paths[index][2] = opp(dir); paths[index][3] = opp(i), paths[index][4] = opp(j);
      index++;
      paths[index][0] = i; paths[index][1] = opp(j); paths[index][2] = opp(dir); paths[index][3] = opp(i), paths[index][4] = j;
      index++;
      paths[index][0] = opp(i); paths[index][1] = j; paths[index][2] = opp(dir); paths[index][3] = i, paths[index][4] = opp(j);
      index++;
      paths[index][0] = opp(i); paths[index][1] = opp(j); paths[index][2] = opp(dir); paths[index][3] = i, paths[index][4] = j;
      index++;
    }
  }

}



void qudaGaugeForce( int precision,
    int num_loop_types,
    double milc_loop_coeff[3],
    double eb3,
    void* milc_sitelink,
    void* milc_momentum
    )
{
  qudamilc_called<true>(__func__);
  const int numPaths = 48;
  QudaGaugeParam qudaGaugeParam = newMILCGaugeParam(localDim, 
      (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION, 
      QUDA_SU3_LINKS);

  static double loop_coeff[numPaths];
  static int length[numPaths];

  if((milc_loop_coeff[0] != loop_coeff[0]) ||
      (milc_loop_coeff[1] != loop_coeff[6]) ||
      (milc_loop_coeff[2] != loop_coeff[24]) ){

    setLoopCoeffs(milc_loop_coeff, loop_coeff);
    for(int i=0; i<6; ++i)  length[i] = 3;
    for(int i=6; i<48; ++i) length[i] = 5;
  }


  int** input_path_buf[4];
  for(int dir=0; dir<4; ++dir){
    input_path_buf[dir] = new int*[numPaths];
    for(int i=0; i<numPaths; ++i){
      input_path_buf[dir][i] = new int[length[i]];
    }
    createGaugeForcePaths(input_path_buf[dir], dir);
  }

  int max_length = 6;
  double timeinfo[3];

  memset(milc_momentum, 0, 4*getVolume(localDim)*10*qudaGaugeParam.cpu_prec);
  computeGaugeForceQuda(milc_momentum, milc_sitelink,  input_path_buf, length,
      loop_coeff, numPaths, max_length, eb3, 
      &qudaGaugeParam, timeinfo);

  for(int dir=0; dir<4; ++dir){
    for(int i=0; i<numPaths; ++i){
      delete[] input_path_buf[dir][i];
    }
    delete[] input_path_buf[dir];
  }
  qudamilc_called<false>(__func__);
  return;
}

#endif // GPU_GAUGE_FORCE

static int getFatLinkPadding(const int dim[4])
{
  int padding = MAX(dim[1]*dim[2]*dim[3]/2, dim[0]*dim[2]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[3]/2);
  padding = MAX(padding, dim[0]*dim[1]*dim[2]/2);
  return padding;
}


#ifdef GPU_STAGGERED_DIRAC
// set the params for the single mass solver
static void setInvertParams(const int dim[4],
    QudaPrecision cpu_prec,
    QudaPrecision cuda_prec,
    QudaPrecision cuda_prec_sloppy,
    QudaPrecision cuda_prec_precondition,
    double mass,
    double target_residual,
    double target_residual_hq,
    int maxiter,
    double reliable_delta,
    QudaParity parity,
    QudaVerbosity verbosity,
    QudaInverterType inverter,
    QudaInvertParam *invertParam)
{
  invertParam->use_sloppy_partial_accumulator = 0;
  invertParam->verbosity = verbosity;
  invertParam->mass = mass;
  invertParam->tol = target_residual;
  invertParam->tol_hq =target_residual_hq;
  invertParam->num_offset = 0;

  invertParam->inv_type = inverter;
  invertParam->maxiter = maxiter;
  invertParam->reliable_delta = reliable_delta;

  invertParam->mass_normalization = QUDA_MASS_NORMALIZATION;
  invertParam->cpu_prec = cpu_prec;
  invertParam->cuda_prec = cuda_prec;
  invertParam->cuda_prec_sloppy = cuda_prec_sloppy;

  invertParam->solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  invertParam->solve_type = QUDA_NORMEQ_PC_SOLVE; 
  invertParam->preserve_source = QUDA_PRESERVE_SOURCE_YES;
  invertParam->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // not used, but required by the code.
  invertParam->dirac_order = QUDA_DIRAC_ORDER;

  invertParam->dslash_type = QUDA_ASQTAD_DSLASH;
  invertParam->tune = QUDA_TUNE_YES;
  invertParam->gflops = 0.0;

  invertParam->input_location = QUDA_CPU_FIELD_LOCATION;
  invertParam->output_location = QUDA_CPU_FIELD_LOCATION;


  if(parity == QUDA_EVEN_PARITY){ // even parity
    invertParam->matpc_type = QUDA_MATPC_EVEN_EVEN;
  }else if(parity == QUDA_ODD_PARITY){
    invertParam->matpc_type = QUDA_MATPC_ODD_ODD;
  }else{
    errorQuda("Invalid parity\n");
    exit(1);
  }

  invertParam->dagger = QUDA_DAG_NO;
  invertParam->sp_pad = dim[0]*dim[1]*dim[2]/2;
  invertParam->use_init_guess = QUDA_USE_INIT_GUESS_YES; 

  // for the preconditioner
  invertParam->inv_type_precondition = QUDA_CG_INVERTER;
  invertParam->tol_precondition = 1e-1;
  invertParam->maxiter_precondition = 2;
  invertParam->verbosity_precondition = QUDA_SILENT;
  invertParam->cuda_prec_precondition = cuda_prec_precondition;


  return;
}




// Set params for the multi-mass solver.
static void setInvertParams(const int dim[4],
    QudaPrecision cpu_prec,
    QudaPrecision cuda_prec,
    QudaPrecision cuda_prec_sloppy,
    QudaPrecision cuda_prec_precondition,
    int num_offset,
    const double offset[],
    const double target_residual_offset[],
    const double target_residual_hq_offset[],
    int maxiter,
    double reliable_delta,
    QudaParity parity,
    QudaVerbosity verbosity,
    QudaInverterType inverter,
    QudaInvertParam *invertParam)
{

  const double null_mass = -1;
  const double null_residual = -1;


  setInvertParams(dim, cpu_prec, cuda_prec, cuda_prec_sloppy, cuda_prec_precondition,
      null_mass, null_residual, null_residual, maxiter, reliable_delta, parity, verbosity, inverter, invertParam);

  invertParam->num_offset = num_offset;
  for(int i=0; i<num_offset; ++i){
    invertParam->offset[i] = offset[i];
    invertParam->tol_offset[i] = target_residual_offset[i];
    //if(invertParam->residual_type & QUDA_HEAVY_QUARK_RESIDUAL){
      invertParam->tol_hq_offset[i] = target_residual_hq_offset[i];
    //}
  }
  return;
}


static void setGaugeParams(const int dim[4],
    QudaPrecision cpu_prec,
    QudaPrecision cuda_prec,
    QudaPrecision cuda_prec_sloppy,
    QudaPrecision cuda_prec_precondition,
    const double tadpole,
    QudaGaugeParam *gaugeParam)   
{

  for(int dir=0; dir<4; ++dir){
    gaugeParam->X[dir] = dim[dir];
  }

  gaugeParam->cpu_prec = cpu_prec;
  gaugeParam->cuda_prec = cuda_prec;
  gaugeParam->cuda_prec_sloppy = cuda_prec_sloppy;
  gaugeParam->reconstruct = QUDA_RECONSTRUCT_NO;
  gaugeParam->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;

  gaugeParam->gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam->anisotropy = 1.0;
  gaugeParam->tadpole_coeff = tadpole;
  gaugeParam->t_boundary = QUDA_PERIODIC_T; // anti-periodic boundary conditions are built into the gauge field
  gaugeParam->gauge_order = QUDA_MILC_GAUGE_ORDER; 
  gaugeParam->ga_pad = getFatLinkPadding(dim);
  gaugeParam->scale = -1.0/(24.0*gaugeParam->tadpole_coeff*gaugeParam->tadpole_coeff);


  // preconditioning...
  gaugeParam->cuda_prec_precondition = cuda_prec_precondition;
  gaugeParam->reconstruct_precondition = QUDA_RECONSTRUCT_NO;

  return;
}



static void setColorSpinorParams(const int dim[4],
    QudaPrecision precision,
    ColorSpinorParam* param)
{

  param->nColor = 3;
  param->nSpin = 1;
  param->nDim = 4;

  for(int dir=0; dir<4; ++dir) param->x[dir] = dim[dir];
  param->x[0] /= 2; 

  param->precision = precision;
  param->pad = 0;
  param->siteSubset = QUDA_PARITY_SITE_SUBSET;
  param->siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param->fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  param->gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless, but required by the code.
  param->create = QUDA_ZERO_FIELD_CREATE;
  return;
} 


static size_t getColorVectorOffset(QudaParity local_parity, bool even_odd_exchange, const int dim[4])
{
  size_t offset;
  int volume = dim[0]*dim[1]*dim[2]*dim[3];  

  if(local_parity == QUDA_EVEN_PARITY){
    offset = even_odd_exchange ? volume*6/2 : 0;
  }else{
    offset = even_odd_exchange ? 0 : volume*6/2;
  }
  return offset;
}


void qudaMultishiftInvert(int external_precision, 
    int quda_precision,
    int num_offsets,
    double* const offset,
    QudaInvertArgs_t inv_args,
    const double target_residual[], 
    const double target_fermilab_residual[],
    const void* const fatlink,
    const void* const longlink,
    const double tadpole,
    void* source,
    void** solutionArray,
    double* const final_residual,
    double* const final_fermilab_residual,
    int *num_iters)
{

  // for(int i=0; i<num_offsets; ++i){
  //   if(target_residual[i] == 0){
  //     errorQuda("qudaMultishiftInvert: target residual cannot be zero\n");
  //     exit(1);
  //   }
  // }
  // for(int i=0; i<num_offsets; ++i){
    static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

    if(target_residual[0] == 0){
      errorQuda("qudaMultishiftInvert: target residual cannot be zero\n");
      exit(1);
    }
  // }

  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const bool use_mixed_precision = (((quda_precision==2) && inv_args.mixed_precision) || 
                                     ((quda_precision==1) && (inv_args.mixed_precision==2)) ) ? true : false;
  QudaPrecision device_precision_sloppy; 
  if(inv_args.mixed_precision == 2){
    device_precision_sloppy = QUDA_HALF_PRECISION;
  }else if(inv_args.mixed_precision == 1){
    device_precision_sloppy = QUDA_SINGLE_PRECISION;
  }else{
    device_precision_sloppy = device_precision; 
  }

  QudaPrecision device_precision_precondition = device_precision_sloppy;





/*
  if(verbosity >= QUDA_VERBOSE){ 
    if(quda_precision == 2){
      printfQuda("Using %s double-precision multi-mass inverter\n", use_mixed_precision?"mixed":"pure");
    }else if(quda_precision == 1){
      printfQuda("Using %s single-precision multi-mass inverter\n", use_mixed_precision?"mixed":"pure");
    }else{
      errorQuda("Unrecognised precision\n");
      exit(1);
    }
  }
*/

  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  setGaugeParams(localDim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition, tadpole, &gaugeParam);

  QudaInvertParam invertParam = newQudaInvertParam();

  invertParam.residual_type = static_cast<QudaResidualType_s>(0);
  invertParam.residual_type = (target_residual[0] != 0) ? static_cast<QudaResidualType_s> ( invertParam.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : invertParam.residual_type;
  invertParam.residual_type = (target_fermilab_residual[0] != 0) ? static_cast<QudaResidualType_s> (invertParam.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : invertParam.residual_type;
  if (invertParam.residual_type ==QUDA_L2_RELATIVE_RESIDUAL) 
    printfQuda("Using QUDA_L2_RELATIVE_RESIDUAL only");
  else{
    if (invertParam.residual_type &QUDA_L2_RELATIVE_RESIDUAL) 
      printfQuda("Using QUDA_L2_RELATIVE_RESIDUAL");      
     if (invertParam.residual_type &QUDA_L2_RELATIVE_RESIDUAL) 
      printfQuda("Using QUDA_HEAVY_QUARK_RESIDUAL"); 
  }

  invertParam.use_sloppy_partial_accumulator = 0;


  const double ignore_mass = 1.0;

  QudaParity local_parity = inv_args.evenodd;
  {
    // need to set this to zero until issue #146 is fixed
    const double reliable_delta = (use_mixed_precision ? 1e-1 :0.0);//1e-1;
    setInvertParams(localDim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition,
        num_offsets, offset, target_residual, target_fermilab_residual, 
        inv_args.max_iter, reliable_delta, local_parity, verbosity, QUDA_CG_INVERTER, &invertParam);
  }  

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);

  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;

//  if(invalidate_quda_gauge){
    const int fat_pad  = getFatLinkPadding(localDim);
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = fat_pad;  // don't know if this is correct
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam); 

    const int long_pad = 3*fat_pad;
    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.ga_pad = long_pad; 
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);

   // invalidate_quda_gauge = false;
//  }

  void** sln_pointer = (void**)malloc(num_offsets*sizeof(void*));
  int quark_offset = getColorVectorOffset(local_parity, false, gaugeParam.X)*host_precision;
  void* src_pointer = (char*)source + quark_offset;

  for(int i=0; i<num_offsets; ++i) sln_pointer[i] = (char*)solutionArray[i] + quark_offset;

  invertMultiShiftQuda(sln_pointer, src_pointer, &invertParam);
  free(sln_pointer); 

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  for(int i=0; i<num_offsets; ++i){
    final_residual[i] = invertParam.true_res_offset[i];
    final_fermilab_residual[i] = invertParam.true_res_hq_offset[i];
  } // end loop over number of offsets

  freeGaugeQuda();
 // if(!create_quda_gauge) invalidateGaugeQuda();
  qudamilc_called<false>(__func__, verbosity);
  return;
} // qudaMultiShiftInvert




void qudaInvert(int external_precision,
    int quda_precision,
    double mass,
    QudaInvertArgs_t inv_args,
    double target_residual, 
    double target_fermilab_residual,
    const void* const fatlink,
    const void* const longlink,
    const double tadpole,
    void* source,
    void* solution,
    double* const final_residual,
    double* const final_fermilab_residual,
    int* num_iters)
{

  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  if(target_fermilab_residual == 0 && target_residual == 0){
    errorQuda("qudaInvert: requesting zero residual\n");
    exit(1);
  }

  const bool use_mixed_precision = (((quda_precision==2) && inv_args.mixed_precision) || 
                                     ((quda_precision==1) && (inv_args.mixed_precision==2) ) ) ? true : false;

  // static const QudaVerbosity verbosity = getVerbosity();
  QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;

  if(inv_args.mixed_precision == 2){
    device_precision_sloppy = QUDA_HALF_PRECISION;
  }else if(inv_args.mixed_precision == 1){
    device_precision_sloppy = QUDA_SINGLE_PRECISION;
  }else{
    device_precision_sloppy = device_precision;
  }
  


  QudaPrecision device_precision_precondition = device_precision_sloppy;
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  // a basic set routine for the gauge parameters
  setGaugeParams(localDim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition, tadpole, &gaugeParam);
  QudaInvertParam invertParam = newQudaInvertParam();

  invertParam.residual_type = static_cast<QudaResidualType_s>(0);
  invertParam.residual_type = (target_residual != 0) ? static_cast<QudaResidualType_s> ( invertParam.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : invertParam.residual_type;
  invertParam.residual_type = (target_fermilab_residual != 0) ? static_cast<QudaResidualType_s> (invertParam.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : invertParam.residual_type;
  

  QudaParity local_parity = inv_args.evenodd;
  //double& target_res = (invertParam.residual_type == QUDA_L2_RELATIVE_RESIDUAL) ? target_residual : target_fermilab_residual;
  double& target_res = target_residual;
  double& target_res_hq = target_fermilab_residual;
  const double reliable_delta = 1e-1;

  setInvertParams(localDim, host_precision, device_precision, device_precision_sloppy, device_precision_precondition,
      mass, target_res, target_res_hq, inv_args.max_iter, reliable_delta, local_parity, verbosity, QUDA_CG_INVERTER, &invertParam);
  invertParam.use_sloppy_partial_accumulator = 0;

  ColorSpinorParam csParam;
  setColorSpinorParams(localDim, host_precision, &csParam);


  const QudaPrecision milc_precision = (external_precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION; 
  const int fat_pad  = getFatLinkPadding(localDim);
  const int long_pad = 3*fat_pad;

//  if(invalidate_quda_gauge){
    gaugeParam.type = QUDA_GENERAL_LINKS;
    gaugeParam.ga_pad = fat_pad; 
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(fatlink), &gaugeParam); 

    gaugeParam.type = QUDA_THREE_LINKS;
    gaugeParam.ga_pad = long_pad; 
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(const_cast<void*>(longlink), &gaugeParam);

    invalidate_quda_gauge = false;
//  }

  int quark_offset = getColorVectorOffset(local_parity, false, gaugeParam.X);

  invertQuda(((char*)solution + quark_offset*host_precision), 
      ((char*)source + quark_offset*host_precision), 
      &invertParam); 

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

  freeGaugeQuda();
  qudamilc_called<false>(__func__, verbosity);
//  if(!create_quda_gauge) invalidateGaugeQuda();
  return;
} // qudaInvert

#endif

#ifdef GPU_CLOVER_DIRAC

static inline void* createExtendedGaugeField(void* gauge, int geometry, int precision, int resident)
{
  qudamilc_called<true>(__func__);
  QudaPrecision qudaPrecision = (precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION; 
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim, qudaPrecision,
      (geometry==1) ? QUDA_GENERAL_LINKS : QUDA_SU3_LINKS);
  gaugeParam.use_resident_gauge = resident ? 1 : 0;
  qudamilc_called<false>(__func__);
  return createExtendedGaugeFieldQuda(gauge, geometry, &gaugeParam);
}

void* qudaCreateExtendedGaugeField(void* gauge, int geometry, int precision)
{
  return createExtendedGaugeField(gauge, geometry, precision, 0);
}

void* qudaResidentExtendedGaugeField(void* gauge, int geometry, int precision)
{
  return createExtendedGaugeField(gauge, geometry, precision, 1);
}


void* qudaCreateGaugeField(void* gauge, int geometry, int precision)
{
  qudamilc_called<true>(__func__);
  QudaPrecision qudaPrecision = (precision==2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim, qudaPrecision,
      (geometry==1) ? QUDA_GENERAL_LINKS : QUDA_SU3_LINKS);
  qudamilc_called<false>(__func__);
  return createGaugeFieldQuda(gauge, geometry, &gaugeParam);
}


void qudaSaveGaugeField(void* gauge, void* inGauge)
{
  qudamilc_called<true>(__func__);
  cudaGaugeField* cudaGauge = reinterpret_cast<cudaGaugeField*>(inGauge);
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim, cudaGauge->Precision(), QUDA_GENERAL_LINKS);
  saveGaugeFieldQuda(gauge, inGauge, &gaugeParam);
  qudamilc_called<false>(__func__);
  return;
}


void qudaDestroyGaugeField(void* gauge)
{
  qudamilc_called<true>(__func__);
  destroyGaugeFieldQuda(gauge);
    qudamilc_called<false>(__func__);
  return;
}


void qudaCloverTrace(void* out, void* clover, int mu, int nu)
{
  qudamilc_called<true>(__func__);
  computeCloverTraceQuda(out, clover, mu, nu, const_cast<int*>(localDim));
  qudamilc_called<false>(__func__);
  return;
}



void qudaCloverDerivative(void* out, void* gauge, void* oprod, int mu, int nu, double coeff, int precision, int parity, int conjugate)
{
  qudamilc_called<true>(__func__);
  QudaParity qudaParity = (parity==2) ? QUDA_EVEN_PARITY : QUDA_ODD_PARITY;
  QudaGaugeParam gaugeParam = newMILCGaugeParam(localDim, 
      (precision==1) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION,
      QUDA_GENERAL_LINKS);

  computeCloverDerivativeQuda(out, gauge, oprod, mu, nu, coeff, qudaParity, &gaugeParam, conjugate);
  qudamilc_called<false>(__func__);
  return;
}



void setGaugeParams(QudaGaugeParam &gaugeParam, const int dim[4], QudaInvertArgs_t &inv_args,
    int external_precision, int quda_precision) {

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy; 

  if(inv_args.mixed_precision == 2){
    device_precision_sloppy = QUDA_HALF_PRECISION;
  }else if(inv_args.mixed_precision == 1){
    device_precision_sloppy = QUDA_SINGLE_PRECISION;
  }else{
    device_precision_sloppy = device_precision;
  }

  for(int dir=0; dir<4; ++dir) gaugeParam.X[dir] = dim[dir];

  gaugeParam.anisotropy               = 1.0;
  gaugeParam.type                     = QUDA_WILSON_LINKS;
  gaugeParam.gauge_order              = QUDA_MILC_GAUGE_ORDER; 

  // Check the boundary conditions
  // Can't have twisted or anti-periodic boundary conditions in the spatial 
  // directions with 12 reconstruct at the moment.
  bool trivial_phase = true;
  for(int dir=0; dir<3; ++dir){
    if(inv_args.boundary_phase[dir] != 0) trivial_phase = false;
  }
  if(inv_args.boundary_phase[3] != 0 && inv_args.boundary_phase[3] != 1) trivial_phase = false;  

  if(trivial_phase){
    gaugeParam.t_boundary               = (inv_args.boundary_phase[3]) ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_12; 
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_12;
  }else{
    gaugeParam.t_boundary               = QUDA_PERIODIC_T;
    gaugeParam.reconstruct              = QUDA_RECONSTRUCT_NO;
    gaugeParam.reconstruct_sloppy       = QUDA_RECONSTRUCT_NO;
  }

  gaugeParam.cpu_prec                 = host_precision;
  gaugeParam.cuda_prec                = device_precision;
  gaugeParam.cuda_prec_sloppy         = device_precision_sloppy;
  gaugeParam.cuda_prec_precondition   = device_precision_sloppy;
  gaugeParam.gauge_fix                = QUDA_GAUGE_FIXED_NO;
  gaugeParam.ga_pad                   = getFatLinkPadding(dim);
}



void setInvertParam(QudaInvertParam &invertParam, QudaInvertArgs_t &inv_args, 
		    int external_precision, int quda_precision, double kappa, double reliable_delta) {

  const QudaPrecision host_precision = (external_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  const QudaPrecision device_precision = (quda_precision == 2) ? QUDA_DOUBLE_PRECISION : QUDA_SINGLE_PRECISION;
  QudaPrecision device_precision_sloppy;
  if(inv_args.mixed_precision == 2){
    device_precision_sloppy = QUDA_HALF_PRECISION;
  }else if(inv_args.mixed_precision == 1){
    device_precision_sloppy = QUDA_SINGLE_PRECISION;
  }else{
    device_precision_sloppy = device_precision;
  }
 
  static const QudaVerbosity verbosity = getVerbosity();

  invertParam.dslash_type                   = QUDA_CLOVER_WILSON_DSLASH;
  invertParam.kappa                         = kappa;
  invertParam.dagger                        = QUDA_DAG_NO;
  invertParam.mass_normalization            = QUDA_KAPPA_NORMALIZATION;
  invertParam.gcrNkrylov                    = 30; 
  invertParam.reliable_delta                = reliable_delta; 
  invertParam.maxiter                       = inv_args.max_iter;

  invertParam.cuda_prec_precondition        = device_precision_sloppy;
  invertParam.verbosity_precondition        = verbosity;
  invertParam.verbosity        = verbosity;
  invertParam.cpu_prec                      = host_precision;
  invertParam.cuda_prec                     = device_precision;
  invertParam.cuda_prec_sloppy              = device_precision_sloppy;
  invertParam.preserve_source               = QUDA_PRESERVE_SOURCE_NO;
  invertParam.gamma_basis                   = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  invertParam.dirac_order                   = QUDA_DIRAC_ORDER;
  invertParam.tune                          = QUDA_TUNE_YES;
  invertParam.sp_pad                        = 0;
  invertParam.cl_pad                        = 0;
  invertParam.clover_cpu_prec               = host_precision;
  invertParam.clover_cuda_prec              = device_precision;
  invertParam.clover_cuda_prec_sloppy       = device_precision_sloppy;
  invertParam.clover_cuda_prec_precondition = device_precision_sloppy;
  invertParam.clover_order                  = QUDA_PACKED_CLOVER_ORDER;
}


void qudaLoadGaugeField(int external_precision, 
    int quda_precision,
    QudaInvertArgs_t inv_args,
    const void* milc_link) {
  qudamilc_called<true>(__func__);
  QudaGaugeParam gaugeParam = newQudaGaugeParam();
  setGaugeParams(gaugeParam, localDim,  inv_args, external_precision, quda_precision);

  loadGaugeQuda(const_cast<void*>(milc_link), &gaugeParam);
    qudamilc_called<false>(__func__);
} // qudaLoadGaugeField


void qudaFreeGaugeField() {
    qudamilc_called<true>(__func__);
  freeGaugeQuda();
    qudamilc_called<false>(__func__);
} // qudaFreeGaugeField


void qudaLoadCloverField(int external_precision, 
    int quda_precision,
    QudaInvertArgs_t inv_args,
    void* milc_clover, 
    void* milc_clover_inv,
    QudaSolutionType solution_type,
    QudaSolveType solve_type,
    double clover_coeff,
    int compute_trlog,
    double *trlog) {
  qudamilc_called<true>(__func__);
  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, 0.0, 0.0);
  invertParam.solution_type = solution_type;
  invertParam.solve_type = solve_type;
  invertParam.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
  invertParam.compute_clover_trlog = compute_trlog;
  invertParam.clover_coeff = clover_coeff;

  if(invertParam.dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (clover_alloc == 0) {
      loadCloverQuda(milc_clover, milc_clover_inv, &invertParam);
      clover_alloc = 1;
    } else {
      errorQuda("Clover term already allocated");
    }
  }

  if (compute_trlog) {
    trlog[0] = invertParam.trlogA[0];
    trlog[1] = invertParam.trlogA[1];
  }
  qudamilc_called<false>(__func__);
} // qudaLoadCoverField



void qudaFreeCloverField() {
  qudamilc_called<true>(__func__);
  if (clover_alloc==1) {
    freeCloverQuda();
    clover_alloc = 0;
  } else {
    errorQuda("Trying to free non-allocated clover term");
  }
  qudamilc_called<false>(__func__);
} // qudaFreeCloverField



void qudaCloverInvert(int external_precision, 
    int quda_precision,
    double kappa,
    double clover_coeff,
    QudaInvertArgs_t inv_args,
    double target_residual,
    double target_fermilab_residual,
    const void* link,
    void* clover, // could be stored in Milc format
    void* cloverInverse,
    void* source,
    void* solution,
    double* const final_residual, 
    double* const final_fermilab_residual,
    int* num_iters)
{
  qudamilc_called<true>(__func__);
  if(target_fermilab_residual == 0 && target_residual == 0){
    errorQuda("qudaCloverInvert: requesting zero residual\n");
    exit(1);
  }

  qudaLoadGaugeField(external_precision, quda_precision, inv_args, link);

//  double clover_coeff = 0.0;
  qudaLoadCloverField(external_precision, quda_precision, inv_args, clover, cloverInverse,
      QUDA_MAT_SOLUTION, QUDA_DIRECT_PC_SOLVE, clover_coeff, 0, 0);

  double reliable_delta = 1e-1;

  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, kappa, reliable_delta);
  invertParam.residual_type = static_cast<QudaResidualType_s>(0);
  invertParam.residual_type = (target_residual != 0) ? static_cast<QudaResidualType_s> ( invertParam.residual_type | QUDA_L2_RELATIVE_RESIDUAL) : invertParam.residual_type;
  invertParam.residual_type = (target_fermilab_residual != 0) ? static_cast<QudaResidualType_s> (invertParam.residual_type | QUDA_HEAVY_QUARK_RESIDUAL) : invertParam.residual_type;

  invertParam.tol =  target_residual;
  invertParam.tol_hq = target_fermilab_residual;

  // solution types
  invertParam.solution_type      = QUDA_MAT_SOLUTION;
  invertParam.solve_type         = QUDA_DIRECT_PC_SOLVE;
  invertParam.inv_type           = QUDA_BICGSTAB_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_ODD_ODD;

  invertQuda(solution, source, &invertParam); 
  *num_iters = invertParam.iter;
  *final_residual = invertParam.true_res;
  *final_fermilab_residual = invertParam.true_res_hq;

  qudaFreeGaugeField();
  qudaFreeCloverField();
  qudamilc_called<false>(__func__);
  return;
} // qudaCloverInvert


void qudaCloverMultishiftInvert(int external_precision, 
    int quda_precision,
    int num_offsets,
    double* const offset,
    double kappa,
    double clover_coeff,
    QudaInvertArgs_t inv_args,
    const double* target_residual_offset,
    const void* milc_link,
    void* milc_clover, 
    void* milc_clover_inv,
    void* source,
    void** solutionArray,
    double* const final_residual, 
    int* num_iters)
{

  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  for(int i=0; i<num_offsets; ++i){
    if(target_residual_offset[i] == 0){
      errorQuda("qudaMultishiftInvert: target residual cannot be zero\n");
      exit(1);
    }
  }

  // if doing a pure double-precision multi-shift solve don't use reliable updates
  double reliable_delta = (inv_args.mixed_precision == 1 || quda_precision == 1) ? 1e-1 : 0.0;
  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, kappa, reliable_delta);
  invertParam.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  invertParam.num_offset = num_offsets;
  for(int i=0; i<num_offsets; ++i){
    invertParam.offset[i] = offset[i];
    invertParam.tol_offset[i] = target_residual_offset[i];
  }
  invertParam.tol = target_residual_offset[0];
  invertParam.clover_coeff = clover_coeff;

  // solution types
  invertParam.solution_type      = QUDA_MATPCDAG_MATPC_SOLUTION;
  invertParam.solve_type         = QUDA_NORMOP_PC_SOLVE;
  invertParam.inv_type           = QUDA_CG_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;

  invertParam.verbosity = verbosity;
  invertParam.verbosity_precondition = QUDA_SILENT;

  invertMultiShiftQuda(solutionArray, source, &invertParam); 

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  for(int i=0; i<num_offsets; ++i) final_residual[i] = invertParam.true_res_offset[i];
  qudamilc_called<false>(__func__, verbosity);
  return;
} // qudaCloverMultishiftInvert





void qudaCloverMultishiftMDInvert(int external_precision, 
    int quda_precision,
    int num_offsets,
    double* const offset,
    double kappa,
    double clover_coeff,
    QudaInvertArgs_t inv_args,
    const double* target_residual_offset,
    const void* milc_link,
    void* milc_clover, 
    void* milc_clover_inv,
    void* source,
    void** psiEven,
    void** psiOdd,
    void** pEven,
    void** pOdd,
    double* const final_residual, 
    int* num_iters)
{
  static const QudaVerbosity verbosity = getVerbosity();
  qudamilc_called<true>(__func__, verbosity);

  for(int i=0; i<num_offsets; ++i){
    if(target_residual_offset[i] == 0){
      errorQuda("qudaMultishiftInvert: target residual cannot be zero\n");
      exit(1);
    }
  }

  // if doing a pure double-precision multi-shift solve don't use reliable updates
  double reliable_delta = (inv_args.mixed_precision == 1 || quda_precision == 1) ? 1e-1 : 0.0;

  QudaInvertParam invertParam = newQudaInvertParam();
  setInvertParam(invertParam, inv_args, external_precision, quda_precision, kappa, reliable_delta);
  invertParam.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  invertParam.num_offset = num_offsets;
  for(int i=0; i<num_offsets; ++i){
    invertParam.offset[i] = offset[i];
    invertParam.tol_offset[i] = target_residual_offset[i];
  }
  invertParam.tol = target_residual_offset[0];
  invertParam.clover_coeff = clover_coeff;

  // solution types
  invertParam.solution_type      = QUDA_MATPCDAG_MATPC_SOLUTION;
  invertParam.solve_type         = QUDA_NORMOP_PC_SOLVE;
  invertParam.inv_type           = QUDA_CG_INVERTER;
  invertParam.matpc_type         = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;

  invertParam.verbosity = verbosity;
  invertParam.verbosity_precondition = QUDA_SILENT;

  invertMultiShiftMDQuda(psiEven, psiOdd, pEven, pOdd, source, &invertParam); 

  // return the number of iterations taken by the inverter
  *num_iters = invertParam.iter;
  for(int i=0; i<num_offsets; ++i) final_residual[i] = invertParam.true_res_offset[i];
  qudamilc_called<false>(__func__, verbosity);
  return;
} // qudaCloverMultishiftMDInvert

#endif // GPU_CLOVER_DIRAC


#endif // BUILD_MILC_INTERFACE
