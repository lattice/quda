#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <quda.h>
#include <quda_internal.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <blas_quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <color_spinor_field.h>
#include <clover_field.h>
#include <llfat_quda.h>
#include <fat_force_quda.h>
#include <hisq_links_quda.h>

#ifdef NUMA_AFFINITY
#include <numa_affinity.h>
#endif

#include <cuda.h>

#ifdef MULTI_GPU
#ifdef MPI_COMMS
#include <mpi.h>
#endif
#ifdef QMP_COMMS
#include <qmp.h>
#endif
#endif // MULTI_GPU

#ifdef GPU_GAUGE_FORCE
#include <gauge_force_quda.h>
#endif

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

#define spinorSiteSize 24 // real numbers per spinor

#define MAX_GPU_NUM_PER_NODE 16

// define newQudaGaugeParam() and newQudaInvertParam()
#define INIT_PARAM
#include "check_params.h"
#undef INIT_PARAM

// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM

// define printQudaGaugeParam() and printQudaInvertParam()
#define PRINT_PARAM
#include "check_params.h"
#undef PRINT_PARAM

#ifdef QMP_COMMS
int rank_QMP;
int num_QMP;
#endif

#include "face_quda.h"

static QudaVerbosity verbosity;
int numa_affinity_enabled = 1;

cudaGaugeField *gaugePrecise = NULL;
cudaGaugeField *gaugeSloppy = NULL;
cudaGaugeField *gaugePrecondition = NULL;

// It's important that these alias the above so that constants are set correctly in Dirac::Dirac()
cudaGaugeField *&gaugeFatPrecise = gaugePrecise;
cudaGaugeField *&gaugeFatSloppy = gaugeSloppy;
cudaGaugeField *&gaugeFatPrecondition = gaugePrecondition;

cudaGaugeField *gaugeLongPrecise = NULL;
cudaGaugeField *gaugeLongSloppy = NULL;
cudaGaugeField *gaugeLongPrecondition = NULL;

cudaCloverField *cloverPrecise = NULL;
cudaCloverField *cloverSloppy = NULL;
cudaCloverField *cloverPrecondition = NULL;

cudaDeviceProp deviceProp;
cudaStream_t *streams;


int getGpuCount()
{
  int count;
  cudaGetDeviceCount(&count);
  if (count <= 0){
    errorQuda("No devices supporting CUDA");
  }
  if(count > MAX_GPU_NUM_PER_NODE){
    errorQuda("GPU count (%d) is larger than limit\n", count);
  }
  return count;
}


void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[], FILE *outfile)
{
  setVerbosity(verbosity);
  setOutputPrefix(prefix);
  setOutputFile(outfile);
}


void initQuda(int dev)
{
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;

#if defined(GPU_DIRECT) && defined(MULTI_GPU) && (CUDA_VERSION == 4000)
  //check if CUDA_NIC_INTEROP is set to 1 in the enviroment
  // not needed for CUDA >= 4.1
  char* cni_str = getenv("CUDA_NIC_INTEROP");
  if(cni_str == NULL){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set\n");
  }
  int cni_int = atoi(cni_str);
  if (cni_int != 1){
    errorQuda("Environment variable CUDA_NIC_INTEROP is not set to 1\n");    
  }
#endif

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    errorQuda("No devices supporting CUDA");
  }

  for(int i=0; i<deviceCount; i++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
    if (getVerbosity() >= QUDA_SUMMARIZE) {
      printfQuda("Found device %d: %s\n", i, deviceProp.name);
    }
  }

#ifdef MULTI_GPU
  comm_init();
#endif

#ifdef QMP_COMMS
  int ndim;
  const int *dim;

  if ( QMP_is_initialized() != QMP_TRUE ) {
    errorQuda("QMP is not initialized");
  }
  num_QMP=QMP_get_number_of_nodes();
  rank_QMP=QMP_get_node_number();
  
  if (dev < 0) {
    dev = rank_QMP % deviceCount;
  }
  ndim = QMP_get_logical_number_of_dimensions();
  dim = QMP_get_logical_dimensions();
#elif defined(MPI_COMMS)
  if (dev < 0) {
    dev=comm_gpuid();
  }
#else
  if (dev < 0) errorQuda("Invalid device number");
#endif
  
  cudaGetDeviceProperties(&deviceProp, dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode
  if (deviceProp.major < 1) {
    errorQuda("Device %d does not support CUDA", dev);
  }
  
  if (getVerbosity() >= QUDA_SUMMARIZE) {
    printfQuda("Using device %d: %s\n", dev, deviceProp.name);
  }
  cudaSetDevice(dev);
  checkCudaErrorNoSync(); // "NoSync" for correctness in HOST_DEBUG mode

#ifdef NUMA_AFFINITY
  if(numa_affinity_enabled){
    setNumaAffinity(dev);
  }
#endif
  // if the device supports host-mapped memory, then enable this
  if(deviceProp.canMapHostMemory) cudaSetDeviceFlags(cudaDeviceMapHost);
  checkCudaError();

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  cudaGetDeviceProperties(&deviceProp, dev);

  streams = new cudaStream_t[Nstream];
  for (int i=0; i<Nstream; i++) {
    cudaStreamCreate(&streams[i]);
  }
  checkCudaError();
  createDslashEvents();

  quda::initBlas();

  loadTuneCache(getVerbosity());
}


void loadGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  checkGaugeParam(param);

  // Set the specific cpu parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);

  cpuGaugeField cpu(gauge_param);

  // switch the parameters for creating the mirror precise cuda gauge field
  gauge_param.create = QUDA_NULL_FIELD_CREATE;
  gauge_param.precision = param->cuda_prec;
  gauge_param.reconstruct = param->reconstruct;
  gauge_param.pad = param->ga_pad;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
		       gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;

  cudaGaugeField *precise = new cudaGaugeField(gauge_param);
  precise->loadCPUField(cpu, QUDA_CPU_FIELD_LOCATION);

  param->gaugeGiB += precise->GBytes();

  // switch the parameters for creating the mirror sloppy cuda gauge field
  gauge_param.precision = param->cuda_prec_sloppy;
  gauge_param.reconstruct = param->reconstruct_sloppy;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
		       gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *sloppy = NULL;
  if (param->cuda_prec != param->cuda_prec_sloppy) {
    sloppy = new cudaGaugeField(gauge_param);
    sloppy->loadCPUField(cpu, QUDA_CPU_FIELD_LOCATION);
    param->gaugeGiB += sloppy->GBytes();
  } else {
    sloppy = precise;
  }

  // switch the parameters for creating the mirror preconditioner cuda gauge field
  gauge_param.precision = param->cuda_prec_precondition;
  gauge_param.reconstruct = param->reconstruct_precondition;
  gauge_param.order = (gauge_param.precision == QUDA_DOUBLE_PRECISION || 
		       gauge_param.reconstruct == QUDA_RECONSTRUCT_NO ) ?
    QUDA_FLOAT2_GAUGE_ORDER : QUDA_FLOAT4_GAUGE_ORDER;
  cudaGaugeField *precondition = NULL;
  if (param->cuda_prec_sloppy != param->cuda_prec_precondition) {
    precondition = new cudaGaugeField(gauge_param);
    precondition->loadCPUField(cpu, QUDA_CPU_FIELD_LOCATION);
    param->gaugeGiB += precondition->GBytes();
  } else {
    precondition = sloppy;
  }
  
  switch (param->type) {
  case QUDA_WILSON_LINKS:
    if (gaugePrecise) errorQuda("Precise gauge field already allocated");
    gaugePrecise = precise;
    if (gaugeSloppy) errorQuda("Sloppy gauge field already allocated");
    gaugeSloppy = sloppy;
    if (gaugePrecondition) errorQuda("Precondition gauge field already allocated");
    gaugePrecondition = precondition;
    break;
  case QUDA_ASQTAD_FAT_LINKS:
    if (gaugeFatPrecise) errorQuda("Precise gauge fat field already allocated");
    gaugeFatPrecise = precise;
    if (gaugeFatSloppy) errorQuda("Sloppy gauge fat field already allocated");
    gaugeFatSloppy = sloppy;
    if (gaugeFatPrecondition) errorQuda("Precondition gauge fat field already allocated");
    gaugeFatPrecondition = precondition;
    break;
  case QUDA_ASQTAD_LONG_LINKS:
    if (gaugeLongPrecise) errorQuda("Precise gauge long field already allocated");
    gaugeLongPrecise = precise;
    if (gaugeLongSloppy) errorQuda("Sloppy gauge long field already allocated");
    gaugeLongSloppy = sloppy;
    if (gaugeLongPrecondition) errorQuda("Precondition gauge long field already allocated");
    gaugeLongPrecondition = precondition;
    break;
  default:
    errorQuda("Invalid gauge type");   
  }

}

void saveGaugeQuda(void *h_gauge, QudaGaugeParam *param)
{
  checkGaugeParam(param);

  // Set the specific cpu parameters and create the cpu gauge field
  GaugeFieldParam gauge_param(h_gauge, *param);
  cpuGaugeField cpuGauge(gauge_param);
  cudaGaugeField *cudaGauge = NULL;
  switch (param->type) {
  case QUDA_WILSON_LINKS:
    cudaGauge = gaugePrecise;
    break;
  case QUDA_ASQTAD_FAT_LINKS:
    cudaGauge = gaugeFatPrecise;
    break;
  case QUDA_ASQTAD_LONG_LINKS:
    cudaGauge = gaugeLongPrecise;
    break;
  default:
    errorQuda("Invalid gauge type");   
  }

  cudaGauge->saveCPUField(cpuGauge, QUDA_CPU_FIELD_LOCATION);
}


void loadCloverQuda(void *h_clover, void *h_clovinv, QudaInvertParam *inv_param)
{

  if (!h_clover && !h_clovinv) {
    errorQuda("loadCloverQuda() called with neither clover term nor inverse");
  }
  if (inv_param->clover_cpu_prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported on CPU");
  }
  if (gaugePrecise == NULL) {
    errorQuda("Gauge field must be loaded before clover");
  }
  if (inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH) {
    errorQuda("Wrong dslash_type in loadCloverQuda()");
  }

  // determines whether operator is preconditioned when calling invertQuda()
  bool pc_solve = (inv_param->solve_type == QUDA_DIRECT_PC_SOLVE ||
		   inv_param->solve_type == QUDA_NORMEQ_PC_SOLVE);

  // determines whether operator is preconditioned when calling MatQuda() or MatDagMatQuda()
  bool pc_solution = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
		      inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  bool asymmetric = (inv_param->matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ||
		     inv_param->matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC);

  // We issue a warning only when it seems likely that the user is screwing up:

  // inverted clover term is required when applying preconditioned operator
  if (!h_clovinv && pc_solve && pc_solution) {
    warningQuda("Inverted clover term not loaded");
  }

  // uninverted clover term is required when applying unpreconditioned operator,
  // but note that dslashQuda() is always preconditioned
  if (!h_clover && !pc_solve && !pc_solution) {
    //warningQuda("Uninverted clover term not loaded");
  }

  // uninverted clover term is also required for "asymmetric" preconditioning
  if (!h_clover && pc_solve && pc_solution && asymmetric) {
    warningQuda("Uninverted clover term not loaded");
  }

  CloverFieldParam clover_param;
  clover_param.nDim = 4;
  for (int i=0; i<4; i++) clover_param.x[i] = gaugePrecise->X()[i];
  clover_param.precision = inv_param->clover_cuda_prec;
  clover_param.pad = inv_param->cl_pad;

  cloverPrecise = new cudaCloverField(h_clover, h_clovinv, inv_param->clover_cpu_prec, 
				      inv_param->clover_order, clover_param);
  inv_param->cloverGiB = cloverPrecise->GBytes();

  // create the mirror sloppy clover field
  if (inv_param->clover_cuda_prec != inv_param->clover_cuda_prec_sloppy) {
    clover_param.precision = inv_param->clover_cuda_prec_sloppy;
    cloverSloppy = new cudaCloverField(h_clover, h_clovinv, inv_param->clover_cpu_prec, 
				       inv_param->clover_order, clover_param); 
    inv_param->cloverGiB += cloverSloppy->GBytes();
  } else {
    cloverSloppy = cloverPrecise;
  }

  // create the mirror preconditioner clover field
  if (inv_param->clover_cuda_prec_sloppy != inv_param->clover_cuda_prec_precondition &&
      inv_param->clover_cuda_prec_precondition != QUDA_INVALID_PRECISION) {
    clover_param.precision = inv_param->clover_cuda_prec_precondition;
    cloverPrecondition = new cudaCloverField(h_clover, h_clovinv, inv_param->clover_cpu_prec, 
					     inv_param->clover_order, clover_param); 
    inv_param->cloverGiB += cloverPrecondition->GBytes();
  } else {
    cloverPrecondition = cloverSloppy;
  }

}

void freeGaugeQuda(void) 
{  
  if (gaugeSloppy != gaugePrecondition && gaugePrecondition) delete gaugePrecondition;
  if (gaugePrecise != gaugeSloppy && gaugeSloppy) delete gaugeSloppy;
  if (gaugePrecise) delete gaugePrecise;

  gaugePrecondition = NULL;
  gaugeSloppy = NULL;
  gaugePrecise = NULL;

  if (gaugeLongSloppy != gaugeLongPrecondition && gaugeLongPrecondition) delete gaugeLongPrecondition;
  if (gaugeLongPrecise != gaugeLongSloppy && gaugeLongSloppy) delete gaugeLongSloppy;
  if (gaugeLongPrecise) delete gaugeLongPrecise;

  gaugeLongPrecondition = NULL;
  gaugeLongSloppy = NULL;
  gaugeLongPrecise = NULL;

  if (gaugeFatSloppy != gaugeFatPrecondition && gaugeFatPrecondition) delete gaugeFatPrecondition;
  if (gaugeFatPrecise != gaugeFatSloppy && gaugeFatSloppy) delete gaugeFatSloppy;
  if (gaugeFatPrecise) delete gaugeFatPrecise;
  
  gaugeFatPrecondition = NULL;
  gaugeFatSloppy = NULL;
  gaugeFatPrecise = NULL;
}


void freeCloverQuda(void)
{
  if (cloverPrecondition != cloverSloppy && cloverPrecondition) delete cloverPrecondition;
  if (cloverSloppy != cloverPrecise && cloverSloppy) delete cloverSloppy;
  if (cloverPrecise) delete cloverPrecise;

  cloverPrecondition = NULL;
  cloverSloppy = NULL;
  cloverPrecise = NULL;
}


void endQuda(void)
{
  cudaColorSpinorField::freeBuffer();
  cudaColorSpinorField::freeGhostBuffer();
  cpuColorSpinorField::freeGhostBuffer();
  freeGaugeQuda();
  freeCloverQuda();

  quda::endBlas();

  if (streams) {
    for (int i=0; i<Nstream; i++) cudaStreamDestroy(streams[i]);
    delete []streams;
    streams = NULL;
  }
  destroyDslashEvents();

  saveTuneCache(getVerbosity());
}


void setDiracParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
{
  double kappa = inv_param->kappa;
  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    kappa *= gaugePrecise->Anisotropy();
  }

  switch (inv_param->dslash_type) {
  case QUDA_WILSON_DSLASH:
    diracParam.type = pc ? QUDA_WILSONPC_DIRAC : QUDA_WILSON_DIRAC;
    break;
  case QUDA_CLOVER_WILSON_DSLASH:
    diracParam.type = pc ? QUDA_CLOVERPC_DIRAC : QUDA_CLOVER_DIRAC;
    break;
  case QUDA_DOMAIN_WALL_DSLASH:
    diracParam.type = pc ? QUDA_DOMAIN_WALLPC_DIRAC : QUDA_DOMAIN_WALL_DIRAC;
//BEGIN NEW :
    diracParam.Ls = inv_param->Ls;
//END NEW    
    break;
  case QUDA_ASQTAD_DSLASH:
    diracParam.type = pc ? QUDA_ASQTADPC_DIRAC : QUDA_ASQTAD_DIRAC;
    break;
  case QUDA_TWISTED_MASS_DSLASH:
    diracParam.type = pc ? QUDA_TWISTED_MASSPC_DIRAC : QUDA_TWISTED_MASS_DIRAC;
    break;
  default:
    errorQuda("Unsupported dslash_type %d", inv_param->dslash_type);
  }

  diracParam.matpcType = inv_param->matpc_type;
  diracParam.dagger = inv_param->dagger;
  diracParam.gauge = gaugePrecise;
  diracParam.fatGauge = gaugeFatPrecise;
  diracParam.longGauge = gaugeLongPrecise;    
  diracParam.clover = cloverPrecise;
  diracParam.kappa = kappa;
  diracParam.mass = inv_param->mass;
  diracParam.m5 = inv_param->m5;
  diracParam.mu = inv_param->mu;
  diracParam.verbose = inv_param->verbosity;

  for (int i=0; i<4; i++) {
    diracParam.commDim[i] = 1;   // comms are always on
  }
}


void setDiracSloppyParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
{
  setDiracParam(diracParam, inv_param, pc);

  diracParam.gauge = gaugeSloppy;
  diracParam.fatGauge = gaugeFatSloppy;
  diracParam.longGauge = gaugeLongSloppy;    
  diracParam.clover = cloverSloppy;

  for (int i=0; i<4; i++) {
    diracParam.commDim[i] = 1;   // comms are always on
  }

}

// The preconditioner currently mimicks the sloppy operator with no comms
void setDiracPreParam(DiracParam &diracParam, QudaInvertParam *inv_param, const bool pc)
{
  setDiracParam(diracParam, inv_param, pc);

  diracParam.gauge = gaugePrecondition;
  diracParam.fatGauge = gaugeFatPrecondition;
  diracParam.longGauge = gaugeLongPrecondition;    
  diracParam.clover = cloverPrecondition;

  for (int i=0; i<4; i++) {
    diracParam.commDim[i] = 0; // comms are always off
  }

}


static void massRescale(QudaDslashType dslash_type, double &kappa, QudaSolutionType solution_type, 
			QudaMassNormalization mass_normalization, cudaColorSpinorField &b)
{   
 if (verbosity >= QUDA_VERBOSE) {
    printfQuda("Mass rescale: Kappa is: %f\n", kappa);
    printfQuda("Mass rescale: mass normalization: %d\n", mass_normalization);
    double nin = norm2(b);
    printfQuda("Mass rescale: norm of source in = %f\n", nin);
  }
 
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (mass_normalization != QUDA_MASS_NORMALIZATION) {
      errorQuda("Staggered code only supports QUDA_MASS_NORMALIZATION");
    }
    return;
  }

  // multiply the source to compensate for normalization of the Dirac operator, if necessary
  switch (solution_type) {
  case QUDA_MAT_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(2.0*kappa, b);
    }
    break;
  case QUDA_MATDAG_MAT_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(4.0*kappa*kappa, b);
    }
    break;
  case QUDA_MATPC_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	axCuda(4.0*kappa*kappa, b);
    } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(2.0*kappa, b);
    }
    break;
  case QUDA_MATPCDAG_MATPC_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	axCuda(16.0*pow(kappa,4), b);
    } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	axCuda(4.0*kappa*kappa, b);
    }
    break;
  default:
    errorQuda("Solution type %d not supported", solution_type);
  }

  if (verbosity >= QUDA_DEBUG_VERBOSE) printfQuda("Mass rescale done\n");   
 if (verbosity >= QUDA_VERBOSE) {
    printfQuda("Mass rescale: Kappa is: %f\n", kappa);
    printfQuda("Mass rescale: mass normalization: %d\n", mass_normalization);
    double nin = norm2(b);
    printfQuda("Mass rescale: norm of source out = %f\n", nin);
  }

}

static void massRescaleCoeff(QudaDslashType dslash_type, double &kappa, QudaSolutionType solution_type, 
			     QudaMassNormalization mass_normalization, double &coeff)
{    
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (mass_normalization != QUDA_MASS_NORMALIZATION) {
      errorQuda("Staggered code only supports QUDA_MASS_NORMALIZATION");
    }
    return;
  }

  // multiply the source to compensate for normalization of the Dirac operator, if necessary
  switch (solution_type) {
  case QUDA_MAT_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      coeff *= 2.0*kappa;
    }
    break;
  case QUDA_MATDAG_MAT_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION ||
	mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      coeff *= 4.0*kappa*kappa;
    }
    break;
  case QUDA_MATPC_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION) {
      coeff *= 4.0*kappa*kappa;
    } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      coeff *= 2.0*kappa;
    }
    break;
  case QUDA_MATPCDAG_MATPC_SOLUTION:
    if (mass_normalization == QUDA_MASS_NORMALIZATION) {
	coeff*=16.0*pow(kappa,4);
    } else if (mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
	coeff*=4.0*kappa*kappa;
    }
    break;
  default:
    errorQuda("Solution type %d not supported", solution_type);
  }

  if (verbosity >= QUDA_DEBUG_VERBOSE) printfQuda("Mass rescale done\n");   
}

/*void QUDA_DiracField(QUDA_DiracParam *param) {
  
  }*/

void dslashQuda(void *h_out, void *h_in, QudaInvertParam *inv_param, QudaParity parity)
{
  ColorSpinorParam cpuParam(h_in, inv_param->input_location, *inv_param, gaugePrecise->X(), 1);

  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) {
    if (parity == QUDA_EVEN_PARITY) {
      parity = QUDA_ODD_PARITY;
    } else {
      parity = QUDA_EVEN_PARITY;
    }
    axCuda(gaugePrecise->Anisotropy(), in);
  }
  bool pc = true;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->Dslash(out, in, parity); // apply the operator
  delete dirac; // clean up

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;
  
  delete out_h;
  delete in_h;
}


void MatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
	     inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, inv_param->input_location, *inv_param, gaugePrecise->X(), pc);
  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);

  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->M(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
	inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.5/kappa, out);
    }
  }

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  delete out_h;
  delete in_h;
}


void MatDagMatQuda(void *h_out, void *h_in, QudaInvertParam *inv_param)
{
  bool pc = (inv_param->solution_type == QUDA_MATPC_SOLUTION ||
	     inv_param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  ColorSpinorParam cpuParam(h_in, inv_param->input_location, *inv_param, gaugePrecise->X(), pc);
  ColorSpinorField *in_h = (inv_param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  ColorSpinorParam cudaParam(cpuParam, *inv_param);
  cudaColorSpinorField in(*in_h, cudaParam);
  
  cudaParam.create = QUDA_NULL_FIELD_CREATE;
  cudaColorSpinorField out(in, cudaParam);

  //  double kappa = inv_param->kappa;
  //  if (inv_param->dirac_order == QUDA_CPS_WILSON_DIRAC_ORDER) kappa *= gaugePrecise->anisotropy;

  DiracParam diracParam;
  setDiracParam(diracParam, inv_param, pc);

  Dirac *dirac = Dirac::create(diracParam); // create the Dirac operator
  dirac->MdagM(out, in); // apply the operator
  delete dirac; // clean up

  double kappa = inv_param->kappa;
  if (pc) {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION) {
      axCuda(1.0/pow(2.0*kappa,4), out);
    } else if (inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  } else {
    if (inv_param->mass_normalization == QUDA_MASS_NORMALIZATION ||
	inv_param->mass_normalization == QUDA_ASYMMETRIC_MASS_NORMALIZATION) {
      axCuda(0.25/(kappa*kappa), out);
    }
  }

  cpuParam.v = h_out;

  ColorSpinorField *out_h = (inv_param->output_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));
  *out_h = out;

  delete out_h;
  delete in_h;
}

void createDirac(Dirac *&d, Dirac *&dSloppy, Dirac *&dPre, QudaInvertParam &param, const bool pc_solve)
{
  DiracParam diracParam;
  DiracParam diracSloppyParam;
  DiracParam diracPreParam;

  setDiracParam(diracParam, &param, pc_solve);
  setDiracSloppyParam(diracSloppyParam, &param, pc_solve);
  setDiracPreParam(diracPreParam, &param, pc_solve);
    
  d = Dirac::create(diracParam); // create the Dirac operator   
  dSloppy = Dirac::create(diracSloppyParam);
  dPre = Dirac::create(diracPreParam);
}

cudaGaugeField* checkGauge(QudaInvertParam *param) {
  cudaGaugeField *cudaGauge = NULL;
  if (param->dslash_type != QUDA_ASQTAD_DSLASH) {
    if (gaugePrecise == NULL) errorQuda("Precise gauge field doesn't exist");
    if (gaugeSloppy == NULL) errorQuda("Sloppy gauge field doesn't exist");
    if (gaugePrecondition == NULL) errorQuda("Precondition gauge field doesn't exist");
    cudaGauge = gaugePrecise;
  } else {
    if (gaugeFatPrecise == NULL) errorQuda("Precise gauge fat field doesn't exist");
    if (gaugeFatSloppy == NULL) errorQuda("Sloppy gauge fat field doesn't exist");
    if (gaugeFatPrecondition == NULL) errorQuda("Precondition gauge fat field doesn't exist");

    if (gaugeLongPrecise == NULL) errorQuda("Precise gauge long field doesn't exist");
    if (gaugeLongSloppy == NULL) errorQuda("Sloppy gauge long field doesn't exist");
    if (gaugeLongPrecondition == NULL) errorQuda("Precondition gauge long field doesn't exist");
    cudaGauge = gaugeFatPrecise;
  }
  return cudaGauge;
}

void invertQuda(void *hp_x, void *hp_b, QudaInvertParam *param)
{
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);

  checkInvertParam(param);
  verbosity = param->verbosity;

  bool pc_solve = (param->solve_type == QUDA_DIRECT_PC_SOLVE ||
		   param->solve_type == QUDA_NORMEQ_PC_SOLVE);

  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
		      param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION);

  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if (!pc_solve) param->spinorGiB *= 2;
  param->spinorGiB *= (param->cuda_prec == QUDA_DOUBLE_PRECISION ? sizeof(double) : sizeof(float));
  if (param->preserve_source == QUDA_PRESERVE_SOURCE_NO) {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 5 : 7)/(double)(1<<30);
  } else {
    param->spinorGiB *= (param->inv_type == QUDA_CG_INVERTER ? 8 : 9)/(double)(1<<30);
  }

  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);

  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;
  Dirac &diracPre = *dPre;

  cudaColorSpinorField *b = NULL;
  cudaColorSpinorField *x = NULL;
  cudaColorSpinorField *in = NULL;
  cudaColorSpinorField *out = NULL;

  const int *X = cudaGauge->X();

  // wrap CPU host side pointers
  ColorSpinorParam cpuParam(hp_b, param->input_location, *param, X, pc_solution);
  ColorSpinorField *h_b = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

  cpuParam.v = hp_x;
  ColorSpinorField *h_x = (param->input_location == QUDA_CPU_FIELD_LOCATION) ?
    static_cast<ColorSpinorField*>(new cpuColorSpinorField(cpuParam)) : static_cast<ColorSpinorField*>(new cudaColorSpinorField(cpuParam));

    
  // download source
  ColorSpinorParam cudaParam(cpuParam, *param);     
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); 


  if (param->use_init_guess == QUDA_USE_INIT_GUESS_YES) { // download initial guess
    x = new cudaColorSpinorField(*h_x, cudaParam); // solution  
  } else { // zero initial guess
    cudaParam.create = QUDA_ZERO_FIELD_CREATE;
    x = new cudaColorSpinorField(cudaParam); // solution
  }
    
  if (param->verbosity >= QUDA_VERBOSE) {
    double nh_b = norm2(*h_b);
    double nb = norm2(*b);
    printfQuda("Source: CPU = %f, CUDA copy = %f\n", nh_b, nb);
  }

  setDslashTuning(param->tune, param->verbosity);
  quda::setBlasTuning(param->tune, param->verbosity);

  dirac.prepare(in, out, *x, *b, param->solution_type);
  if (param->verbosity >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    printfQuda("Prepared source = %f\n", nin);   
  }

  massRescale(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *in);

  if (param->verbosity >= QUDA_VERBOSE) {
    double nin = norm2(*in);
    printfQuda("Prepared source post mass rescale = %f\n", nin);   
  }
  

  switch (param->inv_type) {
  case QUDA_CG_INVERTER:
    if (param->solution_type != QUDA_MATDAG_MAT_SOLUTION && param->solution_type != QUDA_MATPCDAG_MATPC_SOLUTION) {
      copyCuda(*out, *in);
      dirac.Mdag(*in, *out);
    }
    {
      DiracMdagM m(dirac), mSloppy(diracSloppy);
      CG cg(m, mSloppy, *param);
      cg(*out, *in);
    }
    break;
  case QUDA_BICGSTAB_INVERTER:
    if (param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      BiCGstab bicg(m, mSloppy, mPre, *param);
      bicg(*out, *in);
      copyCuda(*in, *out);
    }
    {
      DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      BiCGstab bicg(m, mSloppy, mPre, *param);
      bicg(*out, *in);
    }
    break;
  case QUDA_GCR_INVERTER:
    if (param->solution_type == QUDA_MATDAG_MAT_SOLUTION || param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION) {
      DiracMdag m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      GCR gcr(m, mSloppy, mPre, *param);
      gcr(*out, *in);
      copyCuda(*in, *out);
    }
    {
      DiracM m(dirac), mSloppy(diracSloppy), mPre(diracPre);
      GCR gcr(m, mSloppy, mPre, *param);
      gcr(*out, *in);
    }
    break;
  default:
    errorQuda("Inverter type %d not implemented", param->inv_type);
  }
  
  if (param->verbosity >= QUDA_VERBOSE){
   double nx = norm2(*x);
   printfQuda("Solution = %f\n",nx);
  }
  dirac.reconstruct(*x, *b, param->solution_type);
  
  *h_x = *x;
  
  if (param->verbosity >= QUDA_VERBOSE){
    double nx = norm2(*x);
    double nh_x = norm2(*h_x);
    printfQuda("Reconstructed: CUDA solution = %f, CPU copy = %f\n", nx, nh_x);
  }
  
  delete h_b;
  delete h_x;
  delete b;
  delete x;

  delete d;
  delete dSloppy;
  delete dPre;

  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
}


/*! 
 *
 * Generic version of the multi-shift solver. Should work for
 * most fermions. Note, offset[0] is not folded into the mass parameter 
 */
void invertMultiShiftQuda(void **_hp_x, void *_hp_b, QudaInvertParam *param,
			  double* offsets, int num_offsets, double* residue_sq)
{
  // check the gauge fields have been created
  cudaGaugeField *cudaGauge = checkGauge(param);
  checkInvertParam(param);

  param->num_offset = num_offsets;
  if (param->num_offset > QUDA_MAX_MULTI_SHIFT) 
    errorQuda("Number of shifts %d requested greater than QUDA_MAX_MULTI_SHIFT %d", 
	      param->num_offset, QUDA_MAX_MULTI_SHIFT);
  for (int i=0; i<param->num_offset; i++) {
    param->offset[i] = offsets[i];
    param->tol_offset[i] = residue_sq[i];
  }

  verbosity = param->verbosity;

  // Are we doing a preconditioned solve */
  /* What does NormEq solve mean in the shifted case? 
   */
  if (param->solve_type != QUDA_NORMEQ_PC_SOLVE &&
      param->solve_type != QUDA_NORMEQ_SOLVE) { 
    errorQuda("Direct solve_type is not supported in invertMultiShiftQuda()\n");
  }

  bool pc_solve = (param->solve_type == QUDA_NORMEQ_PC_SOLVE);

  // In principle one can do a MATPC Solution for a hermitian M_pc
  // In practice most of the time I guess one will do a M^\dagger_pc M_pc solution.
  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
		      param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION );

  // No of GiB in a checkerboard of a spinor
  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if( !pc_solve) param->spinorGiB *= 2; // Double volume for non PC solve
  
  // **** WARNING *** this may not match implementation... 
  if( param->inv_type == QUDA_CG_INVERTER ) { 
    // CG-M needs 5 vectors for the smallest shift + 2 for each additional shift
    param->spinorGiB *= (5 + 2*(param->num_offset-1))/(double)(1<<30);
  }
  else {
    // BiCGStab-M needs 7 for the original shift + 2 for each additional shift + 1 auxiliary
    // (Jegerlehner hep-lat/9612014 eq (3.13)
    param->spinorGiB *= (7 + 2*(param->num_offset-1))/(double)(1<<30);
  }

  // Timing and FLOP counters
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  // Find the smallest shift and its offset.
  double low_offset = param->offset[0];
  int low_index = 0;
  for (int i=1;i < param->num_offset;i++){
    if (param->offset[i] < low_offset){
      low_offset = param->offset[i];
      low_index = i;
    }
  }
  
  // Host pointers for x, take a copy of the input host pointers
  void** hp_x;
  hp_x = new void* [ param->num_offset ];

  void* hp_b = _hp_b;
  for(int i=0;i < param->num_offset;i++){
    hp_x[i] = _hp_x[i];
  }
  
  // Now shift things so that the vector with the smallest shift 
  // is in the first position of the array
  if (low_index != 0){
    void* tmp = hp_x[0];
    hp_x[0] = hp_x[low_index] ;
    hp_x[low_index] = tmp;
    
    double tmp1 = param->offset[0];
    param->offset[0]= param->offset[low_index];
    param->offset[low_index] =tmp1;
  }
    
  // Create the matrix.
  // The way this works is that createDirac will create 'd' and 'dSloppy'
  // which are global. We then grab these with references...
  //
  // Balint: Isn't there a  nice construction pattern we could use here? This is 
  // expedient but yucky.
  //  DiracParam diracParam; 
  if (param->dslash_type == QUDA_ASQTAD_DSLASH){
    param->mass = sqrt(param->offset[0]/4);  
  }

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  Dirac &dirac = *d;
  Dirac &diracSloppy = *dSloppy;

  cpuColorSpinorField *h_b = NULL; // Host RHS
  cpuColorSpinorField **h_x = NULL;
  cudaColorSpinorField *b = NULL;   // Cuda RHS
  cudaColorSpinorField **x = NULL;  // Cuda Solutions

  // Grab the dimension array of the input gauge field.
  const int *X = ( param->dslash_type == QUDA_ASQTAD_DSLASH ) ? 
    gaugeFatPrecise->X() : gaugePrecise->X();

  // Wrap CPU host side pointers
  // 
  // Balint: This creates a ColorSpinorParam struct, from the host data pointer, 
  // the definitions in param, the dimensions X, and whether the solution is on 
  // a checkerboard instruction or not. These can then be used as 'instructions' 
  // to create the actual colorSpinorField
  ColorSpinorParam cpuParam(hp_b, QUDA_CPU_FIELD_LOCATION, *param, X, pc_solution);
  h_b = new cpuColorSpinorField(cpuParam);

  h_x = new cpuColorSpinorField* [ param->num_offset ]; // DYNAMIC ALLOCATION
  for(int i=0; i < param->num_offset; i++) { 
    cpuParam.v = hp_x[i];
    h_x[i] = new cpuColorSpinorField(cpuParam);
  }

  // Now I need a colorSpinorParam for the device
  ColorSpinorParam cudaParam(cpuParam, *param);
  // This setting will download a host vector
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); // Creates b and downloads h_b to it

  // Create the solution fields filled with zero
  x = new cudaColorSpinorField* [ param->num_offset ];
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int i=0; i < param->num_offset; i++) { 
    x[i] = new cudaColorSpinorField(cudaParam);
  }

  // Check source norms
  if( param->verbosity >= QUDA_VERBOSE ) {
    double nh_b = norm2(*h_b);
    double nb = norm2(*b);
    printfQuda("Source: CPU= %f, CUDA copy = %f\n", nh_b,nb);
  }

  setDslashTuning(param->tune, param->verbosity);
  quda::setBlasTuning(param->tune, param->verbosity);
  
  massRescale(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *b);
  double *unscaled_shifts = new double [param->num_offset];
  for(int i=0; i < param->num_offset; i++){ 
    unscaled_shifts[i] = param->offset[i];
    massRescaleCoeff(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, param->offset[i]);
  }

  {
    DiracMdagM m(dirac), mSloppy(diracSloppy);
    MultiShiftCG cg_m(m, mSloppy, *param);
    cg_m(x, *b);  
  }

  // restore shifts -- avoid side effects
  for(int i=0; i < param->num_offset; i++) { 
    param->offset[i] = unscaled_shifts[i];
  }

  delete [] unscaled_shifts;

  for(int i=0; i < param->num_offset; i++) { 
    *h_x[i] = *x[i];
  }

  for(int i=0; i < param->num_offset; i++){ 
    delete h_x[i];
    delete x[i];
  }

  delete h_b;
  delete b;

  delete [] h_x;
  delete [] x;

  delete [] hp_x;

  delete d;
  delete dSloppy;
  delete dPre;
  
  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
}

/************************************** Ugly Mixed precision multishift CG solver ****************************/

static void* fatlink;
static int fatlink_pad;
static void* longlink;
static int longlink_pad;
static QudaReconstructType longlink_recon;
static QudaReconstructType longlink_recon_sloppy;
static QudaGaugeParam* gauge_param;

void 
record_gauge(int* X, void *_fatlink, int _fatlink_pad, void* _longlink, int _longlink_pad, 
	     QudaReconstructType _longlink_recon, QudaReconstructType _longlink_recon_sloppy,
	     QudaGaugeParam *_param)
{
  

  //the X and precision in fatlink must be set because we use them in dirac creation
  //See dirac_staggered.cpp
  /*
  for(int i =0;i < 4;i++){
    cudaFatLinkPrecise.X[i] = cudaFatLinkSloppy.X[i] = X[i];
  }
  cudaFatLinkPrecise.precision = _param->cuda_prec;
  cudaFatLinkSloppy.precision = _param->cuda_prec_sloppy;
  */

  fatlink = _fatlink;
  fatlink_pad = _fatlink_pad;
  
  longlink = _longlink;
  longlink_pad = _longlink_pad;
  longlink_recon = _longlink_recon;
  longlink_recon_sloppy = _longlink_recon_sloppy;
  
  gauge_param = _param;

  return;

}


void 
do_create_precise_cuda_gauge(void)
{
  QudaPrecision prec = gauge_param->cuda_prec;
  QudaPrecision prec_sloppy = gauge_param->cuda_prec_sloppy;
  QudaPrecision prec_precondition = gauge_param->cuda_prec_precondition;
  
  //the sloppy gauge field will be filled, and needs to backup
  cudaGaugeField *tmp_fat = gaugeFatSloppy;
  cudaGaugeField *tmp_long= gaugeLongSloppy;
  cudaGaugeField *tmp_fat_precondition = gaugeFatPrecondition;
  cudaGaugeField *tmp_long_precondition = gaugeLongPrecondition;

  
  //create precise links
  if(gauge_param->cuda_prec != gauge_param->cuda_prec_sloppy){
    gaugeFatPrecise = gaugeFatSloppy = gaugeFatPrecondition = NULL;
    gaugeLongPrecise = gaugeLongSloppy = gaugeLongPrecondition = NULL;
    gauge_param->cuda_prec = gauge_param->cuda_prec_sloppy = gauge_param->cuda_prec_precondition = prec;
    gauge_param->type = QUDA_ASQTAD_FAT_LINKS;
    gauge_param->ga_pad = fatlink_pad;
    gauge_param->reconstruct = gauge_param->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    loadGaugeQuda(fatlink, gauge_param);

    
    gauge_param->type = QUDA_ASQTAD_LONG_LINKS;
    gauge_param->ga_pad = longlink_pad;
    gauge_param->reconstruct = longlink_recon;
    gauge_param->reconstruct_sloppy = longlink_recon_sloppy;
    loadGaugeQuda(longlink, gauge_param);
  }else{
    gaugeFatPrecise = gaugeFatSloppy;
    gaugeLongPrecise = gaugeLongSloppy;
  }

  //set prec/prec_sloppy it back
  gauge_param->cuda_prec = prec;
  gauge_param->cuda_prec_sloppy =prec_sloppy;
  gauge_param->cuda_prec_precondition = prec_precondition;

  //set the sloopy gauge filed back
  gaugeFatSloppy = tmp_fat;
  gaugeLongSloppy = tmp_long;
  
  gaugeFatPrecondition = tmp_fat_precondition;
  gaugeLongPrecondition = tmp_long_precondition;
  return;
}

void 
do_create_sloppy_cuda_gauge(void)
{

  QudaPrecision prec = gauge_param->cuda_prec;
  QudaPrecision prec_sloppy = gauge_param->cuda_prec_sloppy;  
  QudaPrecision prec_precondition = gauge_param->cuda_prec_precondition;

  //the precise gauge field will be filled, and needs to backup
  cudaGaugeField *tmp_fat = gaugeFatPrecise;
  cudaGaugeField *tmp_long = gaugeLongPrecise;
  cudaGaugeField *tmp_fat_precondition = gaugeFatPrecondition;
  cudaGaugeField *tmp_long_precondition = gaugeLongPrecondition;
  
  gaugeFatPrecise = gaugeFatSloppy = gaugeFatPrecondition = NULL;
  gaugeLongPrecise = gaugeLongSloppy = gaugeLongPrecondition= NULL;

  //create sloppy links
  gauge_param->cuda_prec = gauge_param->cuda_prec_sloppy = 
    gauge_param->cuda_prec_precondition = prec_sloppy; 
  gauge_param->type = QUDA_ASQTAD_FAT_LINKS;
  gauge_param->ga_pad = fatlink_pad;
  gauge_param->reconstruct = gauge_param->reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  loadGaugeQuda(fatlink, gauge_param);
  
  gauge_param->type = QUDA_ASQTAD_LONG_LINKS;
  gauge_param->ga_pad = longlink_pad;
  gauge_param->reconstruct = longlink_recon;
  gauge_param->reconstruct_sloppy = longlink_recon_sloppy;
  loadGaugeQuda(longlink, gauge_param);
  
  //set prec/prec_sloppy it back
  gauge_param->cuda_prec = prec;
  gauge_param->cuda_prec_sloppy =prec_sloppy;
  gauge_param->cuda_prec_precondition = prec_precondition;
  
  //set the sloopy gauge filed back
  gaugeFatPrecise = tmp_fat;
  gaugeLongPrecise = tmp_long;  
  
  gaugeFatPrecondition = tmp_fat_precondition;
  gaugeLongPrecondition = tmp_long_precondition;

  return;
}


void 
invertMultiShiftQudaMixed(void **_hp_x, void *_hp_b, QudaInvertParam *param,
			  double* offsets, int num_offsets, double* residue_sq)
{


  QudaPrecision high_prec = param->cuda_prec;
  param->cuda_prec = param->cuda_prec_sloppy;
  
  param->num_offset = num_offsets;
  if (param->num_offset > QUDA_MAX_MULTI_SHIFT) 
    errorQuda("Number of shifts %d requested greater than QUDA_MAX_MULTI_SHIFT %d", 
	      param->num_offset, QUDA_MAX_MULTI_SHIFT);
  for (int i=0; i<param->num_offset; i++) {
    param->offset[i] = offsets[i];
    param->tol_offset[i] = residue_sq[i];
  }

  do_create_sloppy_cuda_gauge();

  // check the gauge fields have been created
  //cudaGaugeField *cudaGauge = checkGauge(param);
  cudaGaugeField *cudaGauge = gaugeFatSloppy;


  checkInvertParam(param);
  verbosity = param->verbosity;

  // Are we doing a preconditioned solve */
  /* What does NormEq solve mean in the shifted case? 
   */
  if (param->solve_type != QUDA_NORMEQ_PC_SOLVE &&
      param->solve_type != QUDA_NORMEQ_SOLVE) { 
    errorQuda("Direct solve_type is not supported in invertMultiShiftQuda()\n");
  }

  bool pc_solve = (param->solve_type == QUDA_NORMEQ_PC_SOLVE);

  // In principle one can do a MATPC Solution for a hermitian M_pc
  // In practice most of the time I guess one will do a M^\dagger_pc M_pc solution.
  bool pc_solution = (param->solution_type == QUDA_MATPC_SOLUTION ||
		      param->solution_type == QUDA_MATPCDAG_MATPC_SOLUTION );

  // No of GiB in a checkerboard of a spinor
  param->spinorGiB = cudaGauge->VolumeCB() * spinorSiteSize;
  if( !pc_solve) param->spinorGiB *= 2; // Double volume for non PC solve
  
  // **** WARNING *** this may not match implementation... 
  if( param->inv_type == QUDA_CG_INVERTER ) { 
    // CG-M needs 5 vectors for the smallest shift + 2 for each additional shift
    param->spinorGiB *= (5 + 2*(param->num_offset-1))/(double)(1<<30);
  }
  else {
    // BiCGStab-M needs 7 for the original shift + 2 for each additional shift + 1 auxiliary
    // (Jegerlehner hep-lat/9612014 eq (3.13)
    param->spinorGiB *= (7 + 2*(param->num_offset-1))/(double)(1<<30);
  }

  // Timing and FLOP counters
  param->secs = 0;
  param->gflops = 0;
  param->iter = 0;
  
  // Find the smallest shift and its offset.
  double low_offset = param->offset[0];
  int low_index = 0;
  for (int i=1;i < param->num_offset;i++){
    if (param->offset[i] < low_offset){
      low_offset = param->offset[i];
      low_index = i;
    }
  }
  
  // Host pointers for x, take a copy of the input host pointers
  void** hp_x;
  hp_x = new void* [ param->num_offset ];

  void* hp_b = _hp_b;
  for(int i=0;i < param->num_offset;i++){
    hp_x[i] = _hp_x[i];
  }
  
  // Now shift things so that the vector with the smallest shift 
  // is in the first position of the array
  if (low_index != 0){
    void* tmp = hp_x[0];
    hp_x[0] = hp_x[low_index] ;
    hp_x[low_index] = tmp;
    
    double tmp1 = param->offset[0];
    param->offset[0]= param->offset[low_index];
    param->offset[low_index] =tmp1;
  }
    
  // Create the matrix.
  // The way this works is that createDirac will create 'd' and 'dSloppy'
  // which are global. We then grab these with references...
  //
  // Balint: Isn't there a  nice construction pattern we could use here? This is 
  // expedient but yucky.
  if (param->dslash_type == QUDA_ASQTAD_DSLASH){
    param->mass = sqrt(param->offset[0]/4);  
  }
  //FIXME: Dirty dirty hack
  // At this moment, the precise fat/long gauge is not created (NULL)
  // but we set it to be the same as sloppy to avoid segfault 
  // in creating the dirac since it is needed 
  gaugeFatPrecondition = gaugeFatPrecise = gaugeFatSloppy;
  gaugeLongPrecondition = gaugeLongPrecise = gaugeLongSloppy;

  Dirac *d = NULL;
  Dirac *dSloppy = NULL;
  Dirac *dPre = NULL;

  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);
  // resetting to NULL
  gaugeFatPrecise = NULL; gaugeFatPrecondition = NULL;
  gaugeLongPrecise = NULL; gaugeLongPrecondition = NULL;

  Dirac &diracSloppy = *dSloppy;

  cpuColorSpinorField *h_b = NULL; // Host RHS
  cpuColorSpinorField **h_x = NULL;
  cudaColorSpinorField *b = NULL;   // Cuda RHS
  cudaColorSpinorField **x = NULL;  // Cuda Solutions

  // Grab the dimension array of the input gauge field.
  const int *X = cudaGauge->X();

  // Wrap CPU host side pointers
  // 
  // Balint: This creates a ColorSpinorParam struct, from the host data pointer, 
  // the definitions in param, the dimensions X, and whether the solution is on 
  // a checkerboard instruction or not. These can then be used as 'instructions' 
  // to create the actual colorSpinorField
  ColorSpinorParam cpuParam(hp_b, QUDA_CPU_FIELD_LOCATION, *param, X, pc_solution);
  h_b = new cpuColorSpinorField(cpuParam);

  h_x = new cpuColorSpinorField* [ param->num_offset ]; // DYNAMIC ALLOCATION
  for(int i=0; i < param->num_offset; i++) { 
    cpuParam.v = hp_x[i];
    h_x[i] = new cpuColorSpinorField(cpuParam);
  }

  // Now I need a colorSpinorParam for the device
  ColorSpinorParam cudaParam(cpuParam, *param);
  // This setting will download a host vector
  cudaParam.create = QUDA_COPY_FIELD_CREATE;
  b = new cudaColorSpinorField(*h_b, cudaParam); // Creates b and downloads h_b to it

  // Create the solution fields filled with zero
  x = new cudaColorSpinorField* [ param->num_offset ];
  cudaParam.create = QUDA_ZERO_FIELD_CREATE;
  for(int i=0; i < param->num_offset; i++) { 
    x[i] = new cudaColorSpinorField(cudaParam);
  }

  // Check source norms
  if( param->verbosity >= QUDA_VERBOSE ) {
    double nh_b = norm2(*h_b);
    double nb = norm2(*b);
    printfQuda("Source: CPU= %f, CUDA copy = %f\n", nh_b,nb);
  }

  // tune the Dirac Kernel
  setDslashTuning(param->tune, param->verbosity);
  quda::setBlasTuning(param->tune, param->verbosity);
  // if set, tuning will happen in the first multishift call
  
  massRescale(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, *b);
  double *rescaled_shifts = new double [param->num_offset];
  for(int i=0; i < param->num_offset; i++){ 
    rescaled_shifts[i] = param->offset[i];
    massRescaleCoeff(param->dslash_type, param->kappa, param->solution_type, param->mass_normalization, rescaled_shifts[i]);
  }

  {
    DiracMdagM m(diracSloppy);
    MultiShiftCG cg_m(m, m, *param);
    cg_m(x, *b);  
  }
    
  delete [] rescaled_shifts;
  
  delete b;
  
  int total_iters = 0;
  double total_secs = 0;
  double total_gflops = 0;
  total_iters += param->iter;
  total_secs  += param->secs;
  total_gflops += param->gflops;
  
  cudaParam.precision = high_prec;
  param->cuda_prec = high_prec;
  do_create_precise_cuda_gauge();
  
  b = new cudaColorSpinorField(cudaParam);
  *b = *h_b;
  
  /*FIXME: to avoid setfault*/
  gaugeFatPrecondition =gaugeFatSloppy;
  gaugeLongPrecondition =gaugeLongSloppy;
  
  delete dPre;
  delete dSloppy;
  delete d;
  
  // create the dirac operator
  createDirac(d, dSloppy, dPre, *param, pc_solve);

  gaugeFatPrecondition = NULL;
  gaugeLongPrecondition = NULL;
  {
    Dirac& dirac2 = *d;
    Dirac& diracSloppy2 = *dSloppy;
    
    cudaColorSpinorField* high_x;
    high_x = new cudaColorSpinorField(cudaParam);
    for(int i=0;i < param->num_offset; i++){
      *high_x  = *x[i];
      delete x[i];
      double mass = sqrt(param->offset[i]/4);
      dirac2.setMass(mass);
      diracSloppy2.setMass(mass);
      DiracMdagM m(dirac2), mSloppy(diracSloppy2);
      CG cg(m, mSloppy, *param);
      cg(*high_x, *b);      
      total_iters += param->iter;
      total_secs  += param->secs;
      total_gflops += param->gflops;      
      *h_x[i] = *high_x;
    }
    
    param->iter = total_iters;
    param->secs = total_secs;
    param->gflops = total_gflops;	
    delete high_x;
  }
  
  
  for(int i=0; i < param->num_offset; i++){ 
    delete h_x[i];
  }

  delete h_b;
  delete b;

  delete [] h_x;
  delete [] x;

  delete [] hp_x;
  
  delete dPre;
  delete dSloppy;
  delete d;


  // FIXME: added temporarily so that the cache is written out even if a long benchmarking job gets interrupted
  saveTuneCache(getVerbosity());
}


#ifdef GPU_FATLINK 
/*   @method  
 *   QUDA_COMPUTE_FAT_STANDARD: standard method (default)
 *   QUDA_COMPUTE_FAT_EXTENDED_VOLUME, extended volume method
 *
 */
#include <sys/time.h>

void setFatLinkPadding(QudaComputeFatMethod method, QudaGaugeParam* param)
{
  int* X    = param->X;
  int Vsh_x = X[1]*X[2]*X[3]/2;
  int Vsh_y = X[0]*X[2]*X[3]/2;
  int Vsh_z = X[0]*X[1]*X[3]/2;
  int Vsh_t = X[0]*X[1]*X[2]/2;

  int E1 = X[0] + 4;
  int E2 = X[1] + 4;
  int E3 = X[2] + 4;
  int E4 = X[3] + 4;

  // fat-link padding 
  param->llfat_ga_pad = Vsh_t;

  // site-link padding
  if(method ==  QUDA_COMPUTE_FAT_STANDARD){
#ifdef MULTI_GPU
    int Vh_2d_max = MAX(X[0]*X[1]/2, X[0]*X[2]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[0]*X[3]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[1]*X[2]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[1]*X[3]/2);
    Vh_2d_max = MAX(Vh_2d_max, X[2]*X[3]/2);
    param->site_ga_pad = 3*(Vsh_x+Vsh_y+Vsh_z+Vsh_t) + 4*Vh_2d_max;
#else
    param->site_ga_pad = Vsh_t;
#endif
  }else{
    param->site_ga_pad = E1*E2*E3/2*3;
  }
  param->ga_pad = param->site_ga_pad;

 // staple padding
  if(method == QUDA_COMPUTE_FAT_STANDARD){
#ifdef MULTI_GPU
    param->staple_pad = 3*(Vsh_x + Vsh_y + Vsh_z+ Vsh_t);
#else
    param->staple_pad = 3*Vsh_t;
#endif
  }else{
    param->staple_pad  = E1*E2*E3/2*3;
  }

  return;
}


void computeFatLinkCore(cudaGaugeField* cudaSiteLink, double* act_path_coeff,
			QudaGaugeParam* qudaGaugeParam, QudaComputeFatMethod method,
			cudaGaugeField* cudaFatLink, struct timeval time_array[])
{
 
  gettimeofday(&time_array[0], NULL);
  
  const int flag = qudaGaugeParam->preserve_gauge;
  GaugeFieldParam gParam(0,*qudaGaugeParam);

  if(method == QUDA_COMPUTE_FAT_STANDARD){
    for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam->X[dir];
  }else{
    for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam->X[dir] + 4;
  }


  static cudaGaugeField* cudaStapleField=NULL, *cudaStapleField1=NULL;
  if(cudaStapleField == NULL || cudaStapleField1 == NULL){
    gParam.pad    = qudaGaugeParam->staple_pad;
    gParam.create = QUDA_NULL_FIELD_CREATE;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.geometry = QUDA_SCALAR_GEOMETRY; // only require a scalar matrix field for the staple
    cudaStapleField  = new cudaGaugeField(gParam);
    cudaStapleField1 = new cudaGaugeField(gParam);
  }

  gettimeofday(&time_array[1], NULL);

  if(method == QUDA_COMPUTE_FAT_STANDARD){
    llfat_cuda(*cudaFatLink, *cudaSiteLink, *cudaStapleField, *cudaStapleField1, qudaGaugeParam, act_path_coeff);
  }else{ //method == QUDA_COMPUTE_FAT_EXTENDED_VOLUME
    llfat_cuda_ex(*cudaFatLink, *cudaSiteLink, *cudaStapleField, *cudaStapleField1, qudaGaugeParam, act_path_coeff);
  }
  gettimeofday(&time_array[2], NULL);

  if (!(flag & QUDA_FAT_PRESERVE_GPU_GAUGE) ){
    delete cudaStapleField; cudaStapleField = NULL;
    delete cudaStapleField1; cudaStapleField1 = NULL;
  }
  gettimeofday(&time_array[3], NULL);

  return;

}



int
computeFatLinkQuda(void* fatlink, void** sitelink, double* act_path_coeff, 
		   QudaGaugeParam* qudaGaugeParam, 
		   QudaComputeFatMethod method)
{
#define TDIFF_MS(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))*1000

  struct timeval t0;
  struct timeval t7, t8, t9, t10, t11;

  gettimeofday(&t0, NULL);

  static cpuGaugeField* cpuFatLink=NULL, *cpuSiteLink=NULL;
  static cudaGaugeField* cudaFatLink=NULL, *cudaSiteLink=NULL;
  int flag = qudaGaugeParam->preserve_gauge;

  QudaGaugeParam qudaGaugeParam_ex_buf;
  QudaGaugeParam* qudaGaugeParam_ex = &qudaGaugeParam_ex_buf;
  memcpy(qudaGaugeParam_ex, qudaGaugeParam, sizeof(QudaGaugeParam));

  qudaGaugeParam_ex->X[0] = qudaGaugeParam->X[0]+4;
  qudaGaugeParam_ex->X[1] = qudaGaugeParam->X[1]+4;
  qudaGaugeParam_ex->X[2] = qudaGaugeParam->X[2]+4;
  qudaGaugeParam_ex->X[3] = qudaGaugeParam->X[3]+4;

  GaugeFieldParam gParam_ex(0, *qudaGaugeParam_ex);
  
  // fat-link padding
  setFatLinkPadding(method, qudaGaugeParam);
  qudaGaugeParam_ex->llfat_ga_pad = qudaGaugeParam->llfat_ga_pad;
  qudaGaugeParam_ex->staple_pad   = qudaGaugeParam->staple_pad;
  qudaGaugeParam_ex->site_ga_pad  = qudaGaugeParam->site_ga_pad;
  
  GaugeFieldParam gParam(0, *qudaGaugeParam);

  // create the host fatlink
  if(cpuFatLink == NULL){
    gParam.create = QUDA_REFERENCE_FIELD_CREATE;
    gParam.link_type = QUDA_ASQTAD_FAT_LINKS;
    gParam.order = QUDA_MILC_GAUGE_ORDER;
    gParam.gauge= fatlink;
    cpuFatLink = new cpuGaugeField(gParam);
    if(cpuFatLink == NULL){
      errorQuda("ERROR: Creating cpuFatLink failed\n");
    }
  }else{
    cpuFatLink->setGauge((void**)fatlink);
  }
  
 // create the device fatlink
  if(cudaFatLink == NULL){
    gParam.pad    = qudaGaugeParam->llfat_ga_pad;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    gParam.link_type = QUDA_ASQTAD_FAT_LINKS;
    gParam.order = QUDA_QDP_GAUGE_ORDER;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    cudaFatLink = new cudaGaugeField(gParam);
  }



  // create the host sitelink	
  if(cpuSiteLink == NULL){
    gParam.pad = 0; 
    gParam.create    = QUDA_REFERENCE_FIELD_CREATE;
    gParam.link_type = qudaGaugeParam->type;
    gParam.order     = qudaGaugeParam->gauge_order;
    gParam.gauge     = sitelink;
    if(method != QUDA_COMPUTE_FAT_STANDARD){
      for(int dir=0; dir<4; ++dir) gParam.x[dir] = qudaGaugeParam_ex->X[dir];	
    }
    cpuSiteLink      = new cpuGaugeField(gParam);
    if(cpuSiteLink == NULL){
      errorQuda("ERROR: Creating cpuSiteLink failed\n");
    }
  }else{
    cpuSiteLink->setGauge(sitelink);
  }
  
  if(cudaSiteLink == NULL){
    gParam.pad         = qudaGaugeParam->site_ga_pad;
    gParam.create      = QUDA_NULL_FIELD_CREATE;
    gParam.link_type   = qudaGaugeParam->type;
    gParam.reconstruct = qudaGaugeParam->reconstruct;      
    cudaSiteLink = new cudaGaugeField(gParam);
  }
  
  initLatticeConstants(*cudaFatLink);  
  
  if(method == QUDA_COMPUTE_FAT_STANDARD){
    llfat_init_cuda(qudaGaugeParam);
    gettimeofday(&t7, NULL);
    
#ifdef MULTI_GPU
    if(qudaGaugeParam->gauge_order == QUDA_MILC_GAUGE_ORDER){
      errorQuda("Only QDP-ordered site links are supported in the multi-gpu standard fattening code\n");
    }
#endif
    loadLinkToGPU(cudaSiteLink, cpuSiteLink, qudaGaugeParam);
    
    gettimeofday(&t8, NULL);
  }else{
    llfat_init_cuda_ex(qudaGaugeParam_ex);
#ifdef MULTI_GPU
    int R[4] = {2, 2, 2, 2}; // radius of the extended region in each dimension / direction
    exchange_cpu_sitelink_ex(qudaGaugeParam->X, R, (void**)cpuSiteLink->Gauge_p(), 
			     cpuSiteLink->Order(),qudaGaugeParam->cpu_prec, 0);
#endif
    gettimeofday(&t7, NULL);
    loadLinkToGPU_ex(cudaSiteLink, cpuSiteLink);
    gettimeofday(&t8, NULL);
  }

  // time the subroutines in computeFatLinkCore
  struct timeval time_array[4];
  
  // Actually do the fattening
  computeFatLinkCore(cudaSiteLink, act_path_coeff, qudaGaugeParam, method, cudaFatLink, time_array);
 

  gettimeofday(&t9, NULL);

  storeLinkToCPU(cpuFatLink, cudaFatLink, qudaGaugeParam);
  
  gettimeofday(&t10, NULL);
  
  if (!(flag & QUDA_FAT_PRESERVE_CPU_GAUGE) ){
    delete cpuFatLink; cpuFatLink = NULL;
    delete cpuSiteLink; cpuSiteLink = NULL;
  }  
  if (!(flag & QUDA_FAT_PRESERVE_GPU_GAUGE) ){
    delete cudaFatLink; cudaFatLink = NULL;
    delete cudaSiteLink; cudaSiteLink = NULL;
  }
  
  gettimeofday(&t11, NULL);
#ifdef DSLASH_PROFILING 
  printfQuda("total time: %f ms, init(cuda/cpu gauge field creation,etc)=%f ms,"
	     " sitelink cpu->gpu=%f ms, computation in gpu =%f ms, fatlink gpu->cpu=%f ms\n",
	     TDIFF_MS(t0, t11), TDIFF_MS(t0, t7) + TDIFF_MS(time_array[0],time_array[1]), TDIFF_MS(t7, t8), TDIFF_MS(time_array[1], time_array[2]), TDIFF_MS(t9,t10));
  printfQuda("finally cleanup =%f ms\n", TDIFF_MS(t10, t11) + TDIFF_MS(time_array[2],time_array[3]));
#endif

  return 0;
}
#endif

#ifdef GPU_GAUGE_FORCE
int
computeGaugeForceQuda(void* mom, void* sitelink,  int*** input_path_buf, int* path_length,
                      void* loop_coeff, int num_paths, int max_length, double eb3,
                      QudaGaugeParam* qudaGaugeParam, double* timeinfo)
{
  
  struct timeval t0, t1, t2, t3, t4;
  
  gettimeofday(&t0,NULL);

#ifdef MULTI_GPU
  int E[4];
  QudaGaugeParam qudaGaugeParam_ex_buf;
  QudaGaugeParam* qudaGaugeParam_ex=&qudaGaugeParam_ex_buf;
  memcpy(qudaGaugeParam_ex, qudaGaugeParam, sizeof(QudaGaugeParam));
  E[0] = qudaGaugeParam_ex->X[0] = qudaGaugeParam->X[0] + 4;
  E[1] = qudaGaugeParam_ex->X[1] = qudaGaugeParam->X[1] + 4;
  E[2] = qudaGaugeParam_ex->X[2] = qudaGaugeParam->X[2] + 4;
  E[3] = qudaGaugeParam_ex->X[3] = qudaGaugeParam->X[3] + 4;
#endif

  int* X = qudaGaugeParam->X;
  GaugeFieldParam gParam(0, *qudaGaugeParam);
#ifdef MULTI_GPU
  GaugeFieldParam gParam_ex(0, *qudaGaugeParam_ex);
  GaugeFieldParam& gParamSL = gParam_ex;  
  int pad = E[2]*E[1]*E[0]/2;
#else
  GaugeFieldParam& gParamSL = gParam;
  int pad = X[2]*X[1]*X[0]/2;
#endif
  
  GaugeFieldParam& gParamMom = gParam;
  
  gParamSL.create = QUDA_REFERENCE_FIELD_CREATE;
  gParamSL.gauge = sitelink;
  gParamSL.pad = 0;
  cpuGaugeField* cpuSiteLink = new cpuGaugeField(gParamSL);
  
  gParamSL.create =QUDA_NULL_FIELD_CREATE;
  gParamSL.pad = pad;
  gParamSL.precision = qudaGaugeParam->cuda_prec;
  gParamSL.reconstruct = qudaGaugeParam->reconstruct;
  cudaGaugeField* cudaSiteLink = new cudaGaugeField(gParamSL);  
  qudaGaugeParam->site_ga_pad = gParamSL.pad;//need to record this value

  gParamMom.pad = 0;
  gParamMom.order =QUDA_MILC_GAUGE_ORDER;
  gParamMom.precision = qudaGaugeParam->cpu_prec;
  gParamMom.create =QUDA_REFERENCE_FIELD_CREATE;
  gParamMom.reconstruct =QUDA_RECONSTRUCT_10;  
  gParamMom.gauge=mom;
  gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
  cpuGaugeField* cpuMom = new cpuGaugeField(gParamMom);              


  gParamMom.pad = pad;
  gParamMom.create =QUDA_NULL_FIELD_CREATE;  
  gParamMom.reconstruct = QUDA_RECONSTRUCT_10;
  gParamMom.precision = qudaGaugeParam->cuda_prec;
  gParamMom.link_type = QUDA_ASQTAD_MOM_LINKS;
  cudaGaugeField* cudaMom = new cudaGaugeField(gParamMom);
  qudaGaugeParam->mom_ga_pad = gParamMom.pad; //need to record this value
  
  
  initLatticeConstants(*cudaMom);
  gauge_force_init_cuda(qudaGaugeParam, max_length); 
  
#ifdef MULTI_GPU
  int R[4] = {2, 2, 2, 2}; // radius of the extended region in each dimension / direction
  exchange_cpu_sitelink_ex(qudaGaugeParam->X, R, (void**)cpuSiteLink->Gauge_p(), 
			   cpuSiteLink->Order(), qudaGaugeParam->cpu_prec, 1);
  loadLinkToGPU_ex(cudaSiteLink, cpuSiteLink);
#else  
  loadLinkToGPU(cudaSiteLink, cpuSiteLink, qudaGaugeParam);    
#endif
  
  cudaMom->loadCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);

  gettimeofday(&t1,NULL);  
  
  gauge_force_cuda(*cudaMom, eb3, *cudaSiteLink, qudaGaugeParam, input_path_buf, 
		   path_length, loop_coeff, num_paths, max_length);

  gettimeofday(&t2,NULL);

  cudaMom->saveCPUField(*cpuMom, QUDA_CPU_FIELD_LOCATION);
  
  delete cpuSiteLink;
  delete cpuMom;
  
  delete cudaSiteLink;
  delete cudaMom;
  
  gettimeofday(&t3,NULL); 

  if(timeinfo){
    timeinfo[0] = TDIFF(t0, t1);
    timeinfo[1] = TDIFF(t1, t2);
    timeinfo[2] = TDIFF(t2, t3);
  }

  return 0;  
}

#endif


void initCommsQuda(int argc, char **argv, const int *X, int nDim) {

  if (nDim != 4) errorQuda("Comms dimensions %d != 4", nDim);

#ifdef MULTI_GPU

#ifdef QMP_COMMS
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  QMP_declare_logical_topology(X, nDim);
#elif defined(MPI_COMMS)
  MPI_Init (&argc, &argv);  

  int volume = 1;
  for (int d=0; d<nDim; d++) volume *= X[d];
  int size = -1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (volume != size)
    errorQuda("Number of processes %d must match requested MPI volume %d",
	      size, volume);

  comm_set_gridsize(X[0], X[1], X[2], X[3]);  
  comm_init();
#endif

#endif
}

void endCommsQuda() {
#ifdef MULTI_GPU

#ifdef QMP_COMMS
  QMP_finalize_msg_passing();
#elif defined MPI_COMMS
  comm_cleanup();
#endif 

#endif
}
