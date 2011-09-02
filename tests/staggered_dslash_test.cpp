#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <misc.h>
#include <test_util.h>
#include <staggered_dslash_reference.h>
#include "gauge_quda.h"

#include <face_quda.h>

#include <assert.h>

#define MAX(a,b) ((a)>(b)?(a):(b))
#define staggeredSpinorSiteSize 6
// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
int test_type = 0;
int device = 0;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

FullGauge cudaFatLink;
FullGauge cudaLongLink;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;

cudaColorSpinorField* tmp;

void *hostGauge[4];
void *fatlink[4], *longlink[4];

#ifdef MULTI_GPU
void* ghost_fatlink[4], *ghost_longlink[4];
#endif

const int loops = 100;

QudaParity parity;
extern QudaDagType dagger;
int transfer = 0; // include transfer time in the benchmark?
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;

int X[4];


Dirac* dirac;
extern int Z[4];
extern int V;
extern int Vh;
static int Vs_x, Vs_y, Vs_z, Vs_t;
extern int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
static int Vsh[4];

void
setDimConstants(int *X)
{
  V = 1;
  for (int d=0; d< 4; d++) {
    V *= X[d];
    Z[d] = X[d];
  }
  Vh = V/2;

  Vs_x = X[1]*X[2]*X[3];
  Vs_y = X[0]*X[2]*X[3];
  Vs_z = X[0]*X[1]*X[3];
  Vs_t = X[0]*X[1]*X[2];


  Vsh_x = Vs_x/2;
  Vsh_y = Vs_y/2;
  Vsh_z = Vs_z/2;
  Vsh_t = Vs_t/2;

  Vsh[0] = Vsh_x;
  Vsh[1] = Vsh_y;
  Vsh[2] = Vsh_z;
  Vsh[3] = Vsh_t;
}

void init()
{    

  initQuda(device);

  gaugeParam = newQudaGaugeParam();
  inv_param = newQudaInvertParam();
  
  gaugeParam.X[0] = X[0] = xdim;
  gaugeParam.X[1] = X[1] = ydim;
  gaugeParam.X[2] = X[2] = zdim;
  gaugeParam.X[3] = X[3] = tdim;

  setDims(gaugeParam.X);

  setDimConstants(gaugeParam.X);

  gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;
    
  gaugeParam.tadpole_coeff = 0.8;
  gaugeParam.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gaugeParam.t_boundary = QUDA_ANTI_PERIODIC_T;
  gaugeParam.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gaugeParam.gaugeGiB = 0;
    
  inv_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  inv_param.cuda_prec = prec;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;
  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dagger = dagger;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dslash_type = QUDA_ASQTAD_DSLASH;

  int tmpint = MAX(X[1]*X[2]*X[3], X[0]*X[2]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[2]);
  
  
  gaugeParam.ga_pad = tmpint;
  inv_param.sp_pad = tmpint;

  ColorSpinorParam csParam;
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor=3;
  csParam.nSpin=1;
  csParam.nDim=4;
  for(int d = 0; d < 4; d++) {
    csParam.x[d] = gaugeParam.X[d];
  }
  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  if (test_type < 2) {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;	
  }

  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder  = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis; // this parameter is meaningless for staggered
  csParam.create = QUDA_ZERO_FIELD_CREATE;    

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gaugeParam.X[0];
    
  printfQuda("Randomizing fields ...\n");
    
  spinor->Source(QUDA_RANDOM_SOURCE);

  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    
  for (int dir = 0; dir < 4; dir++) {
    fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  if (fatlink == NULL || longlink == NULL){
    errorQuda("ERROR: malloc failed for fatlink/longlink");
  }
  construct_fat_long_gauge_field(fatlink, longlink, 1, gaugeParam.cpu_prec, &gaugeParam);
  

#ifdef MULTI_GPU

  //exchange_init_dims(X);
  ghost_fatlink[0] = malloc(Vs_x*gaugeSiteSize*gSize);
  ghost_fatlink[1] = malloc(Vs_y*gaugeSiteSize*gSize);
  ghost_fatlink[2] = malloc(Vs_z*gaugeSiteSize*gSize);
  ghost_fatlink[3] = malloc(Vs_t*gaugeSiteSize*gSize);
  ghost_longlink[0] = malloc(3*Vs_x*gaugeSiteSize*gSize);
  ghost_longlink[1] = malloc(3*Vs_y*gaugeSiteSize*gSize);
  ghost_longlink[2] = malloc(3*Vs_z*gaugeSiteSize*gSize);
  ghost_longlink[3] = malloc(3*Vs_t*gaugeSiteSize*gSize);
  if (ghost_fatlink[0] == NULL || ghost_longlink[0] == NULL ||
      ghost_fatlink[1] == NULL || ghost_longlink[1] == NULL ||
      ghost_fatlink[2] == NULL || ghost_longlink[2] == NULL ||
      ghost_fatlink[3] == NULL || ghost_longlink[3] == NULL){
    errorQuda("ERROR: malloc failed for ghost fatlink/longlink");
  }
  //exchange_cpu_links(fatlink, ghost_fatlink[3], longlink, ghost_longlink[3], gaugeParam.cpu_prec);
  //exchange_cpu_links4dir(fatlink, ghost_fatlink, longlink, ghost_longlink, gaugeParam.cpu_prec);

  void *fat_send[4], *long_send[4];
  fat_send[0] = malloc(Vs_x*gaugeSiteSize*gSize);
  fat_send[1] = malloc(Vs_y*gaugeSiteSize*gSize);
  fat_send[2] = malloc(Vs_z*gaugeSiteSize*gSize);
  fat_send[3] = malloc(Vs_t*gaugeSiteSize*gSize);
  long_send[0] = malloc(3*Vs_x*gaugeSiteSize*gSize);
  long_send[1] = malloc(3*Vs_y*gaugeSiteSize*gSize);
  long_send[2] = malloc(3*Vs_z*gaugeSiteSize*gSize);
  long_send[3] = malloc(3*Vs_t*gaugeSiteSize*gSize);

  set_dim(Z);
  pack_ghost(fatlink, fat_send, 1, gaugeParam.cpu_prec);
  pack_ghost(longlink, long_send, 3, gaugeParam.cpu_prec);


  {
    FaceBuffer faceBuf(X, 4, 18, 1, gaugeParam.cpu_prec);
    faceBuf.exchangeCpuLink((void**)ghost_fatlink, (void**)fat_send);
  }

  {    
    FaceBuffer faceBuf(X, 4, 18, 3, gaugeParam.cpu_prec);
    faceBuf.exchangeCpuLink((void**)ghost_longlink, (void**)long_send);
  }

  for (int i=0; i<4; i++) {
    free(fat_send[i]);
    free(long_send[i]);
  }
  
#endif   

  
  gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
#ifdef MULTI_GPU
  int x_face_size = X[1]*X[2]*X[3]/2;
  int y_face_size = X[0]*X[2]*X[3]/2;
  int z_face_size = X[0]*X[1]*X[3]/2;
  int t_face_size = X[0]*X[1]*X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif
  gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  printfQuda("Fat links sending..."); 
  loadGaugeQuda(fatlink, &gaugeParam);
  printfQuda("Fat links sent"); 
  
  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;  
#ifdef MULTI_GPU
  gaugeParam.ga_pad = 3*pad_size;
#endif
  gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = link_recon;
  printfQuda("Long links sending..."); 
  loadGaugeQuda(longlink, &gaugeParam);
  printfQuda("Long links sent..."); 

  cudaFatLink = cudaFatLinkPrecise;
  cudaLongLink = cudaLongLinkPrecise;
  
  printfQuda("Sending fields to GPU..."); 
    
  if (!transfer) {

    //csParam.verbose = QUDA_DEBUG_VERBOSE;
	
    csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (test_type < 2){
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /=2;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);

    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);
	
    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;
	
    cudaThreadSynchronize();
    checkCudaError();
	
    double spinor_norm2 = norm2(*spinor);
    double cuda_spinor_norm2=  norm2(*cudaSpinor);
    printfQuda("Source CPU = %f, CUDA=%f\n", spinor_norm2, cuda_spinor_norm2);
	
    if(test_type == 2){
      csParam.x[0] /=2;
    }
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp = new cudaColorSpinorField(csParam);

    bool pc = (test_type != 2);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.fatGauge = &cudaFatLinkPrecise;
    diracParam.longGauge = &cudaLongLinkPrecise;

    //diracParam.verbose = QUDA_DEBUG_VERBOSE;
    diracParam.tmp1=tmp;

    dirac = Dirac::create(diracParam);
	
    //dirac->Tune(*cudaSpinorOut, *cudaSpinor, *tmp);

  } else {
    errorQuda("Error not suppported");
  }
    
  return;
}

void end(void) 
{
  for (int dir = 0; dir < 4; dir++) {
    free(fatlink[dir]);
    free(longlink[dir]);
  }

#ifdef MULTI_GPU
  for(int i=0;i < 4;i++){
    free(ghost_fatlink[i]);
    free(ghost_longlink[i]);
  }

#endif

  if (!transfer){
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp;
  }
    
  delete spinor;
  delete spinorOut;
  delete spinorRef;
    
  endQuda();
}

double dslashCUDA() {
    
  // execute kernel
  printfQuda("Executing %d kernel loops...", loops);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  for (int i = 0; i < loops; i++) {
    switch (test_type) {
    case 0:
      parity = QUDA_EVEN_PARITY;
      if (transfer){
	//dslashQuda(spinorOdd, spinorEven, &inv_param, parity);
      } else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }	   
      break;
    case 1:
      parity = QUDA_ODD_PARITY;
      if (transfer){
	//MatPCQuda(spinorOdd, spinorEven, &inv_param);
      } else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }
      break;
    case 2:
      if (transfer){
	//MatQuda(spinorGPU, spinor, &inv_param);
      } else {
	dirac->M(*cudaSpinorOut, *cudaSpinor);
      }
    }
  }
    
  cudaEventCreate(&end);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  double secs = runTime / 1000; //stopwatchReadSeconds();

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    errorQuda("with ERROR: %s\n", cudaGetErrorString(stat));
    
  return secs;
}

void staggeredDslashRef()
{
#ifndef MULTI_GPU
  int cpu_parity = 0;
#endif

  // compare to dslash reference implementation
  printfQuda("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
  case 0:    
#ifdef MULTI_GPU

#if 1
    staggered_dslash_mg4dir(spinorRef, fatlink, longlink, ghost_fatlink, ghost_longlink, 
			    spinor, parity, dagger,
			    inv_param.cpu_prec, gaugeParam.cpu_prec);
#else

    staggered_dslash(spinorRef->v, fatlink, longlink, spinor->v, 0, dagger, 
		     inv_param.cpu_prec, gaugeParam.cpu_prec);
#endif


#else
    cpu_parity = 0; //EVEN
    staggered_dslash(spinorRef->v, fatlink, longlink, spinor->v, cpu_parity, dagger, 
		     inv_param.cpu_prec, gaugeParam.cpu_prec);
    
#endif    


    break;
  case 1: 
#ifdef MULTI_GPU
    staggered_dslash_mg4dir(spinorRef, fatlink, longlink, ghost_fatlink, ghost_longlink, 
			    spinor, parity, dagger,
			    inv_param.cpu_prec, gaugeParam.cpu_prec);    
    
#else
    cpu_parity=1; //ODD
    staggered_dslash(spinorRef->v, fatlink, longlink, spinor->v, cpu_parity, dagger, 
		     inv_param.cpu_prec, gaugeParam.cpu_prec);
#endif
    break;
  case 2:
    //mat(spinorRef->v, fatlink, longlink, spinor->v, kappa, dagger, 
    //inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  default:
    errorQuda("Test type not defined");
  }
    
  printfQuda("done.\n");
    
}

static int dslashTest() 
{
  int accuracy_level = 0;
  
  init();
    
  int attempts = 1;
    
  for (int i=0; i<attempts; i++) {
	
    double secs = dslashCUDA();
    
    if (!transfer) *spinorOut = *cudaSpinorOut;
      
    printfQuda("\n%fms per loop\n", 1000*secs);
    staggeredDslashRef();
	
    unsigned long long flops = dirac->Flops();
    int link_floats = 8*gaugeParam.reconstruct+8*18;
    int spinor_floats = 8*6*2 + 6;
    int link_float_size = prec;
    int spinor_float_size = 0;
    
    link_floats = test_type ? (2*link_floats) : link_floats;
    spinor_floats = test_type ? (2*spinor_floats) : spinor_floats;

    int bytes_for_one_site = link_floats * link_float_size + spinor_floats * spinor_float_size;
    if (prec == QUDA_HALF_PRECISION) bytes_for_one_site += (8*2 + 1)*4;	

    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printfQuda("GiB/s = %f\n\n", 1.0*Vh*bytes_for_one_site/((secs/loops)*(1<<30)));
	
    if (!transfer) {
      double spinor_ref_norm2 = norm2(*spinorRef);
      double cuda_spinor_out_norm2 =  norm2(*cudaSpinorOut);
      double spinor_out_norm2 =  norm2(*spinorOut);
      printfQuda("Results: CPU=%f, CUDA=%f, CPU-CUDA=%f\n",  spinor_ref_norm2, cuda_spinor_out_norm2,
		 spinor_out_norm2);
    } else {
      double spinor_ref_norm2 = norm2(*spinorRef);
      double spinor_out_norm2 =  norm2(*spinorOut);
      printfQuda("Result: CPU=%f , CPU-CUDA=%f", spinor_ref_norm2, spinor_out_norm2);
    }
    
    accuracy_level = cpuColorSpinorField::Compare(*spinorRef, *spinorOut);	
  }
  end();
  
  return accuracy_level;
}


void display_test_info()
{
  printfQuda("running the following test:\n");
 
  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %d           %d       %d/%d/%d        %d \n", 
	     get_prec_str(prec), get_recon_str(link_recon), 
	     test_type, dagger, xdim, ydim, zdim, tdim);
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     commDimPartitioned(0),
	     commDimPartitioned(1),
	     commDimPartitioned(2),
	     commDimPartitioned(3));

  return ;
    
}

extern void usage(char** argv );

int main(int argc, char **argv) 
{

  int i;
  for (i =1;i < argc; i++){
    
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }    

    if( strcmp(argv[i], "--device") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      device =  atoi(argv[i+1]);
      if (device < 0){
	errorQuda("Error: invalid device number(%d)", device);
      }
      i++;
      continue;
    }
    
	
    if( strcmp(argv[i], "--test") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      test_type =  atoi(argv[i+1]);
      if (test_type < 0 || test_type > 2){
	errorQuda("Error: invalid test type");
      }
      i++;
      continue;	    
    }
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }
  
  //qudaSetNumaConfig("/usr/local/gpu_numa_config.txt");
  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);
  
  display_test_info();

  int ret =1;
  int accuracy_level = dslashTest();

  printfQuda("accuracy_level =%d\n", accuracy_level);
  if (accuracy_level >= 1) ret = 0;    //probably no error, -1 means no matching  
  endCommsQuda();
  return ret;
}

