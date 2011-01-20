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
#include "exchange_face.h"
#include "gauge_quda.h"
#include <mpi.h>
#include "mpicomm.h"

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
void *cpu_fwd_nbr_spinor, *cpu_back_nbr_spinor;

cudaColorSpinorField* tmp;

void *hostGauge[4];
void *fatlink[4], *longlink[4];
void* ghost_fatlink, *ghost_longlink;


QudaParity parity;
QudaDagType dagger = QUDA_DAG_NO;
int transfer = 0; // include transfer time in the benchmark?
int tdim = 24;
int sdim = 24;
int X[4];

QudaReconstructType link_recon = QUDA_RECONSTRUCT_12;
QudaPrecision prec = QUDA_SINGLE_PRECISION;

Dirac* dirac;

void init()
{    

  initQuda(device);

  int Vs = sdim*sdim*sdim;
  int Vsh = Vs/2;

  gaugeParam = newQudaGaugeParam();
  inv_param = newQudaInvertParam();
  
  gaugeParam.X[0] = X[0] = sdim;
  gaugeParam.X[1] = X[1] = sdim;
  gaugeParam.X[2] = X[2] = sdim;
  gaugeParam.X[3] = X[3] = tdim;

  gaugeParam.X[0] = sdim;
  gaugeParam.X[1] = sdim;
  gaugeParam.X[2] = sdim;
  gaugeParam.X[3] = tdim;

  setDims(gaugeParam.X);

  Vh = sdim*sdim*sdim*tdim/2;

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
  inv_param.dagger = dagger;
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dslash_type = QUDA_ASQTAD_DSLASH;
    
  gaugeParam.ga_pad = sdim*sdim*sdim/2;
  inv_param.sp_pad = sdim*sdim*sdim/2;

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
  csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  csParam.create = QUDA_ZERO_FIELD_CREATE;    

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gaugeParam.X[0];
    
  printfQuda("Randomizing fields ...\n");
    
  spinor->Source(QUDA_RANDOM_SOURCE);

  //create ghost spinors
  cpu_fwd_nbr_spinor = malloc(Vsh* staggeredSpinorSiteSize *3*sizeof(double));
  cpu_back_nbr_spinor = malloc(Vsh*staggeredSpinorSiteSize *3*sizeof(double));
  if (cpu_fwd_nbr_spinor == NULL || cpu_back_nbr_spinor == NULL){
    PRINTF("ERROR: malloc failed for cpu_fwd_nbr_spinor/cpu_back_nbr_spinor\n");
    exit(1);
  }


  size_t gSize = (gaugeParam.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
    
  for (int dir = 0; dir < 4; dir++) {
    fatlink[dir] = malloc(V*gaugeSiteSize*gSize);
    longlink[dir] = malloc(V*gaugeSiteSize*gSize);
  }
  if (fatlink == NULL || longlink == NULL){
    fprintf(stderr, "ERROR: malloc failed for fatlink/longlink\n");
    exit(1);
  }

  ghost_fatlink = malloc(Vs*gaugeSiteSize*gSize);
  ghost_longlink = malloc(3*Vs*gaugeSiteSize*gSize);
  if (ghost_fatlink == NULL || ghost_longlink == NULL){
    PRINTF("ERROR: malloc failed for ghost fatlink/longlink\n");
    exit(1);
  }
   
  construct_fat_long_gauge_field(fatlink, longlink, 1, gaugeParam.cpu_prec, &gaugeParam);
  
  exchange_cpu_links(X, fatlink, ghost_fatlink, longlink, ghost_longlink, gaugeParam.cpu_prec);
  



#if 0
  //printf("links are:\n");
  //display_link(fatlink[0], 1, gaugeParam.cpu_prec);
  //display_link(longlink[0], 1, gaugeParam.cpu_prec);
    
  for (int i =0;i < 4 ;i++){
    int dir = 2*i;
    link_sanity_check(fatlink[i], V, gaugeParam.cpu_prec, dir, &gaugeParam);
    link_sanity_check(longlink[i], V, gaugeParam.cpu_prec, dir, &gaugeParam);
  }

  //printf("spinors are:\n");  
  //display_spinor(spinor, 10, inv_param.cpu_prec);
#endif


#if 0
  gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  loadGaugeQuda(fatlink, &gaugeParam);

#else
  
  int num_faces =1;
  gaugeParam.ga_pad = sdim*sdim*sdim/2;    
  gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  loadGaugeQuda_general_mg(fatlink, ghost_fatlink, &gaugeParam, &cudaFatLinkPrecise, &cudaFatLinkSloppy, num_faces, GAUGE_STAGGERED_FAT);
  
  
#endif
  

#if 0
  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = link_recon;
  loadGaugeQuda(longlink, &gaugeParam);
#else
  num_faces =3;
  gaugeParam.ga_pad = 3*sdim*sdim*sdim/2;    
  gaugeParam.reconstruct= gaugeParam.reconstruct_sloppy = link_recon;
  loadGaugeQuda_general_mg(longlink, ghost_longlink, &gaugeParam, &cudaLongLinkPrecise, &cudaLongLinkSloppy, num_faces, GAUGE_STAGGERED_LONG);

#endif

  cudaFatLink = cudaFatLinkPrecise;
  cudaLongLink = cudaLongLinkPrecise;
    
  printf("Sending fields to GPU..."); fflush(stdout);
    
  if (!transfer) {
	
    csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
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
	
    printf("Source CPU = %f, CUDA=%f\n", norm2(*spinor), norm2(*cudaSpinor));
	
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

    diracParam.verbose = QUDA_VERBOSE;
    diracParam.tmp1=tmp;
    dirac = Dirac::create(diracParam);
	
  } else {
    printf("ERROR: not suppported\n");
  }
    
  return;
}

void end(void) 
{
  for (int dir = 0; dir < 4; dir++) {
    free(fatlink[dir]);
    free(longlink[dir]);
  }
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
  const int LOOPS = 1;
  printf("Executing %d kernel loops...", LOOPS);
  fflush(stdout);
  stopwatchStart();
  for (int i = 0; i < LOOPS; i++) {
    switch (test_type) {
    case 0:
      parity = QUDA_EVEN_PARITY;
      if (transfer){
	//dslashQuda(spinorOdd, spinorEven, &inv_param, parity);
      }
      else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }	   
      break;
    case 1:
      parity = QUDA_ODD_PARITY;
      if (transfer){
	//MatPCQuda(spinorOdd, spinorEven, &inv_param);
      }else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }
      break;
    case 2:
      if (transfer){
	//MatQuda(spinorGPU, spinor, &inv_param);
      }
      else {
	dirac->M(*cudaSpinorOut, *cudaSpinor);
      }
    }
  }
    
  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printf("with ERROR: %s\n", cudaGetErrorString(stat));
    
  cudaThreadSynchronize();
  double secs = stopwatchReadSeconds() / LOOPS;
  printf("done.\n\n");
    
  return secs;
}

void staggeredDslashRef()
{
  int cpu_parity;
  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
  case 0:    
    /*
    cpu_parity = 0; //EVEN
    staggered_dslash(spinorRef->v, fatlink, longlink, spinor->v, cpu_parity, dagger, 
		     inv_param.cpu_prec, gaugeParam.cpu_prec);
    */
    
    staggered_dslash_mg(spinorRef->v, fatlink, longlink, ghost_fatlink, ghost_longlink, 
			spinor->v, cpu_fwd_nbr_spinor, cpu_back_nbr_spinor, parity, dagger,
			inv_param.cpu_prec, gaugeParam.cpu_prec);

    break;
  case 1: 
    cpu_parity=1; //ODD
    staggered_dslash(spinorRef->v, fatlink, longlink, spinor->v, cpu_parity, dagger, 
		     inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  case 2:
    //mat(spinorRef->v, fatlink, longlink, spinor->v, kappa, dagger, 
    //inv_param.cpu_prec, gaugeParam.cpu_prec);
    break;
  default:
    printf("Test type not defined\n");
    exit(-1);
  }
    
  printf("done.\n");
    
}

static int dslashTest() 
{
  int accuracy_level = 0;
  
  init();
    
  int attempts = 1;

  staggeredDslashRef();
    
  for (int i=0; i<attempts; i++) {
	
    double secs = dslashCUDA();
    
    if (!transfer) {
      *spinorOut = *cudaSpinorOut;
    }
      
    printf("\n%fms per loop\n", 1000*secs);
	
    int flops = dirac->Flops();
    int link_floats = 8*gaugeParam.reconstruct+8*18;
    int spinor_floats = 8*6*2 + 6;
    int link_float_size = prec;
    int spinor_float_size = 0;
    
    link_floats = test_type ? (2*link_floats) : link_floats;
    spinor_floats = test_type ? (2*spinor_floats) : spinor_floats;

    int bytes_for_one_site = link_floats * link_float_size + spinor_floats * spinor_float_size;
    if (prec == QUDA_HALF_PRECISION) {
      bytes_for_one_site += (8*2 + 1)*4;	
    }
    printf("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printf("GiB/s = %f\n\n", 1.0*Vh*bytes_for_one_site/(secs*(1<<30)));
	
    if (!transfer) {
      std::cout << "Results: CPU = " << norm2(*spinorRef) << ", CUDA = " << norm2(*cudaSpinorOut) << 
	", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    } else {
      std::cout << "Result: CPU = " << norm2(*spinorRef) << ", CPU-CUDA = " << norm2(*spinorOut) << std::endl;
    }
    
    accuracy_level = cpuColorSpinorField::Compare(*spinorRef, *spinorOut);	
  }
  end();
  
  return accuracy_level;
}


void display_test_info()
{
  printf("running the following test:\n");
 
  printf("prec recon   test_type     dagger   S_dim     T_dimension\n");
  printf("%s   %s       %d           %d       %d        %d \n", 
	 get_prec_str(prec), get_recon_str(link_recon), 
	 test_type, dagger, sdim, tdim);
  return ;
    
}

void usage(char** argv )
{
  printf("Usage: %s <args>\n", argv[0]);
  printf("--prec <double/single/half> \t Precision in GPU\n"); 
  printf("--recon <8/12> \t\t\t Long link reconstruction type\n"); 
  printf("--type <0/1/2> \t\t\t Test type\n"); 
  printf("--dagger \t\t\t Set the dagger to 1\n"); 
  printf("--tdim \t\t\t\t Set T dimention size(default 24)\n");     
  printf("--sdim \t\t\t\t Set space dimention size\n"); 
  printf("--help \t\t\t\t Print out this message\n"); 
  exit(1);
  return ;
}

int main(int argc, char **argv) 
{
  MPI_Init (&argc, &argv);
  comm_init();
  
  int i;
  for (i =1;i < argc; i++){
	
    if( strcmp(argv[i], "--help")== 0){
      usage(argv);
    }

    if( strcmp(argv[i], "--device") == 0){
      if (i+1 >= argc){
	usage(argv);
      }
      device =  atoi(argv[i+1]);
      if (device < 0){
	fprintf(stderr, "Error: invalid device number(%d)\n", device);
	exit(1);
      }
      i++;
      continue;
    }
	
    if( strcmp(argv[i], "--prec") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      prec =  get_prec(argv[i+1]);
      i++;
      continue;	    
    }
	
	
    if( strcmp(argv[i], "--recon") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      link_recon =  get_recon(argv[i+1]);
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--test") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      test_type =  atoi(argv[i+1]);
      if (test_type < 0 || test_type >= 2){
	fprintf(stderr, "Error: invalid test type\n");
	exit(1);
      }
      i++;
      continue;	    
    }

    if( strcmp(argv[i], "--tdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      tdim =  atoi(argv[i+1]);
      if (tdim < 0 || tdim > 128){
	fprintf(stderr, "Error: invalid t dimention\n");
	exit(1);
      }
      i++;
      continue;	    
    }

    if( strcmp(argv[i], "--sdim") == 0){
      if (i+1 >= argc){
	usage(argv);
      }	    
      sdim =  atoi(argv[i+1]);
      if (sdim < 0 || sdim > 128){
	fprintf(stderr, "Error: invalid S dimention\n");
	exit(1);
      }
      i++;
      continue;	    
    }
	
    if( strcmp(argv[i], "--dagger") == 0){
      dagger = QUDA_DAG_YES;
      continue;	    
    }	

    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  display_test_info();

  int ret =1;
  int accuracy_level = dslashTest();

  printf("accuracy_level =%d\n", accuracy_level);
  if (accuracy_level >= 3){
    //probably no error 
    ret = 0;
  }

  return ret;
}
