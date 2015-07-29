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
#include <dslash_util.h>
#include <staggered_dslash_reference.h>
#include <gauge_field.h>

#include <face_quda.h>

#include <assert.h>
#include <gtest.h>

using namespace quda;

#define MAX(a,b) ((a)>(b)?(a):(b))
#define staggeredSpinorSiteSize 6
// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)

extern void usage(char** argv );

extern QudaDslashType dslash_type;

extern int test_type;

extern bool tune;

QudaGaugeParam gaugeParam;
QudaInvertParam inv_param;

cpuGaugeField *cpuFat = NULL;
cpuGaugeField *cpuLong = NULL;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;

cudaColorSpinorField* tmp;

void *hostGauge[4];
void *fatlink[4], *longlink[4];

#ifdef MULTI_GPU
const void **ghost_fatlink, **ghost_longlink;
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

extern int device;
extern bool verify_results;

extern bool kernel_pack_t;

int X[4];

Dirac* dirac;

void init()
{    

  initQuda(device);

  setKernelPackT(kernel_pack_t);

  setVerbosity(QUDA_VERBOSE);

  gaugeParam = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gaugeParam.X[0] = X[0] = xdim;
  gaugeParam.X[1] = X[1] = ydim;
  gaugeParam.X[2] = X[2] = zdim;
  gaugeParam.X[3] = X[3] = tdim;

  setDims(gaugeParam.X);
  setSpinorSiteSize(6);

  gaugeParam.cpu_prec = QUDA_DOUBLE_PRECISION;
  gaugeParam.cuda_prec = prec;
  gaugeParam.reconstruct = link_recon;
  gaugeParam.reconstruct_sloppy = gaugeParam.reconstruct;
  gaugeParam.cuda_prec_sloppy = gaugeParam.cuda_prec;

    // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH &&
      dslash_type != QUDA_ASQTAD_DSLASH)
    dslash_type = QUDA_ASQTAD_DSLASH;

  gaugeParam.anisotropy = 1.0;
  gaugeParam.tadpole_coeff = 0.8;
  gaugeParam.scale = (dslash_type == QUDA_ASQTAD_DSLASH) ? -1.0/(24.0*gaugeParam.tadpole_coeff*gaugeParam.tadpole_coeff) : 1.0;
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
  inv_param.dslash_type = dslash_type;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  int tmpint = MAX(X[1]*X[2]*X[3], X[0]*X[2]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[3]);
  tmpint = MAX(tmpint, X[0]*X[1]*X[2]);


  gaugeParam.ga_pad = tmpint;
  inv_param.sp_pad = tmpint;

  ColorSpinorParam csParam;
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

    if (fatlink[dir] == NULL || longlink[dir] == NULL){
      errorQuda("ERROR: malloc failed for fatlink/longlink");
    }  
  }

  construct_fat_long_gauge_field(fatlink, longlink, 1, gaugeParam.cpu_prec, &gaugeParam, dslash_type);

#ifdef MULTI_GPU
  gaugeParam.type = QUDA_ASQTAD_FAT_LINKS;
  gaugeParam.reconstruct = QUDA_RECONSTRUCT_NO;
  GaugeFieldParam cpuFatParam(fatlink, gaugeParam);
  cpuFat = new cpuGaugeField(cpuFatParam);
  ghost_fatlink = cpuFat->Ghost();

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;
  GaugeFieldParam cpuLongParam(longlink, gaugeParam);
  cpuLong = new cpuGaugeField(cpuLongParam);
  ghost_longlink = cpuLong->Ghost();

  int x_face_size = X[1]*X[2]*X[3]/2;
  int y_face_size = X[0]*X[2]*X[3]/2;
  int z_face_size = X[0]*X[1]*X[3]/2;
  int t_face_size = X[0]*X[1]*X[2]/2;
  int pad_size = MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gaugeParam.ga_pad = pad_size;    
#endif

  gaugeParam.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
  if (dslash_type == QUDA_STAGGERED_DSLASH) {
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = link_recon;
  } else {
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  }
  
  printfQuda("Fat links sending..."); 
  loadGaugeQuda(fatlink, &gaugeParam);
  printfQuda("Fat links sent\n"); 

  gaugeParam.type = QUDA_ASQTAD_LONG_LINKS;  

#ifdef MULTI_GPU
  gaugeParam.ga_pad = 3*pad_size;
#endif

  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    gaugeParam.reconstruct = gaugeParam.reconstruct_sloppy = link_recon;
    printfQuda("Long links sending..."); 
    loadGaugeQuda(longlink, &gaugeParam);
    printfQuda("Long links sent...\n");
  }

  printfQuda("Sending fields to GPU..."); 

  if (!transfer) {

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

    cudaDeviceSynchronize();
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

    diracParam.tmp1=tmp;

    dirac = Dirac::create(diracParam);

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

  if (!transfer){
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp;
  }

  delete spinor;
  delete spinorOut;
  delete spinorRef;

  if (cpuFat) delete cpuFat;
  if (cpuLong) delete cpuLong;

  endQuda();
}

double dslashCUDA(int niter) {

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  for (int i = 0; i < niter; i++) {
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
        errorQuda("Staggered operator acting on full-site not supported");
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

      staggered_dslash_mg4dir(spinorRef, fatlink, longlink, (void**)ghost_fatlink, (void**)ghost_longlink, 
          spinor, parity, dagger, inv_param.cpu_prec, gaugeParam.cpu_prec);
#else
      cpu_parity = 0; //EVEN
      staggered_dslash(spinorRef->V(), fatlink, longlink, spinor->V(), cpu_parity, dagger, 
          inv_param.cpu_prec, gaugeParam.cpu_prec);

#endif    


      break;
    case 1: 
#ifdef MULTI_GPU
      staggered_dslash_mg4dir(spinorRef, fatlink, longlink, (void**)ghost_fatlink, (void**)ghost_longlink, 
          spinor, parity, dagger, inv_param.cpu_prec, gaugeParam.cpu_prec);    

#else
      cpu_parity=1; //ODD
      staggered_dslash(spinorRef->V(), fatlink, longlink, spinor->V(), cpu_parity, dagger, 
          inv_param.cpu_prec, gaugeParam.cpu_prec);
#endif
      break;
    case 2:
      //mat(spinorRef->V(), fatlink, longlink, spinor->V(), kappa, dagger, 
      //inv_param.cpu_prec, gaugeParam.cpu_prec);
      break;
    default:
      errorQuda("Test type not defined");
  }

  printfQuda("done.\n");

}

TEST(dslash, verify) {
  double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
  double tol = (inv_param.cuda_prec == QUDA_DOUBLE_PRECISION ? 1e-12 :
		(inv_param.cuda_prec == QUDA_SINGLE_PRECISION ? 1e-3 : 1e-1));
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

static int dslashTest()
{
  // return code for google test
  int test_rc = 0;
  init();

  int attempts = 1;

  for (int i=0; i<attempts; i++) {

    if (tune) { // warm-up run
      printfQuda("Tuning...\n");
      setTuning(QUDA_TUNE_YES);
      dslashCUDA(1);
    }
    printfQuda("Executing %d kernel loops...", loops);	

    // reset flop counter
    dirac->Flops();

    double secs = dslashCUDA(loops);

    if (!transfer) *spinorOut = *cudaSpinorOut;

    printfQuda("\n%fms per loop\n", 1000*secs);
    staggeredDslashRef();

    unsigned long long flops = dirac->Flops();
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/secs);

    double norm2_cpu = norm2(*spinorRef);
    double norm2_cpu_cuda= norm2(*spinorOut);
    if (!transfer) {
      double norm2_cuda= norm2(*cudaSpinorOut);
      printfQuda("Results: CPU = %f, CUDA=%f, CPU-CUDA = %f\n", norm2_cpu, norm2_cuda, norm2_cpu_cuda);
    } else {
      printfQuda("Result: CPU = %f, CPU-QUDA = %f\n",  norm2_cpu, norm2_cpu_cuda);
    }

    if (verify_results) {
      test_rc = RUN_ALL_TESTS();
      if (test_rc != 0) warningQuda("Tests failed");
    }
  }
  end();

  return test_rc;
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
  printfQuda("                                                0: Even destination spinor\n");
  printfQuda("                                                1: Odd destination spinor\n");
  return ;
}

int main(int argc, char **argv) 
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  for (int i=1 ;i < argc; i++){

    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }    

    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  // return result of RUN_ALL_TESTS
  int test_rc = dslashTest();

  finalizeComms();

  return test_rc;
}

