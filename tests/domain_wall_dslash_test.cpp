#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <test_util.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"
#include "mpi.h"

#define MAX(a,b) ((a)>(b)?(a):(b))

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
const int test_type = 0;

const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
///const QudaDagType dagger = QUDA_DAG_NO;     // apply Dslash or Dslash dagger?
const int transfer = 0; // include transfer time in the benchmark?

const int loops = 1;

const int Ls = 2;
double kappa5;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec = QUDA_DOUBLE_PRECISION;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

//FullGauge gauge;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp=0, *tmp2=0;

void *hostGauge[4];

Dirac *dirac;

//BEGIN NEW
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern bool kernelPackT;
extern QudaDagType dagger;

extern char latfile[];

//END NEW

void init() {
//BEGIN NEW  
  kernelPackT = true; // Set true for kernel T face packing  
//END NEW  

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;
  
  setDims(gauge_param.X, Ls);

  gauge_param.anisotropy = 1.0;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_12;
  gauge_param.reconstruct_sloppy = gauge_param.reconstruct;
  gauge_param.cuda_prec_sloppy = gauge_param.cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  gauge_param.type = QUDA_WILSON_LINKS;

  inv_param.inv_type = QUDA_CG_INVERTER;

  inv_param.mass = 0.01;
  inv_param.m5 = -1.5;
  kappa5 = 0.5/(5 + inv_param.m5);

  inv_param.Ls = Ls;
  
  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = dagger;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  
  if (inv_param.cpu_prec != gauge_param.cpu_prec) 
    errorQuda("Gauge and spinor cpu precisions must match");  
  
#ifndef MULTI_GPU // free parameter for single GPU
  gauge_param.ga_pad = 0;
#else // must be this one c/b face for multi gpu
  int x_face_size = Ls*gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = Ls*gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = Ls*gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = Ls*gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif    
  inv_param.sp_pad = 0;// xdim*ydim*zdim;
  inv_param.cl_pad = 0;

  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (test_type == 2 || test_type == 3) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  } else {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
  }

  inv_param.dslash_type = QUDA_DOMAIN_WALL_DSLASH;

  inv_param.verbosity = QUDA_VERBOSE;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 5;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  csParam.x[4] = Ls;
  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;
  if (test_type < 2) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  }    
  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;
  
  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  spinor->Source(QUDA_RANDOM_SOURCE);

  printfQuda("done.\n"); fflush(stdout);
  
  int dev = 0;
  initQuda(dev);

  printfQuda("Sending gauge field to GPU\n");

  loadGaugeQuda(hostGauge, &gauge_param);
  //gauge = cudaGaugePrecise;

  if (!transfer) {
    csParam.fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    if (test_type < 2) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);
    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    if (test_type == 2 || test_type == 3) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;

    std::cout << "Source: CPU = " << norm2(*spinor) << ", CUDA = " << 
      norm2(*cudaSpinor) << std::endl;

    bool pc = (test_type != 2 || test_type != 3);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.verbose = QUDA_DEBUG_VERBOSE;
    diracParam.tmp1 = tmp;
    diracParam.tmp2 = tmp2;
    
    dirac = Dirac::create(diracParam);

  } else {
    std::cout << "Source: CPU = " << norm2(*spinor) << std::endl;
  }
    
}

void end() {
  if (!transfer) {
    delete dirac;
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp;
  }

  // release memory
  delete spinor;
  delete spinorOut;
  delete spinorRef;

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  endQuda();
}

// execute kernel
double dslashCUDA() {

  printfQuda("Executing %d kernel loops...\n", loops);
  fflush(stdout);

  //if (test_type < 2)
    //dirac->Tune(*cudaSpinorOut, *cudaSpinor, *tmp);
  //else
    //dirac->Tune(cudaSpinorOut->Even(), cudaSpinor->Even(), *tmp);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventRecord(start, 0);
  cudaEventSynchronize(start);

  for (int i = 0; i < loops; i++) {
    switch (test_type) {
    case 0:
      if (transfer) {
	dslashQuda(spinorOut->V(), spinor->V(), &inv_param, parity);
      } else {
	dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
      }
      break;
    case 1:
    case 2:
      if (transfer) {
	MatQuda(spinorOut->V(), spinor->V(), &inv_param);
      } else {
	dirac->M(*cudaSpinorOut, *cudaSpinor);
      }
      break;
    case 3:
      if (transfer) {
	MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
      } else {
	dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
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
    printf("with ERROR: %s\n", cudaGetErrorString(stat));

  printf("done.\n\n");

  return secs;
}

void dslashRef() {

  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // FIXME: remove once reference clover is finished
  if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  } else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
    inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
  }

  // compare to dslash reference implementation
  printf("Calculating reference implementation...");
  fflush(stdout);
  switch (test_type) {
  case 0:
    dw_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, mpi_size);	   
    break;
  case 1:    
    matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, 
	  inv_param.cpu_prec, gauge_param.cpu_prec, inv_param.mass);
    break;
  case 2:
    dw_mat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, mpi_size);
    break;
  case 3:
    dw_matdagmat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, mpi_size);
    break; 
  default:
    printf("Test type not defined\n");
    exit(-1);
  }

  printf("done.\n");
    
}

void display_test_info()
{
  printfQuda("running the following test:\n");
 
  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %d           %d       %d/%d/%d        %d\n", 
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

extern void usage(char**);

int main(int argc, char **argv)
{
  
  for (int i =1;i < argc; i++){    
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }  
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }


  initCommsQuda(argc, argv, gridsize_from_cmdline, 4);

  display_test_info();
  
  init();

  float spinorGiB = (float)Vh*spinorSiteSize*sizeof(inv_param.cpu_prec) / (1 << 30);
  printf("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printf("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  
  int attempts = 1;
  dslashRef();

  for (int i=0; i<attempts; i++) {
    
    double secs = dslashCUDA();

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printf("%fms per loop\n", 1000*secs);
    
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    int floats = test_type ? 2*(7*24+8*gauge_param.reconstruct+24)+24 : 7*24+8*gauge_param.reconstruct+24;
    printf("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printf("GiB/s = %f\n\n", Vh*floats*sizeof(float)/((secs/loops)*(1<<30)));
 
    if (!transfer) {
      double norm2_cpu = norm2(*spinorRef);
      double norm2_cuda= norm2(*cudaSpinorOut);
      double norm2_cpu_cuda= norm2(*spinorOut);
      printfQuda("Results: CPU = %f, CUDA=%f, CPU-CUDA = %f\n", norm2_cpu, norm2_cuda, norm2_cpu_cuda);
    } else {
      double norm2_cpu = norm2(*spinorRef);
      double norm2_cpu_cuda= norm2(*spinorOut);
      printfQuda("Result: CPU = %f, CPU-QUDA = %f\n",  norm2_cpu, norm2_cpu_cuda);
    }
   
    cpuColorSpinorField::Compare(*spinorRef, *spinorOut);
  }
    
  end();
  endCommsQuda();
}
