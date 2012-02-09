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

#include <test_util.h>
#include <wilson_dslash_reference.h>
#include "misc.h"

#include <gauge_qio.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat)
const int test_type = 0;

// Dirac operator type
extern QudaDslashType dslash_type;
//const QudaDslashType dslash_type = QUDA_WILSON_DSLASH;
//const QudaDslashType dslash_type = QUDA_CLOVER_WILSON_DSLASH;
//const QudaDslashType dslash_type = QUDA_TWISTED_MASS_DSLASH;

const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
const int transfer = 0; // include transfer time in the benchmark?

const int loops = 100;

extern bool tune;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp1=0, *tmp2=0;

void *hostGauge[4], *hostClover, *hostCloverInv;

Dirac *dirac;

extern int device;
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

void init(int argc, char **argv) {

  //qudaSetNumaConfig("/usr/local/gpu_numa_config.txt");

  kernelPackT = false; // Set true for kernel T face packing
  cuda_prec= prec;

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;
  setDims(gauge_param.X);

  gauge_param.anisotropy = 1.0;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.kappa = 0.1;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    inv_param.mu = 0.01;
    inv_param.twist_flavor = QUDA_TWIST_MINUS;
  }

  inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
  inv_param.dagger = dagger;

  inv_param.cpu_prec = cpu_prec;
  if (inv_param.cpu_prec != gauge_param.cpu_prec) 
    errorQuda("Gauge and spinor cpu precisions must match");

  inv_param.cuda_prec = cuda_prec;

#ifndef MULTI_GPU // free parameter for single GPU
  gauge_param.ga_pad = 0;
#else // must be this one c/b face for multi gpu
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;    
#endif
  inv_param.sp_pad = 0;
  inv_param.cl_pad = 0;

  //inv_param.sp_pad = 24*24*24;
  //inv_param.cl_pad = 24*24*24;

  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // test code only supports DeGrand-Rossi Basis
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if (test_type == 2) {
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  } else {
    inv_param.solution_type = QUDA_MATPC_SOLUTION;
  }

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.clover_cuda_prec;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    //if (test_type > 0) {
      hostClover = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      hostCloverInv = hostClover; // fake it
      /*} else {
      hostClover = NULL;
      hostCloverInv = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      }*/
  } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {

  }

  //inv_param.verbosity = QUDA_DEBUG_VERBOSE;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.fieldLocation = QUDA_CPU_FIELD_LOCATION;
  csParam.nColor = 3;
  csParam.nSpin = 4;
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    csParam.twistFlavor = inv_param.twist_flavor;
  }
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
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

  //csParam.verbose = QUDA_DEBUG_VERBOSE;

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);

  csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, hostGauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(hostGauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

  spinor->Source(QUDA_RANDOM_SOURCE);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    if (test_type == 2) {
      construct_clover_field(hostClover, norm, diag, inv_param.clover_cpu_prec);
    } else {
      construct_clover_field(hostCloverInv, norm, diag, inv_param.clover_cpu_prec);
    }
  }
  printfQuda("done.\n"); fflush(stdout);
  
  initQuda(device);

  printfQuda("Sending gauge field to GPU\n");
  loadGaugeQuda(hostGauge, &gauge_param);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    printfQuda("Sending clover field to GPU\n");
    loadCloverQuda(hostClover, hostCloverInv, &inv_param);
    //clover = cudaCloverPrecise;
  }

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

    if (test_type == 2) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp1 = new cudaColorSpinorField(csParam);
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
	dslash_type == QUDA_TWISTED_MASS_DSLASH) {
      tmp2 = new cudaColorSpinorField(csParam);
    }

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;

    std::cout << "Source: CPU = " << norm2(*spinor) << ", CUDA = " << 
      norm2(*cudaSpinor) << std::endl;

    bool pc = (test_type != 2);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.verbose = QUDA_VERBOSE;
    diracParam.tmp1 = tmp1;
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
    delete tmp1;
    delete tmp2;
  }

  // release memory
  delete spinor;
  delete spinorOut;
  delete spinorRef;

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (test_type == 2) free(hostClover);
    else free(hostCloverInv);
  }
  endQuda();

}

// execute kernel
double dslashCUDA() {

  if (!transfer && tune) {
    if (test_type < 2) {
      dirac->Tune(*cudaSpinorOut, *cudaSpinor, *tmp1);
    } else {
      dirac->Tune(cudaSpinorOut->Even(), cudaSpinor->Even(), *tmp1);
    }
  }

  printfQuda("Executing %d kernel loops...\n", loops);
  fflush(stdout);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

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
    }
  }
    
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
    printfQuda("with ERROR: %s\n", cudaGetErrorString(stat));

  printfQuda("done.\n\n");

#ifdef DSLASH_PROFILING
  printDslashProfile();
#endif

  return secs;
}

void dslashRef() {

  // FIXME: remove once reference clover is finished
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    if (inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    } else if (inv_param.matpc_type == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      inv_param.matpc_type = QUDA_MATPC_ODD_ODD;
    }
  }

  // compare to dslash reference implementation
  printfQuda("Calculating reference implementation...");
  fflush(stdout);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH ||
      dslash_type == QUDA_WILSON_DSLASH) {
    switch (test_type) {
    case 0:
      wil_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case 1:    
      wil_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.matpc_type, dagger, 
		inv_param.cpu_prec, gauge_param);
      break;
    case 2:
      wil_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec, gauge_param);
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else { // twisted mass
    switch (test_type) {
    case 0:
      tm_dslash(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
		parity, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case 1:    
      tm_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case 2:
      tm_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     dagger, inv_param.cpu_prec, gauge_param);
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  }

  printfQuda("done.\n");
}


void display_test_info()
{
  printfQuda("running the following test:\n");
 
  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension   dslash_type\n");
  printfQuda("%s   %s       %d           %d       %d/%d/%d        %d            %s\n", 
	     get_prec_str(prec), get_recon_str(link_recon), 
	     test_type, dagger, xdim, ydim, zdim, tdim, 
	     get_dslash_type_str(dslash_type));
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

  init(argc, argv);

  float spinorGiB = (float)Vh*spinorSiteSize*inv_param.cuda_prec / (1 << 30);
  printfQuda("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printfQuda("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  
  int attempts = 1;
  dslashRef();
  for (int i=0; i<attempts; i++) {
    double secs = dslashCUDA();

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printfQuda("%fms per loop\n", 1000*secs);
    
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    int spinor_floats = test_type ? 2*(7*24+24)+24 : 7*24+24;
    if (inv_param.cuda_prec == QUDA_HALF_PRECISION) 
      spinor_floats += test_type ? 2*(7*2 + 2) + 2 : 7*2 + 2; // relative size of norm is twice a short
    int gauge_floats = (test_type ? 2 : 1) * (gauge_param.gauge_fix ? 6 : 8) * gauge_param.reconstruct;
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      gauge_floats += test_type ? 72*2 : 72;
    }
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printfQuda("GB/s = %f\n\n", 
	       Vh*(spinor_floats+gauge_floats)*inv_param.cuda_prec/((secs/loops)*1e+9));
    
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
