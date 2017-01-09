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
#include <dslash_util.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"

#include <qio_field.h>
// google test frame work
#include <gtest.h>

#define MAX(a,b) ((a)>(b)?(a):(b))

using namespace quda;

const QudaParity parity = QUDA_EVEN_PARITY; // even or odd?
const int transfer = 0; // include transfer time in the benchmark?

double kappa5;

QudaPrecision cpu_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision cuda_prec;

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef, *spinorTmp;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut, *tmp1=0, *tmp2=0;

void *hostGauge[4], *hostClover, *hostCloverInv;

Dirac *dirac = NULL;
DiracDomainWall4DPC *dirac_4dpc = NULL; // create the 4d preconditioned DWF Dirac operator

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat, 3 = MatPCDagMatPC, 4 = MatDagMat)
extern int test_type;

// Dirac operator type
extern QudaDslashType dslash_type;

// Twisted mass flavor type
extern QudaMatPCType matpc_type;

extern int device;
extern int xdim;
extern int ydim;
extern int zdim;
extern int tdim;
extern int Lsdim;
extern int gridsize_from_cmdline[];
extern QudaReconstructType link_recon;
extern QudaPrecision prec;
extern QudaDagType dagger;
QudaDagType not_dagger;


extern bool verify_results;
extern int niter;
extern char latfile[];

extern bool kernel_pack_t;

QudaVerbosity verbosity = QUDA_VERBOSE;

void init(int argc, char **argv) {

  cuda_prec = prec;

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  gauge_param.X[0] = xdim;
  gauge_param.X[1] = ydim;
  gauge_param.X[2] = zdim;
  gauge_param.X[3] = tdim;

  if (dslash_type == QUDA_ASQTAD_DSLASH || dslash_type == QUDA_STAGGERED_DSLASH) {
    errorQuda("Asqtad not supported.  Please try staggered_dslash_test instead");
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
             dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ) {
    // for these we always use kernel packing
    dw_setDims(gauge_param.X, Lsdim);
    setKernelPackT(true);
  } else {
    setDims(gauge_param.X);
    setKernelPackT(kernel_pack_t);
    Ls = 1;
  }

  setSpinorSiteSize(24);

  gauge_param.anisotropy = 1.0;

  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gauge_param.t_boundary = QUDA_ANTI_PERIODIC_T;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon;
  gauge_param.cuda_prec_sloppy = cuda_prec;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.kappa = 0.1;

  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
             dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ) {
    inv_param.mass = 0.01;
    inv_param.m5 = -1.5;
    kappa5 = 0.5/(5 + inv_param.m5);
  }
  inv_param.Ls = Ls;
  
  inv_param.solve_type = (test_type == 2 || test_type == 4) ? QUDA_DIRECT_SOLVE : QUDA_DIRECT_PC_SOLVE;
  inv_param.matpc_type = matpc_type;
  inv_param.dagger = dagger;
  not_dagger = (QudaDagType)((dagger + 1)%2);

  inv_param.cpu_prec = cpu_prec;
  if (inv_param.cpu_prec != gauge_param.cpu_prec) {
    errorQuda("Gauge and spinor CPU precisions must match");
  }
  inv_param.cuda_prec = cuda_prec;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

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

  //inv_param.sp_pad = xdim*ydim*zdim/2;
  //inv_param.cl_pad = 24*24*24;

  inv_param.gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // test code only supports DeGrand-Rossi Basis
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  if(dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH){
    switch(test_type) {
      case 0:
      case 1:
      case 2:
      case 3:
        inv_param.solution_type = QUDA_MATPC_SOLUTION;
        break;
      case 4:
        inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
        break;
      default:
        errorQuda("Test type %d not defined QUDA_DOMAIN_WALL_4D_DSLASH\n", test_type);
    }
  } else {
    switch(test_type) {
      case 0:
      case 1:
        inv_param.solution_type = QUDA_MATPC_SOLUTION;
        break;
      case 2:
        inv_param.solution_type = QUDA_MAT_SOLUTION;
        break;
      case 3:
        inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
        break;
      case 4:
        inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION;
        break;
      default:
        errorQuda("Test type %d not defined\n", test_type);
    }
  }

  inv_param.dslash_type = dslash_type;

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ) {
    csParam.nDim = 5;
    csParam.x[4] = Ls;
  }
  if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ) {
    csParam.PCtype = QUDA_4D_PC;
  } else {
    csParam.PCtype = QUDA_5D_PC;
  }


  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;

  if(dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
  } else {
    if (test_type < 2 || test_type == 3) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    } else {
      csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    }
  }

  csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  csParam.gammaBasis = inv_param.gamma_basis;
  csParam.create = QUDA_ZERO_FIELD_CREATE;

  spinor = new cpuColorSpinorField(csParam);
  spinorOut = new cpuColorSpinorField(csParam);
  spinorRef = new cpuColorSpinorField(csParam);
  spinorTmp = new cpuColorSpinorField(csParam);

  csParam.x[0] = gauge_param.X[0];
  
  printfQuda("Randomizing fields... ");

  if (strcmp(latfile,"")) {  // load in the command line supplied gauge field
    read_gauge_field(latfile, hostGauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_gauge_field(hostGauge, 2, gauge_param.cpu_prec, &gauge_param);
  } else { // else generate a random SU(3) field
    construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
  }

  spinor->Source(QUDA_RANDOM_SOURCE, 0);

  printfQuda("done.\n"); fflush(stdout);
  
  initQuda(device);

  // set verbosity prior to loadGaugeQuda
  setVerbosity(verbosity);
  inv_param.verbosity = verbosity;

  printfQuda("Sending gauge field to GPU\n");
  loadGaugeQuda(hostGauge, &gauge_param);

  if (!transfer) {
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.precision = inv_param.cuda_prec;
    if (csParam.precision == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }
 
    if(dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH )
    {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    } else
    {
      if (test_type < 2 || test_type == 3) {
        csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
        csParam.x[0] /= 2;
      }
    }

    printfQuda("Creating cudaSpinor\n");
    cudaSpinor = new cudaColorSpinorField(csParam);
    printfQuda("Creating cudaSpinorOut\n");
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    tmp1 = new cudaColorSpinorField(csParam);

    if(dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH )
      if (test_type == 2 || test_type == 4) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp2 = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;
    
    double cpu_norm = blas::norm2(*spinor);
    double cuda_norm = blas::norm2(*cudaSpinor);
    printfQuda("Source: CPU = %e, CUDA = %e\n", cpu_norm, cuda_norm);

    bool pc;
    if(dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH )
      pc = true;
    else
      pc = (test_type != 2 && test_type != 4);
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.tmp1 = tmp1;
    diracParam.tmp2 = tmp2;
   
    if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH){
      dirac_4dpc = new DiracDomainWall4DPC(diracParam);
      dirac = (Dirac*)dirac_4dpc;
    }
    else {
      dirac = Dirac::create(diracParam);
    }
  } else {
    double cpu_norm = blas::norm2(*spinor);
    printfQuda("Source: CPU = %e\n", cpu_norm);
  }
    
}

void end() {
  if (!transfer) {
    if(dirac != NULL)
    {
      delete dirac;
      dirac = NULL;
    }
    delete cudaSpinor;
    delete cudaSpinorOut;
    delete tmp1;
    delete tmp2;
  }

  // release memory
  delete spinor;
  delete spinorOut;
  delete spinorRef;
  delete spinorTmp;

  for (int dir = 0; dir < 4; dir++) free(hostGauge[dir]);
  endQuda();

}

// execute kernel
double dslashCUDA(int niter) {

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, 0);

  for (int i = 0; i < niter; i++) {
    if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH){
      switch (test_type) {
        case 0:
          if (transfer) {
            dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, test_type);
          } else {
            dirac_4dpc->Dslash4(*cudaSpinorOut, *cudaSpinor, parity);
          }
          break;
        case 1:
          if (transfer) {
            dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, test_type);
          } else {
            dirac_4dpc->Dslash5(*cudaSpinorOut, *cudaSpinor, parity);
          }
          break;
        case 2:
          if (transfer) {
            dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, test_type);
          } else {
            dirac_4dpc->Dslash5inv(*cudaSpinorOut, *cudaSpinor, parity, kappa5);
          }
          break;
        case 3:
          if (transfer) {
            MatQuda(spinorOut->V(), spinor->V(), &inv_param);
          } else {
            dirac_4dpc->M(*cudaSpinorOut, *cudaSpinor);
          }
          break;
        case 4:
          if (transfer) {
            MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
          } else {
            dirac_4dpc->MdagM(*cudaSpinorOut, *cudaSpinor);
          }
          break;
      }
    } else {
      switch (test_type) {
        case 0:
          {
            if (transfer) {
              dslashQuda(spinorOut->V(), spinor->V(), &inv_param, parity);
            } else {
              dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
            }
          }
          break;
        case 1:
          if (transfer) {
            MatQuda(spinorOut->V(), spinor->V(), &inv_param);
          } else {
            dirac->M(*cudaSpinorOut, *cudaSpinor);
          }
          break;
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
          break;
        case 4:
          if (transfer) {
            MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
          } else {
            dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
          }
          break;
      }
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

  return secs;
}

void dslashRef() {

  // compare to dslash reference implementation
  printfQuda("Calculating reference implementation...");
  fflush(stdout);

  if (dslash_type == QUDA_WILSON_DSLASH) {
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
    case 3:
      wil_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.matpc_type, dagger, 
		inv_param.cpu_prec, gauge_param);
      wil_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.matpc_type, not_dagger, 
		inv_param.cpu_prec, gauge_param);
      break;
    case 4:
      wil_mat(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec, gauge_param);
      wil_mat(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, not_dagger, inv_param.cpu_prec, gauge_param);
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ){
    switch (test_type) {
    case 0:
      dw_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 1:    
      dw_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 2:
      dw_mat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 3:    
      dw_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      dw_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, inv_param.matpc_type, not_dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 4:
      dw_matdagmat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
    break; 
    default:
      printf("Test type not supported for domain wall\n");
      exit(-1);
    }
  } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH){
    double *kappa_5 = (double*)malloc(Ls*sizeof(double));
    for(int xs = 0; xs < Ls ; xs++)
      kappa_5[xs] = kappa5;
    switch (test_type) {
    case 0:
      dslash_4_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 1:    
      dw_dslash_5_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, true);
      break;
    case 2:    
      dslash_5_inv(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, kappa_5);
      break;
    case 3:
      dw_4d_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 4:    
      dw_4d_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      dw_4d_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, inv_param.matpc_type, not_dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    break; 
    default:
      printf("Test type not supported for domain wall\n");
      exit(-1);
    }
    free(kappa_5);
  } else {
    printfQuda("Unsupported dslash_type\n");
    exit(-1);
  }

  printfQuda("done.\n");
}


void display_test_info()
{
  printfQuda("running the following test:\n");
 
  printfQuda("prec    recon   test_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension dslash_type    niter\n");
  printfQuda("%6s   %2s       %d           %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n", 
	     get_prec_str(prec), get_recon_str(link_recon), 
	     test_type, get_matpc_str(matpc_type), dagger, xdim, ydim, zdim, tdim, Lsdim,
	     get_dslash_str(dslash_type), niter);
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3));

  return ;
    
}

extern void usage(char**);

TEST(dslash, verify) {
  double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
  double tol = (inv_param.cuda_prec == QUDA_DOUBLE_PRECISION ? 1e-12 :
		(inv_param.cuda_prec == QUDA_SINGLE_PRECISION ? 1e-3 : 1e-1));
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;
  for (int i =1;i < argc; i++) {
    if(process_command_line_option(argc, argv, &i) == 0){
      continue;
    }  
    
    fprintf(stderr, "ERROR: Invalid option:%s\n", argv[i]);
    usage(argv);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  init(argc, argv);

  float spinorGiB = (float)Vh*spinorSiteSize*inv_param.cuda_prec / (1 << 30);
  printfQuda("\nSpinor mem: %.3f GiB\n", spinorGiB);
  printfQuda("Gauge mem: %.3f GiB\n", gauge_param.gaugeGiB);
  
  int attempts = 1;
  dslashRef();
  for (int i=0; i<attempts; i++) {

    {
      printfQuda("Tuning...\n");
      dslashCUDA(1); // warm-up run
    }
    printfQuda("Executing %d kernel loops...\n", niter);
    if (!transfer) dirac->Flops();
    double secs = dslashCUDA(niter);
    printfQuda("done.\n\n");

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printfQuda("%fus per kernel call\n", 1e6*secs / niter);
    //FIXME No flops count for twisted-clover yet
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/secs);
    
    double norm2_cpu = blas::norm2(*spinorRef);
    double norm2_cpu_cuda= blas::norm2(*spinorOut);
    if (!transfer) {
      double norm2_cuda= blas::norm2(*cudaSpinorOut);
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

  finalizeComms();
  return test_rc;
}
