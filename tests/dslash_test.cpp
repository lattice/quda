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

// google test frame work
#include <gtest.h>

#include <gauge_qio.h>

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
DiracMobiusDomainWallPC *dirac_mdwf = NULL; // create the MDWF Dirac operator
DiracDomainWall4DPC *dirac_4dpc = NULL; // create the 4d preconditioned DWF Dirac operator

// What test are we doing (0 = dslash, 1 = MatPC, 2 = Mat, 3 = MatPCDagMatPC, 4 = MatDagMat)
extern int test_type;

// Dirac operator type
extern QudaDslashType dslash_type;

// Twisted mass flavor type
extern QudaTwistFlavorType twist_flavor;
extern QudaMatPCType matpc_type;

extern bool tune;

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
             dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
             dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
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

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.mu = 0.01;
    inv_param.epsilon = 0.01; 
    inv_param.twist_flavor = twist_flavor;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
             dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ) {
    inv_param.mass = 0.01;
    inv_param.m5 = -1.5;
    kappa5 = 0.5/(5 + inv_param.m5);
  } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
    inv_param.mass = 0.01;
    inv_param.m5 = -1.5;
    kappa5 = 0.5/(5 + inv_param.m5);
    for(int k = 0; k < Lsdim; k++)
    {
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = 1.50;
      inv_param.c_5[k] = 0.50;
    }
  }

  inv_param.Ls = (inv_param.twist_flavor != QUDA_TWIST_NONDEG_DOUBLET) ? Ls : 2;
  
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
  } else if(dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    switch(test_type) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
        inv_param.solution_type = QUDA_MATPC_SOLUTION;
        break;
      case 5:
        inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
        break;
      default:
        errorQuda("Test type %d not defined on QUDA_MOBIUS_DWF_DSLASH\n", test_type);
    }
  }
  else
  {
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

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = inv_param.clover_cuda_prec;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = 1.5*inv_param.kappa;
    //if (test_type > 0) {
      hostClover = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      hostCloverInv = hostClover; // fake it
      /*} else {
      hostClover = NULL;
      hostCloverInv = malloc(V*cloverSiteSize*inv_param.clover_cpu_prec);
      }*/
  } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {

  }

  setVerbosity(QUDA_VERBOSE);

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc(V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.nColor = 3;
  csParam.nSpin = 4;
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    csParam.twistFlavor = inv_param.twist_flavor;
  }
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
    csParam.nDim = 5;
    csParam.x[4] = Ls;
  }
  if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
    csParam.PCtype = QUDA_4D_PC;
  } else {
    csParam.PCtype = QUDA_5D_PC;
  }

//ndeg_tm    
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    csParam.twistFlavor = inv_param.twist_flavor;
    csParam.nDim = (inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS) ? 4 : 5;
    csParam.x[4] = inv_param.Ls;    
  }


  csParam.precision = inv_param.cpu_prec;
  csParam.pad = 0;

  if(dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || 
      dslash_type == QUDA_MOBIUS_DWF_DSLASH)
  {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;

  } else
  {
    if (test_type < 2 || test_type ==3) {
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

  spinor->Source(QUDA_RANDOM_SOURCE);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    double norm = 0.0; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal

    if (test_type == 2 || test_type == 4) {
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
  }

  if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    printfQuda("Sending clover field to GPU\n");
    loadCloverQuda(NULL, NULL, &inv_param);
  }

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
 
    if(dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || 
        dslash_type == QUDA_MOBIUS_DWF_DSLASH)
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

    if(dslash_type != QUDA_DOMAIN_WALL_4D_DSLASH && 
        dslash_type != QUDA_MOBIUS_DWF_DSLASH)
      if (test_type == 2 || test_type == 4) csParam.x[0] /= 2;

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp2 = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;
    
    double cpu_norm = norm2(*spinor);
    double cuda_norm = norm2(*cudaSpinor);
    printfQuda("Source: CPU = %e, CUDA = %e\n", cpu_norm, cuda_norm);

    bool pc;
    if(dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH || 
       dslash_type == QUDA_MOBIUS_DWF_DSLASH)
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
    else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH){
      dirac_mdwf = new DiracMobiusDomainWallPC(diracParam);
      dirac = (Dirac*)dirac_mdwf;
    }
    else {
      dirac = Dirac::create(diracParam);
    }
  } else {
    double cpu_norm = norm2(*spinor);
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
  if((dslash_type == QUDA_CLOVER_WILSON_DSLASH) || (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)){
    if (hostClover != hostCloverInv && hostClover) free(hostClover);
    free(hostCloverInv);
  }
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
    }
    else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH){
      switch (test_type) {
        case 0:
          if (transfer) {
            dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, test_type);
          } else {
            dirac_mdwf->Dslash4(*cudaSpinorOut, *cudaSpinor, parity);
          }
          break;
        case 1:
          if (transfer) {
            dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, test_type);
          } else {
            dirac_mdwf->Dslash5(*cudaSpinorOut, *cudaSpinor, parity);
          }
          break;
        case 2:
          if (transfer) {
            dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, test_type);
          } else {
            dirac_mdwf->Dslash4pre(*cudaSpinorOut, *cudaSpinor, parity);
          }
          break;
        case 3:
          if (transfer) {
            dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, test_type);
          } else {
            dirac_mdwf->Dslash5inv(*cudaSpinorOut, *cudaSpinor, parity, kappa5);
          }
          break;
        case 4:
          if (transfer) {
            MatQuda(spinorOut->V(), spinor->V(), &inv_param);
          } else {
            dirac_mdwf->M(*cudaSpinorOut, *cudaSpinor);
          }
          break;
        case 5:
          if (transfer) {
            MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
          } else {
            dirac_mdwf->MdagM(*cudaSpinorOut, *cudaSpinor);
          }
          break;
      }
    } else {
      switch (test_type) {
        case 0:
          if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH && (matpc_type == QUDA_MATPC_EVEN_EVEN || matpc_type == QUDA_MATPC_ODD_ODD)) {
            if (transfer) {
              dslashQuda(spinorOut->V(), spinor->V(), &inv_param, parity);
            } else {
	      ((DiracTwistedCloverPC *) dirac)->TwistCloverInv(*tmp1, *cudaSpinor, (parity+1)%2);
              dirac->Dslash(*cudaSpinorOut, *tmp1, parity);
            }
          } else {
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
  } else if((dslash_type == QUDA_TWISTED_MASS_DSLASH) || (dslash_type == QUDA_TWISTED_CLOVER_DSLASH)){ 
    switch (test_type) {
    case 0:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS)
	tm_dslash(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, parity, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = cpu_prec == sizeof(double) ? (void*)((double*)ref1 + tm_offset): (void*)((float*)ref1 + tm_offset);
    
	void *flv1 = spinor->V();
	void *flv2 = cpu_prec == sizeof(double) ? (void*)((double*)flv1 + tm_offset): (void*)((float*)flv1 + tm_offset);
    
	tm_ndeg_dslash(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, 
	               parity, dagger, inv_param.matpc_type, inv_param.cpu_prec, gauge_param);	
      }
      break;
    case 1:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS)      
	tm_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = cpu_prec == sizeof(double) ? (void*)((double*)ref1 + tm_offset): (void*)((float*)ref1 + tm_offset);
    
	void *flv1 = spinor->V();
	void *flv2 = cpu_prec == sizeof(double) ? (void*)((double*)flv1 + tm_offset): (void*)((float*)flv1 + tm_offset);
    
	tm_ndeg_matpc(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);	
      }	
      break;
    case 2:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS)      
	tm_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *evenOut = spinorRef->V();
	void *oddOut  = cpu_prec == sizeof(double) ? (void*)((double*)evenOut + tm_offset): (void*)((float*)evenOut + tm_offset);
    
	void *evenIn = spinor->V();
	void *oddIn  = cpu_prec == sizeof(double) ? (void*)((double*)evenIn + tm_offset): (void*)((float*)evenIn + tm_offset);
    
	tm_ndeg_mat(evenOut, oddOut, hostGauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, dagger, inv_param.cpu_prec, gauge_param);	
      }
      break;
    case 3:    
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS){      
	tm_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
	tm_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
      }
      else
      {
	int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = cpu_prec == sizeof(double) ? (void*)((double*)ref1 + tm_offset): (void*)((float*)ref1 + tm_offset);

	void *flv1 = spinor->V();
	void *flv2 = cpu_prec == sizeof(double) ? (void*)((double*)flv1 + tm_offset): (void*)((float*)flv1 + tm_offset);

	void *tmp1 = spinorTmp->V();
	void *tmp2 = cpu_prec == sizeof(double) ? (void*)((double*)tmp1 + tm_offset): (void*)((float*)tmp1 + tm_offset);

	tm_ndeg_matpc(tmp1, tmp2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
	tm_ndeg_matpc(ref1, ref2, hostGauge, tmp1, tmp2, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
      }
      break;
    case 4:
      if(inv_param.twist_flavor == QUDA_TWIST_PLUS || inv_param.twist_flavor == QUDA_TWIST_MINUS){      
	tm_mat(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     dagger, inv_param.cpu_prec, gauge_param);
	tm_mat(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     not_dagger, inv_param.cpu_prec, gauge_param);
      }
      else
      {
	int tm_offset = 12*spinorRef->Volume();

	void *evenOut = spinorRef->V();
	void *oddOut  = cpu_prec == sizeof(double) ? (void*)((double*)evenOut + tm_offset): (void*)((float*)evenOut + tm_offset);

	void *evenIn = spinor->V();
	void *oddIn  = cpu_prec == sizeof(double) ? (void*)((double*)evenIn + tm_offset): (void*)((float*)evenIn + tm_offset);

	void *evenTmp = spinorTmp->V();
	void *oddTmp = cpu_prec == sizeof(double) ? (void*)((double*)evenTmp + tm_offset): (void*)((float*)evenTmp + tm_offset);

	tm_ndeg_mat(evenTmp, oddTmp, hostGauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, dagger, inv_param.cpu_prec, gauge_param);
	tm_ndeg_mat(evenOut, oddOut, hostGauge, evenTmp, oddTmp, inv_param.kappa, inv_param.mu, inv_param.epsilon, not_dagger, inv_param.cpu_prec, gauge_param);
      }
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
      dw_dslash_5_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
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
  } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH){
    double *kappa_b, *kappa_c, *kappa_5, *kappa_mdwf;
    kappa_b = (double*)malloc(Lsdim*sizeof(double));
    kappa_c = (double*)malloc(Lsdim*sizeof(double));
    kappa_5 = (double*)malloc(Lsdim*sizeof(double));
    kappa_mdwf = (double*)malloc(Lsdim*sizeof(double));
    for(int xs = 0 ; xs < Lsdim ; xs++)
    {
      kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
      kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
      kappa_5[xs] = 0.5*kappa_b[xs]/kappa_c[xs];
      kappa_mdwf[xs] = -kappa_5[xs];
    }
    switch (test_type) {
    case 0:
      dslash_4_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case 1:
      mdw_dslash_5(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, kappa_5);
      break;
    case 2:
      mdw_dslash_4_pre(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5 );
      break;
    case 3:
      dslash_5_inv(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, kappa_mdwf);
      break;
    case 4:    
      mdw_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa_b, kappa_c, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      break;
    case 5:    
      mdw_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa_b, kappa_c, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      mdw_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa_b, kappa_c, inv_param.matpc_type, not_dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      break;
    break; 
    default:
      printf("Test type not supported for domain wall\n");
      exit(-1);
    }
    free(kappa_b);
    free(kappa_c);
    free(kappa_5);
    free(kappa_mdwf);
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

  for (int i =1;i < argc; i++){    
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

    if (tune) { // warm-up run
      printfQuda("Tuning...\n");
      setTuning(QUDA_TUNE_YES);
      dslashCUDA(1);
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
    int spinor_floats = test_type ? 2*(7*24+24)+24 : 7*24+24;
    if (inv_param.cuda_prec == QUDA_HALF_PRECISION) 
      spinor_floats += test_type ? 2*(7*2 + 2) + 2 : 7*2 + 2; // relative size of norm is twice a short
    int gauge_floats = (test_type ? 2 : 1) * (gauge_param.gauge_fix ? 6 : 8) * gauge_param.reconstruct;
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      gauge_floats += test_type ? 72*2 : 72;
    }
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/secs);
    printfQuda("GB/s = %f\n\n", 
	       (double)Vh*(Ls*spinor_floats+gauge_floats)*inv_param.cuda_prec/((secs/niter)*1e+9));
    
    double norm2_cpu = norm2(*spinorRef);
    double norm2_cpu_cuda= norm2(*spinorOut);
    if (!transfer) {
      double norm2_cuda= norm2(*cudaSpinorOut);
      printfQuda("Results: CPU = %f, CUDA=%f, CPU-CUDA = %f\n", norm2_cpu, norm2_cuda, norm2_cpu_cuda);
    } else {
      printfQuda("Result: CPU = %f, CPU-QUDA = %f\n",  norm2_cpu, norm2_cpu_cuda);
    }
  
    if (verify_results) {
      ::testing::InitGoogleTest(&argc, argv);
      if (RUN_ALL_TESTS() != 0) warningQuda("Tests failed");
    }
  }    
  end();

  finalizeComms();
}
