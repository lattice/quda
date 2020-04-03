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
#include <test_params.h>
#include <dslash_util.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"
#include "dslash_test_helpers.h"

#include <qio_field.h>
// google test frame work
#include <gtest/gtest.h>

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

QudaDagType not_dagger;

dslash_test_type dtest_type = dslash_test_type::Dslash;
CLI::TransformPairs<dslash_test_type> dtest_type_map {{"Dslash", dslash_test_type::Dslash},
                                                      {"MatPC", dslash_test_type::MatPC},
                                                      {"Mat", dslash_test_type::Mat},
                                                      {"MatPCDagMatPC", dslash_test_type::MatPCDagMatPC},
                                                      {"MatDagMat", dslash_test_type::MatDagMat},
                                                      {"M5", dslash_test_type::M5},
                                                      {"M5inv", dslash_test_type::M5inv},
                                                      {"Dslash4pre", dslash_test_type::Dslash4pre}};

double getTolerance(QudaPrecision prec)
{
  switch (prec) {
  case QUDA_QUARTER_PRECISION: return 1e-1;
  case QUDA_HALF_PRECISION: return 1e-3;
  case QUDA_SINGLE_PRECISION: return 1e-4;
  case QUDA_DOUBLE_PRECISION: return 1e-11;
  case QUDA_INVALID_PRECISION: return 1.0;
  }
  return 1.0;
}

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
    dw_setDims(gauge_param.X, Lsdim);
  } else {
    setDims(gauge_param.X);
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
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;

  inv_param.kappa = 0.1;

  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.epsilon = epsilon;
    inv_param.twist_flavor = twist_flavor;
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
             dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ) {
    inv_param.m5 = -1.5;
    kappa5 = 0.5/(5 + inv_param.m5);
  } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
    inv_param.m5 = -1.5;
    kappa5 = 0.5/(5 + inv_param.m5);
    for(int k = 0; k < Lsdim; k++)
    {
      // b5[k], c[k] values are chosen for arbitrary values,
      // but the difference of them are same as 1.0
      inv_param.b_5[k] = 1.50; // + 0.5*k;
      inv_param.c_5[k] = 0.50; // - 0.5*k;
    }
  }

  inv_param.mu = mu;
  inv_param.mass = mass;
  inv_param.Ls = (inv_param.twist_flavor != QUDA_TWIST_NONDEG_DOUBLET) ? Ls : 2;

  inv_param.solve_type = (dtest_type == dslash_test_type::Mat || dtest_type == dslash_test_type::MatDagMat) ?
    QUDA_DIRECT_SOLVE :
    QUDA_DIRECT_PC_SOLVE;
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
    switch (dtest_type) {
    case dslash_test_type::Dslash:
    case dslash_test_type::M5:
    case dslash_test_type::M5inv:
    case dslash_test_type::MatPC: inv_param.solution_type = QUDA_MATPC_SOLUTION; break;
    case dslash_test_type::Mat: inv_param.solution_type = QUDA_MAT_SOLUTION; break;
    case dslash_test_type::MatPCDagMatPC: inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; break;
    case dslash_test_type::MatDagMat: inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION; break;
    default: errorQuda("Test type %d not defined QUDA_DOMAIN_WALL_4D_DSLASH\n", static_cast<int>(dtest_type));
    }
  } else if(dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
    switch (dtest_type) {
    case dslash_test_type::Dslash:
    case dslash_test_type::M5:
    case dslash_test_type::Dslash4pre:
    case dslash_test_type::M5inv:
    case dslash_test_type::MatPC: inv_param.solution_type = QUDA_MATPC_SOLUTION; break;
    case dslash_test_type::Mat: inv_param.solution_type = QUDA_MAT_SOLUTION; break;
    case dslash_test_type::MatPCDagMatPC: inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; break;
    case dslash_test_type::MatDagMat: inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION; break;
    default: errorQuda("Test type %d not defined on QUDA_MOBIUS_DWF_DSLASH\n", static_cast<int>(dtest_type));
    }
  }
  else
  {
    switch (dtest_type) {
    case dslash_test_type::Dslash:
    case dslash_test_type::MatPC: inv_param.solution_type = QUDA_MATPC_SOLUTION; break;
    case dslash_test_type::Mat: inv_param.solution_type = QUDA_MAT_SOLUTION; break;
    case dslash_test_type::MatPCDagMatPC: inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; break;
    case dslash_test_type::MatDagMat: inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION; break;
    default: errorQuda("Test type %d not defined\n", static_cast<int>(dtest_type));
    }
  }

  inv_param.dslash_type = dslash_type;

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH
      || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    inv_param.clover_cpu_prec = cpu_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = clover_coeff;
    hostClover = malloc((size_t)V*cloverSiteSize*inv_param.clover_cpu_prec);
    hostCloverInv = malloc((size_t)V*cloverSiteSize*inv_param.clover_cpu_prec);
  }

  // construct input fields
  for (int dir = 0; dir < 4; dir++) hostGauge[dir] = malloc((size_t)V*gaugeSiteSize*gauge_param.cpu_prec);

  ColorSpinorParam csParam;
  
  csParam.nColor = 3;
  csParam.nSpin = 4;
  csParam.nDim = 4;
  for (int d=0; d<4; d++) csParam.x[d] = gauge_param.X[d];
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ||
      dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH ||
      dslash_type == QUDA_MOBIUS_DWF_DSLASH ) {
    csParam.nDim = 5;
    csParam.x[4] = Ls;
  }
  if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
    csParam.pc_type = QUDA_5D_PC;
  } else {
    csParam.pc_type = QUDA_4D_PC;
  }

//ndeg_tm    
  if (dslash_type == QUDA_TWISTED_MASS_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    csParam.twistFlavor = inv_param.twist_flavor;
    csParam.nDim = (inv_param.twist_flavor == QUDA_TWIST_SINGLET) ? 4 : 5;
    csParam.x[4] = inv_param.Ls;    
  }

  csParam.setPrecision(inv_param.cpu_prec);
  csParam.pad = 0;

  if (inv_param.solution_type == QUDA_MAT_SOLUTION || inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION) {
    csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  } else {
    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    csParam.x[0] /= 2;
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
  } else { // else generate an SU(3) field
    if (unit_gauge) {
      // unit SU(3) field
      construct_gauge_field(hostGauge, 0, gauge_param.cpu_prec, &gauge_param);
    } else {
      // random SU(3) field
      construct_gauge_field(hostGauge, 1, gauge_param.cpu_prec, &gauge_param);
    }
  }

  spinor->Source(QUDA_RANDOM_SOURCE, 0);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
      || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
    double norm = 0.1; // clover components are random numbers in the range (-norm, norm)
    double diag = 1.0; // constant added to the diagonal
    construct_clover_field(hostClover, norm, diag, inv_param.clover_cpu_prec);
    memcpy(hostCloverInv, hostClover, (size_t)V*cloverSiteSize*inv_param.clover_cpu_prec);
  }

  printfQuda("done.\n"); fflush(stdout);
  
  initQuda(device);

  // set verbosity prior to loadGaugeQuda
  setVerbosity(verbosity);
  inv_param.verbosity = verbosity;

  printfQuda("Sending gauge field to GPU\n");
  loadGaugeQuda(hostGauge, &gauge_param);

  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
      || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
    if (compute_clover) printfQuda("Computing clover field on GPU\n");
    else printfQuda("Sending clover field to GPU\n");
    inv_param.compute_clover = compute_clover;
    inv_param.return_clover = compute_clover;
    inv_param.compute_clover_inverse = compute_clover;
    inv_param.return_clover_inverse = compute_clover;
    inv_param.return_clover_inverse = true;

    loadCloverQuda(hostClover, hostCloverInv, &inv_param);
  }

  if (!transfer) {
    csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    csParam.pad = inv_param.sp_pad;
    csParam.setPrecision(inv_param.cuda_prec);
    if (csParam.Precision() == QUDA_DOUBLE_PRECISION ) {
      csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    } else {
      /* Single and half */
      csParam.fieldOrder = QUDA_FLOAT4_FIELD_ORDER;
    }

    if (inv_param.solution_type == QUDA_MAT_SOLUTION || inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION) {
      csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    } else {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    }

    printfQuda("Creating cudaSpinor with nParity = %d\n", csParam.siteSubset);
    cudaSpinor = new cudaColorSpinorField(csParam);
    printfQuda("Creating cudaSpinorOut with nParity = %d\n", csParam.siteSubset);
    cudaSpinorOut = new cudaColorSpinorField(csParam);

    tmp1 = new cudaColorSpinorField(csParam);

    if (inv_param.solution_type == QUDA_MAT_SOLUTION || inv_param.solution_type == QUDA_MATDAG_MAT_SOLUTION) {
      csParam.x[0] /= 2;
    }

    csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
    tmp2 = new cudaColorSpinorField(csParam);

    printfQuda("Sending spinor field to GPU\n");
    *cudaSpinor = *spinor;
    
    double cpu_norm = blas::norm2(*spinor);
    double cuda_norm = blas::norm2(*cudaSpinor);
    printfQuda("Source: CPU = %e, CUDA = %e\n", cpu_norm, cuda_norm);

    bool pc = (dtest_type != dslash_test_type::Mat && dtest_type != dslash_test_type::MatDagMat);

    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.tmp1 = tmp1;
    diracParam.tmp2 = tmp2;
    dirac = Dirac::create(diracParam);

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
  if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
      || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
    free(hostClover);
    free(hostCloverInv);
  }
  endQuda();

}

struct DslashTime {
  double event_time;
  double cpu_time;
  double cpu_min;
  double cpu_max;

  DslashTime() : event_time(0.0), cpu_time(0.0), cpu_min(DBL_MAX), cpu_max(0.0) {}
};

// execute kernel
DslashTime dslashCUDA(int niter) {

  DslashTime dslash_time;
  timeval tstart, tstop;

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  comm_barrier();
  cudaEventRecord(start, 0);

  for (int i = 0; i < niter; i++) {

    gettimeofday(&tstart, NULL);

    if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH){
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        if (transfer) {
          dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
        } else {
          static_cast<DiracDomainWall4DPC *>(dirac)->Dslash4(*cudaSpinorOut, *cudaSpinor, parity);
        }
        break;
      case dslash_test_type::M5:
        if (transfer) {
          dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
        } else {
          static_cast<DiracDomainWall4DPC *>(dirac)->Dslash5(*cudaSpinorOut, *cudaSpinor, parity);
        }
        break;
      case dslash_test_type::M5inv:
        if (transfer) {
          dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
        } else {
          static_cast<DiracDomainWall4DPC *>(dirac)->Dslash5inv(*cudaSpinorOut, *cudaSpinor, parity, kappa5);
        }
        break;
      case dslash_test_type::MatPC:
      case dslash_test_type::Mat:
        if (transfer) {
          MatQuda(spinorOut->V(), spinor->V(), &inv_param);
        } else {
          dirac->M(*cudaSpinorOut, *cudaSpinor);
        }
        break;
      case dslash_test_type::MatPCDagMatPC:
      case dslash_test_type::MatDagMat:
        if (transfer) {
          MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
        } else {
          dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
        }
        break;
      default: errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
      }
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        if (transfer) {
          dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
        } else {
          static_cast<DiracMobiusPC *>(dirac)->Dslash4(*cudaSpinorOut, *cudaSpinor, parity);
        }
        break;
      case dslash_test_type::M5:
        if (transfer) {
          dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
        } else {
          static_cast<DiracMobiusPC *>(dirac)->Dslash5(*cudaSpinorOut, *cudaSpinor, parity);
        }
        break;
      case dslash_test_type::Dslash4pre:
        if (transfer) {
          dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
        } else {
          static_cast<DiracMobiusPC *>(dirac)->Dslash4pre(*cudaSpinorOut, *cudaSpinor, parity);
        }
        break;
      case dslash_test_type::M5inv:
        if (transfer) {
          dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
        } else {
          static_cast<DiracMobiusPC *>(dirac)->Dslash5inv(*cudaSpinorOut, *cudaSpinor, parity);
        }
        break;
      case dslash_test_type::MatPC:
      case dslash_test_type::Mat:
        if (transfer) {
          MatQuda(spinorOut->V(), spinor->V(), &inv_param);
        } else {
          dirac->M(*cudaSpinorOut, *cudaSpinor);
        }
        break;
      case dslash_test_type::MatPCDagMatPC:
      case dslash_test_type::MatDagMat:
        if (transfer) {
          MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
        } else {
          dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
        }
        break;
      default: errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
      }
    } else {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
          if (transfer) {
            dslashQuda(spinorOut->V(), spinor->V(), &inv_param, parity);
          } else {
            dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
          }
        } else {
          if (transfer) {
            dslashQuda(spinorOut->V(), spinor->V(), &inv_param, parity);
          } else {
            dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity);
          }
        }
        break;
      case dslash_test_type::MatPC:
      case dslash_test_type::Mat:
        if (transfer) {
          MatQuda(spinorOut->V(), spinor->V(), &inv_param);
        } else {
          dirac->M(*cudaSpinorOut, *cudaSpinor);
        }
        break;
      case dslash_test_type::MatPCDagMatPC:
      case dslash_test_type::MatDagMat:
        if (transfer) {
          MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
        } else {
          dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
        }
        break;
      default: errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
      }
    }

    gettimeofday(&tstop, NULL);
    long ds = tstop.tv_sec - tstart.tv_sec;
    long dus = tstop.tv_usec - tstart.tv_usec;
    double elapsed = ds + 0.000001*dus;

    dslash_time.cpu_time += elapsed;
    // skip first and last iterations since they may skew these metrics if comms are not synchronous
    if (i>0 && i<niter) {
      if (elapsed < dslash_time.cpu_min) dslash_time.cpu_min = elapsed;
      if (elapsed > dslash_time.cpu_max) dslash_time.cpu_max = elapsed;
    }
  }
    
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);
  float runTime;
  cudaEventElapsedTime(&runTime, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  dslash_time.event_time = runTime / 1000;

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    printfQuda("with ERROR: %s\n", cudaGetErrorString(stat));

  return dslash_time;
}

void dslashRef() {

  // compare to dslash reference implementation
  printfQuda("Calculating reference implementation...");
  fflush(stdout);

  if (dslash_type == QUDA_WILSON_DSLASH) {
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      wil_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::MatPC:
      wil_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.matpc_type, dagger, 
		inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::Mat:
      wil_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::MatPCDagMatPC:
      wil_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.matpc_type, dagger, 
		inv_param.cpu_prec, gauge_param);
      wil_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.matpc_type, not_dagger, 
		inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::MatDagMat:
      wil_mat(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec, gauge_param);
      wil_mat(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, not_dagger, inv_param.cpu_prec, gauge_param);
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      clover_dslash(spinorRef->V(), hostGauge, hostCloverInv, spinor->V(), parity, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::MatPC:
      clover_matpc(spinorRef->V(), hostGauge, hostClover, hostCloverInv, spinor->V(), inv_param.kappa, inv_param.matpc_type,
		   dagger, inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::Mat:
      clover_mat(spinorRef->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::MatPCDagMatPC:
      clover_matpc(spinorTmp->V(), hostGauge, hostClover, hostCloverInv, spinor->V(), inv_param.kappa, inv_param.matpc_type,
		   dagger, inv_param.cpu_prec, gauge_param);
      clover_matpc(spinorRef->V(), hostGauge, hostClover, hostCloverInv, spinorTmp->V(), inv_param.kappa, inv_param.matpc_type,
		   not_dagger, inv_param.cpu_prec, gauge_param);
      break;
    case dslash_test_type::MatDagMat:
      clover_mat(spinorTmp->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec, gauge_param);
      clover_mat(spinorRef->V(), hostGauge, hostClover, spinorTmp->V(), inv_param.kappa, not_dagger,
		 inv_param.cpu_prec, gauge_param);
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else if (dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
    printfQuda("HASENBUCH_TWIST Test: kappa=%lf mu=%lf\n", inv_param.kappa, inv_param.mu);
    fflush(stdout);
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      // My dslash should be the same as the clover dslash
      clover_dslash(spinorRef->V(), hostGauge, hostCloverInv, spinor->V(), parity, dagger, inv_param.cpu_prec,
                    gauge_param);
      break;
    case dslash_test_type::MatPC:
      // my matpc op
      cloverHasenbuschTwist_matpc(spinorRef->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa,
                                  inv_param.mu, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);

      break;
    case dslash_test_type::Mat:
      // my mat
      cloverHasenbuchTwist_mat(spinorRef->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, inv_param.mu,
                               dagger, inv_param.cpu_prec, gauge_param, inv_param.matpc_type);
      break;
    case dslash_test_type::MatPCDagMatPC:
      // matpc^\dagger matpc
      // my matpc op
      cloverHasenbuschTwist_matpc(spinorTmp->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa,
                                  inv_param.mu, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);

      cloverHasenbuschTwist_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), hostClover, hostCloverInv, inv_param.kappa,
                                  inv_param.mu, inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);

      break;
    case dslash_test_type::MatDagMat:
      // my mat
      cloverHasenbuchTwist_mat(spinorTmp->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, inv_param.mu,
                               dagger, inv_param.cpu_prec, gauge_param, inv_param.matpc_type);
      cloverHasenbuchTwist_mat(spinorRef->V(), hostGauge, hostClover, spinorTmp->V(), inv_param.kappa, inv_param.mu,
                               not_dagger, inv_param.cpu_prec, gauge_param, inv_param.matpc_type);

      break;
    default: printfQuda("Test type not defined\n"); exit(-1);
    }
  } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET)
        tm_dslash(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, parity,
            inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = (char*)ref1 + tm_offset*cpu_prec;
    
	void *flv1 = spinor->V();
	void *flv2 = (char*)flv1 + tm_offset*cpu_prec;
    
	tm_ndeg_dslash(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, 
	               parity, dagger, inv_param.matpc_type, inv_param.cpu_prec, gauge_param);	
      }
      break;
    case dslash_test_type::MatPC:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET)      
	tm_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = (char*)ref1 + tm_offset*cpu_prec;
    
	void *flv1 = spinor->V();
	void *flv2 = (char*)flv1 + tm_offset*cpu_prec;
    
	tm_ndeg_matpc(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);	
      }	
      break;
    case dslash_test_type::Mat:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET)      
	tm_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, dagger, inv_param.cpu_prec, gauge_param);
      else
      {
        int tm_offset = 12*spinorRef->Volume();

	void *evenOut = spinorRef->V();
	void *oddOut  = (char*)evenOut + tm_offset*cpu_prec;
    
	void *evenIn = spinor->V();
	void *oddIn  = (char*)evenIn + tm_offset*cpu_prec;
    
	tm_ndeg_mat(evenOut, oddOut, hostGauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, dagger, inv_param.cpu_prec, gauge_param);	
      }
      break;
    case dslash_test_type::MatPCDagMatPC:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET) { 
	tm_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
	tm_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
      }
      else
      {
	int tm_offset = 12*spinorRef->Volume();

	void *ref1 = spinorRef->V();
	void *ref2 = (char*)ref1 + tm_offset*cpu_prec;

	void *flv1 = spinor->V();
	void *flv2 = (char*)flv1 + tm_offset*cpu_prec;

	void *tmp1 = spinorTmp->V();
	void *tmp2 = (char*)tmp1 + tm_offset*cpu_prec;

	tm_ndeg_matpc(tmp1, tmp2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
	tm_ndeg_matpc(ref1, ref2, hostGauge, tmp1, tmp2, inv_param.kappa, inv_param.mu, inv_param.epsilon, inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
      }
      break;
    case dslash_test_type::MatDagMat:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
	tm_mat(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     dagger, inv_param.cpu_prec, gauge_param);
	tm_mat(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	     not_dagger, inv_param.cpu_prec, gauge_param);
      }
      else
      {
	int tm_offset = 12*spinorRef->Volume();

	void *evenOut = spinorRef->V();
	void *oddOut  = (char*)evenOut + tm_offset*cpu_prec;

	void *evenIn = spinor->V();
	void *oddIn  = (char*)evenIn + tm_offset*cpu_prec;

	void *evenTmp = spinorTmp->V();
	void *oddTmp = (char*)evenTmp + tm_offset*cpu_prec;

	tm_ndeg_mat(evenTmp, oddTmp, hostGauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon, dagger, inv_param.cpu_prec, gauge_param);
	tm_ndeg_mat(evenOut, oddOut, hostGauge, evenTmp, oddTmp, inv_param.kappa, inv_param.mu, inv_param.epsilon, not_dagger, inv_param.cpu_prec, gauge_param);
      }
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET)
	tmc_dslash(spinorRef->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, parity, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      else
        errorQuda("Not supported\n");
      break;
    case dslash_test_type::MatPC:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET)      
	tmc_matpc(spinorRef->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
      else
        errorQuda("Not supported\n");
      break;
    case dslash_test_type::Mat:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET)      
	tmc_mat(spinorRef->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, dagger, inv_param.cpu_prec, gauge_param);
      else
        errorQuda("Not supported\n");
      break;
    case dslash_test_type::MatPCDagMatPC:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
	tmc_matpc(spinorTmp->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
	tmc_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
	       inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
      } else
        errorQuda("Not supported\n");
      break;
    case dslash_test_type::MatDagMat:
      if(inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
	tmc_mat(spinorTmp->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, dagger, inv_param.cpu_prec, gauge_param);
	tmc_mat(spinorRef->V(), hostGauge, hostClover, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, not_dagger, inv_param.cpu_prec, gauge_param);
      } else
        errorQuda("Not supported\n");
      break;
    default:
      printfQuda("Test type not defined\n");
      exit(-1);
    }
  } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH ){
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      dw_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::MatPC:
      dw_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::Mat:
      dw_mat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::MatPCDagMatPC:
      dw_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      dw_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, inv_param.matpc_type, not_dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::MatDagMat:
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
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      dslash_4_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::M5:
      dw_dslash_5_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, true);
      break;
    case dslash_test_type::M5inv:
      dslash_5_inv(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, kappa_5);
      break;
    case dslash_test_type::MatPC:
      dw_4d_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::Mat:
      dw_4d_mat(
          spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::MatPCDagMatPC:
      dw_4d_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      dw_4d_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, inv_param.matpc_type, not_dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::MatDagMat:
      dw_4d_mat(
          spinorTmp->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      dw_4d_mat(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, not_dagger, gauge_param.cpu_prec, gauge_param,
          inv_param.mass);
      break;
    default:
      printf("Test type not supported for domain wall\n");
      exit(-1);
    }
    free(kappa_5);
  } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH){
    double _Complex *kappa_b = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
    double _Complex *kappa_c = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
    double _Complex *kappa_5 = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
    double _Complex *kappa_mdwf = (double _Complex *)malloc(Lsdim * sizeof(double _Complex));
    for(int xs = 0 ; xs < Lsdim ; xs++)
    {
      kappa_b[xs] = 1.0/(2*(inv_param.b_5[xs]*(4.0 + inv_param.m5) + 1.0));
      kappa_c[xs] = 1.0/(2*(inv_param.c_5[xs]*(4.0 + inv_param.m5) - 1.0));
      kappa_5[xs] = 0.5*kappa_b[xs]/kappa_c[xs];
      kappa_mdwf[xs] = -kappa_5[xs];
    }
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      dslash_4_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
      break;
    case dslash_test_type::M5:
      mdw_dslash_5(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, kappa_5, true);
      break;
    case dslash_test_type::Dslash4pre:
      mdw_dslash_4_pre(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5, true);
      break;
    case dslash_test_type::M5inv:
      mdw_dslash_5_inv(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
          inv_param.mass, kappa_mdwf);
      break;
    case dslash_test_type::MatPC:
      mdw_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa_b, kappa_c, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      break;
    case dslash_test_type::Mat:
      mdw_mat(spinorRef->V(), hostGauge, spinor->V(), kappa_b, kappa_c, dagger, gauge_param.cpu_prec, gauge_param,
          inv_param.mass, inv_param.b_5, inv_param.c_5);
      break;
    case dslash_test_type::MatPCDagMatPC:
      mdw_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa_b, kappa_c, inv_param.matpc_type, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      mdw_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa_b, kappa_c, inv_param.matpc_type, not_dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
      break;
    case dslash_test_type::MatDagMat:
      mdw_mat(spinorTmp->V(), hostGauge, spinor->V(), kappa_b, kappa_c, dagger, gauge_param.cpu_prec, gauge_param,
          inv_param.mass, inv_param.b_5, inv_param.c_5);
      mdw_mat(spinorRef->V(), hostGauge, spinorTmp->V(), kappa_b, kappa_c, not_dagger, gauge_param.cpu_prec,
          gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
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

  printfQuda("prec    recon   dtest_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension "
             "dslash_type    niter\n");
  printfQuda("%6s   %2s       %s           %12s    %d    %3d/%3d/%3d        %3d             %2d   %14s   %d\n",
             get_prec_str(prec), get_recon_str(link_recon), get_string(dtest_type_map, dtest_type).c_str(),
             get_matpc_str(matpc_type), dagger, xdim, ydim, zdim, tdim, Lsdim, get_dslash_str(dslash_type), niter);
  printfQuda("Grid partition info:     X  Y  Z  T\n"); 
  printfQuda("                         %d  %d  %d  %d\n", 
	     dimPartitioned(0),
	     dimPartitioned(1),
	     dimPartitioned(2),
	     dimPartitioned(3));

  return ;
    
}

TEST(dslash, verify) {
  double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
  double tol = getTolerance(inv_param.cuda_prec);
  if (gauge_param.reconstruct == QUDA_RECONSTRUCT_8) tol *= 10; // if recon 8, we tolerate a greater deviation

  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

int main(int argc, char **argv)
{
  // initalize google test, includes command line options
  ::testing::InitGoogleTest(&argc, argv);

  // return code for google test
  int test_rc = 0;
  // command line options
  auto app = make_app();
  app->add_option("--test", dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
  // add_eigen_option_group(app);
  // add_deflation_option_group(app);
  // add_multigrid_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  init(argc, argv);

  int attempts = 1;
  dslashRef();
  for (int i=0; i<attempts; i++) {

    {
      printfQuda("Tuning...\n");
      dslashCUDA(1); // warm-up run
    }
    printfQuda("Executing %d kernel loops...\n", niter);
    if (!transfer) dirac->Flops();
    DslashTime dslash_time = dslashCUDA(niter);
    printfQuda("done.\n\n");

    if (!transfer) *spinorOut = *cudaSpinorOut;

    // print timing information
    printfQuda("%fus per kernel call\n", 1e6*dslash_time.event_time / niter);
    //FIXME No flops count for twisted-clover yet
    unsigned long long flops = 0;
    if (!transfer) flops = dirac->Flops();
    printfQuda(
        "%llu flops per kernel call, %llu flops per site\n", flops / niter, (flops / niter) / cudaSpinor->Volume());
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/dslash_time.event_time);

    printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for aggregate message size %lu bytes\n",
	       1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.event_time, 1.0e-9*2*cudaSpinor->GhostBytes()*niter/dslash_time.cpu_time,
	       1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_max, 1.0e-9*2*cudaSpinor->GhostBytes()/dslash_time.cpu_min,
	       2*cudaSpinor->GhostBytes());

    double norm2_cpu = blas::norm2(*spinorRef);
    double norm2_cpu_cuda = blas::norm2(*spinorOut);
    if (!transfer) {
      double norm2_cuda= blas::norm2(*cudaSpinorOut);
      printfQuda("Results: CPU = %f, CUDA=%f, CPU-CUDA = %f\n", norm2_cpu, norm2_cuda, norm2_cpu_cuda);
    } else {
      printfQuda("Result: CPU = %f, CPU-QUDA = %f\n",  norm2_cpu, norm2_cpu_cuda);
    }

    if (verify_results) {
      ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
      if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

      test_rc = RUN_ALL_TESTS();
      if (test_rc != 0) warningQuda("Tests failed");
    }
  }    
  end();

  finalizeComms();
  return test_rc;
}
