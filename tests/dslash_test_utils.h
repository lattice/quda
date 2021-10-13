#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <wilson_dslash_reference.h>
#include <domain_wall_dslash_reference.h>
#include "misc.h"
#include "dslash_test_helpers.h"

// google test frame work
#include <gtest/gtest.h>

#include <color_spinor_field.h>

using namespace quda;

CLI::TransformPairs<dslash_test_type> dtest_type_map {{"Dslash", dslash_test_type::Dslash},
                                                      {"MatPC", dslash_test_type::MatPC},
                                                      {"Mat", dslash_test_type::Mat},
                                                      {"MatPCDagMatPC", dslash_test_type::MatPCDagMatPC},
                                                      {"MatPCDagMatPCLocal", dslash_test_type::MatPCDagMatPCLocal},
                                                      {"MatDagMat", dslash_test_type::MatDagMat},
                                                      {"M5", dslash_test_type::M5},
                                                      {"M5inv", dslash_test_type::M5inv},
                                                      {"Dslash4pre", dslash_test_type::Dslash4pre}};
struct DslashTime {
  double event_time;
  double cpu_time;
  double cpu_min;
  double cpu_max;

  DslashTime() : event_time(0.0), cpu_time(0.0), cpu_min(DBL_MAX), cpu_max(0.0) { }
};

struct DslashTestWrapper {

  // CPU color spinor fields
  quda::cpuColorSpinorField *spinor = nullptr;
  quda::cpuColorSpinorField *spinorOut = nullptr;
  quda::cpuColorSpinorField *spinorRef = nullptr;
  quda::cpuColorSpinorField *spinorTmp = nullptr;
  // For split grid
  std::vector<quda::cpuColorSpinorField *> vp_spinor;
  std::vector<quda::cpuColorSpinorField *> vp_spinorOut;
  std::vector<quda::cpuColorSpinorField *> vp_spinorRef;

  // CUDA color spinor fields
  quda::cudaColorSpinorField *cudaSpinor = nullptr;
  quda::cudaColorSpinorField *cudaSpinorOut = nullptr;
  quda::cudaColorSpinorField *tmp1 = nullptr;
  quda::cudaColorSpinorField *tmp2 = nullptr;

  // Dirac pointers
  quda::Dirac *dirac = nullptr;
  quda::DiracMobiusPC *dirac_mdwf = nullptr;
  quda::DiracDomainWall4DPC *dirac_4dpc = nullptr;

  // Raw pointers
  void *hostGauge[4] = {nullptr};
  void *hostClover = nullptr;
  void *hostCloverInv = nullptr;

  // Parameters
  QudaGaugeParam gauge_param;
  QudaInvertParam inv_param;

  // Test options
  QudaDagType dagger;
  QudaDagType not_dagger;
  QudaParity parity;
  dslash_test_type dtest_type = dslash_test_type::Dslash;
  bool test_split_grid;
  int num_src;

  const bool transfer = false;

  void init_ctest(int argc, char **argv, int precision, QudaReconstructType link_recon)
  {
    cuda_prec = getPrecision(precision);

    gauge_param = newQudaGaugeParam();
    inv_param = newQudaInvertParam();
    setWilsonGaugeParam(gauge_param);
    setInvertParam(inv_param);

    gauge_param.cuda_prec = cuda_prec;
    gauge_param.cuda_prec_sloppy = cuda_prec;
    gauge_param.cuda_prec_precondition = cuda_prec;
    gauge_param.cuda_prec_refinement_sloppy = cuda_prec;

    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon;
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon;

    inv_param.cuda_prec = cuda_prec;

    init(argc, argv);
  }

  void init_test(int argc, char **argv)
  {
    gauge_param = newQudaGaugeParam();
    inv_param = newQudaInvertParam();
    setWilsonGaugeParam(gauge_param);
    setInvertParam(inv_param);

    init(argc, argv);
  }

  void init(int argc, char **argv)
  {
    num_src = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
    test_split_grid = num_src > 1;
    if (test_split_grid) { dtest_type = dslash_test_type::Dslash; }

    if (dslash_type == QUDA_ASQTAD_DSLASH || dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) {
      errorQuda("Asqtad not supported.  Please try staggered_dslash_test instead");
    } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
               || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
      // for these we always use kernel packing
      dw_setDims(gauge_param.X, Lsdim);
    } else {
      setDims(gauge_param.X);
      Ls = 1;
    }

    inv_param.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;
    not_dagger = dagger ? QUDA_DAG_NO : QUDA_DAG_YES;

    inv_param.solve_type = (dtest_type == dslash_test_type::Mat || dtest_type == dslash_test_type::MatDagMat) ?
      QUDA_DIRECT_SOLVE :
      QUDA_DIRECT_PC_SOLVE;

    if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
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
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
      case dslash_test_type::M5:
      case dslash_test_type::Dslash4pre:
      case dslash_test_type::M5inv:
      case dslash_test_type::MatPC: inv_param.solution_type = QUDA_MATPC_SOLUTION; break;
      case dslash_test_type::Mat: inv_param.solution_type = QUDA_MAT_SOLUTION; break;
      case dslash_test_type::MatPCDagMatPCLocal:
      case dslash_test_type::MatPCDagMatPC: inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; break;
      case dslash_test_type::MatDagMat: inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION; break;
      default: errorQuda("Test type %d not defined on QUDA_MOBIUS_DWF_(EOFA_)DSLASH\n", static_cast<int>(dtest_type));
      }
    } else {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
      case dslash_test_type::MatPC: inv_param.solution_type = QUDA_MATPC_SOLUTION; break;
      case dslash_test_type::Mat: inv_param.solution_type = QUDA_MAT_SOLUTION; break;
      case dslash_test_type::MatPCDagMatPC: inv_param.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION; break;
      case dslash_test_type::MatDagMat: inv_param.solution_type = QUDA_MATDAG_MAT_SOLUTION; break;
      default: errorQuda("Test type %d not defined\n", static_cast<int>(dtest_type));
      }
    }

    if (inv_param.cpu_prec != gauge_param.cpu_prec) errorQuda("Gauge and spinor CPU precisions must match");

    // construct input fields
    for (int dir = 0; dir < 4; dir++) hostGauge[dir] = safe_malloc((size_t)V * gauge_site_size * gauge_param.cpu_prec);

    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH
        || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      hostClover = safe_malloc((size_t)V * clover_site_size * inv_param.clover_cpu_prec);
      hostCloverInv = safe_malloc((size_t)V * clover_site_size * inv_param.clover_cpu_prec);
    }

    ColorSpinorParam csParam;
    csParam.nColor = 3;
    csParam.nSpin = 4;
    csParam.nDim = 4;
    for (int d = 0; d < 4; d++) csParam.x[d] = gauge_param.X[d];
    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH || dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH
        || dslash_type == QUDA_MOBIUS_DWF_DSLASH || dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
      csParam.nDim = 5;
      csParam.x[4] = Ls;
    }

    if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      csParam.pc_type = QUDA_5D_PC;
    } else {
      csParam.pc_type = QUDA_4D_PC;
    }

    // ndeg_tm
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

    spinor->Source(QUDA_RANDOM_SOURCE);

    inv_param.split_grid[0] = grid_partition[0];
    inv_param.split_grid[1] = grid_partition[1];
    inv_param.split_grid[2] = grid_partition[2];
    inv_param.split_grid[3] = grid_partition[3];

    if (test_split_grid) {
      inv_param.num_src = num_src;
      inv_param.num_src_per_sub_partition = 1;
      for (int n = 0; n < num_src; n++) {
        vp_spinor.push_back(new cpuColorSpinorField(csParam));
        vp_spinorOut.push_back(new cpuColorSpinorField(csParam));
        vp_spinorRef.push_back(new cpuColorSpinorField(csParam));
      }
    }

    if (test_split_grid) {
      for (int n = 0; n < num_src; n++) { *vp_spinor[n] = *spinor; }
    }

    csParam.x[0] = gauge_param.X[0];

    // set verbosity prior to loadGaugeQuda
    setVerbosity(verbosity);
    inv_param.verbosity = verbosity;

    printfQuda("Randomizing fields... ");
    constructHostGaugeField(hostGauge, gauge_param, argc, argv);

    printfQuda("Sending gauge field to GPU\n");
    loadGaugeQuda(hostGauge, &gauge_param);

    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
        || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      if (compute_clover)
        printfQuda("Computing clover field on GPU\n");
      else {
        printfQuda("Sending clover field to GPU\n");
        constructHostCloverField(hostClover, hostCloverInv, inv_param);
      }
      inv_param.compute_clover = compute_clover;
      inv_param.return_clover = compute_clover;
      inv_param.compute_clover_inverse = true;
      inv_param.return_clover_inverse = true;

      loadCloverQuda(hostClover, hostCloverInv, &inv_param);
    }

    if (!transfer) {
      csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      csParam.pad = inv_param.sp_pad;
      csParam.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);

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

      if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
        dirac_4dpc = new DiracDomainWall4DPC(diracParam);
        dirac = (Dirac *)dirac_4dpc;
      } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
        dirac_mdwf = new DiracMobiusPC(diracParam);
        dirac = (Dirac *)dirac_mdwf;
      } else {
        dirac = Dirac::create(diracParam);
      }

    } else {
      double cpu_norm = blas::norm2(*spinor);
      printfQuda("Source: CPU = %e\n", cpu_norm);
    }
  }

  void end()
  {
    if (!transfer) {
      if (dirac != nullptr) {
        delete dirac;
        dirac = nullptr;
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

    for (auto p : vp_spinor) { delete p; }
    for (auto p : vp_spinorOut) { delete p; }
    for (auto p : vp_spinorRef) { delete p; }

    vp_spinor.clear();
    vp_spinorOut.clear();
    vp_spinorRef.clear();

    for (int dir = 0; dir < 4; dir++) host_free(hostGauge[dir]);
    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
        || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      host_free(hostClover);
      host_free(hostCloverInv);
    }
  }

  void dslashRef()
  {

    // compare to dslash reference implementation
    printfQuda("Calculating reference implementation...");

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
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        clover_dslash(spinorRef->V(), hostGauge, hostCloverInv, spinor->V(), parity, dagger, inv_param.cpu_prec,
                      gauge_param);
        break;
      case dslash_test_type::MatPC:
        clover_matpc(spinorRef->V(), hostGauge, hostClover, hostCloverInv, spinor->V(), inv_param.kappa,
                     inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::Mat:
        clover_mat(spinorRef->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec,
                   gauge_param);
        break;
      case dslash_test_type::MatPCDagMatPC:
        clover_matpc(spinorTmp->V(), hostGauge, hostClover, hostCloverInv, spinor->V(), inv_param.kappa,
                     inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
        clover_matpc(spinorRef->V(), hostGauge, hostClover, hostCloverInv, spinorTmp->V(), inv_param.kappa,
                     inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatDagMat:
        clover_mat(spinorTmp->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, dagger, inv_param.cpu_prec,
                   gauge_param);
        clover_mat(spinorRef->V(), hostGauge, hostClover, spinorTmp->V(), inv_param.kappa, not_dagger,
                   inv_param.cpu_prec, gauge_param);
        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      printfQuda("HASENBUCH_TWIST Test: kappa=%lf mu=%lf\n", inv_param.kappa, inv_param.mu);
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
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tm_dslash(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                    parity, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
        else {
          int tm_offset = 12 * spinorRef->Volume();

          void *ref1 = spinorRef->V();
          void *ref2 = (char *)ref1 + tm_offset * cpu_prec;

          void *flv1 = spinor->V();
          void *flv2 = (char *)flv1 + tm_offset * cpu_prec;

          tm_ndeg_dslash(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon, parity,
                         dagger, inv_param.matpc_type, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::MatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tm_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
        else {
          int tm_offset = 12 * spinorRef->Volume();

          void *ref1 = spinorRef->V();
          void *ref2 = (char *)ref1 + tm_offset * cpu_prec;

          void *flv1 = spinor->V();
          void *flv2 = (char *)flv1 + tm_offset * cpu_prec;

          tm_ndeg_matpc(ref1, ref2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::Mat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tm_mat(spinorRef->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, dagger,
                 inv_param.cpu_prec, gauge_param);
        else {
          int tm_offset = 12 * spinorRef->Volume();

          void *evenOut = spinorRef->V();
          void *oddOut = (char *)evenOut + tm_offset * cpu_prec;

          void *evenIn = spinor->V();
          void *oddIn = (char *)evenIn + tm_offset * cpu_prec;

          tm_ndeg_mat(evenOut, oddOut, hostGauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                      dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::MatPCDagMatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
          tm_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
        } else {
          int tm_offset = 12 * spinorRef->Volume();

          void *ref1 = spinorRef->V();
          void *ref2 = (char *)ref1 + tm_offset * cpu_prec;

          void *flv1 = spinor->V();
          void *flv2 = (char *)flv1 + tm_offset * cpu_prec;

          void *tmp1 = spinorTmp->V();
          void *tmp2 = (char *)tmp1 + tm_offset * cpu_prec;

          tm_ndeg_matpc(tmp1, tmp2, hostGauge, flv1, flv2, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
          tm_ndeg_matpc(ref1, ref2, hostGauge, tmp1, tmp2, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::MatDagMat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_mat(spinorTmp->V(), hostGauge, spinor->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor, dagger,
                 inv_param.cpu_prec, gauge_param);
          tm_mat(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 not_dagger, inv_param.cpu_prec, gauge_param);
        } else {
          int tm_offset = 12 * spinorRef->Volume();

          void *evenOut = spinorRef->V();
          void *oddOut = (char *)evenOut + tm_offset * cpu_prec;

          void *evenIn = spinor->V();
          void *oddIn = (char *)evenIn + tm_offset * cpu_prec;

          void *evenTmp = spinorTmp->V();
          void *oddTmp = (char *)evenTmp + tm_offset * cpu_prec;

          tm_ndeg_mat(evenTmp, oddTmp, hostGauge, evenIn, oddIn, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                      dagger, inv_param.cpu_prec, gauge_param);
          tm_ndeg_mat(evenOut, oddOut, hostGauge, evenTmp, oddTmp, inv_param.kappa, inv_param.mu, inv_param.epsilon,
                      not_dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tmc_dslash(spinorRef->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu,
                     inv_param.twist_flavor, parity, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
        else
          errorQuda("Not supported\n");
        break;
      case dslash_test_type::MatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tmc_matpc(spinorRef->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
        else
          errorQuda("Not supported\n");
        break;
      case dslash_test_type::Mat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tmc_mat(spinorRef->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, dagger, inv_param.cpu_prec, gauge_param);
        else
          errorQuda("Not supported\n");
        break;
      case dslash_test_type::MatPCDagMatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_matpc(spinorTmp->V(), hostGauge, spinor->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, dagger, inv_param.cpu_prec, gauge_param);
          tmc_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
        } else
          errorQuda("Not supported\n");
        break;
      case dslash_test_type::MatDagMat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_mat(spinorTmp->V(), hostGauge, hostClover, spinor->V(), inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, dagger, inv_param.cpu_prec, gauge_param);
          tmc_mat(spinorRef->V(), hostGauge, hostClover, spinorTmp->V(), inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, not_dagger, inv_param.cpu_prec, gauge_param);
        } else
          errorQuda("Not supported\n");
        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        dw_dslash(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                  inv_param.mass);
        break;
      case dslash_test_type::MatPC:
        dw_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec,
                 gauge_param, inv_param.mass);
        break;
      case dslash_test_type::Mat:
        dw_mat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass);
        break;
      case dslash_test_type::MatPCDagMatPC:
        dw_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec,
                 gauge_param, inv_param.mass);
        dw_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, inv_param.matpc_type, not_dagger,
                 gauge_param.cpu_prec, gauge_param, inv_param.mass);
        break;
      case dslash_test_type::MatDagMat:
        dw_matdagmat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param,
                     inv_param.mass);
        break;
      default: printf("Test type not supported for domain wall\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      double *kappa_5 = (double *)safe_malloc(Ls * sizeof(double));
      for (int xs = 0; xs < Ls; xs++) kappa_5[xs] = kappa5;
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        dslash_4_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                    inv_param.mass);
        break;
      case dslash_test_type::M5:
        dw_dslash_5_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                       inv_param.mass, true);
        break;
      case dslash_test_type::M5inv:
        dslash_5_inv(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                     inv_param.mass, kappa_5);
        break;
      case dslash_test_type::MatPC:
        dw_4d_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec,
                    gauge_param, inv_param.mass);
        break;
      case dslash_test_type::Mat:
        dw_4d_mat(spinorRef->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param,
                  inv_param.mass);
        break;
      case dslash_test_type::MatPCDagMatPC:
        dw_4d_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa5, inv_param.matpc_type, dagger, gauge_param.cpu_prec,
                    gauge_param, inv_param.mass);
        dw_4d_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, inv_param.matpc_type, not_dagger,
                    gauge_param.cpu_prec, gauge_param, inv_param.mass);
        break;
      case dslash_test_type::MatDagMat:
        dw_4d_mat(spinorTmp->V(), hostGauge, spinor->V(), kappa5, dagger, gauge_param.cpu_prec, gauge_param,
                  inv_param.mass);
        dw_4d_mat(spinorRef->V(), hostGauge, spinorTmp->V(), kappa5, not_dagger, gauge_param.cpu_prec, gauge_param,
                  inv_param.mass);
        break;
      default: printf("Test type not supported for domain wall\n"); exit(-1);
      }
      host_free(kappa_5);
    } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_5 = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_mdwf = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
        kappa_5[xs] = 0.5 * kappa_b[xs] / kappa_c[xs];
        kappa_mdwf[xs] = -kappa_5[xs];
      }
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        dslash_4_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                    inv_param.mass);
        break;
      case dslash_test_type::M5:
        mdw_dslash_5(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                     inv_param.mass, kappa_5, true);
        break;
      case dslash_test_type::Dslash4pre:
        mdw_dslash_4_pre(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                         inv_param.mass, inv_param.b_5, inv_param.c_5, true);
        break;
      case dslash_test_type::M5inv:
        mdw_dslash_5_inv(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                         inv_param.mass, kappa_mdwf);
        break;
      case dslash_test_type::MatPC:
        mdw_matpc(spinorRef->V(), hostGauge, spinor->V(), kappa_b, kappa_c, inv_param.matpc_type, dagger,
                  gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::Mat:
        mdw_mat(spinorRef->V(), hostGauge, spinor->V(), kappa_b, kappa_c, dagger, gauge_param.cpu_prec, gauge_param,
                inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::MatPCDagMatPC:
        mdw_matpc(spinorTmp->V(), hostGauge, spinor->V(), kappa_b, kappa_c, inv_param.matpc_type, dagger,
                  gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        mdw_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), kappa_b, kappa_c, inv_param.matpc_type, not_dagger,
                  gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::MatDagMat:
        mdw_mat(spinorTmp->V(), hostGauge, spinor->V(), kappa_b, kappa_c, dagger, gauge_param.cpu_prec, gauge_param,
                inv_param.mass, inv_param.b_5, inv_param.c_5);
        mdw_mat(spinorRef->V(), hostGauge, spinorTmp->V(), kappa_b, kappa_c, not_dagger, gauge_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::MatPCDagMatPCLocal:
        // reference for MdagM local operator
        mdw_mdagm_local(spinorRef->V(), hostGauge, spinor->V(), kappa_b, kappa_c, inv_param.matpc_type,
                        gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      default: printf("Test type not supported for Mobius domain wall\n"); exit(-1);
      }
      host_free(kappa_b);
      host_free(kappa_c);
      host_free(kappa_5);
      host_free(kappa_mdwf);
    } else if (dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
      double _Complex *kappa_b = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_c = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_5 = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      double _Complex *kappa_mdwf = (double _Complex *)safe_malloc(Lsdim * sizeof(double _Complex));
      for (int xs = 0; xs < Lsdim; xs++) {
        kappa_b[xs] = 1.0 / (2 * (inv_param.b_5[xs] * (4.0 + inv_param.m5) + 1.0));
        kappa_c[xs] = 1.0 / (2 * (inv_param.c_5[xs] * (4.0 + inv_param.m5) - 1.0));
        kappa_5[xs] = 0.5 * kappa_b[xs] / kappa_c[xs];
        kappa_mdwf[xs] = -kappa_5[xs];
      }
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        dslash_4_4d(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                    inv_param.mass);
        break;
      case dslash_test_type::M5:
        mdw_eofa_m5(spinorRef->V(), spinor->V(), parity, dagger, inv_param.mass, inv_param.m5,
                    (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                    inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift, gauge_param.cpu_prec);
        break;
      case dslash_test_type::Dslash4pre:
        mdw_dslash_4_pre(spinorRef->V(), hostGauge, spinor->V(), parity, dagger, gauge_param.cpu_prec, gauge_param,
                         inv_param.mass, inv_param.b_5, inv_param.c_5, true);
        break;
      case dslash_test_type::M5inv:
        mdw_eofa_m5inv(spinorRef->V(), spinor->V(), parity, dagger, inv_param.mass, inv_param.m5,
                       (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                       inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift, gauge_param.cpu_prec);
        break;
      case dslash_test_type::Mat:
        mdw_eofa_mat(spinorRef->V(), hostGauge, spinor->V(), dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass,
                     inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1,
                     inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
        break;
      case dslash_test_type::MatDagMat:
        mdw_eofa_mat(spinorTmp->V(), hostGauge, spinor->V(), dagger, gauge_param.cpu_prec, gauge_param, inv_param.mass,
                     inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1,
                     inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
        mdw_eofa_mat(spinorRef->V(), hostGauge, spinorTmp->V(), not_dagger, gauge_param.cpu_prec, gauge_param,
                     inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]),
                     inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
        break;
      case dslash_test_type::MatPC:
        mdw_eofa_matpc(spinorRef->V(), hostGauge, spinor->V(), inv_param.matpc_type, dagger, gauge_param.cpu_prec,
                       gauge_param, inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]),
                       (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm,
                       inv_param.eofa_shift);
        break;
      case dslash_test_type::MatPCDagMatPC:
        mdw_eofa_matpc(spinorTmp->V(), hostGauge, spinor->V(), inv_param.matpc_type, dagger, gauge_param.cpu_prec,
                       gauge_param, inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]),
                       (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm,
                       inv_param.eofa_shift);
        mdw_eofa_matpc(spinorRef->V(), hostGauge, spinorTmp->V(), inv_param.matpc_type, not_dagger,
                       gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]),
                       (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm,
                       inv_param.eofa_shift);
        break;
      default: printf("Test type not supported for Mobius domain wall EOFA\n"); exit(-1);
      }
      host_free(kappa_b);
      host_free(kappa_c);
      host_free(kappa_5);
      host_free(kappa_mdwf);
    } else {
      printfQuda("Unsupported dslash_type\n");
      exit(-1);
    }

    printfQuda("done.\n");
  }

  // execute kernel
  DslashTime dslashCUDA(int niter)
  {
    DslashTime dslash_time;

    host_timer_t host_timer;
    device_timer_t device_timer;

    comm_barrier();
    device_timer.start();

    if (test_split_grid) {

      std::vector<void *> _hp_x(inv_param.num_src);
      std::vector<void *> _hp_b(inv_param.num_src);
      for (int i = 0; i < inv_param.num_src; i++) {
        _hp_x[i] = vp_spinorOut[i]->V();
        _hp_b[i] = vp_spinor[i]->V();
      }

      if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH
          || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        dslashMultiSrcCloverQuda(_hp_x.data(), _hp_b.data(), &inv_param, parity, hostGauge, &gauge_param, hostClover,
                                 hostCloverInv);
      } else {
        dslashMultiSrcQuda(_hp_x.data(), _hp_b.data(), &inv_param, parity, hostGauge, &gauge_param);
      }

    } else {

      for (int i = 0; i < niter; i++) {

        host_timer.start();

        if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
          switch (dtest_type) {
          case dslash_test_type::Dslash:
            if (transfer) {
              dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracDomainWall4DPC *>(dirac)->Dslash4(*cudaSpinorOut, *cudaSpinor, parity);
            }
            break;
          case dslash_test_type::M5:
            if (transfer) {
              dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracDomainWall4DPC *>(dirac)->Dslash5(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::M5inv:
            if (transfer) {
              dslashQuda_4dpc(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracDomainWall4DPC *>(dirac)->M5inv(*cudaSpinorOut, *cudaSpinor);
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
          default:
            errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
          }
        } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
          switch (dtest_type) {
          case dslash_test_type::Dslash:
            if (transfer) {
              dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->Dslash4(*cudaSpinorOut, *cudaSpinor, parity);
            }
            break;
          case dslash_test_type::M5:
            if (transfer) {
              dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->Dslash5(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::Dslash4pre:
            if (transfer) {
              dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->Dslash4pre(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::M5inv:
            if (transfer) {
              dslashQuda_mdwf(spinorOut->V(), spinor->V(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->M5inv(*cudaSpinorOut, *cudaSpinor);
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
          case dslash_test_type::MatPCDagMatPCLocal:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              dirac->MdagMLocal(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          default:
            errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
          }
        } else if (dslash_type == QUDA_MOBIUS_DWF_EOFA_DSLASH) {
          switch (dtest_type) {
          case dslash_test_type::Dslash:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->Dslash4(*cudaSpinorOut, *cudaSpinor, parity);
            }
            break;
          case dslash_test_type::M5:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->m5_eofa(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::Dslash4pre:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->Dslash4pre(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::M5inv:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->m5inv_eofa(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::MatPC:
          case dslash_test_type::Mat:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
              // MatQuda(spinorOut->V(), spinor->V(), &inv_param);
            } else {
              dirac->M(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPC:
          case dslash_test_type::MatDagMat:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
              // MatDagMatQuda(spinorOut->V(), spinor->V(), &inv_param);
            } else {
              dirac->MdagM(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPCLocal:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              dirac->MdagMLocal(*cudaSpinorOut, *cudaSpinor);
            }
            break;
          default: errorQuda("Undefined test type(=%d)\n", static_cast<int>(dtest_type));
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
          default:
            errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
          }
        }

        host_timer.stop();

        dslash_time.cpu_time += host_timer.last();
        // skip first and last iterations since they may skew these metrics if comms are not synchronous
        if (i > 0 && i < niter) {
          dslash_time.cpu_min = std::min(dslash_time.cpu_min, host_timer.last());
          dslash_time.cpu_max = std::max(dslash_time.cpu_max, host_timer.last());
        }
      }
    }
    device_timer.stop();
    dslash_time.event_time = device_timer.last();

    return dslash_time;
  }

  void run_test(int niter, bool = false)
  {
    {
      printfQuda("Tuning...\n");
      dslashCUDA(1); // warm-up run
    }
    printfQuda("Executing %d kernel loops...\n", niter);
    if (!transfer) dirac->Flops();
    DslashTime dslash_time = dslashCUDA(niter);
    printfQuda("done.\n\n");

    if (!test_split_grid) {
      if (!transfer) *spinorOut = *cudaSpinorOut;

      // print timing information
      printfQuda("%fus per kernel call\n", 1e6 * dslash_time.event_time / niter);
      // FIXME No flops count for twisted-clover yet
      unsigned long long flops = 0;
      if (!transfer) flops = dirac->Flops();
      printfQuda("%llu flops per kernel call, %llu flops per site\n", flops / niter,
                 (flops / niter) / cudaSpinor->Volume());
      printfQuda("GFLOPS = %f\n", 1.0e-9 * flops / dslash_time.event_time);

      size_t ghost_bytes = cudaSpinor->GhostBytes();

      printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for "
                 "aggregate message size %lu bytes\n",
                 1.0e-9 * 2 * ghost_bytes * niter / dslash_time.event_time,
                 1.0e-9 * 2 * ghost_bytes * niter / dslash_time.cpu_time, 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_max,
                 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_min, 2 * ghost_bytes);

      ::testing::Test::RecordProperty("Gflops", std::to_string(1.0e-9 * flops / dslash_time.event_time));
      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_GPU",
                                      1.0e-9 * 2 * ghost_bytes * niter / dslash_time.event_time);
      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_CPU",
                                      1.0e-9 * 2 * ghost_bytes * niter / dslash_time.cpu_time);
      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_CPU_min", 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_max);
      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_CPU_max", 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_min);
      ::testing::Test::RecordProperty("Halo_message_size_bytes", 2 * ghost_bytes);
    }
  }

  double verify()
  {
    double deviation;
    if (test_split_grid) {
      for (int n = 0; n < num_src; n++) {
        double norm2_cpu = blas::norm2(*spinorRef);
        double norm2_cpu_cuda = blas::norm2(*vp_spinorOut[n]);
        printfQuda("Result: CPU = %f, CPU-QUDA = %f\n", norm2_cpu, norm2_cpu_cuda);
        deviation = std::max(deviation, pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *vp_spinorOut[n]))));
      }
    } else {
      double norm2_cpu = blas::norm2(*spinorRef);
      double norm2_cpu_cuda = blas::norm2(*spinorOut);
      if (!transfer) {
        double norm2_cuda = blas::norm2(*cudaSpinorOut);
        printfQuda("Results: CPU = %f, CUDA=%f, CPU-CUDA = %f\n", norm2_cpu, norm2_cuda, norm2_cpu_cuda);
      } else {
        printfQuda("Result: CPU = %f, CPU-QUDA = %f\n", norm2_cpu, norm2_cpu_cuda);
      }
      deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
    }
    return deviation;
  }
};
