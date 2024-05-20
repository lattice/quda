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
#include <tune_quda.h>

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
  static inline ColorSpinorField spinor;
  static inline ColorSpinorField spinorOut;
  static inline ColorSpinorField spinorRef;
  static inline ColorSpinorField spinorTmp;
  // For split grid
  static inline std::vector<ColorSpinorField> vp_spinor;
  static inline std::vector<ColorSpinorField> vp_spinorOut;
  static inline std::vector<ColorSpinorField> vp_spinorRef;

  // CUDA color spinor fields
  ColorSpinorField cudaSpinor;
  ColorSpinorField cudaSpinorOut;
  ColorSpinorField cudaSpinorTmp;
  ColorSpinorField cudaSpinorTmp2;
  ColorSpinorField cudaSpinorTmp3;

  // Dirac pointers
  quda::Dirac *dirac = nullptr;
  quda::DiracMobiusPC *dirac_mdwf = nullptr;
  quda::DiracDomainWall4DPC *dirac_4dpc = nullptr;

  // Raw pointers
  static inline void *hostGauge[4] = {nullptr};
  static inline void *hostClover = nullptr;
  static inline void *hostCloverInv = nullptr;

  // Parameters
  static inline QudaGaugeParam gauge_param;
  static inline QudaInvertParam inv_param;
  static inline bool first_time = true;

  // Test options
  QudaParity parity = QUDA_EVEN_PARITY;
  static inline dslash_test_type dtest_type = dslash_test_type::Dslash;
  static inline bool test_split_grid = false;
  static inline bool test_domain_decomposition = false;
  int num_src = 1;

  const bool transfer = false;

  void init_ctest(int argc, char **argv, int precision, QudaReconstructType link_recon)
  {
    if (first_time) {
      gauge_param = newQudaGaugeParam();
      setWilsonGaugeParam(gauge_param);
      inv_param = newQudaInvertParam();
      setInvertParam(inv_param);
      init_host(argc, argv);
      first_time = false;
    }

    cuda_prec = getPrecision(precision);
    gauge_param.cuda_prec = cuda_prec;
    gauge_param.cuda_prec_sloppy = cuda_prec;
    gauge_param.cuda_prec_precondition = cuda_prec;
    gauge_param.cuda_prec_refinement_sloppy = cuda_prec;

    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon;
    gauge_param.reconstruct = link_recon;
    gauge_param.reconstruct_sloppy = link_recon;

    inv_param.cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec = cuda_prec;
    inv_param.clover_cuda_prec_sloppy = cuda_prec;
    inv_param.clover_cuda_prec_precondition = cuda_prec;
    inv_param.clover_cuda_prec_refinement_sloppy = cuda_prec;

    init();
  }

  void init_test(int argc, char **argv)
  {
    if (first_time) {
      gauge_param = newQudaGaugeParam();
      setWilsonGaugeParam(gauge_param);
      inv_param = newQudaInvertParam();
      setInvertParam(inv_param);
      init_host(argc, argv);
      first_time = false;
    }
    init();
  }

  void init_host(int argc, char **argv)
  {
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

    if (inv_param.cpu_prec != gauge_param.cpu_prec) errorQuda("Gauge and spinor CPU precisions must match");

    test_domain_decomposition = dd_red_black;

    for (int i = 0; i < 4; i++) inv_param.split_grid[i] = grid_partition[i];
    num_src = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
    test_split_grid = num_src > 1;
    if (test_split_grid) { dtest_type = dslash_test_type::Dslash; }

    inv_param.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;
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

    // construct input fields
    for (int dir = 0; dir < 4; dir++) hostGauge[dir] = safe_malloc((size_t)V * gauge_site_size * gauge_param.cpu_prec);

    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH
        || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      hostClover = safe_malloc((size_t)V * clover_site_size * inv_param.clover_cpu_prec);
      hostCloverInv = safe_malloc((size_t)V * clover_site_size * inv_param.clover_cpu_prec);

      if (compute_clover)
        printfQuda("Computing clover field on GPU\n");
      else {
        printfQuda("Sending clover field to GPU\n");
        constructHostCloverField(hostClover, hostCloverInv, inv_param);
      }
    }

    printfQuda("Randomizing fields... ");
    constructHostGaugeField(hostGauge, gauge_param, argc, argv);

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

    csParam.pc_type = dslash_type == QUDA_DOMAIN_WALL_DSLASH ? QUDA_5D_PC : QUDA_4D_PC;

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

    csParam.location = QUDA_CPU_FIELD_LOCATION;
    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam.gammaBasis = inv_param.gamma_basis;
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    spinor = ColorSpinorField(csParam);
    spinorOut = ColorSpinorField(csParam);
    spinorRef = ColorSpinorField(csParam);
    spinorTmp = ColorSpinorField(csParam);

    spinor.Source(QUDA_RANDOM_SOURCE);

    if (test_split_grid) {
      inv_param.num_src = num_src;
      inv_param.num_src_per_sub_partition = 1;
      resize(vp_spinor, num_src, csParam);
      resize(vp_spinorOut, num_src, csParam);
      resize(vp_spinorRef, num_src, csParam);

      std::fill(vp_spinor.begin(), vp_spinor.end(), spinor);
    }

    // set verbosity prior to loadGaugeQuda
    setVerbosity(verbosity);
    inv_param.verbosity = verbosity;
  }

  void init()
  {
    printfQuda("Sending gauge field to GPU\n");
    loadGaugeQuda(hostGauge, &gauge_param);

    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
        || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      inv_param.compute_clover = compute_clover;
      inv_param.return_clover = compute_clover;
      inv_param.compute_clover_inverse = true;
      inv_param.return_clover_inverse = true;

      loadCloverQuda(hostClover, hostCloverInv, &inv_param);
    }

    if (!transfer) {
      ColorSpinorParam csParam(spinor);
      csParam.location = QUDA_CUDA_FIELD_LOCATION;
      csParam.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
      csParam.setPrecision(inv_param.cuda_prec, inv_param.cuda_prec, true);

      printfQuda("Creating cudaSpinor with nParity = %d\n", csParam.siteSubset);
      cudaSpinor = ColorSpinorField(csParam);
      printfQuda("Creating cudaSpinorOut with nParity = %d\n", csParam.siteSubset);
      cudaSpinorOut = ColorSpinorField(csParam);
      if (test_domain_decomposition) {
        cudaSpinorTmp = ColorSpinorField(csParam);
        cudaSpinorTmp2 = ColorSpinorField(csParam);
        cudaSpinorTmp3 = ColorSpinorField(csParam);
      }

      printfQuda("Sending spinor field to GPU\n");
      cudaSpinor = spinor;

      double cpu_norm = blas::norm2(spinor);
      double cuda_norm = blas::norm2(cudaSpinor);
      printfQuda("Source: CPU = %e, CUDA = %e\n", cpu_norm, cuda_norm);

      bool pc = (dtest_type != dslash_test_type::Mat && dtest_type != dslash_test_type::MatDagMat);

      DiracParam diracParam;
      setDiracParam(diracParam, &inv_param, pc);

      dirac = Dirac::create(diracParam);

    } else {
      double cpu_norm = blas::norm2(spinor);
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
    }
  }

  static void destroy()
  {
    for (int dir = 0; dir < 4; dir++)
      if (hostGauge[dir]) host_free(hostGauge[dir]);

    if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH
        || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      if (hostClover) host_free(hostClover);
      if (hostCloverInv) host_free(hostCloverInv);
    }

    spinor = {};
    spinorOut = {};
    spinorRef = {};
    spinorTmp = {};

    if (test_split_grid) {
      vp_spinor.clear();
      vp_spinorOut.clear();
      vp_spinorRef.clear();
    }
  }

  void dslashRef()
  {
    const QudaDagType not_dagger = dagger ? QUDA_DAG_NO : QUDA_DAG_YES;
    // compare to dslash reference implementation
    printfQuda("Calculating reference implementation...");

    if (dslash_type == QUDA_WILSON_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        wil_dslash(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatPC:
        wil_matpc(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.matpc_type, inv_param.dagger,
                  inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::Mat:
        wil_mat(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.dagger, inv_param.cpu_prec,
                gauge_param);
        break;
      case dslash_test_type::MatPCDagMatPC:
        wil_matpc(spinorTmp.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.matpc_type, inv_param.dagger,
                  inv_param.cpu_prec, gauge_param);
        wil_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), inv_param.kappa, inv_param.matpc_type, not_dagger,
                  inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatDagMat:
        wil_mat(spinorTmp.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.dagger, inv_param.cpu_prec,
                gauge_param);
        wil_mat(spinorRef.data(), hostGauge, spinorTmp.data(), inv_param.kappa, not_dagger, inv_param.cpu_prec,
                gauge_param);
        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_CLOVER_WILSON_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        clover_dslash(spinorRef.data(), hostGauge, hostCloverInv, spinor.data(), parity, inv_param.dagger,
                      inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatPC:
        clover_matpc(spinorRef.data(), hostGauge, hostClover, hostCloverInv, spinor.data(), inv_param.kappa,
                     inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::Mat:
        clover_mat(spinorRef.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.dagger,
                   inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatPCDagMatPC:
        clover_matpc(spinorTmp.data(), hostGauge, hostClover, hostCloverInv, spinor.data(), inv_param.kappa,
                     inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        clover_matpc(spinorRef.data(), hostGauge, hostClover, hostCloverInv, spinorTmp.data(), inv_param.kappa,
                     inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatDagMat:
        clover_mat(spinorTmp.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.dagger,
                   inv_param.cpu_prec, gauge_param);
        clover_mat(spinorRef.data(), hostGauge, hostClover, spinorTmp.data(), inv_param.kappa, not_dagger,
                   inv_param.cpu_prec, gauge_param);
        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH) {
      printfQuda("HASENBUCH_TWIST Test: kappa=%lf mu=%lf\n", inv_param.kappa, inv_param.mu);
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        // My dslash should be the same as the clover dslash
        clover_dslash(spinorRef.data(), hostGauge, hostCloverInv, spinor.data(), parity, inv_param.dagger,
                      inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatPC:
        // my matpc op
        cloverHasenbuschTwist_matpc(spinorRef.data(), hostGauge, spinor.data(), hostClover, hostCloverInv,
                                    inv_param.kappa, inv_param.mu, inv_param.matpc_type, inv_param.dagger,
                                    inv_param.cpu_prec, gauge_param);

        break;
      case dslash_test_type::Mat:
        // my mat
        cloverHasenbuchTwist_mat(spinorRef.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.mu,
                                 inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.matpc_type);
        break;
      case dslash_test_type::MatPCDagMatPC:
        // matpc^\dagger matpc
        // my matpc op
        cloverHasenbuschTwist_matpc(spinorTmp.data(), hostGauge, spinor.data(), hostClover, hostCloverInv,
                                    inv_param.kappa, inv_param.mu, inv_param.matpc_type, inv_param.dagger,
                                    inv_param.cpu_prec, gauge_param);

        cloverHasenbuschTwist_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), hostClover, hostCloverInv,
                                    inv_param.kappa, inv_param.mu, inv_param.matpc_type, not_dagger, inv_param.cpu_prec,
                                    gauge_param);

        break;
      case dslash_test_type::MatDagMat:
        // my mat
        cloverHasenbuchTwist_mat(spinorTmp.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.mu,
                                 inv_param.dagger, inv_param.cpu_prec, gauge_param, inv_param.matpc_type);
        cloverHasenbuchTwist_mat(spinorRef.data(), hostGauge, hostClover, spinorTmp.data(), inv_param.kappa,
                                 inv_param.mu, not_dagger, inv_param.cpu_prec, gauge_param, inv_param.matpc_type);

        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_TWISTED_MASS_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tm_dslash(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                    parity, inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        else {
          tm_ndeg_dslash(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.epsilon,
                         parity, inv_param.dagger, inv_param.matpc_type, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::MatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tm_matpc(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        else {
          tm_ndeg_matpc(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::Mat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tm_mat(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 inv_param.dagger, inv_param.cpu_prec, gauge_param);
        else {
          tm_ndeg_mat(spinorRef.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.epsilon,
                      inv_param.dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::MatPCDagMatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_matpc(spinorTmp.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
          tm_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                   inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
        } else {
          tm_ndeg_matpc(spinorTmp.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
          tm_ndeg_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), inv_param.kappa, inv_param.mu, inv_param.epsilon,
                        inv_param.matpc_type, not_dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      case dslash_test_type::MatDagMat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tm_mat(spinorTmp.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 inv_param.dagger, inv_param.cpu_prec, gauge_param);
          tm_mat(spinorRef.data(), hostGauge, spinorTmp.data(), inv_param.kappa, inv_param.mu, inv_param.twist_flavor,
                 not_dagger, inv_param.cpu_prec, gauge_param);
        } else {
          tm_ndeg_mat(spinorTmp.data(), hostGauge, spinor.data(), inv_param.kappa, inv_param.mu, inv_param.epsilon,
                      inv_param.dagger, inv_param.cpu_prec, gauge_param);
          tm_ndeg_mat(spinorRef.data(), hostGauge, spinorTmp.data(), inv_param.kappa, inv_param.mu, inv_param.epsilon,
                      not_dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tmc_dslash(spinorRef.data(), hostGauge, spinor.data(), hostClover, hostCloverInv, inv_param.kappa,
                     inv_param.mu, inv_param.twist_flavor, parity, inv_param.matpc_type, inv_param.dagger,
                     inv_param.cpu_prec, gauge_param);
        else
          tmc_ndeg_dslash(spinorRef.data(), hostGauge, spinor.data(), hostClover, hostCloverInv, inv_param.kappa,
                          inv_param.mu, inv_param.epsilon, parity, inv_param.matpc_type, inv_param.dagger,
                          inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tmc_matpc(spinorRef.data(), hostGauge, spinor.data(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        else
          tmc_ndeg_matpc(spinorRef.data(), hostGauge, spinor.data(), hostClover, hostCloverInv, inv_param.kappa,
                         inv_param.mu, inv_param.epsilon, inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec,
                         gauge_param);
        break;
      case dslash_test_type::Mat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET)
          tmc_mat(spinorRef.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        else
          tmc_ndeg_mat(spinorRef.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.mu,
                       inv_param.epsilon, inv_param.dagger, inv_param.cpu_prec, gauge_param);
        break;
      case dslash_test_type::MatPCDagMatPC:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_matpc(spinorTmp.data(), hostGauge, spinor.data(), hostClover, hostCloverInv, inv_param.kappa, inv_param.mu,
                    inv_param.twist_flavor, inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec, gauge_param);
          tmc_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), hostClover, hostCloverInv, inv_param.kappa,
                    inv_param.mu, inv_param.twist_flavor, inv_param.matpc_type, not_dagger, inv_param.cpu_prec,
                    gauge_param);
        } else {
          tmc_ndeg_matpc(spinorTmp.data(), hostGauge, spinor.data(), hostClover, hostCloverInv, inv_param.kappa,
                         inv_param.mu, inv_param.epsilon, inv_param.matpc_type, inv_param.dagger, inv_param.cpu_prec,
                         gauge_param);
          tmc_ndeg_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), hostClover, hostCloverInv, inv_param.kappa,
                         inv_param.mu, inv_param.epsilon, inv_param.matpc_type, not_dagger, inv_param.cpu_prec,
                         gauge_param);
        }
        break;
      case dslash_test_type::MatDagMat:
        if (inv_param.twist_flavor == QUDA_TWIST_SINGLET) {
          tmc_mat(spinorTmp.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, inv_param.dagger, inv_param.cpu_prec, gauge_param);
          tmc_mat(spinorRef.data(), hostGauge, hostClover, spinorTmp.data(), inv_param.kappa, inv_param.mu,
                  inv_param.twist_flavor, not_dagger, inv_param.cpu_prec, gauge_param);
        } else {
          tmc_ndeg_mat(spinorTmp.data(), hostGauge, hostClover, spinor.data(), inv_param.kappa, inv_param.mu,
                       inv_param.epsilon, inv_param.dagger, inv_param.cpu_prec, gauge_param);
          tmc_ndeg_mat(spinorRef.data(), hostGauge, hostClover, spinorTmp.data(), inv_param.kappa, inv_param.mu,
                       inv_param.epsilon, not_dagger, inv_param.cpu_prec, gauge_param);
        }
        break;
      default: printfQuda("Test type not defined\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_DOMAIN_WALL_DSLASH) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        dw_dslash(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                  gauge_param, inv_param.mass);
        break;
      case dslash_test_type::MatPC:
        dw_matpc(spinorRef.data(), hostGauge, spinor.data(), kappa5, inv_param.matpc_type, inv_param.dagger,
                 gauge_param.cpu_prec, gauge_param, inv_param.mass);
        break;
      case dslash_test_type::Mat:
        dw_mat(spinorRef.data(), hostGauge, spinor.data(), kappa5, inv_param.dagger, gauge_param.cpu_prec, gauge_param,
               inv_param.mass);
        break;
      case dslash_test_type::MatPCDagMatPC:
        dw_matpc(spinorTmp.data(), hostGauge, spinor.data(), kappa5, inv_param.matpc_type, inv_param.dagger,
                 gauge_param.cpu_prec, gauge_param, inv_param.mass);
        dw_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), kappa5, inv_param.matpc_type, not_dagger,
                 gauge_param.cpu_prec, gauge_param, inv_param.mass);
        break;
      case dslash_test_type::MatDagMat:
        dw_matdagmat(spinorRef.data(), hostGauge, spinor.data(), kappa5, inv_param.dagger, gauge_param.cpu_prec,
                     gauge_param, inv_param.mass);
        break;
      default: printf("Test type not supported for domain wall\n"); exit(-1);
      }
    } else if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
      double *kappa_5 = (double *)safe_malloc(Ls * sizeof(double));
      for (int xs = 0; xs < Ls; xs++) kappa_5[xs] = kappa5;
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        dslash_4_4d(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                    gauge_param, inv_param.mass);
        break;
      case dslash_test_type::M5:
        dw_dslash_5_4d(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                       gauge_param, inv_param.mass, true);
        break;
      case dslash_test_type::M5inv:
        dslash_5_inv(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                     gauge_param, inv_param.mass, kappa_5);
        break;
      case dslash_test_type::MatPC:
        dw_4d_matpc(spinorRef.data(), hostGauge, spinor.data(), kappa5, inv_param.matpc_type, inv_param.dagger,
                    gauge_param.cpu_prec, gauge_param, inv_param.mass);
        break;
      case dslash_test_type::Mat:
        dw_4d_mat(spinorRef.data(), hostGauge, spinor.data(), kappa5, inv_param.dagger, gauge_param.cpu_prec,
                  gauge_param, inv_param.mass);
        break;
      case dslash_test_type::MatPCDagMatPC:
        dw_4d_matpc(spinorTmp.data(), hostGauge, spinor.data(), kappa5, inv_param.matpc_type, inv_param.dagger,
                    gauge_param.cpu_prec, gauge_param, inv_param.mass);
        dw_4d_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), kappa5, inv_param.matpc_type, not_dagger,
                    gauge_param.cpu_prec, gauge_param, inv_param.mass);
        break;
      case dslash_test_type::MatDagMat:
        dw_4d_mat(spinorTmp.data(), hostGauge, spinor.data(), kappa5, inv_param.dagger, gauge_param.cpu_prec,
                  gauge_param, inv_param.mass);
        dw_4d_mat(spinorRef.data(), hostGauge, spinorTmp.data(), kappa5, not_dagger, gauge_param.cpu_prec, gauge_param,
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
        dslash_4_4d(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                    gauge_param, inv_param.mass);
        break;
      case dslash_test_type::M5:
        mdw_dslash_5(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                     gauge_param, inv_param.mass, kappa_5, true);
        break;
      case dslash_test_type::Dslash4pre:
        mdw_dslash_4_pre(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                         gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5, true);
        break;
      case dslash_test_type::M5inv:
        mdw_dslash_5_inv(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                         gauge_param, inv_param.mass, kappa_mdwf);
        break;
      case dslash_test_type::MatPC:
        mdw_matpc(spinorRef.data(), hostGauge, spinor.data(), kappa_b, kappa_c, inv_param.matpc_type, inv_param.dagger,
                  gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::Mat:
        mdw_mat(spinorRef.data(), hostGauge, spinor.data(), kappa_b, kappa_c, inv_param.dagger, gauge_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::MatPCDagMatPC:
        mdw_matpc(spinorTmp.data(), hostGauge, spinor.data(), kappa_b, kappa_c, inv_param.matpc_type, inv_param.dagger,
                  gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        mdw_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), kappa_b, kappa_c, inv_param.matpc_type, not_dagger,
                  gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::MatDagMat:
        mdw_mat(spinorTmp.data(), hostGauge, spinor.data(), kappa_b, kappa_c, inv_param.dagger, gauge_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        mdw_mat(spinorRef.data(), hostGauge, spinorTmp.data(), kappa_b, kappa_c, not_dagger, gauge_param.cpu_prec,
                gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5);
        break;
      case dslash_test_type::MatPCDagMatPCLocal:
        // reference for MdagM local operator
        mdw_mdagm_local(spinorRef.data(), hostGauge, spinor.data(), kappa_b, kappa_c, inv_param.matpc_type,
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
        dslash_4_4d(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                    gauge_param, inv_param.mass);
        break;
      case dslash_test_type::M5:
        mdw_eofa_m5(spinorRef.data(), spinor.data(), parity, inv_param.dagger, inv_param.mass, inv_param.m5,
                    (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                    inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift, gauge_param.cpu_prec);
        break;
      case dslash_test_type::Dslash4pre:
        mdw_dslash_4_pre(spinorRef.data(), hostGauge, spinor.data(), parity, inv_param.dagger, gauge_param.cpu_prec,
                         gauge_param, inv_param.mass, inv_param.b_5, inv_param.c_5, true);
        break;
      case dslash_test_type::M5inv:
        mdw_eofa_m5inv(spinorRef.data(), spinor.data(), parity, inv_param.dagger, inv_param.mass, inv_param.m5,
                       (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2,
                       inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift, gauge_param.cpu_prec);
        break;
      case dslash_test_type::Mat:
        mdw_eofa_mat(spinorRef.data(), hostGauge, spinor.data(), inv_param.dagger, gauge_param.cpu_prec, gauge_param,
                     inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]),
                     inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
        break;
      case dslash_test_type::MatDagMat:
        mdw_eofa_mat(spinorTmp.data(), hostGauge, spinor.data(), inv_param.dagger, gauge_param.cpu_prec, gauge_param,
                     inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]),
                     inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
        mdw_eofa_mat(spinorRef.data(), hostGauge, spinorTmp.data(), not_dagger, gauge_param.cpu_prec, gauge_param,
                     inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]), (__real__ inv_param.c_5[0]),
                     inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm, inv_param.eofa_shift);
        break;
      case dslash_test_type::MatPC:
        mdw_eofa_matpc(spinorRef.data(), hostGauge, spinor.data(), inv_param.matpc_type, inv_param.dagger,
                       gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]),
                       (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm,
                       inv_param.eofa_shift);
        break;
      case dslash_test_type::MatPCDagMatPC:
        mdw_eofa_matpc(spinorTmp.data(), hostGauge, spinor.data(), inv_param.matpc_type, inv_param.dagger,
                       gauge_param.cpu_prec, gauge_param, inv_param.mass, inv_param.m5, (__real__ inv_param.b_5[0]),
                       (__real__ inv_param.c_5[0]), inv_param.mq1, inv_param.mq2, inv_param.mq3, inv_param.eofa_pm,
                       inv_param.eofa_shift);
        mdw_eofa_matpc(spinorRef.data(), hostGauge, spinorTmp.data(), inv_param.matpc_type, not_dagger,
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
        _hp_x[i] = vp_spinorOut[i].data();
        _hp_b[i] = vp_spinor[i].data();
      }

      if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH
          || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        dslashMultiSrcCloverQuda(_hp_x.data(), _hp_b.data(), &inv_param, parity, hostGauge, &gauge_param, hostClover,
                                 hostCloverInv);
      } else {
        dslashMultiSrcQuda(_hp_x.data(), _hp_b.data(), &inv_param, parity, hostGauge, &gauge_param);
      }

    } else if (test_domain_decomposition) {

      if (dd_red_black) {
        for (auto i = 0u; i < 4; i++) {
          cudaSpinor.dd.blockDim[i] = dd_block_size[i];
          cudaSpinorTmp.dd.blockDim[i] = dd_block_size[i];
          cudaSpinorTmp2.dd.blockDim[i] = dd_block_size[i];
          cudaSpinorTmp3.dd.blockDim[i] = dd_block_size[i];
        }

        blas::zero(cudaSpinorOut);

        // We test that applying all 4 possible operators Drr, Drb, Dbr, Dbb gived D
        for (int col = 0; col < 4; col++) {

          cudaSpinor.dd.reset(DD::red_black_type, col % 2 == 0 ? DD::red_active : DD::black_active);
          cudaSpinorTmp.dd.reset(DD::red_black_type, col / 2 == 0 ? DD::red_active : DD::black_active);

          switch (dtest_type) {
          case dslash_test_type::Dslash: dirac->Dslash(cudaSpinorTmp, cudaSpinor, parity); break;
          case dslash_test_type::MatPC:
          case dslash_test_type::Mat: dirac->M(cudaSpinorTmp, cudaSpinor); break;
          case dslash_test_type::MatPCDagMatPC:
          case dslash_test_type::MatDagMat: dirac->MdagM(cudaSpinorTmp, cudaSpinor); break;
          default:
            errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
          }

          cudaSpinor.dd.reset();
          cudaSpinorTmp.dd.reset();

          blas::xpy(cudaSpinorTmp, cudaSpinorOut);

          // We also test that Dyx is same as D applied to projected in and out spinors
          cudaSpinorTmp2 = spinor;
          cudaSpinorTmp2.dd.reset(DD::red_black_type, col % 2 == 0 ? DD::red_active : DD::black_active);
          cudaSpinorTmp2.projectDD();
          cudaSpinorTmp2.dd.reset();

          switch (dtest_type) {
          case dslash_test_type::Dslash: dirac->Dslash(cudaSpinorTmp3, cudaSpinorTmp2, parity); break;
          case dslash_test_type::MatPC:
          case dslash_test_type::Mat: dirac->M(cudaSpinorTmp3, cudaSpinorTmp2); break;
          case dslash_test_type::MatPCDagMatPC:
          case dslash_test_type::MatDagMat: dirac->MdagM(cudaSpinorTmp3, cudaSpinorTmp2); break;
          default:
            errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
          }

          cudaSpinorTmp3.dd.reset(DD::red_black_type, col / 2 == 0 ? DD::red_active : DD::black_active);
          cudaSpinorTmp3.projectDD();
          cudaSpinorTmp3.dd.reset();

          spinorTmp = cudaSpinorTmp;
          spinorOut = cudaSpinorTmp3;

          double deviation = std::pow(10, -(double)(ColorSpinorField::Compare(spinorTmp, spinorOut)));
          printfQuda("Deviaton for (D-PDP)_{%d,%d}*spinor is %e\n", col % 2, col / 2, deviation);
        }
      } else {
        errorQuda("Test dd type not supported");
      }

    } else {

      for (int i = 0; i < niter; i++) {

        host_timer.start();

        if (dslash_type == QUDA_DOMAIN_WALL_4D_DSLASH) {
          switch (dtest_type) {
          case dslash_test_type::Dslash:
            if (transfer) {
              dslashQuda_4dpc(spinorOut.data(), spinor.data(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracDomainWall4DPC *>(dirac)->Dslash4(cudaSpinorOut, cudaSpinor, parity);
            }
            break;
          case dslash_test_type::M5:
            if (transfer) {
              dslashQuda_4dpc(spinorOut.data(), spinor.data(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracDomainWall4DPC *>(dirac)->Dslash5(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::M5inv:
            if (transfer) {
              dslashQuda_4dpc(spinorOut.data(), spinor.data(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracDomainWall4DPC *>(dirac)->M5inv(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPC:
          case dslash_test_type::Mat:
            if (transfer) {
              MatQuda(spinorOut.data(), spinor.data(), &inv_param);
            } else {
              dirac->M(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPC:
          case dslash_test_type::MatDagMat:
            if (transfer) {
              MatDagMatQuda(spinorOut.data(), spinor.data(), &inv_param);
            } else {
              dirac->MdagM(cudaSpinorOut, cudaSpinor);
            }
            break;
          default:
            errorQuda("Test type %s not support for current Dslash", get_string(dtest_type_map, dtest_type).c_str());
          }
        } else if (dslash_type == QUDA_MOBIUS_DWF_DSLASH) {
          switch (dtest_type) {
          case dslash_test_type::Dslash:
            if (transfer) {
              dslashQuda_mdwf(spinorOut.data(), spinor.data(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->Dslash4(cudaSpinorOut, cudaSpinor, parity);
            }
            break;
          case dslash_test_type::M5:
            if (transfer) {
              dslashQuda_mdwf(spinorOut.data(), spinor.data(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->Dslash5(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::Dslash4pre:
            if (transfer) {
              dslashQuda_mdwf(spinorOut.data(), spinor.data(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->Dslash4pre(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::M5inv:
            if (transfer) {
              dslashQuda_mdwf(spinorOut.data(), spinor.data(), &inv_param, parity, dtest_type);
            } else {
              static_cast<quda::DiracMobiusPC *>(dirac)->M5inv(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPC:
          case dslash_test_type::Mat:
            if (transfer) {
              MatQuda(spinorOut.data(), spinor.data(), &inv_param);
            } else {
              dirac->M(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPC:
          case dslash_test_type::MatDagMat:
            if (transfer) {
              MatDagMatQuda(spinorOut.data(), spinor.data(), &inv_param);
            } else {
              dirac->MdagM(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPCLocal:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              dirac->MdagMLocal(cudaSpinorOut, cudaSpinor);
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
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->Dslash4(cudaSpinorOut, cudaSpinor, parity);
            }
            break;
          case dslash_test_type::M5:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->m5_eofa(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::Dslash4pre:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->Dslash4pre(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::M5inv:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              static_cast<quda::DiracMobiusEofaPC *>(dirac)->m5inv_eofa(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPC:
          case dslash_test_type::Mat:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              dirac->M(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPC:
          case dslash_test_type::MatDagMat:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              dirac->MdagM(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPCLocal:
            if (transfer) {
              errorQuda("(transfer == true) version NOT yet available!\n");
            } else {
              dirac->MdagMLocal(cudaSpinorOut, cudaSpinor);
            }
            break;
          default: errorQuda("Undefined test type(=%d)\n", static_cast<int>(dtest_type));
          }
        } else {
          switch (dtest_type) {
          case dslash_test_type::Dslash:
            if (dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
              if (transfer) {
                dslashQuda(spinorOut.data(), spinor.data(), &inv_param, parity);
              } else {
                dirac->Dslash(cudaSpinorOut, cudaSpinor, parity);
              }
            } else {
              if (transfer) {
                dslashQuda(spinorOut.data(), spinor.data(), &inv_param, parity);
              } else {
                dirac->Dslash(cudaSpinorOut, cudaSpinor, parity);
              }
            }
            break;
          case dslash_test_type::MatPC:
          case dslash_test_type::Mat:
            if (transfer) {
              MatQuda(spinorOut.data(), spinor.data(), &inv_param);
            } else {
              dirac->M(cudaSpinorOut, cudaSpinor);
            }
            break;
          case dslash_test_type::MatPCDagMatPC:
          case dslash_test_type::MatDagMat:
            if (transfer) {
              MatDagMatQuda(spinorOut.data(), spinor.data(), &inv_param);
            } else {
              dirac->MdagM(cudaSpinorOut, cudaSpinor);
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

    auto flops0 = quda::Tunable::flops_global();
    auto bytes0 = quda::Tunable::bytes_global();

    printfQuda("Executing %d kernel loops...\n", niter);
    DslashTime dslash_time = dslashCUDA(niter);
    printfQuda("done.\n\n");

    unsigned long long flops = (quda::Tunable::flops_global() - flops0);
    unsigned long long bytes = (quda::Tunable::bytes_global() - bytes0);

    if (!test_split_grid) {
      if (!transfer) spinorOut = cudaSpinorOut;

      // print timing information
      printfQuda("%fus per kernel call\n", 1e6 * dslash_time.event_time / niter);

      printfQuda("%llu flops per kernel call, %llu flops per site %llu bytes per site\n", flops / niter,
                 (flops / niter) / cudaSpinor.Volume(), (bytes / niter) / cudaSpinor.Volume());

      double gflops = 1.0e-9 * flops / dslash_time.event_time;
      printfQuda("GFLOPS = %f\n", gflops);
      ::testing::Test::RecordProperty("Gflops", std::to_string(gflops));

      double gbytes = 1.0e-9 * bytes / dslash_time.event_time;
      printfQuda("GBYTES = %f\n", gbytes);
      ::testing::Test::RecordProperty("Gbytes", std::to_string(gbytes));

      size_t ghost_bytes = cudaSpinor.GhostBytes();

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
    double deviation = 0.0;
    if (test_split_grid) {
      for (int n = 0; n < num_src; n++) {
        auto norm_cpu = blas::norm2(spinorRef);
        auto norm_cpu_quda = blas::norm2(vp_spinorOut[n]);
        auto max_deviation = blas::max_deviation(spinorRef, vp_spinorOut[n]);

        printfQuda("Results: reference = %f, QUDA = %f, L2 relative deviation = %e, max deviation = %e\n", norm_cpu,
                   norm_cpu_quda, 1.0 - sqrt(norm_cpu_quda / norm_cpu), max_deviation[0]);
        deviation = std::max(deviation, std::pow(10, -(double)(ColorSpinorField::Compare(spinorRef, vp_spinorOut[n]))));
      }
    } else {
      auto norm_cpu = blas::norm2(spinorRef);
      auto norm_cpu_quda = blas::norm2(spinorOut);
      auto max_deviation = blas::max_deviation(spinorRef, spinorOut);

      printfQuda("Results: reference = %f, QUDA = %f, L2 relative deviation = %e, max deviation = %e\n", norm_cpu,
                 norm_cpu_quda, 1.0 - sqrt(norm_cpu_quda / norm_cpu), max_deviation[0]);
      deviation = std::pow(10, -(double)(ColorSpinorField::Compare(spinorRef, spinorOut)));
    }
    return deviation;
  }
};
