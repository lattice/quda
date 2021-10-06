#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include <quda.h>
#include <gauge_field.h>
#include <dirac_quda.h>
#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>

#include "dslash_test_helpers.h"
#include <assert.h>
#include <gtest/gtest.h>

using namespace quda;

dslash_test_type dtest_type = dslash_test_type::Dslash;
CLI::TransformPairs<dslash_test_type> dtest_type_map {
  {"Dslash", dslash_test_type::Dslash}, {"MatPC", dslash_test_type::MatPC}, {"Mat", dslash_test_type::Mat}
  // left here for completeness but not support in staggered dslash test
  // {"MatPCDagMatPC", dslash_test_type::MatPCDagMatPC},
  // {"MatDagMat", dslash_test_type::MatDagMat},
  // {"M5", dslash_test_type::M5},
  // {"M5inv", dslash_test_type::M5inv},
  // {"Dslash4pre", dslash_test_type::Dslash4pre}
};

struct DslashTime {
  double event_time;
  double cpu_time;
  double cpu_min;
  double cpu_max;

  DslashTime() : event_time(0.0), cpu_time(0.0), cpu_min(DBL_MAX), cpu_max(0.0) { }
};

struct StaggeredDslashTestWrapper {

  bool is_ctest = false; // Added to distinguish from being used in dslash_test.

  void *qdp_inlink[4] = {nullptr, nullptr, nullptr, nullptr};

  QudaGaugeParam gauge_param;
  QudaInvertParam inv_param;

  void *milc_fatlink_gpu;
  void *milc_longlink_gpu;

  cpuGaugeField *cpuFat = nullptr;
  cpuGaugeField *cpuLong = nullptr;

  cpuColorSpinorField *spinor = nullptr;
  cpuColorSpinorField *spinorOut = nullptr;
  cpuColorSpinorField *spinorRef = nullptr;
  cpuColorSpinorField *tmpCpu = nullptr;
  cudaColorSpinorField *cudaSpinor = nullptr;
  cudaColorSpinorField *cudaSpinorOut = nullptr;
  cudaColorSpinorField *tmp = nullptr;

  std::vector<cpuColorSpinorField *> vp_spinor;
  std::vector<cpuColorSpinorField *> vp_spinor_out;

  // In the HISQ case, we include building fat/long links in this unit test
  void *qdp_fatlink_cpu[4] = {nullptr, nullptr, nullptr, nullptr};
  void *qdp_longlink_cpu[4] = {nullptr, nullptr, nullptr, nullptr};
  void **ghost_fatlink_cpu, **ghost_longlink_cpu;

  QudaParity parity = QUDA_EVEN_PARITY;

  Dirac *dirac;

  // For loading the gauge fields
  int argc_copy;
  char **argv_copy;

  // Split grid options
  int num_src;
  int test_split_grid;

  void staggeredDslashRef()
  {

    // compare to dslash reference implementation
    printfQuda("Calculating reference implementation...");
    switch (dtest_type) {
    case dslash_test_type::Dslash:
      staggeredDslash(spinorRef, qdp_fatlink_cpu, qdp_longlink_cpu, ghost_fatlink_cpu, ghost_longlink_cpu, spinor,
                      parity, dagger, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
      break;
    case dslash_test_type::MatPC:
      staggeredMatDagMat(spinorRef, qdp_fatlink_cpu, qdp_longlink_cpu, ghost_fatlink_cpu, ghost_longlink_cpu, spinor,
                         mass, 0, inv_param.cpu_prec, gauge_param.cpu_prec, tmpCpu, parity, dslash_type);
      break;
    case dslash_test_type::Mat:
      // the !dagger is to reconcile the QUDA convention of D_stag = {{ 2m, -D_{eo}}, -D_{oe}, 2m}} vs the host convention without the minus signs
      staggeredDslash(reinterpret_cast<cpuColorSpinorField *>(&spinorRef->Even()), qdp_fatlink_cpu, qdp_longlink_cpu,
                      ghost_fatlink_cpu, ghost_longlink_cpu, reinterpret_cast<cpuColorSpinorField *>(&spinor->Odd()),
                      QUDA_EVEN_PARITY, !dagger, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
      staggeredDslash(reinterpret_cast<cpuColorSpinorField *>(&spinorRef->Odd()), qdp_fatlink_cpu, qdp_longlink_cpu,
                      ghost_fatlink_cpu, ghost_longlink_cpu, reinterpret_cast<cpuColorSpinorField *>(&spinor->Even()),
                      QUDA_ODD_PARITY, !dagger, inv_param.cpu_prec, gauge_param.cpu_prec, dslash_type);
      if (dslash_type == QUDA_LAPLACE_DSLASH) {
        xpay(spinor->V(), kappa, spinorRef->V(), spinor->Length(), gauge_param.cpu_prec);
      } else {
        axpy(2 * mass, spinor->V(), spinorRef->V(), spinor->Length(), gauge_param.cpu_prec);
      }
      break;
    default: errorQuda("Test type not defined");
    }
  }

  void init_ctest_once()
  {
    static bool has_been_called = false;
    if (has_been_called) { errorQuda("This function is not supposed to be called twice.\n"); }
    is_ctest = true; // Is being used in dslash_ctest.
    has_been_called = true;
  }

  void end_ctest_once()
  {
    static bool has_been_called = false;
    if (has_been_called) { errorQuda("This function is not supposed to be called twice.\n"); }
    has_been_called = true;
  }

  void init_ctest(int precision, QudaReconstructType link_recon_)
  {
    gauge_param = newQudaGaugeParam();
    inv_param = newQudaInvertParam();

    setStaggeredGaugeParam(gauge_param);
    setStaggeredInvertParam(inv_param);

    auto prec = getPrecision(precision);
    setVerbosity(QUDA_SUMMARIZE);

    gauge_param.cuda_prec = prec;
    gauge_param.cuda_prec_sloppy = prec;
    gauge_param.cuda_prec_precondition = prec;
    gauge_param.cuda_prec_refinement_sloppy = prec;

    inv_param.cuda_prec = prec;

    link_recon = link_recon_;

    init();
  }

  void init_test()
  {
    gauge_param = newQudaGaugeParam();
    inv_param = newQudaInvertParam();

    setStaggeredGaugeParam(gauge_param);
    setStaggeredInvertParam(inv_param);

    init();
  }

  void init()
  {
    inv_param.split_grid[0] = grid_partition[0];
    inv_param.split_grid[1] = grid_partition[1];
    inv_param.split_grid[2] = grid_partition[2];
    inv_param.split_grid[3] = grid_partition[3];

    num_src = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
    test_split_grid = num_src > 1;

    if (test_split_grid) { dtest_type = dslash_test_type::Dslash; }

    inv_param.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;

    setDims(gauge_param.X);
    dw_setDims(gauge_param.X, 1);
    if (Nsrc != 1) {
      warningQuda("Ignoring Nsrc = %d, setting to 1.", Nsrc);
      Nsrc = 1;
    }

    // Allocate a lot of memory because I'm very confused
    void *milc_fatlink_cpu = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
    void *milc_longlink_cpu = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

    milc_fatlink_gpu = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
    milc_longlink_gpu = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

    void *qdp_fatlink_gpu[4];
    void *qdp_longlink_gpu[4];

    for (int dir = 0; dir < 4; dir++) {
      qdp_inlink[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);

      qdp_fatlink_gpu[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink_gpu[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);

      qdp_fatlink_cpu[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink_cpu[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    }

    bool gauge_loaded = false;
    constructStaggeredHostDeviceGaugeField(qdp_inlink, qdp_longlink_cpu, qdp_longlink_gpu, qdp_fatlink_cpu,
                                           qdp_fatlink_gpu, gauge_param, argc_copy, argv_copy, gauge_loaded);

    // Alright, we've created all the void** links.
    // Create the void* pointers
    reorderQDPtoMILC(milc_fatlink_gpu, qdp_fatlink_gpu, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
    reorderQDPtoMILC(milc_fatlink_cpu, qdp_fatlink_cpu, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
    reorderQDPtoMILC(milc_longlink_gpu, qdp_longlink_gpu, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
    reorderQDPtoMILC(milc_longlink_cpu, qdp_longlink_cpu, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
    // Create ghost zones for CPU fields,
    // prepare and load the GPU fields

#ifdef MULTI_GPU
    gauge_param.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
    gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    GaugeFieldParam cpuFatParam(gauge_param, milc_fatlink_cpu);
    cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    cpuFat = new cpuGaugeField(cpuFatParam);
    ghost_fatlink_cpu = cpuFat->Ghost();

    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
      GaugeFieldParam cpuLongParam(gauge_param, milc_longlink_cpu);
      cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
      cpuLong = new cpuGaugeField(cpuLongParam);
      ghost_longlink_cpu = cpuLong ? cpuLong->Ghost() : nullptr;
    }
#endif

    gauge_param.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
    if (dslash_type == QUDA_STAGGERED_DSLASH) {
      gauge_param.reconstruct = gauge_param.reconstruct_sloppy = (link_recon == QUDA_RECONSTRUCT_12) ?
                                             QUDA_RECONSTRUCT_13 :
        (link_recon == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_9 :
                                             link_recon;
    } else {
      gauge_param.reconstruct = gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
    }

    // set verbosity prior to loadGaugeQuda
    setVerbosity(verbosity);

    printfQuda("Sending fat links to GPU\n");
    loadGaugeQuda(milc_fatlink_gpu, &gauge_param);

    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;

#ifdef MULTI_GPU
    gauge_param.ga_pad *= 3;
#endif

    if (dslash_type == QUDA_ASQTAD_DSLASH) {
      gauge_param.staggered_phase_type = QUDA_STAGGERED_PHASE_NO;
      gauge_param.reconstruct = gauge_param.reconstruct_sloppy = (link_recon == QUDA_RECONSTRUCT_12) ?
                                             QUDA_RECONSTRUCT_13 :
        (link_recon == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_9 :
                                             link_recon;
      printfQuda("Sending long links to GPU\n");
      loadGaugeQuda(milc_longlink_gpu, &gauge_param);
    }

    ColorSpinorParam csParam;
    csParam.nColor = 3;
    csParam.nSpin = 1;
    csParam.nDim = 5;
    for (int d = 0; d < 4; d++) { csParam.x[d] = gauge_param.X[d]; }
    csParam.x[4] = 1;

    csParam.setPrecision(inv_param.cpu_prec);
    // inv_param.solution_type = QUDA_MAT_SOLUTION;
    csParam.pad = 0;
    if (dtest_type != dslash_test_type::Mat && dslash_type != QUDA_LAPLACE_DSLASH) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
      inv_param.solution_type = QUDA_MATPC_SOLUTION;
    } else {
      csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
      inv_param.solution_type = QUDA_MAT_SOLUTION;
    }

    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam.gammaBasis = inv_param.gamma_basis; // this parameter is meaningless for staggered
    csParam.create = QUDA_ZERO_FIELD_CREATE;
    csParam.pc_type = QUDA_4D_PC;

    spinor = new cpuColorSpinorField(csParam);
    spinorOut = new cpuColorSpinorField(csParam);
    spinorRef = new cpuColorSpinorField(csParam);
    tmpCpu = new cpuColorSpinorField(csParam);

    spinor->Source(QUDA_RANDOM_SOURCE);

    if (test_split_grid) {
      inv_param.num_src = num_src;
      inv_param.num_src_per_sub_partition = 1;
      for (int n = 0; n < num_src; n++) {
        vp_spinor.push_back(new cpuColorSpinorField(csParam));
        vp_spinor_out.push_back(new cpuColorSpinorField(csParam));
        *vp_spinor[n] = *spinor;
      }
    }

    csParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    csParam.pad = inv_param.sp_pad;
    csParam.setPrecision(inv_param.cuda_prec);

    cudaSpinor = new cudaColorSpinorField(csParam);
    cudaSpinorOut = new cudaColorSpinorField(csParam);
    *cudaSpinor = *spinor;
    tmp = new cudaColorSpinorField(csParam);

    bool pc = (dtest_type == dslash_test_type::MatPC); // For test_type 0, can use either pc or not pc
    // because both call the same "Dslash" directly.
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    diracParam.tmp1 = tmp;
    dirac = Dirac::create(diracParam);

    for (int dir = 0; dir < 4; dir++) {
      host_free(qdp_fatlink_gpu[dir]);
      host_free(qdp_longlink_gpu[dir]);
      host_free(qdp_inlink[dir]);
    }
    host_free(milc_fatlink_cpu);
    host_free(milc_longlink_cpu);
  }

  void end()
  {
    for (int dir = 0; dir < 4; dir++) {
      if (qdp_fatlink_cpu[dir] != nullptr) {
        host_free(qdp_fatlink_cpu[dir]);
        qdp_fatlink_cpu[dir] = nullptr;
      }
      if (qdp_longlink_cpu[dir] != nullptr) {
        host_free(qdp_longlink_cpu[dir]);
        qdp_longlink_cpu[dir] = nullptr;
      }
    }

    if (dirac != nullptr) {
      delete dirac;
      dirac = nullptr;
    }
    if (cudaSpinor != nullptr) {
      delete cudaSpinor;
      cudaSpinor = nullptr;
    }
    if (cudaSpinorOut != nullptr) {
      delete cudaSpinorOut;
      cudaSpinorOut = nullptr;
    }
    if (tmp != nullptr) {
      delete tmp;
      tmp = nullptr;
    }

    if (spinor != nullptr) {
      delete spinor;
      spinor = nullptr;
    }
    if (spinorOut != nullptr) {
      delete spinorOut;
      spinorOut = nullptr;
    }
    if (spinorRef != nullptr) {
      delete spinorRef;
      spinorRef = nullptr;
    }
    if (tmpCpu != nullptr) {
      delete tmpCpu;
      tmpCpu = nullptr;
    }

    if (test_split_grid) {
      for (auto p : vp_spinor) { delete p; }
      for (auto p : vp_spinor_out) { delete p; }
      vp_spinor.clear();
      vp_spinor_out.clear();
    }

    host_free(milc_fatlink_gpu);
    milc_fatlink_gpu = nullptr;
    host_free(milc_longlink_gpu);
    milc_longlink_gpu = nullptr;

    freeGaugeQuda();

    if (cpuFat) {
      delete cpuFat;
      cpuFat = nullptr;
    }
    if (cpuLong) {
      delete cpuLong;
      cpuLong = nullptr;
    }
    commDimPartitionedReset();
  }

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
        _hp_x[i] = vp_spinor_out[i]->V();
        _hp_b[i] = vp_spinor[i]->V();
      }
      dslashMultiSrcStaggeredQuda(_hp_x.data(), _hp_b.data(), &inv_param, parity, milc_fatlink_gpu, milc_longlink_gpu,
                                  &gauge_param);

    } else {

      for (int i = 0; i < niter; i++) {

        host_timer.start();

        switch (dtest_type) {
        case dslash_test_type::Dslash: dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity); break;
        case dslash_test_type::MatPC: dirac->M(*cudaSpinorOut, *cudaSpinor); break;
        case dslash_test_type::Mat: dirac->M(*cudaSpinorOut, *cudaSpinor); break;
        default: errorQuda("Test type %d not defined on staggered dslash.\n", static_cast<int>(dtest_type));
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

  void run_test(int niter, bool print_metrics = false)
  {
    printfQuda("Tuning...\n");
    dslashCUDA(1);

    // reset flop counter
    dirac->Flops();

    DslashTime dslash_time = dslashCUDA(niter);
    *spinorOut = *cudaSpinorOut;

    if (print_metrics) {
      printfQuda("%fus per kernel call\n", 1e6 * dslash_time.event_time / niter);

      unsigned long long flops = dirac->Flops();
      double gflops = 1.0e-9 * flops / dslash_time.event_time;
      printfQuda("GFLOPS = %f\n", gflops);
      ::testing::Test::RecordProperty("Gflops", std::to_string(gflops));

      size_t ghost_bytes = cudaSpinor->GhostBytes();

      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_GPU",
                                      1.0e-9 * 2 * ghost_bytes * niter / dslash_time.event_time);
      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_CPU",
                                      1.0e-9 * 2 * ghost_bytes * niter / dslash_time.cpu_time);
      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_CPU_min", 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_max);
      ::testing::Test::RecordProperty("Halo_bidirectitonal_BW_CPU_max", 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_min);
      ::testing::Test::RecordProperty("Halo_message_size_bytes", 2 * ghost_bytes);

      printfQuda(
        "Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for aggregate "
        "message size %lu bytes\n",
        1.0e-9 * 2 * ghost_bytes * niter / dslash_time.event_time,
        1.0e-9 * 2 * ghost_bytes * niter / dslash_time.cpu_time, 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_max,
        1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_min, 2 * ghost_bytes);
    }
  }

  double verify()
  {
    double deviation = 0.0;

    if (test_split_grid) {
      for (int n = 0; n < num_src; n++) {
        double spinor_ref_norm2 = blas::norm2(*spinorRef);
        double spinor_out_norm2 = blas::norm2(*vp_spinor_out[n]);

        bool failed = false;
        // Catching nans is weird.
        if (std::isnan(spinor_ref_norm2)) { failed = true; }
        if (std::isnan(spinor_out_norm2)) { failed = true; }

        printfQuda("Results: CPU=%f, CPU-CUDA=%f\n", spinor_ref_norm2, spinor_out_norm2);
        deviation = std::max(deviation, pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *vp_spinor_out[n]))));
        if (failed) { deviation = 1.0; }
      }
    } else {
      double spinor_ref_norm2 = blas::norm2(*spinorRef);
      double spinor_out_norm2 = blas::norm2(*spinorOut);

      bool failed = false;
      // Catching nans is weird.
      if (std::isnan(spinor_ref_norm2)) { failed = true; }
      if (std::isnan(spinor_out_norm2)) { failed = true; }

      double cuda_spinor_out_norm2 = blas::norm2(*cudaSpinorOut);
      printfQuda("Results: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", spinor_ref_norm2, cuda_spinor_out_norm2, spinor_out_norm2);
      deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
      if (failed) { deviation = 1.0; }
    }

    return deviation;
  }
};
