#pragma once

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include <tune_quda.h>
#include "test.h"

using namespace quda;

dslash_test_type dtest_type = dslash_test_type::Dslash;
CLI::TransformPairs<dslash_test_type> dtest_type_map {
  {"Dslash", dslash_test_type::Dslash},
  {"MatPC", dslash_test_type::MatPC},
  {"Mat", dslash_test_type::Mat},
  {"MatDagMat", dslash_test_type::MatDagMat},
  // left here for completeness but not supported in staggered dslash test
  // {"MatPCDagMatPC", dslash_test_type::MatPCDagMatPC},
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

  static inline QudaGaugeParam gauge_param;
  static inline QudaInvertParam inv_param;
  static inline bool first_time = true;

  static inline std::vector<ColorSpinorField> spinor;
  static inline std::vector<ColorSpinorField> spinorOut;
  static inline std::vector<ColorSpinorField> spinorRef;

  std::vector<ColorSpinorField> cudaSpinor;
  std::vector<ColorSpinorField> cudaSpinorOut;

  static inline std::vector<ColorSpinorField> vp_spinor;
  static inline std::vector<ColorSpinorField> vp_spinor_out;

  static inline void *qdp_inlink[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *qdp_fatlink[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *qdp_longlink[4] = {nullptr, nullptr, nullptr, nullptr};
  static inline void *milc_fatlink = nullptr;
  static inline void *milc_longlink = nullptr;
  static inline GaugeField cpuFat;
  static inline GaugeField cpuLong;

  QudaParity parity = QUDA_EVEN_PARITY;

  Dirac *dirac;

  // Split grid options
  static inline bool test_split_grid = false;
  int num_src = 1;

  void staggeredDslashRef()
  {
    // compare to dslash reference implementation
    printfQuda("Calculating reference implementation...");
    for (int i = 0; i < Nsrc; i++) {
      switch (dtest_type) {
      case dslash_test_type::Dslash:
        stag_dslash(spinorRef[i], cpuFat, cpuLong, spinor[i], parity, dagger, dslash_type, laplace3D);
        break;
      case dslash_test_type::MatPC:
        stag_matpc(spinorRef[i], cpuFat, cpuLong, spinor[i], mass, 0, parity, dslash_type, laplace3D);
        break;
      case dslash_test_type::Mat:
        stag_mat(spinorRef[i], cpuFat, cpuLong, spinor[i], mass, dagger, dslash_type, laplace3D);
        break;
      case dslash_test_type::MatDagMat:
        stag_matdag_mat(spinorRef[i], cpuFat, cpuLong, spinor[i], mass, dagger, dslash_type, laplace3D);
        break;
      default: errorQuda("Test type %d not defined", static_cast<int>(dtest_type));
      }
    }
  }

  void init_ctest(int precision, QudaReconstructType link_recon_)
  {
    gauge_param = newQudaGaugeParam();
    inv_param = newQudaInvertParam();

    setStaggeredGaugeParam(gauge_param);
    setStaggeredInvertParam(inv_param);

    auto prec = getPrecision(precision);

    gauge_param.cuda_prec = prec;
    gauge_param.cuda_prec_sloppy = prec;
    gauge_param.cuda_prec_precondition = prec;
    gauge_param.cuda_prec_refinement_sloppy = prec;

    inv_param.cuda_prec = prec;

    link_recon = link_recon_;

    if (first_time) {
      init_host();
      first_time = false;
    }
    init();
  }

  void init_test()
  {
    gauge_param = newQudaGaugeParam();
    inv_param = newQudaInvertParam();

    setStaggeredGaugeParam(gauge_param);
    setStaggeredInvertParam(inv_param);

    if (first_time) {
      init_host();
      first_time = false;
    }
    init();
  }

  void init_host()
  {
    setDims(gauge_param.X);
    dw_setDims(gauge_param.X, 1);

    for (int i = 0; i < 4; i++) inv_param.split_grid[i] = grid_partition[i];
    num_src = grid_partition[0] * grid_partition[1] * grid_partition[2] * grid_partition[3];
    test_split_grid = num_src > 1;
    if (test_split_grid) { dtest_type = dslash_test_type::Dslash; }

    for (int dir = 0; dir < 4; dir++) {
      qdp_inlink[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_fatlink[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink[dir] = safe_malloc(V * gauge_site_size * host_gauge_data_type_size);
    }

    bool compute_on_gpu = false; // reference fat/long fields should be computed on cpu
    constructStaggeredHostGaugeField(qdp_inlink, qdp_longlink, qdp_fatlink, gauge_param, 0, nullptr, compute_on_gpu);

    // create the reordered MILC-layout fields
    milc_fatlink = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
    milc_longlink = safe_malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

    reorderQDPtoMILC(milc_fatlink, qdp_fatlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);
    reorderQDPtoMILC(milc_longlink, qdp_longlink, V, gauge_site_size, gauge_param.cpu_prec, gauge_param.cpu_prec);

    // create some host-side spinors up front
    ColorSpinorParam csParam;
    csParam.nColor = 3;
    csParam.nSpin = 1;
    csParam.nDim = 4;
    for (int d = 0; d < 4; d++) { csParam.x[d] = gauge_param.X[d]; }
    csParam.x[4] = 1;

    csParam.setPrecision(inv_param.cpu_prec);
    csParam.pad = 0;
    if (dtest_type != dslash_test_type::Mat && dtest_type != dslash_test_type::MatDagMat) {
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
    csParam.location = QUDA_CPU_FIELD_LOCATION;

    spinor.resize(Nsrc);
    spinorOut.resize(Nsrc);
    spinorRef.resize(Nsrc);

    for (auto i = 0; i < Nsrc; i++) {
      spinor[i] = ColorSpinorField(csParam);
      spinorOut[i] = ColorSpinorField(csParam);
      spinorRef[i] = ColorSpinorField(csParam);
      spinor[i].Source(QUDA_RANDOM_SOURCE);
    }

    if (test_split_grid) {
      inv_param.num_src = num_src;
      inv_param.num_src_per_sub_partition = 1;
      resize(vp_spinor, num_src, csParam);
      resize(vp_spinor_out, num_src, csParam);
      std::fill(vp_spinor.begin(), vp_spinor.end(), spinor[0]);
    }

    inv_param.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;

    // set verbosity prior to loadGaugeQuda
    setVerbosity(verbosity);
  }

  void init()
  {

    // For load, etc
    gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;

    // Create ghost gauge fields in case of multi GPU builds.
    gauge_param.type = (dslash_type == QUDA_STAGGERED_DSLASH || dslash_type == QUDA_LAPLACE_DSLASH) ?
      QUDA_SU3_LINKS :
      QUDA_ASQTAD_FAT_LINKS;
    gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
    gauge_param.location = QUDA_CPU_FIELD_LOCATION;

    GaugeFieldParam cpuFatParam(gauge_param, qdp_fatlink);
    cpuFatParam.order = QUDA_QDP_GAUGE_ORDER;
    cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    cpuFat = GaugeField(cpuFatParam);

    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    GaugeFieldParam cpuLongParam(gauge_param, qdp_longlink);
    cpuLongParam.order = QUDA_QDP_GAUGE_ORDER;
    cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    cpuLong = GaugeField(cpuLongParam);

    // Override link reconstruct as appropriate for staggered or asqtad
    if (is_staggered(dslash_type)) {
      if (link_recon == QUDA_RECONSTRUCT_12) link_recon = QUDA_RECONSTRUCT_13;
      if (link_recon == QUDA_RECONSTRUCT_8) link_recon = QUDA_RECONSTRUCT_9;
    }

    loadFatLongGaugeQuda(milc_fatlink, milc_longlink, gauge_param);

    // reset the reconstruct in gauge param
    gauge_param.reconstruct = link_recon;

    // create device-size spinors
    ColorSpinorParam csParam(spinor[0]);
    csParam.fieldOrder = colorspinor::getNative(inv_param.cuda_prec, 1);
    csParam.pad = 0;
    csParam.setPrecision(inv_param.cuda_prec);
    csParam.location = QUDA_CUDA_FIELD_LOCATION;

    cudaSpinor.resize(Nsrc);
    cudaSpinorOut.resize(Nsrc);
    for (auto i = 0; i < Nsrc; i++) {
      cudaSpinor[i] = ColorSpinorField(csParam);
      cudaSpinorOut[i] = ColorSpinorField(csParam);
      cudaSpinor[i] = spinor[i];
    }

    bool pc = (dtest_type == dslash_test_type::MatPC); // For test_type 0, can use either pc or not pc
    // because both call the same "Dslash" directly.
    DiracParam diracParam;
    setDiracParam(diracParam, &inv_param, pc);
    dirac = Dirac::create(diracParam);

  }

  void end()
  {
    if (dirac != nullptr) {
      delete dirac;
      dirac = nullptr;
    }
    freeGaugeQuda();
    cpuFat = {};
    cpuLong = {};
    commDimPartitionedReset();
  }

  static void destroy()
  {
    for (int dir = 0; dir < 4; dir++) {
      if (qdp_inlink[dir]) host_free(qdp_inlink[dir]);
      if (qdp_fatlink[dir]) host_free(qdp_fatlink[dir]);
      if (qdp_longlink[dir]) host_free(qdp_longlink[dir]);
    }

    if (milc_fatlink) {
      host_free(milc_fatlink);
      milc_fatlink = nullptr;
    }

    if (milc_longlink) {
      host_free(milc_longlink);
      milc_longlink = nullptr;
    }

    spinor = {};
    spinorOut = {};
    spinorRef = {};

    if (test_split_grid) {
      vp_spinor.clear();
      vp_spinor_out.clear();
    }
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
        _hp_x[i] = vp_spinor_out[i].data();
        _hp_b[i] = vp_spinor[i].data();
      }
      dslashMultiSrcQuda(_hp_x.data(), _hp_b.data(), &inv_param, parity);

    } else {

      for (int i = 0; i < niter; i++) {

        host_timer.start();

        if (is_laplace(dslash_type)) {
          switch (dtest_type) {
          case dslash_test_type::Mat: dirac->M(cudaSpinorOut, cudaSpinor); break;
          default: errorQuda("Test type %d not defined on Laplace operator", static_cast<int>(dtest_type));
          }
        } else if (is_staggered(dslash_type)) {
          switch (dtest_type) {
          case dslash_test_type::Dslash: dirac->Dslash(cudaSpinorOut, cudaSpinor, parity); break;
          case dslash_test_type::MatPC: dirac->M(cudaSpinorOut, cudaSpinor); break;
          case dslash_test_type::Mat: dirac->M(cudaSpinorOut, cudaSpinor); break;
          case dslash_test_type::MatDagMat: dirac->MdagM(cudaSpinorOut, cudaSpinor); break;
          default: errorQuda("Test type %d not defined on staggered dslash", static_cast<int>(dtest_type));
          }
        } else {
          errorQuda("Invalid dslash type %d", dslash_type);
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

    auto flops0 = quda::Tunable::flops_global();
    auto bytes0 = quda::Tunable::bytes_global();

    DslashTime dslash_time = dslashCUDA(niter);

    unsigned long long flops = (quda::Tunable::flops_global() - flops0);
    unsigned long long bytes = (quda::Tunable::bytes_global() - bytes0);

    for (auto i = 0; i < Nsrc; i++) spinorOut[i] = cudaSpinorOut[i];

    if (print_metrics) {
      printfQuda("%fus per kernel call\n", 1e6 * dslash_time.event_time / niter);

      printfQuda("%llu flops per kernel call, %llu flops per site %llu bytes per site\n", flops / niter,
                 (flops / niter) / cudaSpinor[0].Volume(), (bytes / niter) / cudaSpinor[0].Volume());

      double gflops = 1.0e-9 * flops / dslash_time.event_time;
      printfQuda("GFLOPS = %f\n", gflops);
      ::testing::Test::RecordProperty("Gflops", std::to_string(gflops));

      double gbytes = 1.0e-9 * bytes / dslash_time.event_time;
      printfQuda("GBYTES = %f\n", gbytes);
      ::testing::Test::RecordProperty("Gbytes", std::to_string(gbytes));

      size_t ghost_bytes = cudaSpinor[0].GhostBytes();

      ::testing::Test::RecordProperty("Halo_bidirectional_BW_GPU",
                                      1.0e-9 * 2 * ghost_bytes * niter / dslash_time.event_time);
      ::testing::Test::RecordProperty("Halo_bidirectional_BW_CPU",
                                      1.0e-9 * 2 * ghost_bytes * niter / dslash_time.cpu_time);
      ::testing::Test::RecordProperty("Halo_bidirectional_BW_CPU_min", 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_max);
      ::testing::Test::RecordProperty("Halo_bidirectional_BW_CPU_max", 1.0e-9 * 2 * ghost_bytes / dslash_time.cpu_min);
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
        auto spinor_ref_norm = blas::norm2(spinorRef[0]);
        auto spinor_out_norm = blas::norm2(vp_spinor_out[n]);
        auto max_deviation = blas::max_deviation(spinorRef[0], vp_spinor_out[n]);

        bool failed = false;
        // Catching nans is weird.
        if (std::isnan(spinor_ref_norm)) { failed = true; }
        if (std::isnan(spinor_out_norm)) { failed = true; }

        printfQuda("Results: reference = %f, QUDA = %f, L2 relative deviation = %e, max deviation = %e\n",
                   spinor_ref_norm, spinor_out_norm, 1.0 - sqrt(spinor_out_norm / spinor_ref_norm), max_deviation[0]);
        deviation = std::max(deviation, pow(10.0, -(double)(ColorSpinorField::Compare(spinorRef[0], vp_spinor_out[n]))));
        if (failed) { deviation = 1.0; }
      }
    } else {
      for (int i = 0; i < Nsrc; i++) {
        auto spinor_ref_norm = blas::norm2(spinorRef[i]);
        auto spinor_out_norm = blas::norm2(spinorOut[i]);
        auto max_deviation = blas::max_deviation(spinorRef[i], spinorOut[i]);

        bool failed = false;
        // Catching nans is weird.
        if (std::isnan(spinor_ref_norm)) { failed = true; }
        if (std::isnan(spinor_out_norm)) { failed = true; }

        printfQuda("Results: reference = %f, QUDA = %f, L2 relative deviation = %e, max deviation = %e\n",
                   spinor_ref_norm, spinor_out_norm, 1.0 - sqrt(spinor_out_norm / spinor_ref_norm), max_deviation[0]);
        deviation = pow(10, -(double)(ColorSpinorField::Compare(spinorRef[i], spinorOut[i])));
        if (failed) { deviation = 1.0; }
      }
    }

    return deviation;
  }
};
