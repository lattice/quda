#pragma once

using namespace quda;

#define staggeredSpinorSiteSize 6

dslash_test_type dtest_type = dslash_test_type::Dslash;
CLI::TransformPairs<dslash_test_type> dtest_type_map {{"Dslash", dslash_test_type::Dslash},
                                                      {"MatPC", dslash_test_type::MatPC},
                                                      {"Mat", dslash_test_type::Mat}
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

  DslashTime() : event_time(0.0), cpu_time(0.0), cpu_min(DBL_MAX), cpu_max(0.0) {}
};

struct StaggeredDslashTestWrapper {

  bool is_ctest = false; // Added to distinguish from being used in dslash_test.

  void *qdp_inlink[4] = { nullptr, nullptr, nullptr, nullptr };

  QudaGaugeParam gauge_param;
  QudaInvertParam inv_param;

  cpuGaugeField *cpuFat = nullptr;
  cpuGaugeField *cpuLong = nullptr;

  cpuColorSpinorField *spinor = nullptr;
  cpuColorSpinorField *spinorOut = nullptr;
  cpuColorSpinorField *spinorRef = nullptr;
  cpuColorSpinorField *tmpCpu = nullptr;
  cudaColorSpinorField *cudaSpinor = nullptr;
  cudaColorSpinorField *cudaSpinorOut = nullptr;
  cudaColorSpinorField* tmp = nullptr;

  // In the HISQ case, we include building fat/long links in this unit test
  void *qdp_fatlink_cpu[4] = { nullptr, nullptr, nullptr, nullptr };
  void *qdp_longlink_cpu[4] = { nullptr, nullptr, nullptr, nullptr };
  void **ghost_fatlink_cpu, **ghost_longlink_cpu;

  // To speed up the unit test, build the CPU field once per partition
#ifdef MULTI_GPU
  void *qdp_fatlink_cpu_backup[16][4];
  void *qdp_longlink_cpu_backup[16][4];
  void *qdp_inlink_backup[16][4];
#else
  void *qdp_fatlink_cpu_backup[1][4];
  void *qdp_longlink_cpu_backup[1][4];
  void *qdp_inlink_backup[1][4];
#endif

  QudaParity parity = QUDA_EVEN_PARITY;

  Dirac* dirac;

  // For loading the gauge fields
  int argc_copy;
  char** argv_copy;

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
      default:
        errorQuda("Test type not defined");
    }
  }

  void init_ctest_once()
  {
    static bool has_been_called = false;
    if (has_been_called) { errorQuda("This function is not supposed to be called twice.\n"); }
    // initialize CPU field backup
    int pmax = 1;
#ifdef MULTI_GPU
    pmax = 16;
#endif
    for (int p = 0; p < pmax; p++) {
      for (int d = 0; d < 4; d++) {
        qdp_fatlink_cpu_backup[p][d] = nullptr;
        qdp_longlink_cpu_backup[p][d] = nullptr;
        qdp_inlink_backup[p][d] = nullptr;
      }
    }
    is_ctest = true; // Is being used in dslash_ctest.
    has_been_called = true;
  }

  void end_ctest_once()
  {
    static bool has_been_called = false;
    if (has_been_called) { errorQuda("This function is not supposed to be called twice.\n"); }
    // Clean up per-partition backup
    int pmax = 1;
#ifdef MULTI_GPU
    pmax = 16;
#endif
    for (int p = 0; p < pmax; p++) {
      for (int d = 0; d < 4; d++) {
        if (qdp_inlink_backup[p][d] != nullptr) { free(qdp_inlink_backup[p][d]); qdp_inlink_backup[p][d] = nullptr; }
        if (qdp_fatlink_cpu_backup[p][d] != nullptr) {
          free(qdp_fatlink_cpu_backup[p][d]);
          qdp_fatlink_cpu_backup[p][d] = nullptr;
        }
        if (qdp_longlink_cpu_backup[p][d] != nullptr) {
          free(qdp_longlink_cpu_backup[p][d]);
          qdp_longlink_cpu_backup[p][d] = nullptr;
        }
      }
    }
    has_been_called = true;
  }

  void init_ctest(int precision, QudaReconstructType link_recon_, int partition)
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
    inv_param.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;

    setDims(gauge_param.X);
    dw_setDims(gauge_param.X, 1);
    if (Nsrc != 1) {
      warningQuda("Ignoring Nsrc = %d, setting to 1.", Nsrc);
      Nsrc = 1;
    }
    setSpinorSiteSize(staggeredSpinorSiteSize);

    // Allocate a lot of memory because I'm very confused
    void *milc_fatlink_cpu = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
    void *milc_longlink_cpu = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

    void *milc_fatlink_gpu = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);
    void *milc_longlink_gpu = malloc(4 * V * gauge_site_size * host_gauge_data_type_size);

    void* qdp_fatlink_gpu[4];
    void* qdp_longlink_gpu[4];

    for (int dir = 0; dir < 4; dir++) {
      qdp_fatlink_gpu[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink_gpu[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);

      qdp_fatlink_cpu[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);
      qdp_longlink_cpu[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size);

      if (qdp_fatlink_gpu[dir] == NULL || qdp_longlink_gpu[dir] == NULL ||
          qdp_fatlink_cpu[dir] == NULL || qdp_longlink_cpu[dir] == NULL) {
        errorQuda("ERROR: malloc failed for fatlink/longlink");
      }
    }

    // create a base field
    for (int dir = 0; dir < 4; dir++) {
      if (qdp_inlink[dir] == nullptr) { qdp_inlink[dir] = malloc(V * gauge_site_size * host_gauge_data_type_size); }
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
    GaugeFieldParam cpuFatParam(milc_fatlink_cpu, gauge_param);
    cpuFatParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    cpuFat = new cpuGaugeField(cpuFatParam);
    ghost_fatlink_cpu = cpuFat->Ghost();

    gauge_param.type = QUDA_ASQTAD_LONG_LINKS;
    GaugeFieldParam cpuLongParam(milc_longlink_cpu, gauge_param);
    cpuLongParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    cpuLong = new cpuGaugeField(cpuLongParam);
    ghost_longlink_cpu = cpuLong->Ghost();
#endif

    gauge_param.type = (dslash_type == QUDA_ASQTAD_DSLASH) ? QUDA_ASQTAD_FAT_LINKS : QUDA_SU3_LINKS;
    if (dslash_type == QUDA_STAGGERED_DSLASH) {
      gauge_param.reconstruct = gauge_param.reconstruct_sloppy = (link_recon == QUDA_RECONSTRUCT_12) ?
        QUDA_RECONSTRUCT_13 :
        (link_recon == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_9 : link_recon;
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
        (link_recon == QUDA_RECONSTRUCT_8) ? QUDA_RECONSTRUCT_9 : link_recon;
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
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    csParam.pad = 0;
    if (dtest_type != dslash_test_type::Mat && dslash_type != QUDA_LAPLACE_DSLASH) {
      csParam.siteSubset = QUDA_PARITY_SITE_SUBSET;
      csParam.x[0] /= 2;
    } else {
      csParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    }

    csParam.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    csParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    csParam.gammaBasis = inv_param.gamma_basis; // this parameter is meaningless for staggered
    csParam.create = QUDA_ZERO_FIELD_CREATE;

    spinor = new cpuColorSpinorField(csParam);
    spinorOut = new cpuColorSpinorField(csParam);
    spinorRef = new cpuColorSpinorField(csParam);
    tmpCpu = new cpuColorSpinorField(csParam);

    spinor->Source(QUDA_RANDOM_SOURCE);

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
      free(qdp_fatlink_gpu[dir]); qdp_fatlink_gpu[dir] = nullptr;
      free(qdp_longlink_gpu[dir]); qdp_longlink_gpu[dir] = nullptr;
    }
    free(milc_fatlink_gpu); milc_fatlink_gpu = nullptr;
    free(milc_longlink_gpu); milc_longlink_gpu = nullptr;
    free(milc_fatlink_cpu); milc_fatlink_cpu = nullptr;
    free(milc_longlink_cpu); milc_longlink_cpu = nullptr;

    gauge_param.reconstruct = link_recon;
  }

  void end()
  {
    for (int dir = 0; dir < 4; dir++) {
      if (qdp_fatlink_cpu[dir] != nullptr) { free(qdp_fatlink_cpu[dir]); qdp_fatlink_cpu[dir] = nullptr; }
      if (qdp_longlink_cpu[dir] != nullptr) { free(qdp_longlink_cpu[dir]); qdp_longlink_cpu[dir] = nullptr; }
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

    if (spinor != nullptr) { delete spinor; spinor = nullptr; }
    if (spinorOut != nullptr) { delete spinorOut; spinorOut = nullptr; }
    if (spinorRef != nullptr) { delete spinorRef; spinorRef = nullptr; }
    if (tmpCpu != nullptr) { delete tmpCpu; tmpCpu = nullptr; }

    freeGaugeQuda();

    if (cpuFat) { delete cpuFat; cpuFat = nullptr; }
    if (cpuLong) { delete cpuLong; cpuLong = nullptr; }
    commDimPartitionedReset();

  }

  DslashTime dslashCUDA(int niter) {

    DslashTime dslash_time;
    timeval tstart, tstop;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    comm_barrier();
    cudaEventRecord(start, 0);

    for (int i = 0; i < niter; i++) {

      gettimeofday(&tstart, NULL);

      switch (dtest_type) {
        case dslash_test_type::Dslash: dirac->Dslash(*cudaSpinorOut, *cudaSpinor, parity); break;
        case dslash_test_type::MatPC: dirac->M(*cudaSpinorOut, *cudaSpinor); break;
        case dslash_test_type::Mat: dirac->M(*cudaSpinorOut, *cudaSpinor); break;
        default: errorQuda("Test type %d not defined on staggered dslash.\n", static_cast<int>(dtest_type));
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

    cudaEventCreate(&end);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float runTime;
    cudaEventElapsedTime(&runTime, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    dslash_time.event_time = runTime / 1000;

    return dslash_time;
  }

};
