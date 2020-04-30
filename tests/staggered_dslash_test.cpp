#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include <quda.h>
#include <quda_internal.h>
#include <dirac_quda.h>
#include <dslash_quda.h>
#include <invert_quda.h>
#include <util_quda.h>
#include <blas_quda.h>

#include <misc.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <staggered_dslash_reference.h>
#include <staggered_gauge_utils.h>
#include <gauge_field.h>
#include <unitarization_links.h>

#include "dslash_test_helpers.h"
#include <assert.h>
#include <gtest/gtest.h>

using namespace quda;

#define staggeredSpinorSiteSize 6

void *qdp_inlink[4] = { nullptr, nullptr, nullptr, nullptr };

QudaGaugeParam gauge_param;
QudaInvertParam inv_param;

cpuGaugeField *cpuFat = nullptr;
cpuGaugeField *cpuLong = nullptr;

cpuColorSpinorField *spinor, *spinorOut, *spinorRef, *tmpCpu;
cudaColorSpinorField *cudaSpinor, *cudaSpinorOut;
cudaColorSpinorField* tmp;

// In the HISQ case, we include building fat/long links in this unit test
void *qdp_fatlink_cpu[4], *qdp_longlink_cpu[4];
void **ghost_fatlink_cpu, **ghost_longlink_cpu;

QudaParity parity = QUDA_EVEN_PARITY;

Dirac* dirac;

// For loading the gauge fields
int argc_copy;
char** argv_copy;

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

void init()
{
  initQuda(device);

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();

  setStaggeredGaugeParam(gauge_param);
  setStaggeredInvertParam(inv_param);
  inv_param.dagger = dagger ? QUDA_DAG_YES : QUDA_DAG_NO;

  setDims(gauge_param.X);
  dw_setDims(gauge_param.X, Nsrc); // so we can use 5-d indexing from dwf
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
  csParam.x[4] = Nsrc; // number of sources becomes the fifth dimension

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

  cudaDeviceSynchronize();
  checkCudaError();

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

  endQuda();
}

struct DslashTime {
  double event_time;
  double cpu_time;
  double cpu_min;
  double cpu_max;

  DslashTime() : event_time(0.0), cpu_time(0.0), cpu_min(DBL_MAX), cpu_max(0.0) {}
};

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

  // check for errors
  cudaError_t stat = cudaGetLastError();
  if (stat != cudaSuccess)
    errorQuda("with ERROR: %s\n", cudaGetErrorString(stat));

  return dslash_time;
}

void staggeredDslashRef()
{

  // compare to dslash reference implementation
  // printfQuda("Calculating reference implementation...");
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
      // Not sure about the !dagger...
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

TEST(dslash, verify) {
  double deviation = pow(10, -(double)(cpuColorSpinorField::Compare(*spinorRef, *spinorOut)));
  double tol = getTolerance(prec);
  ASSERT_LE(deviation, tol) << "CPU and CUDA implementations do not agree";
}

static int dslashTest()
{

  for (int dir = 0; dir < 4; dir++) {
    qdp_fatlink_cpu[dir] = nullptr;
    qdp_longlink_cpu[dir] = nullptr;
  }

  dirac = nullptr;
  cudaSpinor = nullptr;
  cudaSpinorOut = nullptr;
  tmp = nullptr;

  spinor = nullptr;
  spinorOut = nullptr;
  spinorRef = nullptr;
  tmpCpu = nullptr;

  bool failed = false;

  // return code for google test
  int test_rc = 0;
  init();

  int attempts = 1;

  for (int i=0; i<attempts; i++) {

    { // warm-up run
      printfQuda("Tuning...\n");
      dslashCUDA(1);
    }
    printfQuda("Executing %d kernel loops...", niter);

    // reset flop counter
    dirac->Flops();

    DslashTime dslash_time = dslashCUDA(niter);

    *spinorOut = *cudaSpinorOut;

    printfQuda("%fus per kernel call\n", 1e6*dslash_time.event_time / niter);
    staggeredDslashRef();

    double spinor_ref_norm2 = blas::norm2(*spinorRef);
    double spinor_out_norm2 = blas::norm2(*spinorOut);

    // Catching nans is weird.
    if (std::isnan(spinor_ref_norm2)) { failed = true; }
    if (std::isnan(spinor_out_norm2)) { failed = true; }

    unsigned long long flops = dirac->Flops();
    printfQuda("GFLOPS = %f\n", 1.0e-9*flops/dslash_time.event_time);

    if (niter > 2) { // only print this if valid
      printfQuda("Effective halo bi-directional bandwidth (GB/s) GPU = %f ( CPU = %f, min = %f , max = %f ) for "
                 "aggregate message size %lu bytes\n",
                 1.0e-9 * 2 * cudaSpinor->GhostBytes() * niter / dslash_time.event_time,
                 1.0e-9 * 2 * cudaSpinor->GhostBytes() * niter / dslash_time.cpu_time,
                 1.0e-9 * 2 * cudaSpinor->GhostBytes() / dslash_time.cpu_max,
                 1.0e-9 * 2 * cudaSpinor->GhostBytes() / dslash_time.cpu_min, 2 * cudaSpinor->GhostBytes());
    }

    double cuda_spinor_out_norm2 = blas::norm2(*cudaSpinorOut);
    printfQuda("Results: CPU=%f, CUDA=%f, CPU-CUDA=%f\n", spinor_ref_norm2, cuda_spinor_out_norm2, spinor_out_norm2);

    if (verify_results) {
      test_rc = RUN_ALL_TESTS();
      if (test_rc != 0 || failed) warningQuda("Tests failed");
    }
  }
  end();

  return test_rc;
}

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("prec recon   test_type     dagger   S_dim         T_dimension\n");
  printfQuda("%s   %s       %s           %d       %d/%d/%d        %d \n", get_prec_str(prec), get_recon_str(link_recon),
      get_string(dtest_type_map, dtest_type).c_str(), dagger, xdim, ydim, zdim, tdim);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
      dimPartitioned(3));
}

int main(int argc, char **argv)
{
  // hack for loading gauge fields
  argc_copy = argc;
  argv_copy = argv;

  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);

  // command line options
  auto app = make_app();
  app->add_option("--test", dtest_type, "Test method")->transform(CLI::CheckedTransformer(dtest_type_map));
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  initComms(argc, argv, gridsize_from_cmdline);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  // Only these fermions are supported in this file. Ensure a reasonable default,
  // ensure that the default is improved staggered
  if (dslash_type != QUDA_STAGGERED_DSLASH && dslash_type != QUDA_ASQTAD_DSLASH && dslash_type != QUDA_LAPLACE_DSLASH) {
    printfQuda("dslash_type %s not supported, defaulting to %s\n", get_dslash_str(dslash_type),
               get_dslash_str(QUDA_ASQTAD_DSLASH));
    dslash_type = QUDA_ASQTAD_DSLASH;
  }

  // Sanity check: if you pass in a gauge field, want to test the asqtad/hisq dslash,
  // and don't ask to build the fat/long links... it doesn't make sense.
  if (strcmp(latfile,"") && !compute_fatlong && dslash_type == QUDA_ASQTAD_DSLASH) {
    errorQuda("Cannot load a gauge field and test the ASQTAD/HISQ operator without setting \"--compute-fat-long true\".\n");
  }

  // Set n_naiks to 2 if eps_naik != 0.0
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    if (eps_naik != 0.0) {
      if (compute_fatlong) {
        n_naiks = 2;
        printfQuda("Note: epsilon-naik != 0, testing epsilon correction links.\n");
      } else {
        eps_naik = 0.0;
        printfQuda("Not computing fat-long, ignoring epsilon correction.\n");
      }
    } else {
      printfQuda("Note: epsilon-naik = 0, testing original HISQ links.\n");
    }
  }

  if (dslash_type == QUDA_LAPLACE_DSLASH) {
    if (dtest_type != dslash_test_type::Mat) {
      errorQuda("Test type %s is not supported for the Laplace operator.\n", get_string(dtest_type_map, dtest_type).c_str());
    }
  }

  // If we're building fat/long links, there are some
  // tests we have to skip.
  if (dslash_type == QUDA_ASQTAD_DSLASH && compute_fatlong) {
    if (prec == QUDA_HALF_PRECISION /* half */) {
      errorQuda("Half precision unsupported in fat/long compute");
    }
  }
  if (dslash_type == QUDA_LAPLACE_DSLASH && prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision unsupported for Laplace operator.\n");
  }

  display_test_info();

  // return result of RUN_ALL_TESTS
  int test_rc = dslashTest();

  finalizeComms();

  return test_rc;
}
