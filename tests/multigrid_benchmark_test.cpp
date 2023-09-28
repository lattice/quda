#include <stdio.h>
#include <stdlib.h>

#include <quda_internal.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <host_utils.h>
#include <command_line_params.h>
#include <misc.h>

// include because of nasty globals used in the tests
#include <dslash_reference.h>
#include <dirac_quda.h>
#include <gauge_tools.h>
#include <gtest/gtest.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

extern void usage(char **);

using namespace quda;

std::vector<ColorSpinorField> xD, yD;

std::shared_ptr<cudaGaugeField> Y_d, X_d, Xinv_d, Yhat_d;

int Ncolor;

#define MAX(a, b) ((a) > (b) ? (a) : (b))

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("S_dimension T_dimension Ncolor Nsrc\n");
  printfQuda("%3d /%3d / %3d   %3d      %d     %d\n", xdim, ydim, zdim, tdim, Ncolor, Nsrc);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

void initFields(QudaPrecision prec)
{
  ColorSpinorParam param;
  param.nColor = Ncolor;
  param.nSpin = 2;
  param.nDim = 4; // number of spacetime dimensions

  param.pad = 0; // padding must be zero for cpu fields
  param.siteSubset = QUDA_FULL_SITE_SUBSET;
  param.x[0] = xdim;
  param.x[1] = ydim;
  param.x[2] = zdim;
  param.x[3] = tdim;
  param.x[4] = 1;
  param.pc_type = QUDA_4D_PC;

  param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  param.create = QUDA_ZERO_FIELD_CREATE;
  param.setPrecision(prec, prec, true);
  param.location = QUDA_CUDA_FIELD_LOCATION;

  resize(xD, Nsrc, param);
  resize(yD, Nsrc, param);

  GaugeFieldParam gParam;
  gParam.x[0] = xdim;
  gParam.x[1] = ydim;
  gParam.x[2] = zdim;
  gParam.x[3] = tdim;
  gParam.nColor = param.nColor * param.nSpin;
  gParam.reconstruct = QUDA_RECONSTRUCT_NO;
  gParam.order = QUDA_QDP_GAUGE_ORDER;
  gParam.link_type = QUDA_COARSE_LINKS;
  gParam.t_boundary = QUDA_PERIODIC_T;
  gParam.create = QUDA_ZERO_FIELD_CREATE;
  gParam.setPrecision(param.Precision());
  gParam.nDim = 4;
  gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.nFace = 1;
  gParam.order = QUDA_FLOAT2_GAUGE_ORDER;
  gParam.geometry = QUDA_COARSE_GEOMETRY;
  gParam.nFace = 1;

  int x_face_size = gParam.x[1] * gParam.x[2] * gParam.x[3] / 2;
  int y_face_size = gParam.x[0] * gParam.x[2] * gParam.x[3] / 2;
  int z_face_size = gParam.x[0] * gParam.x[1] * gParam.x[3] / 2;
  int t_face_size = gParam.x[0] * gParam.x[1] * gParam.x[2] / 2;
  int pad = MAX(x_face_size, y_face_size);
  pad = MAX(pad, z_face_size);
  pad = MAX(pad, t_face_size);
  gParam.pad = gParam.nFace * pad * 2;

  gParam.setPrecision(prec_sloppy);
  gParam.location = QUDA_CUDA_FIELD_LOCATION;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

  Y_d = std::make_shared<cudaGaugeField>(gParam);
  Yhat_d = std::make_shared<cudaGaugeField>(gParam);

  gParam.geometry = QUDA_SCALAR_GEOMETRY;
  gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
  gParam.nFace = 0;
  X_d = std::make_shared<cudaGaugeField>(gParam);
  Xinv_d = std::make_shared<cudaGaugeField>(gParam);

  // insert random noise into the gauge fields
  {
    quda::RNG rng(xD[0], 1234);
    for (auto &yi : yD) spinorNoise(yi, rng, QUDA_NOISE_GAUSS);

    gaugeNoise(*Y_d, rng, QUDA_NOISE_GAUSS);
    gaugeNoise(*Yhat_d, rng, QUDA_NOISE_GAUSS);
    gaugeNoise(*X_d, rng, QUDA_NOISE_GAUSS);
    gaugeNoise(*Xinv_d, rng, QUDA_NOISE_GAUSS);
  }
}

void freeFields()
{
  xD.clear();
  yD.clear();

  Y_d.reset();
  X_d.reset();
  Xinv_d.reset();
  Yhat_d.reset();
}

DiracCoarse *dirac;
DiracCoarsePC *dirac_pc;

TEST(multi_rhs_test, verify)
{
  printfQuda("\nTesting Multi-RHS correctness...\n\n");

  blas::zero(xD);

  auto xEven = make_parity_subset(xD, QUDA_EVEN_PARITY);
  auto yEven = make_parity_subset(yD, QUDA_EVEN_PARITY);
  auto yOdd = make_parity_subset(yD, QUDA_ODD_PARITY);

  switch (test_type) {
  case 0: dirac->Dslash(xEven, yOdd, QUDA_EVEN_PARITY); break;
  case 1: dirac->M(xD, yD); break;
  case 2: dirac->Clover(xEven, yEven, QUDA_EVEN_PARITY); break;
  case 3: dirac->Mdag(xD, yD); break;
  case 4: dirac->MdagM(xD, yD); break;
  case 5: dirac_pc->M(xEven, yOdd); break;
  case 6: dirac_pc->Mdag(xEven, yOdd); break;
  case 7: dirac_pc->MdagM(xEven, yOdd); break;
  default: errorQuda("Undefined test %d", test_type);
  }

  ColorSpinorField x_ref(yD[0]);
  blas::zero(x_ref);

  for (auto i = 0u; i < yD.size(); i++) {
    switch (test_type) {
    case 0: dirac->Dslash(x_ref.Even(), yD[i].Odd(), QUDA_EVEN_PARITY); break;
    case 1: dirac->M(x_ref, yD[i]); break;
    case 2: dirac->Clover(x_ref.Even(), yD[i].Even(), QUDA_EVEN_PARITY); break;
    case 3: dirac->Mdag(x_ref, yD[i]); break;
    case 4: dirac->MdagM(x_ref, yD[i]); break;
    case 5: dirac_pc->M(x_ref.Even(), yD[i].Odd()); break;
    case 6: dirac_pc->Mdag(x_ref.Even(), yD[i].Odd()); break;
    case 7: dirac_pc->MdagM(x_ref.Even(), yD[i].Odd()); break;
    default: errorQuda("Undefined test %d", test_type);
    }

    auto max_dev = blas::max_deviation(xD[i], x_ref);
    auto x2 = blas::norm2(x_ref);
    auto l2_dev = blas::xmyNorm(xD[i], x_ref);

    // require that the relative L2 norm differs by no more than 2e-6/4e-5
    EXPECT_LE(sqrt(l2_dev / x2), prec_sloppy == QUDA_SINGLE_PRECISION ? 2e-6 : 4e-5);
    // require that each component differs by no more than 1e-3/4e-3
    EXPECT_LE(max_dev[1], prec_sloppy == QUDA_SINGLE_PRECISION ? 1e-3 : 4e-3);
  }
}

double benchmark(int test, const int niter)
{
  printfQuda("\nBenchmarking %s precision with %d iterations...\n\n", get_prec_str(prec), niter);

  device_timer_t device_timer;
  device_timer.start();

  auto xEven = make_parity_subset(xD, QUDA_EVEN_PARITY);
  auto yEven = make_parity_subset(yD, QUDA_EVEN_PARITY);
  auto yOdd = make_parity_subset(yD, QUDA_ODD_PARITY);

  switch (test) {
  case 0:
    for (int i = 0; i < niter; ++i) dirac->Dslash(xEven, yOdd, QUDA_EVEN_PARITY);
    break;
  case 1:
    for (int i = 0; i < niter; ++i) dirac->M(xD, yD);
    break;
  case 2:
    for (int i = 0; i < niter; ++i) dirac->Clover(xEven, yEven, QUDA_EVEN_PARITY);
    break;
  case 3:
    for (int i = 0; i < niter; ++i) dirac->Mdag(xD, yD);
    break;
  case 4:
    for (int i = 0; i < niter; ++i) dirac->MdagM(xD, yD);
    break;
  case 5:
    for (int i = 0; i < niter; ++i) dirac_pc->M(xEven, yOdd);
    break;
  case 6:
    for (int i = 0; i < niter; ++i) dirac_pc->Mdag(xEven, yOdd);
    break;
  case 7:
    for (int i = 0; i < niter; ++i) dirac_pc->MdagM(xEven, yOdd);
    break;
  default: errorQuda("Undefined test %d", test);
  }

  device_timer.stop();
  return device_timer.last();
}

const char *names[] = {"Dslash", "Mat", "Clover", "MatDag", "MatDagMat", "MatPC", "MatPCDag", "MatPCDagMatPC"};

int main(int argc, char **argv)
{
  // initalize google test
  ::testing::InitGoogleTest(&argc, argv);
  // return code for google test
  int test_rc = 0;

  // Set some defaults that lets the benchmark fit in memory if you run it
  // with default parameters.
  xdim = ydim = zdim = tdim = 8;

  // command line options
  auto app = make_app();
  add_multigrid_option_group(app);
  CLI::TransformPairs<int> test_type_map {{"Dslash", 0},    {"Mat", 1},   {"Clover", 2},   {"MatDag", 3},
                                          {"MatDagMat", 4}, {"MatPC", 5}, {"MatPCDag", 6}, {"MatPCDagMatPC", 7}};
  app->add_option("--test", test_type, "Test method")->transform(CLI::CheckedTransformer(test_type_map));

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  Ncolor = nvec[0] == 0 ? 24 : nvec[0];

  initComms(argc, argv, gridsize_from_cmdline);
  display_test_info();
  initQuda(device_ordinal);

  setVerbosity(verbosity);

  initFields(prec);

  DiracParam param;
  param.halo_precision = smoother_halo_prec;
  param.kappa = 1.0;
  param.dagger = QUDA_DAG_NO;
  param.setup_use_mma = mg_setup_use_mma[0];
  param.dslash_use_mma = mg_dslash_use_mma[0];
  param.matpcType = QUDA_MATPC_EVEN_EVEN;
  dirac = new DiracCoarse(param, nullptr, nullptr, nullptr, nullptr, Y_d, X_d, Xinv_d, Yhat_d);
  dirac_pc = new DiracCoarsePC(param, nullptr, nullptr, nullptr, nullptr, Y_d, X_d, Xinv_d, Yhat_d);

  if (verify_results) {
    // Ensure gtest prints only from rank 0
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

    test_rc = RUN_ALL_TESTS();
    if (test_rc != 0) warningQuda("Tests failed");
  }

  // now rerun with more iterations to get accurate speed measurements
  dirac->Flops();    // reset flops counter
  dirac_pc->Flops(); // reset flops counter

  double secs = benchmark(test_type, niter);
  double gflops = ((test_type < 5 ? dirac->Flops() : dirac_pc->Flops()) * 1e-9) / (secs);

  printfQuda("Ncolor = %2d, %-31s: Gflop/s = %6.1f\n", Ncolor, names[test_type], gflops);

  delete dirac;
  delete dirac_pc;
  freeFields();

  endQuda();

  finalizeComms();
  return test_rc;
}
