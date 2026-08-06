#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <baryon_contract_reference.h>
#include "misc.h"

// google test
#include <gtest/gtest.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("baryonContractFTQuda test\n");
  printfQuda("prec    S_dimension T_dimension\n");
  printfQuda("%s   %d/%d/%d          %d\n", get_prec_str(prec), xdim, ydim, zdim, tdim);
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

// Fill a set of buffers with pseudo-random data, seeded by global site
// index so the result is invariant of the process grid.
template <typename Float, int N = 2>
void fill_buffers(std::array<std::vector<Float>, N> &buffs, const std::array<int, 4> &X, const int dofs)
{
  const std::array<int, 4> X0 = {X[0] * comm_coord(0), X[1] * comm_coord(1), X[2] * comm_coord(2), X[3] * comm_coord(3)};
  const std::array<int, 4> XN = {X[0] * comm_dim(0), X[1] * comm_dim(1), X[2] * comm_dim(2), X[3] * comm_dim(3)};

  for (int ix = 0; ix < X[0]; ix++) {
    for (int iy = 0; iy < X[1]; iy++) {
      for (int iz = 0; iz < X[2]; iz++) {
        for (int it = 0; it < X[3]; it++) {

          int l
            = (ix + X0[0]) + (iy + X0[1]) * XN[0] + (iz + X0[2]) * XN[0] * XN[1] + (it + X0[3]) * XN[0] * XN[1] * XN[2];
          int ll = ix + iy * X[0] + iz * X[0] * X[1] + it * X[0] * X[1] * X[2];

          srand(l);
          for (int i = 0; i < dofs; i++) {
            for (int n = 0; n < N; n++) { buffs[n][ll * dofs + i] = 2. * (rand() / (Float)RAND_MAX) - 1.; }
          }
        }
      }
    }
  }
}

template <typename Float, int n_mom>
int launch_baryon_test(const std::array<int, 4> &X, const std::array<int, 4> &source_position,
                       const std::array<int, n_mom * 4> &mom, const std::array<QudaFFTSymmType, n_mom * 4> &fft_type)
{
  constexpr int nSpin = 4;
  constexpr int nprops = nSpin * 3;

  ColorSpinorParam cs_param;
  cs_param.nColor = 3;
  cs_param.nSpin = nSpin;
  cs_param.nDim = 4;
  for (int i = 0; i < 4; i++) cs_param.x[i] = X[i];
  cs_param.x[4] = 1;
  cs_param.siteSubset = QUDA_FULL_SITE_SUBSET;
  cs_param.setPrecision(sizeof(Float) == sizeof(float) ? QUDA_SINGLE_PRECISION : QUDA_DOUBLE_PRECISION);
  cs_param.pad = 0;
  cs_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
  cs_param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
  cs_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  cs_param.create = QUDA_ZERO_FIELD_CREATE;
  cs_param.location = QUDA_CPU_FIELD_LOCATION;
  cs_param.pc_type = QUDA_4D_PC;

  const int spinor_field_floats = V * nSpin * 3 * 2;
  const int dof = nSpin * 3 * 2 * nprops;

  const int red_size = comm_dim(3) * X[3];
  const int n_contract_results = red_size * n_mom * nSpin * nSpin * 2;
  std::vector<double> d_result(n_contract_results, 0.0);

  std::array<std::vector<Float>, 2> buffs {std::vector<Float>(nprops * spinor_field_floats, 0),
                                           std::vector<Float>(nprops * spinor_field_floats, 0)};
  fill_buffers<Float, 2>(buffs, X, dof);

  std::array<void *, nprops> prop_u;
  std::array<void *, nprops> prop_d;
  size_t off = 0;
  for (int s = 0; s < nprops; ++s, off += spinor_field_floats) {
    prop_u[s] = static_cast<void *>(buffs[0].data() + off);
    prop_d[s] = static_cast<void *>(buffs[1].data() + off);
  }

  // Perform GPU contraction:
  void *d_result_ = static_cast<void *>(d_result.data());
  baryonContractFTQuda(prop_u.data(), prop_d.data(), &d_result_, QUDA_CONTRACT_TYPE_BARYON_NUCLEON_FT_T,
                       (void *)(&cs_param), X.data(), source_position.data(), n_mom, mom.data(), fft_type.data());

  // Check results:
  int faults = baryon_ref::baryonContractFT_reference<Float>(prop_u.data(), prop_d.data(), d_result.data(), X.data(),
                                                             source_position.data(), n_mom, mom.data(), fft_type.data());

  printfQuda("Baryon contraction comparison complete with %d/%d faults\n", faults, n_contract_results);

  return faults;
}

int run_baryon_test(QudaPrecision test_prec)
{
  if (xdim % 2) errorQuda("odd local x-dimension is not supported");

  const std::array<int, 4> X = {xdim, ydim, zdim, tdim};
  const std::array<int, 4> source_position = prop_source_position[0];

  const QudaFFTSymmType eo = QUDA_FFT_SYMM_EO;
  const QudaFFTSymmType ev = QUDA_FFT_SYMM_EVEN;
  const QudaFFTSymmType od = QUDA_FFT_SYMM_ODD;

  constexpr int n_mom = 4;
  const std::array<int, n_mom * 4> mom = {0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0};
  const std::array<QudaFFTSymmType, n_mom * 4> fft_type = {
    eo, eo, eo, eo, // (0,0,0) exp
    eo, eo, eo, eo, // (1,0,0) exp
    ev, ev, ev, eo, // (0,1,0) cos
    ev, od, ev, eo, // (0,1,1) mixed
  };

  int faults = 0;
  if (test_prec == QUDA_SINGLE_PRECISION) {
    faults = launch_baryon_test<float, n_mom>(X, source_position, mom, fft_type);
  } else if (test_prec == QUDA_DOUBLE_PRECISION) {
    faults = launch_baryon_test<double, n_mom>(X, source_position, mom, fft_type);
  } else {
    errorQuda("Unsupported precision %d", test_prec);
  }

  return faults;
}

TEST(BaryonContractFT, ChargeConjugationConvention)
{
  // C = gamma_4 gamma_2 must satisfy C gamma_mu C^{-1} = -gamma_mu^T in
  // the DeGrand-Rossi basis; the contraction conventions rest on this.
  EXPECT_LT(baryon_ref::check_charge_conjugation(), 1e-14);
}

TEST(BaryonContractFT, DoublePrecision) { EXPECT_EQ(run_baryon_test(QUDA_DOUBLE_PRECISION), 0); }

TEST(BaryonContractFT, SinglePrecision) { EXPECT_EQ(run_baryon_test(QUDA_SINGLE_PRECISION), 0); }

int main(int argc, char **argv)
{
  // Start Google Test Suite
  ::testing::InitGoogleTest(&argc, argv);

  // command line options
  auto app = make_app();
  add_propagator_option_group(app);
  add_testing_option_group(app);
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  display_test_info();

  // Initialize the QUDA library
  initQuda(device_ordinal);

  // Ensure gtest prints only from rank 0
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }

  // call srand() with a rank-dependent seed
  initRand();

  std::array<int, 4> X = {xdim, ydim, zdim, tdim}; // local dims
  setDims(X.data());

  int result = RUN_ALL_TESTS();

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return result;
}
