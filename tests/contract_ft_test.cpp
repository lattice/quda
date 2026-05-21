#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>
#include <dslash_reference.h>
#include <contract_ft_reference.h>
#include "misc.h"
#include "test.h"

// google test
#include <contract_ft_test_gtest.hpp>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>
#include <color_spinor_field.h>

void display_test_info()
{
  printfQuda("running the following test:\n");

  printfQuda("contraction_type prec    S_dimension T_dimension Ls_dimension\n");
  printfQuda("%s             %s   %d/%d/%d          %d         %d\n", get_contract_str(contract_type),
             get_prec_str(prec), xdim, ydim, zdim, tdim, Lsdim);

  printfQuda("contractFTQuda test");
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
  return;
}

int main(int argc, char **argv)
{
  // Start Google Test Suite
  ::testing::InitGoogleTest(&argc, argv);

  // QUDA initialise
  // command line options:
  auto app = make_app();

  add_propagator_option_group(app);
  add_contraction_option_group(app);
  add_testing_option_group(app);

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  // Set values for precisions via the command line.
  setQudaPrecisions();

  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);

  // All parameters have been set. Display the parameters via stdout
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

  // Check for correctness:
  int result = 0;

  if (enable_testing) { // tests are defined in invert_test_gtest.hpp
    ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (quda::comm_rank() != 0) { delete listeners.Release(listeners.default_result_printer()); }
    result = RUN_ALL_TESTS();
  } else { //
    contract(test_t {contract_type, prec});
  }

  // finalize the QUDA library
  endQuda();

  // finalize the communications layer
  finalizeComms();

  return result;
}

template <typename Float, int N = 2>
inline void fill_buffers(std::array<std::vector<Float>, N> &buffs, const std::array<int, 4> &X, const int dofs)
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

template <typename Float, int nSpin, int src_colors, int n_mom>
inline int launch_contract_test(const QudaContractType cType, const std::array<int, 4> &X, const int red_size,
                                const std::array<int, 4> &source_position, const std::array<int, n_mom * 4> &mom,
                                const std::array<QudaFFTSymmType, n_mom * 4> &fft_type)
{
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
  cs_param.gammaBasis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS; // meaningless for staggered, but required by the code.
  cs_param.create = QUDA_ZERO_FIELD_CREATE;
  cs_param.location = QUDA_CPU_FIELD_LOCATION;
  cs_param.pc_type = QUDA_4D_PC;

  const int my_spinor_site_size = nSpin * 3; // nSpin X nColor

  const int spinor_field_floats = V * my_spinor_site_size * 2; // DMH: Vol * spinor elems * 2(re,im)

  const int n_contract_results = red_size * n_mom * nSpin * nSpin * 2;

  std::vector<double> d_result(n_contract_results, 0.0);

  constexpr int nprops = nSpin * src_colors;

  const int dof = my_spinor_site_size * 2 * nprops;

  // array of spinor field for each source spin and color
  size_t off = 0;

  // array of spinor field for each source spin and color
  std::array<void *, nprops> spinorX;
  std::array<void *, nprops> spinorY;

  std::array<std::vector<Float>, 2> buffs {std::vector<Float>(nprops * spinor_field_floats, 0),
                                           std::vector<Float>(nprops * spinor_field_floats, 0)};

  fill_buffers<Float, 2>(buffs, X, dof);

  for (int s = 0; s < nprops; ++s, off += spinor_field_floats) {
    spinorX[s] = static_cast<void *>(buffs[0].data() + off);
    spinorY[s] = static_cast<void *>(buffs[1].data() + off);
  }
  // Perform GPU contraction:
  void *d_result_ = static_cast<void *>(d_result.data());

  contractFTQuda(spinorX.data(), spinorY.data(), &d_result_, cType, (void *)(&cs_param), src_colors, X.data(),
                 source_position.data(), n_mom, mom.data(), fft_type.data());
  // Check results:
  int faults = contractionFT_reference<Float>(spinorX.data(), spinorY.data(), d_result.data(), cType, src_colors,
                                              X.data(), source_position.data(), n_mom, mom.data(), fft_type.data());

  return faults;
}

template <typename Float, int src_colors, int n_mom>
int launch_contract_test(const QudaContractType cType, const std::array<int, 4> &X, const int nspin, const int red_size,
                         const std::array<int, 4> &source_position, const std::array<int, n_mom * 4> &mom,
                         const std::array<QudaFFTSymmType, n_mom * 4> &fft_type)
{
  int faults = 0;

  if (nspin == 1) {
    faults = launch_contract_test<Float, 1, src_colors, n_mom>(cType, X, red_size, source_position, mom, fft_type);
    //} else  if ( nspin == 4 ){ //TODO : must be enabled when spin=4 case will be re-activated
    // faults = launch_contract_test<Float, 4, src_colors, n_mom>(cType, X, red_size, source_position, mom, fft_type );
  } else {
    errorQuda("Unsupported spin.\n");
  }

  return faults;
}

// Functions used for Google testing
// Performs the CPU GPU comparison with the given parameters
int contract(test_t param)
{
  if (xdim % 2) errorQuda("odd local x-dimension is not supported");

  const std::array<int, 4> X = {xdim, ydim, zdim, tdim};

  QudaContractType cType = ::testing::get<0>(param);
  QudaPrecision test_prec = ::testing::get<1>(param);

  const int nSpin = cType == QUDA_CONTRACT_TYPE_STAGGERED_FT_T ? 1 : 4;
  const int red_size = cType == QUDA_CONTRACT_TYPE_STAGGERED_FT_T || cType == QUDA_CONTRACT_TYPE_DR_FT_T ?
    comm_dim(3) * X[3] :
    comm_dim(2) * X[2]; // WARNING : check if needed

  const QudaFFTSymmType eo = QUDA_FFT_SYMM_EO;
  const QudaFFTSymmType ev = QUDA_FFT_SYMM_EVEN;
  const QudaFFTSymmType od = QUDA_FFT_SYMM_ODD;

  const std::array<int, 4> source_position = prop_source_position[0]; // make command option

  constexpr int n_mom = 18;

  const std::array<int, n_mom * 4> mom
    = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,  0, -1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,  0,  0, 0, -1, 0, 0, 0, 1, 0, 0,
       0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0,  0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, -1, -1, 0, 0, 1,  1, 0, 0, 1, 1, 0};

  const std::array<QudaFFTSymmType, n_mom * 4> fft_type
    = {eo, eo, eo, eo,                                                 // (0,0,0)
       ev, ev, ev, eo, eo, eo, eo, eo,                                 // (1,0,0)
       eo, eo, eo, eo, ev, ev, ev, eo, od, ev, ev, eo, eo, eo, eo, eo, // (0,1,0)
       eo, eo, eo, eo, ev, ev, ev, eo, ev, od, ev, eo, eo, eo, eo, eo, // (0,0,1)
       eo, eo, eo, eo, ev, ev, ev, eo, ev, ev, od, eo, eo, eo, eo, eo, // (0,1,1)
       eo, eo, eo, eo, ev, ev, ev, eo, ev, od, od, eo};

  int faults = 0;

  constexpr int src_colors = 1;

  if (test_prec == QUDA_SINGLE_PRECISION) {
    faults = launch_contract_test<float, src_colors, n_mom>(cType, X, nSpin, red_size, source_position, mom, fft_type);
  } else if (test_prec == QUDA_DOUBLE_PRECISION) {
    faults = launch_contract_test<double, src_colors, n_mom>(cType, X, nSpin, red_size, source_position, mom, fft_type);
  } else {
    errorQuda("Unsupported precision.\n");
  }

  const int n_contract_results = red_size * n_mom * nSpin * nSpin * 2;

  printfQuda("Contraction comparison for contraction type %s complete with %d/%d faults\n", get_contract_str(cType),
             faults, n_contract_results);

  return faults;
}
