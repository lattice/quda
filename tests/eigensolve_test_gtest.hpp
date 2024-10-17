#include <instantiate.h>
#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaPrecision, QudaEigType, QudaBoolean, QudaBoolean, QudaBoolean, QudaEigSpectrumType>;

bool skip_test(test_t param)
{
  auto prec = ::testing::get<0>(param);
  if (!quda::is_enabled(prec)) return true; // skip if precision is not enabled
  // dwf-style solves must use a normal solver
  if (is_chiral(dslash_type) && (::testing::get<2>(param) == QUDA_BOOLEAN_FALSE)) return true;
  return false;
}

class EigensolveTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  EigensolveTest() : param(GetParam()) { }

  virtual void SetUp()
  {
    if (skip_test(GetParam())) GTEST_SKIP();

    // check if outer precision has changed and update if it has
    if (::testing::get<0>(param) != last_prec) {
      if (last_prec != QUDA_INVALID_PRECISION) {
        freeGaugeQuda();
        if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) freeCloverQuda();
      }

      // Load the gauge field to the device
      gauge_param.cuda_prec = ::testing::get<0>(param);
      gauge_param.cuda_prec_sloppy = ::testing::get<0>(param);
      gauge_param.cuda_prec_precondition = ::testing::get<0>(param);
      gauge_param.cuda_prec_refinement_sloppy = ::testing::get<0>(param);
      gauge_param.cuda_prec_eigensolver = ::testing::get<0>(param);
      loadGaugeQuda(gauge.data(), &gauge_param);

      if (dslash_type == QUDA_CLOVER_WILSON_DSLASH || dslash_type == QUDA_TWISTED_CLOVER_DSLASH) {
        // Load the clover terms to the device
        eig_inv_param.clover_cuda_prec = ::testing::get<0>(param);
        eig_inv_param.clover_cuda_prec_sloppy = ::testing::get<0>(param);
        eig_inv_param.clover_cuda_prec_precondition = ::testing::get<0>(param);
        eig_inv_param.clover_cuda_prec_refinement_sloppy = ::testing::get<0>(param);
        eig_inv_param.clover_cuda_prec_eigensolver = ::testing::get<0>(param);
        loadCloverQuda(clover.data(), clover_inv.data(), &eig_inv_param);
      }
      last_prec = ::testing::get<0>(param);
    }

    // Compute plaquette as a sanity check
    double plaq[3];
    plaqQuda(plaq);
    printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }
};

std::vector<double> eigensolve(test_t test_param);

TEST_P(EigensolveTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();

  auto tol = ::testing::get<0>(GetParam()) == QUDA_SINGLE_PRECISION ? 1e-5 : 1e-12;
  eig_param.tol = tol;

  // The IRAM eigensolver will sometimes report convergence with tolerances slightly
  // higher than requested. The same phenomenon occurs in ARPACK. This factor
  // prevents failure when IRAM has solved to say 2e-6 when 1e-6 is requested.
  // The solution to avoid this is to use a Krylov space (eig-n-kr) about 3-4 times the
  // size of the search space (eig-n-ev), or use a well chosen Chebyshev polynomial,
  // or use a tighter than necessary tolerance.
  if (eig_param.eig_type == QUDA_EIG_IR_ARNOLDI || eig_param.eig_type == QUDA_EIG_BLK_IR_ARNOLDI) tol *= 15;
  for (auto rsd : eigensolve(GetParam())) EXPECT_LE(rsd, tol);
}

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string name;
  name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
  name += get_eig_type_str(::testing::get<1>(param.param)) + std::string("_");
  name += (::testing::get<2>(param.param) == QUDA_BOOLEAN_TRUE ? std::string("normop") : std::string("direct"))
    + std::string("_");
  name += (::testing::get<3>(param.param) == QUDA_BOOLEAN_TRUE ? std::string("evenodd") : std::string("full"))
    + std::string("_");
  name += (::testing::get<4>(param.param) == QUDA_BOOLEAN_TRUE ? std::string("withSVD") : std::string("noSVD"))
    + std::string("_");
  name += get_eig_spectrum_str(::testing::get<5>(param.param));
  return name;
}

using ::testing::Combine;
using ::testing::Values;

// Can solve hermitian systems
auto hermitian_solvers = Values(QUDA_EIG_TR_LANCZOS, QUDA_EIG_BLK_TR_LANCZOS, QUDA_EIG_IR_ARNOLDI);

// Can solve non-hermitian systems
auto non_hermitian_solvers = Values(QUDA_EIG_IR_ARNOLDI);

// Eigensolver spectrum types
auto hermitian_spectrum = Values(QUDA_SPECTRUM_LR_EIG, QUDA_SPECTRUM_SR_EIG);
auto non_hermitian_spectrum = Values(QUDA_SPECTRUM_LR_EIG, QUDA_SPECTRUM_SR_EIG, QUDA_SPECTRUM_LM_EIG,
                                     QUDA_SPECTRUM_SM_EIG, QUDA_SPECTRUM_LI_EIG, QUDA_SPECTRUM_SI_EIG);

auto precisions = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION);

// preconditioned normal solves
INSTANTIATE_TEST_SUITE_P(NormalEvenOdd, EigensolveTest,
                         ::testing::Combine(precisions, hermitian_solvers, Values(QUDA_BOOLEAN_TRUE),
                                            Values(QUDA_BOOLEAN_TRUE), Values(QUDA_BOOLEAN_TRUE), hermitian_spectrum),
                         gettestname);

// full system normal solve
INSTANTIATE_TEST_SUITE_P(NormalFull, EigensolveTest,
                         ::testing::Combine(precisions, hermitian_solvers, Values(QUDA_BOOLEAN_TRUE),
                                            Values(QUDA_BOOLEAN_TRUE), Values(QUDA_BOOLEAN_TRUE), hermitian_spectrum),
                         gettestname);

// preconditioned direct solves
INSTANTIATE_TEST_SUITE_P(DirectEvenOdd, EigensolveTest,
                         ::testing::Combine(precisions, non_hermitian_solvers, Values(QUDA_BOOLEAN_FALSE),
                                            Values(QUDA_BOOLEAN_TRUE), Values(QUDA_BOOLEAN_FALSE), non_hermitian_spectrum),
                         gettestname);
// full system direct solve
INSTANTIATE_TEST_SUITE_P(DirectFull, EigensolveTest,
                         ::testing::Combine(precisions, non_hermitian_solvers, Values(QUDA_BOOLEAN_FALSE),
                                            Values(QUDA_BOOLEAN_FALSE), Values(QUDA_BOOLEAN_FALSE),
                                            non_hermitian_spectrum),
                         gettestname);
