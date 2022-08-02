#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaEigType, QudaBoolean, QudaBoolean, QudaBoolean, QudaEigSpectrumType>;

class EigensolveTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  EigensolveTest() : param(GetParam()) { }
};

bool is_chiral(QudaDslashType type)
{
  switch (type) {
  case QUDA_DOMAIN_WALL_DSLASH:
  case QUDA_DOMAIN_WALL_4D_DSLASH:
  case QUDA_MOBIUS_DWF_DSLASH:
  case QUDA_MOBIUS_DWF_EOFA_DSLASH: return true;
  default: return false;
  }
}

bool skip_test(test_t param)
{
  // dwf-style solves must use a normal solver
  if (is_chiral(dslash_type) && (::testing::get<1>(param) == QUDA_BOOLEAN_FALSE)) return true;
  return false;
}

std::vector<double> eigensolve(test_t test_param);

TEST_P(EigensolveTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();
  double factor = 1.0;
  // The IRAM eigensolver will sometimes report convergence with tolerances slightly
  // higher than requested. The same phenomenon occurs in ARPACK. This factor
  // prevents failure when IRAM has solved to say 2e-6 when 1e-6 is requested.
  // The solution to avoid this is to use a Krylov space (eig-n-kr) about 3-4 times the
  // size of the search space (eig-n-ev), or use a well chosen Chebyshev polynomial,
  // or use a tighter than necessary tolerance.
  if (eig_param.eig_type == QUDA_EIG_IR_ARNOLDI || eig_param.eig_type == QUDA_EIG_BLK_IR_ARNOLDI) factor *= 5;
  auto tol = factor * eig_param.tol;
  for (auto rsd : eigensolve(GetParam())) EXPECT_LE(rsd, tol);
}

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string name;
  name += get_eig_type_str(::testing::get<0>(param.param)) + std::string("_");
  name += (::testing::get<1>(param.param) == QUDA_BOOLEAN_TRUE ? std::string("normop") : std::string("direct"))
    + std::string("_");
  name += (::testing::get<2>(param.param) == QUDA_BOOLEAN_TRUE ? std::string("evenodd") : std::string("full"))
    + std::string("_");
  name += (::testing::get<3>(param.param) == QUDA_BOOLEAN_TRUE ? std::string("withSVD") : std::string("noSVD"))
    + std::string("_");
  name += get_eig_spectrum_str(::testing::get<4>(param.param));
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

// preconditioned normal solves
INSTANTIATE_TEST_SUITE_P(NormalEvenOdd, EigensolveTest,
                         ::testing::Combine(hermitian_solvers, Values(QUDA_BOOLEAN_TRUE), Values(QUDA_BOOLEAN_TRUE),
                                            Values(QUDA_BOOLEAN_TRUE), hermitian_spectrum),
                         gettestname);

// full system normal solve
INSTANTIATE_TEST_SUITE_P(NormalFull, EigensolveTest,
                         ::testing::Combine(hermitian_solvers, Values(QUDA_BOOLEAN_TRUE), Values(QUDA_BOOLEAN_TRUE),
                                            Values(QUDA_BOOLEAN_TRUE), hermitian_spectrum),
                         gettestname);

// preconditioned direct solves
INSTANTIATE_TEST_SUITE_P(DirectEvenOdd, EigensolveTest,
                         ::testing::Combine(non_hermitian_solvers, Values(QUDA_BOOLEAN_FALSE), Values(QUDA_BOOLEAN_TRUE),
                                            Values(QUDA_BOOLEAN_FALSE), non_hermitian_spectrum),
                         gettestname);
// full system direct solve
INSTANTIATE_TEST_SUITE_P(DirectFull, EigensolveTest,
                         ::testing::Combine(non_hermitian_solvers, Values(QUDA_BOOLEAN_FALSE), Values(QUDA_BOOLEAN_FALSE),
                                            Values(QUDA_BOOLEAN_FALSE), non_hermitian_spectrum),
                         gettestname);
