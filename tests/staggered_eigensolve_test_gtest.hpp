#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaEigType, QudaBoolean, QudaBoolean, QudaBoolean, QudaEigSpectrumType>;

class StaggeredEigensolveTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  StaggeredEigensolveTest() : param(GetParam()) { }
};

// Get the solve type that this combination corresponds to
QudaSolveType get_solve_type(QudaBoolean use_norm_op, QudaBoolean use_pc, QudaBoolean compute_svd)
{
  if (use_norm_op == QUDA_BOOLEAN_FALSE && use_pc == QUDA_BOOLEAN_TRUE && compute_svd == QUDA_BOOLEAN_FALSE)
    return QUDA_DIRECT_PC_SOLVE;
  else if (use_norm_op == QUDA_BOOLEAN_TRUE && use_pc == QUDA_BOOLEAN_FALSE && compute_svd == QUDA_BOOLEAN_TRUE)
    return QUDA_NORMOP_SOLVE;
  else if (use_norm_op == QUDA_BOOLEAN_FALSE && use_pc == QUDA_BOOLEAN_FALSE && compute_svd == QUDA_BOOLEAN_FALSE)
    return QUDA_DIRECT_SOLVE;
  else
    return QUDA_INVALID_SOLVE;
}

bool skip_test(test_t test_param)
{
  auto eig_type = ::testing::get<0>(test_param);
  auto use_norm_op = ::testing::get<1>(test_param);
  auto use_pc = ::testing::get<2>(test_param);
  auto compute_svd = ::testing::get<3>(test_param);
  auto spectrum = ::testing::get<4>(test_param);

  // 3-d operator only supported for Laplace
  if (eig_type == QUDA_EIG_TR_LANCZOS_3D && dslash_type != QUDA_LAPLACE_DSLASH) return true;

  // Reverse engineer the operator type
  QudaSolveType combo_solve_type = get_solve_type(use_norm_op, use_pc, compute_svd);
  if (combo_solve_type == QUDA_DIRECT_PC_SOLVE) {
    // matpc

    // this is only legal for the staggered and asqtad op
    if (!is_staggered(dslash_type)) return true;

    // we can only compute the real part for Lanczos, and real or magnitude for Arnoldi
    switch (eig_type) {
    case QUDA_EIG_TR_LANCZOS:
    case QUDA_EIG_BLK_TR_LANCZOS:
      if (spectrum != QUDA_SPECTRUM_LR_EIG && spectrum != QUDA_SPECTRUM_SR_EIG) return true;
      break;
    case QUDA_EIG_IR_ARNOLDI:
      if (spectrum == QUDA_SPECTRUM_LI_EIG || spectrum == QUDA_SPECTRUM_SI_EIG) return true;
      break;
    default: break;
    }
  } else if (combo_solve_type == QUDA_NORMOP_SOLVE) {
    // matdag_mat

    // this is only legal for the staggered and asqtad op
    if (!is_staggered(dslash_type)) return true;

    switch (eig_type) {
    case QUDA_EIG_TR_LANCZOS:
    case QUDA_EIG_BLK_TR_LANCZOS:
      if (spectrum != QUDA_SPECTRUM_LR_EIG && spectrum != QUDA_SPECTRUM_SR_EIG) return true;
      break;
    case QUDA_EIG_IR_ARNOLDI:
      // if (spectrum == QUDA_SPECTRUM_LI_EIG || spectrum == QUDA_SPECTRUM_SI_EIG) return true;
      return true; // we skip this because it takes an unnecessarily long time and it's covered elsewhere
    default: return true;
    }
  } else if (combo_solve_type == QUDA_DIRECT_SOLVE) {
    // mat

    switch (dslash_type) {
    case QUDA_STAGGERED_DSLASH:
      // only Arnoldi, imaginary part or magnitude works (real part is degenerate)
      // We skip SM because it takes an unnecessarily long time and it's
      // covered by HISQ
      if (eig_type != QUDA_EIG_IR_ARNOLDI) return true;
      if (spectrum != QUDA_SPECTRUM_LI_EIG && spectrum != QUDA_SPECTRUM_SI_EIG && spectrum != QUDA_SPECTRUM_LM_EIG)
        return true;
      break;
    case QUDA_ASQTAD_DSLASH:
      // only Arnoldi, imaginary part or magnitude works (real part is degenerate)
      if (eig_type != QUDA_EIG_IR_ARNOLDI) return true;
      if (spectrum == QUDA_SPECTRUM_LR_EIG || spectrum == QUDA_SPECTRUM_SR_EIG) return true;
      break;
    case QUDA_LAPLACE_DSLASH:
      switch (eig_type) {
      case QUDA_EIG_TR_LANCZOS:
      case QUDA_EIG_TR_LANCZOS_3D:
      case QUDA_EIG_BLK_TR_LANCZOS:
        if (spectrum != QUDA_SPECTRUM_LR_EIG && spectrum != QUDA_SPECTRUM_SR_EIG) return true;
        break;
      case QUDA_EIG_IR_ARNOLDI:
        if (spectrum == QUDA_SPECTRUM_LI_EIG || spectrum == QUDA_SPECTRUM_SI_EIG) return true;
        break;
      default: return true;
      }
      break;
    default: return true;
    }
  }
  return false;
}

std::vector<double> eigensolve(test_t test_param);

TEST_P(StaggeredEigensolveTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();
  double factor = 1.0;
  // The IRAM eigensolver will sometimes report convergence with tolerances slightly
  // higher than requested. The same phenomenon occurs in ARPACK. This factor
  // prevents failure when IRAM has solved to say 2e-6 when 1e-6 is requested.
  // The solution to avoid this is to use a Krylov space (eig-n-kr) about 3-4 times the
  // size of the search space (eig-n-ev), or use a well chosen Chebyshev polynomial,
  // or use a tighter than necessary tolerance.
  auto eig_type = ::testing::get<0>(GetParam());
  if (eig_type == QUDA_EIG_IR_ARNOLDI || eig_type == QUDA_EIG_BLK_IR_ARNOLDI) factor *= 10;
  auto tol = factor * eig_param.tol;
  if (dslash_type == QUDA_LAPLACE_DSLASH) laplace3D = eig_type == QUDA_EIG_TR_LANCZOS_3D ? 3 : 4;

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

// Batched solvers for 3-d operators
auto batched_solvers = Values(QUDA_EIG_TR_LANCZOS_3D);

// Eigensolver spectrum types
auto hermitian_spectrum = Values(QUDA_SPECTRUM_LR_EIG, QUDA_SPECTRUM_SR_EIG);
auto non_hermitian_spectrum = Values(QUDA_SPECTRUM_LR_EIG, QUDA_SPECTRUM_SR_EIG, QUDA_SPECTRUM_LM_EIG,
                                     QUDA_SPECTRUM_SM_EIG, QUDA_SPECTRUM_LI_EIG, QUDA_SPECTRUM_SI_EIG);

// using test_t = ::testing::tuple<QudaEigType,          // different types of Lanczos/Arnoldi
//                                QudaBoolean,          // Norm op or not
//                                QudaBoolean,          // Preconditioned op or not
//                                QudaBoolean,          // SVD or not
//                                QudaEigSpectrumType>; // Largest real, smallest real, etc

// Preconditioned direct operators, which are HPD for staggered!
INSTANTIATE_TEST_SUITE_P(DirectEvenOdd, StaggeredEigensolveTest,
                         ::testing::Combine(hermitian_solvers, Values(QUDA_BOOLEAN_FALSE), Values(QUDA_BOOLEAN_TRUE),
                                            Values(QUDA_BOOLEAN_FALSE), hermitian_spectrum),
                         gettestname);

// full system normal solve
INSTANTIATE_TEST_SUITE_P(NormalFull, StaggeredEigensolveTest,
                         ::testing::Combine(hermitian_solvers, Values(QUDA_BOOLEAN_TRUE), Values(QUDA_BOOLEAN_FALSE),
                                            Values(QUDA_BOOLEAN_TRUE), hermitian_spectrum),
                         gettestname);

// full system direct solve
INSTANTIATE_TEST_SUITE_P(DirectFull, StaggeredEigensolveTest,
                         ::testing::Combine(hermitian_solvers, Values(QUDA_BOOLEAN_FALSE), Values(QUDA_BOOLEAN_FALSE),
                                            Values(QUDA_BOOLEAN_FALSE), non_hermitian_spectrum),
                         gettestname);

// 3-d full system direct solve
INSTANTIATE_TEST_SUITE_P(DirectFull3D, StaggeredEigensolveTest,
                         ::testing::Combine(batched_solvers, Values(QUDA_BOOLEAN_FALSE), Values(QUDA_BOOLEAN_FALSE),
                                            Values(QUDA_BOOLEAN_FALSE), hermitian_spectrum),
                         gettestname);
