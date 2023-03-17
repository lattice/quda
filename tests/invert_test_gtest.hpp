#include <gtest/gtest.h>
#include <quda_arch.h>

// tuple containing parameters for Schwarz solver
using schwarz_t = ::testing::tuple<QudaSchwarzType, QudaInverterType, QudaPrecision>;

using test_t = ::testing::tuple<QudaInverterType, QudaSolutionType, QudaSolveType, QudaPrecision, int, int, schwarz_t>;

class InvertTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  InvertTest() : param(GetParam()) { }
};

bool is_normal_residual(QudaInverterType type)
{
  switch (type) {
  case QUDA_CGNR_INVERTER:
  case QUDA_CA_CGNR_INVERTER: return true;
  default: return false;
  }
}

bool is_preconditioned_solve(QudaSolveType type)
{
  switch (type) {
  case QUDA_DIRECT_PC_SOLVE:
  case QUDA_NORMOP_PC_SOLVE: return true;
  default: return false;
  }
}

bool is_full_solution(QudaSolutionType type)
{
  switch (type) {
  case QUDA_MAT_SOLUTION:
  case QUDA_MATDAG_MAT_SOLUTION: return true;
  default: return false;
  }
}

bool is_normal_solve(test_t param)
{
  auto inv_type = ::testing::get<0>(param);
  auto solve_type = ::testing::get<2>(param);

  switch (solve_type) {
  case QUDA_NORMOP_SOLVE:
  case QUDA_NORMOP_PC_SOLVE: return true;
  default:
    switch (inv_type) {
    case QUDA_CGNR_INVERTER:
    case QUDA_CGNE_INVERTER:
    case QUDA_CA_CGNR_INVERTER:
    case QUDA_CA_CGNE_INVERTER: return true;
    default: return false;
    }
  }
}

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

bool support_solution_accumulator_pipeline(QudaInverterType type)
{
  switch (type) {
  case QUDA_CG_INVERTER:
  case QUDA_CA_CG_INVERTER:
  case QUDA_CGNR_INVERTER:
  case QUDA_CGNE_INVERTER:
  case QUDA_PCG_INVERTER: return true;
  default: return false;
  }
}

bool skip_test(test_t param)
{
  auto inverter_type = ::testing::get<0>(param);
  auto solution_type = ::testing::get<1>(param);
  auto prec_sloppy = ::testing::get<3>(param);
  auto multishift = ::testing::get<4>(param);
  auto solution_accumulator_pipeline = ::testing::get<5>(param);
  auto schwarz_param = ::testing::get<6>(param);
  auto prec_precondition = ::testing::get<2>(schwarz_param);

  if (prec < prec_sloppy) return true;              // outer precision >= sloppy precision
  if (!(QUDA_PRECISION & prec_sloppy)) return true; // precision not enabled so skip it
  if (!(QUDA_PRECISION & prec_precondition) && prec_precondition != QUDA_INVALID_PRECISION)
    return true; // precision not enabled so skip it
  if (prec_sloppy < prec_precondition) return true; // sloppy precision >= preconditioner precision

  // dwf-style solves must use a normal solver
  if (is_chiral(dslash_type) && !is_normal_solve(param)) return true;
  // FIXME this needs to be added to dslash_reference.cpp
  if (is_chiral(dslash_type) && multishift > 1) return true;
  // FIXME this needs to be added to dslash_reference.cpp
  if (is_chiral(dslash_type) && solution_type == QUDA_MATDAG_MAT_SOLUTION) return true;
  // Skip if the inverter does not support batched update and batched update is greater than one
  if (!support_solution_accumulator_pipeline(inverter_type) && solution_accumulator_pipeline > 1) return true;
  // MdagMLocal only support for Mobius at present
  if (is_normal_solve(param) && ::testing::get<0>(schwarz_param) != QUDA_INVALID_SCHWARZ) {
#ifdef QUDA_MMA_AVAILABLE
    if (dslash_type != QUDA_MOBIUS_DWF_DSLASH) return true;
#else
    return true;
#endif
  }

  return false;
}

std::vector<double> solve(test_t param);

TEST_P(InvertTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();
  auto tol = inv_param.tol;
  // FIXME eventually we should build in refinement to the *NR solvers to remove the need for this
  if (is_normal_residual(::testing::get<0>(GetParam()))) tol *= 50;
  // Slight loss of precision possible when reconstructing full solution
  if (is_full_solution(::testing::get<1>(GetParam())) && is_preconditioned_solve(::testing::get<2>(GetParam())))
    tol *= 10;

  for (auto rsd : solve(GetParam())) EXPECT_LE(rsd, tol);
}

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string name;
  name += get_solver_str(::testing::get<0>(param.param)) + std::string("_");
  name += get_solution_str(::testing::get<1>(param.param)) + std::string("_");
  name += get_solve_str(::testing::get<2>(param.param)) + std::string("_");
  name += get_prec_str(::testing::get<3>(param.param));
  if (::testing::get<4>(param.param) > 1)
    name += std::string("_shift") + std::to_string(::testing::get<4>(param.param));
  if (::testing::get<5>(param.param) > 1)
    name += std::string("_solution_accumulator_pipeline") + std::to_string(::testing::get<5>(param.param));
  auto &schwarz_param = ::testing::get<6>(param.param);
  if (::testing::get<0>(schwarz_param) != QUDA_INVALID_SCHWARZ) {
    name += std::string("_") + get_schwarz_str(::testing::get<0>(schwarz_param));
    name += std::string("_") + get_solver_str(::testing::get<1>(schwarz_param));
    name += std::string("_") + get_prec_str(::testing::get<2>(schwarz_param));
  }
  return name;
}

using ::testing::Combine;
using ::testing::Values;
auto normal_solvers = Values(QUDA_CG_INVERTER, QUDA_CA_CG_INVERTER, QUDA_PCG_INVERTER);

auto direct_solvers
  = Values(QUDA_CGNE_INVERTER, QUDA_CGNR_INVERTER, QUDA_CA_CGNE_INVERTER, QUDA_CA_CGNR_INVERTER, QUDA_GCR_INVERTER,
           QUDA_CA_GCR_INVERTER, QUDA_BICGSTAB_INVERTER, QUDA_BICGSTABL_INVERTER, QUDA_MR_INVERTER);

auto sloppy_precisions
  = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION);

auto solution_accumulator_pipelines = Values(1, 8);

auto no_schwarz = Combine(Values(QUDA_INVALID_SCHWARZ), Values(QUDA_INVALID_INVERTER), Values(QUDA_INVALID_PRECISION));

// preconditioned normal solves
INSTANTIATE_TEST_SUITE_P(NormalEvenOdd, InvertTest,
                         Combine(normal_solvers,
                                 Values(QUDA_MATPCDAG_MATPC_SOLUTION, QUDA_MAT_SOLUTION),
                                 Values(QUDA_NORMOP_PC_SOLVE), sloppy_precisions, Values(1),
                                 solution_accumulator_pipelines,
                                 no_schwarz),
                         gettestname);

// full system normal solve
INSTANTIATE_TEST_SUITE_P(NormalFull, InvertTest,
                         Combine(normal_solvers, Values(QUDA_MATDAG_MAT_SOLUTION),
                                 Values(QUDA_NORMOP_SOLVE),
                                 sloppy_precisions, Values(1),
                                 solution_accumulator_pipelines,
                                 no_schwarz),
                         gettestname);

// preconditioned direct solves
INSTANTIATE_TEST_SUITE_P(EvenOdd, InvertTest,
                         Combine(direct_solvers, Values(QUDA_MATPC_SOLUTION, QUDA_MAT_SOLUTION),
                                 Values(QUDA_DIRECT_PC_SOLVE), sloppy_precisions, Values(1),
                                 solution_accumulator_pipelines,
                                 no_schwarz),
                         gettestname);

// full system direct solve
INSTANTIATE_TEST_SUITE_P(Full, InvertTest,
                         Combine(direct_solvers, Values(QUDA_MAT_SOLUTION),
                                 Values(QUDA_DIRECT_SOLVE),
                                 sloppy_precisions, Values(1), solution_accumulator_pipelines,
                                 no_schwarz),
                         gettestname);

// preconditioned multi-shift solves
INSTANTIATE_TEST_SUITE_P(MultiShiftEvenOdd, InvertTest,
                         Combine(Values(QUDA_CG_INVERTER), Values(QUDA_MATPCDAG_MATPC_SOLUTION),
                                 Values(QUDA_NORMOP_PC_SOLVE), sloppy_precisions, Values(10),
                                 solution_accumulator_pipelines,
                                 no_schwarz),
                         gettestname);

// Schwarz-preconditioned normal solves
INSTANTIATE_TEST_SUITE_P(SchwarzNormal, InvertTest,
                         Combine(Values(QUDA_PCG_INVERTER),
                                 Values(QUDA_MATPCDAG_MATPC_SOLUTION),
                                 Values(QUDA_NORMOP_PC_SOLVE), sloppy_precisions,
                                 Values(1),
                                 solution_accumulator_pipelines,
                                 Combine(Values(QUDA_ADDITIVE_SCHWARZ),
                                         Values(QUDA_CG_INVERTER, QUDA_CA_CG_INVERTER),
                                         Values(QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION))),
                         gettestname);

// Schwarz-preconditioned direct solves
INSTANTIATE_TEST_SUITE_P(SchwarzEvenOdd, InvertTest,
                         Combine(Values(QUDA_GCR_INVERTER),
                                 Values(QUDA_MATPC_SOLUTION),
                                 Values(QUDA_DIRECT_PC_SOLVE), sloppy_precisions,
                                 Values(1),
                                 solution_accumulator_pipelines,
                                 Combine(Values(QUDA_ADDITIVE_SCHWARZ),
                                         Values(QUDA_MR_INVERTER, QUDA_CA_GCR_INVERTER),
                                         Values(QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION))),
                         gettestname);
