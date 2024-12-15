#include <gtest/gtest.h>
#include <quda_arch.h>

// tuple containing parameters for Schwarz solver
using schwarz_t = ::testing::tuple<QudaSchwarzType, QudaInverterType, QudaPrecision>;

using test_t
  = ::testing::tuple<QudaInverterType, QudaSolutionType, QudaSolveType, QudaPrecision, int, int, schwarz_t, QudaResidualType>;

class StaggeredInvertTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  StaggeredInvertTest() : param(GetParam()) { }
};

bool skip_test(test_t param)
{
  auto inverter_type = ::testing::get<0>(param);
  auto solution_type = ::testing::get<1>(param);
  auto solve_type = ::testing::get<2>(param);
  auto prec_sloppy = ::testing::get<3>(param);
  auto multishift = ::testing::get<4>(param);
  auto solution_accumulator_pipeline = ::testing::get<5>(param);
  auto schwarz_param = ::testing::get<6>(param);
  auto prec_precondition = ::testing::get<2>(schwarz_param);

  if (prec < prec_sloppy) return true;              // outer precision >= sloppy precision
  if (!(QUDA_PRECISION & prec_sloppy)) return true; // precision not enabled so skip it
  if (!(QUDA_PRECISION & prec_precondition) && prec_precondition != QUDA_INVALID_PRECISION)
    return true;                                    // precision not enabled so skip it
  if (prec_sloppy < prec_precondition) return true; // sloppy precision >= preconditioner precision

  // Skip if the inverter does not support batched update and batched update is greater than one
  if (!support_solution_accumulator_pipeline(inverter_type) && solution_accumulator_pipeline > 1) return true;
  // There's no MLocal or MdagMLocal support yet, this is left in for reference
  // if (is_normal_solve(param) && ::testing::get<0>(schwarz_param) != QUDA_INVALID_SCHWARZ)
  //  if (dslash_type != QUDA_MOBIUS_DWF_DSLASH) return true;

  if (is_laplace(dslash_type)) {
    if (multishift > 1) return true; // Laplace doesn't support multishift
    if (solution_type != QUDA_MAT_SOLUTION || solve_type != QUDA_DIRECT_SOLVE)
      return true; // Laplace only supports direct solves
  }

  if (is_staggered(dslash_type)) {
    // the staggered and asqtad operators aren't HPD
    if (solution_type == QUDA_MAT_SOLUTION && solve_type == QUDA_DIRECT_SOLVE && is_hermitian_solver(inverter_type))
      return true;

    // MR struggles with the staggered and asqtad spectrum, it's not MR's fault
    if (solution_type == QUDA_MAT_SOLUTION && solve_type == QUDA_DIRECT_SOLVE && inverter_type == QUDA_MR_INVERTER)
      return true;

    // BiCGStab-L is unstable with staggered operator with low precision
    // we could likely restore this when 20-bit quark fields are merged in
    if (inverter_type == QUDA_BICGSTABL_INVERTER && prec_sloppy < QUDA_SINGLE_PRECISION) return true;
  }

  // CG3 is rather unstable with low precision
  if ((inverter_type == QUDA_CG3_INVERTER || inverter_type == QUDA_CG3NE_INVERTER || inverter_type == QUDA_CG3NR_INVERTER)
      && prec_sloppy < QUDA_DOUBLE_PRECISION)
    return true;

  // split-grid doesn't support multigrid at present
  if (use_split_grid && multishift > 1) return true;

  return false;
}

std::vector<std::array<double, 2>> solve(test_t param);

TEST_P(StaggeredInvertTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();

  inv_param.tol = 0.0;
  inv_param.tol_hq = 0.0;
  auto res_t = ::testing::get<7>(GetParam());
  if (res_t & QUDA_L2_RELATIVE_RESIDUAL) inv_param.tol = tol;
  if (res_t & QUDA_HEAVY_QUARK_RESIDUAL) inv_param.tol_hq = tol_hq;

  auto inverter_type = ::testing::get<0>(param);
  auto solution_type = ::testing::get<1>(param);
  auto solve_type = ::testing::get<2>(param);

  // Make a local copy of "tol" for modification in place
  auto verify_tol = tol;

  // FIXME eventually we should build in refinement to the *NR solvers to remove the need for this
  // The mass squared is a proxy for the condition number
  if (is_normal_residual(inverter_type)) verify_tol /= (0.25 * mass * mass);

  // To solve the direct operator to a given tolerance, grind the preconditioned
  // operator to 0.5 * mass * tol... to keep the target tolerance in inv_param
  // in check, we shift the requirement to the verified tolerance instead.
  if (solution_type == QUDA_MAT_SOLUTION) {
    if (solve_type == QUDA_DIRECT_PC_SOLVE)
      verify_tol /= (0.5 * mass); // to solve the full operator to eps, solve the preconditioned to mass * eps
    if (solve_type == QUDA_NORMOP_SOLVE) verify_tol /= (0.5 * mass); // a proxy for the condition number
  }

  // The power iterations method of determining the Chebyshev window
  // breaks down due to the nature of the spectrum of the direct operator
  auto ca_basis_tmp = inv_param.ca_basis;
  if (solve_type == QUDA_DIRECT_SOLVE && inverter_type == QUDA_CA_GCR_INVERTER) inv_param.ca_basis = QUDA_POWER_BASIS;

  // Single precision needs a tiny bump due to small host/device precision deviations
  if (prec == QUDA_SINGLE_PRECISION) verify_tol *= 1.01;

  for (auto rsd : solve(GetParam())) {
    if (res_t & QUDA_L2_RELATIVE_RESIDUAL) { EXPECT_LE(rsd[0], verify_tol); }
    if (res_t & QUDA_HEAVY_QUARK_RESIDUAL) { EXPECT_LE(rsd[1], tol_hq); }
  }

  inv_param.ca_basis = ca_basis_tmp;
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
  auto res_t = ::testing::get<7>(param.param);
  if (res_t & QUDA_L2_RELATIVE_RESIDUAL) name += std::string("_l2");
  if (res_t & QUDA_HEAVY_QUARK_RESIDUAL) name += std::string("_heavy_quark");
  return name;
}

using ::testing::Combine;
using ::testing::Values;

auto staggered_pc_solvers
  = Values(QUDA_CG_INVERTER, QUDA_CA_CG_INVERTER, QUDA_CG3_INVERTER, QUDA_PCG_INVERTER, QUDA_GCR_INVERTER,
           QUDA_CA_GCR_INVERTER, QUDA_BICGSTAB_INVERTER, QUDA_BICGSTABL_INVERTER, QUDA_MR_INVERTER);

auto normal_solvers = Values(QUDA_CG_INVERTER, QUDA_CA_CG_INVERTER, QUDA_CG3_INVERTER, QUDA_PCG_INVERTER);

auto direct_solvers = Values(QUDA_CGNE_INVERTER, QUDA_CGNR_INVERTER, QUDA_CA_CGNE_INVERTER, QUDA_CA_CGNR_INVERTER,
                             QUDA_CG3NE_INVERTER, QUDA_CG3NR_INVERTER, QUDA_GCR_INVERTER, QUDA_CA_GCR_INVERTER,
                             QUDA_BICGSTAB_INVERTER, QUDA_BICGSTABL_INVERTER, QUDA_MR_INVERTER);

auto sloppy_precisions
  = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION);

auto solution_accumulator_pipelines = Values(1, 8);

auto no_schwarz = Combine(Values(QUDA_INVALID_SCHWARZ), Values(QUDA_INVALID_INVERTER), Values(QUDA_INVALID_PRECISION));

auto no_heavy_quark = Values(QUDA_L2_RELATIVE_RESIDUAL);

// the staggered PC op doesn't support "normal" operators since it's already
// Hermitian positive definite

// preconditioned solves
INSTANTIATE_TEST_SUITE_P(EvenOdd, StaggeredInvertTest,
                         Combine(staggered_pc_solvers, Values(QUDA_MATPC_SOLUTION, QUDA_MAT_SOLUTION),
                                 Values(QUDA_DIRECT_PC_SOLVE), sloppy_precisions, Values(1),
                                 solution_accumulator_pipelines, no_schwarz, no_heavy_quark),
                         gettestname);

// full system normal solve
INSTANTIATE_TEST_SUITE_P(NormalFull, StaggeredInvertTest,
                         Combine(normal_solvers, Values(QUDA_MATDAG_MAT_SOLUTION, QUDA_MAT_SOLUTION),
                                 Values(QUDA_NORMOP_SOLVE), sloppy_precisions, Values(1),
                                 solution_accumulator_pipelines, no_schwarz, no_heavy_quark),
                         gettestname);

// full system direct solve
INSTANTIATE_TEST_SUITE_P(Full, StaggeredInvertTest,
                         Combine(direct_solvers, Values(QUDA_MAT_SOLUTION), Values(QUDA_DIRECT_SOLVE), sloppy_precisions,
                                 Values(1), solution_accumulator_pipelines, no_schwarz, no_heavy_quark),
                         gettestname);

// preconditioned multi-shift solves
INSTANTIATE_TEST_SUITE_P(MultiShiftEvenOdd, StaggeredInvertTest,
                         Combine(Values(QUDA_CG_INVERTER), Values(QUDA_MATPC_SOLUTION), Values(QUDA_DIRECT_PC_SOLVE),
                                 sloppy_precisions, Values(10), solution_accumulator_pipelines, no_schwarz,
                                 no_heavy_quark),
                         gettestname);

// Heavy-Quark preconditioned solves
INSTANTIATE_TEST_SUITE_P(HeavyQuarkEvenOdd, StaggeredInvertTest,
                         Combine(Values(QUDA_CG_INVERTER), Values(QUDA_MATPC_SOLUTION), Values(QUDA_DIRECT_PC_SOLVE),
                                 sloppy_precisions, Values(1), solution_accumulator_pipelines, no_schwarz,
                                 Values(QUDA_L2_RELATIVE_RESIDUAL | QUDA_HEAVY_QUARK_RESIDUAL, QUDA_HEAVY_QUARK_RESIDUAL)),
                         gettestname);

// These are left in but commented out for future reference

// Schwarz-preconditioned normal solves
// INSTANTIATE_TEST_SUITE_P(SchwarzNormal, StaggeredInvertTest,
//                         Combine(Values(QUDA_PCG_INVERTER), Values(QUDA_MATPCDAG_MATPC_SOLUTION),
//                                 Values(QUDA_NORMOP_PC_SOLVE), sloppy_precisions, Values(1),
//                                 solution_accumulator_pipelines,
//                                 Combine(Values(QUDA_ADDITIVE_SCHWARZ), Values(QUDA_CG_INVERTER, QUDA_CA_CG_INVERTER),
//                                         Values(QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION)),
//                                 no_heavy_quark),
//                         gettestname);

// Schwarz-preconditioned direct solves
// INSTANTIATE_TEST_SUITE_P(SchwarzEvenOdd, StaggeredInvertTest,
//                         Combine(Values(QUDA_GCR_INVERTER), Values(QUDA_MATPC_SOLUTION), Values(QUDA_DIRECT_PC_SOLVE),
//                                 sloppy_precisions, Values(1), solution_accumulator_pipelines,
//                                 Combine(Values(QUDA_ADDITIVE_SCHWARZ), Values(QUDA_MR_INVERTER, QUDA_CA_GCR_INVERTER),
//                                         Values(QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION)),
//                                 no_heavy_quark),
//                         gettestname);
