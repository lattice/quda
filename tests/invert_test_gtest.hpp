#include <gtest/gtest.h>
#include <quda_arch.h>
#include <cmath>

// tuple containing parameters for Schwarz solver
using schwarz_t = ::testing::tuple<QudaSchwarzType, QudaInverterType, QudaPrecision>;

using test_t = ::testing::tuple<QudaPrecision, QudaPrecision, QudaInverterType, QudaSolutionType, QudaSolveType, int,
                                int, schwarz_t, QudaResidualType>;

bool skip_test(test_t param)
{
  auto prec = ::testing::get<0>(param);
  auto prec_sloppy = ::testing::get<1>(param);
  auto inverter_type = ::testing::get<2>(param);
  auto solution_type = ::testing::get<3>(param);
  auto solve_type = ::testing::get<4>(param);
  auto multishift = ::testing::get<5>(param);
  auto solution_accumulator_pipeline = ::testing::get<6>(param);
  auto schwarz_param = ::testing::get<7>(param);
  auto prec_precondition = ::testing::get<2>(schwarz_param);

  if (prec < prec_sloppy) return true;              // outer precision >= sloppy precision
  if (!(QUDA_PRECISION & prec)) return true;        // precision not enabled so skip it
  if (!(QUDA_PRECISION & prec_sloppy)) return true; // precision not enabled so skip it
  if (!(QUDA_PRECISION & prec_precondition) && prec_precondition != QUDA_INVALID_PRECISION)
    return true; // precision not enabled so skip it
  if (prec_sloppy < prec_precondition) return true; // sloppy precision >= preconditioner precision

  // dwf-style solves must use a normal solver
  if (is_chiral(dslash_type) && !is_normal_solve(inverter_type, solve_type)) return true;
  // FIXME this needs to be added to dslash_reference.cpp
  if (is_chiral(dslash_type) && multishift > 1) return true;
  // FIXME this needs to be added to dslash_reference.cpp
  if (is_chiral(dslash_type) && solution_type == QUDA_MATDAG_MAT_SOLUTION) return true;
  // Skip if the inverter does not support batched update and batched update is greater than one
  if (!support_solution_accumulator_pipeline(inverter_type) && solution_accumulator_pipeline > 1) return true;
  // MdagMLocal only support for Mobius at present
  if (is_normal_solve(inverter_type, solve_type) && ::testing::get<0>(schwarz_param) != QUDA_INVALID_SCHWARZ) {
#ifdef QUDA_MMA_AVAILABLE
    if (dslash_type != QUDA_MOBIUS_DWF_DSLASH) return true;
#else
    return true;
#endif
  }
  // CG3 is rather unstable with low precision
  if ((inverter_type == QUDA_CG3_INVERTER || inverter_type == QUDA_CG3NE_INVERTER || inverter_type == QUDA_CG3NR_INVERTER)
      && prec_sloppy < QUDA_DOUBLE_PRECISION)
    return true;
  // split-grid doesn't support multishift at present
  if (use_split_grid && multishift > 1) return true;
  if ((distance_pc_alpha0 != 0 && distance_pc_t0 >= 0) && (multishift > 1)) return true;

  return false;
}

class InvertTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  InvertTest() : param(GetParam()) { }

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
        inv_param.clover_cuda_prec = ::testing::get<0>(param);
        inv_param.clover_cuda_prec_sloppy = ::testing::get<0>(param);
        inv_param.clover_cuda_prec_precondition = ::testing::get<0>(param);
        inv_param.clover_cuda_prec_refinement_sloppy = ::testing::get<0>(param);
        inv_param.clover_cuda_prec_eigensolver = ::testing::get<0>(param);
        loadCloverQuda(clover.data(), clover_inv.data(), &inv_param);
      }
      last_prec = ::testing::get<0>(param);
    }

    // Compute plaquette as a sanity check
    double plaq[3];
    plaqQuda(plaq);
    printfQuda("Computed plaquette is %e (spatial = %e, temporal = %e)\n", plaq[0], plaq[1], plaq[2]);
  }
};

std::vector<std::array<double, 2>> solve(test_t param);

TEST_P(InvertTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();

  auto tol = ::testing::get<0>(GetParam()) == QUDA_SINGLE_PRECISION ? 1e-6 : 1e-12;
  auto tol_hq = ::testing::get<0>(GetParam()) == QUDA_SINGLE_PRECISION ? 1e-6 : 1e-12;
  inv_param.tol = 0.0;
  inv_param.tol_hq = 0.0;
  auto res_t = ::testing::get<8>(GetParam());
  if (res_t & QUDA_L2_RELATIVE_RESIDUAL) inv_param.tol = tol;
  if (res_t & QUDA_HEAVY_QUARK_RESIDUAL) inv_param.tol_hq = tol_hq;

  if (is_chiral(inv_param.dslash_type)) { tol *= std::sqrt(static_cast<double>(inv_param.Ls)); }
  // FIXME eventually we should build in refinement to the *NR solvers to remove the need for this
  if (is_normal_residual(::testing::get<2>(GetParam()))) tol *= 50;
  // Slight loss of precision possible when reconstructing full solution
  if (is_full_solution(::testing::get<3>(GetParam())) && is_preconditioned_solve(::testing::get<4>(GetParam())))
    tol *= 10;

  for (auto rsd : solve(GetParam())) {
    if (res_t & QUDA_L2_RELATIVE_RESIDUAL) { EXPECT_LE(rsd[0], tol); }
    if (res_t & QUDA_HEAVY_QUARK_RESIDUAL) { EXPECT_LE(rsd[1], tol_hq); }
  }
}

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string name;
  name += get_prec_str(::testing::get<0>(param.param)) + std::string("_");
  name += get_prec_str(::testing::get<1>(param.param)) + std::string("_");
  name += get_solver_str(::testing::get<2>(param.param)) + std::string("_");
  name += get_solution_str(::testing::get<3>(param.param)) + std::string("_");
  name += get_solve_str(::testing::get<4>(param.param));
  if (::testing::get<5>(param.param) > 1)
    name += std::string("_shift") + std::to_string(::testing::get<5>(param.param));
  if (::testing::get<6>(param.param) > 1)
    name += std::string("_solution_accumulator_pipeline") + std::to_string(::testing::get<6>(param.param));
  auto &schwarz_param = ::testing::get<7>(param.param);
  if (::testing::get<0>(schwarz_param) != QUDA_INVALID_SCHWARZ) {
    name += std::string("_") + get_schwarz_str(::testing::get<0>(schwarz_param));
    name += std::string("_") + get_solver_str(::testing::get<1>(schwarz_param));
    name += std::string("_") + get_prec_str(::testing::get<2>(schwarz_param));
  }
  auto res_t = ::testing::get<8>(param.param);
  if (res_t & QUDA_L2_RELATIVE_RESIDUAL) name += std::string("_l2");
  if (res_t & QUDA_HEAVY_QUARK_RESIDUAL) name += std::string("_heavy_quark");
  return name;
}

using ::testing::Combine;
using ::testing::Values;
auto normal_solvers
  = Values(QUDA_CG_INVERTER, QUDA_CA_CG_INVERTER, QUDA_CG3_INVERTER, QUDA_PCG_INVERTER, QUDA_SD_INVERTER);

auto direct_solvers = Values(QUDA_CGNE_INVERTER, QUDA_CGNR_INVERTER, QUDA_CA_CGNE_INVERTER, QUDA_CA_CGNR_INVERTER,
                             QUDA_CG3NE_INVERTER, QUDA_CG3NR_INVERTER, QUDA_GCR_INVERTER, QUDA_CA_GCR_INVERTER,
                             QUDA_BICGSTAB_INVERTER, QUDA_BICGSTABL_INVERTER, QUDA_MR_INVERTER);

auto precisions = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION);

auto sloppy_precisions
  = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION, QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION);

auto solution_accumulator_pipelines = Values(1, 8);

auto no_schwarz = Combine(Values(QUDA_INVALID_SCHWARZ), Values(QUDA_INVALID_INVERTER), Values(QUDA_INVALID_PRECISION));

auto no_heavy_quark = Values(QUDA_L2_RELATIVE_RESIDUAL);

// preconditioned normal solves
INSTANTIATE_TEST_SUITE_P(NormalEvenOdd, InvertTest,
                         Combine(precisions, sloppy_precisions, normal_solvers,
                                 Values(QUDA_MATPCDAG_MATPC_SOLUTION, QUDA_MAT_SOLUTION), Values(QUDA_NORMOP_PC_SOLVE),
                                 Values(1), solution_accumulator_pipelines, no_schwarz, no_heavy_quark),
                         gettestname);

// full system normal solve
INSTANTIATE_TEST_SUITE_P(NormalFull, InvertTest,
                         Combine(precisions, sloppy_precisions, normal_solvers, Values(QUDA_MATDAG_MAT_SOLUTION),
                                 Values(QUDA_NORMOP_SOLVE), Values(1), solution_accumulator_pipelines, no_schwarz,
                                 no_heavy_quark),
                         gettestname);

// preconditioned direct solves
INSTANTIATE_TEST_SUITE_P(EvenOdd, InvertTest,
                         Combine(precisions, sloppy_precisions, direct_solvers,
                                 Values(QUDA_MATPC_SOLUTION, QUDA_MAT_SOLUTION), Values(QUDA_DIRECT_PC_SOLVE),
                                 Values(1), solution_accumulator_pipelines, no_schwarz, no_heavy_quark),
                         gettestname);

// full system direct solve
INSTANTIATE_TEST_SUITE_P(Full, InvertTest,
                         Combine(precisions, sloppy_precisions, direct_solvers, Values(QUDA_MAT_SOLUTION),
                                 Values(QUDA_DIRECT_SOLVE), Values(1), solution_accumulator_pipelines, no_schwarz,
                                 no_heavy_quark),
                         gettestname);

// preconditioned multi-shift solves
INSTANTIATE_TEST_SUITE_P(MultiShiftEvenOdd, InvertTest,
                         Combine(precisions, sloppy_precisions, Values(QUDA_CG_INVERTER),
                                 Values(QUDA_MATPCDAG_MATPC_SOLUTION), Values(QUDA_NORMOP_PC_SOLVE), Values(10),
                                 solution_accumulator_pipelines, no_schwarz, no_heavy_quark),
                         gettestname);

// Schwarz-preconditioned normal solves
INSTANTIATE_TEST_SUITE_P(SchwarzNormal, InvertTest,
                         Combine(precisions, sloppy_precisions, Values(QUDA_PCG_INVERTER),
                                 Values(QUDA_MATPCDAG_MATPC_SOLUTION), Values(QUDA_NORMOP_PC_SOLVE), Values(1),
                                 solution_accumulator_pipelines,
                                 Combine(Values(QUDA_ADDITIVE_SCHWARZ), Values(QUDA_CG_INVERTER, QUDA_CA_CG_INVERTER),
                                         Values(QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION)),
                                 no_heavy_quark),
                         gettestname);

// Schwarz-preconditioned direct solves
INSTANTIATE_TEST_SUITE_P(SchwarzEvenOdd, InvertTest,
                         Combine(precisions, sloppy_precisions, Values(QUDA_GCR_INVERTER), Values(QUDA_MATPC_SOLUTION),
                                 Values(QUDA_DIRECT_PC_SOLVE), Values(1), solution_accumulator_pipelines,
                                 Combine(Values(QUDA_ADDITIVE_SCHWARZ), Values(QUDA_MR_INVERTER, QUDA_CA_GCR_INVERTER),
                                         Values(QUDA_HALF_PRECISION, QUDA_QUARTER_PRECISION)),
                                 no_heavy_quark),
                         gettestname);

// Heavy-Quark preconditioned solves
INSTANTIATE_TEST_SUITE_P(HeavyQuarkEvenOdd, InvertTest,
                         Combine(precisions, sloppy_precisions, Values(QUDA_CG_INVERTER), Values(QUDA_MATPC_SOLUTION),
                                 Values(QUDA_NORMOP_PC_SOLVE), Values(1), solution_accumulator_pipelines, no_schwarz,
                                 Values(QUDA_L2_RELATIVE_RESIDUAL | QUDA_HEAVY_QUARK_RESIDUAL, QUDA_HEAVY_QUARK_RESIDUAL)),
                         gettestname);
