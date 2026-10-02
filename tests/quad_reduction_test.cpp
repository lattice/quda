#include <quda_internal.h>
#include <gtest/gtest.h>
#include <type_traits>

#if defined(QUDA_TARGET_CUDA) && defined(QUDA_FPMP_DOUBLEDOUBLE)
#include <cuda/fpmp>
#if defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_LOW)
static_assert(std::is_same_v<doubledouble, cuda::experimental::fp64mp2_low>);
#elif defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_MID)
static_assert(std::is_same_v<doubledouble, cuda::experimental::fp64mp2_mid>);
#elif defined(QUDA_FPMP_DOUBLEDOUBLE_ACCURACY_HIGH)
static_assert(std::is_same_v<doubledouble, cuda::experimental::fp64mp2_high>);
#endif
#endif

namespace quda
{
#if defined(QUDA_REDUCTION_IS_FLOATFLOAT)
  TEST(QuadReduction, floatfloat_reduction_type)
  {
    // The reduction accuracy is selected independently of the bulk accuracy.
#if defined(QUDA_REDUCTION_FLOATFLOAT_ACCURACY_LOW)
    static_assert(std::is_same_v<reduction_t, floatfloat_low>);
#elif defined(QUDA_REDUCTION_FLOATFLOAT_ACCURACY_MID)
    static_assert(std::is_same_v<reduction_t, floatfloat_mid>);
#elif defined(QUDA_REDUCTION_FLOATFLOAT_ACCURACY_HIGH)
    static_assert(std::is_same_v<reduction_t, floatfloat_high>);
#endif
    static_assert(sizeof(reduction_t) == 2 * sizeof(float));
    static_assert(std::is_trivially_copyable_v<reduction_t>);

    const reduction_t x(1.0, 1e-8);
    EXPECT_EQ(x.hi(), 1.0f);
    EXPECT_EQ(x.lo(), 1e-8f);
  }
#endif

#if defined(QUDA_REDUCTION_IS_DOUBLEDOUBLE) && defined(QUDA_FPMP_DOUBLEDOUBLE)
  TEST(QuadReduction, doubledouble_reduction_type)
  {
#if defined(QUDA_REDUCTION_DOUBLEDOUBLE_ACCURACY_LOW)
    static_assert(std::is_same_v<reduction_t, doubledouble_low>);
#elif defined(QUDA_REDUCTION_DOUBLEDOUBLE_ACCURACY_MID)
    static_assert(std::is_same_v<reduction_t, doubledouble_mid>);
#elif defined(QUDA_REDUCTION_DOUBLEDOUBLE_ACCURACY_HIGH)
    static_assert(std::is_same_v<reduction_t, doubledouble_high>);
#endif
    static_assert(sizeof(reduction_t) == 2 * sizeof(double));
    static_assert(std::is_trivially_copyable_v<reduction_t>);

    const reduction_t x(1.0, 1e-20);
    EXPECT_EQ(x.hi(), 1.0);
    EXPECT_EQ(x.lo(), 1e-20);
  }
#endif

  TEST(QuadReduction, doubledouble_layout_and_components)
  {
    static_assert(sizeof(doubledouble) == 2 * sizeof(double));
    static_assert(std::is_trivially_copyable_v<doubledouble>);

    const doubledouble x(1.0, 1e-20);
    EXPECT_EQ(x.hi(), 1.0);
    EXPECT_EQ(x.lo(), 1e-20);
  }
} // namespace quda

#ifdef QUDA_USE_QUAD_SCALAR

#include <quad_scalar_test_utils.h>

namespace quda
{
  TEST(QuadReduction, doubledouble_to_real_uses_tail)
  {
    const doubledouble x(1.0, 1e-20);
    const float128_t from_dd = static_cast<float128_t>(x);
    const float128_t from_head = static_cast<float128_t>(x.hi());
    const float128_t ref = static_cast<float128_t>(1.0) + static_cast<float128_t>(1e-20);
    const float128_t tol = static_cast<float128_t>(1e-30);

    EXPECT_LT(fp128::fabs(from_dd - ref), tol);
    EXPECT_GT(fp128::fabs(from_dd - from_head), static_cast<float128_t>(1e-25));
    EXPECT_EQ(from_dd, static_cast<float128_t>(x));
  }

  TEST(QuadReduction, doubledouble_to_real_large_cancellation_sum)
  {
    const float128_t ref = static_cast<float128_t>(1e16) + static_cast<float128_t>(1.0);
    const doubledouble x(static_cast<double>(1e16), static_cast<double>(1.0));
    const float128_t got = static_cast<float128_t>(x);

    const double head_only = x.hi();
    EXPECT_GT(fp128::fabs(got - static_cast<float128_t>(head_only)), static_cast<float128_t>(1e-6));
    EXPECT_LT(fp128::fabs(got - ref) / ref, static_cast<float128_t>(1e-30));
  }

  TEST(QuadReduction, rel_error_in_quad_precision)
  {
    const float128_t a = static_cast<float128_t>(1.0) + static_cast<float128_t>(1e-20);
    const float128_t b = static_cast<float128_t>(1.0);
    const float128_t err = rel_error(a, b);
    const double err_d = to_double(err);

    const float128_t target = static_cast<float128_t>(1e-20);
    EXPECT_LT(fp128::fabs(err - target) / target, static_cast<float128_t>(1e-10));
    // rel_error keeps ~1e-20 in float128_t; folding through double would crush this to 0
    EXPECT_GT(err_d, 1e-21);
    EXPECT_LT(err_d, 1e-15);
  }

} // namespace quda

#endif // QUDA_USE_QUAD_SCALAR

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
