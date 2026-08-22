#include <quda_internal.h>
#include <gtest/gtest.h>

#ifdef QUDA_USE_QUAD_SCALAR

#include <quadmath.h>
#include <quad_scalar_test_utils.h>

namespace quda
{

  TEST(QuadReduction, doubledouble_to_real_uses_tail)
  {
    const doubledouble x(1.0, 1e-20);
    const __float128 from_dd = static_cast<__float128>(x);
    const __float128 from_head = static_cast<__float128>(x.head());
    const __float128 ref = static_cast<__float128>(1.0) + static_cast<__float128>(1e-20);
    const __float128 tol = static_cast<__float128>(1e-30);

    EXPECT_LT(fabsq(from_dd - ref), tol);
    EXPECT_GT(fabsq(from_dd - from_head), static_cast<__float128>(1e-25));
    EXPECT_EQ(from_dd, static_cast<__float128>(x));
  }

  TEST(QuadReduction, doubledouble_to_real_large_cancellation_sum)
  {
    const __float128 ref = static_cast<__float128>(1e16) + static_cast<__float128>(1.0);
    const doubledouble x(static_cast<double>(1e16), static_cast<double>(1.0));
    const __float128 got = static_cast<__float128>(x);

    const double head_only = x.head();
    EXPECT_GT(fabsq(got - static_cast<__float128>(head_only)), static_cast<__float128>(1e-6));
    EXPECT_LT(fabsq(got - ref) / ref, static_cast<__float128>(1e-30));
  }

  TEST(QuadReduction, rel_error_in_quad_precision)
  {
    const __float128 a = static_cast<__float128>(1.0) + static_cast<__float128>(1e-20);
    const __float128 b = static_cast<__float128>(1.0);
    const __float128 err = rel_error(a, b);
    const double err_d = to_double(err);

    const __float128 target = static_cast<__float128>(1e-20);
    EXPECT_LT(fabsq(err - target) / target, static_cast<__float128>(1e-10));
    // rel_error keeps ~1e-20 in __float128; folding through double would crush this to 0
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
