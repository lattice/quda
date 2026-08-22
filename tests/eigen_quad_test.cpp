#include <quda_internal.h>
#include <eigen_helper.h>
#include <gtest/gtest.h>

#ifdef QUDA_USE_QUAD_SCALAR

#include <complex>
#include <algorithm>
#include <quad_scalar_test_utils.h>

namespace quda
{

  using T = float128_t;
  using C = std::complex<T>;
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using RealVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

  // TRLM-style arrow matrix used historically to expose bad Eigen eigenvectors
  // for float128_t (residuals ~1e-2).
  template <typename R> Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> make_arrow_matrix()
  {
    const int dim = 8;
    const int arrow_pos = 4;
    const double alpha_d[8] = {0.186, 0.191, 0.231, 0.274, 0.285, 0.325, 0.363, 0.411};
    const double beta_d[8] = {0.0049, 0.0348, 0.0119, 0.0414, 0.0304, 0.0272, 0.0664, 0.0687};

    Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> A
      = Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>::Zero(dim, dim);
    for (int i = 0; i < dim; i++) A(i, i) = static_cast<R>(alpha_d[i]);
    for (int i = 0; i < arrow_pos; i++) {
      A(i, arrow_pos) = static_cast<R>(beta_d[i]);
      A(arrow_pos, i) = static_cast<R>(beta_d[i]);
    }
    for (int i = arrow_pos; i < dim - 1; i++) {
      A(i, i + 1) = static_cast<R>(beta_d[i]);
      A(i + 1, i) = static_cast<R>(beta_d[i]);
    }
    return A;
  }

  template <typename R> Eigen::Matrix<std::complex<R>, Eigen::Dynamic, Eigen::Dynamic> make_hermitian_matrix()
  {
    using Complex = std::complex<R>;
    const int dim = 6;
    Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> A
      = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>::Zero(dim, dim);
    for (int i = 0; i < dim; i++) A(i, i) = Complex(static_cast<R>(0.2 + 0.13 * i), static_cast<R>(0));
    for (int i = 0; i < dim - 1; i++) {
      Complex v(static_cast<R>(0.03 + 0.01 * i), static_cast<R>(0.02 - 0.004 * i));
      A(i, i + 1) = v;
      A(i + 1, i) = std::conj(v);
    }
    A(0, dim - 1) = Complex(static_cast<R>(0.017), static_cast<R>(-0.009));
    A(dim - 1, 0) = std::conj(A(0, dim - 1));
    return A;
  }

  template <typename R>
  double max_eigenpair_residual(const Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic> &A,
                                const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<R, Eigen::Dynamic, Eigen::Dynamic>> &es)
  {
    const int dim = A.rows();
    double max_res = 0.0;
    for (int i = 0; i < dim; i++) {
      Eigen::Matrix<R, Eigen::Dynamic, 1> x = es.eigenvectors().col(i);
      R lam = es.eigenvalues()[i];
      Eigen::Matrix<R, Eigen::Dynamic, 1> r = A * x - lam * x;
      const double res = static_cast<double>(std::sqrt(static_cast<R>(r.dot(r))));
      max_res = std::max(max_res, res);
    }
    return max_res;
  }

  template <typename R>
  double max_eigenpair_residual(
    const Eigen::Matrix<std::complex<R>, Eigen::Dynamic, Eigen::Dynamic> &A,
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<std::complex<R>, Eigen::Dynamic, Eigen::Dynamic>> &es)
  {
    using Complex = std::complex<R>;
    const int dim = A.rows();
    double max_res = 0.0;
    for (int i = 0; i < dim; i++) {
      Eigen::Matrix<Complex, Eigen::Dynamic, 1> x = es.eigenvectors().col(i);
      R lam = es.eigenvalues()[i];
      Eigen::Matrix<Complex, Eigen::Dynamic, 1> r = A * x - Complex(lam) * x;
      const double res = static_cast<double>(std::sqrt(static_cast<R>(r.dot(r).real())));
      max_res = std::max(max_res, res);
    }
    return max_res;
  }

  double frobenius(const Mat &M) { return static_cast<double>(M.norm()); }

  // ---------------------------------------------------------------------------
  // NumTraits diagnostics
  // ---------------------------------------------------------------------------

  TEST(EigenQuad, NumTraits_epsilon_matches_binary128)
  {
    const T eps = Eigen::NumTraits<T>::epsilon();
    const T expected = std::pow(static_cast<T>(2), static_cast<T>(-112));
    EXPECT_EQ(eps, expected);
  }

  TEST(EigenQuad, NumTraits_highest_is_finite_max)
  {
    const T hi = Eigen::NumTraits<T>::highest();
    const T lo = Eigen::NumTraits<T>::lowest();
    const T two = static_cast<T>(2);
    const T expected = (two - std::pow(two, static_cast<T>(-112))) * std::pow(two, static_cast<T>(16383));
    EXPECT_TRUE(std::isfinite(hi));
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_FALSE(std::isinf(hi));
    EXPECT_FALSE(std::isinf(lo));
    EXPECT_EQ(hi, expected);
    EXPECT_EQ(lo, -expected);
  }

  TEST(EigenQuad, NumTraits_IsSigned)
  {
    // Portable workaround: do not rely on std::numeric_limits<float128_t>::is_signed.
    // Without IsSigned=1, Eigen's numext::abs<float128_t> is a no-op (unsigned path).
    EXPECT_EQ(Eigen::NumTraits<T>::IsSigned, 1);
    const T x = static_cast<T>(-2.5);
    EXPECT_EQ(Eigen::numext::abs(x), static_cast<T>(2.5));
  }

  // ---------------------------------------------------------------------------
  // Stage splits: tridiagonalization vs QR vs full solve
  // ---------------------------------------------------------------------------

  TEST(EigenQuad, stage_tridiagonalization_residual)
  {
    const Mat A = make_arrow_matrix<T>();
    Eigen::Tridiagonalization<Mat> tri(A);
    const Mat Q = tri.matrixQ();
    const Mat Tmat = tri.matrixT();

    // Q should be orthogonal and A ≈ Q T Q^T
    const Mat QtQ = Q.transpose() * Q;
    const Mat recon = Q * Tmat * Q.transpose();
    const double orth = frobenius(QtQ - Mat::Identity(A.rows(), A.cols()));
    const double recon_err = frobenius(A - recon) / frobenius(A);

    EXPECT_LT(orth, 1e-12) << "||Q^T Q - I||_F = " << orth;
    EXPECT_LT(recon_err, 1e-12) << "||A - Q T Q^T||_F / ||A||_F = " << recon_err;

    // Strict tridiagonal structure (above the first superdiagonal must be ~0)
    double off = 0.0;
    for (int i = 0; i < Tmat.rows(); i++) {
      for (int j = 0; j < Tmat.cols(); j++) {
        if (std::abs(i - j) > 1) off = std::max(off, static_cast<double>(std::abs(Tmat(i, j))));
      }
    }
    EXPECT_LT(off, 1e-30) << "max |T[i,j]| for |i-j|>1 = " << off;
  }

  TEST(EigenQuad, stage_QR_from_known_tridiagonal)
  {
    // Build a well-conditioned real tridiagonal in double, cast to quad, then
    // run only Eigen's QR-from-tridiagonal stage.
    const int n = 8;
    Eigen::Matrix<double, Eigen::Dynamic, 1> diag_d(n), sub_d(n - 1);
    for (int i = 0; i < n; i++) diag_d[i] = 0.2 + 0.07 * i;
    for (int i = 0; i < n - 1; i++) sub_d[i] = 0.03 + 0.01 * i;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> esd;
    esd.computeFromTridiagonal(diag_d, sub_d);
    ASSERT_EQ(esd.info(), Eigen::Success);

    RealVec diag = diag_d.cast<T>();
    RealVec sub = sub_d.cast<T>();
    Eigen::SelfAdjointEigenSolver<Mat> es;
    es.computeFromTridiagonal(diag, sub);
    ASSERT_EQ(es.info(), Eigen::Success);

    Mat Tcheck = Mat::Zero(n, n);
    for (int i = 0; i < n; i++) Tcheck(i, i) = static_cast<T>(diag_d[i]);
    for (int i = 0; i < n - 1; i++) {
      Tcheck(i, i + 1) = static_cast<T>(sub_d[i]);
      Tcheck(i + 1, i) = static_cast<T>(sub_d[i]);
    }

    const double max_res = max_eigenpair_residual(Tcheck, es);
    EXPECT_LT(max_res, 1e-12) << "QR-from-tridiagonal max ||Tx-lx|| = " << max_res;

    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(static_cast<double>(es.eigenvalues()[i]), esd.eigenvalues()[i], 1e-12);
    }
  }

  TEST(EigenQuad, stage_eigenvalues_only)
  {
    const Mat A = make_arrow_matrix<T>();
    Eigen::SelfAdjointEigenSolver<Mat> es;
    es.compute(A, Eigen::EigenvaluesOnly);
    ASSERT_EQ(es.info(), Eigen::Success);

    auto Ad = make_arrow_matrix<double>();
    Eigen::SelfAdjointEigenSolver<decltype(Ad)> esd;
    esd.compute(Ad, Eigen::EigenvaluesOnly);
    ASSERT_EQ(esd.info(), Eigen::Success);

    for (int i = 0; i < A.rows(); i++) {
      EXPECT_NEAR(static_cast<double>(es.eigenvalues()[i]), esd.eigenvalues()[i], 1e-12)
        << "eigenvalue[" << i << "]";
    }
  }

  TEST(EigenQuad, SelfAdjointEigenSolver_real_arrow_residuals)
  {
    auto A = make_arrow_matrix<T>();
    Eigen::SelfAdjointEigenSolver<decltype(A)> es;
    es.compute(A);
    ASSERT_EQ(es.info(), Eigen::Success);

    const double max_res = max_eigenpair_residual(A, es);
    EXPECT_LT(max_res, 1e-12) << "max ||Ax-lx|| = " << max_res;

    auto Ad = make_arrow_matrix<double>();
    Eigen::SelfAdjointEigenSolver<decltype(Ad)> esd;
    esd.compute(Ad);
    ASSERT_EQ(esd.info(), Eigen::Success);
    for (int i = 0; i < A.rows(); i++) {
      EXPECT_NEAR(static_cast<double>(es.eigenvalues()[i]), esd.eigenvalues()[i], 1e-12);
    }
  }

  TEST(EigenQuad, SelfAdjointEigenSolver_hermitian_residuals)
  {
    auto A = make_hermitian_matrix<T>();
    Eigen::SelfAdjointEigenSolver<decltype(A)> es(A);
    ASSERT_EQ(es.info(), Eigen::Success);

    const double max_res = max_eigenpair_residual(A, es);
    EXPECT_LT(max_res, 1e-12) << "max ||Ax-lx|| = " << max_res;

    auto Ad = make_hermitian_matrix<double>();
    Eigen::SelfAdjointEigenSolver<decltype(Ad)> esd(Ad);
    ASSERT_EQ(esd.info(), Eigen::Success);
    for (int i = 0; i < A.rows(); i++) {
      EXPECT_NEAR(static_cast<double>(es.eigenvalues()[i]), esd.eigenvalues()[i], 1e-12);
    }
  }

  TEST(EigenQuad, SelfAdjointEigenSolver_double_reference_still_tight)
  {
    auto A = make_arrow_matrix<double>();
    Eigen::SelfAdjointEigenSolver<decltype(A)> es;
    es.compute(A);
    ASSERT_EQ(es.info(), Eigen::Success);
    EXPECT_LT(max_eigenpair_residual(A, es), 1e-14);
  }

} // namespace quda

#endif // QUDA_USE_QUAD_SCALAR

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
