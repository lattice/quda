#include <invert_quda.h>
#include <blas_quda.h>
#include <eigen_helper.h>

namespace quda
{

  MinResExt::MinResExt(const DiracMatrix &mat, bool orthogonal, bool apply_mat, bool hermitian, TimeProfile &profile) :
    mat(mat), orthogonal(orthogonal), apply_mat(apply_mat), hermitian(hermitian), profile(profile)
  {
  }

  /* Solve the equation A p_k psi_k = b by minimizing the residual and
     using Eigen's SVD algorithm for numerical stability */
  void MinResExt::solve(std::vector<Complex> &psi_, std::vector<ColorSpinorField> &p, std::vector<ColorSpinorField> &q,
                        const ColorSpinorField &b, bool hermitian)
  {
    typedef Matrix<Complex, Dynamic, Dynamic> matrix;
    typedef Matrix<Complex, Dynamic, 1> vector;

    const int N = q.size();
    vector phi(N), psi(N);
    matrix A(N, N);

    // form the a Nx(N+1) matrix using only a single reduction - this
    // presently requires forgoing the matrix symmetry, but the improvement is well worth it

    std::vector<Complex> A_(N * (N + 1));

    if (hermitian) {
      // linear system is Hermitian, solve directly
      // compute rhs vector phi = P* b = (q_i, b) and construct the matrix
      // P* Q = P* A P = (p_i, q_j) = (p_i, A p_j)
      blas::cDotProduct(A_, p, {q, b});
    } else {
      // linear system is not Hermitian, solve the normal system
      // compute rhs vector phi = Q* b = (q_i, b) and construct the matrix
      // Q* Q = (A P)* (A P) = (q_i, q_j) = (A p_i, A p_j)
      blas::cDotProduct(A_, q, {q, b});
    }

    for (int i = 0; i < N; i++) {
      phi(i) = A_[i * (N + 1) + N];
      for (int j = 0; j < N; j++) { A(i, j) = A_[i * (N + 1) + j]; }
    }

    profile.TPSTOP(QUDA_PROFILE_CHRONO);
    profile.TPSTART(QUDA_PROFILE_EIGEN);

    LDLT<matrix> cholesky(A);
    psi = cholesky.solve(phi);

    profile.TPSTOP(QUDA_PROFILE_EIGEN);
    profile.TPSTART(QUDA_PROFILE_CHRONO);

    for (int i = 0; i < N; i++) psi_[i] = psi(i);
  }

  /*
    We want to find the best initial guess of the solution of
    A x = b, and we have N previous solutions x_i.
    The method goes something like this:

    1. Orthonormalise the p_i and q_i
    2. Form the matrix G_ij = x_i^dagger A x_j
    3. Form the vector B_i = x_i^dagger b
    4. solve A_ij a_j  = B_i
    5. x = a_i p_i
  */
  void MinResExt::operator()(ColorSpinorField &x, const ColorSpinorField &b, std::vector<ColorSpinorField> &p,
                             std::vector<ColorSpinorField> &q)
  {
    bool running = profile.isRunning(QUDA_PROFILE_CHRONO);
    if (!running) profile.TPSTART(QUDA_PROFILE_CHRONO);

    const int N = p.size();
    logQuda(QUDA_VERBOSE, "Constructing minimum residual extrapolation with basis size %d\n", N);

    if (N <= 1) {
      if (N == 0)
        blas::zero(x);
      else
        blas::copy(x, p[0]);
      if (!running) profile.TPSTOP(QUDA_PROFILE_CHRONO);
      return;
    }

    // Orthonormalise the vector basis
    if (orthogonal) {
      for (int i = 0; i < N; i++) {
        double p2 = blas::norm2(p[i]);
        blas::ax(1 / sqrt(p2), p[i]);
        if (!apply_mat) blas::ax(1 / sqrt(p2), q[i]);

        if (i + 1 < N) {
          std::vector<Complex> alpha(N - (i + 1));
          blas::cDotProduct(alpha, {p[i]}, {p.begin() + i + 1, p.end()});
          for (auto &a : alpha) a = -a;
          blas::caxpy(alpha, {p[i]}, {p.begin() + i + 1, p.end()});

          if (!apply_mat) {
            // if not applying the matrix below then orthogonalize q
            blas::caxpy(alpha, {q[i]}, {q.begin() + i + 1, q.end()});
          }
        }
      }
    }

    // if operator hasn't already been applied then apply
    if (apply_mat)
      for (int i = 0; i < N; i++) mat(q[i], p[i]);

    // Solution coefficient vectors
    std::vector<Complex> alpha(N);

    if (b.Precision() != p[0].Precision()) { // need to make a sloppy copy of b
      ColorSpinorParam param(b);
      param.setPrecision(p[0].Precision(), p[0].Precision(), true);
      param.create = QUDA_COPY_FIELD_CREATE;
      solve(alpha, p, q, ColorSpinorField(param), hermitian);
    } else {
      solve(alpha, p, q, b, hermitian);
    }

    blas::zero(x);
    blas::caxpy(alpha, p, x);

    if (getVerbosity() >= QUDA_SUMMARIZE) {
      // compute the residual only if we're going to print it
      ColorSpinorField r(b);
      for (auto &a : alpha) a = -a;
      blas::caxpy(alpha, q, r);
      printfQuda("MinResExt: N = %d, |res| / |src| = %e\n", N, sqrt(blas::norm2(r) / blas::norm2(b)));
    }

    if (!running) profile.TPSTOP(QUDA_PROFILE_CHRONO);
  }

} // namespace quda
