#pragma once

#include <cmath>
#include <complex>

#include <gauge_field.h>

// declare omp reductions for std::complex:
#pragma omp declare reduction(+ : std::complex <double> : omp_out += omp_in) initializer(omp_priv = 0)
#pragma omp declare reduction(+ : std::complex <float> : omp_out += omp_in) initializer(omp_priv = 0)

template <typename real_t_> struct su3_matrix {
  using real_t = real_t_;
  using complex_t = std::complex<real_t>;
  complex_t e[3][3];
};

template <typename real_t_> struct su3_vector {
  using real_t = real_t_;
  using complex_t = std::complex<real_t>;
  complex_t c[3];
};

template <typename real_t_> struct anti_hermitmat {
  using real_t = real_t_;
  using complex_t = std::complex<real_t>;
  complex_t m01, m02, m12;
  real_t m00im, m11im, m22im;
  real_t space;
};

template <typename real_t> su3_matrix<real_t> *get_su3_matrix(quda::GaugeField &p, int idx, int dir)
{
  auto data = static_cast<su3_matrix<real_t> *>(p.data(dir));
  return data + idx;
}

template <typename real_t> const su3_matrix<real_t> *get_su3_matrix(const quda::GaugeField &p, int idx, int dir)
{
  auto data = static_cast<const su3_matrix<real_t> *>(p.data(dir));
  return data + idx;
}

template <typename real_t>
void su3_projector(const su3_vector<real_t> *a, const su3_vector<real_t> *b, su3_matrix<real_t> *c)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) c->e[i][j] = a->c[i] * std::conj(b->c[j]);
}

template <typename real_t> void su3_adjoint(const su3_matrix<real_t> *a, su3_matrix<real_t> *b)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { b->e[i][j] = std::conj(a->e[j][i]); }
  }
}

template <typename real_t> void make_anti_hermitian(const su3_matrix<real_t> *m3, anti_hermitmat<real_t> *ah3)
{
  using complex = std::complex<real_t>;

  auto temp = (m3->e[0][0].imag() + m3->e[1][1].imag() + m3->e[2][2].imag()) * 0.33333333333333333;
  ah3->m00im = m3->e[0][0].imag() - temp;
  ah3->m11im = m3->e[1][1].imag() - temp;
  ah3->m22im = m3->e[2][2].imag() - temp;
  ah3->m01 = complex((m3->e[0][1].real() - m3->e[1][0].real()) * 0.5, (m3->e[0][1].imag() + m3->e[1][0].imag()) * 0.5);
  ah3->m02 = complex((m3->e[0][2].real() - m3->e[2][0].real()) * 0.5, (m3->e[0][2].imag() + m3->e[2][0].imag()) * 0.5);
  ah3->m12 = complex((m3->e[1][2].real() - m3->e[2][1].real()) * 0.5, (m3->e[1][2].imag() + m3->e[2][1].imag()) * 0.5);
}

template <typename real_t>
static void uncompress_anti_hermitian(const anti_hermitmat<real_t> *mat_antihermit, su3_matrix<real_t> *mat_su3)
{
  using complex = std::complex<real_t>;

  mat_su3->e[0][0] = complex(0, mat_antihermit->m00im);
  mat_su3->e[1][1] = complex(0, mat_antihermit->m11im);
  mat_su3->e[2][2] = complex(0, mat_antihermit->m22im);

  mat_su3->e[0][1] = mat_antihermit->m01;
  mat_su3->e[1][0] = -std::conj(mat_antihermit->m01);

  mat_su3->e[0][2] = mat_antihermit->m02;
  mat_su3->e[2][0] = -std::conj(mat_antihermit->m02);

  mat_su3->e[1][2] = mat_antihermit->m12;
  mat_su3->e[2][1] = -std::conj(mat_antihermit->m12);
}

template <typename real_t>
void scalar_mult_sub_su3_matrix(const su3_matrix<real_t> *a, const su3_matrix<real_t> *b, real_t s, su3_matrix<real_t> *c)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) c->e[i][j] = a->e[i][j] - s * b->e[i][j];
}

template <typename real_t>
void scalar_mult_add_su3_matrix(const su3_matrix<real_t> *a, const su3_matrix<real_t> *b, real_t s, su3_matrix<real_t> *c)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) c->e[i][j] = a->e[i][j] + s * b->e[i][j];
}

template <typename real_t>
void mult_su3_nn(const su3_matrix<real_t> *a, const su3_matrix<real_t> *b, su3_matrix<real_t> *c)
{
  using complex = std::complex<real_t>;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      complex x = 0;
      for (int k = 0; k < 3; k++) { x = x + a->e[i][k] * b->e[k][j]; }
      c->e[i][j] = x;
    }
  }
}

template <typename real_t>
void mult_su3_an(const su3_matrix<real_t> *a, const su3_matrix<real_t> *b, su3_matrix<real_t> *c)
{
  using complex = std::complex<real_t>;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      complex x = 0;
      for (int k = 0; k < 3; k++) { x = x + std::conj(a->e[k][i]) * b->e[k][j]; }
      c->e[i][j] = x;
    }
  }
}

template <typename real_t>
void mult_su3_na(const su3_matrix<real_t> *a, const su3_matrix<real_t> *b, su3_matrix<real_t> *c)
{
  using complex = std::complex<real_t>;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      complex x = 0;
      for (int k = 0; k < 3; k++) { x = x + a->e[i][k] * std::conj(b->e[j][k]); }
      c->e[i][j] = x;
    }
  }
}

template <typename real_t> void add_su3(const su3_matrix<real_t> *a, su3_matrix<real_t> *b, real_t eb3)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { b->e[i][j] = b->e[i][j] + eb3 * a->e[i][j]; }
  }
}

template <typename real_t> std::complex<real_t> trace_su3(const su3_matrix<real_t> *a)
{
  return (a->e[0][0] + a->e[1][1] + a->e[2][2]);
}

template <typename real_t> void print_su3_matrix(const su3_matrix<real_t> *a)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { printf("(%f %f)\t", a->e[i][j].real(), a->e[i][j].imag()); }
    printf("\n");
  }
}

template <class T, class U> struct Promote {
  typedef T Type;
};

template <> struct Promote<int, float> {
  typedef float Type;
};

template <> struct Promote<float, int> {
  typedef float Type;
};

template <> struct Promote<int, double> {
  typedef double Type;
};

template <> struct Promote<double, int> {
  typedef double Type;
};

template <> struct Promote<float, double> {
  typedef double Type;
};

template <> struct Promote<double, float> {
  typedef double Type;
};

template <> struct Promote<int, std::complex<float>> {
  typedef std::complex<float> Type;
};

template <> struct Promote<std::complex<float>, int> {
  typedef std::complex<float> Type;
};

template <> struct Promote<float, std::complex<float>> {
  typedef std::complex<float> Type;
};

template <> struct Promote<int, std::complex<double>> {
  typedef std::complex<double> Type;
};

template <> struct Promote<std::complex<double>, int> {
  typedef std::complex<double> Type;
};

template <> struct Promote<float, std::complex<double>> {
  typedef std::complex<double> Type;
};

template <> struct Promote<std::complex<double>, float> {
  typedef std::complex<double> Type;
};

template <> struct Promote<double, std::complex<double>> {
  typedef std::complex<double> Type;
};

template <> struct Promote<std::complex<double>, double> {
  typedef std::complex<double> Type;
};

template <int N, class T> class Matrix
{
private:
  T data[N][N];

public:
  Matrix(); // default constructor
  Matrix(const Matrix<N, T> &) = default;
  Matrix(Matrix<N, T> &&) = default;
  Matrix &operator=(const Matrix<N, T> &) = default;
  Matrix &operator=(Matrix<N, T> &&) = default;
  Matrix &operator+=(const Matrix<N, T> &mat);
  Matrix &operator-=(const Matrix<N, T> &mat);
  const T &operator()(int i, int j) const;
  T &operator()(int i, int j);
  T determinant() const;
  Matrix inverse() const;
};

template <int N, class T> Matrix<N, T>::Matrix()
{
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { data[i][j] = static_cast<T>(0); }
  }
}

template <int N, class T> T &Matrix<N, T>::operator()(int i, int j) { return data[i][j]; }

template <int N, class T> const T &Matrix<N, T>::operator()(int i, int j) const { return data[i][j]; }

template <int N, class T> T Matrix<N, T>::determinant() const
{
  static_assert(N == 3, "Matrix::determinant is implemented only for 3x3 matrices");
  return (*this)(0, 0) * ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1))
    - (*this)(0, 1) * ((*this)(1, 0) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 0))
    + (*this)(0, 2) * ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0));
}

template <int N, class T> Matrix<N, T> Matrix<N, T>::inverse() const
{
  static_assert(N == 3, "Matrix::inverse is implemented only for 3x3 matrices");
  Matrix<N, T> out;
  const auto det = determinant();
  out(0, 0) = ((*this)(1, 1) * (*this)(2, 2) - (*this)(1, 2) * (*this)(2, 1)) / det;
  out(0, 1) = ((*this)(0, 2) * (*this)(2, 1) - (*this)(0, 1) * (*this)(2, 2)) / det;
  out(0, 2) = ((*this)(0, 1) * (*this)(1, 2) - (*this)(0, 2) * (*this)(1, 1)) / det;
  out(1, 0) = ((*this)(1, 2) * (*this)(2, 0) - (*this)(1, 0) * (*this)(2, 2)) / det;
  out(1, 1) = ((*this)(0, 0) * (*this)(2, 2) - (*this)(0, 2) * (*this)(2, 0)) / det;
  out(1, 2) = ((*this)(0, 2) * (*this)(1, 0) - (*this)(0, 0) * (*this)(1, 2)) / det;
  out(2, 0) = ((*this)(1, 0) * (*this)(2, 1) - (*this)(1, 1) * (*this)(2, 0)) / det;
  out(2, 1) = ((*this)(0, 1) * (*this)(2, 0) - (*this)(0, 0) * (*this)(2, 1)) / det;
  out(2, 2) = ((*this)(0, 0) * (*this)(1, 1) - (*this)(0, 1) * (*this)(1, 0)) / det;
  return out;
}

template <int N, class T> Matrix<N, T> &Matrix<N, T>::operator+=(const Matrix<N, T> &mat)
{
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { data[i][j] += mat.data[i][j]; }
  }
  return *this;
}

template <int N, class T> Matrix<N, T> &Matrix<N, T>::operator-=(const Matrix<N, T> &mat)
{
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { data[i][j] -= mat.data[i][j]; }
  }
  return *this;
}

template <int N, class T> Matrix<N, T> operator+(const Matrix<N, T> &a, const Matrix<N, T> &b)
{
  Matrix<N, T> result(a);
  result += b;
  return result;
}

template <int N, class T> Matrix<N, T> operator-(const Matrix<N, T> &a, const Matrix<N, T> &b)
{
  Matrix<N, T> result(a);
  result -= b;
  return result;
}

template <int N, class T> Matrix<N, T> operator*(const Matrix<N, T> &a, const Matrix<N, T> &b)
{
  Matrix<N, T> result;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      result(i, j) = static_cast<T>(0);
      for (int k = 0; k < N; ++k) { result(i, j) += a(i, k) * b(k, j); }
    }
  }
  return result;
}

template <int N, class T> Matrix<N, std::complex<T>> conj(const Matrix<N, std::complex<T>> &mat)
{
  Matrix<N, std::complex<T>> result;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { result(i, j) = std::conj(mat(j, i)); }
  }
  return result;
}

/**
 * @brief Replace a matrix with its traceless Hermitian projection.
 *
 * @param[in,out] m Matrix to project.
 */
template <typename real_t> void make_herm(Matrix<3, std::complex<real_t>> &m)
{
  auto anti_hermitian = conj(m) - m;
  real_t trace = 0;
  for (int i = 0; i < 3; i++) trace += anti_hermitian(i, i).imag();
  for (int i = 0; i < 3; i++) anti_hermitian(i, i).imag(anti_hermitian(i, i).imag() - trace / 3);
  m = std::complex<real_t>(0, static_cast<real_t>(0.5)) * anti_hermitian;
}

/**
 * @brief Compute exp(i Q) for a traceless Hermitian SU(3) generator.
 *
 * @param[in] q Traceless Hermitian generator.
 * @return Matrix exponential exp(i Q).
 */
template <typename real_t> Matrix<3, std::complex<real_t>> exponentiate_iQ(const Matrix<3, std::complex<real_t>> &q)
{
  using complex = std::complex<real_t>;
  constexpr real_t inv3 = static_cast<real_t>(1.0 / 3.0);
  constexpr real_t inv_pi = static_cast<real_t>(1.0 / M_PI);
  constexpr real_t inv_3pi = static_cast<real_t>(1.0 / (3.0 * M_PI));

  const auto q2 = q * q;
  real_t c0 = q.determinant().real();
  const real_t c1 = static_cast<real_t>(0.5) * trace(q2).real();
  const real_t sqrt_c1_inv3 = std::sqrt(c1 * inv3);
  const real_t c0_max = 2 * c1 * inv3 * sqrt_c1_inv3;
  Matrix<3, complex> identity;
  for (int i = 0; i < 3; i++) identity(i, i) = static_cast<real_t>(1.0);

  if (c1 == 0) return identity;

  int parity = 0;
  if (c0 < 0) {
    c0 = -c0;
    parity = 1;
  }

  const real_t theta = std::acos(c0 / c0_max);
  const real_t u = std::cos(theta * inv_3pi * static_cast<real_t>(M_PI)) * sqrt_c1_inv3;
  const real_t w = std::sin(theta * inv_3pi * static_cast<real_t>(M_PI)) * std::sqrt(c1);
  const real_t u_sq = u * u;
  const real_t w_sq = w * w;
  const real_t denom_inv = static_cast<real_t>(1.0) / (9 * u_sq - w_sq);
  const real_t exp_iu_re = std::cos(u * inv_pi * static_cast<real_t>(M_PI));
  const real_t exp_iu_im = std::sin(u * inv_pi * static_cast<real_t>(M_PI));
  const real_t exp_2iu_re = exp_iu_re * exp_iu_re - exp_iu_im * exp_iu_im;
  const real_t exp_2iu_im = 2 * exp_iu_re * exp_iu_im;
  const real_t cos_w = std::cos(w * inv_pi * static_cast<real_t>(M_PI));
  const real_t sinc_w = std::abs(w) < static_cast<real_t>(0.05) ?
    static_cast<real_t>(1.0) - w_sq / 6 * (static_cast<real_t>(1.0) - w_sq * static_cast<real_t>(0.05)
                                             * (static_cast<real_t>(1.0) - w_sq / 42
                                                * (static_cast<real_t>(1.0) - w_sq / 72))) :
    std::sin(w * inv_pi * static_cast<real_t>(M_PI)) / w;

  real_t h_re = (u_sq - w_sq) * exp_2iu_re + 8 * u_sq * cos_w * exp_iu_re
    + 2 * u * (3 * u_sq + w_sq) * sinc_w * exp_iu_im;
  real_t h_im = (u_sq - w_sq) * exp_2iu_im - 8 * u_sq * cos_w * exp_iu_im
    + 2 * u * (3 * u_sq + w_sq) * sinc_w * exp_iu_re;
  complex f0(h_re * denom_inv, h_im * denom_inv);

  h_re = 2 * u * exp_2iu_re - 2 * u * cos_w * exp_iu_re + (3 * u_sq - w_sq) * sinc_w * exp_iu_im;
  h_im = 2 * u * exp_2iu_im + 2 * u * cos_w * exp_iu_im + (3 * u_sq - w_sq) * sinc_w * exp_iu_re;
  complex f1(h_re * denom_inv, h_im * denom_inv);

  h_re = exp_2iu_re - cos_w * exp_iu_re - 3 * u * sinc_w * exp_iu_im;
  h_im = exp_2iu_im + cos_w * exp_iu_im - 3 * u * sinc_w * exp_iu_re;
  complex f2(h_re * denom_inv, h_im * denom_inv);

  if (parity) {
    f0.imag(-f0.imag());
    f1.real(-f1.real());
    f2.imag(-f2.imag());
  }

  return f0 * identity + f1 * q + f2 * q2;
}

template <int N, class T> Matrix<N, T> transpose(const Matrix<N, std::complex<T>> &mat)
{
  Matrix<N, T> result;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { result(i, j) = mat(j, i); }
  }
  return result;
}

template <int N, class T> T trace(const Matrix<N, T> &mat)
{
  T tr {};
  for (int i = 0; i < N; i++) tr += mat(i, i);

  return tr;
}

template <int N, class T, class U>
Matrix<N, typename Promote<T, U>::Type> operator*(const Matrix<N, T> &mat, const U &scalar)
{
  typedef typename Promote<T, U>::Type return_type;
  Matrix<N, return_type> result;

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { result(i, j) = scalar * mat(i, j); }
  }
  return result;
}

template <int N, class T, class U>
Matrix<N, typename Promote<T, U>::Type> operator*(const U &scalar, const Matrix<N, T> &mat)
{
  return mat * scalar;
}

template <int N, class T> struct Identity {
  Matrix<N, T> operator()() const
  {
    Matrix<N, T> id;
    for (int i = 0; i < N; ++i) { id(i, i) = static_cast<T>(1); }
    return id;
  } // operator()
};

template <int N, class T> struct Zero {
  // the default constructor zeros all matrix elements
  Matrix<N, T> operator()() const { return Matrix<N, T>(); }
};

template <int N, class T> std::ostream &operator<<(std::ostream &os, const Matrix<N, T> &m)
{
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { os << m(i, j) << " "; }
    if (i < N - 1) os << std::endl;
  }
  return os;
}
