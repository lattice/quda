#pragma once

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
};

template <int N, class T> Matrix<N, T>::Matrix()
{
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) { data[i][j] = static_cast<T>(0); }
  }
}

template <int N, class T> T &Matrix<N, T>::operator()(int i, int j) { return data[i][j]; }

template <int N, class T> const T &Matrix<N, T>::operator()(int i, int j) const { return data[i][j]; }

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
  T tr;
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
