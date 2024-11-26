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
  auto data = static_cast<su3_matrix<real_t> *const>(p.data(dir));
  return data + idx;
}

template <typename real_t> const su3_matrix<real_t> *get_su3_matrix(const quda::GaugeField &p, int idx, int dir)
{
  auto data = static_cast<const su3_matrix<real_t> *const>(p.data(dir));
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
