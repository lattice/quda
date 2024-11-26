#pragma once

#include <complex>

#include <gauge_field.h>

// declare omp reductions for std::complex:
#pragma omp declare reduction(+ : std::complex <double> : omp_out += omp_in) initializer(omp_priv = 0)
#pragma omp declare reduction(+ : std::complex <float> : omp_out += omp_in) initializer(omp_priv = 0)

struct fsu3_matrix {
  using real_t = float;
  using complex_t = std::complex<real_t>;
  complex_t e[3][3];
};

struct dsu3_matrix {
  using real_t = double;
  using complex_t = std::complex<real_t>;
  complex_t e[3][3];
};

struct fsu3_vector {
  using real_t = float;
  using complex_t = std::complex<real_t>;
  complex_t c[3];
};

struct dsu3_vector {
  using real_t = double;
  using complex_t = std::complex<real_t>;
  complex_t c[3];
};

struct fanti_hermitmat {
  using real_t = float;
  using complex_t = std::complex<real_t>;
  complex_t m01, m02, m12;
  real_t m00im, m11im, m22im;
  real_t space;
};

struct danti_hermitmat {
  using real_t = double;
  using complex_t = std::complex<real_t>;
  complex_t m01, m02, m12;
  real_t m00im, m11im, m22im;
  real_t space;
};

template <typename su3_matrix> su3_matrix *get_su3_matrix(quda::GaugeField &p, int idx, int dir)
{
  auto data = static_cast<su3_matrix *>(p.data(dir));
  return data + idx;
}

template <typename su3_vector, typename su3_matrix> void su3_projector(su3_vector *a, su3_vector *b, su3_matrix *c)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) c->e[i][j] = a->c[i] * std::conj(b->c[j]);
}

template <typename su3_matrix> void su3_adjoint(su3_matrix *a, su3_matrix *b)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { b->e[i][j] = std::conj(a->e[j][i]); }
  }
}

template <typename su3_matrix, typename anti_hermitmat> void make_anti_hermitian(su3_matrix *m3, anti_hermitmat *ah3)
{
  using complex = typename su3_matrix::complex_t;

  auto temp = (m3->e[0][0].imag() + m3->e[1][1].imag() + m3->e[2][2].imag()) * 0.33333333333333333;
  ah3->m00im = m3->e[0][0].imag() - temp;
  ah3->m11im = m3->e[1][1].imag() - temp;
  ah3->m22im = m3->e[2][2].imag() - temp;
  ah3->m01 = complex((m3->e[0][1].real() - m3->e[1][0].real()) * 0.5, (m3->e[0][1].imag() + m3->e[1][0].imag()) * 0.5);
  ah3->m02 = complex((m3->e[0][2].real() - m3->e[2][0].real()) * 0.5, (m3->e[0][2].imag() + m3->e[2][0].imag()) * 0.5);
  ah3->m12 = complex((m3->e[1][2].real() - m3->e[2][1].real()) * 0.5, (m3->e[1][2].imag() + m3->e[2][1].imag()) * 0.5);
}

template <typename anti_hermitmat, typename su3_matrix>
static void uncompress_anti_hermitian(anti_hermitmat *mat_antihermit, su3_matrix *mat_su3)
{
  using complex = typename su3_matrix::complex_t;

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

template <typename su3_matrix, typename real_t>
static void scalar_mult_sub_su3_matrix(su3_matrix *a, su3_matrix *b, real_t s, su3_matrix *c)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) c->e[i][j] = a->e[i][j] - s * b->e[i][j];
}

template <typename su3_matrix, typename real_t>
static void scalar_mult_add_su3_matrix(su3_matrix *a, su3_matrix *b, real_t s, su3_matrix *c)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) c->e[i][j] = a->e[i][j] + s * b->e[i][j];
}

template <typename su3_matrix> static void mult_su3_nn(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  using complex = typename su3_matrix::complex_t;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      complex x = 0;
      for (int k = 0; k < 3; k++) { x = x + a->e[i][k] * b->e[k][j]; }
      c->e[i][j] = x;
    }
  }
}

template <typename su3_matrix> static void mult_su3_an(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  using complex = typename su3_matrix::complex_t;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      complex x = 0;
      for (int k = 0; k < 3; k++) { x = x + std::conj(a->e[k][i]) * b->e[k][j]; }
      c->e[i][j] = x;
    }
  }
}

template <typename su3_matrix> static void mult_su3_na(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  using complex = typename su3_matrix::complex_t;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      complex x = 0;
      for (int k = 0; k < 3; k++) { x = x + a->e[i][k] * std::conj(b->e[j][k]); }
      c->e[i][j] = x;
    }
  }
}

template <typename su3_matrix, typename real_t> static void add_su3(su3_matrix *a, su3_matrix *b, real_t eb3)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { b->e[i][j] = b->e[i][j] + eb3 * a->e[i][j]; }
  }
}

template <typename su3_matrix> static typename su3_matrix::complex_t trace_su3(su3_matrix *a)
{
  return (a->e[0][0] + a->e[1][1] + a->e[2][2]);
}

template <typename su3_matrix> void print_su3_matrix(su3_matrix *a)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { printf("(%f %f)\t", a->e[i][j].real(), a->e[i][j].imag()); }
    printf("\n");
  }
}
