#pragma once

#include <gauge_field.h>

#define CADD(a, b, c)                                                                                                  \
  {                                                                                                                    \
    (c).real = (a).real + (b).real;                                                                                    \
    (c).imag = (a).imag + (b).imag;                                                                                    \
  }
#define CMUL(a, b, c)                                                                                                  \
  {                                                                                                                    \
    (c).real = (a).real * (b).real - (a).imag * (b).imag;                                                              \
    (c).imag = (a).real * (b).imag + (a).imag * (b).real;                                                              \
  }
#define CSUM(a, b)                                                                                                     \
  {                                                                                                                    \
    (a).real += (b).real;                                                                                              \
    (a).imag += (b).imag;                                                                                              \
  }

/* rescale by real scalar */
#define CSCALE(a, b)                                                                                                   \
  {                                                                                                                    \
    (a).real *= b;                                                                                                     \
    (a).imag *= b;                                                                                                     \
  }

/* c = a* * b */
#define CMULJ_(a, b, c)                                                                                                \
  {                                                                                                                    \
    (c).real = (a).real * (b).real + (a).imag * (b).imag;                                                              \
    (c).imag = (a).real * (b).imag - (a).imag * (b).real;                                                              \
  }

/* c = a * b* */
#define CMUL_J(a, b, c)                                                                                                \
  {                                                                                                                    \
    (c).real = (a).real * (b).real + (a).imag * (b).imag;                                                              \
    (c).imag = (a).imag * (b).real - (a).real * (b).imag;                                                              \
  }

#define CONJG(a, b)                                                                                                    \
  {                                                                                                                    \
    (b).real = (a).real;                                                                                               \
    (b).imag = -(a).imag;                                                                                              \
  }

struct fcomplex {
  float real;
  float imag;

  void operator+=(const fcomplex &other)
  {
    real += other.real;
    imag += other.imag;
  }
};

#pragma omp declare reduction(fcomplex_sum:fcomplex : omp_out += omp_in)

/* specific for double complex */
struct dcomplex {
  double real;
  double imag;

  void operator+=(const dcomplex &other)
  {
    real += other.real;
    imag += other.imag;
  }
};

#pragma omp declare reduction(dcomplex_sum:dcomplex : omp_out += omp_in)

struct fsu3_matrix {
  using real_t = float;
  using complex_t = fcomplex;
  fcomplex e[3][3];
};

struct fsu3_vector {
  using real_t = float;
  using complex_t = fcomplex;
  fcomplex c[3];
};

struct dsu3_matrix {
  using real_t = double;
  using complex_t = dcomplex;
  dcomplex e[3][3];
};

struct dsu3_vector {
  using real_t = double;
  using complex_t = dcomplex;
  dcomplex c[3];
};

struct fanti_hermitmat {
  using real_t = float;
  using complex_t = fcomplex;
  fcomplex m01, m02, m12;
  float m00im, m11im, m22im;
  float space;
};

struct danti_hermitmat {
  using real_t = double;
  using complex_t = dcomplex;
  dcomplex m01, m02, m12;
  double m00im, m11im, m22im;
  double space;
};

template <typename su3_matrix> su3_matrix *get_su3_matrix(quda::GaugeField &p, int idx, int dir)
{
  auto data = static_cast<su3_matrix *>(p.data(dir));
  return data + idx;
}

template <typename su3_vector, typename su3_matrix> void su3_projector(su3_vector *a, su3_vector *b, su3_matrix *c)
{
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) CMUL_J(a->c[i], b->c[j], c->e[i][j]);
}

template <typename su3_matrix> void su3_adjoint(su3_matrix *a, su3_matrix *b)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { CONJG(a->e[j][i], b->e[i][j]); }
  }
}

template <typename su3_matrix, typename anti_hermitmat> void make_anti_hermitian(su3_matrix *m3, anti_hermitmat *ah3)
{
  auto temp = (m3->e[0][0].imag + m3->e[1][1].imag + m3->e[2][2].imag) * 0.33333333333333333;
  ah3->m00im = m3->e[0][0].imag - temp;
  ah3->m11im = m3->e[1][1].imag - temp;
  ah3->m22im = m3->e[2][2].imag - temp;
  ah3->m01.real = (m3->e[0][1].real - m3->e[1][0].real) * 0.5;
  ah3->m02.real = (m3->e[0][2].real - m3->e[2][0].real) * 0.5;
  ah3->m12.real = (m3->e[1][2].real - m3->e[2][1].real) * 0.5;
  ah3->m01.imag = (m3->e[0][1].imag + m3->e[1][0].imag) * 0.5;
  ah3->m02.imag = (m3->e[0][2].imag + m3->e[2][0].imag) * 0.5;
  ah3->m12.imag = (m3->e[1][2].imag + m3->e[2][1].imag) * 0.5;
}

template <typename anti_hermitmat, typename su3_matrix>
static void uncompress_anti_hermitian(anti_hermitmat *mat_antihermit, su3_matrix *mat_su3)
{
  typename anti_hermitmat::real_t temp1;
  // typename std::remove_reference<decltype(mat_antihermit->m00im)>::type temp1;
  mat_su3->e[0][0].imag = mat_antihermit->m00im;
  mat_su3->e[0][0].real = 0.;
  mat_su3->e[1][1].imag = mat_antihermit->m11im;
  mat_su3->e[1][1].real = 0.;
  mat_su3->e[2][2].imag = mat_antihermit->m22im;
  mat_su3->e[2][2].real = 0.;
  mat_su3->e[0][1].imag = mat_antihermit->m01.imag;
  temp1 = mat_antihermit->m01.real;
  mat_su3->e[0][1].real = temp1;
  mat_su3->e[1][0].real = -temp1;
  mat_su3->e[1][0].imag = mat_antihermit->m01.imag;
  mat_su3->e[0][2].imag = mat_antihermit->m02.imag;
  temp1 = mat_antihermit->m02.real;
  mat_su3->e[0][2].real = temp1;
  mat_su3->e[2][0].real = -temp1;
  mat_su3->e[2][0].imag = mat_antihermit->m02.imag;
  mat_su3->e[1][2].imag = mat_antihermit->m12.imag;
  temp1 = mat_antihermit->m12.real;
  mat_su3->e[1][2].real = temp1;
  mat_su3->e[2][1].real = -temp1;
  mat_su3->e[2][1].imag = mat_antihermit->m12.imag;
}

template <typename su3_matrix, typename Float>
static void scalar_mult_sub_su3_matrix(su3_matrix *a, su3_matrix *b, Float s, su3_matrix *c)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c->e[i][j].real = a->e[i][j].real - s * b->e[i][j].real;
      c->e[i][j].imag = a->e[i][j].imag - s * b->e[i][j].imag;
    }
  }
}

template <typename su3_matrix, typename Float>
static void scalar_mult_add_su3_matrix(su3_matrix *a, su3_matrix *b, Float s, su3_matrix *c)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      c->e[i][j].real = a->e[i][j].real + s * b->e[i][j].real;
      c->e[i][j].imag = a->e[i][j].imag + s * b->e[i][j].imag;
    }
  }
}

template <typename su3_matrix> static void mult_su3_nn(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  typename std::remove_reference<decltype(a->e[0][0])>::type x, y;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      x.real = x.imag = 0.0;
      for (int k = 0; k < 3; k++) {
        CMUL(a->e[i][k], b->e[k][j], y);
        CSUM(x, y);
      }
      c->e[i][j] = x;
    }
  }
}

template <typename su3_matrix> static void mult_su3_an(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  typename std::remove_reference<decltype(a->e[0][0])>::type x, y;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      x.real = x.imag = 0.0;
      for (int k = 0; k < 3; k++) {
        CMULJ_(a->e[k][i], b->e[k][j], y);
        CSUM(x, y);
      }
      c->e[i][j] = x;
    }
  }
}

template <typename su3_matrix> static void mult_su3_na(su3_matrix *a, su3_matrix *b, su3_matrix *c)
{
  typename std::remove_reference<decltype(a->e[0][0])>::type x, y;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      x.real = x.imag = 0.0;
      for (int k = 0; k < 3; k++) {
        CMUL_J(a->e[i][k], b->e[j][k], y);
        CSUM(x, y);
      }
      c->e[i][j] = x;
    }
  }
}

template <typename su3_matrix, typename Float> static void add_su3(su3_matrix *a, su3_matrix *b, Float eb3)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      b->e[i][j].real += eb3 * a->e[i][j].real;
      b->e[i][j].imag += eb3 * a->e[i][j].imag;
    }
  }
}

template <typename su3_matrix> static typename su3_matrix::complex_t trace_su3(su3_matrix *a)
{
  typename su3_matrix::complex_t tmp;
  CADD(a->e[0][0], a->e[1][1], tmp);
  CADD(a->e[2][2], tmp, tmp);
  return tmp;
}

template <typename su3_matrix> void print_su3_matrix(su3_matrix *a)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { printf("(%f %f)\t", a->e[i][j].real, a->e[i][j].imag); }
    printf("\n");
  }
}
