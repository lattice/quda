#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <type_traits>

#include "quda.h"
#include "gauge_field.h"
#include "host_utils.h"
#include "misc.h"
#include "gauge_force_reference.h"

extern int Z[4];
extern int V;
extern int Vh;
extern int Vh_ex;
extern int E[4];

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

typedef struct {
  float real;
  float imag;
} fcomplex;

/* specific for double complex */
typedef struct {
  double real;
  double imag;
} dcomplex;

typedef struct {
  fcomplex e[3][3];
} fsu3_matrix;
typedef struct {
  fcomplex c[3];
} fsu3_vector;
typedef struct {
  dcomplex e[3][3];
} dsu3_matrix;
typedef struct {
  dcomplex c[3];
} dsu3_vector;

typedef struct {
  fcomplex m01, m02, m12;
  float m00im, m11im, m22im;
  float space;
} fanti_hermitmat;

typedef struct {
  dcomplex m01, m02, m12;
  double m00im, m11im, m22im;
  double space;
} danti_hermitmat;

extern int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1);

template <typename su3_matrix> void su3_adjoint(su3_matrix *a, su3_matrix *b)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { CONJG(a->e[j][i], b->e[i][j]); }
  }
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

template <typename su3_matrix> void print_su3_matrix(su3_matrix *a)
{
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { printf("(%f %f)\t", a->e[i][j].real, a->e[i][j].imag); }
    printf("\n");
  }
}

int gf_neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1)
{
  int oddBit = 0;
  int half_idx = i;
  if (i >= Vh) {
    oddBit = 1;
    half_idx = i - Vh;
  }
  int X1 = Z[0];
  int X2 = Z[1];
  int X3 = Z[2];
  // int X4 = Z[3];
  int X1h = X1 / 2;

  int za = half_idx / X1h;
  int x1h = half_idx - za * X1h;
  int zb = za / X2;
  int x2 = za - zb * X2;
  int x4 = zb / X3;
  int x3 = zb - x4 * X3;
  int x1odd = (x2 + x3 + x4 + oddBit) & 1;
  int x1 = 2 * x1h + x1odd;

#ifdef MULTI_GPU
  x4 = x4 + dx4;
  x3 = x3 + dx3;
  x2 = x2 + dx2;
  x1 = x1 + dx1;

  int nbr_half_idx = ((x4 + 2) * (E[2] * E[1] * E[0]) + (x3 + 2) * (E[1] * E[0]) + (x2 + 2) * (E[0]) + (x1 + 2)) / 2;
#else
  x4 = (x4 + dx4 + Z[3]) % Z[3];
  x3 = (x3 + dx3 + Z[2]) % Z[2];
  x2 = (x2 + dx2 + Z[1]) % Z[1];
  x1 = (x1 + dx1 + Z[0]) % Z[0];

  int nbr_half_idx = (x4 * (Z[2] * Z[1] * Z[0]) + x3 * (Z[1] * Z[0]) + x2 * (Z[0]) + x1) / 2;
#endif

  int oddBitChanged = (dx4 + dx3 + dx2 + dx1) % 2;
  if (oddBitChanged) { oddBit = 1 - oddBit; }
  int ret = nbr_half_idx;
  if (oddBit) {
#ifdef MULTI_GPU
    ret = Vh_ex + nbr_half_idx;
#else
    ret = Vh + nbr_half_idx;
#endif
  }

  return ret;
}

// this functon compute one path for all lattice sites
template <typename su3_matrix>
static double compute_loop(su3_matrix **sitelink, su3_matrix **sitelink_ex_2d, int *path, int len)
{
  double sum = 0.0;
  su3_matrix prev_matrix, curr_matrix, tmat, U[4];
  int dx[4];

  for (int i = 0; i < V; i++) {
    memset(dx, 0, sizeof(dx));
    memset(&curr_matrix, 0, sizeof(curr_matrix));

    curr_matrix.e[0][0].real = 1.0;
    curr_matrix.e[1][1].real = 1.0;
    curr_matrix.e[2][2].real = 1.0;

    for (int j = 0; j < len; j++) {
      int lnkdir;

      prev_matrix = curr_matrix;
      if (GOES_FORWARDS(path[j])) {
        // dx[path[j]] +=1;
        lnkdir = path[j];
      } else {
        dx[OPP_DIR(path[j])] -= 1;
        lnkdir = OPP_DIR(path[j]);
      }

      int nbr_idx = gf_neighborIndexFullLattice(i, dx[3], dx[2], dx[1], dx[0]);
#ifdef MULTI_GPU
      su3_matrix *lnk = sitelink_ex_2d[lnkdir] + nbr_idx;
#else
      su3_matrix *lnk = sitelink[lnkdir] + nbr_idx;
#endif
      U[j] = *lnk;
      if (GOES_FORWARDS(path[j])) {
        mult_su3_nn(&prev_matrix, lnk, &curr_matrix);
      } else {
        mult_su3_na(&prev_matrix, lnk, &curr_matrix);
      }

      if (GOES_FORWARDS(path[j])) {
        dx[path[j]] += 1;
      } else {
        // we already substract one in the code above
      }

    } // j

    su3_adjoint(&curr_matrix, &tmat);
    double rtr = 0.0;
    rtr += tmat.e[0][0].real;
    rtr += tmat.e[1][1].real;
    rtr += tmat.e[2][2].real;
    //if(i==0) {
      double t0 = U[0].e[0][0].real;
      double t1 = U[1].e[0][0].real;
      double t2 = U[2].e[0][0].real;
      double t3 = U[3].e[0][0].real;
      //printf("U: %g\t%g\t%g\t%g\n", t0, t1, t2, t3);
      printf("plaqr %i: %g\n", i, rtr);
      //}
    sum += rtr;
  } // i
  // FIXME: global reduction
  return sum;
}

void plaq_reference_impl(double plaq[3], void **sitelink, void **sitelink_ex_2d, int prec)
{
  double plaqs = 0.0;
  double plaqt = 0.0;
  for (int i = 1; i < 4; i++) {
    for (int j = 0; j < i; j++) {
      double t = 0.0;
      int length = 4;
      int path_dir[4] = { j, i, 7-j, 7-i };
      if (prec == QUDA_DOUBLE_PRECISION) {
	t = compute_loop((dsu3_matrix **)sitelink, (dsu3_matrix **)sitelink_ex_2d, path_dir, length);
      } else {
	t = compute_loop((fsu3_matrix **)sitelink, (fsu3_matrix **)sitelink_ex_2d, path_dir, length);
      }
      if(i==3) plaqt += t;
      else plaqs += t;
    }
  }
  double s = 1.0/(double)(3*3*3*V);
  plaq[1] = s * plaqs;
  plaq[2] = s * plaqt;
  plaq[0] = 0.5*(plaq[1] + plaq[2]);
}

void plaq_reference(double plaq[3], void **sitelink, int prec)
{
  // created extended field
  int R[] = {2, 2, 2, 2};
  QudaGaugeParam param = newQudaGaugeParam();
  setGaugeParam(param);
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;

  auto qdp_ex = quda::createExtendedGauge((void **)sitelink, param, R);

  plaq_reference_impl(plaq, sitelink, (void **)qdp_ex->Gauge_p(), prec);

  delete qdp_ex;
}
