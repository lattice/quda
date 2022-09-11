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
};

/* specific for double complex */
struct dcomplex {
  double real;
  double imag;
};

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

// convenience struct for passing around lattice meta data
struct lattice_t {
  int n_color;
  size_t volume;
  size_t volume_ex;
  int x[4];
  int r[4];
  int e[4];

  lattice_t(const quda::GaugeField &lat) : n_color(lat.Ncolor()), volume(1), volume_ex(lat.Volume())
  {
    for (int d = 0; d < 4; d++) {
      x[d] = lat.X()[d] - 2 * lat.R()[d];
      r[d] = lat.R()[d];
      e[d] = lat.X()[d];
      volume *= x[d];
    }
  };
};

extern int neighborIndexFullLattice(int i, int dx4, int dx3, int dx2, int dx1);

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

int gf_neighborIndexFullLattice(size_t i, int dx[], const lattice_t &lat)
{
  int oddBit = 0;
  int x[4];
  auto half_idx = i;
  if (i >= lat.volume / 2) {
    oddBit = 1;
    half_idx = i - lat.volume / 2;
  }

  auto za = half_idx / (lat.x[0] / 2);
  auto x0h = half_idx - za * (lat.x[0] / 2);
  auto zb = za / lat.x[1];
  x[1] = za - zb * lat.x[1];
  x[3] = zb / lat.x[2];
  x[2] = zb - x[3] * lat.x[2];
  auto x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2 * x0h + x1odd;

  for (int d = 0; d < 4; d++) {
    x[d] = quda::comm_dim_partitioned(d) ? x[d] + dx[d] : (x[d] + dx[d] + lat.x[d]) % lat.x[d];
  }
  size_t nbr_half_idx = ((x[3] + lat.r[3]) * (lat.e[2] * lat.e[1] * lat.e[0]) + (x[2] + lat.r[2]) * (lat.e[1] * lat.e[0])
                         + (x[1] + lat.r[1]) * (lat.e[0]) + (x[0] + lat.r[0]))
    / 2;

  int oddBitChanged = (dx[3] + dx[2] + dx[1] + dx[0]) % 2;
  if (oddBitChanged) { oddBit = 1 - oddBit; }
  int ret = nbr_half_idx;
  if (oddBit) ret += lat.volume_ex / 2;

  return ret;
}

/**
   @brief Calculates an arbitary gauge path, returning the product matrix
   @return The product of the gauge path
   @param[in] sitelink Gauge link structure
   @param[in] i Full lattice index of origin
   @param[in] path Gauge link path
   @param[in] length Length of gauge path
   @param[in] dx Memory for a relative coordinate shift; can be non-zero
   @param[in] lat Utility lattice information
*/
template <typename su3_matrix>
static su3_matrix compute_gauge_path(su3_matrix **sitelink, int i, int *path, int len, int dx[4], const lattice_t &lat)
{
  su3_matrix prev_matrix, curr_matrix;

  memset(&curr_matrix, 0, sizeof(curr_matrix));

  curr_matrix.e[0][0].real = 1;
  curr_matrix.e[1][1].real = 1;
  curr_matrix.e[2][2].real = 1;

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

    int nbr_idx = gf_neighborIndexFullLattice(i, dx, lat);
    su3_matrix *lnk = sitelink[lnkdir] + nbr_idx;

    if (GOES_FORWARDS(path[j])) {
      mult_su3_nn(&prev_matrix, lnk, &curr_matrix);
    } else {
      mult_su3_na(&prev_matrix, lnk, &curr_matrix);
    }

    if (GOES_FORWARDS(path[j])) {
      dx[path[j]] += 1;
    } else {
      // we already subtract one in the code above
    }
  } // j

  return curr_matrix;
}

// this function compute one path for all lattice sites
template <typename su3_matrix, typename Float>
static void compute_path_product(su3_matrix *staple, su3_matrix **sitelink, int *path, int len, Float loop_coeff,
                                 int dir, const lattice_t &lat)
{
  su3_matrix curr_matrix, tmat;
  int dx[4];

  for (size_t i = 0; i < lat.volume; i++) {
    memset(dx, 0, sizeof(dx));

    dx[dir] = 1;

    curr_matrix = compute_gauge_path(sitelink, i, path, len, dx, lat);

    su3_adjoint(&curr_matrix, &tmat);
    scalar_mult_add_su3_matrix(staple + i, &tmat, loop_coeff, staple + i);
  } // i
}

template <typename su3_matrix>
static dcomplex compute_loop_trace(su3_matrix **sitelink, int *path, int len, double loop_coeff, const lattice_t &lat)
{
  su3_matrix tmat;
  dcomplex accum;
  memset(&accum, 0, sizeof(accum));
  int dx[4];

  for (size_t i = 0; i < lat.volume; i++) {
    memset(dx, 0, sizeof(dx));
    tmat = compute_gauge_path(sitelink, i, path, len, dx, lat);
    auto tr = trace_su3(&tmat);
    CSUM(accum, tr);
  }

  CSCALE(accum, loop_coeff);

  return accum;
};

template <typename su3_matrix, typename anti_hermitmat, typename Float>
static void update_mom(anti_hermitmat *momentum, int dir, su3_matrix **sitelink, su3_matrix *staple, Float eb3,
                       const lattice_t &lat)
{
  for (size_t i = 0; i < lat.volume; i++) {
    su3_matrix tmat1;
    su3_matrix tmat2;
    su3_matrix tmat3;

    su3_matrix *lnk = sitelink[dir] + i;
    su3_matrix *stp = staple + i;
    anti_hermitmat *mom = momentum + 4 * i + dir;

    mult_su3_na(lnk, stp, &tmat1);
    uncompress_anti_hermitian(mom, &tmat2);

    scalar_mult_sub_su3_matrix(&tmat2, &tmat1, eb3, &tmat3);
    make_anti_hermitian(&tmat3, mom);
  }
}

template <typename su3_matrix, typename Float>
static void update_gauge(su3_matrix *gauge, int dir, su3_matrix **sitelink, su3_matrix *staple, Float eb3,
                         const lattice_t &lat)
{
  for (size_t i = 0; i < lat.volume; i++) {
    su3_matrix tmat;

    su3_matrix *lnk = sitelink[dir] + i;
    su3_matrix *stp = staple + i;
    su3_matrix *out = gauge + 4 * i + dir;

    mult_su3_na(lnk, stp, &tmat);

    add_su3(&tmat, out, eb3);
  }
}

/* This function only computes one direction @dir
 *
 */
void gauge_force_reference_dir(void *refMom, int dir, double eb3, void **sitelink, void **sitelink_ex,
                               QudaPrecision prec, int **path_dir, int *length, void *loop_coeff, int num_paths,
                               const lattice_t &lat, bool compute_force)
{
  size_t size = V * 2 * lat.n_color * lat.n_color * prec;
  void *staple = safe_malloc(size);
  memset(staple, 0, size);

  for (int i = 0; i < num_paths; i++) {
    if (prec == QUDA_DOUBLE_PRECISION) {
      double *my_loop_coeff = (double *)loop_coeff;
      compute_path_product((dsu3_matrix *)staple, (dsu3_matrix **)sitelink_ex, path_dir[i], length[i], my_loop_coeff[i],
                           dir, lat);
    } else {
      float *my_loop_coeff = (float *)loop_coeff;
      compute_path_product((fsu3_matrix *)staple, (fsu3_matrix **)sitelink_ex, path_dir[i], length[i], my_loop_coeff[i],
                           dir, lat);
    }
  }

  if (compute_force) {
    if (prec == QUDA_DOUBLE_PRECISION) {
      update_mom((danti_hermitmat *)refMom, dir, (dsu3_matrix **)sitelink, (dsu3_matrix *)staple, (double)eb3, lat);
    } else {
      update_mom((fanti_hermitmat *)refMom, dir, (fsu3_matrix **)sitelink, (fsu3_matrix *)staple, (float)eb3, lat);
    }
  } else {
    if (prec == QUDA_DOUBLE_PRECISION) {
      update_gauge((dsu3_matrix *)refMom, dir, (dsu3_matrix **)sitelink, (dsu3_matrix *)staple, (double)eb3, lat);
    } else {
      update_gauge((fsu3_matrix *)refMom, dir, (fsu3_matrix **)sitelink, (fsu3_matrix *)staple, (float)eb3, lat);
    }
  }
  host_free(staple);
}

void gauge_force_reference(void *refMom, double eb3, void **sitelink, QudaPrecision prec, int ***path_dir, int *length,
                           void *loop_coeff, int num_paths, bool compute_force)
{
  // created extended field
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam param = newQudaGaugeParam();
  setGaugeParam(param);
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;

  auto qdp_ex = quda::createExtendedGauge((void **)sitelink, param, R);
  lattice_t lat(*qdp_ex);

  for (int dir = 0; dir < 4; dir++) {
    gauge_force_reference_dir(refMom, dir, eb3, sitelink, (void **)qdp_ex->Gauge_p(), prec, path_dir[dir], length,
                              loop_coeff, num_paths, lat, compute_force);
  }

  delete qdp_ex;
}

void gauge_loop_trace_reference(void **sitelink, QudaPrecision prec, std::vector<quda::Complex> &loop_traces,
                                double factor, int **input_path, int *length, double *path_coeff, int num_paths)
{
  // create extended field
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam param = newQudaGaugeParam();
  setGaugeParam(param);
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;

  auto qdp_ex = quda::createExtendedGauge((void **)sitelink, param, R);
  lattice_t lat(*qdp_ex);
  void **sitelink_ex = (void **)qdp_ex->Gauge_p();

  std::vector<double> loop_tr_dbl(2 * num_paths);

  for (int i = 0; i < num_paths; i++) {
    if (prec == QUDA_DOUBLE_PRECISION) {
      dcomplex tr = compute_loop_trace((dsu3_matrix **)sitelink_ex, input_path[i], length[i], path_coeff[i], lat);
      loop_tr_dbl[2 * i] = factor * tr.real;
      loop_tr_dbl[2 * i + 1] = factor * tr.imag;
    } else {
      dcomplex tr = compute_loop_trace((fsu3_matrix **)sitelink_ex, input_path[i], length[i], path_coeff[i], lat);
      loop_tr_dbl[2 * i] = factor * tr.real;
      loop_tr_dbl[2 * i + 1] = factor * tr.imag;
    }
  }

  quda::comm_allreduce_sum(loop_tr_dbl);

  for (int i = 0; i < num_paths; i++) loop_traces[i] = quda::Complex(loop_tr_dbl[2 * i], loop_tr_dbl[2 * i + 1]);

  delete qdp_ex;
}
