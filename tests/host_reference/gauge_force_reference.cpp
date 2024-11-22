#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <type_traits>

#include "host_utils.h"
#include "force_utils.hpp"
#include "index_utils.hpp"
#include "misc.h"
#include "gauge_force_reference.h"
#include "timer.h"

extern int Z[4];
extern int V;
extern int Vh;
extern int Vh_ex;
extern int E[4];

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
  su3_matrix prev_matrix, curr_matrix = {};

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
#pragma omp parallel for
  for (size_t i = 0; i < lat.volume; i++) {
    int dx[4] = {};
    dx[dir] = 1;

    su3_matrix curr_matrix = compute_gauge_path(sitelink, i, path, len, dx, lat);

    su3_matrix tmat;
    su3_adjoint(&curr_matrix, &tmat);
    scalar_mult_add_su3_matrix(staple + i, &tmat, loop_coeff, staple + i);
  } // i
}

template <typename su3_matrix>
static dcomplex compute_loop_trace(su3_matrix **sitelink, int *path, int len, double loop_coeff, const lattice_t &lat)
{
  dcomplex accum = {};

#pragma omp parallel for reduction(dcomplex_sum : accum)
  for (size_t i = 0; i < lat.volume; i++) {
    int dx[4] = {};
    su3_matrix tmat = compute_gauge_path(sitelink, i, path, len, dx, lat);
    auto tr = trace_su3(&tmat);
    accum += dcomplex {tr.real, tr.imag};
  }

  CSCALE(accum, loop_coeff);

  return accum;
};

template <typename su3_matrix, typename anti_hermitmat, typename Float>
static void update_mom(anti_hermitmat *momentum, int dir, su3_matrix **sitelink, su3_matrix *staple, Float eb3,
                       const lattice_t &lat)
{
#pragma omp parallel for
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
#pragma omp parallel for
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
void gauge_force_reference_dir(void *refMom, int dir, double eb3, quda::GaugeField &u, quda::GaugeField &u_ex,
                               QudaPrecision prec, int **path_dir, int *length, void *loop_coeff, int num_paths,
                               const lattice_t &lat, bool compute_force)
{
  size_t size = size_t(V) * 2 * lat.n_color * lat.n_color * prec;
  void *staple = safe_malloc(size);
  memset(staple, 0, size);

  for (int i = 0; i < num_paths; i++) {
    if (prec == QUDA_DOUBLE_PRECISION) {
      double *my_loop_coeff = (double *)loop_coeff;
      compute_path_product((dsu3_matrix *)staple, u_ex.data_array<dsu3_matrix *>().data, path_dir[i], length[i],
                           my_loop_coeff[i], dir, lat);
    } else {
      float *my_loop_coeff = (float *)loop_coeff;
      compute_path_product((fsu3_matrix *)staple, u_ex.data_array<fsu3_matrix *>().data, path_dir[i], length[i],
                           my_loop_coeff[i], dir, lat);
    }
  }

  if (compute_force) {
    if (prec == QUDA_DOUBLE_PRECISION) {
      update_mom((danti_hermitmat *)refMom, dir, u.data_array<dsu3_matrix *>().data, (dsu3_matrix *)staple, (double)eb3,
                 lat);
    } else {
      update_mom((fanti_hermitmat *)refMom, dir, u.data_array<fsu3_matrix *>().data, (fsu3_matrix *)staple, (float)eb3,
                 lat);
    }
  } else {
    if (prec == QUDA_DOUBLE_PRECISION) {
      update_gauge((dsu3_matrix *)refMom, dir, u.data_array<dsu3_matrix *>().data, (dsu3_matrix *)staple, (double)eb3,
                   lat);
    } else {
      update_gauge((fsu3_matrix *)refMom, dir, u.data_array<fsu3_matrix *>().data, (fsu3_matrix *)staple, (float)eb3,
                   lat);
    }
  }
  host_free(staple);
}

void gauge_force_reference(void *refMom, double eb3, quda::GaugeField &u, int ***path_dir, int *length,
                           void *loop_coeff, int num_paths, bool compute_force)
{
  // created extended field
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam param = newQudaGaugeParam();
  setGaugeParam(param);
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;

  auto qdp_ex = quda::createExtendedGauge(u.data_array().data, param, R);
  lattice_t lat(*qdp_ex);

  for (int dir = 0; dir < 4; dir++) {
    gauge_force_reference_dir(refMom, dir, eb3, u, *qdp_ex, u.Precision(), path_dir[dir], length, loop_coeff, num_paths,
                              lat, compute_force);
  }

  delete qdp_ex;
}

void gauge_loop_trace_reference(quda::GaugeField &u, std::vector<quda::Complex> &loop_traces, double factor,
                                int **input_path, int *length, double *path_coeff, int num_paths)
{
  // create extended field
  quda::lat_dim_t R;
  for (int d = 0; d < 4; d++) R[d] = 2 * quda::comm_dim_partitioned(d);
  QudaGaugeParam param = newQudaGaugeParam();
  setGaugeParam(param);
  param.gauge_order = QUDA_QDP_GAUGE_ORDER;
  param.t_boundary = QUDA_PERIODIC_T;
  auto qdp_ex = quda::createExtendedGauge(u.data_array().data, param, R);
  lattice_t lat(*qdp_ex);

  std::vector<double> loop_tr_dbl(2 * num_paths);

  for (int i = 0; i < num_paths; i++) {
    if (u.Precision() == QUDA_DOUBLE_PRECISION) {
      dcomplex tr
        = compute_loop_trace(qdp_ex->data_array<dsu3_matrix *>().data, input_path[i], length[i], path_coeff[i], lat);
      loop_tr_dbl[2 * i] = factor * tr.real;
      loop_tr_dbl[2 * i + 1] = factor * tr.imag;
    } else {
      dcomplex tr
        = compute_loop_trace(qdp_ex->data_array<fsu3_matrix *>().data, input_path[i], length[i], path_coeff[i], lat);
      loop_tr_dbl[2 * i] = factor * tr.real;
      loop_tr_dbl[2 * i + 1] = factor * tr.imag;
    }
  }

  quda::comm_allreduce_sum(loop_tr_dbl);

  for (int i = 0; i < num_paths; i++) loop_traces[i] = quda::Complex(loop_tr_dbl[2 * i], loop_tr_dbl[2 * i + 1]);

  delete qdp_ex;
}
