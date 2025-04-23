#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <type_traits>

#include "host_utils.h"
#include "force_utils.hpp"
#include "index_utils.hpp"
#include "instantiate_host.hpp"
#include "misc.h"
#include "gauge_force_reference.h"
#include "timer.h"

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
 * @brief Calculates an arbitary gauge path, returning the product matrix
 * @return The product of the gauge path
 * @tparam real_t The precision of the calculation
 * @param[in] sitelink Gauge link structure
 * @param[in] i Full lattice index of origin
 * @param[in] path Gauge link path
 * @param[in] length Length of gauge path
 * @param[in] dx Memory for a relative coordinate shift; can be non-zero
 * @param[in] lat Utility lattice information
 */
template <typename real_t>
Matrix<3, std::complex<real_t>> compute_gauge_path(const Matrix<3, std::complex<real_t>> *const *const sitelink, int i,
                                                   const int *const path, int len, int dx[4], const lattice_t &lat)
{
  using matrix = Matrix<3, std::complex<real_t>>;

  matrix prev_matrix, curr_matrix = Identity<3, std::complex<real_t>>()();

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
    auto &lnk = sitelink[lnkdir][nbr_idx];

    if (GOES_FORWARDS(path[j])) {
      curr_matrix = prev_matrix * lnk;
    } else {
      curr_matrix = prev_matrix * conj(lnk);
    }

    if (GOES_FORWARDS(path[j])) {
      dx[path[j]] += 1;
    } else {
      // we already subtract one in the code above
    }
  } // j

  return curr_matrix;
}

template <typename real_t> struct ComputePathProduct {
  void operator()(void *const staple_, const void *const *const sitelink_, const int *const path, int len,
                  const void *const loop_coeff_, int coeff_index, int dir, const lattice_t &lat)
  {
    using matrix = Matrix<3, std::complex<real_t>>;
    auto sitelink = reinterpret_cast<const matrix *const *const>(sitelink_);

    auto staple = reinterpret_cast<matrix *const>(staple_);
    auto loop_coeff = reinterpret_cast<const real_t *const>(loop_coeff_);
    auto coeff = loop_coeff[coeff_index];

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      int dx[4] = {};
      dx[dir] = 1;

      auto curr_matrix = compute_gauge_path(sitelink, i, path, len, dx, lat);

      staple[i] = staple[i] + coeff * conj(curr_matrix);
    } // i
  }
};

/**
 * @brief Compute and store the product of a set of links along a path
 *
 * @param[in,out] staple The output product of links along a path, assumed to be zero init'd
 * @param[in] sitelink Gauge link structure
 * @param[in] path Gauge link path
 * @param[in] len Length of gauge path
 * @param[in] loop_coeff An array of loop rescaling coefficients
 * @param[in] coeff_index An index into the loop_coeff
 * @param[in] dir The starting displacement direction
 * @param[in] lat Utility lattice information
 * @param[in] precision The floating point precision of all of the links
 */
void compute_path_product(void *const staple, const void *const *const sitelink, const int *const path, int len,
                          const void *const loop_coeff, int coeff_index, int dir, const lattice_t &lat,
                          QudaPrecision precision)
{
  instantiate_host<ComputePathProduct>(precision, staple, sitelink, path, len, loop_coeff, coeff_index, dir, lat);
}

template <typename real_t> struct ComputeLoopTrace {
  std::complex<double> operator()(const void *const *const sitelink_, int *path, int len, double loop_coeff,
                                  const lattice_t &lat)
  {
    using matrix = Matrix<3, std::complex<real_t>>;
    auto sitelink = reinterpret_cast<const matrix *const *const>(sitelink_);

    std::complex<double> accum = 0;

#pragma omp parallel for reduction(+ : accum)
    for (size_t i = 0; i < lat.volume; i++) {
      int dx[4] = {};
      auto tmat = compute_gauge_path(sitelink, i, path, len, dx, lat);
      auto tr = trace(tmat);
      accum += tr;
    }

    accum *= loop_coeff;

    return accum;
  }
};

/**
 * @brief Compute trace of a closed loop of gauge links
 *
 * @return Complex-valued trace
 * @param[in] sitelink Gauge link structure
 * @param[in] path Gauge link path
 * @param[in] len Length of gauge path
 * @param[in] loop_coeff The loop scaling coefficient
 * @param[in] lat Utility lattice information
 * @param[in] precision The floating point precision of all of the links
 */
std::complex<double> compute_loop_trace(const void *const *const sitelink, int *path, int len, double loop_coeff,
                                        const lattice_t &lat, QudaPrecision precision)
{
  return instantiate_host_reduce<ComputeLoopTrace, std::complex<double>>(precision, sitelink, path, len, loop_coeff, lat);
}

template <typename real_t> struct UpdateMomentum {
  void operator()(void *const momentum_, int dir, const void *const *const sitelink_, const void *const staple_,
                  real_t eb3, const lattice_t &lat)
  {
    using matrix = Matrix<3, std::complex<real_t>>;

    auto momentum = reinterpret_cast<anti_hermitmat<real_t> *const>(momentum_);
    auto sitelink = reinterpret_cast<const matrix *const *const>(sitelink_);
    auto staple = reinterpret_cast<const matrix *const>(staple_);

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      const auto &lnk = sitelink[dir][i];
      const auto &stp = staple[i];
      auto &mom = momentum[4 * i + dir];

      matrix mom_full;
      uncompress_anti_hermitian(&mom, reinterpret_cast<su3_matrix<real_t> *>(&mom_full));

      mom_full = mom_full - eb3 * lnk * conj(stp);

      make_anti_hermitian(reinterpret_cast<const su3_matrix<real_t> *>(&mom_full), &mom);
    }
  }
};

/**
 * @brief Update the momentum with the gauge force
 *
 * @param[in,out] momentum The momentum to be updated
 * @param[in] dir The direction of the momentum being computed
 * @param[in] sitelink Gauge link structure
 * @param[in] staple The product of links along the path beginning and ending at the momentum site
 * @param[in] eb3 The contribution scaling coefficient
 * @param[in] lat Utility lattice information
 * @param[in] precision The floating point precision of all of the links
 */
void update_momentum(void *const momentum, int dir, const void *const *const sitelink, const void *const staple,
                     double eb3, const lattice_t &lat, QudaPrecision precision)
{
  instantiate_host<UpdateMomentum>(precision, momentum, dir, sitelink, staple, eb3, lat);
}

template <typename real_t> struct UpdateGauge {
  void operator()(void *const gauge_, int dir, const void *const *const sitelink_, const void *const staple_,
                  real_t eb3, const lattice_t &lat)
  {
    using matrix = Matrix<3, std::complex<real_t>>;

    auto gauge = reinterpret_cast<matrix *const>(gauge_);
    auto sitelink = reinterpret_cast<const matrix *const *const>(sitelink_);
    auto staple = reinterpret_cast<const matrix *const>(staple_);

#pragma omp parallel for
    for (size_t i = 0; i < lat.volume; i++) {
      const auto &lnk = sitelink[dir][i];
      const auto &stp = staple[i];
      auto &out = gauge[4 * i + dir];

      out += eb3 * lnk * conj(stp);
    }
  }
};

/**
 * @brief Compute the product of gauge links along a path in a direction and add to/overwrite the output field
 *
 * @param[in,out] out The output field to be updated
 * @param[in] dir The direction being summed
 * @param[in] sitelink The gauge field from which we compute the products of gauge links
 * @param[in] staple The product of links along the path beginning and ending at the starting gauge link
 * @param[in] eb3 The contribution scaling coefficient
 * @param[in] lat Utility lattice information
 * @param[in] precision The floating point precision of all of the links
 */
void update_gauge(void *const gauge, int dir, const void *const *const sitelink, const void *const staple, double eb3,
                  const lattice_t &lat, QudaPrecision precision)
{
  instantiate_host<UpdateGauge>(precision, gauge, dir, sitelink, staple, eb3, lat);
}

/**
 * @brief Compute the product of gauge links along a path in a single direction
 * and contribute to an open gauge loop or momentum field
 *
 * @param[in,out] refMom The output momentum field or open gauge loop field
 * @param[in] dir The direction of the path product loop
 * @param[in] u The gauge field from which we compute the products of gauge links
 * @param[in] u_ex The corresponding radius-2 extended gauge field
 * @param[in] prec The floating point precision of all of the links
 * @param[in] path_dir[num_paths][path_length] The paths for a specific direction
 * @param[in] length One less than the number of links in a loop (e.g., 3 for a staple)
 * @param[in] loop_coeff Coefficients for each loop
 * @param[in] num_paths How many contributions from path_length different "staples"
 * @param[in] lat Utility lattice information
 * @param[in] compute_force Whether we're updating the momentum (true) or computing a gauge loop (false)
 */
void gauge_force_reference_dir(void *refMom, int dir, double eb3, quda::GaugeField &u, quda::GaugeField &u_ex,
                               QudaPrecision prec, int **path_dir, int *length, void *loop_coeff, int num_paths,
                               const lattice_t &lat, bool compute_force)
{
  size_t size = size_t(V) * 2 * lat.n_color * lat.n_color * prec;
  void *staple = safe_malloc(size);
  memset(staple, 0, size);

  for (int i = 0; i < num_paths; i++) {
    compute_path_product(staple, u_ex.data_array<void *>().data, path_dir[i], length[i], loop_coeff, i, dir, lat, prec);
  }

  if (compute_force) {
    update_momentum(refMom, dir, u.data_array<void *>().data, staple, eb3, lat, prec);
  } else {
    update_gauge(refMom, dir, u.data_array<void *>().data, staple, eb3, lat, prec);
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
    auto tr = compute_loop_trace(qdp_ex->data_array<void *>().data, input_path[i], length[i], path_coeff[i], lat,
                                 u.Precision());
    loop_tr_dbl[2 * i] = factor * tr.real();
    loop_tr_dbl[2 * i + 1] = factor * tr.imag();
  }

  quda::comm_allreduce_sum(loop_tr_dbl);

  for (int i = 0; i < num_paths; i++) loop_traces[i] = quda::Complex(loop_tr_dbl[2 * i], loop_tr_dbl[2 * i + 1]);

  delete qdp_ex;
}
