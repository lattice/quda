#pragma once

#include <quda.h>
#include <comm_quda.h>
#include <gauge_field.h>

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

/**
 * @brief Compute the full neighbor index on an extended lattice from the full coordinate
 * index on the local lattice
 * @tparam Coord Indexable displacement type, e.g. int[4], int * or std::array<int, 4>
 * @return Full neighbor index on the extended lattice
 * @param[in] i Full index on the local lattice
 * @param[in] dx Displacement
 * @param[in] lat Utility lattice information
 */
template <typename Coord> inline int gf_neighborIndexFullLattice(size_t i, const Coord &dx, const lattice_t &lat)
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
 * @brief Compute the product of gauge links along a path and contribute to an open gauge
 * loop or momentum field
 *
 * @param[in,out] refMom The output momentum field or open gauge loop field
 * @param[in] eb3 The contribution scaling coefficient
 * @param[in] u The gauge field from which we compute the products of gauge links
 * @param[in] path_dir[dim][num_paths][path_length]
 * @param[in] length One less that the number of links in a loop (e.g., 3 for a staple)
 * @param[in] loop_coeff Coefficients of the different loops in the Symanzik action
 * @param[in] num_paths How many contributions from path_length different "staples"
 * @param[in] compute_force Whether we're updating the momentum (true) or computing a gauge loop (false)
 */
void gauge_force_reference(void *refMom, double eb3, quda::GaugeField &u, int ***path_dir, int *length,
                           void *loop_coeff, int num_paths, bool compute_force);

/**
 * Compute the traces of products of gauge links along paths using the resident field
 *
 * @param[in] u The gauge field from which we compute the trace of the gauge loop
 * @param[out] loop_traces The computed loop traces
 * @param[in] factor An overall normalization factor
 * @param[in] input_path The closed input paths
 * @param[in] length The length of each path
 * @param[in] path_coeff Independent scaling coefficients for each loop
 * @param[in] num_paths Total number of loops
 */
void gauge_loop_trace_reference(quda::GaugeField &u, std::vector<quda::complex_t> &loop_traces, double factor,
                                int **input_path, int *length, double *path_coeff, int num_paths);
