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
 * @return Full neighbor index on the extended lattice
 * @param[in] i Full index on the local lattice
 * @param[in] dx Displacement
 * @param[in] lat Utility lattice information
 */
int gf_neighborIndexFullLattice(size_t i, int dx[], const lattice_t &lat);

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
void gauge_loop_trace_reference(quda::GaugeField &u, std::vector<quda::Complex> &loop_traces, double factor,
                                int **input_path, int *length, double *path_coeff, int num_paths);
