#pragma once
#include "quda.h"
#include <comm_quda.h>
#include "gauge_field.h"
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

int gf_neighborIndexFullLattice(size_t i, int dx[], const lattice_t &lat);

#include <gauge_field.h>

void gauge_force_reference(void *refMom, double eb3, quda::GaugeField &u, int ***path_dir, int *length,
                           void *loop_coeff, int num_paths, bool compute_force);

void gauge_loop_trace_reference(quda::GaugeField &u, std::vector<quda::Complex> &loop_traces, double factor,
                                int **input_path, int *length, double *path_coeff, int num_paths);
