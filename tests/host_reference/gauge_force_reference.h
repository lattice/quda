#pragma once

#include <gauge_field.h>

void gauge_force_reference(void *refMom, double eb3, quda::GaugeField &u, int ***path_dir, int *length,
                           void *loop_coeff, int num_paths, bool compute_force);

void gauge_loop_trace_reference(quda::GaugeField &u, std::vector<quda::Complex> &loop_traces, double factor,
                                int **input_path, int *length, double *path_coeff, int num_paths);
