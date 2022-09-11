#pragma once

void gauge_force_reference(void *refMom, double eb3, void **sitelink, QudaPrecision prec, int ***path_dir, int *length,
                           void *loop_coeff, int num_paths, bool compute_force);

void gauge_loop_trace_reference(void **sitelink, QudaPrecision prec, std::vector<quda::Complex> &loop_traces,
                                double factor, int **input_path, int *length, double *path_coeff, int num_paths);
