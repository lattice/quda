#pragma once

void gauge_force_reference(void *refMom, double eb3, void *const *sitelink, QudaPrecision prec, int ***path_dir,
                           int *length, void *loop_coeff, int num_paths, bool compute_force);
