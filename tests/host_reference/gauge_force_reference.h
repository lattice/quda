#pragma once

void gauge_force_reference(void *refMom, double eb3, void **sitelink, QudaPrecision prec, int ***path_dir, int *length,
                           void *loop_coeff, int num_paths);
