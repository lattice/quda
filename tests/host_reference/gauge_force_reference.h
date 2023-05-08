#pragma once

#include <gauge_field.h>

void gauge_force_reference(void *refMom, double eb3, quda::GaugeField &u, QudaPrecision prec, int ***path_dir,
                           int *length, void *loop_coeff, int num_paths, bool compute_force);
