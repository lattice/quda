#pragma once

#include <complex>
#include <quda.h>
#include <quda_internal.h>

void display_driver_info();

void laphSinkProject(void *host_quark, void *host_evec, void *host_sinks,
		     QudaInvertParam inv_param, const int X[4], int t_size);
