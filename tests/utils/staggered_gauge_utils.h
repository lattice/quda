#pragma once
#include <quda_internal.h>
#include "color_spinor_field.h"

extern int Z[4];
extern int Vh;
extern int V;

using namespace quda;

void setDims(int *);

// Wrap everything for the GPU construction of fat/long links here
void computeHISQLinksGPU(void **qdp_fatlink, void **qdp_longlink, void **qdp_fatlink_eps, void **qdp_longlink_eps,
                         void **qdp_inlink, QudaGaugeParam &gauge_param, double **act_path_coeffs, double eps_naik,
                         size_t gSize, int n_naiks);

void computeFatLongGPU(void **qdp_fatlink, void **qdp_longlink, void **qdp_inlink, QudaGaugeParam &gauge_param,
                       size_t gSize, int n_naiks, double eps_naik);

void computeFatLongGPUandCPU(void **qdp_fatlink_gpu, void **qdp_longlink_gpu, void **qdp_fatlink_cpu,
                             void **qdp_longlink_cpu, void **qdp_inlink, QudaGaugeParam &gauge_param, size_t gSize,
                             int n_naiks, double eps_naik);

// Routine that takes in a QDP-ordered field and outputs the plaquette.
// Assumes the gauge fields already have phases on them (unless it's the Laplace op),
// so it corrects the sign as appropriate.
void computeStaggeredPlaquetteQDPOrder(void** qdp_link, double plaq[3], const QudaGaugeParam &gauge_param_in,
                                       const QudaDslashType dslash_type);

