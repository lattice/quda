#pragma once
#include <gauge_field.h>
void TMCloverForce_reference(void *h_mom, void **h_x, double *coeff, int nvector, std::array<void *, 4> gauge,
                             std::vector<char> clover, std::vector<char> clover_inv, QudaGaugeParam *gauge_param,
                             QudaInvertParam *inv_param, int detratio);