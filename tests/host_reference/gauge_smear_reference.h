#pragma once

#include <quda.h>
#include <gauge_field.h>

/**
 * @brief Apply a one-step host reference gauge smearing operation.
 *
 * @param[out] out Smeared gauge field.
 * @param[in] in Input gauge field.
 * @param[in] smear_param Gauge smearing parameters.
 */
void gauge_smear_reference(quda::GaugeField &out, const quda::GaugeField &in, const QudaGaugeSmearParam &smear_param);
