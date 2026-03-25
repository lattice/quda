#pragma once

#include <complex>
#include <quda.h>
#include <gauge_field.h>

#include "host_utils.h"

using quda::GaugeField;
using quda::GaugeFieldParam;

// Local enums for gauge field construction types
enum class GaugeFieldConstructionType { UNIT_GAUGE, RANDOM_GAUGE, LOAD_GAUGE };

/**
 * @brief Constructs a QDP-ordered gauge field: either unit, random, or based on a file
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] gauge_param Information about the desired gauge field
 * @param[in] argc Input command-line argument count
 * @param[in] argv Input command-line arguments used in the case of loading a file
 */
void constructHostGaugeField(void *const *gauge, const QudaGaugeParam &gauge_param, int argc, char **argv);

/**
 * @brief Constructs a gauge field: either unit, random, or based on a file
 *
 * @param[out] gauge Generated gauge field
 * @param[in] gauge_param Information about the desired gauge field
 * @param[in] argc Input command-line argument count
 * @param[in] argv Input command-line arguments used in the case of loading a file
 */
void constructHostGaugeField(quda::GaugeField &gauge, const QudaGaugeParam &gauge_param, int argc, char **argv);

/**
 * @brief Constructs a gauge field based on a construction type
 *
 * @param[in,out] gauge Generated QDP-ordered gauge field
 * @param[in] type Type of construction gauge field
 * @param[in] param Additional information about the desired gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructQudaGaugeField(void *const *gauge, GaugeFieldConstructionType type, const QudaGaugeParam &param,
                             QudaPrecision precision);

/**
 * @brief Apply staggered phases to a QDP-ordered gauge field
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] X Full-parity local dimensions
 * @param[in] precision Gauge field floating point precision
 * @param[in] t_boundary Temporal boundary conditions
 * @param[in] phase_type Staggered phase type
 */
void applyGaugeStaggeredPhase(void *const *gauge, int Vh, const int X[], QudaPrecision precision,
                              QudaTboundary t_boundary = QUDA_ANTI_PERIODIC_T,
                              QudaStaggeredPhase phase_type = QUDA_STAGGERED_PHASE_MILC);

/**
 * @brief Apply spatial scaling, anti-periodic boundary conditions, or temporal gauge fixing as requested

 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] param Additional information about the desired gauge field
 * @param[in] precision Gauge field floating point precision
 */
void applyGaugeFieldScaling(void *const *gauge, int Vh, const QudaGaugeParam &param, QudaPrecision precision);

/**
 * @brief Apply staggered phases as well as HISQ long-link scaling factors as requested
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] param Additional information about the desired gauge field
 * @param[in] dslash_type Requested dslash type which informs the matrix type
 * @param[in] precision Gauge field floating point precision
 */
void applyGaugeFieldScaling_long(void *const *gauge, int Vh, const QudaGaugeParam &param, QudaDslashType dslash_type,
                                 QudaPrecision precision);

/**
 * @brief Constructs a 3x3 identity gauge field
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructIdentityGaugeField(void *const *gauge, QudaPrecision precision);

/**
 * @brief Constructs a random SU(3) gauge field
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructRandomSU3GaugeField(void *const *gauge, QudaPrecision precision);

/**
 * @brief Rescales a gauge field with a per-site random phase
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] precision Gauge field floating point precision
 */
void applyRandomU1Phase(void *const *gauge, QudaPrecision precision);

/**
 * @brief Constructs a "gauge field" of random (sane, invertable) matrices
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructRandomMatrixGaugeField(void *const *gauge, QudaPrecision precision);

/**
 * @brief Adds some non-unitary noise to a gauge field
 *
 * @param[in,out] gauge Generated QDP-ordered gauge field
 * @param[in] noise_max Scale of noise added to the gauge field
 * @param[in] precision Gauge field floating point precision
 */
void addNoiseToGaugeField(void *const *gauge, double noise_max, QudaPrecision precision);

/**
 * @brief Constructs a random gauge field, which may be SU(3), U(3), or arbitrary as requested
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] param Additional information about the desired gauge field
 * @param[in] precision Gauge field floating point precision
 * @param[in] dslash_type Requested dslash type which informs the matrix type
 */
void constructRandomGaugeField(void *const *gauge, const QudaGaugeParam &param, QudaPrecision precision,
                               QudaDslashType dslash_type = QUDA_WILSON_DSLASH);
