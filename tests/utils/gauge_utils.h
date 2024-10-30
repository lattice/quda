#pragma once

#include <complex>
#include <quda.h>
#include <gauge_field.h>

#include "host_utils.h"

using quda::GaugeField;
using quda::GaugeFieldParam;

// Local enums for gauge field construction types
enum class GaugeFieldConstructionType {
  UNIT_GAUGE,
  RANDOM_GAUGE,
  LOAD_GAUGE,
  PHASE_GAUGE // applies phases to existing fields?
};

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
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] type Type of construction gauge field
 * @param[in] param Additional information about the desired gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructQudaGaugeField(void *const *gauge, GaugeFieldConstructionType type, const QudaGaugeParam &param,
                             QudaPrecision precision);

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
 * @brief Constructs a unit Nc = 3 gauge field
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] param Additional information about the desired gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructUnitGaugeField(void *const *gauge, const QudaGaugeParam &param, QudaPrecision precision);

/**
 * @brief Constructs a random unitary gauge field
 *
 * FIXME: use gauge_random.cu routines to create a random field via a hypercubic distribution
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructRandomUnitaryGaugeField(void *const *gauge, QudaPrecision precision);

/**
 * @brief Constructs a random gauge field, which may be SU(3), U(3), or arbitrary as requested
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] param Additional information about the desired gauge field
 * @param[in] precision Gauge field floating point precision
 * @param[in] dslash_type Requested dslash type which informs the matrix type
 */
void constructRandomGaugeField(void *const *gauge, const QudaGaugeParam &param, QudaPrecision precision,
                               QudaDslashType dslash_type = QUDA_WILSON_DSLASH);
