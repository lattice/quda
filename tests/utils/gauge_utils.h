#pragma once

#include <complex>
#include <quda.h>
#include <gauge_field.h>

#include "host_utils.h"

using quda::GaugeField;
using quda::GaugeFieldParam;

// Local enums for gauge field construction types
enum class GaugeFieldConstructionType { UNIT_GAUGE, RANDOM_GAUGE, LOAD_GAUGE };

enum class GaugeInputMode { HAAR, UNIT, GAUSSIAN_SU3, LOAD };

/**
 * @brief Immutable host gauge input shared by independent test cases.
 */
class HostGaugeInput
{
  quda::GaugeField input;

public:
  /**
   * @brief Construct an immutable QDP-ordered host gauge field for test cases.
   *
   * @param[in] gauge_param Parameters that define the host field.
   * @param[in] argc Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param).
   * @param[in] argv Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param).
   * @param[in] default_mode Input mode used when no explicit mode is requested.
   * @param[in] phase Staggered phase convention to apply (NO or MILC).
   */
  HostGaugeInput(const QudaGaugeParam &gauge_param, int argc, char **argv,
                 GaugeInputMode default_mode = GaugeInputMode::HAAR, QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_NO);

  const quda::GaugeField &field() const { return input; }
};

/**
 * @brief Resolve the active gauge-input mode from command-line options.
 *
 * @param[in] default_mode Input mode used without --gauge-input, --load-gauge or --unit-gauge.
 */
GaugeInputMode resolveGaugeInputMode(GaugeInputMode default_mode = GaugeInputMode::HAAR);

/**
 * @brief Return the command-line name of a gauge-input mode.
 *
 * @param[in] mode Gauge-input mode.
 */
const char *getGaugeInputStr(GaugeInputMode mode);

/**
 * @brief Print a gauge-input mode, and the Gaussian width for gaussian-su3.
 *
 * @param[in] mode Gauge-input mode.
 */
void printGaugeInput(GaugeInputMode mode);

/**
 * @brief Constructs a QDP-ordered gauge field: either unit, random, or based on a file
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] gauge_param Information about the desired gauge field
 * @param[in] argc Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param)
 * @param[in] argv Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param)
 */
void constructHostGaugeField(void *const *gauge, const QudaGaugeParam &gauge_param, int argc, char **argv);

/**
 * @brief Constructs a gauge field: either unit, random, or based on a file
 *
 * @param[out] gauge Generated gauge field
 * @param[in] gauge_param Information about the desired gauge field
 * @param[in] argc Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param)
 * @param[in] argv Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param)
 */
void constructHostGaugeField(quda::GaugeField &gauge, const QudaGaugeParam &gauge_param, int argc, char **argv);

/**
 * @brief A host gauge field referencing QDP-ordered arrays of plain links: general links, periodic t, no
 * staggered phases, no ghost zone.  For GaugeIO, which then never converts the links.
 *
 * @param[in] gauge The arrays, one per direction
 * @param[in] precision Floating-point precision of the arrays
 * @param[in] X The local lattice dimensions
 * @return The reference field
 */
quda::GaugeField plainHostGaugeFieldReference(void *const *gauge, QudaPrecision precision, const int *X);

/**
 * @brief Load a gauge field file into QDP-ordered host arrays as the plain links it holds, with the format and
 * options of gauge_param.file_io_param; the caller applies any boundary conditions and phases
 *
 * @param[in] file The file
 * @param[out] gauge The arrays, one per direction
 * @param[in] gauge_param The precision (cpu_prec), local lattice (X) and file options of the arrays
 */
void loadPlainHostGaugeField(const std::string &file, void *const *gauge, const QudaGaugeParam &gauge_param);

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
 * @tparam real Floating point type of the gauge field
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
 * @brief Apply temporal boundary conditions
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] t_boundary Temporal boundary conditions
 * @param[in] precision Floating-point precision of the field
 * @param[in] three_link Whether or not this is for the asqtad long links, default false
 */
void applyGaugeTemporalBCs(void *const *gauge, int Vh, QudaTboundary t_boundary, QudaPrecision precision,
                           bool three_link = false);

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

/**
 * @brief Construct a gauge field from independent Gaussian SU(3) links.
 *
 * @param[out] gauge Generated QDP-ordered gauge field.
 * @param[in] precision Gauge field floating point precision.
 * @param[in] width Gaussian width sigma applied to each matrix component.
 */
void constructGaussianSU3GaugeField(void *const *gauge, QudaPrecision precision, double width);

/**
 * @brief Construct a host gauge field from the resolved gauge-input mode.
 *
 * @param[out] gauge Generated QDP-ordered gauge field.
 * @param[in] gauge_param Information about the desired gauge field.
 * @param[in] argc Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param).
 * @param[in] argv Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param).
 * @param[in] default_mode Input mode used when no explicit mode is requested.
 * @param[in] phase Staggered phase convention to apply (NO or MILC).
 */
void constructHostGaugeInputField(void *const *gauge, const QudaGaugeParam &gauge_param, int argc, char **argv,
                                  GaugeInputMode default_mode = GaugeInputMode::HAAR,
                                  QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_NO);

/**
 * @brief Construct a host gauge field from the resolved gauge-input mode.
 *
 * @param[out] gauge Generated gauge field.
 * @param[in] gauge_param Information about the desired gauge field.
 * @param[in] argc Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param).
 * @param[in] argv Unused; kept for the API (a loaded field goes through GaugeIO with gauge_param.file_io_param).
 * @param[in] default_mode Input mode used when no explicit mode is requested.
 * @param[in] phase Staggered phase convention to apply (NO or MILC).
 */
void constructHostGaugeInputField(quda::GaugeField &gauge, const QudaGaugeParam &gauge_param, int argc, char **argv,
                                  GaugeInputMode default_mode = GaugeInputMode::HAAR,
                                  QudaStaggeredPhase phase = QUDA_STAGGERED_PHASE_NO);
