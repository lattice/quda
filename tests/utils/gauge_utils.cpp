#include <cmath>
#include <quda.h>
#include <gauge_field.h>
#include <qio_field.h>

#include "command_line_params.h"
#include "host_utils.h"
#include "rng_utils.hpp"
#include "force_utils.hpp"
#include "gauge_utils.h"
#include "index_utils.hpp"
#include "instantiate_host.hpp"

/**
 * @brief Apply spatial anisotropic scaling
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] anisotropy Spatial anisotropy factor
 */
template <typename real_t> struct ApplyGaugeSpatialAnisotropy {
  void operator()(void *const *gauge_, int Vh, double anisotropy)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);
    auto inv_anisotropy = 1.0 / anisotropy;
    for (int d = 0; d < 3; d++) {
#pragma omp parallel for
      for (auto i = 0lu; i < gauge_site_size * Vh * 2; i++) { gauge[d][i] *= inv_anisotropy; }
    }
  }
};

/**
 * @brief Apply spatial anisotropic scaling
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] anisotropy Spatial anisotropy factor
 * @param[in] precision Floating-point precision of the field
 */
template <typename real_t>
void applyGaugeSpatialAnisotropy(real_t *const *gauge, int Vh, double anisotropy, QudaPrecision precision)
{
  instantiate_host<ApplyGaugeSpatialAnisotropy>(precision, gauge, Vh, anisotropy);
}

/**
 * @brief Apply temporal boundary conditions
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] t_boundary Temporal boundary conditions
 * @param[in] three_link Whether or not this is for the asqtad long links, default false
 */
template <typename real_t> struct ApplyGaugeTemporalBCs {
  void operator()(void *const *gauge_, int Vh, QudaTboundary t_boundary, bool three_link = false)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);
    if (t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t()) {
      int lower_bound = (Z[0] / 2) * Z[1] * Z[2];
      lower_bound *= three_link ? (Z[3] - 3) : (Z[3] - 1);
#pragma omp parallel for
      for (int j = lower_bound; j < Vh; j++) {
        for (auto i = 0lu; i < gauge_site_size; i++) {
          gauge[3][j * gauge_site_size + i] *= -1.0;
          gauge[3][(Vh + j) * gauge_site_size + i] *= -1.0;
        }
      }
    }
  }
};

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
                           bool three_link = false)
{
  instantiate_host<ApplyGaugeTemporalBCs>(precision, gauge, Vh, t_boundary, three_link);
}

/**
 * @brief Emulate imposing the temporal gauge.
 *
 * Emulate imposing the temporal gauge by setting all gauge links
 * except for the last Z[0]*Z[1]*Z[2]/2 to the identity.
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 */
template <typename real_t> struct EmulateGaugeTemporalGauge {
  void operator()(void *const *gauge_, int Vh)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    int iMax = (last_node_in_t() ? (Z[0] / 2) * Z[1] * Z[2] * (Z[3] - 1) : Vh);
    int dir = 3; // time direction only
    real_t *gaugeEven = gauge[dir];
    real_t *gaugeOdd = gauge[dir] + Vh * gauge_site_size;
#pragma omp parallel for
    for (int i = 0; i < iMax; i++) {
      for (int m = 0; m < 3; m++) {
        for (int n = 0; n < 3; n++) {
          gaugeEven[i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = (m == n) ? 1 : 0;
          gaugeEven[i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 0.0;
          gaugeOdd[i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = (m == n) ? 1 : 0;
          gaugeOdd[i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 0.0;
        }
      }
    }
  }
};

/**
 * @brief Emulate imposing the temporal gauge.
 *
 * Emulate imposing the temporal gauge by setting all gauge links
 * except for the last Z[0]*Z[1]*Z[2]/2 to the identity.
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] precision Floating-point precision of the field
 */
void emulateGaugeTemporalGauge(void *const *gauge, int Vh, QudaPrecision precision)
{
  instantiate_host<EmulateGaugeTemporalGauge>(precision, gauge, Vh);
}

/**
 * @brief Rescale long links by the appropriate coefficient, 1/(24 * tadpole^2)
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] tadpole Tadpole coefficient
 */
template <typename real_t> struct ApplyGaugeLongLinkScaling {
  void operator()(void *const *gauge_, int Vh, double tadpole_coeff)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    real_t inv_scale = -1.0 / (24.0 * tadpole_coeff * tadpole_coeff);
    for (int d = 0; d < 4; d++) {
#pragma omp parallel for
      for (auto i = 0lu; i < gauge_site_size * Vh * 2; i++) { gauge[d][i] *= inv_scale; }
    }
  }
};

/**
 * @brief Rescale long links by the appropriate coefficient, 1/(24 * tadpole^2)
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] tadpole Tadpole coefficient
 * @param[in] precision Floating-point precision of the field
 */
void applyGaugeLongLinkScaling(void *const *gauge, int Vh, double tadpole_coeff, QudaPrecision precision)
{
  instantiate_host<ApplyGaugeLongLinkScaling>(precision, gauge, Vh, tadpole_coeff);
}

/**
 * @brief Get the staggered phase (without boundary conditions) for a given coordinate and phase type.
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[in] x Local x coordinate
 * @param[in] y Local y coordinate
 * @param[in] z Local z coordinate
 * @param[in] t Local t coordinate
 * @param[in] X Full-parity local dimensions
 * @param[in] dim Gauge link direction
 * @param[in] t_boundary Temporal boundary conditions
 * @param[in] phase_type Staggered phase type
 * @param
 */
template <typename real_t>
real_t getStaggeredPhase(int x, int y, int z, int t, const int X[], int dim,
                         QudaTboundary t_boundary = QUDA_ANTI_PERIODIC_T,
                         QudaStaggeredPhase phase_type = QUDA_STAGGERED_PHASE_MILC)
{
  real_t t_boundary_factor = (last_node_in_t() && t == (X[3] - 1)) ? t_boundary : 1;

  real_t phase = 1.0;
  if (phase_type == QUDA_STAGGERED_PHASE_NO) {
    if (dim == 3) phase = t_boundary_factor;
  } else if (phase_type == QUDA_STAGGERED_PHASE_MILC) {
    if (dim == 0) {
      phase = (1.0 - 2.0 * (t % 2));
    } else if (dim == 1) {
      phase = (1.0 - 2.0 * ((t + x) % 2));
    } else if (dim == 2) {
      phase = (1.0 - 2.0 * ((t + x + y) % 2));
    } else if (dim == 3) {
      phase = t_boundary_factor;
    }
  } else if (phase_type == QUDA_STAGGERED_PHASE_TIFR) {
    if (dim == 0) {
      phase = (1.0 - 2.0 * ((3 + t + z + y) % 2));
    } else if (dim == 1) {
      phase = (1.0 - 2.0 * ((2 + t + z) % 2));
    } else if (dim == 2) {
      phase = (1.0 - 2.0 * ((1 + t) % 2));
    } else if (dim == 3) {
      phase = t_boundary_factor;
    }
  } else if (phase_type == QUDA_STAGGERED_PHASE_CHROMA) {
    // Chroma follows CPS convention, but uses -Dslash instead of Dslash compared to QUDA
    if (dim == 0) {
      phase = -1.0;
    } else if (dim == 1) {
      phase = (1.0 - 2.0 * ((1 + x) % 2));
    } else if (dim == 2) {
      phase = (1.0 - 2.0 * ((1 + x + y) % 2));
    } else if (dim == 3) {
      phase = t_boundary_factor * (1.0 - 2 * ((1 + x + y + z) % 2));
    }
  } else {
    errorQuda("Invalid phase %d", phase_type);
  }
  return phase;
}

/**
 * @brief Apply staggered phases to a gauge field
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] X Full-parity local dimensions
 * @param[in] t_boundary Temporal boundary conditions
 * @param[in] phase_type Staggered phase type
 */
template <typename real_t> struct ApplyGaugeStaggeredPhase {
  void operator()(void *const *gauge_, int Vh, const int X[], QudaTboundary t_boundary = QUDA_ANTI_PERIODIC_T,
                  QudaStaggeredPhase phase_type = QUDA_STAGGERED_PHASE_MILC)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    int X1 = X[0];
    int X2 = X[1];
    int X3 = X[2];

    // apply the staggered phases
    for (int d = 0; d < 3; d++) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        for (int parity = 0; parity < 2; parity++) {
          int index = fullLatticeIndex(i, parity);
          int i4 = index / (X3 * X2 * X1);
          int i3 = (index - i4 * (X3 * X2 * X1)) / (X2 * X1);
          int i2 = (index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1)) / X1;
          int i1 = index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;

          real_t sign = getStaggeredPhase<real_t>(i1, i2, i3, i4, X, d, t_boundary, phase_type);

          for (int j = 0; j < 18; j++) { gauge[d][(parity * Vh + i) * gauge_site_size + j] *= sign; }
        }
      }
    }
  }
};

void applyGaugeStaggeredPhase(void *const *gauge, int Vh, const int X[], QudaPrecision precision,
                              QudaTboundary t_boundary, QudaStaggeredPhase phase_type)
{
  instantiate_host<ApplyGaugeStaggeredPhase>(precision, gauge, Vh, X, t_boundary, phase_type);
}

void applyGaugeFieldScaling(void *const *gauge, int Vh, const QudaGaugeParam &param, QudaPrecision precision)
{
  if (param.anisotropy != 1) applyGaugeSpatialAnisotropy(gauge, Vh, param.anisotropy, precision);

  applyGaugeTemporalBCs(gauge, Vh, param.t_boundary, precision);

  if (param.gauge_fix) emulateGaugeTemporalGauge(gauge, Vh, precision);
}

/**
 * @brief Constructs a 3x3 identity gauge field
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 */
template <typename real_t> struct ConstructIdentityGaugeField {
  void operator()(void *const *gauge_)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    real_t *gaugeOdd[4], *gaugeEven[4];
    for (int dir = 0; dir < 4; dir++) {
      gaugeEven[dir] = gauge[dir];
      gaugeOdd[dir] = gauge[dir] + Vh * gauge_site_size;
    }

    for (int dir = 0; dir < 4; dir++) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        for (int m = 0; m < 3; m++) {
          for (int n = 0; n < 3; n++) {
            gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = (m == n) ? 1 : 0;
            gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 0.0;
            gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = (m == n) ? 1 : 0;
            gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 0.0;
          }
        }
      }
    }
  }
};

void constructIdentityGaugeField(void *const *gauge, QudaPrecision precision)
{
  instantiate_host<ConstructIdentityGaugeField>(precision, gauge);
}

/**
 * @brief Constructs a random unitary gauge field
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 */
template <typename real_t> struct ConstructRandomSU3GaugeField {
  void operator()(void *const *gauge_)
  {
    using complex = std::complex<real_t>;

    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    real_t *gaugeOdd[4], *gaugeEven[4];
    for (int dir = 0; dir < 4; dir++) {
      gaugeEven[dir] = gauge[dir];
      gaugeOdd[dir] = gauge[dir] + Vh * gauge_site_size;
    }

    // Define normalize, orthogonalize, and accumulateConjugateProduct locally since they aren't
    // used anywhere else

    // normalize the vector a
    auto normalize = [](complex *a, int len) -> void {
      double sum = 0.0;
      for (int i = 0; i < len; i++) sum += norm(a[i]);
      for (int i = 0; i < len; i++) a[i] /= sqrt(sum);
    };

    // orthogonalize vector b to vector a
    auto orthogonalize = [](complex *a, complex *b, int len) -> void {
      std::complex<double> dot = 0.0;
      for (int i = 0; i < len; i++) dot += conj(a[i]) * b[i];
      for (int i = 0; i < len; i++) b[i] -= static_cast<complex>(dot) * a[i];
    };

    // accumulate a conjugate product
    auto accumulateConjugateProduct = [](real_t *a, real_t *b, real_t *c, int sign) -> void {
      a[0] += sign * (b[0] * c[0] - b[1] * c[1]);
      a[1] -= sign * (b[0] * c[1] + b[1] * c[0]);
    };

    for (int dir = 0; dir < 4; dir++) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        for (int m = 1; m < 3; m++) {   // last 2 rows
          for (int n = 0; n < 3; n++) { // 3 columns
            gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = random_uniform_host<real_t>(i, 0);
            gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = random_uniform_host<real_t>(i, 0);
            gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = random_uniform_host<real_t>(i, 1);
            gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = random_uniform_host<real_t>(i, 1);
          }
        }
        normalize(reinterpret_cast<complex *>(gaugeEven[dir] + (i * 3 + 1) * 3 * 2), 3);
        orthogonalize(reinterpret_cast<complex *>(gaugeEven[dir] + (i * 3 + 1) * 3 * 2),
                      reinterpret_cast<complex *>(gaugeEven[dir] + (i * 3 + 2) * 3 * 2), 3);
        normalize(reinterpret_cast<complex *>(gaugeEven[dir] + (i * 3 + 2) * 3 * 2), 3);

        normalize(reinterpret_cast<complex *>(gaugeOdd[dir] + (i * 3 + 1) * 3 * 2), 3);
        orthogonalize(reinterpret_cast<complex *>(gaugeOdd[dir] + (i * 3 + 1) * 3 * 2),
                      reinterpret_cast<complex *>(gaugeOdd[dir] + (i * 3 + 2) * 3 * 2), 3);
        normalize(reinterpret_cast<complex *>(gaugeOdd[dir] + (i * 3 + 2) * 3 * 2), 3);

        {
          real_t *w = gaugeEven[dir] + (i * 3 + 0) * 3 * 2;
          real_t *u = gaugeEven[dir] + (i * 3 + 1) * 3 * 2;
          real_t *v = gaugeEven[dir] + (i * 3 + 2) * 3 * 2;

          for (int n = 0; n < 6; n++) w[n] = 0.0;
          accumulateConjugateProduct(w + 0 * (2), u + 1 * (2), v + 2 * (2), +1);
          accumulateConjugateProduct(w + 0 * (2), u + 2 * (2), v + 1 * (2), -1);
          accumulateConjugateProduct(w + 1 * (2), u + 2 * (2), v + 0 * (2), +1);
          accumulateConjugateProduct(w + 1 * (2), u + 0 * (2), v + 2 * (2), -1);
          accumulateConjugateProduct(w + 2 * (2), u + 0 * (2), v + 1 * (2), +1);
          accumulateConjugateProduct(w + 2 * (2), u + 1 * (2), v + 0 * (2), -1);
        }

        {
          real_t *w = gaugeOdd[dir] + (i * 3 + 0) * 3 * 2;
          real_t *u = gaugeOdd[dir] + (i * 3 + 1) * 3 * 2;
          real_t *v = gaugeOdd[dir] + (i * 3 + 2) * 3 * 2;

          for (int n = 0; n < 6; n++) w[n] = 0.0;
          accumulateConjugateProduct(w + 0 * (2), u + 1 * (2), v + 2 * (2), +1);
          accumulateConjugateProduct(w + 0 * (2), u + 2 * (2), v + 1 * (2), -1);
          accumulateConjugateProduct(w + 1 * (2), u + 2 * (2), v + 0 * (2), +1);
          accumulateConjugateProduct(w + 1 * (2), u + 0 * (2), v + 2 * (2), -1);
          accumulateConjugateProduct(w + 2 * (2), u + 0 * (2), v + 1 * (2), +1);
          accumulateConjugateProduct(w + 2 * (2), u + 1 * (2), v + 0 * (2), -1);
        }
      }
    }
  }
};

void constructRandomSU3GaugeField(void *const *gauge, QudaPrecision precision)
{
  instantiate_host<ConstructRandomSU3GaugeField>(precision, gauge);
}

/**
 * @brief Construct independent Gaussian SU(3) links by algebra projection and exponentiation.
 *
 * @tparam real_t Floating point type of the gauge field.
 * @param[out] gauge Generated QDP-ordered gauge field.
 * @param[in] width Gaussian width sigma applied to each matrix component.
 */
template <typename real_t> struct ConstructGaussianSU3GaugeField {
  void operator()(void *const *gauge_, double width)
  {
    using complex = std::complex<real_t>;
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    for (int dir = 0; dir < 4; dir++) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        for (int parity = 0; parity < 2; parity++) {
          Matrix<3, complex> matrix;
          for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
              matrix(row, col)
                = complex(random_gaussian_host<real_t>(i, parity, 0, width), random_gaussian_host<real_t>(i, parity, 0, width));
            }
          }
          make_herm(matrix);
          const auto link = exponentiate_iQ(matrix);

          auto out = reinterpret_cast<complex *>(gauge[dir] + (parity * Vh + i) * gauge_site_size);
          for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) { out[row * 3 + col] = link(row, col); }
          }
        }
      }
    }
  }
};

void constructGaussianSU3GaugeField(void *const *gauge, QudaPrecision precision, double width)
{
  instantiate_host<ConstructGaussianSU3GaugeField>(precision, gauge, width);
}

const char *getGaugeInputStr(GaugeInputMode mode)
{
  switch (mode) {
  case GaugeInputMode::HAAR: return "haar";
  case GaugeInputMode::UNIT: return "unit";
  case GaugeInputMode::GAUSSIAN_SU3: return "gaussian-su3";
  case GaugeInputMode::LOAD: return "load";
  }
  errorQuda("Invalid gauge input mode %d", static_cast<int>(mode));
  return "invalid";
}

HostGaugeInput::HostGaugeInput(const QudaGaugeParam &gauge_param, int argc, char **argv, GaugeInputMode default_mode)
{
  quda::GaugeFieldParam field_param(gauge_param);
  field_param.location = QUDA_CPU_FIELD_LOCATION;
  field_param.order = QUDA_QDP_GAUGE_ORDER;
  field_param.create = QUDA_NULL_FIELD_CREATE;
  input = quda::GaugeField(field_param);
  constructHostGaugeInputField(input, gauge_param, argc, argv, default_mode);
}

GaugeInputMode resolveGaugeInputMode(GaugeInputMode default_mode)
{
  const bool explicit_input = !gauge_input.empty();
  const bool has_load = !latfile.empty();

  if (explicit_input) {
    GaugeInputMode mode = GaugeInputMode::HAAR;
    if (gauge_input == "unit")
      mode = GaugeInputMode::UNIT;
    else if (gauge_input == "gaussian-su3")
      mode = GaugeInputMode::GAUSSIAN_SU3;
    else if (gauge_input == "load")
      mode = GaugeInputMode::LOAD;

    if (mode != GaugeInputMode::GAUSSIAN_SU3 && gauge_input_width_explicit) {
      errorQuda("--gauge-input-width requires --gauge-input gaussian-su3");
    }
    if (mode != GaugeInputMode::LOAD && has_load) {
      errorQuda("--load-gauge conflicts with explicit --gauge-input %s", gauge_input.c_str());
    }
    if (mode != GaugeInputMode::UNIT && unit_gauge) {
      errorQuda("--unit-gauge conflicts with explicit --gauge-input %s", gauge_input.c_str());
    }
    if (mode == GaugeInputMode::LOAD && !has_load) { errorQuda("--gauge-input load requires --load-gauge"); }
    if (mode == GaugeInputMode::GAUSSIAN_SU3 && (!std::isfinite(gauge_input_width) || gauge_input_width < 0)) {
      errorQuda("--gauge-input-width must be finite and nonnegative");
    }
    return mode;
  }

  if (has_load) return GaugeInputMode::LOAD;
  if (unit_gauge) return GaugeInputMode::UNIT;
  return default_mode;
}

void constructHostGaugeInputField(void *const *gauge, const QudaGaugeParam &gauge_param, int argc, char **argv,
                                  GaugeInputMode default_mode)
{
  switch (resolveGaugeInputMode(default_mode)) {
  case GaugeInputMode::HAAR:
    constructRandomSU3GaugeField(gauge, gauge_param.cpu_prec);
    applyGaugeFieldScaling(gauge, Vh, gauge_param, gauge_param.cpu_prec);
    break;
  case GaugeInputMode::UNIT:
    constructIdentityGaugeField(gauge, gauge_param.cpu_prec);
    applyGaugeFieldScaling(gauge, Vh, gauge_param, gauge_param.cpu_prec);
    break;
  case GaugeInputMode::GAUSSIAN_SU3:
    constructGaussianSU3GaugeField(gauge, gauge_param.cpu_prec, gauge_input_width);
    applyGaugeFieldScaling(gauge, Vh, gauge_param, gauge_param.cpu_prec);
    break;
  case GaugeInputMode::LOAD:
    logQuda(QUDA_VERBOSE, "Loading the gauge field in %s\n", latfile.c_str());
    read_gauge_field(latfile.c_str(), (void **)gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    applyGaugeFieldScaling(gauge, Vh, gauge_param, gauge_param.cpu_prec);
    break;
  }
}

void constructHostGaugeInputField(quda::GaugeField &gauge, const QudaGaugeParam &gauge_param, int argc, char **argv,
                                  GaugeInputMode default_mode)
{
  if (gauge.Order() == QUDA_QDP_GAUGE_ORDER) {
    constructHostGaugeInputField(static_cast<void *const *>(gauge.raw_pointer()), gauge_param, argc, argv, default_mode);
  } else {
    GaugeFieldParam param(gauge);
    param.order = QUDA_QDP_GAUGE_ORDER;
    param.create = QUDA_NULL_FIELD_CREATE;
    GaugeField u(param);
    constructHostGaugeInputField(static_cast<void *const *>(u.raw_pointer()), gauge_param, argc, argv, default_mode);
    gauge = u;
  }
}

/**
 * @brief Rescales a gauge field with a per-site random phase
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[in,out] gauge Generated QDP-ordered gauge field
 */
template <typename real_t> struct ApplyRandomU1Phase {
  void operator()(void *const *gauge_)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    for (int dir = 0; dir < 4; dir++) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        for (int parity = 0; parity < 2; parity++) {
          // create a random phase
          real_t phase = random_uniform_host<real_t>(i, parity, 0, 2 * M_PI);
          std::complex<real_t> u1 = std::polar(static_cast<real_t>(1), phase);

          // rescale bottom row
          auto link = reinterpret_cast<std::complex<real_t> *>(gauge[dir] + (parity * Vh + i) * gauge_site_size);
          constexpr int m = 2;          // bottom row
          for (int n = 0; n < 3; n++) { // 3 columns
            link[m * 3 + n] *= u1;
          }
        }
      }
    }
  }
};

void applyRandomU1Phase(void *const *gauge, QudaPrecision precision)
{
  instantiate_host<ApplyRandomU1Phase>(precision, gauge);
}

/**
 * @brief Constructs a "gauge field" of random (sane, invertable) matrices
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 */
template <typename real_t> struct ConstructRandomMatrixGaugeField {
  void operator()(void *const *gauge_)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);

    // Encapsulate creating a random matrix in a lambda to simplify making sure
    // the matrix is invertable
    auto randomMatrix = [](real_t *link, int i, int parity) -> void {
      for (int m = 0; m < 3; m++) {   // 3 rows
        for (int n = 0; n < 3; n++) { // 3 columns
          link[m * (3 * 2) + n * (2) + 0] = 0.8 * random_uniform_host<real_t>(i, parity, -1, 1);
          link[m * (3 * 2) + n * (2) + 1] = 1.3 * random_uniform_host<real_t>(i, parity, -1, 1);
        }
      }
    };

    auto isReasonable = [](real_t *link) -> bool {
      auto mat = reinterpret_cast<std::complex<real_t> *>(link);

      auto det = mat[0 * 3 + 0] * (mat[1 * 3 + 1] * mat[2 * 3 + 2] - mat[1 * 3 + 2] * mat[2 * 3 + 1])
        - mat[0 * 3 + 1] * (mat[1 * 3 + 0] * mat[2 * 3 + 2] - mat[1 * 3 + 2] * mat[2 * 3 + 0])
        + mat[0 * 3 + 2] * (mat[1 * 3 + 0] * mat[2 * 3 + 1] - mat[1 * 3 + 1] * mat[2 * 3 + 0]);

      // the 1e-3 is arbitrary
      if (std::abs(det) > 1e-3)
        return true;
      else
        return false;
    };

    for (int dir = 0; dir < 4; dir++) {
#pragma omp parallel for
      for (int i = 0; i < Vh; i++) {
        for (int parity = 0; parity < 2; parity++) {
          real_t *link = gauge[dir] + (parity * Vh + i) * gauge_site_size;
          do {
            randomMatrix(link, i, parity);
          } while (!isReasonable(link));
        }
      }
    }
  }
};

void constructRandomMatrixGaugeField(void *const *gauge, QudaPrecision precision)
{
  instantiate_host<ConstructRandomMatrixGaugeField>(precision, gauge);
}

/**
 * @brief Adds some non-unitary noise to a gauge field
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[in,out] gauge Generated QDP-ordered gauge field
 * @param[in] noise_max Scale of noise added to the gauge field
 */
template <typename real_t> struct AddNoiseToGaugeField {
  void operator()(void *const *gauge_, double noise_max)
  {
    auto gauge = reinterpret_cast<real_t *const *>(gauge_);
    // originally in createNoisyLinkCPU in tests/hisq_unitarize_force_test.cpp
    for (int dir = 0; dir < 4; ++dir) {
#pragma omp parallel for
      for (int i = 0; i < Vh; ++i) {
        for (int parity = 0; parity < 2; parity++) {
          for (size_t c = 0lu; c < gauge_site_size; c++) {
            real_t *link = gauge[dir] + (parity * Vh + i) * gauge_site_size + c;
            *link += random_uniform_host<real_t>(i, parity, -noise_max, noise_max);
          }
        }
      }
    }
  }
};

void addNoiseToGaugeField(void *const *gauge, double noise_max, QudaPrecision precision)
{
  instantiate_host<AddNoiseToGaugeField>(precision, gauge, noise_max);
}

void constructRandomGaugeField(void *const *gauge, const QudaGaugeParam &param, QudaPrecision precision,
                               QudaDslashType dslash_type)
{

  // Next set boundary conditions, phases, scalings, etc based on what the dslash_type may dictate wanting
  if (param.type == QUDA_WILSON_LINKS) {
    // Generate a random SU3 field, apply boundary conditions, scalings, etc
    constructRandomSU3GaugeField(gauge, precision);
    applyGaugeFieldScaling(gauge, Vh, param, precision);
  } else if (param.type == QUDA_ASQTAD_LONG_LINKS) {
    // Generate a random SU3 field, apply staggered phases, boundary conditions,
    // scalings, etc, depending on if we're actually requesting staggered or
    // hisq long links
    constructRandomSU3GaugeField(gauge, precision);
    applyGaugeFieldScaling_long(gauge, Vh, param, dslash_type, precision);
  } else if (param.type == QUDA_ASQTAD_FAT_LINKS) {
    // Generate an invertable and sane but otherwise random 3x3 matrix field
    constructRandomMatrixGaugeField(gauge, precision);
  } else {
    errorQuda("Invalid dslash_type %d", dslash_type);
  }
}

void constructQudaGaugeField(void *const *gauge, GaugeFieldConstructionType type, const QudaGaugeParam &param,
                             QudaPrecision precision)
{
  if (type == GaugeFieldConstructionType::UNIT_GAUGE) {
    // populate an identity gauge field
    constructIdentityGaugeField(gauge, precision);

    // apply anisotropy, boundary conditions
    applyGaugeFieldScaling(gauge, Vh, param, precision);
  } else if (type == GaugeFieldConstructionType::RANDOM_GAUGE) {
    constructRandomGaugeField(gauge, param, precision);
  } else {
    // Loaded a field, apply anisotropy, boundary conditions
    applyGaugeFieldScaling(gauge, Vh, param, precision);
  }
}

void constructHostGaugeField(void *const *gauge, const QudaGaugeParam &gauge_param, int argc, char **argv)
{
  GaugeFieldConstructionType construct_type;
  if (latfile.size() > 0) {
    // load in the command line supplied gauge field using QIO and LIME
    logQuda(QUDA_VERBOSE, "Loading the gauge field in %s\n", latfile.c_str());
    read_gauge_field(latfile.c_str(), (void **)gauge, gauge_param.cpu_prec, gauge_param.X, argc, argv);
    construct_type = GaugeFieldConstructionType::LOAD_GAUGE;
  } else {
    if (unit_gauge)
      construct_type = GaugeFieldConstructionType::UNIT_GAUGE;
    else
      construct_type = GaugeFieldConstructionType::RANDOM_GAUGE;
  }
  constructQudaGaugeField(gauge, construct_type, gauge_param, gauge_param.cpu_prec);
}

void constructHostGaugeField(quda::GaugeField &gauge, const QudaGaugeParam &gauge_param, int argc, char **argv)
{
  if (gauge.Order() == QUDA_QDP_GAUGE_ORDER) {
    constructHostGaugeField(static_cast<void *const *>(gauge.raw_pointer()), gauge_param, argc, argv);
  } else {
    GaugeFieldParam param(gauge);
    param.order = QUDA_QDP_GAUGE_ORDER;
    param.create = QUDA_NULL_FIELD_CREATE;
    GaugeField u(param);
    constructHostGaugeField(static_cast<void *const *>(u.raw_pointer()), gauge_param, argc, argv);
    gauge = u;
  }
}

void applyGaugeFieldScaling_long(void *const *gauge, int Vh, const QudaGaugeParam &param, QudaDslashType dslash_type,
                                 QudaPrecision precision)
{
  // rescale long links by the appropriate coefficient
  if (dslash_type == QUDA_ASQTAD_DSLASH) applyGaugeLongLinkScaling(gauge, Vh, param.tadpole_coeff, precision);

  // apply staggered phases WITHOUT the temporal boundary conditions
  // The temporal boundary conditions are (carefully) applied in
  // applyGaugeTemporalBCs
  applyGaugeStaggeredPhase(gauge, Vh, param.X, precision, QUDA_PERIODIC_T, QUDA_STAGGERED_PHASE_MILC);

  applyGaugeTemporalBCs(gauge, Vh, param.t_boundary, precision, dslash_type == QUDA_ASQTAD_DSLASH);
}
