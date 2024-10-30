#include <quda.h>
#include <gauge_field.h>
#include <qio_field.h>

#include "command_line_params.h"
#include "host_utils.h"
#include "gauge_utils.h"
#include "index_utils.hpp"

/**
 * @brief Apply spatial scaling, anti-periodic boundary conditions, or temporal gauge fixing as requested
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] param Additional information about the desired gauge field
 */
template <typename real_t> void applyGaugeFieldScaling(real_t *const *gauge, int Vh, const QudaGaugeParam &param)
{
  // Apply spatial scaling factor (u0) to spatial links
  for (int d = 0; d < 3; d++) {
#pragma omp parallel for
    for (auto i = 0lu; i < gauge_site_size * Vh * 2; i++) { gauge[d][i] /= param.anisotropy; }
  }

  // Apply boundary conditions to temporal links
  if (param.t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t()) {
#pragma omp parallel for
    for (int j = (Z[0] / 2) * Z[1] * Z[2] * (Z[3] - 1); j < Vh; j++) {
      for (auto i = 0lu; i < gauge_site_size; i++) {
        gauge[3][j * gauge_site_size + i] *= -1.0;
        gauge[3][(Vh + j) * gauge_site_size + i] *= -1.0;
      }
    }
  }

  if (param.gauge_fix) {
    // set all gauge links (except for the last Z[0]*Z[1]*Z[2]/2) to the identity,
    // to simulate fixing to the temporal gauge.
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
}

void applyGaugeFieldScaling(void *const *gauge, int Vh, const QudaGaugeParam &param, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    applyGaugeFieldScaling((double *const *)gauge, Vh, param);
  else
    applyGaugeFieldScaling((float *const *)gauge, Vh, param);
}

/**
 * @brief Constructs a 3x3 identity gauge field
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 */
template <typename real_t> void constructIdentityGaugeField(real_t *const *gauge)
{
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

/**
 * @brief Constructs a 3x3 identity gauge field
 *
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] precision Gauge field floating point precision
 */
void constructIdentityGaugeField(void *const *gauge, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    constructIdentityGaugeField((double *const *)gauge);
  else
    constructIdentityGaugeField((float *const *)gauge);
}

void constructUnitGaugeField(void *const *gauge, const QudaGaugeParam &param, QudaPrecision precision)
{
  // Compute the identity matrix
  constructIdentityGaugeField(gauge, precision);

  // Apply spatial isotropy, temporal boundary conditions, and temporal gauge fixing
  applyGaugeFieldScaling(gauge, Vh, param, precision);
}

/**
 * @brief Constructs a random unitary gauge field
 *
 * FIXME: use gauge_random.cu routines to create a random field via a hypercubic distribution
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 */
template <typename real_t> void constructRandomUnitaryGaugeField(real_t *const *gauge)
{
  using complex = std::complex<real_t>;

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
    for (int i = 0; i < Vh; i++) {
      for (int m = 1; m < 3; m++) {   // last 2 rows
        for (int n = 0; n < 3; n++) { // 3 columns
          gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = rand() / static_cast<real_t>(RAND_MAX);
          gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / static_cast<real_t>(RAND_MAX);
          gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = rand() / static_cast<real_t>(RAND_MAX);
          gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = rand() / static_cast<real_t>(RAND_MAX);
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

void constructRandomUnitaryGaugeField(void *const *gauge, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    constructRandomUnitaryGaugeField((double *const *)gauge);
  else
    constructRandomUnitaryGaugeField((float *const *)gauge);
}

void constructRandomGaugeField(void *const *gauge, const QudaGaugeParam &param, QudaPrecision precision,
                               QudaDslashType dslash_type)
{
  // Start by generating a unitary gauge field gauge field
  constructRandomUnitaryGaugeField(gauge, precision);

  // Next set boundary conditions, phases, scalings, etc based on what the dslash_type may dictate wanting
  if (param.type == QUDA_WILSON_LINKS) {
    applyGaugeFieldScaling(gauge, Vh, param, precision);
  } else if (param.type == QUDA_ASQTAD_LONG_LINKS) {
    applyGaugeFieldScaling_long(gauge, Vh, param, dslash_type, precision);
  } else if (param.type == QUDA_ASQTAD_FAT_LINKS) {
    if (precision == QUDA_DOUBLE_PRECISION) {
      double *gaugeOdd[4], *gaugeEven[4];
      for (int dir = 0; dir < 4; dir++) {
        gaugeEven[dir] = (double *)gauge[dir];
        gaugeOdd[dir] = (double *)gauge[dir] + Vh * gauge_site_size;
      }
      for (int dir = 0; dir < 4; dir++) {
        for (int i = 0; i < Vh; i++) {
          for (int m = 0; m < 3; m++) {   // last 2 rows
            for (int n = 0; n < 3; n++) { // 3 columns
              gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = 1.0 * rand() / static_cast<double>(RAND_MAX);
              gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 2.0 * rand() / static_cast<double>(RAND_MAX);
              gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = 3.0 * rand() / static_cast<double>(RAND_MAX);
              gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 4.0 * rand() / static_cast<double>(RAND_MAX);
            }
          }
        }
      }
    } else {
      float *gaugeOdd[4], *gaugeEven[4];
      for (int dir = 0; dir < 4; dir++) {
        gaugeEven[dir] = (float *)gauge[dir];
        gaugeOdd[dir] = (float *)gauge[dir] + Vh * gauge_site_size;
      }
      for (int dir = 0; dir < 4; dir++) {
        for (int i = 0; i < Vh; i++) {
          for (int m = 0; m < 3; m++) {   // last 2 rows
            for (int n = 0; n < 3; n++) { // 3 columns
              gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = 1.0 * rand() / static_cast<float>(RAND_MAX);
              gaugeEven[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 2.0 * rand() / static_cast<float>(RAND_MAX);
              gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 0] = 3.0 * rand() / static_cast<float>(RAND_MAX);
              gaugeOdd[dir][i * (3 * 3 * 2) + m * (3 * 2) + n * (2) + 1] = 4.0 * rand() / static_cast<float>(RAND_MAX);
            }
          }
        }
      }
    }
  } else {
    errorQuda("Invalid dslash_type %d", dslash_type);
  }
}

void constructQudaGaugeField(void *const *gauge, GaugeFieldConstructionType type, const QudaGaugeParam &param,
                             QudaPrecision precision)
{
  if (type == GaugeFieldConstructionType::UNIT_GAUGE) {
    constructUnitGaugeField(gauge, param, precision);
  } else if (type == GaugeFieldConstructionType::RANDOM_GAUGE) {
    constructRandomGaugeField(gauge, param, precision);
  } else {
    // Loaded a field, applying some type of post-processing
    applyGaugeFieldScaling(gauge, Vh, param, precision);
  }
}

void constructHostGaugeField(void *const *gauge, const QudaGaugeParam &gauge_param, int argc, char **argv)
{
  // 0 = unit gauge
  // 1 = random SU(3)
  // 2 = supplied field
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

/**
 * @brief Apply staggered phases as well as HISQ long-link scaling factors as requested
 *
 * @tparam real_t Floating point type of the gauge field
 * @param[out] gauge Generated QDP-ordered gauge field
 * @param[in] Vh One-half of the local volume
 * @param[in] param Additional information about the desired gauge field
 * @param[in] dslash_type Requested dslash type which informs the matrix type
 */
template <typename real_t>
void applyGaugeFieldScaling_long(real_t *const *gauge, int Vh, const QudaGaugeParam &param, QudaDslashType dslash_type)
{
  int X1h = param.X[0] / 2;
  int X1 = param.X[0];
  int X2 = param.X[1];
  int X3 = param.X[2];
  int X4 = param.X[3];

  // rescale long links by the appropriate coefficient
  if (dslash_type == QUDA_ASQTAD_DSLASH) {
    for (int d = 0; d < 4; d++) {
      for (size_t i = 0; i < V * gauge_site_size; i++) {
        gauge[d][i] /= (-24 * param.tadpole_coeff * param.tadpole_coeff);
      }
    }
  }

  // apply the staggered phases
  for (int d = 0; d < 3; d++) {

    // even
#pragma omp parallel for
    for (int i = 0; i < Vh; i++) {

      int index = fullLatticeIndex(i, 0);
      int i4 = index / (X3 * X2 * X1);
      int i3 = (index - i4 * (X3 * X2 * X1)) / (X2 * X1);
      int i2 = (index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1)) / X1;
      int i1 = index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;
      int sign = 1;

      if (d == 0) {
        if (i4 % 2 == 1) { sign = -1; }
      }

      if (d == 1) {
        if ((i4 + i1) % 2 == 1) { sign = -1; }
      }
      if (d == 2) {
        if ((i4 + i1 + i2) % 2 == 1) { sign = -1; }
      }

      for (int j = 0; j < 18; j++) { gauge[d][i * gauge_site_size + j] *= sign; }
    }
    // odd
    for (int i = 0; i < Vh; i++) {
      int index = fullLatticeIndex(i, 1);
      int i4 = index / (X3 * X2 * X1);
      int i3 = (index - i4 * (X3 * X2 * X1)) / (X2 * X1);
      int i2 = (index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1)) / X1;
      int i1 = index - i4 * (X3 * X2 * X1) - i3 * (X2 * X1) - i2 * X1;
      int sign = 1;

      if (d == 0) {
        if (i4 % 2 == 1) { sign = -1; }
      }

      if (d == 1) {
        if ((i4 + i1) % 2 == 1) { sign = -1; }
      }
      if (d == 2) {
        if ((i4 + i1 + i2) % 2 == 1) { sign = -1; }
      }

      for (int j = 0; j < 18; j++) { gauge[d][(Vh + i) * gauge_site_size + j] *= sign; }
    }
  }

  // Apply boundary conditions to temporal links
  if (param.t_boundary == QUDA_ANTI_PERIODIC_T && last_node_in_t()) {
#pragma omp parallel for
    for (int j = 0; j < Vh; j++) {
      int sign = 1;
      if (dslash_type == QUDA_ASQTAD_DSLASH) {
        if (j >= (X4 - 3) * X1h * X2 * X3) { sign = -1; }
      } else {
        if (j >= (X4 - 1) * X1h * X2 * X3) { sign = -1; }
      }

      for (int i = 0; i < 18; i++) {
        gauge[3][j * gauge_site_size + i] *= sign;
        gauge[3][(Vh + j) * gauge_site_size + i] *= sign;
      }
    }
  }
}

void applyGaugeFieldScaling_long(void *const *gauge, int Vh, const QudaGaugeParam &param, QudaDslashType dslash_type,
                                 QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION) {
    applyGaugeFieldScaling_long((double *const *)gauge, Vh, param, dslash_type);
  } else if (precision == QUDA_SINGLE_PRECISION) {
    applyGaugeFieldScaling_long((float *const *)gauge, Vh, param, dslash_type);
  } else {
    errorQuda("Invalid precision %d", precision);
  }
}
