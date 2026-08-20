#pragma once

#include <array>
#include <complex>
#include <vector>

#include <gauge_field.h>

/**
 * @brief Host plaquette and rectangle observables, ordered as total, spatial, and temporal.
 */
struct PlaquetteRectangleReference {
  std::array<double, 3> plaquette; /**< Total, spatial, and temporal plaquette. */
  std::array<double, 3> rectangle; /**< Total, spatial, and temporal rectangle. */
};

/**
 * @brief Host mean link determinant and trace with cancellation-safe comparison scales.
 */
struct LinkDeterminantTraceReference {
  std::complex<double> determinant; /**< Mean link determinant. */
  std::complex<double> trace;       /**< Mean link trace. */
  double determinant_scale;         /**< Mean absolute link determinant. */
  double trace_scale;               /**< Mean absolute link trace. */
};

/**
 * @brief Compute the total, spatial, and temporal plaquette on the host.
 *
 * @param[in] u QDP-ordered host gauge field.
 * @return Total, spatial, and temporal plaquette values.
 */
std::array<double, 3> plaquette_reference(const quda::GaugeField &u);

/**
 * @brief Compute the total, spatial, and temporal plaquette and rectangle on the host.
 *
 * @param[in] u QDP-ordered host gauge field.
 * @return Plaquette and rectangle values.
 */
PlaquetteRectangleReference plaquette_rectangle_reference(const quda::GaugeField &u);

/**
 * @brief Compute the real and imaginary parts of the temporal Polyakov loop on the host.
 *
 * @param[in] u QDP-ordered host gauge field.
 * @return Real and imaginary Polyakov-loop values.
 */
std::array<double, 2> polyakov_loop_reference(const quda::GaugeField &u);

/**
 * @brief Compute mean link determinant and trace values and their mean magnitudes.
 *
 * @param[in] u QDP-ordered host gauge field.
 * @return Determinant and trace values with comparison scales.
 */
LinkDeterminantTraceReference link_determinant_trace_reference(const quda::GaugeField &u);

/**
 * @brief Host observables derived from a field-strength tensor.
 */
struct FieldStrengthObservableReference {
  std::array<double, 3> energy; /**< Total, spatial, and temporal gauge energy. */
  double qcharge;               /**< Global topological charge. */
  std::vector<double> qdensity; /**< Local per-site topological-charge density. */
  double qcharge_scale;         /**< Global sum of absolute per-site charge contributions. */
};

/**
 * @brief Compute the clover field-strength tensor on the host.
 *
 * @param[out] fmunu QDP-ordered, tensor-geometry output field.
 * @param[in] u QDP-ordered host gauge field.
 */
void compute_fmunu_reference(quda::GaugeField &fmunu, const quda::GaugeField &u);

/**
 * @brief Compute gauge energy, topological charge, and charge density from a host field-strength tensor.
 *
 * @param[in] fmunu QDP-ordered host field-strength tensor.
 * @return Derived field-strength observables and charge comparison scale.
 */
FieldStrengthObservableReference field_strength_observable_reference(const quda::GaugeField &fmunu);
