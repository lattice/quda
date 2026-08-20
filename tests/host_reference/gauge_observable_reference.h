#pragma once

#include <array>
#include <complex>

#include <gauge_field.h>

struct PlaquetteRectangleReference {
  std::array<double, 3> plaquette;
  std::array<double, 3> rectangle;
};

struct LinkDeterminantTraceReference {
  std::complex<double> determinant;
  std::complex<double> trace;
  double determinant_scale;
  double trace_scale;
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
