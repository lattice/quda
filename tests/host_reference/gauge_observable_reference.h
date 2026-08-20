#pragma once

#include <array>

#include <gauge_field.h>

struct PlaquetteRectangleReference {
  std::array<double, 3> plaquette;
  std::array<double, 3> rectangle;
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
