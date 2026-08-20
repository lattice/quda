#pragma once

#include <array>

#include <gauge_field.h>

/**
 * @brief Compute the total, spatial, and temporal plaquette on the host.
 *
 * @param[in] u QDP-ordered host gauge field.
 * @return Total, spatial, and temporal plaquette values.
 */
std::array<double, 3> plaquette_reference(const quda::GaugeField &u);
