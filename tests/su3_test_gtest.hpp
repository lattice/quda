#pragma once

#include <gtest/gtest.h>

void plaquette_test();
void polyakov_loop_test();
void topological_charge_and_density_test();
void gauge_smearing_or_flow_test();

TEST(SU3Test, Plaquette) { plaquette_test(); }

TEST(SU3Test, PolyakovLoop) { polyakov_loop_test(); }

TEST(SU3Test, TopologicalChargeAndDensity) { topological_charge_and_density_test(); }

TEST(SU3Test, GaugeSmearingOrFlow) { gauge_smearing_or_flow_test(); }
