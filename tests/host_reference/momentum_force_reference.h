#pragma once

#include <gauge_field.h>

void hisqCompleteForceToForceCPU(quda::GaugeField &force_out, const quda::GaugeField &oprod,
                                 const quda::GaugeField &link);

void updateMomentumFromForceCPU(quda::GaugeField &mom, double coeff, const quda::GaugeField &force);
