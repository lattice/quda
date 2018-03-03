#pragma once

#include <quda_internal.h>
#include <quda.h>
#include <gauge_field.h>

namespace quda {
  namespace fermion_force {

    void hisqStaplesForce(GaugeField &newOprod,
                          const GaugeField& oprod,
                          const GaugeField& link,
                          const double path_coeff[6],
                          long long* flops = nullptr);

    void hisqLongLinkForce(GaugeField &newOprod,
                           const GaugeField &oprod,
                           const GaugeField &link,
                           double coeff,
                           long long* flops = nullptr);

    void hisqCompleteForce(GaugeField &force,
                           const GaugeField &oprod,
                           const GaugeField &link,
                           long long* flops = nullptr);

    void setUnitarizeForceConstants(double unitarize_eps, double hisq_force_filter, double max_det_error,
                                    bool allow_svd, bool svd_only,
                                    double svd_rel_error,
                                    double svd_abs_error);

    void unitarizeForce(cudaGaugeField &newForce,
                        const cudaGaugeField &oldForce,
                        const cudaGaugeField &gauge,
                        int* unitarization_failed,
                        long long* flops = NULL);

    void unitarizeForceCPU( cpuGaugeField &newForce,
                            const cpuGaugeField &oldForce,
                            const cpuGaugeField &gauge);

 } // namespace fermion_force
}  // namespace quda
