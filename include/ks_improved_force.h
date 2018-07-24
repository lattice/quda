#pragma once

#include <quda_internal.h>
#include <quda.h>
#include <gauge_field.h>

namespace quda {
  namespace fermion_force {

    /**
       @brief Compute the fat-link contribution to the fermion force
       @param[out] newOprod The computed force output
       @param[in] oprod The previously computed input force
       @param[in] link Thin-link gauge field
       @param[in] path_coeff Coefficients of the contributions to the operator
       @param[out] flops number of flops performed
    */
    void hisqStaplesForce(GaugeField &newOprod,
                          const GaugeField& oprod,
                          const GaugeField& link,
                          const double path_coeff[6],
                          long long* flops = nullptr);

    /**
       @brief Compute the long-link contribution to the fermion force
       @param[out] newOprod The computed force output
       @param[in] oprod The previously computed input force
       @param[in] link Thin-link gauge field
       @param[in] coeff Long-link Coefficient
       @param[out] flops number of flops performed
    */
    void hisqLongLinkForce(GaugeField &newOprod,
                           const GaugeField &oprod,
                           const GaugeField &link,
                           double coeff,
                           long long* flops = nullptr);

    /**
       @brief Multiply the computed the force matrix by the gauge
       field and perform traceless anti-hermitian projection
       @param[out] momentum The computed momentum
       @param[in] oprod The previously computed force
       @param[in] link Thin-link gauge field
       @param[out] flops number of flops performed
    */
    void hisqCompleteForce(GaugeField &momentum,
                           const GaugeField &oprod,
                           const GaugeField &link,
                           long long* flops = nullptr);

    /**
       @brief Set the constant parameters for the force unitarization
    */
    void setUnitarizeForceConstants(double unitarize_eps, double hisq_force_filter, double max_det_error,
                                    bool allow_svd, bool svd_only,
                                    double svd_rel_error,
                                    double svd_abs_error);

    /**
       @brief Unitarize the fermion force
       @param[in] newForce Unitarized output
       @param[in] oldForce Input force
       @param[in] gauge Gauge field
       @param[out] unitarization_failed Whether the unitarization failed (number of failures)
       @param[out] flops number of flops performed
    */
    void unitarizeForce(cudaGaugeField &newForce,
                        const cudaGaugeField &oldForce,
                        const cudaGaugeField &gauge,
                        int* unitarization_failed,
                        long long* flops = NULL);

    /**
       @brief Unitarize the fermion force on CPU
       @param[in] newForce Unitarized output
       @param[in] oldForce Input force
       @param[in] gauge Gauge field
    */
    void unitarizeForceCPU(cpuGaugeField &newForce,
                           const cpuGaugeField &oldForce,
                           const cpuGaugeField &gauge);

 } // namespace fermion_force
}  // namespace quda
