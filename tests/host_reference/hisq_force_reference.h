#ifndef _HISQ_FORCE_REFERENCE_H
#define _HISQ_FORCE_REFERENCE_H

#include <array>

#include <quda.h>
#include <enum_quda.h>
#include <gauge_field.h>
#include <color_spinor_field.h>

/**
   @brief Compute a staggered spinor outer product for some offset, CPU version
   @param[in] src Full-parity host staggered spinor field; its ghost zones are exchanged here
   @param[out] dest Reference to a gauge field for the outer product
   @param[in] separation Odd offset for outer product (1 for fat links, 3 for long links)
*/
void computeLinkOrderedOuterProduct(const quda::ColorSpinorField &src, quda::GaugeField &dest, int separation);

/**
   @brief Accumulate the staggered outer product(s) of a quark field, CPU version of computeStaggeredOprod
   @param[in,out] out Host QDP-ordered fields to accumulate into: out[0] one-hop, out[1] three-hop (nFace = 3 only;
   otherwise unused)
   @param[in] in Full-parity host staggered quark field, of the same precision as out
   @param[in] coeff Coefficients for the one- and three-hop terms
   @param[in] nFace 1 (naive staggered, odd sites take -coeff[0]) or 3 (HISQ/asqtad)
*/
void computeStaggeredOprodCPU(std::array<quda::GaugeField, 2> &out, const quda::ColorSpinorField &in,
                              const double coeff[], int nFace);

/**
   @brief Compute the force contribution from the fat links, CPU version
   @param[in] path_coeff Input HISQ coefficients
   @param[in] oprod Input force outer product
   @param[in] link Gauge field links
   @param[out] newOprod Force accumulated with fat link contributions
*/
void hisqStaplesForceCPU(const double *path_coeff, quda::GaugeField &oprod, quda::GaugeField &link,
                         quda::GaugeField *newOprod);

/**
   @brief Compute the force contribution from the long link, CPU version
   @param[in] coeff Long-link contribution (path_coeff[1])
   @param[in] oprod Input force outer product
   @param[in] link Gauge field links
   @param[out] newOprod Force accumulated with fat link contributions
*/
void hisqLongLinkForceCPU(double coeff, quda::GaugeField &oprod, quda::GaugeField &link, quda::GaugeField *newOprod);

/**
   @brief Accumulate the force contributions into the momentum field, CPU version
   @param[in] oprod Input force outer product
   @param[in] link Gauge field links
   @param[out] mom Accumulated momentum
*/
void hisqCompleteForceCPU(quda::GaugeField &oprod, quda::GaugeField &link, quda::GaugeField *mom);

#endif
