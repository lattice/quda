#ifndef _HISQ_FORCE_REFERENCE_H
#define _HISQ_FORCE_REFERENCE_H

#include <quda.h>
#include <enum_quda.h>
#include <gauge_field.h>

/**
   @brief Compute a staggered spinor outer product for some offset, CPU version
   @param[in] src Pointer to an appropriately sized host staggered spinor field
   @param[out] dest Reference to a gauge field for the outer product
   @param[in] precision Precision of data (single or double)
   @param[in] separation Offset for outer product (1 for fat links, 3 for long links)
*/
void computeLinkOrderedOuterProduct(void *src, quda::GaugeField &dest, QudaPrecision precision, size_t separation);

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
