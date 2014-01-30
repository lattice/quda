#ifndef __KS_FORCE_QUDA_H__
#define __KS_FORCE_QUDA_H__

#include <gauge_field.h>


namespace quda {

void completeKSForce(GaugeField &mom, const GaugeField &oprod, const GaugeField &gauge, QudaFieldLocation location);

} // namespace quda

#endif
