#include <quda_internal.h>
#include <timer.h>
#include <gauge_field.h>
#include <assert.h>
#include <string.h>
#include <typeinfo>

namespace quda {

  cpuGaugeField::cpuGaugeField(const GaugeFieldParam &param) :
    GaugeField(param)
  {
    // compute the fat link max now in case it is needed later (i.e., for half precision)
    if (param.compute_fat_link_max) fat_link_max = this->abs_max();
  }

} // namespace quda
