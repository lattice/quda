#include <quda_internal.h>
#include <timer.h>
#include <gauge_field.h>
#include <assert.h>
#include <string.h>
#include <typeinfo>

namespace quda {

  cpuGaugeField::cpuGaugeField(const GaugeFieldParam &param) : GaugeField(param) {}

} // namespace quda
