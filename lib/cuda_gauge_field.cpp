#include <cstring>
#include <typeinfo>
#include <gauge_field.h>
#include <timer.h>
#include <blas_quda.h>
#include <device.h>

namespace quda {

  cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) : GaugeField(param) {}

} // namespace quda
