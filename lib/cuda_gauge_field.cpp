#include <cuda_gauge_field.h>

cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) :
  GaugeField(param, QUDA_CUDA_FIELD_LOCATION)
{


}

cudaGaugeField::~cudaGaugeField() {

}
