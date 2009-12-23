#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <color_spinor_field.h>

cpuColorSpinorField::cpuColorSpinorField() : 
  ColorSpinorField(), init(false) {

}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) :
  ColorSpinorField(param), init(false) {
  create();
}

cpuColorSpinorField::cpuColorSpinorField(const cpuColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create();
  memcpy(v,src.v,bytes);
}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create();
  if (src.fieldType() == CPU_FIELD) {
    memcpy(v, dynamic_cast<const cpuColorSpinorField&>(src).v, bytes);
  } else if (src.fieldType() == CUDA_FIELD) {
    dynamic_cast<const cudaColorSpinorField&>(src).saveCPUSpinorField(*this);
  }
}

cpuColorSpinorField::~cpuColorSpinorField() {
  destroy();
}

void cpuColorSpinorField::create() {
  if (pad != 0) {
    errorQuda("Non-zero pad not supported");
  }
  
  if (prec == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported");
  }

  if (basis != DEGRAND_ROSSI_BASIS) {
    errorQuda("Basis not implemented");
  }

  if (order != SPACE_COLOR_SPIN_ORDER || order != SPACE_SPIN_COLOR_ORDER) {
    errorQuda("Field order not supported");
  }

  v = (void*)malloc(bytes);
  
  init = true;
}

void cpuColorSpinorField::destroy() {

  if (init) free(v);

}
