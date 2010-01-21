#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <color_spinor_field.h>

cpuColorSpinorField::cpuColorSpinorField() : 
  ColorSpinorField(), init(false) {

}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) :
  ColorSpinorField(param), init(false) {
  create(param.create);
  if (param.create == QUDA_NULL_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_CREATE) {
    zero();
  } else if (param.create == QUDA_REFERENCE_CREATE) {
    v = param.v;
  } else {
    errorQuda("Creation type not supported");
  }
}

cpuColorSpinorField::cpuColorSpinorField(const cpuColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create(QUDA_COPY_CREATE);
  memcpy(v,src.v,bytes);
}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create(QUDA_COPY_CREATE);
  if (src.fieldType() == QUDA_CPU_FIELD) {
    memcpy(v, dynamic_cast<const cpuColorSpinorField&>(src).v, bytes);
  } else if (src.fieldType() == QUDA_CUDA_FIELD) {
    dynamic_cast<const cudaColorSpinorField&>(src).saveCPUSpinorField(*this);
  } else {
    errorQuda("FieldType not supported");
  }
}

cpuColorSpinorField::~cpuColorSpinorField() {
  destroy();
}

cpuColorSpinorField& cpuColorSpinorField::operator=(const cpuColorSpinorField &src) {
  if (&src != this) {
    destroy();
    ColorSpinorField::operator=(src);
    type = QUDA_CPU_FIELD;
    create(QUDA_COPY_CREATE);
    copy(src);
  }
  return *this;
}

void cpuColorSpinorField::create(const FieldCreate create) {
  if (pad != 0) {
    errorQuda("Non-zero pad not supported");
  }
  
  if (precision == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported");
  }

  if (basis != QUDA_DEGRAND_ROSSI_BASIS) {
    errorQuda("Basis not implemented");
  }

  if (order != QUDA_SPACE_COLOR_SPIN_ORDER || order != QUDA_SPACE_SPIN_COLOR_ORDER) {
    errorQuda("Field order not supported");
  }

  if (create != QUDA_REFERENCE_CREATE) {
    v = (void*)malloc(bytes);
    init = true;
  }
  
}

void cpuColorSpinorField::destroy() {

  if (init) free(v);

}

void cpuColorSpinorField::copy(const cpuColorSpinorField &src) {
  checkField(*this, src);
  memcpy(v, src.v, bytes);
}

void cpuColorSpinorField::zero() {
  memset(v, '0', bytes);
}
