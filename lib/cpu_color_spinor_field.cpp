#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <color_spinor_field.h>
#include <comm_quda.h> // for comm_drand()

namespace quda {

  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) : ColorSpinorField(param)
  {
    create2(param.create);

    if (param.create == QUDA_NULL_FIELD_CREATE || param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else {
      errorQuda("Creation type %d not supported", param.create);
    }
  }

  cpuColorSpinorField::cpuColorSpinorField(const cpuColorSpinorField &src) : ColorSpinorField(src)
  {
    create2(QUDA_COPY_FIELD_CREATE);
    memcpy(v,src.v,bytes);
  }

  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : ColorSpinorField(src)
  {
    create2(QUDA_COPY_FIELD_CREATE);
    if (typeid(src) == typeid(cpuColorSpinorField)) {
      memcpy(v, dynamic_cast<const cpuColorSpinorField&>(src).v, bytes);
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copy(*this);
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
  }

  /*
    This is special case constructor used to create parity subset references with in a full field
   */
  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src, const ColorSpinorParam &param) : ColorSpinorField(src)
  {
    // can only overide if we parity subset reference special case
    if ( param.create == QUDA_REFERENCE_FIELD_CREATE &&
	 src.SiteSubset() == QUDA_FULL_SITE_SUBSET &&
	 param.siteSubset == QUDA_PARITY_SITE_SUBSET &&
	 typeid(src) == typeid(cpuColorSpinorField) ) {
      reset(param);
    } else {
      errorQuda("Undefined behaviour"); // else silent bug possible?
    }

    // need to set this before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = (void*)src.V();
      norm = (void*)src.Norm();
    }

    create2(param.create);
  }

  cpuColorSpinorField::~cpuColorSpinorField() { destroy2(); }

  void cpuColorSpinorField::unpackGhost(void *, const int, const QudaDirection)
  {
    if (this->siteSubset == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in unpackGhost for cpu");
    }
  }

} // namespace quda
