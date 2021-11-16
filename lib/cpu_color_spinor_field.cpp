#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <color_spinor_field.h>
#include <comm_quda.h> // for comm_drand()

namespace quda {

#if 0
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
#endif

} // namespace quda
