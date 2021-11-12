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
    create(param.create);

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
    create(QUDA_COPY_FIELD_CREATE);
    memcpy(v,src.v,bytes);
  }

  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : ColorSpinorField(src)
  {
    create(QUDA_COPY_FIELD_CREATE);
    if (typeid(src) == typeid(cpuColorSpinorField)) {
      memcpy(v, dynamic_cast<const cpuColorSpinorField&>(src).v, bytes);
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      dynamic_cast<const cudaColorSpinorField&>(src).saveSpinorField(*this);
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

    create(param.create);
  }

  cpuColorSpinorField::~cpuColorSpinorField() { destroy(); }

  ColorSpinorField& cpuColorSpinorField::operator=(const ColorSpinorField &src) {
    if (typeid(src) == typeid(cudaColorSpinorField)) {
      *this = (dynamic_cast<const cudaColorSpinorField&>(src));
    } else if (typeid(src) == typeid(cpuColorSpinorField)) {
      *this = (dynamic_cast<const cpuColorSpinorField&>(src));
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
    return *this;
  }

  cpuColorSpinorField& cpuColorSpinorField::operator=(const cpuColorSpinorField &src) {
    if (&src != this) {
      if (!reference) {
	destroy();
	// keep current attributes unless unset
	if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
	create(QUDA_COPY_FIELD_CREATE);
      }
      copy(src);
    }
    return *this;
  }

  cpuColorSpinorField& cpuColorSpinorField::operator=(const cudaColorSpinorField &src) {
    if (!reference) { // if the field is a reference, then we must maintain the current state
      destroy();
      // keep current attributes unless unset
      if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
      create(QUDA_COPY_FIELD_CREATE);
    }
    src.saveSpinorField(*this);
    return *this;
  }

  void cpuColorSpinorField::create(const QudaFieldCreate create) {
    // these need to be reset to ensure no ghost zones for the cpu
    // fields since we can't determine during the parent's constructor
    // whether the field is a cpu or cuda field

    // set this again here.  this is a hack since we can determine we
    // have a cpu or cuda field in ColorSpinorField::create(), which
    // means a ghost zone is set.  So we unset it here.  This will be
    // fixed when clean up the ghost code with the peer-2-peer branch
    bytes = length * precision;
    if (isNative()) bytes = (siteSubset == QUDA_FULL_SITE_SUBSET && fieldOrder != QUDA_QDPJIT_FIELD_ORDER) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);

    if (pad != 0) errorQuda("Non-zero pad not supported");
    if (precision < QUDA_SINGLE_PRECISION) errorQuda("Fixed-point precision not supported");

    if (fieldOrder != QUDA_SPACE_COLOR_SPIN_FIELD_ORDER && 
	fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER &&
	fieldOrder != QUDA_QDPJIT_FIELD_ORDER           &&
	fieldOrder != QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
      errorQuda("Field order %d not supported", fieldOrder);
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      v = safe_malloc(bytes);
      alloc = true;
    }
 
    if (siteSubset == QUDA_FULL_SITE_SUBSET && fieldOrder != QUDA_QDPJIT_FIELD_ORDER) {
      ColorSpinorParam param(*this);
      param.siteSubset = QUDA_PARITY_SITE_SUBSET;
      param.nDim = nDim;
      memcpy(param.x, x, nDim*sizeof(int));
      param.x[0] /= 2;
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      param.v = v;
      param.norm = norm;
      param.is_composite  = false;
      param.composite_dim = 0;
      param.is_component  = composite_descr.is_component;
      param.component_id  = composite_descr.id;
      even = new cpuColorSpinorField(*this, param);
      odd = new cpuColorSpinorField(*this, param);

      // need this hackery for the moment (need to locate the odd pointers half way into the full field)
      (dynamic_cast<cpuColorSpinorField*>(odd))->v = (void*)((char*)v + bytes/2);
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
	(dynamic_cast<cpuColorSpinorField*>(odd))->norm = (void*)((char*)norm + norm_bytes/2);

      if (bytes != 2*even->Bytes() || bytes != 2*odd->Bytes())
	errorQuda("dual-parity fields should have double the size of a single-parity field (%lu,%lu,%lu)\n",
		  bytes, even->Bytes(), odd->Bytes());
    }

  }

  void cpuColorSpinorField::destroy() {
  
    if (alloc) {
      host_free(v);
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      if (even) delete even;
      if (odd) delete odd;
    }

  }

  void cpuColorSpinorField::Source(QudaSourceType source_type, int x, int s, int c) {
    genericSource(*this, source_type, x, s, c);
  }

  int cpuColorSpinorField::Compare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, 
				   const int tol) {    
    checkField(a,b);
    return genericCompare(a, b, tol);
  }

  // print out the vector at volume point x
  void cpuColorSpinorField::PrintVector(unsigned int x) const { genericPrintVector(*this, x); }

  void cpuColorSpinorField::freeGhostBuffer(void)
  {
    if(!initGhostFaceBuffer) return;

    for(int i=0; i < 4; i++){  // make nDimComms static?
      host_free(fwdGhostFaceBuffer[i]); fwdGhostFaceBuffer[i] = NULL;
      host_free(backGhostFaceBuffer[i]); backGhostFaceBuffer[i] = NULL;
      host_free(fwdGhostFaceSendBuffer[i]); fwdGhostFaceSendBuffer[i] = NULL;
      host_free(backGhostFaceSendBuffer[i]);  backGhostFaceSendBuffer[i] = NULL;
    } 
    initGhostFaceBuffer = 0;
  }


  void cpuColorSpinorField::unpackGhost(void *, const int, const QudaDirection)
  {
    if (this->siteSubset == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in unpackGhost for cpu");
    }
  }

} // namespace quda
