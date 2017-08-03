#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <color_spinor_field.h>
#include <comm_quda.h> // for comm_drand()

namespace quda {

  int cpuColorSpinorField::initGhostFaceBuffer =0;
  void* cpuColorSpinorField::fwdGhostFaceBuffer[QUDA_MAX_DIM]; 
  void* cpuColorSpinorField::backGhostFaceBuffer[QUDA_MAX_DIM];
  void* cpuColorSpinorField::fwdGhostFaceSendBuffer[QUDA_MAX_DIM]; 
  void* cpuColorSpinorField::backGhostFaceSendBuffer[QUDA_MAX_DIM];

  size_t cpuColorSpinorField::ghostFaceBytes[QUDA_MAX_DIM] = { };

  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) :
    ColorSpinorField(param), init(false), reference(false) {

    // need to set this before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      reference = true;
    }

    create(param.create);

    if (param.create == QUDA_NULL_FIELD_CREATE || param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else {
      errorQuda("Creation type %d not supported", param.create);
    }
  }

  cpuColorSpinorField::cpuColorSpinorField(const cpuColorSpinorField &src) : 
    ColorSpinorField(src), init(false), reference(false) {
    create(QUDA_COPY_FIELD_CREATE);
    memcpy(v,src.v,bytes);
  }

  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : 
    ColorSpinorField(src), init(false), reference(false) {
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
  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src, const ColorSpinorParam &param) : 
    ColorSpinorField(src), init(false), reference(false) {

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

  cpuColorSpinorField::~cpuColorSpinorField() {
    destroy();
  }

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
    ghost_length = 0;
    ghost_norm_length = 0;

    // set this again here.  this is a hack since we can determine we
    // have a cpu or cuda field in ColorSpinorField::create(), which
    // means a ghost zone is set.  So we unset it here.  This will be
    // fixed when clean up the ghost code with the peer-2-peer branch
    bytes = length * precision;
    if (isNative()) bytes = (siteSubset == QUDA_FULL_SITE_SUBSET && fieldOrder != QUDA_QDPJIT_FIELD_ORDER) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);


    if (pad != 0) errorQuda("Non-zero pad not supported");  
    if (precision == QUDA_HALF_PRECISION) errorQuda("Half precision not supported");

    if (fieldOrder != QUDA_SPACE_COLOR_SPIN_FIELD_ORDER && 
	fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER &&
	fieldOrder != QUDA_QOP_DOMAIN_WALL_FIELD_ORDER  &&
	fieldOrder != QUDA_QDPJIT_FIELD_ORDER           &&
	fieldOrder != QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
      errorQuda("Field order %d not supported", fieldOrder);
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      // array of 4-d fields
      if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
        int Ls = x[nDim-1];
        v = (void**)safe_malloc(Ls * sizeof(void*));
        for (int i=0; i<Ls; i++) ((void**)v)[i] = safe_malloc(bytes / Ls);
      } else {
        v = safe_malloc(bytes);
      }
      init = true;
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
      if (precision == QUDA_HALF_PRECISION)
	(dynamic_cast<cpuColorSpinorField*>(odd))->norm = (void*)((char*)norm + norm_bytes/2);

      if (bytes != 2*even->Bytes() || bytes != 2*odd->Bytes())
	errorQuda("dual-parity fields should have double the size of a single-parity field (%lu,%lu,%lu)\n",
		  bytes, even->Bytes(), odd->Bytes());
    }

  }

  void cpuColorSpinorField::destroy() {
  
    if (init) {
      if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
	for (int i=0; i<x[nDim-1]; i++) host_free(((void**)v)[i]);
      host_free(v);
      init = false;
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      if (even) delete even;
      if (odd) delete odd;
    }

  }

  void cpuColorSpinorField::copy(const cpuColorSpinorField &src) {
    checkField(*this, src);
    if (fieldOrder == src.fieldOrder && bytes == src.Bytes()) {
      if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
        for (int i=0; i<x[nDim-1]; i++) memcpy(((void**)v)[i], ((void**)src.v)[i], bytes/x[nDim-1]);
      else 
        memcpy(v, src.v, bytes);
    } else {
      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION);
    }
  }

  void cpuColorSpinorField::backup() const {
    if (backed_up) errorQuda("Field already backed up");

    backup_h = new char[bytes];
    memcpy(backup_h, v, bytes);

    if (norm_bytes) {
      backup_norm_h = new char[norm_bytes];
      memcpy(backup_norm_h, norm, norm_bytes);
    }

    backed_up = true;
  }

  void cpuColorSpinorField::restore() {
    if (!backed_up) errorQuda("Cannot restore since not backed up");

    memcpy(v, backup_h, bytes);
    delete []backup_h;
    if (norm_bytes) {
      memcpy(norm, backup_norm_h, norm_bytes);
      delete []backup_norm_h;
    }

    backed_up = false;
  }

  void cpuColorSpinorField::zero() {
    if (fieldOrder != QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) memset(v, '\0', bytes);
    else for (int i=0; i<x[nDim-1]; i++) memset(((void**)v)[i], '\0', bytes/x[nDim-1]);
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
  void cpuColorSpinorField::PrintVector(unsigned int x) { genericPrintVector(*this, x); }

  void cpuColorSpinorField::allocateGhostBuffer(int nFace) const
  {
    int spinor_size = 2*nSpin*nColor*precision;
    bool resize = false;

    // resize face only if requested size is larger than previously allocated one
    for (int i=0; i<nDimComms; i++) {
      size_t nbytes = siteSubset*nFace*surfaceCB[i]*spinor_size;
      resize = (nbytes > ghostFaceBytes[i]) ? true : resize;
      ghostFaceBytes[i] = (nbytes > ghostFaceBytes[i]) ? nbytes : ghostFaceBytes[i];
    }

    if (!initGhostFaceBuffer || resize) {
      freeGhostBuffer();
      for (int i=0; i<nDimComms; i++) {
	fwdGhostFaceBuffer[i] = safe_malloc(ghostFaceBytes[i]);
	backGhostFaceBuffer[i] = safe_malloc(ghostFaceBytes[i]);
	fwdGhostFaceSendBuffer[i] = safe_malloc(ghostFaceBytes[i]);
	backGhostFaceSendBuffer[i] = safe_malloc(ghostFaceBytes[i]);
      }
      initGhostFaceBuffer = 1;
    }
  }


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


  void cpuColorSpinorField::packGhost(void **ghost, const QudaParity parity, const int nFace, const int dagger) const
  {
    genericPackGhost(ghost, *this, parity, nFace, dagger);
    return;
  }

  void cpuColorSpinorField::unpackGhost(void* ghost_spinor, const int dim, 
					const QudaDirection dir, const int dagger)
  {
    if (this->siteSubset == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in unpackGhost for cpu");
    }
  }

  void cpuColorSpinorField::exchangeGhost(QudaParity parity, int nFace, int dagger, const MemoryLocation *dummy1,
					  const MemoryLocation *dummy2, bool dummy3, bool dummy4) const
  {
    // allocate ghost buffer if not yet allocated
    allocateGhostBuffer(nFace);

    void **sendbuf = static_cast<void**>(safe_malloc(nDimComms * 2 * sizeof(void*)));

    for (int i=0; i<nDimComms; i++) {
      sendbuf[2*i + 0] = backGhostFaceSendBuffer[i];
      sendbuf[2*i + 1] = fwdGhostFaceSendBuffer[i];
      ghost_buf[2*i + 0] = backGhostFaceBuffer[i];
      ghost_buf[2*i + 1] = fwdGhostFaceBuffer[i];
    }

    packGhost(sendbuf, parity, nFace, dagger);

    exchange(ghost_buf, sendbuf, nFace);

    host_free(sendbuf);
  }

} // namespace quda
