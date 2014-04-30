#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <color_spinor_field.h>
#include <comm_quda.h> // for comm_drand()

/*
Maybe this will be useful at some point

#define myalloc(type, n, m0) (type *) aligned_malloc(n*sizeof(type), m0)

#define ALIGN 16
void *
aligned_malloc(size_t n, void **m0)
{
  size_t m = (size_t) safe_malloc(n+ALIGN);
  *m0 = (void*)m;
  size_t r = m % ALIGN;
  if(r) m += (ALIGN - r);
  return (void *)m;
}
*/

namespace quda {

  /*cpuColorSpinorField::cpuColorSpinorField() : 
    ColorSpinorField(), init(false) {

    }*/


  int cpuColorSpinorField::initGhostFaceBuffer =0;
  void* cpuColorSpinorField::fwdGhostFaceBuffer[QUDA_MAX_DIM]; 
  void* cpuColorSpinorField::backGhostFaceBuffer[QUDA_MAX_DIM];
  void* cpuColorSpinorField::fwdGhostFaceSendBuffer[QUDA_MAX_DIM]; 
  void* cpuColorSpinorField::backGhostFaceSendBuffer[QUDA_MAX_DIM];

  cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) :
    ColorSpinorField(param), init(false), reference(false) {
    create(param.create);
    if (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      reference = true;
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
    total_length = length;
    total_norm_length = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*stride : stride;
    bytes = total_length * precision; // includes pads and ghost zones
    bytes = ALIGNMENT_ADJUST(bytes);

    if (pad != 0) errorQuda("Non-zero pad not supported");  
    if (precision == QUDA_HALF_PRECISION) errorQuda("Half precision not supported");

    if (fieldOrder != QUDA_SPACE_COLOR_SPIN_FIELD_ORDER && 
	fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER &&
	fieldOrder != QUDA_QOP_DOMAIN_WALL_FIELD_ORDER &&
	fieldOrder != QUDA_QDPJIT_FIELD_ORDER) {
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
 
  }

  void cpuColorSpinorField::destroy() {
  
    if (init) {
      if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
	for (int i=0; i<x[nDim-1]; i++) host_free(((void**)v)[i]);
      host_free(v);
      init = false;
    }

  }

  void cpuColorSpinorField::copy(const cpuColorSpinorField &src) {
    checkField(*this, src);
    if (fieldOrder == src.fieldOrder) {
      if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
	for (int i=0; i<x[nDim-1]; i++) memcpy(((void**)v)[i], ((void**)src.v)[i], bytes);
      else 
	memcpy(v, src.v, bytes);
    } else {
      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION);
    }
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

  void cpuColorSpinorField::allocateGhostBuffer(void)
  {
    if (initGhostFaceBuffer) return;

    if (this->siteSubset == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in alllocateGhostBuffer\n");
    }
  
    int X1 = this->x[0]*2;
    int X2 = this->x[1];
    int X3 = this->x[2];
    int X4 = this->x[3];
    int X5 = this->nDim == 5 ? this->x[4] : 1;
  
    int Vsh[4]={ X2*X3*X4*X5/2,
		 X1*X3*X4*X5/2,
		 X1*X2*X4*X5/2,
		 X1*X2*X3*X5/2};
  
    int num_faces = 1;
    if(this->nSpin == 1) num_faces = 3; // staggered

    int spinor_size = 2*this->nSpin*this->nColor*this->precision;
    for (int i=0; i<4; i++) {
      size_t nbytes = num_faces*Vsh[i]*spinor_size;

      fwdGhostFaceBuffer[i] = safe_malloc(nbytes);
      backGhostFaceBuffer[i] = safe_malloc(nbytes);
      fwdGhostFaceSendBuffer[i] = safe_malloc(nbytes);
      backGhostFaceSendBuffer[i] = safe_malloc(nbytes);
    }
    initGhostFaceBuffer = 1;
  }


  void cpuColorSpinorField::freeGhostBuffer(void)
  {
    if(!initGhostFaceBuffer) return;

    for(int i=0;i < 4; i++){
      host_free(fwdGhostFaceBuffer[i]); fwdGhostFaceBuffer[i] = NULL;
      host_free(backGhostFaceBuffer[i]); backGhostFaceBuffer[i] = NULL;
      host_free(fwdGhostFaceSendBuffer[i]); fwdGhostFaceSendBuffer[i] = NULL;
      host_free(backGhostFaceSendBuffer[i]);  backGhostFaceSendBuffer[i] = NULL;
    } 
    initGhostFaceBuffer = 0;
  }


  void cpuColorSpinorField::packGhost(void* ghost_spinor, const int dim, 
				      const QudaDirection dir, const QudaParity oddBit, const int dagger)
  {
    if (this->siteSubset == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in packGhost for cpu");
    }
  
    if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      errorQuda("Field order %d not supported", fieldOrder);
    }

    int num_faces=1;
    if(this->nSpin == 1){ //staggered
      num_faces=3;
    }
    int spinor_size = 2*this->nSpin*this->nColor*this->precision;

    int X1 = this->x[0]*2;
    int X2 = this->x[1];
    int X3 = this->x[2];
    int X4 = this->x[3];
    int X5 = this->nDim == 5 ? this->x[4]: 1;


    for(int i=0;i < this->volume;i++){ 
    
      int X1h = X1/2;
    
      int sid =i;
      int za = sid/X1h;
      int x1h = sid - za*X1h;
      int zb = za/X2;
      int x2 = za - zb*X2;
      int zc = zb / X3;
      int x3 = zb - zc*X3;
      int x5 = zc / X4; //this->nDim == 5 ? zz / X4 : 0;
      int x4 = zc - x5*X4;
      int x1odd = (x2 + x3 + x4 + x5 + oddBit) & 1;
      int x1 = 2*x1h + x1odd;

      int ghost_face_idx ;
    
      //NOTE: added extra dimension for DW and TM dslash    

      switch(dim){            
      case 0: //X dimension
	if (dir == QUDA_BACKWARDS){
	  if (x1 < num_faces){
	    ghost_face_idx =  (x1*X5*X4*X3*X2 + x5*X4*X3*X2 + x4*(X3*X2)+x3*X2 +x2)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);
	  }
	}else{  // QUDA_FORWARDS
	  if (x1 >=X1 - num_faces){
	    ghost_face_idx = ((x1-X1+num_faces)*X5*X4*X3*X2 + x5*X4*X3*X2 + x4*(X3*X2)+x3*X2 +x2)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);	  
	  }
	}
	break;      
      
      case 1: //Y dimension
	if (dir == QUDA_BACKWARDS){
	  if (x2 < num_faces){
	    ghost_face_idx = (x2*X5*X4*X3*X1 +x5*X4*X3*X1 + x4*X3*X1+x3*X1+x1)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);	  
	  }
	}else{ // QUDA_FORWARDS      
	  if (x2 >= X2 - num_faces){
	    ghost_face_idx = ((x2-X2+num_faces)*X5*X4*X3*X1 +x5*X4*X3*X1+ x4*X3*X1+x3*X1+x1)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);	  
	  }
	}
	break;

      case 2: //Z dimension      
	if (dir == QUDA_BACKWARDS){
	  if (x3 < num_faces){
	    ghost_face_idx = (x3*X5*X4*X2*X1 + x5*X4*X2*X1 + x4*X2*X1+x2*X1+x1)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);	  
	  }
	}else{ // QUDA_FORWARDS     
	  if (x3 >= X3 - num_faces){
	    ghost_face_idx = ((x3-X3+num_faces)*X5*X4*X2*X1 + x5*X4*X2*X1 + x4*X2*X1 + x2*X1 + x1)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);	  
	  }
	}
	break;
      
      case 3:  //T dimension      
	if (dir == QUDA_BACKWARDS){
	  if (x4 < num_faces){
	    ghost_face_idx = (x4*X5*X3*X2*X1 + x5*X3*X2*X1 + x3*X2*X1+x2*X1+x1)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);	  
	  }
	}else{ // QUDA_FORWARDS     
	  if (x4 >= X4 - num_faces){
	    ghost_face_idx = ((x4-X4+num_faces)*X5*X3*X2*X1 + x5*X3*X2*X1 + x3*X2*X1+x2*X1+x1)>>1;
	    memcpy( ((char*)ghost_spinor) + ghost_face_idx*spinor_size, ((char*)v)+i*spinor_size, spinor_size);	  
	  }
	}
	break;
      default:
	errorQuda("Invalid dim value\n");
      }//switch
    }//for i
    return;
  }

  void cpuColorSpinorField::unpackGhost(void* ghost_spinor, const int dim, 
					const QudaDirection dir, const int dagger)
  {
    if (this->siteSubset == QUDA_FULL_SITE_SUBSET){
      errorQuda("Full spinor is not supported in unpackGhost for cpu");
    }
  }

  // Return the location of the field
  QudaFieldLocation cpuColorSpinorField::Location() const { 
    return QUDA_CPU_FIELD_LOCATION;
  }

} // namespace quda
