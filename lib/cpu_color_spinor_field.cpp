#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <color_spinor_field.h>
#include <color_spinor_field_order.h>

/*
Maybe this will be useful at some point

#define myalloc(type, n, m0) (type *) aligned_malloc(n*sizeof(type), m0)

#define ALIGN 16
void *
aligned_malloc(size_t n, void **m0)
{
  size_t m = (size_t) malloc(n+ALIGN);
  *m0 = (void*)m;
  size_t r = m % ALIGN;
  if(r) m += (ALIGN - r);
  return (void *)m;
}
*/

 /*cpuColorSpinorField::cpuColorSpinorField() : 
  ColorSpinorField(), init(false) {

  }*/


int cpuColorSpinorField::initGhostFaceBuffer =0;
void* cpuColorSpinorField::fwdGhostFaceBuffer[QUDA_MAX_DIM]; 
void* cpuColorSpinorField::backGhostFaceBuffer[QUDA_MAX_DIM];
void* cpuColorSpinorField::fwdGhostFaceSendBuffer[QUDA_MAX_DIM]; 
void* cpuColorSpinorField::backGhostFaceSendBuffer[QUDA_MAX_DIM];

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorParam &param) :
  ColorSpinorField(param), init(false), order_double(NULL), order_single(NULL) {
  create(param.create);
  if (param.create == QUDA_NULL_FIELD_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
    zero();
  } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
    v = param.v;
  } else {
    errorQuda("Creation type %d not supported", param.create);
  }

  if (fieldLocation != QUDA_CPU_FIELD_LOCATION) 
    errorQuda("Location incorrectly set");
}

cpuColorSpinorField::cpuColorSpinorField(const cpuColorSpinorField &src) : 
  ColorSpinorField(src), init(false), order_double(NULL), order_single(NULL) {
  create(QUDA_COPY_FIELD_CREATE);
  memcpy(v,src.v,bytes);

  if (fieldLocation != QUDA_CPU_FIELD_LOCATION) 
    errorQuda("Location incorrectly set");
}

cpuColorSpinorField::cpuColorSpinorField(const ColorSpinorField &src) : 
  ColorSpinorField(src), init(false), order_double(NULL), order_single(NULL) {
  create(QUDA_COPY_FIELD_CREATE);
  if (src.FieldLocation() == QUDA_CPU_FIELD_LOCATION) {
    memcpy(v, dynamic_cast<const cpuColorSpinorField&>(src).v, bytes);
  } else if (src.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) {
    dynamic_cast<const cudaColorSpinorField&>(src).saveCPUSpinorField(*this);
  } else {
    errorQuda("FieldType not supported");
  }

  if (fieldLocation != QUDA_CPU_FIELD_LOCATION) 
    errorQuda("Location incorrectly set");
}

cpuColorSpinorField::~cpuColorSpinorField() {
  destroy();
}

cpuColorSpinorField& cpuColorSpinorField::operator=(const cpuColorSpinorField &src) {
  if (&src != this) {
    destroy();
    // keep current attributes unless unset
    if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
    fieldLocation = QUDA_CPU_FIELD_LOCATION;
    create(QUDA_COPY_FIELD_CREATE);
    copy(src);
  }
  return *this;
}

cpuColorSpinorField& cpuColorSpinorField::operator=(const cudaColorSpinorField &src) {
  destroy();
  // keep current attributes unless unset
  if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
  fieldLocation = QUDA_CPU_FIELD_LOCATION;
  create(QUDA_COPY_FIELD_CREATE);
  src.saveCPUSpinorField(*this);
  return *this;
}

void cpuColorSpinorField::create(const QudaFieldCreate create) {
  if (pad != 0) {
    errorQuda("Non-zero pad not supported");
  }
  
  if (precision == QUDA_HALF_PRECISION) {
    errorQuda("Half precision not supported");
  }

  if (fieldOrder != QUDA_SPACE_COLOR_SPIN_FIELD_ORDER && 
      fieldOrder != QUDA_SPACE_SPIN_COLOR_FIELD_ORDER &&
      fieldOrder != QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
    errorQuda("Field order %d not supported", fieldOrder);
  }

  if (create != QUDA_REFERENCE_FIELD_CREATE) {
    // array of 4-d fields
    if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
      int Ls = x[nDim-1];
      v = (void**)malloc(Ls * sizeof(void*));
      for (int i=0; i<Ls; i++) ((void**)v)[i] = (void*)malloc(bytes / Ls);
    } else {
      v = (void*)malloc(bytes);
      init = true;
    }
  }
 
  createOrder(); // need to do this for references?
}

void cpuColorSpinorField::createOrder() {

  if (precision == QUDA_DOUBLE_PRECISION) {
    if (fieldOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) 
      order_double = new SpaceSpinColorOrder<double>(*this);
    else if (fieldOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) 
      order_double = new SpaceColorSpinOrder<double>(*this);
    else if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
      order_double = new QOPDomainWallOrder<double>(*this);
    else
      errorQuda("Order %d not supported in cpuColorSpinorField", fieldOrder);
  } else if (precision == QUDA_SINGLE_PRECISION) {
    if (fieldOrder == QUDA_SPACE_SPIN_COLOR_FIELD_ORDER) 
      order_single = new SpaceSpinColorOrder<float>(*this);
    else if (fieldOrder == QUDA_SPACE_COLOR_SPIN_FIELD_ORDER) 
      order_single = new SpaceColorSpinOrder<float>(*this);
    else if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
      order_single = new QOPDomainWallOrder<float>(*this);
    else
      errorQuda("Order %d not supported in cpuColorSpinorField", fieldOrder);
  } else {
    errorQuda("Precision %d not supported", precision);
  }
  
}

void cpuColorSpinorField::destroy() {
  
  if (precision == QUDA_DOUBLE_PRECISION) {
    delete order_double;
  } else if (precision == QUDA_SINGLE_PRECISION) {
    delete order_single;
  } else {
    errorQuda("Precision %d not supported", precision);
  }
  
  if (init) {
    if (fieldOrder == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) 
      for (int i=0; i<x[nDim-1]; i++) free(((void**)v)[i]);
    free(v);
    init = false;
  }

}

template <class D, class S>
void genericCopy(D &dst, const S &src) {

  for (int x=0; x<dst.Volume(); x++) {
    for (int s=0; s<dst.Nspin(); s++) {
      for (int c=0; c<dst.Ncolor(); c++) {
	for (int z=0; z<2; z++) {
	  dst(x, s, c, z) = src(x, s, c, z);
	}
      }
    }
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
    if (precision == QUDA_DOUBLE_PRECISION) {
      if (src.precision == QUDA_DOUBLE_PRECISION) {
	genericCopy(*order_double, *(src.order_double));
      } else {
	genericCopy(*order_double, *(src.order_single));
      }
    } else {
      if (src.precision == QUDA_DOUBLE_PRECISION) {
	genericCopy(*order_single, *(src.order_double));
      } else {
	genericCopy(*order_single, *(src.order_single));
      }
    }
  }
}

void cpuColorSpinorField::zero() {
  if (fieldOrder != QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) memset(v, '\0', bytes);
  else for (int i=0; i<x[nDim-1]; i++) memset(((void**)v)[i], '\0', bytes/x[nDim-1]);
}

// Random number insertion over all field elements
template <class T>
void random(T &t) {
  for (int x=0; x<t.Volume(); x++) {
    for (int s=0; s<t.Nspin(); s++) {
      for (int c=0; c<t.Ncolor(); c++) {
	for (int z=0; z<2; z++) {
	  t(x,s,c,z) = rand() / (double)RAND_MAX;
	}
      }
    }
  }
}

// Create a point source at spacetime point x, spin s and colour c
template <class T>
void point(T &t, const int x, const int s, const int c) { t(x, s, c, 0) = 1.0; }

void cpuColorSpinorField::Source(const QudaSourceType sourceType, const int x,
				 const int s, const int c) {

  switch(sourceType) {

  case QUDA_RANDOM_SOURCE:
    if (precision == QUDA_DOUBLE_PRECISION) random(*order_double);
    else if (precision == QUDA_SINGLE_PRECISION) random(*order_single);
    else errorQuda("Precision not supported");
    break;

  case QUDA_POINT_SOURCE:
    zero();
    if (precision == QUDA_DOUBLE_PRECISION) point(*order_double, x, s, c);
    else if (precision == QUDA_SINGLE_PRECISION) point(*order_single, x, s, c);
    else errorQuda("Precision not supported");
    break;

  default:
    errorQuda("Source type %d not implemented", sourceType);

  }

}

template <class U, class V>
int compareSpinor(const U &u, const V &v, const int tol) {
  int fail_check = 16*tol;
  int *fail = new int[fail_check];
  for (int f=0; f<fail_check; f++) fail[f] = 0;

  int N = 2*u.Nspin()*u.Ncolor();
  int *iter = new int[N];
  for (int i=0; i<N; i++) iter[i] = 0;

  for (int x=0; x<u.Volume(); x++) {
    for (int s=0; s<u.Nspin(); s++) {
      for (int c=0; c<u.Ncolor(); c++) {
	for (int z=0; z<2; z++) {
	  double diff = fabs(u(x,s,c,z) - v(x,s,c,z));

	  for (int f=0; f<fail_check; f++)
	    if (diff > pow(10.0,-(f+1)/(double)tol)) fail[f]++;

	  int j = (s*u.Ncolor() + c)*2+z;
	  if (diff > 1e-3) iter[j]++;
	}
      }
    }
  }

  for (int i=0; i<N; i++) printfQuda("%d fails = %d\n", i, iter[i]);
    
  int accuracy_level =0;
  for (int f=0; f<fail_check; f++) {
    if (fail[f] == 0) accuracy_level = f+1;
  }

  for (int f=0; f<fail_check; f++) {
    printfQuda("%e Failures: %d / %d  = %e\n", pow(10.0,-(f+1)/(double)tol), 
	       fail[f], u.Volume()*N, fail[f] / (double)(u.Volume()*N));
  }
  
  delete []iter;
  delete []fail;
  
  return accuracy_level;
}

int cpuColorSpinorField::Compare(const cpuColorSpinorField &a, const cpuColorSpinorField &b, 
				 const int tol) {
  checkField(a, b);

  int ret = 0;
  if (a.precision == QUDA_DOUBLE_PRECISION) 
    if (b.precision == QUDA_DOUBLE_PRECISION)
      ret = compareSpinor(*(a.order_double), *(b.order_double), tol);
    else
      ret = compareSpinor(*(a.order_double), *(b.order_single), tol);
  else 
    if (b.precision == QUDA_DOUBLE_PRECISION)
      ret = compareSpinor(*(a.order_single), *(b.order_double), tol);
    else
      ret =compareSpinor(*(a.order_single), *(b.order_single), tol);

  return ret;
}

template <class Order>
void print_vector(const Order &o, unsigned int x) {

  for (int s=0; s<o.Nspin(); s++) {
    for (int c=0; c<o.Ncolor(); c++) {
      for (int z=0; z<2; z++) {
	std::cout << o(x, s, c, z) << std::endl;
      }
    }
    std::cout << std::endl;
  }

}

// print out the vector at volume point x
void cpuColorSpinorField::PrintVector(unsigned int x) {
  
  switch(precision) {
  case QUDA_DOUBLE_PRECISION:
    print_vector(*order_double, x);
    break;
  case QUDA_SINGLE_PRECISION:
    print_vector(*order_single, x);
    break;
  default:
    errorQuda("Precision %d not implemented", precision); 
  }

}

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
//BEGIN NEW  
  int X5 = this->nDim == 5 ? this->x[4] : 1;
  
  int Vsh[4]={ X2*X3*X4*X5/2,
	       X1*X3*X4*X5/2,
	       X1*X2*X4*X5/2,
	       X1*X2*X3*X5/2};
//END NEW  
  
  int num_faces = 1;
  if(this->nSpin == 1) num_faces = 3; // staggered

  int spinor_size = 2*this->nSpin*this->nColor*this->precision;
  for(int i=0;i < 4; i++){
    fwdGhostFaceBuffer[i] = malloc(num_faces*Vsh[i]*spinor_size);
    backGhostFaceBuffer[i] = malloc(num_faces*Vsh[i]*spinor_size);

    fwdGhostFaceSendBuffer[i] = malloc(num_faces*Vsh[i]*spinor_size);
    backGhostFaceSendBuffer[i] = malloc(num_faces*Vsh[i]*spinor_size);
    
    if(fwdGhostFaceBuffer[i]== NULL || backGhostFaceBuffer[i] == NULL||
       fwdGhostFaceSendBuffer[i]== NULL || backGhostFaceSendBuffer[i]==NULL){
      errorQuda("malloc for ghost buf in cpu spinor failed\n");
    }
  }
  
  initGhostFaceBuffer = 1;
  return;
}

void cpuColorSpinorField::freeGhostBuffer(void)
{
  if(!initGhostFaceBuffer) return;

  for(int i=0;i < 4; i++){
    free(fwdGhostFaceBuffer[i]); fwdGhostFaceBuffer[i] = NULL;
    free(backGhostFaceBuffer[i]); backGhostFaceBuffer[i] = NULL;
    free(fwdGhostFaceSendBuffer[i]); fwdGhostFaceSendBuffer[i] = NULL;
    free(backGhostFaceSendBuffer[i]);  backGhostFaceSendBuffer[i] = NULL;
  } 

  initGhostFaceBuffer = 0;
  
  return;
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
//BEGIN NEW 
  int X5 = this->nDim == 5 ? this->x[4]: 1;
//END NEW    


  for(int i=0;i < this->volume;i++){ 
    
    int X1h = X1/2;
    
    int sid =i;
    int za = sid/X1h;
    int x1h = sid - za*X1h;
    int zb = za/X2;
    int x2 = za - zb*X2;
//BEGIN NEW
    int zc = zb / X3;
    int x3 = zb - zc*X3;
    int x5 = zc / X4; //this->nDim == 5 ? zz / X4 : 0;
    int x4 = zc - x5*X4;
    int x1odd = (x2 + x3 + x4 + x5 + oddBit) & 1;
//END NEW
    int x1 = 2*x1h + x1odd;
    int X = 2*sid + x1odd; 


    int ghost_face_idx ;
    
//NOTE: added extra dimension for DW dslash    

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
