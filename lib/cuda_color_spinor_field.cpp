#include <stdlib.h>
#include <stdio.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <string.h>
#include <iostream>
#include "misc_helpers.h"
#include <face_quda.h>
#include <dslash_quda.h>

// Easy to switch between overlapping communication or not
#ifdef OVERLAP_COMMS
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpyAsync(dst, src, size, type, stream)
#else
#define CUDAMEMCPY(dst, src, size, type, stream) cudaMemcpy(dst, src, size, type)
#endif

void* cudaColorSpinorField::buffer = 0;
bool cudaColorSpinorField::bufferInit = false;
size_t cudaColorSpinorField::bufferBytes = 0;

int cudaColorSpinorField::initGhostFaceBuffer = 0;
void* cudaColorSpinorField::fwdGhostFaceBuffer[QUDA_MAX_DIM]; //gpu memory
void* cudaColorSpinorField::backGhostFaceBuffer[QUDA_MAX_DIM]; //gpu memory
QudaPrecision cudaColorSpinorField::facePrecision; 

extern bool kernelPackT;

/*cudaColorSpinorField::cudaColorSpinorField() : 
  ColorSpinorField(), v(0), norm(0), alloc(false), init(false) {

  }*/

cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
  ColorSpinorField(param), v(0), norm(0), alloc(false), init(true) {
  create(param.create);
  if  (param.create == QUDA_NULL_FIELD_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
    zero();
  } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
    v = param.v;
    norm = param.norm;
  } else if (param.create == QUDA_COPY_FIELD_CREATE){
    errorQuda("not implemented");
  }
}

cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : 
  ColorSpinorField(src), v(0), norm(0), alloc(false), init(true) {
  create(QUDA_COPY_FIELD_CREATE);
  copy(src);
}

// creates a copy of src, any differences defined in param
cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, const ColorSpinorParam &param) :
  ColorSpinorField(src), v(0), norm(0), alloc(false), init(true) {  
// can only overide if we are not using a reference or parity special case

  if (param.create != QUDA_REFERENCE_FIELD_CREATE || 
      (param.create == QUDA_REFERENCE_FIELD_CREATE && 
       src.SiteSubset() == QUDA_FULL_SITE_SUBSET && 
       param.siteSubset == QUDA_PARITY_SITE_SUBSET && 
       src.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) ) {
    reset(param);
  } else {
    errorQuda("Undefined behaviour"); // else silent bug possible?
  }

  fieldLocation = QUDA_CUDA_FIELD_LOCATION;
  create(param.create);

  if (param.create == QUDA_NULL_FIELD_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
    zero();
  } else if (param.create == QUDA_COPY_FIELD_CREATE) {
    if (src.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) {
      copy(dynamic_cast<const cudaColorSpinorField&>(src));    
    } else if (src.FieldLocation() == QUDA_CPU_FIELD_LOCATION) {
      loadCPUSpinorField(dynamic_cast<const cpuColorSpinorField&>(src));
    } else {
      errorQuda("FieldLocation %d not supported", src.FieldLocation());
    }
  } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
    if (src.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) {
      v = (dynamic_cast<const cudaColorSpinorField&>(src)).v;
      norm = (dynamic_cast<const cudaColorSpinorField&>(src)).norm;
    } else {
      errorQuda("Cannot reference a non cuda field");
    }
  } else {
    errorQuda("CreateType %d not implemented", param.create);
  }

}

cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) 
  : ColorSpinorField(src), alloc(false), init(true) {
  fieldLocation = QUDA_CUDA_FIELD_LOCATION;
  create(QUDA_COPY_FIELD_CREATE);
  if (src.FieldLocation() == QUDA_CUDA_FIELD_LOCATION) {
    copy(dynamic_cast<const cudaColorSpinorField&>(src));
  } else if (src.FieldLocation() == QUDA_CPU_FIELD_LOCATION) {
    loadCPUSpinorField(src);
  } else {
    errorQuda("FieldLocation not supported");
  }
}

cudaColorSpinorField& cudaColorSpinorField::operator=(const cudaColorSpinorField &src) {
  if (&src != this) {
    destroy();
    // keep current attributes unless unset
    if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
    fieldLocation = QUDA_CUDA_FIELD_LOCATION;
    create(QUDA_COPY_FIELD_CREATE);
    copy(src);
  }
  return *this;
}

cudaColorSpinorField& cudaColorSpinorField::operator=(const cpuColorSpinorField &src) {
  destroy();
  // keep current attributes unless unset
  if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
  fieldLocation = QUDA_CUDA_FIELD_LOCATION;
  create(QUDA_COPY_FIELD_CREATE);
  loadCPUSpinorField(src);
  return *this;
}

cudaColorSpinorField::~cudaColorSpinorField() {
  destroy();
}


void cudaColorSpinorField::create(const QudaFieldCreate create) {

  if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER) {
    errorQuda("Subset not implemented");
  }

  //FIXME: This addition is temporary to ensure we have the correct
  //field order for a given precision
  if (precision == QUDA_DOUBLE_PRECISION) fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
  else fieldOrder = (nSpin == 4) ? QUDA_FLOAT4_FIELD_ORDER : QUDA_FLOAT2_FIELD_ORDER;

  if (create != QUDA_REFERENCE_FIELD_CREATE) {
    // Overallocate to hold tface bytes extra
    if (cudaMalloc((void**)&v, bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating spinor: bytes=%lu", (unsigned long)bytes);
    }

    if (precision == QUDA_HALF_PRECISION) {
      if (cudaMalloc((void**)&norm, norm_bytes) == cudaErrorMemoryAllocation) {
	errorQuda("Error allocating norm");
      }
    }
    alloc = true;
  }

  // Check if buffer isn't big enough
  if (bytes > bufferBytes && bufferInit) {
    cudaFreeHost(buffer);
    bufferInit = false;
  }

  if (!bufferInit) {
    bufferBytes = bytes;
    cudaMallocHost(&buffer, bufferBytes);    
    bufferInit = true;
  }

  if (siteSubset == QUDA_FULL_SITE_SUBSET) {
    // create the associated even and odd subsets
    ColorSpinorParam param;
    param.siteSubset = QUDA_PARITY_SITE_SUBSET;
    param.nDim = nDim;
    memcpy(param.x, x, nDim*sizeof(int));
    param.x[0] /= 2; // set single parity dimensions
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.v = v;
    param.norm = norm;
    even = new cudaColorSpinorField(*this, param);
    odd = new cudaColorSpinorField(*this, param);
    // need this hackery for the moment (need to locate the odd pointer half way into the full field)
    (dynamic_cast<cudaColorSpinorField*>(odd))->v = (void*)((unsigned long)v + bytes/2);
    if (precision == QUDA_HALF_PRECISION) 
      (dynamic_cast<cudaColorSpinorField*>(odd))->norm = (void*)((unsigned long)norm + norm_bytes/2);
  }

  if (siteSubset != QUDA_FULL_SITE_SUBSET) {
    zeroPad();
  } else {
    (dynamic_cast<cudaColorSpinorField*>(even))->zeroPad();
    (dynamic_cast<cudaColorSpinorField*>(odd))->zeroPad();
  }
  
}
void cudaColorSpinorField::freeBuffer() {
  if (bufferInit) cudaFreeHost(buffer);
}

void cudaColorSpinorField::destroy() {
  if (alloc) {
    cudaFree(v);
    if (precision == QUDA_HALF_PRECISION) cudaFree(norm);
    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      delete even;
      delete odd;
    }
    alloc = false;
  }
}


cudaColorSpinorField& cudaColorSpinorField::Even() const { 
  if (siteSubset == QUDA_FULL_SITE_SUBSET) {
    return *(dynamic_cast<cudaColorSpinorField*>(even)); 
  }

  errorQuda("Cannot return even subset of %d subset", siteSubset);
  exit(-1);
}

cudaColorSpinorField& cudaColorSpinorField::Odd() const {
  if (siteSubset == QUDA_FULL_SITE_SUBSET) {
    return *(dynamic_cast<cudaColorSpinorField*>(odd)); 
  }

  errorQuda("Cannot return odd subset of %d subset", siteSubset);
  exit(-1);
}

// cuda's floating point format, IEEE-754, represents the floating point
// zero as 4 zero bytes
void cudaColorSpinorField::zero() {
  cudaMemset(v, 0, bytes);
  if (precision == QUDA_HALF_PRECISION) cudaMemset(norm, 0, norm_bytes);
}


void cudaColorSpinorField::zeroPad() {
  size_t pad_bytes = (stride - volume) * precision * fieldOrder;
  int Npad = nColor * nSpin * 2 / fieldOrder;
  for (int i=0; i<Npad; i++) {
    if (pad_bytes) cudaMemset((char*)v + (volume + i*stride)*fieldOrder*precision, 0, pad_bytes);      
  }
}

void cudaColorSpinorField::copy(const cudaColorSpinorField &src) {
  checkField(*this, src);
  copyCuda(*this, src);
}

#include <pack_spinor.h>

void cudaColorSpinorField::loadCPUSpinorField(const cpuColorSpinorField &src) {

  
  if (nDim != src.Ndim()) {
    errorQuda("Number of dimensions %d %d don't match", nDim, src.Ndim());
  }

  if (volume != src.volume) {
    errorQuda("Volumes %d %d don't match", volume, src.volume);
  }

  if (SiteOrder() != src.SiteOrder()) {
    errorQuda("Subset orders don't match");
  }

  if (nColor != 3) {
    errorQuda("Nc != 3 not yet supported");
  }

  if (siteSubset != src.siteSubset) {
    errorQuda("Subset types do not match %d %d", siteSubset, src.siteSubset);
  }
  if (precision == QUDA_HALF_PRECISION) {
    ColorSpinorParam param(*this); // acquire all attributes of this
    param.precision = QUDA_SINGLE_PRECISION; // change precision
    param.create = QUDA_COPY_FIELD_CREATE;
    cudaColorSpinorField tmp(src, param);
    copy(tmp);
    return;
  }

  // no native support for this yet - copy to a native supported order
  if (src.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
    ColorSpinorParam param(src); // acquire all attributes of this
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.create = QUDA_NULL_FIELD_CREATE;
    cpuColorSpinorField tmp(param);
    tmp.copy(src);
    loadCPUSpinorField(tmp);
    return;
  }

  // (temporary?) bug fix for padding
  memset(buffer, 0, bufferBytes);
  
#define LOAD_SPINOR_CPU_TO_GPU(myNs)					\
  if (precision == QUDA_DOUBLE_PRECISION) {				\
      if (src.precision == QUDA_DOUBLE_PRECISION) {			\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      packSpinor<3,myNs,1>((double*)buffer, (double*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((double*)buffer, (double*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      packSpinor<3,myNs,1>((double*)buffer, (float*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((double*)buffer, (float*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      }									\
  } else {								\
      if (src.precision == QUDA_DOUBLE_PRECISION) {			\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      packSpinor<3,myNs,1>((float*)buffer, (double*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((float*)buffer, (double*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      packSpinor<3,myNs,4>((float*)buffer, (double*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      packSpinor<3,myNs,1>((float*)buffer, (float*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((float*)buffer, (float*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      packSpinor<3,myNs,4>((float*)buffer, (float*)src.v, volume, pad, x, total_length, src.total_length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  }								\
      }									\
}

  switch(nSpin){
  case 1:
      LOAD_SPINOR_CPU_TO_GPU(1);
      break;
  case 4:
      LOAD_SPINOR_CPU_TO_GPU(4);
      break;
  default:
      errorQuda("invalid number of spinors in function %s\n", __FUNCTION__);

  }
  
#undef LOAD_SPINOR_CPU_TO_GPU

  /*  for (int i=0; i<length; i++) {
    std::cout << i << " " << ((float*)src.v)[i] << " " << ((float*)buffer)[i] << std::endl;
    }*/

  cudaMemcpy(v, buffer, bytes, cudaMemcpyHostToDevice);
  return;
}


void cudaColorSpinorField::saveCPUSpinorField(cpuColorSpinorField &dest) const {

  if (nDim != dest.Ndim()) {
    errorQuda("Number of dimensions %d %d don't match", nDim, dest.Ndim());
  }

  if (volume != dest.volume) {
    errorQuda("Volumes %d %d don't match", volume, dest.volume);
  }

  if (SiteOrder() != dest.SiteOrder()) {
    errorQuda("Subset orders don't match");
  }

  if (nColor != 3) {
    errorQuda("Nc != 3 not yet supported");
  }


  if (siteSubset != dest.siteSubset) {
    errorQuda("Subset types do not match %d %d", siteSubset, dest.siteSubset);
  }

  if (precision == QUDA_HALF_PRECISION) {
    ColorSpinorParam param(*this); // acquire all attributes of this
    param.precision = QUDA_SINGLE_PRECISION; // change precision
    param.create = QUDA_COPY_FIELD_CREATE; 
    cudaColorSpinorField tmp(*this, param);
    tmp.saveCPUSpinorField(dest);
    return;
  }

  // no native support for this yet - copy to a native supported order
  if (dest.FieldOrder() == QUDA_QOP_DOMAIN_WALL_FIELD_ORDER) {
    ColorSpinorParam param(dest); // acquire all attributes of this
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.create = QUDA_NULL_FIELD_CREATE;
    cpuColorSpinorField tmp(param);
    saveCPUSpinorField(tmp);
    dest.copy(tmp);
    return;
  }

  // (temporary?) bug fix for padding
  memset(buffer, 0, bufferBytes);

  cudaMemcpy(buffer, v, bytes, cudaMemcpyDeviceToHost);


#define SAVE_SPINOR_GPU_TO_CPU(myNs)				\
  if (precision == QUDA_DOUBLE_PRECISION) {				\
      if (dest.precision == QUDA_DOUBLE_PRECISION) {			\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      unpackSpinor<3,myNs,1>((double*)dest.v, (double*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((double*)dest.v, (double*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      unpackSpinor<3,myNs,1>((float*)dest.v, (double*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((float*)dest.v, (double*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      }									\
  } else {								\
      if (dest.precision == QUDA_DOUBLE_PRECISION) {			\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      unpackSpinor<3,myNs,1>((double*)dest.v, (float*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((double*)dest.v, (float*)buffer, volume, pad, x, dest.total_length, total_length,	\
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,4>((double*)dest.v, (float*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      unpackSpinor<3,myNs,1>((float*)dest.v, (float*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((float*)dest.v, (float*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,4>((float*)dest.v, (float*)buffer, volume, pad, x, dest.total_length, total_length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  }								\
      }									\
  }

  switch(nSpin){
  case 1:
      SAVE_SPINOR_GPU_TO_CPU(1);
      break;
  case 4:
      SAVE_SPINOR_GPU_TO_CPU(4);
      break;
  default:
      errorQuda("invalid number of spinors in function %s\n", __FUNCTION__);      
  }
#undef SAVE_SPINOR_GPU_TO_CPU
  return;
}

void cudaColorSpinorField::allocateGhostBuffer(void) {
  int nFace = (nSpin == 1) ? 3 : 1; //3 faces for asqtad
  int Nint = nColor * nSpin * 2; // number of internal degrees of freedom
  if (nSpin == 4) Nint /= 2; // spin projection for Wilson

  if(this->initGhostFaceBuffer == 0 || precision > facePrecision){    
    for (int i=0; i<4; i++) {
      if(!commDimPartitioned(i)){
	continue;
      }
      size_t faceBytes = nFace*ghostFace[i]*Nint*precision;
      // add extra space for the norms for half precision
      if (precision == QUDA_HALF_PRECISION) faceBytes += nFace*ghostFace[i]*sizeof(float);
      
      if (this->initGhostFaceBuffer) { // only free-ed if precision is higher than previous allocation
	//cudaFree(this->fwdGhostFaceBuffer[i]); 
	cudaFree(this->backGhostFaceBuffer[i]); this->backGhostFaceBuffer[i] = NULL;
	this->fwdGhostFaceBuffer[i] = NULL;
      }
      //cudaMalloc((void**)&this->fwdGhostFaceBuffer[i], faceBytes);
      cudaMalloc((void**)&this->backGhostFaceBuffer[i], 2*faceBytes);
      fwdGhostFaceBuffer[i] = (void*)(((char*)backGhostFaceBuffer[i]) + faceBytes);
    }   
    CUERR;
    
    this->facePrecision = precision;
    this->initGhostFaceBuffer = 1;
  }

  for (int i=0; i<4; i++) {
    if(!commDimPartitioned(i)){
      continue;
    }
    size_t faceBytes = nFace*ghostFace[i]*Nint*precision;
    // add extra space for the norms for half precision
    if (precision == QUDA_HALF_PRECISION) faceBytes += nFace*ghostFace[i]*sizeof(float);
    fwdGhostFaceBuffer[i] = (void*)(((char*)backGhostFaceBuffer[i]) + faceBytes);
  }


}

void cudaColorSpinorField::freeGhostBuffer(void) {
  if (!initGhostFaceBuffer) return;
  
  for(int i=0;i < 4; i++){
    if(!commDimPartitioned(i)){
      continue;
    }
    //cudaFree(fwdGhostFaceBuffer[i]); 
    cudaFree(backGhostFaceBuffer[i]); backGhostFaceBuffer[i] = NULL;
    fwdGhostFaceBuffer[i] = NULL;
  } 

  initGhostFaceBuffer = 0;  
}

// pack the ghost zone into a contiguous buffer for communications
void cudaColorSpinorField::packGhost(const int dim, const QudaParity parity, const int dagger, cudaStream_t *stream) 
{
#ifdef MULTI_GPU
  if (dim !=3 || kernelPackT) { // use kernels to pack into contiguous buffers then a single cudaMemcpy
    void* gpu_buf = this->backGhostFaceBuffer[dim];
    packFace(gpu_buf, *this, dim, dagger, parity, *stream); 
  }
#else
  errorQuda("packGhost not built on single-GPU build");
#endif

  CUERR;
}
 
// send the ghost zone to the host
void cudaColorSpinorField::sendGhost(void *ghost_spinor, const int dim, const QudaDirection dir,
				     const int dagger, cudaStream_t *stream) {

#ifdef MULTI_GPU
  int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
  int nFace = (nSpin == 1) ? 3 : 1; //3 faces for asqtad
  int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom

  if (dim !=3 || kernelPackT) { // use kernels to pack into contiguous buffers then a single cudaMemcpy

    size_t bytes = nFace*Nint*ghostFace[dim]*precision;
    if (precision == QUDA_HALF_PRECISION) bytes += nFace*ghostFace[dim]*sizeof(float);
    void* gpu_buf = 
      (dir == QUDA_BACKWARDS) ? this->backGhostFaceBuffer[dim] : this->fwdGhostFaceBuffer[dim];

    CUDAMEMCPY(ghost_spinor, gpu_buf, bytes, cudaMemcpyDeviceToHost, *stream); 
  } else { // do multiple cudaMemcpys 

    int Npad = Nint / Nvec; // number Nvec buffers we have
    int Nt_minus1_offset = (volume - nFace*ghostFace[3]); // N_t -1 = Vh-Vsh
    
    int offset = 0;
    if (nSpin == 1) {
      offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset;
    } else if (nSpin == 4) {    
      // !dagger: send lower components backwards, send upper components forwards
      // dagger: send upper components backwards, send lower components forwards
      bool upper = dagger ? true : false; // Fwd is !Back  
      if (dir == QUDA_FORWARDS) upper = !upper;
      int lower_spin_offset = Npad*stride;	    
      if (upper) offset = (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
      else offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? 0 : Nt_minus1_offset);
    }
    
    // QUDA Memcpy NPad's worth. 
    //  -- Dest will point to the right beginning PAD. 
    //  -- Each Pad has size Nvec*Vsh Floats. 
    //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next

    void *dst = (char*)ghost_spinor;
    void *src = (char*)v + offset*Nvec*precision;
    size_t len = nFace*ghostFace[3]*Nvec*precision;     
    size_t spitch = stride*Nvec*precision;
    cudaMemcpy2DAsync(dst, len, src, spitch, len, Npad, cudaMemcpyDeviceToHost, *stream);

    /*for(int i=0; i < Npad; i++) {
      int len = nFace*ghostFace[3]*Nvec*precision;     
      void *dst = (char*)ghost_spinor + i*len;
      void *src = (char*)v + (offset + i*stride)* Nvec*precision;
      CUDAMEMCPY(dst, src, len, cudaMemcpyDeviceToHost, *stream); 
    }*/
    
    if (precision == QUDA_HALF_PRECISION) {
      int norm_offset = (dir == QUDA_BACKWARDS) ? 0 : Nt_minus1_offset*sizeof(float);
      void *dst = (char*)ghost_spinor + nFace*Nint*ghostFace[3]*precision;
      void *src = (char*)norm + norm_offset;
      CUDAMEMCPY(dst, src, nFace*ghostFace[3]*sizeof(float), cudaMemcpyDeviceToHost, *stream); 
    }
  }
#else
  errorQuda("sendGhost not built on single-GPU build");
#endif

  CUERR;
}

void cudaColorSpinorField::unpackGhost(void* ghost_spinor, const int dim, 
				       const QudaDirection dir, 
				       const int dagger, cudaStream_t* stream) 
{
  int nFace = (nSpin == 1) ? 3 : 1; //3 faces for asqtad
  int Nint = (nColor * nSpin * 2) / (nSpin == 4 ? 2 : 1);  // (spin proj.) degrees of freedom

  int len = nFace*ghostFace[dim]*Nint;
  int offset = length + ghostOffset[dim]*nColor*nSpin*2;
  offset += (dir == QUDA_BACKWARDS) ? 0 : len;

  void *dst = (char*)v + precision*offset;
  void *src = ghost_spinor;

  CUDAMEMCPY(dst, src, len*precision, cudaMemcpyHostToDevice, *stream);
    
  if (precision == QUDA_HALF_PRECISION) {
    int normlen = nFace*ghostFace[dim];
    int norm_offset = stride + ghostNormOffset[dim];
    norm_offset += (dir == QUDA_BACKWARDS) ? 0 : normlen;

    void *dst = (char*)norm + norm_offset*sizeof(float);
    void *src = (char*)ghost_spinor+nFace*Nint*ghostFace[dim]*precision; // norm region of host ghost zone
    CUDAMEMCPY(dst, src, normlen*sizeof(float), cudaMemcpyHostToDevice, *stream);
  }

  CUERR;
}

std::ostream& operator<<(std::ostream &out, const cudaColorSpinorField &a) {
  out << (const ColorSpinorField)a;
  out << "v = " << a.v << std::endl;
  out << "norm = " << a.norm << std::endl;
  out << "alloc = " << a.alloc << std::endl;
  out << "init = " << a.init << std::endl;
  return out;
}
