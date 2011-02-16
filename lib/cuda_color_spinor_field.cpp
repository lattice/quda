#include <stdlib.h>
#include <stdio.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <string.h>
#include <iostream>

void* cudaColorSpinorField::buffer = 0;
bool cudaColorSpinorField::bufferInit = false;
size_t cudaColorSpinorField::bufferBytes = 0;

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


void cudaColorSpinorField::packGhost(void *ghost_spinor, void *ghost_norm,
				     const int dim, const QudaDirection dir,
				     cudaStream_t *stream) {

  if (dim != 3) errorQuda("Not supported");

  int num_faces = 3; //3 faces for asqtad
  int FloatN = 2; // always use Float2 for staggered
  int M = nColor * nSpin * 2 / FloatN; // number FloatN buffers we have

  int Vh = this->volume;
  int Vsh = x[0]*x[1]*x[2];
  int sizeOfFloatN = FloatN*precision;
  int len = num_faces*Vsh*sizeOfFloatN; 

  int offset = (dir == QUDA_BACKWARDS) ? 0 : Vh - num_faces*Vsh;

  for (int i=0; i<M; i++) {
    void *dst = (char*)ghost_spinor + i*len;
    void *src = (char*)v + (offset + i*stride) * sizeOfFloatN;
    cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
  }

}

void cudaColorSpinorField::unpackGhost(void* ghost_spinor, void* ghost_norm, 
				       const int dim, const QudaDirection dir, 
				       cudaStream_t* stream) 
{

  if (dim != 3) errorQuda("Not supported");

  int num_faces = 3; //3 faces for asqtad
  int FloatN = 2; // always use Float2 for staggered
  int M = nColor * nSpin * 2 / FloatN; // number FloatN buffers we have

  int Vsh = x[0]*x[1]*x[2];  
  int sizeOfFloatN = FloatN*precision;
  int len = num_faces*Vsh*sizeOfFloatN;
  
  int offset = (dir == QUDA_BACKWARDS) ? 0 : M*len;
  void* dst = ((char*)v) + M*stride*sizeOfFloatN + offset; // into the endzone
  void* src = ghost_spinor;
  cudaMemcpyAsync(dst, src, M*len, cudaMemcpyHostToDevice, *stream);CUERR;
  
  if (precision == QUDA_HALF_PRECISION){
    int normlen = num_faces*Vsh*sizeof(float);
    int offset = (dir == QUDA_BACKWARDS) ? 0 : normlen;
    void* dst = ((char*)norm) + stride*sizeof(float) + offset;
    void* src = ghost_norm;
    cudaMemcpyAsync(dst, src, normlen, cudaMemcpyHostToDevice, *stream); CUERR;
  }  

  return;
}

void
cudaColorSpinorField::packGhostSpinor(void* fwd_ghost_spinor, void* back_ghost_spinor, 
				      void* f_norm, void* b_norm, cudaStream_t* stream) 
{
  int Vh = this->volume;
  int Vsh = x[0]*x[1]*x[2];
  int i;
  
  int sizeOfFloatN = 2*precision;
  int len = 3*Vsh*sizeOfFloatN; //3 faces

  for (i =0; i < 3;i ++){
    void* dst = ((char*)back_ghost_spinor) + i*len; 
    void* src = ((char*)v) + i*stride*sizeOfFloatN;
    cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
  }

  
  for (i =0; i < 3;i ++){
    void* dst = ((char*)fwd_ghost_spinor) + i*len; 
    void* src = ((char*)v) + (Vh - 3*Vsh + i*stride)*sizeOfFloatN;
    cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
  }  
  
  if (precision == QUDA_HALF_PRECISION){
    int normlen = 3*Vsh*sizeof(float);
    void* dst = b_norm;
    void* src = norm;
    cudaMemcpyAsync(dst, src, normlen, cudaMemcpyDeviceToHost, *stream); CUERR;
    
    dst = f_norm;
    src = ((char*)norm) + (Vh-3*Vsh)*sizeof(float);
    cudaMemcpyAsync(dst, src, normlen, cudaMemcpyDeviceToHost, *stream); CUERR;
  }  
  

  return;
}

void
cudaColorSpinorField::unpackGhostSpinor(void* fwd_ghost_spinor, void* back_ghost_spinor, 
					void* f_norm, void* b_norm, cudaStream_t* stream) 
{
  int Vsh = x[0]*x[1]*x[2];
  
  int sizeOfFloatN = 2*precision;
  int len = 3*Vsh*sizeOfFloatN; //3 faces
  
  void* dst = ((char*)v) + 3*stride*sizeOfFloatN;
  void* src =back_ghost_spinor;
  cudaMemcpyAsync(dst, src, 3*len, cudaMemcpyHostToDevice, *stream); CUERR;
  
  dst = ((char*)v) + 3*stride*sizeOfFloatN + 3*len;
  src = fwd_ghost_spinor;
  cudaMemcpyAsync(dst, src, 3*len, cudaMemcpyHostToDevice, *stream);CUERR;
  
  if (precision == QUDA_HALF_PRECISION){
    int normlen = 3*Vsh*sizeof(float);
    void* dst = ((char*)norm) + stride*sizeof(float);
    void* src = b_norm;
    cudaMemcpyAsync(dst, src, normlen, cudaMemcpyHostToDevice, *stream); CUERR;
    
    dst = ((char*)norm) + (stride + 3*Vsh)*sizeof(float);
    src = f_norm;
    cudaMemcpyAsync(dst, src, normlen, cudaMemcpyHostToDevice, *stream); CUERR;
  }  
  


  return;

}
