#include <stdlib.h>
#include <stdio.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <string.h>
#include <iostream>
#include "misc_helpers.h"

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


void cudaColorSpinorField::packGhost(void *ghost_spinor, const int dim, const QudaDirection dir,
				     const QudaParity parity, const int dagger,
				     cudaStream_t *stream) 
{

#ifndef GPU_STAGGERED_DIRAC
  CUERR; // check error state

  int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
  int num_faces = (nSpin == 1) ? 3 : 1; //3 faces for asqtad
  int Vh = this->volume;
  int Vsh = x[0]*x[1]*x[2];
  int Nint = nColor * nSpin * 2; // number of internal degrees of freedom
  if (nSpin == 4) Nint /= 2; // spin projection for Wilson
  int Npad = Nint / Nvec; // number Nvec buffers we have

  if (dim != 3) errorQuda("Not supported");

  int t0_offset=0; // T=0 is the first VS block
  int Nt_minus1_offset = (Vh - num_faces*Vsh); // N_t -1 = Vh-Vsh

  int offset = 0;
  if (nSpin == 1) {
    offset = (dir == QUDA_BACKWARDS) ? t0_offset : Nt_minus1_offset;
  } else if (nSpin == 4) {    
    // !dagger: send lower components backwards, send upper components forwards
    // dagger: send upper components backwards, send lower components forwards
    bool upper = dagger ? true : false; // Fwd is !Back  
    if (dir == QUDA_FORWARDS) upper = !upper;
    int lower_spin_offset=Npad*stride;	    
    if (upper) offset = (dir == QUDA_BACKWARDS ? t0_offset : Nt_minus1_offset);
    else offset = lower_spin_offset + (dir == QUDA_BACKWARDS ? t0_offset : Nt_minus1_offset);
  }
    
  // QUDA Memcpy NPad's worth. 
  //  -- Dest will point to the right beginning PAD. 
  //  -- Each Pad has size Nvec*Vsh Floats. 
  //  --  There is Nvec*Stride Floats from the start of one PAD to the start of the next
  for(int i=0; i < Npad; i++) {
    int len = num_faces*Vsh*Nvec*precision;     
    void *dst = (char*)ghost_spinor + i*len;
    void *src = (char*)v + (offset + i*stride)* Nvec*precision;
    CUDAMEMCPY(dst, src, len, cudaMemcpyDeviceToHost, *stream); CUERR;
  }

  if (precision == QUDA_HALF_PRECISION) {
    int normlen = num_faces*Vsh*sizeof(float);
    int norm_offset = (dir == QUDA_BACKWARDS) ? t0_offset : Nt_minus1_offset*sizeof(float);
    void *dst = (char*)ghost_spinor + num_faces*Nint*Vsh*precision;
    void *src = (char*)norm + norm_offset;
    CUDAMEMCPY(dst, src, normlen, cudaMemcpyDeviceToHost, *stream); CUERR;
  }
#else
  
  int Vsh_x = x[1]*x[2]*x[3]/2;
  int Vsh_y = x[0]*x[2]*x[3];
  int Vsh_z = x[0]*x[1]*x[3];
  int Vsh_t = x[0]*x[1]*x[2];
  int Vsh_xyzt[4]={Vsh_x, Vsh_y, Vsh_z, Vsh_t};
  int num_faces = 3; //3 faces for asqtad
  int FloatN = 2; // always use Float2 for staggered
  int sizeOfFloatN = FloatN*precision; 
  int len = num_faces*Vsh_xyzt[dim]*sizeOfFloatN;
  
  if(this->initGhostFaceBuffer == 0 || precision > facePrecision){
    
    cudaMalloc((void**)&this->fwdGhostFaceBuffer[0], 9*Vsh_x*sizeOfFloatN);
    cudaMalloc((void**)&this->fwdGhostFaceBuffer[1], 9*Vsh_y*sizeOfFloatN);
    cudaMalloc((void**)&this->fwdGhostFaceBuffer[2], 9*Vsh_z*sizeOfFloatN);
    cudaMalloc((void**)&this->fwdGhostFaceBuffer[3], 9*Vsh_t*sizeOfFloatN);
  
    cudaMalloc((void**)&this->backGhostFaceBuffer[0], 9*Vsh_x*sizeOfFloatN);
    cudaMalloc((void**)&this->backGhostFaceBuffer[1], 9*Vsh_y*sizeOfFloatN);
    cudaMalloc((void**)&this->backGhostFaceBuffer[2], 9*Vsh_z*sizeOfFloatN);
    cudaMalloc((void**)&this->backGhostFaceBuffer[3], 9*Vsh_t*sizeOfFloatN);
    CUERR;
    
    this->facePrecision = precision;
    this->initGhostFaceBuffer = 1;
  }

  void* gpu_buf;
  if(dir== QUDA_BACKWARDS){
    gpu_buf = this->backGhostFaceBuffer[dim];
  }else{
    gpu_buf = this->fwdGhostFaceBuffer[dim];
  }
  
  collectGhostSpinor(this->v, this->norm, gpu_buf, dim, dir, parity, this, stream); CUERR;
  cudaMemcpyAsync(ghost_spinor, gpu_buf, 3*len, cudaMemcpyDeviceToHost, *stream); CUERR;

#endif

}


void cudaColorSpinorField::unpackGhost(void* ghost_spinor, const int dim, 
				       const QudaDirection dir, 
				       const int dagger, cudaStream_t* stream) 
{
#if 1
  CUERR;
  int Nvec = (nSpin == 1 || precision == QUDA_DOUBLE_PRECISION) ? 2 : 4;
  int num_faces = (nSpin == 1) ? 3 : 1; //3 faces for asqtad
  int Vsh_xyzt[4]= {
    x[1]*x[2]*x[3]/2,
    x[0]*x[2]*x[3],
    x[0]*x[1]*x[3],
    x[0]*x[1]*x[2]
  };
  int Vsh = Vsh_xyzt[dim];
  int Nint = nColor * nSpin * 2; // number of internal degrees of freedom
  if (nSpin == 4) Nint /= 2; // spin projection for Wilson
  int Npad = Nint / Nvec; // number Nvec buffers we have


  // Wilson only
  // !dagger: receive lower components forwards, receive upper components backwards
  // dagger: receive upper components forwards, receive lower components backwards
  bool upper = dagger? false : true;
  if (dir == QUDA_FORWARDS) upper = !upper;
    
  int len = num_faces*Vsh*Nvec*Npad;

  int offset = length + ghostOffset[dim]*nColor*nSpin*2;
  if (nSpin == 1) offset += (dir == QUDA_BACKWARDS) ? 0 : len;
  else offset += (upper ? 0 : len);    
  void *dst = (char*)v + precision*offset;
  void *src = ghost_spinor;

  CUDAMEMCPY(dst, src, len*precision, cudaMemcpyHostToDevice, *stream); CUERR;
    
  if (precision == QUDA_HALF_PRECISION) {
    // FIXME: Convention difference
    // Staggered: backwards goes in 1st norm zone, forwards in 2nd norm zone
    // Wilson: upper goes in the 1st norm zone, lower in the 2nd norm zone (changes depending on dagger)
    int normlen = num_faces*Vsh*sizeof(float);
    int norm_offset = (nSpin == 1) ? ((dir == QUDA_BACKWARDS) ? 0 : normlen ) : (upper ? 0 : normlen);
    void *dst = (char*)norm + stride*sizeof(float) + norm_offset;
    void *src = (char*)ghost_spinor+num_faces*Nint*Vsh*precision;
    CUDAMEMCPY(dst, src, normlen, cudaMemcpyHostToDevice, *stream);  CUERR;
  }

#else
  //x[0] is already half of X dimension length
  int Vsh_x = x[1]*x[2]*x[3]/2;
  int Vsh_y = x[0]*x[2]*x[3];
  int Vsh_z = x[0]*x[1]*x[3];
  int Vsh_t = x[0]*x[1]*x[2];

  int sizeOfFloatN = 2*precision;

  void* src =ghost_spinor;
  void* dst;
  
  //put X dimension ghost data in place
  int len_x = 3*Vsh_x*sizeOfFloatN; //3 faces  
  if (dim == 0 && dir == QUDA_BACKWARDS){
    void* dst = ((char*)v) + 3*stride*sizeOfFloatN;
    cudaMemcpyAsync(dst, src, 3*len_x, cudaMemcpyHostToDevice, *stream); CUERR;
  }
  if (dim == 0 && dir == QUDA_FORWARDS){
    dst = ((char*)v) + 3*stride*sizeOfFloatN + 3*len_x;
    cudaMemcpyAsync(dst, src, 3*len_x, cudaMemcpyHostToDevice, *stream);CUERR;
  }


  //put Y dimension ghost data in place
  int len_y = 3*Vsh_y*sizeOfFloatN; //3 faces  
  if (dim == 1 && dir == QUDA_BACKWARDS){
    dst = ((char*)v) + 3*stride*sizeOfFloatN + 6*(Vsh_x)*(3*sizeOfFloatN);
    cudaMemcpyAsync(dst, src, 3*len_y, cudaMemcpyHostToDevice, *stream); CUERR;
  }
  if (dim == 1 && dir == QUDA_FORWARDS){
    dst = ((char*)v) + 3*stride*sizeOfFloatN + 6*(Vsh_x)*(3*sizeOfFloatN)+ 3*len_y;
    cudaMemcpyAsync(dst, src, 3*len_y, cudaMemcpyHostToDevice, *stream);CUERR;
  }

  //put Z dimension ghost data in place
  int len_z = 3*Vsh_z*sizeOfFloatN; //3 faces  
  if (dim == 2 && dir == QUDA_BACKWARDS){
    dst = ((char*)v) + 3*stride*sizeOfFloatN + 6*(Vsh_x+Vsh_y)*(3*sizeOfFloatN);
    cudaMemcpyAsync(dst, src, 3*len_z, cudaMemcpyHostToDevice, *stream); CUERR;
  }
  if (dim == 2 && dir == QUDA_FORWARDS){
    dst = ((char*)v) + 3*stride*sizeOfFloatN + 6*(Vsh_x+Vsh_y)*(3*sizeOfFloatN)+ 3*len_z;
    cudaMemcpyAsync(dst, src, 3*len_z, cudaMemcpyHostToDevice, *stream);CUERR;
  }
  
  //put T dimension ghost data in place
  int len_t = 3*Vsh_t*sizeOfFloatN; //3 faces  
  if (dim == 3 && dir == QUDA_BACKWARDS){
    dst = ((char*)v) + 3*stride*sizeOfFloatN + 6*(Vsh_x+Vsh_y+Vsh_z)*(3*sizeOfFloatN);
    cudaMemcpyAsync(dst, src, 3*len_t, cudaMemcpyHostToDevice, *stream); CUERR;
  }
  if (dim == 3 && dir == QUDA_FORWARDS){
    dst = ((char*)v) + 3*stride*sizeOfFloatN + 6*(Vsh_x+Vsh_y+Vsh_z)*(3*sizeOfFloatN)+ 3*len_t;
    cudaMemcpyAsync(dst, src, 3*len_t, cudaMemcpyHostToDevice, *stream);CUERR;
  }


#endif

  


}
