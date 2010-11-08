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
    if (cudaMalloc((void**)&v, bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating spinor: bytes=%d", (int)bytes);
    }
    
    if (precision == QUDA_HALF_PRECISION) {
      if (cudaMalloc((void**)&norm, bytes/(nColor*nSpin)) == cudaErrorMemoryAllocation) {
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
      (dynamic_cast<cudaColorSpinorField*>(odd))->norm = (void*)((unsigned long)norm + bytes/(2*nColor*nSpin));
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
  } else {
    errorQuda("Cannot return even subset of %d subset", siteSubset);
  }
}

cudaColorSpinorField& cudaColorSpinorField::Odd() const {
  if (siteSubset == QUDA_FULL_SITE_SUBSET) {
    return *(dynamic_cast<cudaColorSpinorField*>(odd)); 
  } else {
    errorQuda("Cannot return odd subset of %d subset", siteSubset);
  }
}

// cuda's floating point format, IEEE-754, represents the floating point
// zero as 4 zero bytes
void cudaColorSpinorField::zero() {
  cudaMemset(v, 0, bytes);
  if (precision == QUDA_HALF_PRECISION) cudaMemset(norm, 0, bytes/(nColor*nSpin));
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
    ColorSpinorParam param;
    fill(param); // acquire all attributes of this
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
	      packSpinor<3,myNs,1>((double*)buffer, (double*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((double*)buffer, (double*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      packSpinor<3,myNs,1>((double*)buffer, (float*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((double*)buffer, (float*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      }									\
  } else {								\
      if (src.precision == QUDA_DOUBLE_PRECISION) {			\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      packSpinor<3,myNs,1>((float*)buffer, (double*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((float*)buffer, (double*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      packSpinor<3,myNs,4>((float*)buffer, (double*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      packSpinor<3,myNs,1>((float*)buffer, (float*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      packSpinor<3,myNs,2>((float*)buffer, (float*)src.v, volume, pad, x, length, src.length, \
				src.SiteSubset(), src.SiteOrder(), gammaBasis, src.GammaBasis(), src.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      packSpinor<3,myNs,4>((float*)buffer, (float*)src.v, volume, pad, x, length, src.length, \
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
    ColorSpinorParam param;
    param.create = QUDA_COPY_FIELD_CREATE;
    param.precision = QUDA_SINGLE_PRECISION;
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
	      unpackSpinor<3,myNs,1>((double*)dest.v, (double*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((double*)dest.v, (double*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      unpackSpinor<3,myNs,1>((float*)dest.v, (double*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((float*)dest.v, (double*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      errorQuda("double4 not supported");			\
	  }								\
      }									\
  } else {								\
      if (dest.precision == QUDA_DOUBLE_PRECISION) {			\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      unpackSpinor<3,myNs,1>((double*)dest.v, (float*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((double*)dest.v, (float*)buffer, volume, pad, x, dest.length, length,	\
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,4>((double*)dest.v, (float*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  }								\
      } else {								\
	  if (fieldOrder == QUDA_FLOAT_FIELD_ORDER) {				\
	      unpackSpinor<3,myNs,1>((float*)dest.v, (float*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder()); \
	  } else if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,2>((float*)dest.v, (float*)buffer, volume, pad, x, dest.length, length, \
				  dest.SiteSubset(), dest.SiteOrder(), dest.GammaBasis(), gammaBasis, dest.FieldOrder());	\
	  } else if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) {			\
	      unpackSpinor<3,myNs,4>((float*)dest.v, (float*)buffer, volume, pad, x, dest.length, length, \
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

