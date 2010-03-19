#include <stdlib.h>
#include <stdio.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

#include <string.h>
#include <iostream>

void* cudaColorSpinorField::buffer = 0;
bool cudaColorSpinorField::bufferInit = false;
size_t cudaColorSpinorField::bufferBytes = 0;

cudaColorSpinorField::cudaColorSpinorField() : 
  ColorSpinorField(), v(0), norm(0), alloc(false), init(false) {

}

cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
  ColorSpinorField(param), v(0), norm(0), alloc(false), init(true) {
  create(param.create);
  if  (param.create == QUDA_NULL_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_CREATE) {
    zero();
  } else if (param.create == QUDA_REFERENCE_CREATE) {
    v = param.v;
    norm = param.norm;
  } else if (param.create == QUDA_COPY_CREATE){
    errorQuda("not implemented");
  }
}

cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : 
  ColorSpinorField(src), v(0), norm(0), alloc(false), init(true) {
  create(QUDA_COPY_CREATE);
  copy(src);
}

// creates a copy of src, any differences defined in param
cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, const ColorSpinorParam &param) :
  ColorSpinorField(src), v(0), norm(0), alloc(false), init(true) {  
// can only overide if we are not using a reference or parity special case
  if (param.create != QUDA_REFERENCE_CREATE || 
      (param.create == QUDA_REFERENCE_CREATE && 
       src.fieldSubset() == QUDA_FULL_FIELD_SUBSET && 
       param.fieldSubset == QUDA_PARITY_FIELD_SUBSET && 
       src.fieldType() == QUDA_CUDA_FIELD) ) {
    reset(param);
  } else {
    errorQuda("Undefined behaviour"); // else silent bug possible?
  }

  type = QUDA_CUDA_FIELD;
  create(param.create);

  if (param.create == QUDA_NULL_CREATE) {
    // do nothing
  } else if (param.create == QUDA_ZERO_CREATE) {
    zero();
  } else if (param.create == QUDA_COPY_CREATE) {
    if (src.fieldType() == QUDA_CUDA_FIELD) {
      copy(dynamic_cast<const cudaColorSpinorField&>(src));    
    } else if (src.fieldType() == QUDA_CPU_FIELD) {
      loadCPUSpinorField(dynamic_cast<const cpuColorSpinorField&>(src));
    } else {
      errorQuda("FieldType %d not supported", src.fieldType());
    }
  } else if (param.create == QUDA_REFERENCE_CREATE) {
    if (src.fieldType() == QUDA_CUDA_FIELD) {
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
  type = QUDA_CUDA_FIELD;
  create(QUDA_COPY_CREATE);
  if (src.fieldType() == QUDA_CUDA_FIELD) {
    copy(dynamic_cast<const cudaColorSpinorField&>(src));
  } else if (src.fieldType() == QUDA_CPU_FIELD) {
    loadCPUSpinorField(src);
  } else {
    errorQuda("FieldType not supported");
  }
}

cudaColorSpinorField& cudaColorSpinorField::operator=(const cudaColorSpinorField &src) {
  if (&src != this) {
    destroy();
    // keep current attributes unless unset
    if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
    type = QUDA_CUDA_FIELD;
    create(QUDA_COPY_CREATE);
    copy(src);
  }
  return *this;
}

cudaColorSpinorField& cudaColorSpinorField::operator=(const cpuColorSpinorField &src) {
  destroy();
  // keep current attributes unless unset
  if (!ColorSpinorField::init) ColorSpinorField::operator=(src);
  type = QUDA_CUDA_FIELD;
  create(QUDA_COPY_CREATE);
  loadCPUSpinorField(src);
  return *this;
}

cudaColorSpinorField::~cudaColorSpinorField() {
  destroy();
}

void cudaColorSpinorField::create(const FieldCreate create) {

  if (subset == QUDA_FULL_FIELD_SUBSET && subset_order != QUDA_EVEN_ODD_SUBSET_ORDER) {
    errorQuda("Subset not implemented");
  }
									     
  if (create != QUDA_REFERENCE_CREATE) {
    if (cudaMalloc((void**)&v, bytes) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating spinor");
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

  if (subset == QUDA_FULL_FIELD_SUBSET) {
    // create the associated even and odd subsets
    ColorSpinorParam param;
    param.fieldSubset = QUDA_PARITY_FIELD_SUBSET;
    param.nDim = nDim;
    memcpy(param.x, x, nDim*sizeof(int));
    param.x[0] /= 2; // set single parity dimensions
    param.create = QUDA_REFERENCE_CREATE;
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
  if (bufferInit) cudaFree(buffer);
}

void cudaColorSpinorField::destroy() {
  if (alloc) {
    cudaFree(v);
    if (precision == QUDA_HALF_PRECISION) cudaFree(norm);
    if (subset == QUDA_FULL_FIELD_SUBSET) {
      delete even;
      delete odd;
    }
    alloc = false;
  }
}


cudaColorSpinorField& cudaColorSpinorField::Even() const { 
  if (subset == QUDA_FULL_FIELD_SUBSET) {
    return *(dynamic_cast<cudaColorSpinorField*>(even)); 
  } else {
    errorQuda("Cannot return even subset of %d subset", subset);
  }
}

cudaColorSpinorField& cudaColorSpinorField::Odd() const {
  if (subset == QUDA_FULL_FIELD_SUBSET) {
    return *(dynamic_cast<cudaColorSpinorField*>(odd)); 
  } else {
    errorQuda("Cannot return odd subset of %d subset", subset);
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

  if (volume != src.volume) {
    errorQuda("Volumes %d %d don't match", volume, src.volume);
  }

  if (subsetOrder() != src.subsetOrder()) {
    errorQuda("Subset orders don't match");
  }

  if (nColor != 3) {
    errorQuda("Nc != 3 not yet supported");
  }

  if (nSpin != 4) {
    errorQuda("Ns != 4 not yet supported");
  }

  if (subset != src.subset) {
    errorQuda("Subset types do not match %d %d", subset, src.subset);
  }

  if (precision == QUDA_HALF_PRECISION) {
    ColorSpinorParam param;
    fill(param); // acquire all attributes of this
    param.precision = QUDA_SINGLE_PRECISION; // change precision
    param.create = QUDA_COPY_CREATE;
    cudaColorSpinorField tmp(src, param);
    copy(tmp);
    return;
  }

  // (temporary?) bug fix for padding
  memset(buffer, 0, bufferBytes);

  if (precision == QUDA_DOUBLE_PRECISION) {
    if (src.precision == QUDA_DOUBLE_PRECISION) {
      if (order == QUDA_FLOAT_ORDER) {
	packSpinor<3,4,1>((double*)buffer, (double*)src.v, volume, pad, x, length, 
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	packSpinor<3,4,2>((double*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    } else {
      if (order == QUDA_FLOAT_ORDER) {
	packSpinor<3,4,1>((double*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	packSpinor<3,4,2>((double*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    }
  } else {
    if (src.precision == QUDA_DOUBLE_PRECISION) {
      if (order == QUDA_FLOAT_ORDER) {
	packSpinor<3,4,1>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	packSpinor<3,4,2>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	packSpinor<3,4,4>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      }
    } else {
      if (order == QUDA_FLOAT_ORDER) {
	packSpinor<3,4,1>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	packSpinor<3,4,2>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	packSpinor<3,4,4>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      }
    }
  }

  /*  for (int i=0; i<length; i++) {
    std::cout << i << " " << ((float*)src.v)[i] << " " << ((float*)buffer)[i] << std::endl;
    }*/

  cudaMemcpy(v, buffer, bytes, cudaMemcpyHostToDevice);
  return;
}


void cudaColorSpinorField::saveCPUSpinorField(cpuColorSpinorField &dest) const {

  if (volume != dest.volume) {
    errorQuda("Volumes %d %d don't match", volume, dest.volume);
  }

  if (subsetOrder() != dest.subsetOrder()) {
    errorQuda("Subset orders don't match");
  }

  if (nColor != 3) {
    errorQuda("Nc != 3 not yet supported");
  }

  if (nSpin != 4) {
    errorQuda("Ns != 4 not yet supported");
  }

  if (subset != dest.subset) {
    errorQuda("Subset types do not match %d %d", subset, dest.subset);
  }

  if (precision == QUDA_HALF_PRECISION) {
    ColorSpinorParam param;
    param.create = QUDA_COPY_CREATE;
    param.precision = QUDA_SINGLE_PRECISION;
    cudaColorSpinorField tmp(*this, param);
    tmp.saveCPUSpinorField(dest);
    return;
  }

  // (temporary?) bug fix for padding
  memset(buffer, 0, bufferBytes);

  cudaMemcpy(buffer, v, bytes, cudaMemcpyDeviceToHost);

  if (precision == QUDA_DOUBLE_PRECISION) {
    if (dest.precision == QUDA_DOUBLE_PRECISION) {
      if (order == QUDA_FLOAT_ORDER) {
	unpackSpinor<3,4,1>((double*)dest.v, (double*)buffer, volume, pad, x, length, 
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((double*)dest.v, (double*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    } else {
      if (order == QUDA_FLOAT_ORDER) {
	unpackSpinor<3,4,1>((float*)dest.v, (double*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((float*)dest.v, (double*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    }
  } else {
    if (dest.precision == QUDA_DOUBLE_PRECISION) {
      if (order == QUDA_FLOAT_ORDER) {
	unpackSpinor<3,4,1>((double*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((double*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	unpackSpinor<3,4,4>((double*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      }
    } else {
      if (order == QUDA_FLOAT_ORDER) {
	unpackSpinor<3,4,1>((float*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((float*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      } else if (order == QUDA_FLOAT4_ORDER) {
	unpackSpinor<3,4,4>((float*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), dest.gammaBasis(), basis, dest.fieldOrder());
      }
    }
  }

  return;
}

