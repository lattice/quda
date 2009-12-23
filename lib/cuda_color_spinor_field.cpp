#include <stdlib.h>
#include <stdio.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

void zeroField(cudaColorSpinorField &a);
void copyField(cudaColorSpinorField &a, const cudaColorSpinorField &b);

bool cudaColorSpinorField::bufferInit = false;

cudaColorSpinorField::cudaColorSpinorField() : 
  ColorSpinorField(), init(false) {

}

cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : 
  ColorSpinorField(param), init(false) {
  create();
  if (param.create == CREATE_ZERO) {
    zeroField(*this);
  } else {
    errorQuda("not implemented");
  }

}

cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : 
  ColorSpinorField(src), init(false) {
  create();
  copyField(*this, src);
}

cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) 
  : ColorSpinorField(src), init(false) {
  create();
  type = CUDA_FIELD;
  if (src.fieldType() == CUDA_FIELD) {
    copyField(*this, dynamic_cast<const cudaColorSpinorField&>(src));
  } else if (src.fieldType() == CPU_FIELD) {
    loadCPUSpinorField(src);
  }
}

cudaColorSpinorField::~cudaColorSpinorField() {
  destroy();
}

void cudaColorSpinorField::create() {
  if (basis != UKQCD_BASIS) {
    errorQuda("Basis not implemented");
  }

  if (subset == FULL_FIELD_SUBSET && subset_order != EVEN_ODD_SUBSET_ORDER) {
    errorQuda("Subset not implemented");
  }
									     

  if (cudaMalloc((void**)&v, bytes) == cudaErrorMemoryAllocation) {
    errorQuda("Error allocating spinor");
  }

  if (prec == QUDA_HALF_PRECISION) {
    if (cudaMalloc((void**)&norm, bytes/12) == cudaErrorMemoryAllocation) {
      errorQuda("Error allocating norm");
    }
  }

  // Check if buffer isn't big enough
  if (bytes > bufferBytes && bufferInit) {
    cudaFree(buffer);
    bufferInit = false;
  }

  if (!bufferInit) {
    bufferBytes = bytes;
    cudaMallocHost(&buffer, bufferBytes);    
    bufferInit = true;
  }

  init = true;
}

void cudaColorSpinorField::destroy() {
  if (init) {
    cudaFree(v);
    if (prec == QUDA_HALF_PRECISION) cudaFree(norm);
  }
  if (bufferInit) cudaFree(buffer);
}

#include <pack_spinor.h>

void cudaColorSpinorField::loadCPUSpinorField(const cpuColorSpinorField &src) {

  if (volume != src.volume) {
    errorQuda("Volumes don't match");
  }

  if (nColor != 3) {
    errorQuda("Nc != 3 not yet supported");
  }

  if (nSpin != 4) {
    errorQuda("Ns != 4 not yet supported");
  }

  if (prec == QUDA_HALF_PRECISION) {
    cudaColorSpinorField tmp(src);
    copyField(*this, tmp);
    return;
  }

  if (prec == QUDA_DOUBLE_PRECISION) {
    if (src.precision() == QUDA_DOUBLE_PRECISION) {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((double*)buffer, (double*)src.v, volume, pad, x, length, 
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((double*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    } else {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((double*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((double*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    }
  } else {
    if (src.precision() == QUDA_DOUBLE_PRECISION) {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	packSpinor<3,4,4>((float*)buffer, (double*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      }
    } else {
      if (order == FLOAT_ORDER) {
	packSpinor<3,4,1>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	packSpinor<3,4,2>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	packSpinor<3,4,4>((float*)buffer, (float*)src.v, volume, pad, x, length,
			  src.fieldSubset(), src.subsetOrder(), basis, src.gammaBasis(), src.fieldOrder());
      }
    }
  }

  cudaMemcpy(v, buffer, bytes, cudaMemcpyHostToDevice);
  return;
}


void cudaColorSpinorField::saveCPUSpinorField(cpuColorSpinorField &dest) const {

  if (volume != dest.volume) {
    errorQuda("Volumes don't match");
  }

  if (nColor != 3) {
    errorQuda("Nc != 3 not yet supported");
  }

  if (nSpin != 4) {
    errorQuda("Ns != 4 not yet supported");
  }

  // Need to either do packing or do extra copy
  if (prec == QUDA_HALF_PRECISION) {
    errorQuda("Not implemented");
  }

  cudaMemcpy(buffer, v, bytes, cudaMemcpyDeviceToHost);

  if (prec == QUDA_DOUBLE_PRECISION) {
    if (dest.precision() == QUDA_DOUBLE_PRECISION) {
      if (order == FLOAT_ORDER) {
	unpackSpinor<3,4,1>((double*)dest.v, (double*)buffer, volume, pad, x, length, 
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((double*)dest.v, (double*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    } else {
      if (order == FLOAT_ORDER) {
	unpackSpinor<3,4,1>((double*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((double*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	errorQuda("double4 not supported");
      }
    }
  } else {
    if (dest.precision() == QUDA_DOUBLE_PRECISION) {
      if (order == FLOAT_ORDER) {
	unpackSpinor<3,4,1>((float*)dest.v, (double*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((float*)dest.v, (double*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	unpackSpinor<3,4,4>((float*)dest.v, (double*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      }
    } else {
      if (order == FLOAT_ORDER) {
	unpackSpinor<3,4,1>((float*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT2_ORDER) {
	unpackSpinor<3,4,2>((float*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      } else if (order == FLOAT4_ORDER) {
	unpackSpinor<3,4,4>((float*)dest.v, (float*)buffer, volume, pad, x, length,
			  dest.fieldSubset(), dest.subsetOrder(), basis, dest.gammaBasis(), dest.fieldOrder());
      }
    }
  }

  return;
}

