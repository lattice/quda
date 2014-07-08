#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <typeinfo>

#include <quda_internal.h>
#include <clover_field.h>
#include <gauge_field.h>

namespace quda {

  CloverField::CloverField(const CloverFieldParam &param) :
    LatticeField(param), bytes(0), norm_bytes(0), nColor(3), nSpin(4), 
    clover(0), norm(0), cloverInv(0), invNorm(0), order(param.order), create(param.create),
    trlog(static_cast<double*>(pinned_malloc(2*sizeof(double))))
  {
    if (nDim != 4) errorQuda("Number of dimensions must be 4, not %d", nDim);

    if (order == QUDA_QDPJIT_CLOVER_ORDER && create != QUDA_REFERENCE_FIELD_CREATE)
      errorQuda("QDPJIT ordered clover fields only supported for reference fields");

    real_length = 2*volumeCB*nColor*nColor*nSpin*nSpin/2;  // block-diagonal Hermitian (72 reals)
    length = 2*stride*nColor*nColor*nSpin*nSpin/2;

    bytes = length*precision;
    bytes = ALIGNMENT_ADJUST(bytes);
    if (precision == QUDA_HALF_PRECISION) {
      norm_bytes = sizeof(float)*2*stride*2; // 2 chirality
      norm_bytes = ALIGNMENT_ADJUST(norm_bytes);
    }
//for twisted mass only:
    twisted = false;//param.twisted;
    mu2 = 0.0; //param.mu2;
  }
  
  CloverField::~CloverField() {
    host_free(trlog);
  }

  cudaCloverField::cudaCloverField(const CloverFieldParam &param) : CloverField(param) {
    
    if (create != QUDA_NULL_FIELD_CREATE && create != QUDA_REFERENCE_FIELD_CREATE) 
      errorQuda("Create type %d not supported", create);

    if (param.direct) {
      if (create != QUDA_REFERENCE_FIELD_CREATE) {
	clover = device_malloc(bytes);
	if (precision == QUDA_HALF_PRECISION) norm = device_malloc(norm_bytes);
      } else {
	clover = param.clover;
	norm = param.norm;
      }

      even = clover;
      odd = (char*)clover + bytes/2;
    
      evenNorm = norm;
      oddNorm = (char*)norm + norm_bytes/2;

      total_bytes += bytes + norm_bytes;
    } 

    if (param.inverse) {
      if (create != QUDA_REFERENCE_FIELD_CREATE) {
	cloverInv = device_malloc(bytes);
	if (precision == QUDA_HALF_PRECISION) invNorm = device_malloc(norm_bytes);
      } else {
	cloverInv = param.cloverInv;
	invNorm = param.invNorm;
      }

      evenInv = cloverInv;
      oddInv = (char*)cloverInv + bytes/2;
    
      evenInvNorm = invNorm;
      oddInvNorm = (char*)invNorm + norm_bytes/2;

      total_bytes += bytes + norm_bytes;

      // this is a hack to ensure that we can autotune the clover
      // operator when just using symmetric preconditioning
      if (!param.direct) {
	clover = cloverInv;
	even = evenInv;
	odd = oddInv;
	norm = invNorm;
	evenNorm = evenInvNorm;
	oddNorm = oddInvNorm;
      }
    } 

#ifdef USE_TEXTURE_OBJECTS
    createTexObject(evenTex, evenNormTex, even, evenNorm);
    createTexObject(oddTex, oddNormTex, odd, oddNorm);
    createTexObject(evenInvTex, evenInvNormTex, evenInv, evenInvNorm);
    createTexObject(oddInvTex, oddInvNormTex, oddInv, oddInvNorm);
#endif
    twisted = param.twisted;
    mu2 = param.mu2;

  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaCloverField::createTexObject(cudaTextureObject_t &tex, cudaTextureObject_t &texNorm,
					void *field, void *norm) {

    if (order == QUDA_FLOAT2_CLOVER_ORDER || order == QUDA_FLOAT4_CLOVER_ORDER) {
      // create the texture for the field components
      
      cudaChannelFormatDesc desc;
      memset(&desc, 0, sizeof(cudaChannelFormatDesc));
      if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
      else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2
      
      // always four components regardless of precision
      desc.x = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
      desc.y = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
      desc.z = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
      desc.w = (precision == QUDA_DOUBLE_PRECISION) ? 8*sizeof(int) : 8*precision;
      
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = field;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = bytes/2;
      
      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
      else texDesc.readMode = cudaReadModeElementType;
      
      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
      checkCudaError();
      
      // create the texture for the norm components
      if (precision == QUDA_HALF_PRECISION) {
	cudaChannelFormatDesc desc;
	memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	desc.f = cudaChannelFormatKindFloat;
	desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;
	
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = norm;
	resDesc.res.linear.desc = desc;
	resDesc.res.linear.sizeInBytes = norm_bytes/2;
	
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	
	cudaCreateTextureObject(&texNorm, &resDesc, &texDesc, NULL);
	checkCudaError();
      }
    }

  }

  void cudaCloverField::destroyTexObject() {
    if (order == QUDA_FLOAT2_CLOVER_ORDER || order == QUDA_FLOAT4_CLOVER_ORDER) {
      cudaDestroyTextureObject(evenTex);
      cudaDestroyTextureObject(oddTex);
      cudaDestroyTextureObject(evenInvTex);
      cudaDestroyTextureObject(oddInvTex);
      if (precision == QUDA_HALF_PRECISION) {
	cudaDestroyTextureObject(evenNormTex);
	cudaDestroyTextureObject(oddNormTex);
	cudaDestroyTextureObject(evenInvNormTex);
	cudaDestroyTextureObject(oddInvNormTex);
      }
      checkCudaError();
    }
  }
#endif

  cudaCloverField::~cudaCloverField()
  {
#ifdef USE_TEXTURE_OBJECTS
    destroyTexObject();
#endif

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (clover != cloverInv) {
	if (clover) device_free(clover);
	if (norm) device_free(norm);
      }
      if (cloverInv) device_free(cloverInv);
      if (invNorm) device_free(invNorm);
    }
    
    checkCudaError();
  }

  void cudaCloverField::copy(const CloverField &src, bool inverse) {

    checkField(src);
    
    if (typeid(src) == typeid(cudaCloverField)) {
      if (src.V(false))	copyGenericClover(*this, src, false, QUDA_CUDA_FIELD_LOCATION);
      if (src.V(true)) copyGenericClover(*this, src, true, QUDA_CUDA_FIELD_LOCATION);
    } else if (typeid(src) == typeid(cpuCloverField)) {
      resizeBufferPinned(bytes + norm_bytes);
      void *packClover = bufferPinned[0];
      void *packCloverNorm = (precision == QUDA_HALF_PRECISION) ? (char*)(bufferPinned[0]) + bytes : 0;
      
      if (src.V(false)) {
	copyGenericClover(*this, src, false, QUDA_CPU_FIELD_LOCATION, packClover, 0, packCloverNorm, 0);
	cudaMemcpy(clover, packClover, bytes, cudaMemcpyHostToDevice);
	if (precision == QUDA_HALF_PRECISION) 
	  cudaMemcpy(norm, packCloverNorm, norm_bytes, cudaMemcpyHostToDevice);
      }
      
      if (src.V(true) && inverse) {
	copyGenericClover(*this, src, true, QUDA_CPU_FIELD_LOCATION, packClover, 0, packCloverNorm, 0);
	cudaMemcpy(cloverInv, packClover, bytes, cudaMemcpyHostToDevice);
	if (precision == QUDA_HALF_PRECISION) 
	  cudaMemcpy(invNorm, packCloverNorm, norm_bytes, cudaMemcpyHostToDevice);
      }
    } else {
      errorQuda("Invalid clover field type");
    }

    checkCudaError();
  }

  void cudaCloverField::loadCPUField(const cpuCloverField &cpu) { copy(cpu); }

  /**
     Computes Fmunu given the gauge field U
  */
  void cudaCloverField::compute(const cudaGaugeField &gauge) {

    if (gauge.Precision() != precision) 
      errorQuda("Gauge and clover precisions must match");

    computeClover(*this, gauge, 1.0, QUDA_CUDA_FIELD_LOCATION);

  }

  cpuCloverField::cpuCloverField(const CloverFieldParam &param) : CloverField(param) {
    if (create != QUDA_REFERENCE_FIELD_CREATE) errorQuda("Create type %d not supported", create);

    if (create == QUDA_REFERENCE_FIELD_CREATE) {
      clover = param.clover;
      norm = param.norm;
      cloverInv = param.cloverInv;
      invNorm = param.invNorm;
    }
  }

  cpuCloverField::~cpuCloverField() { 
    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (clover) host_free(clover);
      if (norm) host_free(norm);
      if (cloverInv) host_free(cloverInv);
      if (invNorm) host_free(invNorm);      
    }
  }

} // namespace quda
