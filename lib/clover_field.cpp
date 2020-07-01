#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <typeinfo>

#include <quda_internal.h>
#include <clover_field.h>
#include <gauge_field.h>
#include <color_spinor_field.h>
#include <blas_quda.h>

namespace quda {

  CloverFieldParam::CloverFieldParam(const CloverField &a) :
      LatticeFieldParam(a),
      direct(false),
      inverse(false),
      clover(NULL),
      norm(NULL),
      cloverInv(NULL),
      invNorm(NULL),
      csw(a.Csw()),
      twisted(a.Twisted()),
      mu2(a.Mu2()),
      rho(a.Rho()),
      order(a.Order()),
      create(QUDA_NULL_FIELD_CREATE)
  {
    precision = a.Precision();
    nDim = a.Ndim();
    pad = a.Pad();
    siteSubset = QUDA_FULL_SITE_SUBSET;
    for (int dir = 0; dir < nDim; ++dir) x[dir] = a.X()[dir];
  }

  CloverField::CloverField(const CloverFieldParam &param) :
    LatticeField(param), bytes(0), norm_bytes(0), nColor(3), nSpin(4), 
    clover(0), norm(0), cloverInv(0), invNorm(0), csw(param.csw), rho(param.rho),
    order(param.order), create(param.create), trlog{0, 0}
  {
    if (nDim != 4) errorQuda("Number of dimensions must be 4, not %d", nDim);

    if (order == QUDA_QDPJIT_CLOVER_ORDER && create != QUDA_REFERENCE_FIELD_CREATE)
      errorQuda("QDPJIT ordered clover fields only supported for reference fields");

    real_length = 2 * ((size_t)volumeCB) * nColor * nColor * nSpin * nSpin / 2; // block-diagonal Hermitian (72 reals)
    length = 2 * ((size_t)stride) * nColor * nColor * nSpin * nSpin / 2;

    bytes = length * precision;
    if (isNative()) bytes = 2*ALIGNMENT_ADJUST(bytes/2);
    if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
      norm_bytes = sizeof(float)*2*stride*2; // 2 chirality
      if (isNative()) norm_bytes = 2*ALIGNMENT_ADJUST(norm_bytes/2);
    }
//for twisted mass only:
    twisted = false;//param.twisted;
    mu2 = 0.0; //param.mu2;
  }
  
  CloverField::~CloverField() { }

  bool CloverField::isNative() const {
    if (precision == QUDA_DOUBLE_PRECISION) {
      if (order  == QUDA_FLOAT2_CLOVER_ORDER) return true;
    } else if (precision == QUDA_SINGLE_PRECISION || precision == QUDA_HALF_PRECISION
        || precision == QUDA_QUARTER_PRECISION) {
      if (order == QUDA_FLOAT4_CLOVER_ORDER) return true;
    }
    return false;
  }

  void CloverField::setRho(double rho_)
  {
    rho = rho_;
  }

  cudaCloverField::cudaCloverField(const CloverFieldParam &param) : CloverField(param) {
    
    if (create != QUDA_NULL_FIELD_CREATE && create != QUDA_REFERENCE_FIELD_CREATE) 
      errorQuda("Create type %d not supported", create);

    if (param.direct) {
      if (create != QUDA_REFERENCE_FIELD_CREATE) {
	clover = bytes ? pool_device_malloc(bytes) : nullptr;
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          norm = norm_bytes ? pool_device_malloc(norm_bytes) : nullptr;
      } else {
	clover = param.clover;
	norm = param.norm;
      }

      even = clover;
      odd = static_cast<char*>(clover) + bytes/2;
    
      evenNorm = norm;
      oddNorm = static_cast<char*>(norm) + norm_bytes/2;

      total_bytes += bytes + norm_bytes;

      // this is a hack to prevent us allocating a texture object for an unallocated inverse field
      if (!param.inverse) {
	cloverInv = clover;
	evenInv = even;
	oddInv = odd;
	invNorm = norm;
	evenInvNorm = evenNorm;
	oddInvNorm = oddNorm;
      }
    } 

    if (param.inverse) {
      if (create != QUDA_REFERENCE_FIELD_CREATE) {
	cloverInv = bytes ? pool_device_malloc(bytes) : nullptr;
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          invNorm = norm_bytes ? pool_device_malloc(norm_bytes) : nullptr;
      } else {
	cloverInv = param.cloverInv;
	invNorm = param.invNorm;
      }

      evenInv = cloverInv;
      oddInv = static_cast<char*>(cloverInv) + bytes/2;
    
      evenInvNorm = invNorm;
      oddInvNorm = static_cast<char*>(invNorm) + norm_bytes/2;

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

    if (!param.inverse) {
      cloverInv = clover;
      evenInv = even;
      oddInv = odd;
      invNorm = norm;
      evenInvNorm = evenNorm;
      oddInvNorm = oddNorm;
    }

#ifdef USE_TEXTURE_OBJECTS
    createTexObject(tex, normTex, clover, norm, true);
    createTexObject(invTex, invNormTex, cloverInv, invNorm, true);

    createTexObject(evenTex, evenNormTex, even, evenNorm, false);
    createTexObject(oddTex, oddNormTex, odd, oddNorm, false);

    createTexObject(evenInvTex, evenInvNormTex, evenInv, evenInvNorm, false);
    createTexObject(oddInvTex, oddInvNormTex, oddInv, oddInvNorm, false);
#endif
    twisted = param.twisted;
    mu2 = param.mu2;

  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaCloverField::createTexObject(cudaTextureObject_t &tex, cudaTextureObject_t &texNorm,
					void *field, void *norm, bool full) {
    if (isNative()) {
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
      int texel_size = 4 * (precision == QUDA_DOUBLE_PRECISION ? sizeof(int) : precision);

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = field;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = bytes/(!full ? 2 : 1);

      if (resDesc.res.linear.sizeInBytes % deviceProp.textureAlignment != 0
          || !is_aligned(resDesc.res.linear.devPtr, deviceProp.textureAlignment)) {
        errorQuda("Allocation size %lu does not have correct alignment for textures (%lu)",
                  resDesc.res.linear.sizeInBytes, deviceProp.textureAlignment);
      }

      unsigned long texels = resDesc.res.linear.sizeInBytes / texel_size;
      if (texels > (unsigned)deviceProp.maxTexture1DLinear) {
	errorQuda("Attempting to bind too large a texture %lu > %d", texels, deviceProp.maxTexture1DLinear);
      }

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
        texDesc.readMode = cudaReadModeNormalizedFloat;
      else
        texDesc.readMode = cudaReadModeElementType;

      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
      checkCudaError();
      
      // create the texture for the norm components
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
        cudaChannelFormatDesc desc;
	memset(&desc, 0, sizeof(cudaChannelFormatDesc));
	desc.f = cudaChannelFormatKindFloat;
	desc.x = 8*QUDA_SINGLE_PRECISION; desc.y = 0; desc.z = 0; desc.w = 0;
	
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = norm;
	resDesc.res.linear.desc = desc;
	resDesc.res.linear.sizeInBytes = norm_bytes/(!full ? 2 : 1);

        if (!is_aligned(resDesc.res.linear.devPtr, deviceProp.textureAlignment)) {
          errorQuda("Allocation size %lu does not have correct alignment for textures (%lu)",
                    resDesc.res.linear.sizeInBytes, deviceProp.textureAlignment);
        }

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&texNorm, &resDesc, &texDesc, NULL);
	checkCudaError();
      }
    }

  }

  void cudaCloverField::destroyTexObject() {
    if (isNative()) {
      cudaDestroyTextureObject(tex);
      cudaDestroyTextureObject(invTex);
      cudaDestroyTextureObject(evenTex);
      cudaDestroyTextureObject(oddTex);
      cudaDestroyTextureObject(evenInvTex);
      cudaDestroyTextureObject(oddInvTex);
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
        cudaDestroyTextureObject(normTex);
	cudaDestroyTextureObject(invNormTex);
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
	if (clover) pool_device_free(clover);
	if (norm) pool_device_free(norm);
      }
      if (cloverInv) pool_device_free(cloverInv);
      if (invNorm) pool_device_free(invNorm);
    }
    
    checkCudaError();
  }

  void cudaCloverField::copy(const CloverField &src, bool inverse) {

    checkField(src);
    
    if (typeid(src) == typeid(cudaCloverField)) {
      if (src.V(false))	copyGenericClover(*this, src, false, QUDA_CUDA_FIELD_LOCATION);
      if (src.V(true)) copyGenericClover(*this, src, true, QUDA_CUDA_FIELD_LOCATION);
    } else if (reorder_location() == QUDA_CPU_FIELD_LOCATION && typeid(src) == typeid(cpuCloverField)) {
      void *packClover = pool_pinned_malloc(bytes + norm_bytes);
      void *packCloverNorm = (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) ?
          static_cast<char *>(packClover) + bytes :
          0;

      if (src.V(false)) {
	copyGenericClover(*this, src, false, QUDA_CPU_FIELD_LOCATION, packClover, 0, packCloverNorm, 0);
	qudaMemcpy(clover, packClover, bytes, cudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(norm, packCloverNorm, norm_bytes, cudaMemcpyHostToDevice);
      }
      
      if (src.V(true) && inverse) {
	copyGenericClover(*this, src, true, QUDA_CPU_FIELD_LOCATION, packClover, 0, packCloverNorm, 0);
	qudaMemcpy(cloverInv, packClover, bytes, cudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(invNorm, packCloverNorm, norm_bytes, cudaMemcpyHostToDevice);
      }

      pool_pinned_free(packClover);
    } else if (reorder_location() == QUDA_CUDA_FIELD_LOCATION && typeid(src) == typeid(cpuCloverField)) {
      void *packClover = pool_device_malloc(src.Bytes() + src.NormBytes());
      void *packCloverNorm = (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) ?
          static_cast<char *>(packClover) + src.Bytes() :
          0;

      if (src.V(false)) {
	qudaMemcpy(packClover, src.V(false), src.Bytes(), cudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(packCloverNorm, src.Norm(false), src.NormBytes(), cudaMemcpyHostToDevice);

	copyGenericClover(*this, src, false, QUDA_CUDA_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
      }

      if (src.V(true) && inverse) {
	qudaMemcpy(packClover, src.V(true), src.Bytes(), cudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(packCloverNorm, src.Norm(true), src.NormBytes(), cudaMemcpyHostToDevice);

	copyGenericClover(*this, src, true, QUDA_CUDA_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
      }

      pool_device_free(packClover);
    } else {
      errorQuda("Invalid clover field type");
    }

    qudaDeviceSynchronize();
    checkCudaError();
  }

  void cudaCloverField::loadCPUField(const cpuCloverField &cpu) { copy(cpu); }

  void cudaCloverField::saveCPUField(cpuCloverField &cpu) const {
    checkField(cpu);

    // we know we are copying from GPU to CPU here, so for now just
    // assume that reordering is on CPU
    void *packClover = pool_pinned_malloc(bytes + norm_bytes);
    void *packCloverNorm = (precision == QUDA_HALF_PRECISION) ? static_cast<char*>(packClover) + bytes : 0;

    // first copy over the direct part (if it exists)
    if (V(false) && cpu.V(false)) {
      qudaMemcpy(packClover, clover, bytes, cudaMemcpyDeviceToHost);
      if (precision == QUDA_HALF_PRECISION)
	qudaMemcpy(packCloverNorm, norm, norm_bytes, cudaMemcpyDeviceToHost);
      copyGenericClover(cpu, *this, false, QUDA_CPU_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
    } else if((V(false) && !cpu.V(false)) || (!V(false) && cpu.V(false))) {
      errorQuda("Mismatch between Clover field GPU V(false) and CPU.V(false)");
    }

    // now copy the inverse part (if it exists)
    if (V(true) && cpu.V(true)) {
      qudaMemcpy(packClover, cloverInv, bytes, cudaMemcpyDeviceToHost);
	if (precision == QUDA_HALF_PRECISION)
	  qudaMemcpy(packCloverNorm, invNorm, norm_bytes, cudaMemcpyDeviceToHost);
      copyGenericClover(cpu, *this, true, QUDA_CPU_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
    } else if ((V(true) && !cpu.V(true)) || (!V(true) && cpu.V(true))) {
      errorQuda("Mismatch between Clover field GPU V(true) and CPU.V(true)");
    } 

    pool_pinned_free(packClover);

    qudaDeviceSynchronize();
    checkCudaError();
  }

  void cudaCloverField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    prefetch(mem_space, stream, CloverPrefetchType::BOTH_CLOVER_PREFETCH_TYPE);
  }

  void cudaCloverField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream, CloverPrefetchType type,
                                 QudaParity parity) const
  {
    if (is_prefetch_enabled()) {
      int dev_id = 0;
      if (mem_space == QUDA_CUDA_FIELD_LOCATION)
        dev_id = comm_gpuid();
      else if (mem_space == QUDA_CPU_FIELD_LOCATION)
        dev_id = cudaCpuDeviceId;
      else
        errorQuda("Invalid QudaFieldLocation.");

      auto clover_parity = clover;
      auto norm_parity = norm;
      auto cloverInv_parity = cloverInv;
      auto invNorm_parity = invNorm;
      auto bytes_parity = bytes;
      auto norm_bytes_parity = norm_bytes;
      if (parity != QUDA_INVALID_PARITY) {
        bytes_parity /= 2;
        norm_bytes_parity /= 2;
        if (parity == QUDA_EVEN_PARITY) {
          clover_parity = even;
          norm_parity = evenNorm;
          cloverInv_parity = evenInv;
          invNorm_parity = evenInvNorm;
        } else { // odd
          clover_parity = odd;
          norm_parity = oddNorm;
          cloverInv_parity = oddInv;
          invNorm_parity = oddInvNorm;
        }
      }

      switch (type) {
      case CloverPrefetchType::BOTH_CLOVER_PREFETCH_TYPE:
        if (clover_parity) cudaMemPrefetchAsync(clover_parity, bytes_parity, dev_id, stream);
        if (norm_parity) cudaMemPrefetchAsync(norm_parity, norm_bytes_parity, dev_id, stream);
        if (clover_parity != cloverInv_parity) {
          if (cloverInv_parity) cudaMemPrefetchAsync(cloverInv_parity, bytes_parity, dev_id, stream);
          if (invNorm_parity) cudaMemPrefetchAsync(invNorm_parity, norm_bytes_parity, dev_id, stream);
        }
        break;
      case CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE:
        if (clover_parity) cudaMemPrefetchAsync(clover_parity, bytes_parity, dev_id, stream);
        if (norm_parity) cudaMemPrefetchAsync(norm_parity, norm_bytes_parity, dev_id, stream);
        break;
      case CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE:
        if (cloverInv_parity) cudaMemPrefetchAsync(cloverInv_parity, bytes_parity, dev_id, stream);
        if (invNorm_parity) cudaMemPrefetchAsync(invNorm_parity, norm_bytes_parity, dev_id, stream);
        break;
      default: errorQuda("Invalid CloverPrefetchType.");
      }
    }
  }

  /**
     Computes Fmunu given the gauge field U
  */
  void cudaCloverField::compute(const cudaGaugeField &gauge) {

    if (gauge.Precision() != precision) 
      errorQuda("Gauge and clover precisions must match");

    computeClover(*this, gauge, 1.0, QUDA_CUDA_FIELD_LOCATION);

  }

  cpuCloverField::cpuCloverField(const CloverFieldParam &param) : CloverField(param) {

    if (create == QUDA_NULL_FIELD_CREATE || create == QUDA_ZERO_FIELD_CREATE) {
      if(order != QUDA_PACKED_CLOVER_ORDER) {errorQuda("cpuCloverField only supports QUDA_PACKED_CLOVER_ORDER");}
      clover = (void *) safe_malloc(bytes);
      if (precision == QUDA_HALF_PRECISION) norm = (void *) safe_malloc(norm_bytes);
      if(param.inverse) {
	cloverInv = (void *) safe_malloc(bytes);
	if (precision == QUDA_HALF_PRECISION) invNorm = (void *) safe_malloc(norm_bytes);
      }

      if(create == QUDA_ZERO_FIELD_CREATE) {
	memset(clover, '\0', bytes);
	if(param.inverse) memset(cloverInv, '\0', bytes);
	if(precision == QUDA_HALF_PRECISION) memset(norm, '\0', norm_bytes);
	if(param.inverse && precision ==QUDA_HALF_PRECISION) memset(invNorm, '\0', norm_bytes);
      }
    } else if (create == QUDA_REFERENCE_FIELD_CREATE) {
      clover = param.clover;
      norm = param.norm;
      cloverInv = param.cloverInv;
      invNorm = param.invNorm;
    } else {
      errorQuda("Create type %d not supported", create);
    }

    if (param.pad != 0) errorQuda("%s pad must be zero", __func__);
  }

  cpuCloverField::~cpuCloverField() { 
    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (clover) host_free(clover);
      if (norm) host_free(norm);
      if (cloverInv) host_free(cloverInv);
      if (invNorm) host_free(invNorm);      
    }
  }

  // This doesn't really live here, but is fine for the moment
  std::ostream& operator<<(std::ostream& output, const CloverFieldParam& param)
  {
    output << static_cast<const LatticeFieldParam&>(param);
    output << "direct = "    << param.direct << std::endl;
    output << "inverse = "   << param.inverse << std::endl;
    output << "clover = "    << param.clover << std::endl;
    output << "norm = "      << param.norm << std::endl;
    output << "cloverInv = " << param.cloverInv << std::endl;
    output << "invNorm = "   << param.invNorm << std::endl;
    output << "csw = "       << param.csw << std::endl;
    output << "twisted = "   << param.twisted << std::endl;
    output << "mu2 = "       << param.mu2 << std::endl;
    output << "rho = "       << param.rho << std::endl;
    output << "order = "     << param.order << std::endl;
    output << "create = "    << param.create << std::endl;
    return output;  // for multiple << operators.
  }

  ColorSpinorParam colorSpinorParam(const CloverField &a, bool inverse) {

    if (a.Precision() == QUDA_HALF_PRECISION)
      errorQuda("Casting a CloverField into ColorSpinorField not possible in half precision");

    ColorSpinorParam spinor_param;
    // 72 = 9 * 4 * 2
    spinor_param.nColor = 9;
    spinor_param.nSpin = 4;
    spinor_param.nDim = a.Ndim();
    for (int d=0; d<a.Ndim(); d++) spinor_param.x[d] = a.X()[d];
    spinor_param.setPrecision(a.Precision());
    spinor_param.pad = a.Pad();
    spinor_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    spinor_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    spinor_param.fieldOrder = a.Precision() == QUDA_DOUBLE_PRECISION ?
      QUDA_FLOAT2_FIELD_ORDER : QUDA_FLOAT4_FIELD_ORDER;
    spinor_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    spinor_param.create = QUDA_REFERENCE_FIELD_CREATE;
    spinor_param.v = (void*)a.V(inverse);
    spinor_param.location = a.Location();
    return spinor_param;
  }

  // Return the L2 norm squared of the clover field
  double norm2(const CloverField &a, bool inverse) {
    ColorSpinorField *b = ColorSpinorField::Create(colorSpinorParam(a, inverse));
    double nrm2 = blas::norm2(*b);
    delete b;
    return nrm2;
  }

  // Return the L1 norm of the clover field
  double norm1(const CloverField &a, bool inverse) {
    ColorSpinorField *b = ColorSpinorField::Create(colorSpinorParam(a, inverse));
    double nrm1 = blas::norm1(*b);
    delete b;
    return nrm1;
  }

} // namespace quda
