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
    direct(a.V(false)),
    inverse(a.V(true)),
    clover(nullptr),
    norm(nullptr),
    cloverInv(nullptr),
    invNorm(nullptr),
    csw(a.Csw()),
    coeff(a.Coeff()),
    twisted(a.Twisted()),
    mu2(a.Mu2()),
    rho(a.Rho()),
    order(a.Order()),
    create(QUDA_NULL_FIELD_CREATE),
    location(a.Location())
  {
    precision = a.Precision();
    nDim = a.Ndim();
    pad = a.Pad();
    siteSubset = QUDA_FULL_SITE_SUBSET;
    for (int dir = 0; dir < nDim; ++dir) x[dir] = a.X()[dir];
  }

  CloverField::CloverField(const CloverFieldParam &param) :
    LatticeField(param),
    bytes(0),
    norm_bytes(0),
    nColor(3),
    nSpin(4),
    clover(0),
    norm(0),
    cloverInv(0),
    invNorm(0),
    csw(param.csw),
    coeff(param.coeff),
    rho(param.rho),
    order(param.order),
    create(param.create),
    trlog {0, 0}
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

    setTuningString();
  }

  CloverField::~CloverField() { }

  void CloverField::setTuningString()
  {
    LatticeField::setTuningString();
    int aux_string_n = TuneKey::aux_n / 2;
    int check
      = snprintf(aux_string, aux_string_n, "vol=%lu,stride=%lu,precision=%d,Nc=%d", volume, stride, precision, nColor);
    if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
  }

  void CloverField::backup() const
  {
    if (backup_h) errorQuda("Already allocated host backup");
    backup_h = static_cast<char *>(safe_malloc(2 * bytes));
    if (norm_bytes) backup_norm_h = static_cast<char *>(safe_malloc(2 * norm_bytes));

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(backup_h, clover, bytes, qudaMemcpyDeviceToHost);
      qudaMemcpy(backup_h + bytes, cloverInv, bytes, qudaMemcpyDeviceToHost);
      if (norm_bytes) {
        qudaMemcpy(backup_norm_h, norm, norm_bytes, qudaMemcpyDeviceToHost);
        qudaMemcpy(backup_norm_h + norm_bytes, invNorm, norm_bytes, qudaMemcpyDeviceToHost);
      }
    } else {
      memcpy(backup_h, clover, bytes);
      memcpy(backup_h + bytes, cloverInv, bytes);
      if (norm_bytes) {
        memcpy(backup_norm_h, norm, norm_bytes);
        memcpy(backup_norm_h + norm_bytes, invNorm, norm_bytes);
      }
    }
  }

  void CloverField::restore() const
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(clover, backup_h, bytes, qudaMemcpyHostToDevice);
      qudaMemcpy(cloverInv, backup_h + bytes, bytes, qudaMemcpyHostToDevice);
      if (norm_bytes) {
        qudaMemcpy(norm, backup_norm_h, norm_bytes, qudaMemcpyHostToDevice);
        qudaMemcpy(invNorm, backup_norm_h + norm_bytes, norm_bytes, qudaMemcpyHostToDevice);
      }
    } else {
      memcpy(clover, backup_h, bytes);
      memcpy(cloverInv, backup_h + bytes, bytes);
      if (norm_bytes) {
        memcpy(norm, backup_norm_h, norm_bytes);
        memcpy(invNorm, backup_norm_h + norm_bytes, norm_bytes);
      }
    }

    host_free(backup_h);
    backup_h = nullptr;
    if (norm_bytes) {
      host_free(backup_norm_h);
      backup_norm_h = nullptr;
    }
  }

  CloverField *CloverField::Create(const CloverFieldParam &param)
  {
    CloverField *field = nullptr;
    if (param.location == QUDA_CPU_FIELD_LOCATION) {
      field = new cpuCloverField(param);
    } else if (param.location == QUDA_CUDA_FIELD_LOCATION) {
      field = new cudaCloverField(param);
    } else {
      errorQuda("Invalid field location %d", param.location);
    }

    return field;
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

    twisted = param.twisted;
    mu2 = param.mu2;
  }

  cudaCloverField::~cudaCloverField()
  {
    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (clover != cloverInv) {
	if (clover) pool_device_free(clover);
	if (norm) pool_device_free(norm);
      }
      if (cloverInv) pool_device_free(cloverInv);
      if (invNorm) pool_device_free(invNorm);
    }
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
        qudaMemcpy(clover, packClover, bytes, qudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(norm, packCloverNorm, norm_bytes, qudaMemcpyHostToDevice);
      }

      if (src.V(true) && inverse) {
	copyGenericClover(*this, src, true, QUDA_CPU_FIELD_LOCATION, packClover, 0, packCloverNorm, 0);
        qudaMemcpy(cloverInv, packClover, bytes, qudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(invNorm, packCloverNorm, norm_bytes, qudaMemcpyHostToDevice);
      }

      pool_pinned_free(packClover);
    } else if (reorder_location() == QUDA_CUDA_FIELD_LOCATION && typeid(src) == typeid(cpuCloverField)) {
      void *packClover = pool_device_malloc(src.Bytes() + src.NormBytes());
      void *packCloverNorm = (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) ?
          static_cast<char *>(packClover) + src.Bytes() :
          0;

      if (src.V(false)) {
        qudaMemcpy(packClover, src.V(false), src.Bytes(), qudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(packCloverNorm, src.Norm(false), src.NormBytes(), qudaMemcpyHostToDevice);

        copyGenericClover(*this, src, false, QUDA_CUDA_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
      }

      if (src.V(true) && inverse) {
        qudaMemcpy(packClover, src.V(true), src.Bytes(), qudaMemcpyHostToDevice);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
          qudaMemcpy(packCloverNorm, src.Norm(true), src.NormBytes(), qudaMemcpyHostToDevice);

        copyGenericClover(*this, src, true, QUDA_CUDA_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
      }

      pool_device_free(packClover);
    } else {
      errorQuda("Invalid clover field type");
    }

    qudaDeviceSynchronize();
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
      qudaMemcpy(packClover, clover, bytes, qudaMemcpyDeviceToHost);
      if (precision == QUDA_HALF_PRECISION) qudaMemcpy(packCloverNorm, norm, norm_bytes, qudaMemcpyDeviceToHost);
      copyGenericClover(cpu, *this, false, QUDA_CPU_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
    } else if((V(false) && !cpu.V(false)) || (!V(false) && cpu.V(false))) {
      errorQuda("Mismatch between Clover field GPU V(false) and CPU.V(false)");
    }

    // now copy the inverse part (if it exists)
    if (V(true) && cpu.V(true)) {
      qudaMemcpy(packClover, cloverInv, bytes, qudaMemcpyDeviceToHost);
      if (precision == QUDA_HALF_PRECISION) qudaMemcpy(packCloverNorm, invNorm, norm_bytes, qudaMemcpyDeviceToHost);
      copyGenericClover(cpu, *this, true, QUDA_CPU_FIELD_LOCATION, 0, packClover, 0, packCloverNorm);
    } else if ((V(true) && !cpu.V(true)) || (!V(true) && cpu.V(true))) {
      errorQuda("Mismatch between Clover field GPU V(true) and CPU.V(true)");
    }

    pool_pinned_free(packClover);

    qudaDeviceSynchronize();
  }

  void cudaCloverField::copy_to_buffer(void *buffer) const
  {

    size_t buffer_offset = 0;
    if (V(false)) { // direct
      qudaMemcpy(buffer, clover, bytes, qudaMemcpyDeviceToHost);
      if (precision < QUDA_SINGLE_PRECISION) {
        qudaMemcpy(static_cast<char *>(buffer) + bytes, norm, norm_bytes, qudaMemcpyDeviceToHost);
      }
      buffer_offset += bytes + norm_bytes;
    }

    if (V(true)) { // inverse
      qudaMemcpy(static_cast<char *>(buffer) + buffer_offset, cloverInv, bytes, qudaMemcpyDeviceToHost);
      if (precision < QUDA_SINGLE_PRECISION) {
        qudaMemcpy(static_cast<char *>(buffer) + buffer_offset + bytes, invNorm, norm_bytes, qudaMemcpyDeviceToHost);
      }
    }
  }

  void cudaCloverField::copy_from_buffer(void *buffer)
  {

    size_t buffer_offset = 0;
    if (V(false)) { // direct
      qudaMemcpy(clover, static_cast<char *>(buffer), bytes, qudaMemcpyHostToDevice);
      if (precision < QUDA_SINGLE_PRECISION) {
        qudaMemcpy(norm, static_cast<char *>(buffer) + bytes, norm_bytes, qudaMemcpyHostToDevice);
      }
      buffer_offset += bytes + norm_bytes;
    }

    if (V(true)) { // inverse
      qudaMemcpy(cloverInv, static_cast<char *>(buffer) + buffer_offset, bytes, qudaMemcpyHostToDevice);
      if (precision < QUDA_SINGLE_PRECISION) {
        qudaMemcpy(invNorm, static_cast<char *>(buffer) + buffer_offset + bytes, norm_bytes, qudaMemcpyHostToDevice);
      }
    }
  }

  void cudaCloverField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    prefetch(mem_space, stream, CloverPrefetchType::BOTH_CLOVER_PREFETCH_TYPE);
  }

  void cudaCloverField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream, CloverPrefetchType type,
                                 QudaParity parity) const
  {
    if (is_prefetch_enabled()) {
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
        if (clover_parity) qudaMemPrefetchAsync(clover_parity, bytes_parity, mem_space, stream);
        if (norm_parity) qudaMemPrefetchAsync(norm_parity, norm_bytes_parity, mem_space, stream);
        if (clover_parity != cloverInv_parity) {
          if (cloverInv_parity) qudaMemPrefetchAsync(cloverInv_parity, bytes_parity, mem_space, stream);
          if (invNorm_parity) qudaMemPrefetchAsync(invNorm_parity, norm_bytes_parity, mem_space, stream);
        }
        break;
      case CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE:
        if (clover_parity) qudaMemPrefetchAsync(clover_parity, bytes_parity, mem_space, stream);
        if (norm_parity) qudaMemPrefetchAsync(norm_parity, norm_bytes_parity, mem_space, stream);
        break;
      case CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE:
        if (cloverInv_parity) qudaMemPrefetchAsync(cloverInv_parity, bytes_parity, mem_space, stream);
        if (invNorm_parity) qudaMemPrefetchAsync(invNorm_parity, norm_bytes_parity, mem_space, stream);
        break;
      default: errorQuda("Invalid CloverPrefetchType.");
      }
    }
  }

  /**
     Computes Fmunu given the gauge field U
  */
  void cudaCloverField::compute(const cudaGaugeField &gauge) { computeClover(*this, gauge, 1.0); }

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

  cpuCloverField::~cpuCloverField()
  {
    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (clover) host_free(clover);
      if (norm) host_free(norm);
      if (cloverInv) host_free(cloverInv);
      if (invNorm) host_free(invNorm);
    }
  }

  void cpuCloverField::copy_to_buffer(void *buffer) const
  {

    size_t buffer_offset = 0;
    if (V(false)) { // direct
      std::memcpy(static_cast<char *>(buffer), clover, bytes);
      if (precision < QUDA_SINGLE_PRECISION) { std::memcpy(static_cast<char *>(buffer) + bytes, norm, norm_bytes); }
      buffer_offset += bytes + norm_bytes;
    }

    if (V(true)) { // inverse
      std::memcpy(static_cast<char *>(buffer) + buffer_offset, cloverInv, bytes);
      if (precision < QUDA_SINGLE_PRECISION) {
        std::memcpy(static_cast<char *>(buffer) + buffer_offset + bytes, invNorm, norm_bytes);
      }
    }
  }

  void cpuCloverField::copy_from_buffer(void *buffer)
  {

    size_t buffer_offset = 0;
    if (V(false)) { // direct
      std::memcpy(clover, static_cast<char *>(buffer), bytes);
      if (precision < QUDA_SINGLE_PRECISION) { std::memcpy(norm, static_cast<char *>(buffer) + bytes, norm_bytes); }
      buffer_offset += bytes + norm_bytes;
    }

    if (V(true)) { // inverse
      std::memcpy(cloverInv, static_cast<char *>(buffer) + buffer_offset, bytes);
      if (precision < QUDA_SINGLE_PRECISION) {
        std::memcpy(invNorm, static_cast<char *>(buffer) + buffer_offset + bytes, norm_bytes);
      }
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
    output << "coeff = " << param.coeff << std::endl;
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
