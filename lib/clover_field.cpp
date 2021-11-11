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
    reconstruct(clover::reconstruct()),
    inverse(a.V(true)),
    clover(nullptr),
    cloverInv(nullptr),
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
    reconstruct(param.reconstruct),
    bytes(0),
    nColor(3),
    nSpin(4),
    clover(0),
    cloverInv(0),
    diagonal(0),
    max {0, 0},
    csw(param.csw),
    coeff(param.coeff),
    twisted(param.twisted),
    mu2(param.mu2),
    rho(param.rho),
    order(param.order),
    create(param.create),
    location(param.location),
    trlog {0, 0}
  {
    if (nDim != 4) errorQuda("Number of dimensions must be 4, not %d", nDim);
    if (!isNative() && precision < QUDA_SINGLE_PRECISION)
      errorQuda("Fixed-point precision only supported on native field");
    if (!isNative() && param.pad != 0) errorQuda("pad must be zero");
    if (order == QUDA_QDPJIT_CLOVER_ORDER && create != QUDA_REFERENCE_FIELD_CREATE)
      errorQuda("QDPJIT ordered clover fields only supported for reference fields");
    if (create != QUDA_NULL_FIELD_CREATE && create != QUDA_REFERENCE_FIELD_CREATE && create != QUDA_ZERO_FIELD_CREATE)
      errorQuda("Create type %d not supported", create);

    // for now we only support compressed blocks for Nc = 3
    if (reconstruct && !isNative()) errorQuda("Clover reconstruct only supported on native fields");
    compressed_block = (reconstruct && nColor == 3 && nSpin == 4) ? 28 : (nColor * nSpin / 2) * (nColor * nSpin / 2);
    real_length = 2 * ((size_t)volumeCB) * 2 * compressed_block; // block-diagonal Hermitian (72 reals)
    length = 2 * ((size_t)stride) * 2 * compressed_block;

    bytes = length * precision;
    if (isNative()) bytes = 2*ALIGNMENT_ADJUST(bytes/2);

    setTuningString();

    if (bytes) {
      if (create != QUDA_REFERENCE_FIELD_CREATE) {
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          clover = pool_device_malloc(bytes);
        } else {
          clover = safe_malloc(bytes);
        }

      } else {
        clover = param.clover;
      }

      total_bytes += bytes;

      if (param.inverse) {
        if (create != QUDA_REFERENCE_FIELD_CREATE) {
          if (location == QUDA_CUDA_FIELD_LOCATION) {
            cloverInv = pool_device_malloc(bytes);
          } else {
            cloverInv = safe_malloc(bytes);
          }
        } else {
          cloverInv = param.cloverInv;
        }

        total_bytes += bytes;
      }

      if (create == QUDA_ZERO_FIELD_CREATE) {
        if (location == QUDA_CUDA_FIELD_LOCATION) {
          qudaMemset(clover, '\0', bytes);
          if (param.inverse) qudaMemset(cloverInv, '\0', bytes);
        } else {
          memset(clover, '\0', bytes);
          if (param.inverse) memset(cloverInv, '\0', bytes);
        }
      }
    }
  }

  CloverField::~CloverField()
  {
    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (location == QUDA_CUDA_FIELD_LOCATION) {
        if (clover) pool_device_free(clover);
        if (cloverInv) pool_device_free(cloverInv);
      } else {
        if (clover) host_free(clover);
        if (cloverInv) host_free(cloverInv);
      }
    }
  }

  void CloverField::setTuningString()
  {
    LatticeField::setTuningString();
    int aux_string_n = TuneKey::aux_n / 2;
    int check
      = snprintf(aux_string, aux_string_n, "vol=%lu,stride=%lu,precision=%d,Nc=%d", volume, stride, precision, nColor);
    if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
  }

  void CloverField::backup(bool which) const
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(backup_h + which * bytes, V(which), bytes, qudaMemcpyDeviceToHost);
    } else {
      memcpy(backup_h + which * bytes, V(which), bytes);
    }
  }

  void CloverField::backup() const
  {
    if (backup_h) errorQuda("Already allocated host backup");
    backup_h = static_cast<char *>(safe_malloc(2 * bytes));

    if (V(false)) backup(false);
    if (V(true)) backup(true);
  }

  void CloverField::restore(bool which) const
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy((void *)V(which), backup_h + which * bytes, bytes, qudaMemcpyHostToDevice);
    } else {
      memcpy((void *)V(which), backup_h + which * bytes, bytes);
    }
  }

  void CloverField::restore() const
  {
    if (V(false)) restore(false);
    if (V(true)) restore(true);

    host_free(backup_h);
    backup_h = nullptr;
  }

  CloverField *CloverField::Create(const CloverFieldParam &param) { return new CloverField(param); }

  void CloverField::setRho(double rho_)
  {
    rho = rho_;
  }

  void CloverField::copy(const CloverField &src, bool is_inverse)
  {
    // special case where we wish to make a copy of the inverse field when dynamic_inverse is enabled
    static bool dynamic_inverse_copy = false;
    if (is_inverse && clover::dynamic_inverse() && V(true) && !src.V(true) && !dynamic_inverse_copy) {
      dynamic_inverse_copy = true;
      // create a copy of the clover field that we will invert in place and use as the source
      CloverFieldParam param(src);
      param.inverse = false;
      param.reconstruct = false; // we cannot use a compressed field for storing the inverse
      CloverField clover_inverse(param);
      clover_inverse.copy(src, false);
      cloverInvert(clover_inverse, true);
      copy(clover_inverse, true);
      dynamic_inverse_copy = false;
      return;
    }

    checkField(src);
    if (!V(is_inverse)) errorQuda("Destination field's is_inverse=%d component does not exist", is_inverse);
    if (!src.V(is_inverse) && !dynamic_inverse_copy)
      errorQuda("Source field's is_inverse=%d component does not exist", is_inverse);

    auto src_v = dynamic_inverse_copy ? src.V(false) : src.V(is_inverse);

    // if we copying to a reconstruction field, we must find the overall scale factor to allow us to reconstruct
    if (Reconstruct()) {
      if (src.Reconstruct())
        Diagonal(src.Diagonal());
      else
        Diagonal(-1);
    }

    if (precision < QUDA_SINGLE_PRECISION) {
      // if the destination is fixed point, then we must set the global norm
      max[is_inverse] = src.Precision() >= QUDA_SINGLE_PRECISION ? src.abs_max(is_inverse) : src.max_element(is_inverse);
    }

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      if (src.Location() == QUDA_CUDA_FIELD_LOCATION) {
        copyGenericClover(*this, src, is_inverse, QUDA_CUDA_FIELD_LOCATION, 0, src_v);
      } else if (reorder_location() == QUDA_CPU_FIELD_LOCATION && src.Location() == QUDA_CPU_FIELD_LOCATION) {
        void *packClover = pool_pinned_malloc(bytes);

        copyGenericClover(*this, src, is_inverse, QUDA_CPU_FIELD_LOCATION, packClover, src_v);
        qudaMemcpy(V(is_inverse), packClover, bytes, qudaMemcpyHostToDevice);

        pool_pinned_free(packClover);
      } else if (reorder_location() == QUDA_CUDA_FIELD_LOCATION && src.Location() == QUDA_CPU_FIELD_LOCATION) {
        void *packClover = pool_device_malloc(src.Bytes());

        qudaMemcpy(packClover, src_v, src.Bytes(), qudaMemcpyHostToDevice);
        copyGenericClover(*this, src, is_inverse, QUDA_CUDA_FIELD_LOCATION, 0, packClover);

        pool_device_free(packClover);
      }
    } else if (Location() == QUDA_CPU_FIELD_LOCATION) {
      if (src.Location() == QUDA_CPU_FIELD_LOCATION) {
        copyGenericClover(*this, src, is_inverse, QUDA_CPU_FIELD_LOCATION, 0, src_v);
      } else if (reorder_location() == QUDA_CPU_FIELD_LOCATION && src.Location() == QUDA_CUDA_FIELD_LOCATION) {
        void *packClover = pool_pinned_malloc(src.Bytes());

        qudaMemcpy(packClover, src_v, src.Bytes(), qudaMemcpyDeviceToHost);
        copyGenericClover(*this, src, is_inverse, QUDA_CPU_FIELD_LOCATION, 0, packClover);

        pool_pinned_free(packClover);
      } else if (reorder_location() == QUDA_CUDA_FIELD_LOCATION && src.Location() == QUDA_CUDA_FIELD_LOCATION) {
        void *packClover = pool_device_malloc(bytes);

        copyGenericClover(*this, src, is_inverse, QUDA_CUDA_FIELD_LOCATION, packClover, src_v);
        qudaMemcpy(V(is_inverse), packClover, bytes, qudaMemcpyDeviceToHost);

        pool_device_free(packClover);
      }
    }

    qudaDeviceSynchronize();
  }

  void CloverField::copy(const CloverField &src)
  {
    copy(src, false);
    if (!clover::dynamic_inverse()) copy(src, true);
  }

  void CloverField::copy_to_buffer(void *buffer) const
  {
    size_t buffer_offset = 0;
    if (V(false)) { // direct
      qudaMemcpy(buffer, clover, bytes, qudaMemcpyDefault);
      buffer_offset += bytes;
    }

    if (V(true)) { // inverse
      qudaMemcpy(static_cast<char *>(buffer) + buffer_offset, cloverInv, bytes, qudaMemcpyDefault);
    }
  }

  void CloverField::copy_from_buffer(void *buffer)
  {
    size_t buffer_offset = 0;
    if (V(false)) { // direct
      qudaMemcpy(clover, static_cast<char *>(buffer), bytes, qudaMemcpyDefault);
      buffer_offset += bytes;
    }

    if (V(true)) { // inverse
      qudaMemcpy(cloverInv, static_cast<char *>(buffer) + buffer_offset, bytes, qudaMemcpyDefault);
    }
  }

  void CloverField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    if (location == QUDA_CUDA_FIELD_LOCATION)
      prefetch(mem_space, stream, CloverPrefetchType::BOTH_CLOVER_PREFETCH_TYPE);
  }

  void CloverField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream, CloverPrefetchType type,
                             QudaParity parity) const
  {
    if (location == QUDA_CUDA_FIELD_LOCATION && is_prefetch_enabled()) {
      auto clover_parity = clover;
      auto cloverInv_parity = cloverInv;
      auto bytes_parity = parity == QUDA_INVALID_PARITY ? bytes : bytes / 2;
      if (parity != QUDA_INVALID_PARITY && parity == QUDA_ODD_PARITY) {
        clover_parity = static_cast<char *>(clover_parity) + bytes_parity;
        cloverInv_parity = static_cast<char *>(cloverInv_parity) + bytes_parity;
      }

      switch (type) {
      case CloverPrefetchType::BOTH_CLOVER_PREFETCH_TYPE:
        if (clover_parity) qudaMemPrefetchAsync(clover_parity, bytes_parity, mem_space, stream);
        if (clover_parity != cloverInv_parity) {
          if (cloverInv_parity) qudaMemPrefetchAsync(cloverInv_parity, bytes_parity, mem_space, stream);
        }
        break;
      case CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE:
        if (clover_parity) qudaMemPrefetchAsync(clover_parity, bytes_parity, mem_space, stream);
        break;
      case CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE:
        if (cloverInv_parity) qudaMemPrefetchAsync(cloverInv_parity, bytes_parity, mem_space, stream);
        break;
      default: errorQuda("Invalid CloverPrefetchType.");
      }
    }
  }

  // This doesn't really live here, but is fine for the moment
  std::ostream& operator<<(std::ostream& output, const CloverFieldParam& param)
  {
    output << static_cast<const LatticeFieldParam&>(param);
    output << "reconstruct = " << param.reconstruct << std::endl;
    output << "inverse = "   << param.inverse << std::endl;
    output << "clover = "    << param.clover << std::endl;
    output << "cloverInv = " << param.cloverInv << std::endl;
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
    spinor_param.nColor = (2 * a.compressed_block_size()) / (2 * a.Nspin());
    spinor_param.nSpin = a.Nspin();
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
