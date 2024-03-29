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

  CloverFieldParam::CloverFieldParam(const CloverField &a) : LatticeFieldParam(a)
  {
    a.fill(*this);
    create = QUDA_NULL_FIELD_CREATE;
  }

  CloverField::CloverField(const CloverFieldParam &param) : LatticeField(param)
  {
    create(param); // create switch-case here for create type

    switch (param.create) {
    case QUDA_NULL_FIELD_CREATE:
    case QUDA_REFERENCE_FIELD_CREATE: break;
    case QUDA_ZERO_FIELD_CREATE:
      qudaMemset(clover, '\0', bytes);
      if (inverse) qudaMemset(cloverInv, '\0', bytes);
      break;
    case QUDA_COPY_FIELD_CREATE: copy(*param.field); break;
    default: errorQuda("Unexpected create type %d", param.create);
    }
  }

  CloverField::CloverField(const CloverField &field) noexcept : LatticeField(field)
  {
    CloverFieldParam param;
    field.fill(param);
    param.create = QUDA_COPY_FIELD_CREATE;
    create(param);
    copy(field);
  }

  CloverField::CloverField(CloverField &&field) noexcept : LatticeField(std::move(field)) { move(std::move(field)); }

  CloverField &CloverField::operator=(const CloverField &src)
  {
    if (&src != this) {
      if (!init) { // keep current attributes unless unset
        LatticeField::operator=(src);
        CloverFieldParam param;
        src.fill(param);
        param.create = QUDA_COPY_FIELD_CREATE;
        create(param);
      }

      copy(src);
    }
    return *this;
  }

  CloverField &CloverField::operator=(CloverField &&src)
  {
    if (&src != this) {
      // if field not already initialized then move the field
      if (!init) {
        LatticeField::operator=(std::move(src));
        move(std::move(src));
      } else {
        // we error if the field is not compatible with this
        errorQuda("Moving to already created field");
      }
    }
    return *this;
  }

  void CloverField::create(const CloverFieldParam &param)
  {
    reconstruct = param.reconstruct;
    nColor = 3;
    nSpin = 4;
    inverse = param.inverse;
    csw = param.csw;
    coeff = param.coeff;
    twist_flavor = param.twist_flavor;
    mu2 = param.mu2;
    rho = param.rho;
    order = param.order;

    if (siteSubset != QUDA_FULL_SITE_SUBSET) errorQuda("Unexpected siteSubset %d", siteSubset);
    if (nDim != 4) errorQuda("Number of dimensions must be 4, not %d", nDim);
    if (!isNative() && precision < QUDA_SINGLE_PRECISION)
      errorQuda("Fixed-point precision only supported on native field");
    if (order == QUDA_QDPJIT_CLOVER_ORDER && param.create != QUDA_REFERENCE_FIELD_CREATE)
      errorQuda("QDPJIT ordered clover fields only supported for reference fields");
    if (param.create != QUDA_NULL_FIELD_CREATE && param.create != QUDA_REFERENCE_FIELD_CREATE
        && param.create != QUDA_ZERO_FIELD_CREATE && param.create != QUDA_COPY_FIELD_CREATE)
      errorQuda("Create type %d not supported", param.create);

    // for now we only support compressed blocks for Nc = 3
    if (reconstruct && !isNative()) errorQuda("Clover reconstruct only supported on native fields");

    compressed_block = (reconstruct && nColor == 3 && nSpin == 4) ? 28 : (nColor * nSpin / 2) * (nColor * nSpin / 2);
    real_length = 2 * ((size_t)volumeCB) * 2 * compressed_block; // block-diagonal Hermitian (72 reals)
    length = 2 * ((size_t)stride) * 2 * compressed_block;

    bytes = length * precision;
    if (isNative()) bytes = 2*ALIGNMENT_ADJUST(bytes/2);

    setTuningString();

    if (bytes) {
      if (param.create != QUDA_REFERENCE_FIELD_CREATE) {
        clover = quda_ptr(mem_type, bytes);
      } else {
        clover = quda_ptr(param.clover, mem_type);
      }

      total_bytes += bytes;

      if (inverse) {
        if (param.create != QUDA_REFERENCE_FIELD_CREATE) {
          cloverInv = quda_ptr(mem_type, bytes);
        } else {
          cloverInv = quda_ptr(param.cloverInv, mem_type);
        }

        total_bytes += bytes;
      }
    }

    twist_flavor = param.twist_flavor;
    mu2 = param.mu2;
    epsilon2 = param.epsilon2;

    init = true;
  }

  // Fills the param with the contents of this field
  void CloverField::fill(CloverFieldParam &param) const
  {
    LatticeField::fill(param);
    param.reconstruct = reconstruct;
    param.inverse = inverse;
    param.field = const_cast<CloverField *>(this);
    param.clover = data(false);
    if (inverse) param.cloverInv = data(true);
    param.csw = csw;
    param.coeff = coeff;
    param.twist_flavor = twist_flavor;
    param.mu2 = mu2;
    param.epsilon2 = epsilon2;
    param.rho = rho;
    param.order = order;
  }

  void CloverField::move(CloverField &&src)
  {
    reconstruct = std::exchange(src.reconstruct, false);
    bytes = std::exchange(src.bytes, 0);
    length = std::exchange(src.length, 0);
    real_length = std::exchange(src.real_length, 0);
    compressed_block = std::exchange(src.compressed_block, 0);
    nColor = std::exchange(src.nColor, 0);
    nSpin = std::exchange(src.nSpin, 0);
    clover.exchange(src.clover, {});
    if (src.inverse) cloverInv.exchange(src.cloverInv, {});
    inverse = std::exchange(src.inverse, false);
    diagonal = std::exchange(src.diagonal, 0.0);
    max = std::exchange(src.max, {});
    csw = std::exchange(src.csw, 0.0);
    coeff = std::exchange(src.coeff, 0.0);
    twist_flavor = std::exchange(src.twist_flavor, QUDA_TWIST_INVALID);
    mu2 = std::exchange(src.mu2, 0.0);
    epsilon2 = std::exchange(src.epsilon2, 0.0);
    rho = std::exchange(src.rho, 0.0);
    order = std::exchange(src.order, QUDA_INVALID_CLOVER_ORDER);
    trlog = std::exchange(src.trlog, {});
    init = std::exchange(src.init, false);
  }

  void CloverField::setTuningString()
  {
    LatticeField::setTuningString();
    std::stringstream aux_ss;
    aux_ss << "vol=" << volume << "precision=" << precision << "Nc=" << nColor;
    aux_string = aux_ss.str();
    if (aux_string.size() >= TuneKey::aux_n / 2) errorQuda("Aux string too large %lu", aux_string.size());
  }

  void CloverField::backup(bool which) const
  {
    qudaMemcpy(backup_h[which], which ? cloverInv : clover, bytes, qudaMemcpyDefault);
  }

  void CloverField::backup() const
  {
    if (backup_h.size()) errorQuda("Already allocated host backup");
    backup_h.resize(2);
    for (auto &b : backup_h) b = quda_ptr(QUDA_MEMORY_HOST, bytes);

    backup(false);
    if (inverse) backup(true);
  }

  void CloverField::restore(bool which) const
  {
    qudaMemcpy(which ? cloverInv : clover, backup_h[which], bytes, qudaMemcpyDefault);
  }

  void CloverField::restore() const
  {
    if (!backup_h.size()) errorQuda("Cannot restore since not backed up");
    restore(false);
    if (inverse) restore(true);

    backup_h.resize(0);
  }

  CloverField *CloverField::Create(const CloverFieldParam &param) { return new CloverField(param); }

  void CloverField::setRho(double rho_)
  {
    rho = rho_;
  }

  void CloverField::copy(const CloverField &src, bool is_inverse)
  {
    if (src.Location() == QUDA_CUDA_FIELD_LOCATION && location == QUDA_CPU_FIELD_LOCATION) {
      getProfile().TPSTART(QUDA_PROFILE_D2H);
    } else if (src.Location() == QUDA_CPU_FIELD_LOCATION && location == QUDA_CUDA_FIELD_LOCATION) {
      getProfile().TPSTART(QUDA_PROFILE_H2D);
    }

    // special case where we wish to make a copy of the inverse field when dynamic_inverse is enabled
    static bool dynamic_inverse_copy = false;
    if (is_inverse && clover::dynamic_inverse() && inverse && !src.inverse && !dynamic_inverse_copy) {
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
    if (is_inverse && !inverse) errorQuda("Destination field's is_inverse=%d component does not exist", is_inverse);
    if (is_inverse && !src.Inverse() && !dynamic_inverse_copy)
      errorQuda("Source field's is_inverse=%d component does not exist", is_inverse);

    auto src_v = dynamic_inverse_copy ? src.data(false) : src.data(is_inverse);

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
        qudaMemcpy(data(is_inverse), packClover, bytes, qudaMemcpyHostToDevice);

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
        qudaMemcpy(data(is_inverse), packClover, bytes, qudaMemcpyDeviceToHost);

        pool_device_free(packClover);
      }
    }

    if (src.Location() == QUDA_CUDA_FIELD_LOCATION && location == QUDA_CPU_FIELD_LOCATION) {
      getProfile().TPSTOP(QUDA_PROFILE_D2H);
    } else if (src.Location() == QUDA_CPU_FIELD_LOCATION && location == QUDA_CUDA_FIELD_LOCATION) {
      getProfile().TPSTOP(QUDA_PROFILE_H2D);
    }
  }

  void CloverField::copy(const CloverField &src)
  {
    copy(src, false);
    if (!clover::dynamic_inverse()) copy(src, true);
  }

  void CloverField::copy_to_buffer(void *buffer) const
  {
    size_t buffer_offset = 0;
    qudaMemcpy(buffer, clover.data(), bytes, qudaMemcpyDefault);
    buffer_offset += bytes;

    if (inverse) { // inverse
      qudaMemcpy(static_cast<char *>(buffer) + buffer_offset, cloverInv.data(), bytes, qudaMemcpyDefault);
    }
  }

  void CloverField::copy_from_buffer(void *buffer)
  {
    size_t buffer_offset = 0;
    qudaMemcpy(clover.data(), static_cast<char *>(buffer), bytes, qudaMemcpyDefault);
    buffer_offset += bytes;

    if (inverse) { // inverse
      qudaMemcpy(cloverInv.data(), static_cast<char *>(buffer) + buffer_offset, bytes, qudaMemcpyDefault);
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
      auto bytes_parity = parity == QUDA_INVALID_PARITY ? bytes : bytes / 2;
      auto clover_parity = clover.data();
      auto cloverInv_parity = inverse ? cloverInv.data() : nullptr;
      if (parity == QUDA_ODD_PARITY) {
        clover_parity = static_cast<char *>(clover_parity) + bytes_parity;
        cloverInv_parity = inverse ? static_cast<char *>(cloverInv_parity) + bytes_parity : nullptr;
      }

      switch (type) {
      case CloverPrefetchType::BOTH_CLOVER_PREFETCH_TYPE:
        if (clover_parity) qudaMemPrefetchAsync(clover_parity, bytes_parity, mem_space, stream);
        if (cloverInv_parity) qudaMemPrefetchAsync(cloverInv_parity, bytes_parity, mem_space, stream);
        break;
      case CloverPrefetchType::CLOVER_CLOVER_PREFETCH_TYPE:
        if (clover_parity) qudaMemPrefetchAsync(clover_parity, bytes_parity, mem_space, stream);
        break;
      case CloverPrefetchType::INVERSE_CLOVER_PREFETCH_TYPE:
        if (!clover::dynamic_inverse()) {
          if (cloverInv_parity) qudaMemPrefetchAsync(cloverInv_parity, bytes_parity, mem_space, stream);
        } else {
          if (clover_parity) qudaMemPrefetchAsync(clover_parity, bytes_parity, mem_space, stream);
        }
        break;
      default: errorQuda("Invalid CloverPrefetchType %d", static_cast<int>(type));
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
    output << "twist_flavor = " << param.twist_flavor << std::endl;
    output << "mu2 = " << param.mu2 << std::endl;
    output << "epsilon2 = " << param.epsilon2 << std::endl;
    output << "rho = " << param.rho << std::endl;
    output << "order = " << param.order << std::endl;
    output << "create = " << param.create << std::endl;
    return output;  // for multiple << operators.
  }

  ColorSpinorParam colorSpinorParam(const CloverField &a, bool inverse)
  {
    if (a.Precision() == QUDA_HALF_PRECISION)
      errorQuda("Casting a CloverField into ColorSpinorField not possible in half precision");

    ColorSpinorParam spinor_param;
    spinor_param.nColor = (2 * a.compressed_block_size()) / (2 * a.Nspin());
    spinor_param.nSpin = a.Nspin();
    spinor_param.nDim = a.Ndim();
    for (int d=0; d<a.Ndim(); d++) spinor_param.x[d] = a.X()[d];
    spinor_param.setPrecision(a.Precision());
    spinor_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    spinor_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    spinor_param.fieldOrder = colorspinor::getNative(a.Precision(), a.Nspin());
    spinor_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    spinor_param.create = QUDA_REFERENCE_FIELD_CREATE;
    spinor_param.v = a.data(inverse);
    spinor_param.location = a.Location();
    return spinor_param;
  }

  // Return the L2 norm squared of the clover field
  double norm2(const CloverField &a, bool inverse)
  {
    ColorSpinorField b(colorSpinorParam(a, inverse));
    return blas::norm2(b);
  }

  // Return the L1 norm of the clover field
  double norm1(const CloverField &a, bool inverse)
  {
    ColorSpinorField b(colorSpinorParam(a, inverse));
    return blas::norm1(b);
  }

} // namespace quda
