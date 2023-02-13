#include <string.h>
#include <iostream>
#include <typeinfo>

#include <color_spinor_field.h>
#include <dslash_quda.h>
#include <field_cache.h>
#include <uint_to_char.h>

static bool zeroCopy = false;

namespace quda
{

  ColorSpinorParam::ColorSpinorParam(const ColorSpinorField &field) : LatticeFieldParam()
  {
    field.fill(*this);
    init = true;
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param) :
    LatticeField(param),
    composite_descr(param.is_composite, param.composite_dim, param.is_component, param.component_id),
    components(0)
  {
    // this must come before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      norm_offset = param.norm_offset;
      reference = true;
    } else if (param.create == QUDA_GHOST_FIELD_CREATE) {
      ghost_only = true;
    }

    create(param);

    switch (param.create) {
    case QUDA_NULL_FIELD_CREATE:
    case QUDA_REFERENCE_FIELD_CREATE:
    case QUDA_GHOST_FIELD_CREATE: break; // do nothing;
    case QUDA_ZERO_FIELD_CREATE: zero(); break;
    case QUDA_COPY_FIELD_CREATE: copy(*param.field); break;
    default: errorQuda("Unexpected create type %d", param.create);
    }
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorField &field) noexcept :
    LatticeField(field),
    composite_descr(field.composite_descr),
    components(0)
  {
    ColorSpinorParam param;
    field.fill(param);
    param.create = QUDA_COPY_FIELD_CREATE;
    create(param);
    copy(field);
  }

  ColorSpinorField::ColorSpinorField(ColorSpinorField &&field) noexcept : LatticeField(std::move(field))
  {
    move(std::move(field));
  }

  ColorSpinorField::~ColorSpinorField() { destroy(); }

  ColorSpinorField &ColorSpinorField::operator=(const ColorSpinorField &src)
  {
    if (&src != this) {
      if (!init) { // keep current attributes unless unset
        LatticeField::operator=(src);
        composite_descr = src.composite_descr;
        ColorSpinorParam param;
        src.fill(param);
        param.create = QUDA_COPY_FIELD_CREATE;
        create(param);
      }

      copy(src);
    }
    return *this;
  }

  ColorSpinorField &ColorSpinorField::operator=(ColorSpinorField &&src)
  {
    if (&src != this) {
      // if field not already initialized then move the field
      if (!init || are_compatible(*this, src)) {
        if (init) destroy();
        LatticeField::operator=(std::move(src));
        move(std::move(src));
      } else {
        // we error if the field is not compatible with this
        errorQuda("Moving to already created field");
      }
    }
    return *this;
  }

  void ColorSpinorField::create(const ColorSpinorParam &param)
  {
    if (param.create == QUDA_INVALID_FIELD_CREATE) errorQuda("Invalid create type");

    siteOrder = param.siteOrder;
    fieldOrder = param.fieldOrder;
    gammaBasis = param.gammaBasis;
    nColor = param.nColor;
    nSpin = param.nSpin;
    nVec = param.nVec;
    twistFlavor = param.twistFlavor;

    if (param.pc_type != QUDA_5D_PC && param.pc_type != QUDA_4D_PC) errorQuda("Unexpected pc_type %d", param.pc_type);
    pc_type = param.pc_type;
    suggested_parity = param.suggested_parity;

    precision = param.Precision();

    if (twistFlavor == QUDA_TWIST_NONDEG_DOUBLET && x[4] != 2) // two flavors
      errorQuda(
        "Must be two flavors for non-degenerate twisted mass spinor (while provided with %d number of components)",
        x[4]);

    if (param.pad != 0) errorQuda("Padding must be zero");
    length = siteSubset * volumeCB * nColor * nSpin * 2;
    bytes_raw = length * precision;
    if (precision < QUDA_SINGLE_PRECISION) bytes_raw += siteSubset * volumeCB * sizeof(float);

    norm_offset = volumeCB * nColor * nSpin * 2 * precision; // this is the offset in bytes to start of the norm

    // alignment must be done on parity boundaries
    if (isNative())
      bytes = siteSubset * ALIGNMENT_ADJUST(bytes_raw / siteSubset);
    else
      bytes = bytes_raw;

    //! stuff for deflated solvers (eigenvector sets):
    if (composite_descr.is_composite) {
      if (composite_descr.is_component) errorQuda("Composite type is not implemented");
      composite_descr.volume = volume;
      composite_descr.volumeCB = volumeCB;
      composite_descr.length = length;
      composite_descr.bytes = bytes;

      volume *= composite_descr.dim;
      volumeCB *= composite_descr.dim;
      length *= composite_descr.dim;
      bytes_raw *= composite_descr.dim;
      bytes *= composite_descr.dim;
    } else if (composite_descr.is_component) {
      composite_descr.dim = 0;
      composite_descr.volume = 0;
      composite_descr.volumeCB = 0;
      composite_descr.length = 0;
      composite_descr.bytes = 0;
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER)
      errorQuda("Subset not implemented");

    if (param.create != QUDA_REFERENCE_FIELD_CREATE && param.create != QUDA_GHOST_FIELD_CREATE) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        v = safe_malloc(bytes);
      } else if (location == QUDA_CUDA_FIELD_LOCATION) {
        switch (mem_type) {
        case QUDA_MEMORY_DEVICE: v = pool_device_malloc(bytes); break;
        case QUDA_MEMORY_MAPPED:
          v_h = mapped_malloc(bytes);
          v = get_mapped_device_pointer(v_h);
          break;
        default: errorQuda("Unsupported memory type %d", mem_type);
        }
      } else {
        errorQuda("Unexpected field location %d", location);
      }
      alloc = true;
    }

    if (composite_descr.is_composite && param.create != QUDA_REFERENCE_FIELD_CREATE
        && param.create != QUDA_GHOST_FIELD_CREATE) {
      ColorSpinorParam param;
      fill(param);
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      param.is_composite = false;
      param.composite_dim = 0;
      param.is_component = true;

      components.reserve(composite_descr.dim);
      for (int cid = 0; cid < composite_descr.dim; cid++) {
        param.component_id = cid;
        param.v = static_cast<void *>(static_cast<char *>(v) + cid * bytes / composite_descr.dim);
        components.push_back(new ColorSpinorField(param));
      }
    }

    // create the associated even and odd subsets
    if (siteSubset == QUDA_FULL_SITE_SUBSET && fieldOrder != QUDA_QDPJIT_FIELD_ORDER && !composite_descr.is_composite) {
      ColorSpinorParam param;
      fill(param);
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      param.siteSubset = QUDA_PARITY_SITE_SUBSET;
      param.x[0] /= 2; // set single parity dimensions
      param.is_composite = false;
      param.composite_dim = 0;
      param.is_component = composite_descr.is_component;
      param.component_id = composite_descr.id;
      even = new ColorSpinorField(param);
      param.v = static_cast<char *>(v) + bytes / 2;
      odd = new ColorSpinorField(param);
    }

    if (isNative() && param.create != QUDA_REFERENCE_FIELD_CREATE && param.create != QUDA_GHOST_FIELD_CREATE) {
      if (!(siteSubset == QUDA_FULL_SITE_SUBSET && composite_descr.is_composite)) {
        zeroPad();
      } else { // temporary hack for the full spinor field sets, manual zeroPad for each component:
        for (int cid = 0; cid < composite_descr.dim; cid++) {
          components[cid]->Even().zeroPad();
          components[cid]->Odd().zeroPad();
        }
      }
    }

    dslash_constant = static_cast<DslashConstant *>(safe_malloc(sizeof(DslashConstant)));
    init = true;
    setTuningString();
  }

  void ColorSpinorField::zeroPad()
  {
    // zero the region added for alignment reasons
    if (bytes != bytes_raw) {
      size_t subset_bytes = bytes / siteSubset;
      size_t subset_bytes_raw = bytes_raw / siteSubset;
      for (int subset = 0; subset < siteSubset; subset++) {
        if (location == QUDA_CUDA_FIELD_LOCATION)
          qudaMemsetAsync(static_cast<char *>(v) + subset_bytes_raw + subset_bytes * subset, 0,
                          subset_bytes - subset_bytes_raw, device::get_default_stream());
        else
          memset(static_cast<char *>(v) + subset_bytes_raw + subset_bytes * subset, 0, subset_bytes - subset_bytes_raw);
      }
    }
  }

  void ColorSpinorField::move(ColorSpinorField &&src)
  {
    init = std::exchange(src.init, false);
    alloc = std::exchange(src.alloc, false);
    reference = std::exchange(src.reference, false);
    ghost_precision_allocated = std::exchange(src.ghost_precision_allocated, QUDA_INVALID_PRECISION);
    nColor = std::exchange(src.nColor, 0);
    nSpin = std::exchange(src.nSpin, 0);
    nVec = std::exchange(src.nVec, 0);
    twistFlavor = std::exchange(src.twistFlavor, QUDA_TWIST_INVALID);
    pc_type = std::exchange(src.pc_type, QUDA_PC_INVALID);
    suggested_parity = std::exchange(src.suggested_parity, QUDA_INVALID_PARITY);
    length = std::exchange(src.length, 0);
    v = std::exchange(src.v, nullptr);
    v_h = std::exchange(src.v_h, nullptr);
    norm_offset = std::exchange(src.norm_offset, 0);
    ghost = std::exchange(src.ghost, {});
    ghostFace = std::exchange(src.ghostFace, {});
    ghostFaceCB = std::exchange(src.ghostFaceCB, {});
    ghost_buf = std::exchange(src.ghost_buf, {});
    dslash_constant = std::exchange(src.dslash_constant, nullptr);
    bytes = std::exchange(src.bytes, 0);
    bytes_raw = std::exchange(src.bytes_raw, 0);
    siteOrder = std::exchange(src.siteOrder, QUDA_INVALID_SITE_ORDER);
    fieldOrder = std::exchange(src.fieldOrder, QUDA_INVALID_FIELD_ORDER);
    gammaBasis = std::exchange(src.gammaBasis, QUDA_INVALID_GAMMA_BASIS);
    even = std::exchange(src.even, nullptr);
    odd = std::exchange(src.odd, nullptr);
    composite_descr = std::exchange(src.composite_descr, CompositeColorSpinorFieldDescriptor());
    components = std::move(src.components);
  }

  void ColorSpinorField::destroy()
  {
    if (alloc) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        host_free(v);
      } else { // device field
        switch (mem_type) {
        case QUDA_MEMORY_DEVICE: pool_device_free(v); break;
        case QUDA_MEMORY_MAPPED: host_free(v_h); break;
        default: errorQuda("Unsupported memory type %d", mem_type);
        }
      }
      alloc = false;
      v = nullptr;
      v_h = nullptr;

      if (composite_descr.is_composite) {
        CompositeColorSpinorField::iterator vec;
        for (vec = components.begin(); vec != components.end(); vec++) {
          delete *vec;
          *vec = nullptr;
        }
        components.resize(0);
      }
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET && !composite_descr.is_composite) {
      if (even) {
        delete even;
        even = nullptr;
      }
      if (odd) {
        delete odd;
        odd = nullptr;
      }
    }

    if (dslash_constant) {
      host_free(dslash_constant);
      dslash_constant = nullptr;
    }

    init = false;
  }

  void ColorSpinorField::setTuningString()
  {
    LatticeField::setTuningString();
    if (init) {
      std::stringstream aux_ss;
      aux_ss << "vol=" << volume << ",parity=" << siteSubset << ",precision=" << precision << ",order=" << fieldOrder
             << ",Ns=" << nSpin << ",Nc=" << nColor;
      if (twistFlavor != QUDA_TWIST_NO && twistFlavor != QUDA_TWIST_INVALID) aux_ss << ",TwistFlavor=" << twistFlavor;
      aux_string = aux_ss.str();
      if (aux_string.size() >= TuneKey::aux_n / 2) errorQuda("Aux string too large %lu", aux_string.size());
    }
  }

  void ColorSpinorField::createGhostZone(int nFace, bool spin_project) const
  {
    if (ghost_precision == QUDA_INVALID_PRECISION) errorQuda("Invalid requested ghost precision");
    if (ghost_precision_allocated == ghost_precision) return;

    bool is_fixed = (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION);
    int nSpinGhost = (nSpin == 4 && spin_project) ? 2 : nSpin;
    size_t site_size = nSpinGhost * nColor * 2 * ghost_precision + (is_fixed ? sizeof(float) : 0);

    // calculate size of ghost zone required
    int dims = nDim == 5 ? (nDim - 1) : nDim;
    int x5 = nDim == 5 ? x[4] : 1; /// includes DW and non-degenerate TM ghosts
    const int ghost_align
      = 1; // TODO perhaps in the future we should align each ghost dim/dir, e.g., along 32-byte boundaries
    ghost_bytes = 0;

    for (int i = 0; i < dims; i++) {
      ghostFace[i] = 0;
      if (comm_dim_partitioned(i)) {
        ghostFace[i] = 1;
        for (int j = 0; j < dims; j++) {
          if (i == j) continue;
          ghostFace[i] *= x[j];
        }
        ghostFace[i] *= x5; // temporary hack : extra dimension for DW ghosts
        if (i == 0 && siteSubset != QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
      }

      ghost_face_bytes[i] = nFace * ghostFace[i] * site_size;
      ghost_face_bytes_aligned[i] = ((ghost_face_bytes[i] + ghost_align - 1) / ghost_align) * ghost_align;
      ghost_offset[i][0] = i == 0 ? 0 : ghost_offset[i - 1][0] + 2 * ghost_face_bytes_aligned[i - 1];
      ghost_offset[i][1] = ghost_offset[i][0] + ghost_face_bytes_aligned[i];
      ghost_bytes += 2 * ghost_face_bytes_aligned[i];

      ghostFaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET ? ghostFace[i] / 2 : ghostFace[i]);
    } // dim

    if (isNative()) ghost_bytes = ALIGNMENT_ADJUST(ghost_bytes);

    { // compute temporaries needed by dslash and packing kernels
      auto &dc = *dslash_constant;
      auto &X = dc.X;
      for (int dim = 0; dim < nDim; dim++) X[dim] = x[dim];
      for (int dim = nDim; dim < QUDA_MAX_DIM; dim++) X[dim] = 1;
      if (siteSubset == QUDA_PARITY_SITE_SUBSET) X[0] = 2 * X[0];

      for (int i = 0; i < nDim; i++) dc.Xh[i] = X[i] / 2;

      dc.Ls = X[4];
      dc.volume_4d_cb = volumeCB / (nDim == 5 ? x[4] : 1);
      dc.volume_4d = 2 * dc.volume_4d_cb;

      int face[4];
      for (int dim = 0; dim < 4; dim++) {
        for (int j = 0; j < 4; j++) face[j] = X[j];
        face[dim] = nFace;
        dc.face_X[dim] = face[0];
        dc.face_Y[dim] = face[1];
        dc.face_Z[dim] = face[2];
        dc.face_T[dim] = face[3];
        dc.face_XY[dim] = dc.face_X[dim] * face[1];
        dc.face_XYZ[dim] = dc.face_XY[dim] * face[2];
        dc.face_XYZT[dim] = dc.face_XYZ[dim] * face[3];
      }

      dc.Vh = (X[3] * X[2] * X[1] * X[0]) / 2;
      dc.ghostFace[0] = X[1] * X[2] * X[3];
      dc.ghostFace[1] = X[0] * X[2] * X[3];
      dc.ghostFace[2] = X[0] * X[1] * X[3];
      dc.ghostFace[3] = X[0] * X[1] * X[2];
      for (int d = 0; d < 4; d++) dc.ghostFaceCB[d] = dc.ghostFace[d] / 2;

      dc.X2X1 = X[1] * X[0];
      dc.X3X2X1 = X[2] * X[1] * X[0];
      dc.X4X3X2X1 = X[3] * X[2] * X[1] * X[0];
      dc.X2X1mX1 = (X[1] - 1) * X[0];
      dc.X3X2X1mX2X1 = (X[2] - 1) * X[1] * X[0];
      dc.X4X3X2X1mX3X2X1 = (X[3] - 1) * X[2] * X[1] * X[0];
      dc.X5X4X3X2X1mX4X3X2X1 = (X[4] - 1) * X[3] * X[2] * X[1] * X[0];
      dc.X4X3X2X1hmX3X2X1h = dc.X4X3X2X1mX3X2X1 / 2;

      // used by indexFromFaceIndexStaggered
      dc.dims[0][0] = X[1];
      dc.dims[0][1] = X[2];
      dc.dims[0][2] = X[3];

      dc.dims[1][0] = X[0];
      dc.dims[1][1] = X[2];
      dc.dims[1][2] = X[3];

      dc.dims[2][0] = X[0];
      dc.dims[2][1] = X[1];
      dc.dims[2][2] = X[3];

      dc.dims[3][0] = X[0];
      dc.dims[3][1] = X[1];
      dc.dims[3][2] = X[2];
    }
    ghost_precision_allocated = ghost_precision;
  } // createGhostZone

  void ColorSpinorField::zero()
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemsetAsync(v, 0, bytes, device::get_default_stream());
    } else {
      memset(v, '\0', bytes);
    }
  }

  void ColorSpinorField::copy(const ColorSpinorField &src)
  {
    test_compatible_weak(*this, src);
    if (Location() == src.Location()) { // H2H and D2D

      copyGenericColorSpinor(*this, src, Location());

    } else if (Location() == QUDA_CUDA_FIELD_LOCATION && src.Location() == QUDA_CPU_FIELD_LOCATION) { // H2D

      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // reorder on host
        void *buffer = pool_pinned_malloc(bytes);
        memset(buffer, 0, bytes); // FIXME (temporary?) bug fix for padding
        copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, 0);
        qudaMemcpy(v, buffer, bytes, qudaMemcpyDefault);
        pool_pinned_free(buffer);

      } else { // reorder on device

        if (src.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
          // special case where we use mapped memory to read/write directly from application's array
          void *src_d = get_mapped_device_pointer(src.V());
          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, v, src_d);
        } else {
          void *Src = nullptr, *buffer = nullptr;
          if (!zeroCopy) {
            buffer = pool_device_malloc(src.Bytes());
            Src = buffer;
            qudaMemcpy(Src, src.V(), src.Bytes(), qudaMemcpyDefault);
          } else {
            buffer = pool_pinned_malloc(src.Bytes());
            memcpy(buffer, src.V(), src.Bytes());
            Src = get_mapped_device_pointer(buffer);
          }

          qudaMemsetAsync(v, 0, bytes, device::get_default_stream()); // FIXME (temporary?) bug fix for padding
          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, 0, Src);

          if (zeroCopy)
            pool_pinned_free(buffer);
          else
            pool_device_free(buffer);
        }
      }
      qudaDeviceSynchronize(); // include sync here for accurate host-device profiling

    } else if (Location() == QUDA_CPU_FIELD_LOCATION && src.Location() == QUDA_CUDA_FIELD_LOCATION) { // D2H

      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // reorder on the host
        void *buffer = pool_pinned_malloc(bytes);
        qudaMemcpy(buffer, v, bytes, qudaMemcpyDefault);
        copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, 0, buffer);
        pool_pinned_free(buffer);

      } else { // reorder on the device

        if (FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
          // special case where we use zero-copy memory to read/write directly from application's array
          void *dest_d = get_mapped_device_pointer(v);
          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, dest_d, src.V());
        } else {
          void *dst = nullptr, *buffer = nullptr;
          if (!zeroCopy) {
            buffer = pool_device_malloc(bytes);
            dst = buffer;
          } else {
            buffer = pool_pinned_malloc(bytes);
            dst = get_mapped_device_pointer(buffer);
          }

          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, dst, 0);

          if (!zeroCopy) {
            qudaMemcpy(v, dst, Bytes(), qudaMemcpyDefault);
          } else {
            qudaDeviceSynchronize();
            memcpy(v, buffer, bytes);
          }

          if (zeroCopy)
            pool_pinned_free(buffer);
          else
            pool_device_free(buffer);
        }
      }

      qudaDeviceSynchronize(); // need to sync before data can be used on CPU
    }
  }

  // Fills the param with the contents of this field
  void ColorSpinorField::fill(ColorSpinorParam &param) const
  {
    LatticeField::fill(param);
    param.field = const_cast<ColorSpinorField *>(this);
    param.v = v;
    param.nColor = nColor;
    param.nSpin = nSpin;
    param.nVec = nVec;
    param.twistFlavor = twistFlavor;
    param.fieldOrder = fieldOrder;
    param.setPrecision(precision, ghost_precision); // intentionally called here and not in LatticeField
    param.is_composite = composite_descr.is_composite;
    param.composite_dim = composite_descr.dim;
    param.is_component = false; // always either a regular spinor or a composite object
    param.component_id = 0;
    param.siteOrder = siteOrder;
    param.gammaBasis = gammaBasis;
    param.pc_type = pc_type;
    param.suggested_parity = suggested_parity;
    param.create = QUDA_NULL_FIELD_CREATE;
  }

  void ColorSpinorField::exchange(void **ghost, void **sendbuf, int nFace) const
  {
    // FIXME: use LatticeField MsgHandles
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];
    size_t bytes[4];

    const int Ninternal = 2 * nColor * nSpin;
    size_t total_bytes = 0;
    for (int i = 0; i < nDimComms; i++) {
      bytes[i] = siteSubset * nFace * surfaceCB[i] * Ninternal * ghost_precision;
      if (comm_dim_partitioned(i)) total_bytes += 2 * bytes[i]; // 2 for fwd/bwd
    }

    void *total_send = nullptr;
    void *total_recv = nullptr;
    void *send_fwd[4];
    void *send_back[4];
    void *recv_fwd[4];
    void *recv_back[4];

    // leave this option in there just in case
    bool no_comms_fill = false;

    // If this is set to false, then we are assuming that the send and
    // ghost buffers are in a single contiguous memory space.  Setting
    // to false means we aggregate all qudaMemcpys which reduces
    // latency.
    bool fine_grained_memcpy = false;

    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      for (int i = 0; i < nDimComms; i++) {
        if (comm_dim_partitioned(i)) {
          send_back[i] = sendbuf[2 * i + 0];
          send_fwd[i] = sendbuf[2 * i + 1];
          recv_fwd[i] = ghost[2 * i + 1];
          recv_back[i] = ghost[2 * i + 0];
        } else if (no_comms_fill) {
          memcpy(ghost[2 * i + 1], sendbuf[2 * i + 0], bytes[i]);
          memcpy(ghost[2 * i + 0], sendbuf[2 * i + 1], bytes[i]);
        }
      }
    } else { // FIXME add GPU_COMMS support
      if (total_bytes) {
        total_send = pool_pinned_malloc(total_bytes);
        total_recv = pool_pinned_malloc(total_bytes);
      }
      size_t offset = 0;
      for (int i = 0; i < nDimComms; i++) {
        if (comm_dim_partitioned(i)) {
          send_back[i] = static_cast<char *>(total_send) + offset;
          recv_back[i] = static_cast<char *>(total_recv) + offset;
          offset += bytes[i];
          send_fwd[i] = static_cast<char *>(total_send) + offset;
          recv_fwd[i] = static_cast<char *>(total_recv) + offset;
          offset += bytes[i];
          if (fine_grained_memcpy) {
            qudaMemcpy(send_back[i], sendbuf[2 * i + 0], bytes[i], qudaMemcpyDeviceToHost);
            qudaMemcpy(send_fwd[i], sendbuf[2 * i + 1], bytes[i], qudaMemcpyDeviceToHost);
          }
        } else if (no_comms_fill) {
          qudaMemcpy(ghost[2 * i + 1], sendbuf[2 * i + 0], bytes[i], qudaMemcpyDeviceToDevice);
          qudaMemcpy(ghost[2 * i + 0], sendbuf[2 * i + 1], bytes[i], qudaMemcpyDeviceToDevice);
        }
      }
      if (!fine_grained_memcpy && total_bytes) {
        // find first non-zero pointer
        void *send_ptr = nullptr;
        for (int i = 0; i < nDimComms; i++) {
          if (comm_dim_partitioned(i)) {
            send_ptr = sendbuf[2 * i];
            break;
          }
        }
        qudaMemcpy(total_send, send_ptr, total_bytes, qudaMemcpyDeviceToHost);
      }
    }

    for (int i = 0; i < nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      mh_send_fwd[i] = comm_declare_send_relative(send_fwd[i], i, +1, bytes[i]);
      mh_send_back[i] = comm_declare_send_relative(send_back[i], i, -1, bytes[i]);
      mh_from_fwd[i] = comm_declare_receive_relative(recv_fwd[i], i, +1, bytes[i]);
      mh_from_back[i] = comm_declare_receive_relative(recv_back[i], i, -1, bytes[i]);
    }

    for (int i = 0; i < nDimComms; i++) {
      if (comm_dim_partitioned(i)) {
        comm_start(mh_from_back[i]);
        comm_start(mh_from_fwd[i]);
        comm_start(mh_send_fwd[i]);
        comm_start(mh_send_back[i]);
      }
    }

    for (int i = 0; i < nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_wait(mh_send_fwd[i]);
      comm_wait(mh_send_back[i]);
      comm_wait(mh_from_back[i]);
      comm_wait(mh_from_fwd[i]);
    }

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      for (int i = 0; i < nDimComms; i++) {
        if (!comm_dim_partitioned(i)) continue;
        if (fine_grained_memcpy) {
          qudaMemcpy(ghost[2 * i + 0], recv_back[i], bytes[i], qudaMemcpyHostToDevice);
          qudaMemcpy(ghost[2 * i + 1], recv_fwd[i], bytes[i], qudaMemcpyHostToDevice);
        }
      }

      if (!fine_grained_memcpy && total_bytes) {
        // find first non-zero pointer
        void *ghost_ptr = nullptr;
        for (int i = 0; i < nDimComms; i++) {
          if (comm_dim_partitioned(i)) {
            ghost_ptr = ghost[2 * i];
            break;
          }
        }
        qudaMemcpy(ghost_ptr, total_recv, total_bytes, qudaMemcpyHostToDevice);
      }

      if (total_bytes) {
        pool_pinned_free(total_send);
        pool_pinned_free(total_recv);
      }
    }

    for (int i = 0; i < nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_free(mh_send_fwd[i]);
      comm_free(mh_send_back[i]);
      comm_free(mh_from_back[i]);
      comm_free(mh_from_fwd[i]);
    }
  }

  bool ColorSpinorField::are_compatible_weak(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    return (a.SiteSubset() == b.SiteSubset() && a.VolumeCB() == b.VolumeCB() && a.Ncolor() == b.Ncolor()
            && a.Nspin() == b.Nspin() && a.Nvec() == b.Nvec() && a.TwistFlavor() == b.TwistFlavor());
  }

  bool ColorSpinorField::are_compatible(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    return (a.Precision() == b.Precision() && a.FieldOrder() == b.FieldOrder() && are_compatible_weak(a, b));
  }

  void ColorSpinorField::test_compatible_weak(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    if (a.SiteSubset() != b.SiteSubset()) errorQuda("siteSubsets do not match: %d %d", a.SiteSubset(), b.SiteSubset());
    if (a.VolumeCB() != b.VolumeCB()) errorQuda("volumes do not match: %lu %lu", a.VolumeCB(), b.VolumeCB());
    if (a.Ncolor() != b.Ncolor()) errorQuda("colors do not match: %d %d", a.Ncolor(), b.Ncolor());
    if (a.Nspin() != b.Nspin()) errorQuda("spins do not match: %d %d", a.Nspin(), b.Nspin());
    if (a.Nvec() != b.Nvec()) errorQuda("nVec does not match: %d %d", a.Nvec(), b.Nvec());
    if (a.TwistFlavor() != b.TwistFlavor())
      errorQuda("twist flavors do not match: %d %d", a.TwistFlavor(), b.TwistFlavor());
  }

  void ColorSpinorField::test_compatible(const ColorSpinorField &a, const ColorSpinorField &b)
  {
    test_compatible_weak(a, b);
    if (a.Precision() != b.Precision()) errorQuda("precisions do not match: %d %d", a.Precision(), b.Precision());
    if (a.FieldOrder() != b.FieldOrder()) errorQuda("orders do not match: %d %d", a.FieldOrder(), b.FieldOrder());
  }

  const ColorSpinorField &ColorSpinorField::Even() const
  {
    if (siteSubset != QUDA_FULL_SITE_SUBSET) errorQuda("Cannot return even subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER) errorQuda("Cannot return even subset of QDPJIT field");
    return *even;
  }

  const ColorSpinorField &ColorSpinorField::Odd() const
  {
    if (siteSubset != QUDA_FULL_SITE_SUBSET) errorQuda("Cannot return odd subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER) errorQuda("Cannot return even subset of QDPJIT field");
    return *odd;
  }

  ColorSpinorField &ColorSpinorField::Even()
  {
    if (siteSubset != QUDA_FULL_SITE_SUBSET) errorQuda("Cannot return even subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER) errorQuda("Cannot return even subset of QDPJIT field");
    return *even;
  }

  ColorSpinorField &ColorSpinorField::Odd()
  {
    if (siteSubset != QUDA_FULL_SITE_SUBSET) errorQuda("Cannot return odd subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER) errorQuda("Cannot return even subset of QDPJIT field");
    return *odd;
  }

  ColorSpinorField& ColorSpinorField::Component(int idx)
  {
    if (!IsComposite()) errorQuda("Not composite field");
    if (idx >= CompositeDim()) errorQuda("Invalid component index %d (size = %d)", idx, CompositeDim());
    return *(components[idx]);
  }

  const ColorSpinorField &ColorSpinorField::Component(int idx) const
  {
    if (!IsComposite()) errorQuda("Not composite field");
    if (idx >= CompositeDim()) errorQuda("Invalid component index %d (size = %d)", idx, CompositeDim());
    return *(components[idx]);
  }

  void *const *ColorSpinorField::Ghost() const { return ghost_buf.data; }

  const void *ColorSpinorField::Ghost2() const
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      return nullptr;
    } else {
      if (bufferIndex < 2) {
        return ghost_recv_buffer_d[bufferIndex];
      } else {
        return ghost_pinned_recv_buffer_hd[bufferIndex % 2];
      }
    }
  }

  /*
    Convert from 1-dimensional index to the n-dimensional spatial index.
    With full fields, we assume that the field is even-odd ordered.  The
    lattice coordinates that are computed here are full-field
    coordinates.
  */
  void ColorSpinorField::LatticeIndex(int *y, int i) const
  {
    auto z = x;

    // parity is the slowest running dimension
    int parity = 0;
    if (siteSubset == QUDA_FULL_SITE_SUBSET) z[0] /= 2;

    for (int d = 0; d < nDim; d++) {
      y[d] = i % z[d];
      i /= z[d];
    }

    parity = i;

    // convert into the full-field lattice coordinate
    int oddBit = parity;
    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      for (int d = 1; d < nDim; d++) oddBit += y[d];
      oddBit = oddBit & 1;
    }
    y[0] = 2 * y[0] + oddBit; // compute the full x coordinate
  }

  /*
    Convert from n-dimensional spatial index to the 1-dimensional index.
    With full fields, we assume that the field is even-odd ordered.  The
    input lattice coordinates are always full-field coordinates.
  */
  void ColorSpinorField::OffsetIndex(int &i, int *y) const
  {
    int parity = 0;
    auto z = x;
    int savey0 = y[0];

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      for (int d = 0; d < nDim; d++) parity += y[d];
      parity = parity & 1;
      y[0] /= 2;
      z[0] /= 2;
    }

    i = parity;
    for (int d = nDim - 1; d >= 0; d--) {
      i = z[d] * i + y[d];
      // printf("z[%d]=%d y[%d]=%d ", d, z[d], d, y[d]);
    }
    // printf("\nparity = %d\n", parity);

    if (siteSubset == QUDA_FULL_SITE_SUBSET) y[0] = savey0;
  }

  FieldTmp<ColorSpinorField> ColorSpinorField::create_comms_batch(cvector_ref<const ColorSpinorField> &v)
  {
    // first create a dummy ndim+1 field
    if (v[0].Ndim() == 5) errorQuda("Cannot batch together 5-d fields");
    ColorSpinorParam param(v[0]);
    param.nDim++;
    param.x[param.nDim - 1] = v.size();
    param.create = QUDA_GHOST_FIELD_CREATE;

    // we use a custom cache key for ghost-only fields
    FieldKey<ColorSpinorField> key;
    key.volume = v[0].VolString();
    key.aux = v[0].AuxString();
    char aux[32];
    strcpy(aux, ",ghost_batch=");
    u32toa(aux + 13, v.size());
    key.aux += aux;

    return FieldTmp<ColorSpinorField>(key, param);
  }

  ColorSpinorField ColorSpinorField::create_alias(const ColorSpinorParam &param_)
  {
    if (param_.init && param_.Precision() > precision)
      errorQuda("Cannot create an alias to source with lower precision than the alias");
    ColorSpinorParam param = param_.init ? param_ : ColorSpinorParam(*this);
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.v = V();

    return ColorSpinorField(param);
  }

  ColorSpinorField *ColorSpinorField::CreateAlias(const ColorSpinorParam &param_)
  {
    if (param_.Precision() > precision)
      errorQuda("Cannot create an alias to source with lower precision than the alias");
    ColorSpinorParam param(param_);
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.v = V();

    return new ColorSpinorField(param);
  }

  ColorSpinorField *ColorSpinorField::CreateCoarse(const int *geoBlockSize, int spinBlockSize, int Nvec,
                                                   QudaPrecision new_precision, QudaFieldLocation new_location,
                                                   QudaMemoryType new_mem_type)
  {
    ColorSpinorParam coarseParam(*this);
    for (int d = 0; d < nDim; d++) coarseParam.x[d] = x[d] / geoBlockSize[d];

    int geoBlockVolume = 1;
    for (int d = 0; d < nDim; d++) { geoBlockVolume *= geoBlockSize[d]; }

    // Detect if the "coarse" op is the Kahler-Dirac op or something else
    // that still acts on a fine staggered ColorSpinorField
    if (geoBlockVolume == 1 && Nvec == nColor && nSpin == 1) {
      coarseParam.nSpin = nSpin;
      coarseParam.nColor = nColor;
    } else {
      coarseParam.nSpin = (nSpin == 1) ? 2 : (nSpin / spinBlockSize); // coarsening staggered check
      coarseParam.nColor = Nvec;
    }

    coarseParam.siteSubset = QUDA_FULL_SITE_SUBSET; // coarse grid is always full
    coarseParam.create = QUDA_ZERO_FIELD_CREATE;

    // if new precision is not set, use this->precision
    new_precision = (new_precision == QUDA_INVALID_PRECISION) ? Precision() : new_precision;

    // if new location is not set, use this->location
    new_location = (new_location == QUDA_INVALID_FIELD_LOCATION) ? Location() : new_location;

    coarseParam.fieldOrder = (new_location == QUDA_CUDA_FIELD_LOCATION) ?
      colorspinor::getNative(new_precision, coarseParam.nSpin) :
      QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    coarseParam.setPrecision(new_precision);

    // set where we allocate the field
    coarseParam.mem_type = (new_mem_type != QUDA_MEMORY_INVALID) ?
      new_mem_type :
      (new_location == QUDA_CUDA_FIELD_LOCATION ? QUDA_MEMORY_DEVICE : QUDA_MEMORY_PINNED);

    return new ColorSpinorField(coarseParam);
  }

  ColorSpinorField *ColorSpinorField::CreateFine(const int *geoBlockSize, int spinBlockSize, int Nvec,
                                                 QudaPrecision new_precision, QudaFieldLocation new_location,
                                                 QudaMemoryType new_mem_type)
  {
    ColorSpinorParam fineParam(*this);
    for (int d = 0; d < nDim; d++) fineParam.x[d] = x[d] * geoBlockSize[d];
    fineParam.nSpin = nSpin * spinBlockSize;
    fineParam.nColor = Nvec;
    fineParam.siteSubset = QUDA_FULL_SITE_SUBSET; // FIXME fine grid is always full
    fineParam.create = QUDA_ZERO_FIELD_CREATE;

    // if new precision is not set, use this->precision
    new_precision = (new_precision == QUDA_INVALID_PRECISION) ? Precision() : new_precision;

    // if new location is not set, use this->location
    new_location = (new_location == QUDA_INVALID_FIELD_LOCATION) ? Location() : new_location;

    // for GPU fields, always use native ordering to ensure coalescing
    if (new_location == QUDA_CUDA_FIELD_LOCATION) {
      fineParam.setPrecision(new_precision, new_precision, true);
    } else {
      fineParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      fineParam.setPrecision(new_precision);
    }

    // set where we allocate the field
    fineParam.mem_type = (new_mem_type != QUDA_MEMORY_INVALID) ?
      new_mem_type :
      (new_location == QUDA_CUDA_FIELD_LOCATION ? QUDA_MEMORY_DEVICE : QUDA_MEMORY_PINNED);

    return new ColorSpinorField(fineParam);
  }

  // legacy CPU static ghost destructor
  void ColorSpinorField::freeGhostBuffer(void)
  {
    if (!initGhostFaceBuffer) return;

    for (int i = 0; i < 4; i++) { // make nDimComms static?
      host_free(fwdGhostFaceBuffer[i]);
      fwdGhostFaceBuffer[i] = NULL;
      host_free(backGhostFaceBuffer[i]);
      backGhostFaceBuffer[i] = NULL;
      host_free(fwdGhostFaceSendBuffer[i]);
      fwdGhostFaceSendBuffer[i] = NULL;
      host_free(backGhostFaceSendBuffer[i]);
      backGhostFaceSendBuffer[i] = NULL;
    }
    initGhostFaceBuffer = 0;
  }

  void ColorSpinorField::allocateGhostBuffer(int nFace, bool spin_project) const
  {
    createGhostZone(nFace, spin_project);
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      if (spin_project) errorQuda("Not yet implemented");

      int spinor_size = 2 * nSpin * nColor * precision;
      bool resize = false;

      // resize face only if requested size is larger than previously allocated one
      for (int i = 0; i < nDimComms; i++) {
        size_t nbytes = siteSubset * nFace * surfaceCB[i] * spinor_size;
        resize = (nbytes > ghostFaceBytes[i]) ? true : resize;
        ghostFaceBytes[i] = (nbytes > ghostFaceBytes[i]) ? nbytes : ghostFaceBytes[i];
      }

      if (!initGhostFaceBuffer || resize) {
        freeGhostBuffer();
        for (int i = 0; i < nDimComms; i++) {
          fwdGhostFaceBuffer[i] = safe_malloc(ghostFaceBytes[i]);
          backGhostFaceBuffer[i] = safe_malloc(ghostFaceBytes[i]);
          fwdGhostFaceSendBuffer[i] = safe_malloc(ghostFaceBytes[i]);
          backGhostFaceSendBuffer[i] = safe_malloc(ghostFaceBytes[i]);
        }
        initGhostFaceBuffer = 1;
      }
    } else {
      LatticeField::allocateGhostBuffer(ghost_bytes);
    }
  }

  void ColorSpinorField::createComms(int nFace, bool spin_project)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    allocateGhostBuffer(nFace, spin_project); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs its comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face_h[0] != ghost_pinned_send_buffer_h[0]) || (my_face_h[1] != ghost_pinned_send_buffer_h[1])
      || (from_face_h[0] != ghost_pinned_recv_buffer_h[0]) || (from_face_h[1] != ghost_pinned_recv_buffer_h[1])
      || (my_face_d[0] != ghost_send_buffer_d[0]) || (my_face_d[1] != ghost_send_buffer_d[1]) ||  // send buffers
      (from_face_d[0] != ghost_recv_buffer_d[0]) || (from_face_d[1] != ghost_recv_buffer_d[1]) || // receive buffers
      ghost_precision_reset; // ghost_precision has changed

    if (!initComms || comms_reset) {

      LatticeField::createComms();

      // reinitialize the ghost receive pointers
      for (int i = 0; i < nDimComms; ++i) {
        if (commDimPartitioned(i)) {
          for (int b = 0; b < 2; b++) {
            ghost[b][i] = static_cast<char *>(ghost_recv_buffer_d[b]) + ghost_offset[i][0];
          }
        }
      }

      ghost_precision_reset = false;
    }

    if (ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  // pack the ghost zone into a contiguous buffer for communications
  void ColorSpinorField::packGhost(const int nFace, const QudaParity parity, const int dagger, const qudaStream_t &stream,
                                   MemoryLocation location[2 * QUDA_MAX_DIM], MemoryLocation location_label,
                                   bool spin_project, double a, double b, double c, int shmem)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    void *packBuffer[4 * QUDA_MAX_DIM] = {};

    for (int dim = 0; dim < 4; dim++) {
      for (int dir = 0; dir < 2; dir++) {
        switch (location[2 * dim + dir]) {

        case Device: // pack to local device buffer
          packBuffer[2 * dim + dir] = my_face_dim_dir_d[bufferIndex][dim][dir];
          packBuffer[2 * QUDA_MAX_DIM + 2 * dim + dir] = nullptr;
          break;
        case Shmem:
          // this is the remote buffer when using shmem ...
          // if the ghost_remote_send_buffer_d exists we can directly use it
          // - else we need pack locally and send data to the recv buffer
          packBuffer[2 * dim + dir] = ghost_remote_send_buffer_d[bufferIndex][dim][dir] != nullptr ?
            static_cast<char *>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + ghost_offset[dim][1 - dir] :
            my_face_dim_dir_d[bufferIndex][dim][dir];
          packBuffer[2 * QUDA_MAX_DIM + 2 * dim + dir] = ghost_remote_send_buffer_d[bufferIndex][dim][dir] != nullptr ?
            nullptr :
            static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + ghost_offset[dim][1 - dir];
          break;
        case Host: // pack to zero-copy memory
          packBuffer[2 * dim + dir] = my_face_dim_dir_hd[bufferIndex][dim][dir];
          break;
        case Remote: // pack to remote peer memory
          packBuffer[2 * dim + dir]
            = static_cast<char *>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + ghost_offset[dim][1 - dir];
          break;
        default: errorQuda("Undefined location %d", location[2 * dim + dir]);
        }
      }
    }

    PackGhost(packBuffer, *this, location_label, nFace, dagger, parity, spin_project, a, b, c, shmem, stream);
  }

  void ColorSpinorField::pack(int nFace, int parity, int dagger, const qudaStream_t &stream,
                              MemoryLocation location[2 * QUDA_MAX_DIM], MemoryLocation location_label,
                              bool spin_project, double a, double b, double c, int shmem)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    createComms(nFace, spin_project); // must call this first

    packGhost(nFace, (QudaParity)parity, dagger, stream, location, location_label, spin_project, a, b, c, shmem);
  }

  void ColorSpinorField::sendGhost(void *ghost_spinor, const int dim, const QudaDirection dir, const qudaStream_t &stream)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    void *gpu_buf
      = (dir == QUDA_BACKWARDS) ? my_face_dim_dir_d[bufferIndex][dim][0] : my_face_dim_dir_d[bufferIndex][dim][1];
    qudaMemcpyAsync(ghost_spinor, gpu_buf, ghost_face_bytes[dim], qudaMemcpyDeviceToHost, stream);
  }

  void ColorSpinorField::unpackGhost(const void *ghost_spinor, const int dim, const QudaDirection dir,
                                     const qudaStream_t &stream)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    const void *src = ghost_spinor;
    auto offset = (dir == QUDA_BACKWARDS) ? ghost_offset[dim][0] : ghost_offset[dim][1];
    void *ghost_dst = static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + offset;

    qudaMemcpyAsync(ghost_dst, src, ghost_face_bytes[dim], qudaMemcpyHostToDevice, stream);
  }

  void ColorSpinorField::gather(int dir, const qudaStream_t &stream)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    int dim = dir / 2;

    if (dir % 2 == 0) {
      // backwards copy to host
      if (comm_peer2peer_enabled(0, dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][0], dim, QUDA_BACKWARDS, stream);
    } else {
      // forwards copy to host
      if (comm_peer2peer_enabled(1, dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][1], dim, QUDA_FORWARDS, stream);
    }
  }

  void ColorSpinorField::recvStart(int d, const qudaStream_t &, bool gdr)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d / 2;
    int dir = d % 2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (comm_peer2peer_enabled(1 - dir, dim)) {
      comm_start(mh_recv_p2p[bufferIndex][dim][1 - dir]);
    } else if (gdr) {
      comm_start(mh_recv_rdma[bufferIndex][dim][1 - dir]);
    } else {
      comm_start(mh_recv[bufferIndex][dim][1 - dir]);
    }
  }

  void ColorSpinorField::sendStart(int d, const qudaStream_t &stream, bool gdr, bool remote_write)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d / 2;
    int dir = d % 2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (!comm_peer2peer_enabled(dir, dim)) {
      if (gdr)
        comm_start(mh_send_rdma[bufferIndex][dim][dir]);
      else
        comm_start(mh_send[bufferIndex][dim][dir]);
    } else { // doing peer-to-peer

      // if not using copy engine then the packing kernel will remotely write the halos
      if (!remote_write) {
        // all goes here
        void *ghost_dst
          = static_cast<char *>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + ghost_offset[dim][(dir + 1) % 2];

        qudaMemcpyP2PAsync(ghost_dst, my_face_dim_dir_d[bufferIndex][dim][dir], ghost_face_bytes[dim], stream);
      } // remote_write

        // record the event
      qudaEventRecord(ipcCopyEvent[bufferIndex][dim][dir], stream);
      // send to the processor in the -1 direction
      comm_start(mh_send_p2p[bufferIndex][dim][dir]);
    }
  }

  void ColorSpinorField::commsStart(int dir, const qudaStream_t &stream, bool gdr_send, bool gdr_recv)
  {
    recvStart(dir, stream, gdr_recv);
    sendStart(dir, stream, gdr_send);
  }

  static bool complete_recv[QUDA_MAX_DIM][2] = {};
  static bool complete_send[QUDA_MAX_DIM][2] = {};

  int ColorSpinorField::commsQuery(int d, const qudaStream_t &, bool gdr_send, bool gdr_recv)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d / 2;
    int dir = d % 2;

    if (!commDimPartitioned(dim)) return 1;
    if ((gdr_send || gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    // first query send to backwards
    if (comm_peer2peer_enabled(dir, dim)) {
      if (!complete_send[dim][dir]) complete_send[dim][dir] = comm_query(mh_send_p2p[bufferIndex][dim][dir]);
    } else if (gdr_send) {
      if (!complete_send[dim][dir]) complete_send[dim][dir] = comm_query(mh_send_rdma[bufferIndex][dim][dir]);
    } else {
      if (!complete_send[dim][dir]) complete_send[dim][dir] = comm_query(mh_send[bufferIndex][dim][dir]);
    }

    // second query receive from forwards
    if (comm_peer2peer_enabled(1 - dir, dim)) {
      if (!complete_recv[dim][1 - dir])
        complete_recv[dim][1 - dir] = comm_query(mh_recv_p2p[bufferIndex][dim][1 - dir]);
    } else if (gdr_recv) {
      if (!complete_recv[dim][1 - dir])
        complete_recv[dim][1 - dir] = comm_query(mh_recv_rdma[bufferIndex][dim][1 - dir]);
    } else {
      if (!complete_recv[dim][1 - dir]) complete_recv[dim][1 - dir] = comm_query(mh_recv[bufferIndex][dim][1 - dir]);
    }

    if (complete_recv[dim][1 - dir] && complete_send[dim][dir]) {
      complete_send[dim][dir] = false;
      complete_recv[dim][1 - dir] = false;
      return 1;
    } else {
      return 0;
    }
  }

  void ColorSpinorField::commsWait(int d, const qudaStream_t &, bool gdr_send, bool gdr_recv)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d / 2;
    int dir = d % 2;

    if (!commDimPartitioned(dim)) return;
    if ((gdr_send && gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    // first wait on send to "dir"
    if (comm_peer2peer_enabled(dir, dim)) {
      comm_wait(mh_send_p2p[bufferIndex][dim][dir]);
      qudaEventSynchronize(ipcCopyEvent[bufferIndex][dim][dir]);
    } else if (gdr_send) {
      comm_wait(mh_send_rdma[bufferIndex][dim][dir]);
    } else {
      comm_wait(mh_send[bufferIndex][dim][dir]);
    }

    // second wait on receive from "1 - dir"
    if (comm_peer2peer_enabled(1 - dir, dim)) {
      comm_wait(mh_recv_p2p[bufferIndex][dim][1 - dir]);
      qudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][dim][1 - dir]);
    } else if (gdr_recv) {
      comm_wait(mh_recv_rdma[bufferIndex][dim][1 - dir]);
    } else {
      comm_wait(mh_recv[bufferIndex][dim][1 - dir]);
    }
  }

  void ColorSpinorField::scatter(int dim_dir, const qudaStream_t &stream)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so input expects dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards), so here we need flip to receive centric

    int dim = dim_dir / 2;
    int dir = (dim_dir + 1) % 2; // dir = 1 - receive from forwards, dir == 0 recive from backwards
    if (!commDimPartitioned(dim)) return;
    if (comm_peer2peer_enabled(dir, dim)) return;

    unpackGhost(from_face_dim_dir_h[bufferIndex][dim][dir], dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, stream);
  }

  void ColorSpinorField::exchangeGhost(QudaParity parity, int nFace, int dagger,
                                       const MemoryLocation *pack_destination_, const MemoryLocation *halo_location_,
                                       bool gdr_send, bool gdr_recv, QudaPrecision ghost_precision_, int shmem,
                                       cvector_ref<const ColorSpinorField> v) const
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      // allocate ghost buffer if not yet allocated
      allocateGhostBuffer(nFace, false);

      void **sendbuf = static_cast<void **>(safe_malloc(nDimComms * 2 * sizeof(void *)));

      for (int i = 0; i < nDimComms; i++) {
        sendbuf[2 * i + 0] = backGhostFaceSendBuffer[i];
        sendbuf[2 * i + 1] = fwdGhostFaceSendBuffer[i];
        ghost_buf[2 * i + 0] = backGhostFaceBuffer[i];
        ghost_buf[2 * i + 1] = fwdGhostFaceBuffer[i];
      }

      genericPackGhost(sendbuf, *this, parity, nFace, dagger, nullptr, 0, v);
      exchange(ghost_buf.data, sendbuf, nFace);

      host_free(sendbuf);
    } else {
      bool shmem_async = shmem & 2;
      // first set default values to device if needed
      MemoryLocation pack_destination[2 * QUDA_MAX_DIM], halo_location[2 * QUDA_MAX_DIM];
      for (int i = 0; i < 2 * nDimComms; i++) {
        pack_destination[i] = pack_destination_ ? pack_destination_[i] : (comm_nvshmem_enabled() ? Shmem : Device);
        halo_location[i] = halo_location_ ? halo_location_[i] : Device;
      }
      // we are overriding the ghost precision, and it doesn't match what has already been allocated
      if (ghost_precision_ != QUDA_INVALID_PRECISION && ghost_precision != ghost_precision_) {
        ghost_precision_reset = true;
        ghost_precision = ghost_precision_;
      }

      // not overriding the ghost precision, but we did previously so need to update
      if (ghost_precision == QUDA_INVALID_PRECISION && ghost_precision != precision) {
        ghost_precision_reset = true;
        ghost_precision = precision;
      }

      if ((gdr_send || gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");
      const_cast<ColorSpinorField &>(*this).createComms(nFace, false);

      if (pack_destination[0] != Shmem) {

        // Contiguous send buffers and we aggregate copies to reduce
        // latency.  Only if all locations are "Device" and no p2p
        bool fused_pack_memcpy = true;

        // Contiguous recv buffers and we aggregate copies to reduce
        // latency.  Only if all locations are "Device" and no p2p
        bool fused_halo_memcpy = true;

        bool pack_host = false; // set to true if any of the ghost packing is being done to Host memory
        bool halo_host = false; // set to true if the final halos will be left in Host memory

        void *send[4 * QUDA_MAX_DIM];
        for (int d = 0; d < nDimComms; d++) {
          for (int dir = 0; dir < 2; dir++) {
            send[2 * d + dir] = pack_destination[2 * d + dir] == Host ? my_face_dim_dir_hd[bufferIndex][d][dir] :
                                                                        my_face_dim_dir_d[bufferIndex][d][dir];
            send[2 * QUDA_MAX_DIM + 2 * d + dir] = nullptr;
            ghost_buf[2 * d + dir] = halo_location[2 * d + dir] == Host ? from_face_dim_dir_hd[bufferIndex][d][dir] :
                                                                          from_face_dim_dir_d[bufferIndex][d][dir];
          }

          // if doing p2p, then we must pack to and load the halo from device memory
          for (int dir = 0; dir < 2; dir++) {
            if (comm_peer2peer_enabled(dir, d)) {
              pack_destination[2 * d + dir] = Device;
              halo_location[2 * d + 1 - dir] = Device;
            }
          }

          // if zero-copy packing or p2p is enabled then we cannot do fused memcpy
          if (pack_destination[2 * d + 0] != Device || pack_destination[2 * d + 1] != Device
              || comm_peer2peer_enabled_global())
            fused_pack_memcpy = false;
          // if zero-copy halo read or p2p is enabled then we cannot do fused memcpy
          if (halo_location[2 * d + 0] != Device || halo_location[2 * d + 1] != Device || comm_peer2peer_enabled_global())
            fused_halo_memcpy = false;

          if (pack_destination[2 * d + 0] == Host || pack_destination[2 * d + 1] == Host) pack_host = true;
          if (halo_location[2 * d + 0] == Host || halo_location[2 * d + 1] == Host) halo_host = true;
        }

        // Error if zero-copy and p2p for now
        if ((pack_host || halo_host) && comm_peer2peer_enabled_global())
          errorQuda("Cannot use zero-copy memory with peer-to-peer comms yet");

        genericPackGhost(send, *this, parity, nFace, dagger, pack_destination, 0,
                         v); // FIXME - need support for asymmetric topologies

        size_t total_bytes = 0;
        for (int i = 0; i < nDimComms; i++)
          if (comm_dim_partitioned(i)) total_bytes += 2 * ghost_face_bytes_aligned[i]; // 2 for fwd/bwd

        if (!gdr_send) {
          if (!fused_pack_memcpy) {
            for (int i = 0; i < nDimComms; i++) {
              if (comm_dim_partitioned(i)) {
                if (pack_destination[2 * i + 0] == Device && !comm_peer2peer_enabled(0, i)
                    && // fuse forwards and backwards if possible
                    pack_destination[2 * i + 1] == Device && !comm_peer2peer_enabled(1, i)) {
                  qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][i][0], my_face_dim_dir_d[bufferIndex][i][0],
                                  2 * ghost_face_bytes_aligned[i], qudaMemcpyDeviceToHost, device::get_default_stream());
                } else {
                  if (pack_destination[2 * i + 0] == Device && !comm_peer2peer_enabled(0, i))
                    qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][i][0], my_face_dim_dir_d[bufferIndex][i][0],
                                    ghost_face_bytes[i], qudaMemcpyDeviceToHost, device::get_default_stream());
                  if (pack_destination[2 * i + 1] == Device && !comm_peer2peer_enabled(1, i))
                    qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][i][1], my_face_dim_dir_d[bufferIndex][i][1],
                                    ghost_face_bytes[i], qudaMemcpyDeviceToHost, device::get_default_stream());
                }
              }
            }
          } else if (total_bytes && !pack_host) {
            qudaMemcpyAsync(my_face_h[bufferIndex], ghost_send_buffer_d[bufferIndex], total_bytes,
                            qudaMemcpyDeviceToHost, device::get_default_stream());
          }
        }

        // prepost receive
        for (int i = 0; i < 2 * nDimComms; i++)
          const_cast<ColorSpinorField *>(this)->recvStart(i, device::get_default_stream(), gdr_recv);

        // FIXME use events to properly synchronize streams, logic below failed when using p2p in all 4 dimensions (DGX2)
        bool sync = true;
        /* bool sync = true; pack_host ? true : false; // no p2p if pack_host so we need to synchronize
        // if not p2p in any direction then need to synchronize before MPI
        for (int i = 0; i < nDimComms; i++)
          if (!comm_peer2peer_enabled(0, i) || !comm_peer2peer_enabled(1, i)) sync = true;
        */
        if (sync)
          qudaDeviceSynchronize(); // need to make sure packing and/or memcpy has finished before kicking off MPI

        for (int p2p = 0; p2p < 2; p2p++) {
          for (int dim = 0; dim < nDimComms; dim++) {
            for (int dir = 0; dir < 2; dir++) {
              if ((comm_peer2peer_enabled(dir, dim) + p2p) % 2 == 0) { // issue non-p2p transfers first
                const_cast<ColorSpinorField *>(this)->sendStart(2 * dim + dir, device::get_stream(2 * dim + dir),
                                                                gdr_send);
              }
            }
          }
        }

        bool comms_complete[2 * QUDA_MAX_DIM] = {};
        int comms_done = 0;
        while (comms_done < 2 * nDimComms) { // non-blocking query of each exchange and exit once all have completed
          for (int dim = 0; dim < nDimComms; dim++) {
            for (int dir = 0; dir < 2; dir++) {
              if (!comms_complete[dim * 2 + dir]) {
                comms_complete[2 * dim + dir] = const_cast<ColorSpinorField *>(this)->commsQuery(
                  2 * dim + dir, device::get_default_stream(), gdr_send, gdr_recv);
                if (comms_complete[2 * dim + dir]) {
                  comms_done++;
                  if (comm_peer2peer_enabled(1 - dir, dim))
                    qudaStreamWaitEvent(device::get_default_stream(), ipcRemoteCopyEvent[bufferIndex][dim][1 - dir], 0);
                }
              }
            }
          }
        }

        if (!gdr_recv) {
          if (!fused_halo_memcpy) {
            for (int i = 0; i < nDimComms; i++) {
              if (comm_dim_partitioned(i)) {
                if (halo_location[2 * i + 0] == Device && !comm_peer2peer_enabled(0, i)
                    && // fuse forwards and backwards if possible
                    halo_location[2 * i + 1] == Device && !comm_peer2peer_enabled(1, i)) {
                  qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][i][0], from_face_dim_dir_h[bufferIndex][i][0],
                                  2 * ghost_face_bytes_aligned[i], qudaMemcpyHostToDevice, device::get_default_stream());
                } else {
                  if (halo_location[2 * i + 0] == Device && !comm_peer2peer_enabled(0, i))
                    qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][i][0], from_face_dim_dir_h[bufferIndex][i][0],
                                    ghost_face_bytes[i], qudaMemcpyHostToDevice, device::get_default_stream());
                  if (halo_location[2 * i + 1] == Device && !comm_peer2peer_enabled(1, i))
                    qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][i][1], from_face_dim_dir_h[bufferIndex][i][1],
                                    ghost_face_bytes[i], qudaMemcpyHostToDevice, device::get_default_stream());
                }
              }
            }
          } else if (total_bytes && !halo_host) {
            qudaMemcpyAsync(ghost_recv_buffer_d[bufferIndex], from_face_h[bufferIndex], total_bytes,
                            qudaMemcpyHostToDevice, device::get_default_stream());
          }
        }

        // ensure that the p2p sending is completed before returning
        for (int dim = 0; dim < nDimComms; dim++) {
          if (!comm_dim_partitioned(dim)) continue;
          for (int dir = 0; dir < 2; dir++) {
            if (comm_peer2peer_enabled(dir, dim))
              qudaStreamWaitEvent(device::get_default_stream(), ipcCopyEvent[bufferIndex][dim][dir], 0);
          }
        }
      } else {
        void *packBuffer[4 * QUDA_MAX_DIM];
        shmem = (shmem | 1);
        if (!shmem_async) shmem = (shmem | 4);
        for (int dim = 0; dim < nDimComms; dim++) {
          for (int dir = 0; dir < 2; dir++) {
            switch (pack_destination[2 * dim + dir]) {
            case Shmem: {
              bool intranode = ghost_remote_send_buffer_d[bufferIndex][dim][dir] != nullptr;
              packBuffer[2 * dim + dir] = intranode ?
                static_cast<char *>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + ghost_offset[dim][1 - dir] :
                my_face_dim_dir_d[bufferIndex][dim][dir];
              packBuffer[2 * QUDA_MAX_DIM + 2 * dim + dir] = intranode ?
                nullptr :
                static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + ghost_offset[dim][1 - dir];
              ghost_buf[2 * dim + dir] = from_face_dim_dir_d[bufferIndex][dim][dir];
            } break;
            case Host:
            case Device:
            default: errorQuda("Undefined / unexpected pack_destination %d", pack_destination[2 * dim + dir]);
            }
          }
        }
        genericPackGhost(packBuffer, *this, parity, nFace, dagger, pack_destination, shmem,
                         v); // FIXME - need support for asymmetric topologies
      }
    }
  }

  void ColorSpinorField::backup() const
  {
    if (backed_up) errorQuda("ColorSpinorField already backed up");

    backup_h = new char[bytes];
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(backup_h, v, bytes, qudaMemcpyDefault);
    } else {
      memcpy(backup_h, v, bytes);
    }

    backed_up = true;
  }

  void ColorSpinorField::restore() const
  {
    if (!backed_up) errorQuda("Cannot restore since not backed up");

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(v, backup_h, bytes, qudaMemcpyDefault);
      delete[] backup_h;
    } else {
      memcpy(v, backup_h, bytes);
      delete[] backup_h;
    }

    backed_up = false;
  }

  void ColorSpinorField::copy_to_buffer(void *buffer) const
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(buffer, v, bytes, qudaMemcpyDeviceToHost);
    } else {
      std::memcpy(buffer, v, bytes);
    }
  }

  void ColorSpinorField::copy_from_buffer(void *buffer)
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(v, buffer, bytes, qudaMemcpyHostToDevice);
    } else {
      std::memcpy(v, buffer, bytes);
    }
  }

  void ColorSpinorField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      // conditionals based on destructor
      if (is_prefetch_enabled() && alloc && mem_type == QUDA_MEMORY_DEVICE)
        qudaMemPrefetchAsync(v, bytes, mem_space, stream);
    }
  }

  void ColorSpinorField::Source(QudaSourceType source_type, unsigned int x, int s, int c)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      genericSource(*this, source_type, x, s, c);
    } else {
      ColorSpinorParam param(*this);
      param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      param.location = QUDA_CPU_FIELD_LOCATION;
      param.setPrecision((param.Precision() == QUDA_HALF_PRECISION || param.Precision() == QUDA_QUARTER_PRECISION) ?
                           QUDA_SINGLE_PRECISION :
                           param.Precision());
      param.create = (source_type == QUDA_POINT_SOURCE ? QUDA_ZERO_FIELD_CREATE : QUDA_NULL_FIELD_CREATE);

      // since CPU fields cannot be low precision, use single precision instead
      if (precision < QUDA_SINGLE_PRECISION) param.setPrecision(QUDA_SINGLE_PRECISION, QUDA_INVALID_PRECISION, false);

      ColorSpinorField tmp(param);
      tmp.Source(source_type, x, s, c);
      *this = tmp;
    }
  }

  void ColorSpinorField::PrintVector(int parity, unsigned int x_cb, int rank) const
  {
    genericPrintVector(*this, parity, x_cb, rank);
  }

  int ColorSpinorField::Compare(const ColorSpinorField &a, const ColorSpinorField &b, const int tol)
  {
    if (checkLocation(a, b) == QUDA_CUDA_FIELD_LOCATION) errorQuda("device field not implemented");
    test_compatible_weak(a, b);
    return genericCompare(a, b, tol);
  }

  std::ostream &operator<<(std::ostream &out, const ColorSpinorField &a)
  {
    out << "location = " << a.Location() << std::endl;
    out << "v = " << a.v << std::endl;
    out << "alloc = " << a.alloc << std::endl;
    out << "reference = " << a.reference << std::endl;
    out << "init = " << a.init << std::endl;
    out << "nColor = " << a.nColor << std::endl;
    out << "nSpin = " << a.nSpin << std::endl;
    out << "twistFlavor = " << a.twistFlavor << std::endl;
    out << "nDim = " << a.nDim << std::endl;
    for (int d = 0; d < a.nDim; d++) out << "x[" << d << "] = " << a.x[d] << std::endl;
    out << "volume = " << a.volume << std::endl;
    out << "pc_type = " << a.pc_type << std::endl;
    out << "suggested_parity = " << a.suggested_parity << std::endl;
    out << "precision = " << a.precision << std::endl;
    out << "ghost_precision = " << a.ghost_precision << std::endl;
    out << "length = " << a.length << std::endl;
    out << "bytes = " << a.bytes << std::endl;
    out << "siteSubset = " << a.siteSubset << std::endl;
    out << "siteOrder = " << a.siteOrder << std::endl;
    out << "fieldOrder = " << a.fieldOrder << std::endl;
    out << "gammaBasis = " << a.gammaBasis << std::endl;
    out << "Is composite = " << a.composite_descr.is_composite << std::endl;
    if (a.composite_descr.is_composite) {
      out << "Composite Dim = " << a.composite_descr.dim << std::endl;
      out << "Composite Volume = " << a.composite_descr.volume << std::endl;
      out << "Composite Length = " << a.composite_descr.length << std::endl;
    }
    out << "Is component = " << a.composite_descr.is_component << std::endl;
    if (a.composite_descr.is_composite) out << "Component ID = " << a.composite_descr.id << std::endl;
    out << "pc_type = " << a.pc_type << std::endl;
    return out;
  }

} // namespace quda
