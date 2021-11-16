#include <string.h>
#include <iostream>
#include <typeinfo>

#include <color_spinor_field.h>
#include <dslash_quda.h>

static bool zeroCopy = false;

namespace quda {

  /*ColorSpinorField::ColorSpinorField() : init(false) {

    }*/

  ColorSpinorParam::ColorSpinorParam(const ColorSpinorField &field) : LatticeFieldParam()  {
    field.fill(*this);
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param, QudaFieldLocation location_) :
    LatticeField(param),
    init(false),
    alloc(false),
    reference(false),
    ghost_precision_allocated(QUDA_INVALID_PRECISION),
    v(0),
    norm(0),
    ghost(),
    ghostNorm(),
    ghostFace(),
    dslash_constant(static_cast<DslashConstant *>(safe_malloc(sizeof(DslashConstant)))),
    bytes(0),
    norm_bytes(0),
    location(param.location),
    even(0),
    odd(0),
    composite_descr(param.is_composite, param.composite_dim, param.is_component, param.component_id),
    components(0)
  {
    if (param.location != location_) errorQuda("location bork %d != %d", param.location, location_);

    // this must come before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      norm = param.norm;
      reference = true;
    }

    if (param.create == QUDA_INVALID_FIELD_CREATE) errorQuda("Invalid create type");
    for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) ghost_buf[i] = nullptr;
    create(param.nDim, param.x, param.nColor, param.nSpin, param.nVec, param.twistFlavor, param.Precision(), param.pad,
           param.siteSubset, param.siteOrder, param.fieldOrder, param.gammaBasis, param.pc_type, param.suggested_parity);

    create2(param.create);

    switch (param.create) {
    case QUDA_NULL_FIELD_CREATE:
    case QUDA_REFERENCE_FIELD_CREATE: break; // do nothing;
    case QUDA_ZERO_FIELD_CREATE: zero(); break;
    case QUDA_COPY_FIELD_CREATE: copy(*param.field); break;
    default: errorQuda("Unexpected create type %d", param.create);
    }
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorField &field) :
    LatticeField(field),
    init(false),
    alloc(false),
    reference(false),
    ghost_precision_allocated(QUDA_INVALID_PRECISION),
    v(0),
    norm(0),
    ghost(),
    ghostNorm(),
    ghostFace(),
    dslash_constant(static_cast<DslashConstant *>(safe_malloc(sizeof(DslashConstant)))),
    bytes(0),
    norm_bytes(0),
    location(field.Location()),
    even(0),
    odd(0),
    composite_descr(field.composite_descr),
    components(0)
  {
    for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) ghost_buf[i] = nullptr;
    create(field.nDim, field.x, field.nColor, field.nSpin, field.nVec, field.twistFlavor, field.Precision(), field.pad,
           field.siteSubset, field.siteOrder, field.fieldOrder, field.gammaBasis, field.pc_type, field.suggested_parity);

    create2(QUDA_COPY_FIELD_CREATE);
    copy(field);
  }

  ColorSpinorField::~ColorSpinorField() {
    if (dslash_constant) host_free(dslash_constant);
    destroy();
    destroy2();
    if (Location() == QUDA_CUDA_FIELD_LOCATION) destroyComms();
  }

  void ColorSpinorField::create(int Ndim, const int *X, int Nc, int Ns, int Nvec, QudaTwistFlavorType Twistflavor,
                                QudaPrecision Prec, int Pad, QudaSiteSubset siteSubset, QudaSiteOrder siteOrder,
                                QudaFieldOrder fieldOrder, QudaGammaBasis gammaBasis, QudaPCType pc_type,
                                QudaParity suggested_parity)
  {
    this->siteSubset = siteSubset;
    this->siteOrder = siteOrder;
    this->fieldOrder = fieldOrder;
    this->gammaBasis = gammaBasis;

    if (Ndim > QUDA_MAX_DIM) errorQuda("Number of dimensions nDim = %d too great", Ndim);
    nDim = Ndim;
    nColor = Nc;
    nSpin = Ns;
    nVec = Nvec;
    twistFlavor = Twistflavor;

    if (pc_type != QUDA_5D_PC && pc_type != QUDA_4D_PC) errorQuda("Unexpected pc_type %d", pc_type);
    this->pc_type = pc_type;
    this->suggested_parity = suggested_parity;

    precision = Prec;
    // Copy all data in X
    for (int d = 0; d < QUDA_MAX_DIM; d++) x[d] = X[d];
    volume = 1;
    for (int d=0; d<nDim; d++) volume *= x[d];
    volumeCB = siteSubset == QUDA_PARITY_SITE_SUBSET ? volume : volume/2;

   if ((twistFlavor == QUDA_TWIST_NONDEG_DOUBLET || twistFlavor == QUDA_TWIST_DEG_DOUBLET) && x[4] != 2) //two flavors
     errorQuda("Must be two flavors for non-degenerate twisted mass spinor (while provided with %d number of components)\n", x[4]);

    pad = Pad;
    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      stride = volume/2 + pad; // padding is based on half volume
      length = 2*stride*nColor*nSpin*2;
    } else {
      stride = volume + pad;
      length = stride*nColor*nSpin*2;
    }

    real_length = volume*nColor*nSpin*2; // physical length

    bytes = (size_t)length * precision; // includes pads and ghost zones
    if (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER)
      bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);

    if (precision < QUDA_SINGLE_PRECISION) {
      norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET ? 2*stride : stride) * sizeof(float);
      if (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER)
        norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(norm_bytes/2) : ALIGNMENT_ADJUST(norm_bytes);
    } else {
      norm_bytes = 0;
    }

    init = true;

    //! stuff for deflated solvers (eigenvector sets):
    if (composite_descr.is_composite) {
      if (composite_descr.is_component) errorQuda("Composite type is not implemented");

      composite_descr.volume   = volume;
      composite_descr.volumeCB = volumeCB;
      composite_descr.stride = stride;
      composite_descr.length = length;
      composite_descr.real_length = real_length;
      composite_descr.bytes       = bytes;
      composite_descr.norm_bytes  = norm_bytes;

      volume *= composite_descr.dim;
      volumeCB *= composite_descr.dim;
      stride *= composite_descr.dim;
      length *= composite_descr.dim;
      real_length *= composite_descr.dim;

      bytes *= composite_descr.dim;
      norm_bytes *= composite_descr.dim;
    } else if (composite_descr.is_component) {
      composite_descr.dim         = 0;
      composite_descr.volume      = 0;
      composite_descr.volumeCB    = 0;
      composite_descr.stride      = 0;
      composite_descr.length      = 0;
      composite_descr.real_length = 0;
      composite_descr.bytes       = 0;
      composite_descr.norm_bytes  = 0;
    }

    setTuningString();
  }

  void ColorSpinorField::create2(const QudaFieldCreate create)
  {
    if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER) errorQuda("Subset not implemented");

    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      // these need to be reset to ensure no ghost zones for the cpu
      // fields since we can't determine during the parent's constructor
      // whether the field is a cpu or cuda field

      // set this again here.  this is a hack since we can determine we
      // have a cpu or cuda field in ColorSpinorField::create(), which
      // means a ghost zone is set.  So we unset it here.  This will be
      // fixed when clean up the ghost code with the peer-2-peer branch
      bytes = length * precision;
      if (isNative()) bytes = siteSubset == QUDA_FULL_SITE_SUBSET ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        v = safe_malloc(bytes);
      } else {
        switch(mem_type) {
        case QUDA_MEMORY_DEVICE:
          v = pool_device_malloc(bytes);
          if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) norm = pool_device_malloc(norm_bytes);
          break;
        case QUDA_MEMORY_MAPPED:
          v_h = mapped_malloc(bytes);
          v = get_mapped_device_pointer(v_h);
          if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
            norm_h = mapped_malloc(norm_bytes);
            norm = get_mapped_device_pointer(norm_h); // set the matching device pointer
          }
          break;
        default: errorQuda("Unsupported memory type %d", mem_type);
        }
      }
      alloc = true;
    }

    if (composite_descr.is_composite && create != QUDA_REFERENCE_FIELD_CREATE) {
      ColorSpinorParam param;
      fill(param);
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      param.is_composite   = false;
      param.composite_dim  = 0;
      param.is_component = true;

      components.reserve(composite_descr.dim);
      for (int cid = 0; cid < composite_descr.dim; cid++) {
        param.component_id = cid;
        param.v = static_cast<void*>(static_cast<char*>(v) + cid * bytes / composite_descr.dim);
        param.norm = static_cast<void*>(static_cast<char*>(norm) + cid * norm_bytes / composite_descr.dim);
        components.push_back(ColorSpinorField::Create(param));
      }
    }

    // create the associated even and odd subsets
    if (siteSubset == QUDA_FULL_SITE_SUBSET && fieldOrder != QUDA_QDPJIT_FIELD_ORDER && !composite_descr.is_composite) {
      ColorSpinorParam param;
      fill(param);
      param.create = QUDA_REFERENCE_FIELD_CREATE;
      param.siteSubset = QUDA_PARITY_SITE_SUBSET;
      param.x[0] /= 2; // set single parity dimensions
      param.is_composite  = false;
      param.composite_dim = 0;
      param.is_component  = composite_descr.is_component;
      param.component_id  = composite_descr.id;
      even = ColorSpinorField::Create(param);
      param.v = static_cast<char*>(v) + bytes / 2;
      param.norm = static_cast<char*>(norm) + norm_bytes / 2;
      odd = ColorSpinorField::Create(param);
    }

    if (isNative() && create != QUDA_REFERENCE_FIELD_CREATE) {
      if (!(siteSubset == QUDA_FULL_SITE_SUBSET && composite_descr.is_composite)) {
        zeroPad();
      } else { //temporary hack for the full spinor field sets, manual zeroPad for each component:
        for (int cid = 0; cid < composite_descr.dim; cid++) {
          components[cid]->Even().zeroPad();
          components[cid]->Odd().zeroPad();
        }
      }
    }
  }

  void ColorSpinorField::destroy() { init = false; }

  void ColorSpinorField::destroy2()
  {
    if (alloc) {
      if (location == QUDA_CPU_FIELD_LOCATION) {
        host_free(v);
        if (norm_bytes) host_free(norm);
      } else { // device field
        switch(mem_type) {
        case QUDA_MEMORY_DEVICE:
          pool_device_free(v);
          if (norm_bytes) pool_device_free(norm);
          break;
        case QUDA_MEMORY_MAPPED:
          host_free(v_h);
          if (norm_bytes) host_free(norm_h);
          break;
        default: errorQuda("Unsupported memory type %d", mem_type);
        }
      }

      if (composite_descr.is_composite) {
        CompositeColorSpinorField::iterator vec;
        for (vec = components.begin(); vec != components.end(); vec++) delete *vec;
      }
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET && !composite_descr.is_composite) {
      if (even) delete even;
      if (odd) delete odd;
    }
  }

  void ColorSpinorField::setTuningString() {
    {
      //LatticeField::setTuningString(); // FIXME - LatticeField needs correct dims for single-parity
      char vol_tmp[TuneKey::volume_n];
      int check  = snprintf(vol_string, TuneKey::volume_n, "%d", x[0]);
      if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
      for (int d=1; d<nDim; d++) {
        strcpy(vol_tmp, vol_string);
        check = snprintf(vol_string, TuneKey::volume_n, "%sx%d", vol_tmp, x[d]);
        if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
      }
    }

    {
      constexpr int aux_string_n = TuneKey::aux_n / 2;
      char aux_tmp[aux_string_n];
      int check = snprintf(aux_string, aux_string_n, "vol=%lu,stride=%lu,precision=%d,order=%d,Ns=%d,Nc=%d", volume,
                           stride, precision, fieldOrder, nSpin, nColor);
      if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
      if (twistFlavor != QUDA_TWIST_NO && twistFlavor != QUDA_TWIST_INVALID) {
        strcpy(aux_tmp, aux_string);
        check = snprintf(aux_string, aux_string_n, "%s,TwistFlavour=%d", aux_tmp, twistFlavor);
        if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
      }
    }
  }

  void ColorSpinorField::zeroPad()
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    if (!isNative()) errorQuda("Field order %d, precision %d not supported", fieldOrder, precision);

    size_t pad_bytes = (stride - volumeCB) * precision * fieldOrder;
    int Npad = nColor * nSpin * 2 / fieldOrder;

    if (composite_descr.is_composite && !composite_descr.is_component) {//we consider the whole eigenvector set:
      Npad      *= composite_descr.dim;
      pad_bytes /= composite_descr.dim;
    }

    size_t pitch = ((!composite_descr.is_composite || composite_descr.is_component) ? stride : composite_descr.stride)*fieldOrder*precision;
    char   *dst  = (char*)v + ((!composite_descr.is_composite || composite_descr.is_component) ? volumeCB : composite_descr.volumeCB)*fieldOrder*precision;
    if (pad_bytes)
      for (int subset=0; subset<siteSubset; subset++) {
        qudaMemset2DAsync(dst + subset * bytes / siteSubset, pitch, 0, pad_bytes, Npad, device::get_default_stream());
      }

    if (norm_bytes > 0) { // zero initialize the norm pad
      size_t pad_bytes = (stride - volumeCB) * sizeof(float);
      if (pad_bytes)
        for (int subset=0; subset<siteSubset; subset++) {
          qudaMemsetAsync((char *)norm + volumeCB * sizeof(float), 0, (stride - volumeCB) * sizeof(float),
                          device::get_default_stream());
        }
    }

    // zero the region added for alignment reasons
    if (bytes != (size_t)length*precision) {
      size_t subset_bytes = bytes/siteSubset;
      size_t subset_length = length/siteSubset;
      for (int subset=0; subset < siteSubset; subset++) {
        qudaMemsetAsync((char *)v + subset_length * precision + subset_bytes * subset, 0,
                        subset_bytes - subset_length * precision, device::get_default_stream());
      }
    }

    // zero the region added for alignment reasons (norm)
    if (norm_bytes && norm_bytes != siteSubset*stride*sizeof(float)) {
      size_t subset_bytes = norm_bytes/siteSubset;
      for (int subset=0; subset < siteSubset; subset++) {
        qudaMemsetAsync((char *)norm + (size_t)stride * sizeof(float) + subset_bytes * subset, 0,
                        subset_bytes - (size_t)stride * sizeof(float), device::get_default_stream());
      }
    }
  }

  void ColorSpinorField::createGhostZone(int nFace, bool spin_project) const
  {
    if (ghost_precision_allocated == ghost_precision) return;

    bool is_fixed = (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION);
    int nSpinGhost = (nSpin == 4 && spin_project) ? 2 : nSpin;
    size_t site_size = nSpinGhost * nColor * 2 * ghost_precision + (is_fixed ? sizeof(float) : 0);

    // calculate size of ghost zone required
    int ghost_volume = 0;
    int dims = nDim == 5 ? (nDim - 1) : nDim;
    int x5 = nDim == 5 ? x[4] : 1; /// includes DW and non-degenerate TM ghosts
    const int ghost_align
      = 1; // TODO perhaps in the future we should align each ghost dim/dir, e.g., along 32-byte boundaries
    ghost_bytes = 0;
    for (int i=0; i<dims; i++) {
      ghostFace[i] = 0;
      if (comm_dim_partitioned(i)) {
	ghostFace[i] = 1;
	for (int j=0; j<dims; j++) {
	  if (i==j) continue;
	  ghostFace[i] *= x[j];
	}
        ghostFace[i] *= x5; // temporary hack : extra dimension for DW ghosts
        if (i == 0 && siteSubset != QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
        ghost_volume += 2 * nFace * ghostFace[i];
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
      for (int dim=0; dim<nDim; dim++) X[dim] = x[dim];
      for (int dim=nDim; dim<QUDA_MAX_DIM; dim++) X[dim] = 1;
      if (siteSubset == QUDA_PARITY_SITE_SUBSET) X[0] = 2*X[0];

      for (int i = 0; i < nDim; i++) dc.Xh[i] = X[i] / 2;

      dc.Ls = X[4];
      dc.volume_4d_cb = volumeCB / (nDim == 5 ? x[4] : 1);
      dc.volume_4d = 2 * dc.volume_4d_cb;

      int face[4];
      for (int dim=0; dim<4; dim++) {
        for (int j=0; j<4; j++) face[j] = X[j];
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
      if (norm_bytes) qudaMemsetAsync(norm, 0, norm_bytes, device::get_default_stream());
    } else {
      memset(v, '\0', bytes);
      if (norm_bytes) memset(norm, 0, norm_bytes);
    }
  }

  ColorSpinorField& ColorSpinorField::operator=(const ColorSpinorField &src)
  {
    if (&src != this) {
      if (!init) { // keep current attributes unless unset
        if(src.composite_descr.is_composite){
          this->composite_descr.is_composite = true;
          this->composite_descr.dim          = src.composite_descr.dim;
          this->composite_descr.is_component = false;
          this->composite_descr.id           = 0;
        } else if(src.composite_descr.is_component){
          this->composite_descr.is_composite = false;
          this->composite_descr.dim          = 0;
          //this->composite_descr.is_component = false;
          //this->composite_descr.id           = 0;
        }

        create(src.nDim, src.x, src.nColor, src.nSpin, src.nVec, src.twistFlavor, src.precision, src.pad, src.siteSubset,
               src.siteOrder, src.fieldOrder, src.gammaBasis, src.pc_type, src.suggested_parity);
      }

      if (!reference) {
	destroy2();
	create2(QUDA_COPY_FIELD_CREATE);
      }
      copy(src);
    }
    return *this;
  }

  void ColorSpinorField::copy(const ColorSpinorField &src)
  {
    checkField(*this, src);
    if (Location() == src.Location()) { // H2H and D2D

      copyGenericColorSpinor(*this, src, Location());

    } else if (Location() == QUDA_CUDA_FIELD_LOCATION && src.Location() == QUDA_CPU_FIELD_LOCATION) { // H2D

      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // reorder on host
        void *buffer = pool_pinned_malloc(bytes + norm_bytes);
        memset(buffer, 0, bytes+norm_bytes); // FIXME (temporary?) bug fix for padding
        copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, 0, static_cast<char*>(buffer)+bytes, 0);
        qudaMemcpy(v, buffer, bytes, qudaMemcpyDefault);
        qudaMemcpy(norm, static_cast<char *>(buffer) + bytes, norm_bytes, qudaMemcpyDefault);
        pool_pinned_free(buffer);

      } else { // reorder on device

        if (src.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
          // special case where we use mapped memory to read/write directly from application's array
          void *src_d = get_mapped_device_pointer(src.V());
          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, v, src_d);
        } else {
          void *Src=nullptr, *srcNorm=nullptr, *buffer=nullptr;
          if (!zeroCopy) {
            buffer = pool_device_malloc(src.Bytes()+src.NormBytes());
            Src = buffer;
            srcNorm = static_cast<char*>(Src) + src.Bytes();
            qudaMemcpy(Src, src.V(), src.Bytes(), qudaMemcpyDefault);
            qudaMemcpy(srcNorm, src.Norm(), src.NormBytes(), qudaMemcpyDefault);
          } else {
            buffer = pool_pinned_malloc(src.Bytes()+src.NormBytes());
            memcpy(buffer, src.V(), src.Bytes());
            memcpy(static_cast<char*>(buffer)+src.Bytes(), src.Norm(), src.NormBytes());
            Src = get_mapped_device_pointer(buffer);
            srcNorm = static_cast<char*>(Src) + src.Bytes();
          }

          qudaMemsetAsync(v, 0, bytes, device::get_default_stream()); // FIXME (temporary?) bug fix for padding
          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, 0, Src, 0, srcNorm);

          if (zeroCopy) pool_pinned_free(buffer);
          else pool_device_free(buffer);
        }
      }
      qudaDeviceSynchronize(); // include sync here for accurate host-device profiling

    } else if (Location() == QUDA_CPU_FIELD_LOCATION && src.Location() == QUDA_CUDA_FIELD_LOCATION) { //D2H

      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // reorder on the host
        void *buffer = pool_pinned_malloc(bytes+norm_bytes);
        qudaMemcpy(buffer, v, bytes, qudaMemcpyDefault);
        qudaMemcpy(static_cast<char *>(buffer) + bytes, norm, norm_bytes, qudaMemcpyDefault);
        copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, 0, buffer, 0, static_cast<char*>(buffer)+bytes);
        pool_pinned_free(buffer);

      } else { // reorder on the device

        if (FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
          // special case where we use zero-copy memory to read/write directly from application's array
          void *dest_d = get_mapped_device_pointer(v);
          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, dest_d, src.V());
        } else {
          void *dst = nullptr, *dstNorm = nullptr, *buffer = nullptr;
          if (!zeroCopy) {
            buffer = pool_device_malloc(bytes + norm_bytes);
            dst = buffer;
            dstNorm = static_cast<char*>(dst) + Bytes();
          } else {
            buffer = pool_pinned_malloc(bytes + norm_bytes);
            dst = get_mapped_device_pointer(buffer);
            dstNorm = static_cast<char*>(dst) + bytes;
          }

          copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION, dst, 0, dstNorm, 0);

          if (!zeroCopy) {
            qudaMemcpy(v, dst, Bytes(), qudaMemcpyDefault);
            qudaMemcpy(norm, dstNorm, NormBytes(), qudaMemcpyDefault);
          } else {
            qudaDeviceSynchronize();
            memcpy(v, buffer, bytes);
            memcpy(norm, static_cast<char*>(buffer) + bytes, norm_bytes);
          }

          if (zeroCopy) pool_pinned_free(buffer);
          else pool_device_free(buffer);
        }
      }

      qudaDeviceSynchronize(); // need to sync before data can be used on CPU
    }
  }

  // Fills the param with the contents of this field
  void ColorSpinorField::fill(ColorSpinorParam &param) const
  {
    param.field = const_cast<ColorSpinorField*>(this);
    param.v = v;
    param.norm = norm;

    param.location = location;
    param.nColor = nColor;
    param.nSpin = nSpin;
    param.nVec = nVec;
    param.twistFlavor = twistFlavor;
    param.fieldOrder = fieldOrder;
    param.setPrecision(precision, ghost_precision);
    param.nDim = nDim;

    param.is_composite  = composite_descr.is_composite;
    param.composite_dim = composite_descr.dim;
    param.is_component  = false;//always either a regular spinor or a composite object
    param.component_id  = 0;

    memcpy(param.x, x, QUDA_MAX_DIM*sizeof(int));
    param.pad = pad;
    param.siteSubset = siteSubset;
    param.siteOrder = siteOrder;
    param.gammaBasis = gammaBasis;
    param.pc_type = pc_type;
    param.suggested_parity = suggested_parity;
    param.create = QUDA_NULL_FIELD_CREATE;

    param.mem_type = mem_type;
  }

  void ColorSpinorField::exchange(void **ghost, void **sendbuf, int nFace) const {

    // FIXME: use LatticeField MsgHandles
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];
    size_t bytes[4];

    const int Ninternal = 2*nColor*nSpin;
    size_t total_bytes = 0;
    for (int i=0; i<nDimComms; i++) {
      bytes[i] = siteSubset*nFace*surfaceCB[i]*Ninternal*ghost_precision;
      if (comm_dim_partitioned(i)) total_bytes += 2*bytes[i]; // 2 for fwd/bwd
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
      for (int i=0; i<nDimComms; i++) {
	if (comm_dim_partitioned(i)) {
	  send_back[i] = sendbuf[2*i + 0];
	  send_fwd[i]  = sendbuf[2*i + 1];
	  recv_fwd[i]  =   ghost[2*i + 1];
	  recv_back[i] =   ghost[2*i + 0];
	} else if (no_comms_fill) {
	  memcpy(ghost[2*i+1], sendbuf[2*i+0], bytes[i]);
	  memcpy(ghost[2*i+0], sendbuf[2*i+1], bytes[i]);
	}
      }
    } else { // FIXME add GPU_COMMS support
      if (total_bytes) {
	total_send = pool_pinned_malloc(total_bytes);
	total_recv = pool_pinned_malloc(total_bytes);
      }
      size_t offset = 0;
      for (int i=0; i<nDimComms; i++) {
	if (comm_dim_partitioned(i)) {
	  send_back[i] = static_cast<char*>(total_send) + offset;
	  recv_back[i] = static_cast<char*>(total_recv) + offset;
	  offset += bytes[i];
	  send_fwd[i] = static_cast<char*>(total_send) + offset;
	  recv_fwd[i] = static_cast<char*>(total_recv) + offset;
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
	for (int i=0; i<nDimComms; i++) {
	  if (comm_dim_partitioned(i)) {
	    send_ptr = sendbuf[2*i];
	    break;
	  }
	}
        qudaMemcpy(total_send, send_ptr, total_bytes, qudaMemcpyDeviceToHost);
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      mh_send_fwd[i] = comm_declare_send_relative(send_fwd[i], i, +1, bytes[i]);
      mh_send_back[i] = comm_declare_send_relative(send_back[i], i, -1, bytes[i]);
      mh_from_fwd[i] = comm_declare_receive_relative(recv_fwd[i], i, +1, bytes[i]);
      mh_from_back[i] = comm_declare_receive_relative(recv_back[i], i, -1, bytes[i]);
    }

    for (int i=0; i<nDimComms; i++) {
      if (comm_dim_partitioned(i)) {
	comm_start(mh_from_back[i]);
	comm_start(mh_from_fwd[i]);
	comm_start(mh_send_fwd[i]);
	comm_start(mh_send_back[i]);
      }
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_wait(mh_send_fwd[i]);
      comm_wait(mh_send_back[i]);
      comm_wait(mh_from_back[i]);
      comm_wait(mh_from_fwd[i]);
    }

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      for (int i=0; i<nDimComms; i++) {
	if (!comm_dim_partitioned(i)) continue;
	if (fine_grained_memcpy) {
          qudaMemcpy(ghost[2 * i + 0], recv_back[i], bytes[i], qudaMemcpyHostToDevice);
          qudaMemcpy(ghost[2 * i + 1], recv_fwd[i], bytes[i], qudaMemcpyHostToDevice);
        }
      }

      if (!fine_grained_memcpy && total_bytes) {
	// find first non-zero pointer
	void *ghost_ptr = nullptr;
	for (int i=0; i<nDimComms; i++) {
	  if (comm_dim_partitioned(i)) {
	    ghost_ptr = ghost[2*i];
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

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_free(mh_send_fwd[i]);
      comm_free(mh_send_back[i]);
      comm_free(mh_from_back[i]);
      comm_free(mh_from_fwd[i]);
    }
  }

  // For kernels with precision conversion built in
  void ColorSpinorField::checkField(const ColorSpinorField &a, const ColorSpinorField &b) {
    if (a.Length() != b.Length()) {
      errorQuda("checkSpinor: lengths do not match: %lu %lu", a.Length(), b.Length());
    }

    if (a.Ncolor() != b.Ncolor()) {
      errorQuda("checkSpinor: colors do not match: %d %d", a.Ncolor(), b.Ncolor());
    }

    if (a.Nspin() != b.Nspin()) {
      errorQuda("checkSpinor: spins do not match: %d %d", a.Nspin(), b.Nspin());
    }

    if (a.Nvec() != b.Nvec()) {
      errorQuda("checkSpinor: nVec does not match: %d %d", a.Nvec(), b.Nvec());
    }

    if (a.TwistFlavor() != b.TwistFlavor()) {
      errorQuda("checkSpinor: twist flavors do not match: %d %d", a.TwistFlavor(), b.TwistFlavor());
    }
  }

  const ColorSpinorField& ColorSpinorField::Even() const {
    if (siteSubset != QUDA_FULL_SITE_SUBSET)
      errorQuda("Cannot return even subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER)
      errorQuda("Cannot return even subset of QDPJIT field");
    return *even;
  }

  const ColorSpinorField& ColorSpinorField::Odd() const {
    if (siteSubset != QUDA_FULL_SITE_SUBSET)
      errorQuda("Cannot return odd subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER)
      errorQuda("Cannot return even subset of QDPJIT field");
    return *odd;
  }

  ColorSpinorField& ColorSpinorField::Even() {
    if (siteSubset != QUDA_FULL_SITE_SUBSET)
      errorQuda("Cannot return even subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER)
      errorQuda("Cannot return even subset of QDPJIT field");
    return *even;
  }

  ColorSpinorField& ColorSpinorField::Odd() {
    if (siteSubset != QUDA_FULL_SITE_SUBSET)
      errorQuda("Cannot return odd subset of %d subset", siteSubset);
    if (fieldOrder == QUDA_QDPJIT_FIELD_ORDER)
      errorQuda("Cannot return even subset of QDPJIT field");
    return *odd;
  }

  void* ColorSpinorField::Ghost(const int i) {
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghost[i];
  }

  const void* ColorSpinorField::Ghost(const int i) const {
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghost[i];
  }


  void* ColorSpinorField::GhostNorm(const int i){
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghostNorm[i];
  }

  const void* ColorSpinorField::GhostNorm(const int i) const{
    if(siteSubset != QUDA_PARITY_SITE_SUBSET) errorQuda("Site Subset %d is not supported",siteSubset);
    return ghostNorm[i];
  }

  void* const* ColorSpinorField::Ghost() const {
    return ghost_buf;
  }

  const void* ColorSpinorField::Ghost2() const
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
  void ColorSpinorField::LatticeIndex(int *y, int i) const {
    int z[QUDA_MAX_DIM];
    memcpy(z, x, QUDA_MAX_DIM*sizeof(int));

    // parity is the slowest running dimension
    int parity = 0;
    if (siteSubset == QUDA_FULL_SITE_SUBSET) z[0] /= 2;

    for (int d=0; d<nDim; d++) {
      y[d] = i % z[d];
      i /= z[d];
    }

    parity = i;

    // convert into the full-field lattice coordinate
    int oddBit = parity;
    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      for (int d=1; d<nDim; d++) oddBit += y[d];
      oddBit = oddBit & 1;
    }
    y[0] = 2*y[0] + oddBit;  // compute the full x coordinate
  }

  /*
    Convert from n-dimensional spatial index to the 1-dimensional index.
    With full fields, we assume that the field is even-odd ordered.  The
    input lattice coordinates are always full-field coordinates.
  */
  void ColorSpinorField::OffsetIndex(int &i, int *y) const {

    int parity = 0;
    int z[QUDA_MAX_DIM];
    memcpy(z, x, QUDA_MAX_DIM*sizeof(int));
    int savey0 = y[0];

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      for (int d=0; d<nDim; d++) parity += y[d];
      parity = parity & 1;
      y[0] /= 2;
      z[0] /= 2;
    }

    i = parity;
    for (int d=nDim-1; d>=0; d--) {
      i = z[d]*i + y[d];
      //printf("z[%d]=%d y[%d]=%d ", d, z[d], d, y[d]);
    }

    //printf("\nparity = %d\n", parity);

    if (siteSubset == QUDA_FULL_SITE_SUBSET) y[0] = savey0;
  }

  ColorSpinorField* ColorSpinorField::Create(const ColorSpinorParam &param)
  {
    ColorSpinorField *field = nullptr;
    if (param.location == QUDA_CPU_FIELD_LOCATION) {
      field = new cpuColorSpinorField(param);
    } else if (param.location== QUDA_CUDA_FIELD_LOCATION) {
      field = new cudaColorSpinorField(param);
    } else {
      errorQuda("Invalid field location %d", param.location);
    }

    return field;
  }

  ColorSpinorField *ColorSpinorField::CreateAlias(const ColorSpinorParam &param_)
  {
    if (param_.Precision() > precision)
      errorQuda("Cannot create an alias to source with lower precision than the alias");
    ColorSpinorParam param(param_);
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.v = V();

    // if norm field in the source exists, use it, else use the second
    // half of main field for norm storage, ensuring that the start of
    // the norm field is on an alignment boundary if we're using an
    // internal field
    if (param.Precision() < QUDA_SINGLE_PRECISION) {
      auto norm_offset = (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) ?
        (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2 * ALIGNMENT_ADJUST(Bytes() / 4) : ALIGNMENT_ADJUST(Bytes() / 2) :
        0;
      param.norm = Norm() ? Norm() : static_cast<char *>(V()) + norm_offset;
    }

    auto alias = ColorSpinorField::Create(param);

    if (alias->Bytes() > Bytes()) errorQuda("Alias footprint %lu greater than source %lu", alias->Bytes(), Bytes());
    if (alias->Precision() < QUDA_SINGLE_PRECISION) {
      // check that norm does not overlap with body
      if (static_cast<char *>(alias->V()) + alias->Bytes() > alias->Norm())
        errorQuda("Overlap between alias body and norm");
      // check that norm does fall off the end
      if (static_cast<char *>(alias->Norm()) + alias->NormBytes() > static_cast<char *>(V()) + Bytes())
        errorQuda("Norm is not contained in the srouce field");
    }

    return alias;
  }

  ColorSpinorField* ColorSpinorField::CreateCoarse(const int *geoBlockSize, int spinBlockSize, int Nvec,
                                                   QudaPrecision new_precision, QudaFieldLocation new_location,
                                                   QudaMemoryType new_mem_type) {
    ColorSpinorParam coarseParam(*this);
    for (int d=0; d<nDim; d++) coarseParam.x[d] = x[d]/geoBlockSize[d];

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

    // for GPU fields, always use native ordering to ensure coalescing
    if (new_location == QUDA_CUDA_FIELD_LOCATION) coarseParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;
    else coarseParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;

    coarseParam.setPrecision(new_precision);

    // set where we allocate the field
    coarseParam.mem_type = (new_mem_type != QUDA_MEMORY_INVALID) ? new_mem_type :
      (new_location == QUDA_CUDA_FIELD_LOCATION ? QUDA_MEMORY_DEVICE : QUDA_MEMORY_PINNED);

    ColorSpinorField *coarse = NULL;
    if (new_location == QUDA_CPU_FIELD_LOCATION) {
      coarse = new cpuColorSpinorField(coarseParam);
    } else if (new_location== QUDA_CUDA_FIELD_LOCATION) {
      coarse = new cudaColorSpinorField(coarseParam);
    } else {
      errorQuda("Invalid field location %d", new_location);
    }

    return coarse;
  }

  ColorSpinorField* ColorSpinorField::CreateFine(const int *geoBlockSize, int spinBlockSize, int Nvec,
                                                 QudaPrecision new_precision, QudaFieldLocation new_location,
                                                 QudaMemoryType new_mem_type) {
    ColorSpinorParam fineParam(*this);
    for (int d=0; d<nDim; d++) fineParam.x[d] = x[d] * geoBlockSize[d];
    fineParam.nSpin = nSpin * spinBlockSize;
    fineParam.nColor = Nvec;
    fineParam.siteSubset = QUDA_FULL_SITE_SUBSET; // FIXME fine grid is always full
    fineParam.create = QUDA_ZERO_FIELD_CREATE;

    // if new precision is not set, use this->precision
    new_precision = (new_precision == QUDA_INVALID_PRECISION) ? Precision() : new_precision;

    // if new location is not set, use this->location
    new_location = (new_location == QUDA_INVALID_FIELD_LOCATION) ? Location(): new_location;

    // for GPU fields, always use native ordering to ensure coalescing
    if (new_location == QUDA_CUDA_FIELD_LOCATION) {
      fineParam.setPrecision(new_precision, new_precision, true);
    } else {
      fineParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
      fineParam.setPrecision(new_precision);
    }

    // set where we allocate the field
    fineParam.mem_type = (new_mem_type != QUDA_MEMORY_INVALID) ? new_mem_type :
      (new_location == QUDA_CUDA_FIELD_LOCATION ? QUDA_MEMORY_DEVICE : QUDA_MEMORY_PINNED);

    ColorSpinorField *fine = NULL;
    if (new_location == QUDA_CPU_FIELD_LOCATION) {
      fine = new cpuColorSpinorField(fineParam);
    } else if (new_location == QUDA_CUDA_FIELD_LOCATION) {
      fine = new cudaColorSpinorField(fineParam);
    } else {
      errorQuda("Invalid field location %d", new_location);
    }
    return fine;
  }

  // legacy CPU static ghost destructor
  void ColorSpinorField::freeGhostBuffer(void)
  {
    if (!initGhostFaceBuffer) return;

    for(int i=0; i < 4; i++){  // make nDimComms static?
      host_free(fwdGhostFaceBuffer[i]); fwdGhostFaceBuffer[i] = NULL;
      host_free(backGhostFaceBuffer[i]); backGhostFaceBuffer[i] = NULL;
      host_free(fwdGhostFaceSendBuffer[i]); fwdGhostFaceSendBuffer[i] = NULL;
      host_free(backGhostFaceSendBuffer[i]);  backGhostFaceSendBuffer[i] = NULL;
    }
    initGhostFaceBuffer = 0;
  }


  void ColorSpinorField::allocateGhostBuffer(int nFace, bool spin_project) const
  {
    createGhostZone(nFace, spin_project);
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      if (spin_project) errorQuda("Not yet implemented");

      int spinor_size = 2*nSpin*nColor*precision;
      bool resize = false;

      // resize face only if requested size is larger than previously allocated one
      for (int i=0; i<nDimComms; i++) {
        size_t nbytes = siteSubset*nFace*surfaceCB[i]*spinor_size;
        resize = (nbytes > ghostFaceBytes[i]) ? true : resize;
        ghostFaceBytes[i] = (nbytes > ghostFaceBytes[i]) ? nbytes : ghostFaceBytes[i];
      }

      if (!initGhostFaceBuffer || resize) {
        freeGhostBuffer();
        for (int i=0; i<nDimComms; i++) {
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
    allocateGhostBuffer(nFace,spin_project); // allocate the ghost buffer if not yet allocated

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
      for (int i=0; i<nDimComms; ++i) {
	if (commDimPartitioned(i)) {
	  for (int b=0; b<2; b++) {
            ghost[b][i] = static_cast<char *>(ghost_recv_buffer_d[b]) + ghost_offset[i][0];
            if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION)
              ghostNorm[b][i] = static_cast<char *>(ghost[b][i])
                + nFace * surface[i] * (nSpin / (spin_project ? 2 : 1)) * nColor * 2 * ghost_precision;
          }
        }
      }

      ghost_precision_reset = false;
    }

    if (ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  // pack the ghost zone into a contiguous buffer for communications
  void ColorSpinorField::packGhost(const int nFace, const QudaParity parity, const int dagger,
                                   const qudaStream_t &stream, MemoryLocation location[2 * QUDA_MAX_DIM],
                                   MemoryLocation location_label, bool spin_project, double a, double b, double c,
                                   int shmem)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    void *packBuffer[4 * QUDA_MAX_DIM] = {};

    for (int dim=0; dim<4; dim++) {
      for (int dir=0; dir<2; dir++) {
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
	case Host:   // pack to zero-copy memory
	  packBuffer[2*dim+dir] = my_face_dim_dir_hd[bufferIndex][dim][dir];
          break;
        case Remote: // pack to remote peer memory
          packBuffer[2 * dim + dir]
            = static_cast<char *>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + ghost_offset[dim][1 - dir];
          break;
	default: errorQuda("Undefined location %d", location[2*dim+dir]);
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

  // FIXME reconcile with above
  void ColorSpinorField::packGhostHost(void **ghost, const QudaParity parity, const int nFace, const int dagger) const
  {
    genericPackGhost(ghost, *this, parity, nFace, dagger);
  }

  void ColorSpinorField::sendGhost(void *ghost_spinor, const int dim, const QudaDirection dir,
                                   const qudaStream_t &stream)
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
    int dim = dir/2;

    if (dir%2 == 0) {
      // backwards copy to host
      if (comm_peer2peer_enabled(0,dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][0], dim, QUDA_BACKWARDS, stream);
    } else {
      // forwards copy to host
      if (comm_peer2peer_enabled(1,dim)) return;

      sendGhost(my_face_dim_dir_h[bufferIndex][dim][1], dim, QUDA_FORWARDS, stream);
    }
  }

  void ColorSpinorField::recvStart(int d, const qudaStream_t &, bool gdr)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (dir == 0) { // receive from forwards
      // receive from the processor in the +1 direction
      if (comm_peer2peer_enabled(1,dim)) {
	comm_start(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr) {
        comm_start(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_fwd[bufferIndex][dim]);
      }
    } else { // receive from backwards
      // receive from the processor in the -1 direction
      if (comm_peer2peer_enabled(0,dim)) {
	comm_start(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr) {
        comm_start(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_back[bufferIndex][dim]);
      }
    }
  }

  void ColorSpinorField::sendStart(int d, const qudaStream_t &stream, bool gdr, bool remote_write)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;
    if (!commDimPartitioned(dim)) return;
    if (gdr && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (!comm_peer2peer_enabled(dir,dim)) {
      if (dir == 0)
	if (gdr) comm_start(mh_send_rdma_back[bufferIndex][dim]);
	else comm_start(mh_send_back[bufferIndex][dim]);
      else
	if (gdr) comm_start(mh_send_rdma_fwd[bufferIndex][dim]);
	else comm_start(mh_send_fwd[bufferIndex][dim]);
    } else { // doing peer-to-peer

      // if not using copy engine then the packing kernel will remotely write the halos
      if (!remote_write) {
        // all goes here
        void *ghost_dst
          = static_cast<char *>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + ghost_offset[dim][(dir + 1) % 2];

        qudaMemcpyP2PAsync(ghost_dst, my_face_dim_dir_d[bufferIndex][dim][dir], ghost_face_bytes[dim], stream);
      } // remote_write

      if (dir == 0) {
	// record the event
        qudaEventRecord(ipcCopyEvent[bufferIndex][0][dim], stream);
        // send to the processor in the -1 direction
	comm_start(mh_send_p2p_back[bufferIndex][dim]);
      } else {
        qudaEventRecord(ipcCopyEvent[bufferIndex][1][dim], stream);
        // send to the processor in the +1 direction
	comm_start(mh_send_p2p_fwd[bufferIndex][dim]);
      }
    }
  }

  void ColorSpinorField::commsStart(int dir, const qudaStream_t &stream, bool gdr_send, bool gdr_recv)
  {
    recvStart(dir, stream, gdr_recv);
    sendStart(dir, stream, gdr_send);
  }

  static bool complete_recv_fwd[QUDA_MAX_DIM] = { };
  static bool complete_recv_back[QUDA_MAX_DIM] = { };
  static bool complete_send_fwd[QUDA_MAX_DIM] = { };
  static bool complete_send_back[QUDA_MAX_DIM] = { };

  int ColorSpinorField::commsQuery(int d, const qudaStream_t &, bool gdr_send, bool gdr_recv)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;

    if (!commDimPartitioned(dim)) return 1;
    if ((gdr_send || gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (dir==0) {

      // first query send to backwards
      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_p2p_back[bufferIndex][dim]);
      } else if (gdr_send) {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_send_back[dim]) complete_send_back[dim] = comm_query(mh_send_back[bufferIndex][dim]);
      }

      // second query receive from forwards
      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (gdr_recv) {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_recv_fwd[dim]) complete_recv_fwd[dim] = comm_query(mh_recv_fwd[bufferIndex][dim]);
      }

      if (complete_recv_fwd[dim] && complete_send_back[dim]) {
	complete_send_back[dim] = false;
	complete_recv_fwd[dim] = false;
	return 1;
      }

    } else { // dir == 1

      // first query send to forwards
      if (comm_peer2peer_enabled(1,dim)) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_p2p_fwd[bufferIndex][dim]);
      } else if (gdr_send) {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	if (!complete_send_fwd[dim]) complete_send_fwd[dim] = comm_query(mh_send_fwd[bufferIndex][dim]);
      }

      // second query receive from backwards
      if (comm_peer2peer_enabled(0,dim)) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (gdr_recv) {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	if (!complete_recv_back[dim]) complete_recv_back[dim] = comm_query(mh_recv_back[bufferIndex][dim]);
      }

      if (complete_recv_back[dim] && complete_send_fwd[dim]) {
	complete_send_fwd[dim] = false;
	complete_recv_back[dim] = false;
	return 1;
      }

    }

    return 0;
  }

  void ColorSpinorField::commsWait(int d, const qudaStream_t &, bool gdr_send, bool gdr_recv)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards)

    int dim = d/2;
    int dir = d%2;

    if (!commDimPartitioned(dim)) return;
    if ( (gdr_send && gdr_recv) && !comm_gdr_enabled()) errorQuda("Requesting GDR comms but GDR is not enabled");

    if (dir==0) {

      // first wait on send to backwards
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_send_p2p_back[bufferIndex][dim]);
        qudaEventSynchronize(ipcCopyEvent[bufferIndex][0][dim]);
      } else if (gdr_send) {
	comm_wait(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_back[bufferIndex][dim]);
      }

      // second wait on receive from forwards
      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_recv_p2p_fwd[bufferIndex][dim]);
        qudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][1][dim]);
      } else if (gdr_recv) {
	comm_wait(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_fwd[bufferIndex][dim]);
      }

    } else {

      // first wait on send to forwards
      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_send_p2p_fwd[bufferIndex][dim]);
        qudaEventSynchronize(ipcCopyEvent[bufferIndex][1][dim]);
      } else if (gdr_send) {
	comm_wait(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_fwd[bufferIndex][dim]);
      }

      // second wait on receive from backwards
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_recv_p2p_back[bufferIndex][dim]);
        qudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][0][dim]);
      } else if (gdr_recv) {
	comm_wait(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_back[bufferIndex][dim]);
      }

    }
  }

  void ColorSpinorField::scatter(int dim_dir, const qudaStream_t &stream)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) errorQuda("Host field not supported");
    // note this is scatter centric, so input expects dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards), so here we need flip to receive centric

    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 recive from backwards
    if (!commDimPartitioned(dim)) return;
    if (comm_peer2peer_enabled(dir,dim)) return;

    unpackGhost(from_face_dim_dir_h[bufferIndex][dim][dir], dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, stream);
  }

  void* ColorSpinorField::fwdGhostFaceBuffer[QUDA_MAX_DIM];
  void* ColorSpinorField::backGhostFaceBuffer[QUDA_MAX_DIM];
  void* ColorSpinorField::fwdGhostFaceSendBuffer[QUDA_MAX_DIM];
  void* ColorSpinorField::backGhostFaceSendBuffer[QUDA_MAX_DIM];
  int ColorSpinorField::initGhostFaceBuffer =0;
  size_t ColorSpinorField::ghostFaceBytes[QUDA_MAX_DIM] = { };

  void ColorSpinorField::exchangeGhost(QudaParity parity, int nFace, int dagger,
                                       const MemoryLocation *pack_destination_, const MemoryLocation *halo_location_,
                                       bool gdr_send, bool gdr_recv, QudaPrecision ghost_precision_) const
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      // allocate ghost buffer if not yet allocated
      allocateGhostBuffer(nFace, false);

      void **sendbuf = static_cast<void**>(safe_malloc(nDimComms * 2 * sizeof(void*)));

      for (int i=0; i<nDimComms; i++) {
        sendbuf[2*i + 0] = backGhostFaceSendBuffer[i];
        sendbuf[2*i + 1] = fwdGhostFaceSendBuffer[i];
        ghost_buf[2*i + 0] = backGhostFaceBuffer[i];
        ghost_buf[2*i + 1] = fwdGhostFaceBuffer[i];
      }

      packGhostHost(sendbuf, parity, nFace, dagger);

      exchange(ghost_buf, sendbuf, nFace);

      host_free(sendbuf);
    } else {
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
      reinterpret_cast<cudaColorSpinorField&>(const_cast<ColorSpinorField&>(*this)).createComms(nFace, false);

      // first set default values to device if needed
      MemoryLocation pack_destination[2*QUDA_MAX_DIM], halo_location[2*QUDA_MAX_DIM];
      for (int i=0; i<2*nDimComms; i++) {
        pack_destination[i] = pack_destination_ ? pack_destination_[i] : Device;
        halo_location[i] = halo_location_ ? halo_location_[i] : Device;
      }

      // Contiguous send buffers and we aggregate copies to reduce
      // latency.  Only if all locations are "Device" and no p2p
      bool fused_pack_memcpy = true;

      // Contiguous recv buffers and we aggregate copies to reduce
      // latency.  Only if all locations are "Device" and no p2p
      bool fused_halo_memcpy = true;

      bool pack_host = false; // set to true if any of the ghost packing is being done to Host memory
      bool halo_host = false; // set to true if the final halos will be left in Host memory

      void *send[2*QUDA_MAX_DIM];
      for (int d=0; d<nDimComms; d++) {
        for (int dir=0; dir<2; dir++) {
          send[2*d+dir] = pack_destination[2*d+dir] == Host ? my_face_dim_dir_hd[bufferIndex][d][dir] : my_face_dim_dir_d[bufferIndex][d][dir];
          ghost_buf[2*d+dir] = halo_location[2*d+dir] == Host ? from_face_dim_dir_hd[bufferIndex][d][dir] : from_face_dim_dir_d[bufferIndex][d][dir];
        }

        // if doing p2p, then we must pack to and load the halo from device memory
        for (int dir=0; dir<2; dir++) {
          if (comm_peer2peer_enabled(dir,d)) { pack_destination[2*d+dir] = Device; halo_location[2*d+1-dir] = Device; }
        }

        // if zero-copy packing or p2p is enabled then we cannot do fused memcpy
        if (pack_destination[2*d+0] != Device || pack_destination[2*d+1] != Device || comm_peer2peer_enabled_global()) fused_pack_memcpy = false;
        // if zero-copy halo read or p2p is enabled then we cannot do fused memcpy
        if (halo_location[2*d+0] != Device || halo_location[2*d+1] != Device || comm_peer2peer_enabled_global()) fused_halo_memcpy = false;

        if (pack_destination[2*d+0] == Host || pack_destination[2*d+1] == Host) pack_host = true;
        if (halo_location[2*d+0] == Host || halo_location[2*d+1] == Host) halo_host = true;
      }

      // Error if zero-copy and p2p for now
      if ( (pack_host || halo_host) && comm_peer2peer_enabled_global()) errorQuda("Cannot use zero-copy memory with peer-to-peer comms yet");

      genericPackGhost(send, *this, parity, nFace, dagger, pack_destination); // FIXME - need support for asymmetric topologies

      size_t total_bytes = 0;
      for (int i = 0; i < nDimComms; i++)
        if (comm_dim_partitioned(i)) total_bytes += 2 * ghost_face_bytes_aligned[i]; // 2 for fwd/bwd

      if (!gdr_send)  {
        if (!fused_pack_memcpy) {
          for (int i=0; i<nDimComms; i++) {
            if (comm_dim_partitioned(i)) {
              if (pack_destination[2*i+0] == Device && !comm_peer2peer_enabled(0,i) && // fuse forwards and backwards if possible
                  pack_destination[2*i+1] == Device && !comm_peer2peer_enabled(1,i)) {
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
          qudaMemcpyAsync(my_face_h[bufferIndex], ghost_send_buffer_d[bufferIndex], total_bytes, qudaMemcpyDeviceToHost,
                          device::get_default_stream());
        }
      }

      // prepost receive
      for (int i = 0; i < 2 * nDimComms; i++)
        const_cast<ColorSpinorField *>(this)->recvStart(i, device::get_default_stream(), gdr_recv);

      bool sync = pack_host ? true : false; // no p2p if pack_host so we need to synchronize
      // if not p2p in any direction then need to synchronize before MPI
      for (int i=0; i<nDimComms; i++) if (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i)) sync = true;
      if (sync) qudaDeviceSynchronize(); // need to make sure packing and/or memcpy has finished before kicking off MPI

      for (int p2p=0; p2p<2; p2p++) {
        for (int dim=0; dim<nDimComms; dim++) {
          for (int dir=0; dir<2; dir++) {
            if ( (comm_peer2peer_enabled(dir,dim) + p2p) % 2 == 0 ) { // issue non-p2p transfers first
              const_cast<ColorSpinorField *>(this)->sendStart(2 * dim + dir, device::get_stream(2 * dim + dir),
                                                                  gdr_send);
            }
	}
        }
      }

      bool comms_complete[2*QUDA_MAX_DIM] = { };
      int comms_done = 0;
      while (comms_done < 2*nDimComms) { // non-blocking query of each exchange and exit once all have completed
        for (int dim=0; dim<nDimComms; dim++) {
          for (int dir=0; dir<2; dir++) {
            if (!comms_complete[dim*2+dir]) {
              comms_complete[2 * dim + dir] = const_cast<ColorSpinorField *>(this)->commsQuery(
              2 * dim + dir, device::get_default_stream(), gdr_send, gdr_recv);
              if (comms_complete[2*dim+dir]) {
                comms_done++;
                if (comm_peer2peer_enabled(1 - dir, dim))
                  qudaStreamWaitEvent(device::get_default_stream(), ipcRemoteCopyEvent[bufferIndex][1 - dir][dim], 0);
              }
            }
          }
        }
      }

      if (!gdr_recv) {
        if (!fused_halo_memcpy) {
          for (int i=0; i<nDimComms; i++) {
            if (comm_dim_partitioned(i)) {
              if (halo_location[2*i+0] == Device && !comm_peer2peer_enabled(0,i) && // fuse forwards and backwards if possible
                  halo_location[2*i+1] == Device && !comm_peer2peer_enabled(1,i)) {
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
          qudaMemcpyAsync(ghost_recv_buffer_d[bufferIndex], from_face_h[bufferIndex], total_bytes, qudaMemcpyHostToDevice,
                          device::get_default_stream());
        }
      }

      // ensure that the p2p sending is completed before returning
      for (int dim = 0; dim < nDimComms; dim++) {
        if (!comm_dim_partitioned(dim)) continue;
        for (int dir = 0; dir < 2; dir++) {
          if (comm_peer2peer_enabled(dir, dim))
            qudaStreamWaitEvent(device::get_default_stream(), ipcCopyEvent[bufferIndex][dir][dim], 0);
        }
      }
    }
  }

  void ColorSpinorField::backup() const
  {
    if (backed_up) errorQuda("ColorSpinorField already backed up");

    backup_h = new char[bytes];
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(backup_h, v, bytes, qudaMemcpyDefault);
      if (norm_bytes) {
        backup_norm_h = new char[norm_bytes];
        qudaMemcpy(backup_norm_h, norm, norm_bytes, qudaMemcpyDefault);
      }
    } else {
      memcpy(backup_h, v, bytes);
      if (norm_bytes) {
        backup_norm_h = new char[norm_bytes];
        memcpy(backup_norm_h, norm, norm_bytes);
      }
    }

    backed_up = true;
  }

  void ColorSpinorField::restore() const
  {
    if (!backed_up) errorQuda("Cannot restore since not backed up");

    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(v, backup_h, bytes, qudaMemcpyDefault);
      delete []backup_h;
      if (norm_bytes) {
        qudaMemcpy(norm, backup_norm_h, norm_bytes, qudaMemcpyDefault);
        delete []backup_norm_h;
      }
    } else {
      memcpy(v, backup_h, bytes);
      delete []backup_h;
      if (norm_bytes) {
        memcpy(norm, backup_norm_h, norm_bytes);
        delete []backup_norm_h;
      }
    }

    backed_up = false;
  }

  void ColorSpinorField::copy_to_buffer(void *buffer) const
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(buffer, v, bytes, qudaMemcpyDeviceToHost);
      if (precision < QUDA_SINGLE_PRECISION) {
        qudaMemcpy(static_cast<char *>(buffer) + bytes, norm, norm_bytes, qudaMemcpyDeviceToHost);
      }
    } else {
      std::memcpy(buffer, v, bytes);
      if (precision < QUDA_SINGLE_PRECISION) { std::memcpy(static_cast<char *>(buffer) + bytes, norm, norm_bytes); }
    }
  }

  void ColorSpinorField::copy_from_buffer(void *buffer)
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(v, buffer, bytes, qudaMemcpyHostToDevice);
      if (precision < QUDA_SINGLE_PRECISION) {
        qudaMemcpy(norm, static_cast<char *>(buffer) + bytes, norm_bytes, qudaMemcpyHostToDevice);
      }
    } else {
      std::memcpy(v, buffer, bytes);
      if (precision < QUDA_SINGLE_PRECISION) { std::memcpy(norm, static_cast<char *>(buffer) + bytes, norm_bytes); }
    }
  }

  void ColorSpinorField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    if (Location() == QUDA_CUDA_FIELD_LOCATION) {
      // conditionals based on destructor
      if (is_prefetch_enabled() && alloc && mem_type == QUDA_MEMORY_DEVICE) {
        qudaMemPrefetchAsync(v, bytes, mem_space, stream);
        if ((precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) && norm_bytes > 0)
          qudaMemPrefetchAsync(norm, norm_bytes, mem_space, stream);
      }
    }
  }

  void ColorSpinorField::Source(QudaSourceType source_type, unsigned int x, int s, int c)
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      genericSource(*reinterpret_cast<cpuColorSpinorField*>(this), source_type, x, s, c);
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

      cpuColorSpinorField tmp(param);
      tmp.Source(source_type, x, s, c);
      *this = tmp;
    }
  }

  void ColorSpinorField::PrintVector(unsigned int x) const
  {
    if (Location() == QUDA_CPU_FIELD_LOCATION) genericPrintVector(*this, x);
    else genericCudaPrintVector(*this, x);
  }

  int ColorSpinorField::Compare(const ColorSpinorField &a, const ColorSpinorField &b, const int tol)
  {
    if (checkLocation(a, b) == QUDA_CUDA_FIELD_LOCATION) errorQuda("device field not implemented");
    checkField(a, b);
    return genericCompare(a, b, tol);
  }

  std::ostream& operator<<(std::ostream &out, const ColorSpinorField &a) {
    out << "location = " << a.Location() << std::endl;
    out << "v = " << a.v << std::endl;
    out << "norm = " << a.norm << std::endl;
    out << "alloc = " << a.alloc << std::endl;
    out << "reference = " << a.reference << std::endl;
    out << "init = " << a.init << std::endl;
    out << "nColor = " << a.nColor << std::endl;
    out << "nSpin = " << a.nSpin << std::endl;
    out << "twistFlavor = " << a.twistFlavor << std::endl;
    out << "nDim = " << a.nDim << std::endl;
    for (int d=0; d<a.nDim; d++) out << "x[" << d << "] = " << a.x[d] << std::endl;
    out << "volume = " << a.volume << std::endl;
    out << "pc_type = " << a.pc_type << std::endl;
    out << "suggested_parity = " << a.suggested_parity << std::endl;
    out << "precision = " << a.precision << std::endl;
    out << "ghost_precision = " << a.ghost_precision << std::endl;
    out << "pad = " << a.pad << std::endl;
    out << "stride = " << a.stride << std::endl;
    out << "real_length = " << a.real_length << std::endl;
    out << "length = " << a.length << std::endl;
    out << "bytes = " << a.bytes << std::endl;
    out << "norm_bytes = " << a.norm_bytes << std::endl;
    out << "siteSubset = " << a.siteSubset << std::endl;
    out << "siteOrder = " << a.siteOrder << std::endl;
    out << "fieldOrder = " << a.fieldOrder << std::endl;
    out << "gammaBasis = " << a.gammaBasis << std::endl;
    out << "Is composite = " << a.composite_descr.is_composite << std::endl;
    if(a.composite_descr.is_composite)
    {
      out << "Composite Dim = " << a.composite_descr.dim << std::endl;
      out << "Composite Volume = " << a.composite_descr.volume << std::endl;
      out << "Composite Stride = " << a.composite_descr.stride << std::endl;
      out << "Composite Length = " << a.composite_descr.length << std::endl;
    }
    out << "Is component = " << a.composite_descr.is_component << std::endl;
    if(a.composite_descr.is_composite) out << "Component ID = " << a.composite_descr.id << std::endl;
    out << "pc_type = " << a.pc_type << std::endl;
    return out;
  }

} // namespace quda
