#include <color_spinor_field.h>
#include <string.h>
#include <iostream>
#include <typeinfo>

namespace quda {

  /*ColorSpinorField::ColorSpinorField() : init(false) {

    }*/

  ColorSpinorParam::ColorSpinorParam(const ColorSpinorField &field) : LatticeFieldParam()  {
    field.fill(*this);
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param)
    : LatticeField(param), init(false), ghost_precision_allocated(QUDA_INVALID_PRECISION), v(0), norm(0),
      ghost( ), ghostNorm( ), ghostFace( ),
      bytes(0), norm_bytes(0), even(0), odd(0),
      composite_descr(param.is_composite, param.composite_dim, param.is_component, param.component_id),
      components(0)
  {
    if (param.create == QUDA_INVALID_FIELD_CREATE) errorQuda("Invalid create type");
    for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) ghost_buf[i] = nullptr;
    create(param.nDim, param.x, param.nColor, param.nSpin, param.nVec, param.twistFlavor, param.Precision(), param.pad,
           param.siteSubset, param.siteOrder, param.fieldOrder, param.gammaBasis, param.pc_type, param.suggested_parity);
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorField &field)
    : LatticeField(field), init(false), ghost_precision_allocated(QUDA_INVALID_PRECISION), v(0), norm(0),
      ghost( ), ghostNorm( ), ghostFace( ),
      bytes(0), norm_bytes(0), even(0), odd(0),
     composite_descr(field.composite_descr), components(0)
  {
    for (int i = 0; i < 2 * QUDA_MAX_DIM; i++) ghost_buf[i] = nullptr;
    create(field.nDim, field.x, field.nColor, field.nSpin, field.nVec, field.twistFlavor, field.Precision(), field.pad,
           field.siteSubset, field.siteOrder, field.fieldOrder, field.gammaBasis, field.pc_type, field.suggested_parity);
  }

  ColorSpinorField::~ColorSpinorField() {
    destroy();
  }

  void ColorSpinorField::createGhostZone(int nFace, bool spin_project) const {

    if ( typeid(*this) == typeid(cpuColorSpinorField) || ghost_precision_allocated == ghost_precision ) return;

    // For Wilson we half the number of effective faces if the fields are spin projected.
    int num_faces = ((nSpin == 4 && spin_project) ? 1 : 2) * nFace;
    int num_norm_faces = 2*nFace;

    // calculate size of ghost zone required
    int ghostVolume = 0;
    int dims = nDim == 5 ? (nDim - 1) : nDim;
    int x5   = nDim == 5 ? x[4] : 1; ///includes DW  and non-degenerate TM ghosts
    for (int i=0; i<dims; i++) {
      ghostFace[i] = 0;
      if (comm_dim_partitioned(i)) {
	ghostFace[i] = 1;
	for (int j=0; j<dims; j++) {
	  if (i==j) continue;
	  ghostFace[i] *= x[j];
	}
	ghostFace[i] *= x5; ///temporal hack : extra dimension for DW ghosts
	if (i==0 && siteSubset != QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
	ghostVolume += ghostFace[i];
      }
      if (i==0) {
	ghostOffset[i][0] = 0;
      } else {
        if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) {
          ghostOffset[i][0] = (ghostNormOffset[i-1][1] + num_norm_faces*ghostFace[i-1]/2)*sizeof(float)/ghost_precision;
          // Ensure that start of ghostOffset is aligned on four word boundaries (check if this is needed)
          ghostOffset[i][0] = 4*((ghostOffset[i][0] + 3)/4);
        } else {
	  ghostOffset[i][0] = ghostOffset[i-1][0] + num_faces*ghostFace[i-1]*nSpin*nColor*2;
        }
      }

      if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) {
        ghostNormOffset[i][0] = (ghostOffset[i][0] + (num_faces*ghostFace[i]*nSpin*nColor*2/2))*ghost_precision/sizeof(float);
        ghostOffset[i][1] = (ghostNormOffset[i][0] + num_norm_faces*ghostFace[i]/2)*sizeof(float)/ghost_precision;
	// Ensure that start of ghostOffset is aligned on four word boundaries (check if this is needed)
        ghostOffset[i][1] = 4*((ghostOffset[i][1] + 3)/4);
        ghostNormOffset[i][1] = (ghostOffset[i][1] + (num_faces*ghostFace[i]*nSpin*nColor*2/2))*ghost_precision/sizeof(float);
      } else {
        ghostOffset[i][1] = ghostOffset[i][0] + num_faces*ghostFace[i]*nSpin*nColor*2/2;
      }

      int Nint = nColor * nSpin * 2 / (nSpin == 4 && spin_project ? 2 : 1); // number of internal degrees of freedom
      ghost_face_bytes[i] = nFace*ghostFace[i]*Nint*ghost_precision;
      if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) ghost_face_bytes[i] += nFace*ghostFace[i]*sizeof(float);

      if(GhostOffset(i,0)%FieldOrder()) errorQuda("ghostOffset(%d,0) %d is not a multiple of FloatN\n", i, GhostOffset(i,0));
      if(GhostOffset(i,1)%FieldOrder()) errorQuda("ghostOffset(%d,1) %d is not a multiple of FloatN\n", i, GhostOffset(i,1));

      ghostFaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET ? ghostFace[i] / 2 : ghostFace[i]);
    } // dim

    int ghostNormVolume = num_norm_faces * ghostVolume;
    ghostVolume *= num_faces;

    size_t ghost_length = ghostVolume*nColor*nSpin*2;
    size_t ghost_norm_length = (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) ? ghostNormVolume : 0;

    ghost_bytes = (size_t)ghost_length*ghost_precision;
    if (ghost_precision == QUDA_HALF_PRECISION || ghost_precision == QUDA_QUARTER_PRECISION) ghost_bytes += ghost_norm_length*sizeof(float);
    if (isNative()) ghost_bytes = ALIGNMENT_ADJUST(ghost_bytes);

    { // compute temporaries needed by dslash and packing kernels
      auto &X = dslash_constant.X;
      for (int dim=0; dim<nDim; dim++) X[dim] = x[dim];
      for (int dim=nDim; dim<QUDA_MAX_DIM; dim++) X[dim] = 1;
      if (siteSubset == QUDA_PARITY_SITE_SUBSET) X[0] = 2*X[0];

      for (int i=0; i<nDim; i++) dslash_constant.Xh[i] = X[i]/2;

      dslash_constant.Ls = X[4];
      dslash_constant.volume_4d_cb = volumeCB / (nDim == 5 ? x[4] : 1);
      dslash_constant.volume_4d = 2 * dslash_constant.volume_4d_cb;

      int face[4];
      for (int dim=0; dim<4; dim++) {
        for (int j=0; j<4; j++) face[j] = X[j];
        face[dim] = nFace;
        dslash_constant.face_X[dim] = face[0];
        dslash_constant.face_Y[dim] = face[1];
        dslash_constant.face_Z[dim] = face[2];
        dslash_constant.face_T[dim] = face[3];
        dslash_constant.face_XY[dim] = dslash_constant.face_X[dim] * face[1];
        dslash_constant.face_XYZ[dim] = dslash_constant.face_XY[dim] * face[2];
        dslash_constant.face_XYZT[dim] = dslash_constant.face_XYZ[dim] * face[3];
      }

      dslash_constant.Vh = (X[3]*X[2]*X[1]*X[0])/2;
      dslash_constant.ghostFace[0] = X[1] * X[2] * X[3];
      dslash_constant.ghostFace[1] = X[0] * X[2] * X[3];
      dslash_constant.ghostFace[2] = X[0] * X[1] * X[3];
      dslash_constant.ghostFace[3] = X[0] * X[1] * X[2];
      for (int d = 0; d < 4; d++) dslash_constant.ghostFaceCB[d] = dslash_constant.ghostFace[d] / 2;

      dslash_constant.X2X1 = X[1]*X[0];
      dslash_constant.X3X2X1 = X[2]*X[1]*X[0];
      dslash_constant.X4X3X2X1 = X[3] * X[2] * X[1] * X[0];
      dslash_constant.X2X1mX1 = (X[1]-1)*X[0];
      dslash_constant.X3X2X1mX2X1 = (X[2]-1)*X[1]*X[0];
      dslash_constant.X4X3X2X1mX3X2X1 = (X[3]-1)*X[2]*X[1]*X[0];
      dslash_constant.X5X4X3X2X1mX4X3X2X1 = (X[4] - 1) * X[3] * X[2] * X[1] * X[0];
      dslash_constant.X4X3X2X1hmX3X2X1h = dslash_constant.X4X3X2X1mX3X2X1/2;

      // used by indexFromFaceIndexStaggered
      dslash_constant.dims[0][0]=X[1];
      dslash_constant.dims[0][1]=X[2];
      dslash_constant.dims[0][2]=X[3];

      dslash_constant.dims[1][0]=X[0];
      dslash_constant.dims[1][1]=X[2];
      dslash_constant.dims[1][2]=X[3];

      dslash_constant.dims[2][0]=X[0];
      dslash_constant.dims[2][1]=X[1];
      dslash_constant.dims[2][2]=X[3];

      dslash_constant.dims[3][0]=X[0];
      dslash_constant.dims[3][1]=X[1];
      dslash_constant.dims[3][2]=X[2];
    }
    ghost_precision_allocated = ghost_precision;

  } // createGhostZone

  void ColorSpinorField::create(int Ndim, const int *X, int Nc, int Ns, int Nvec, QudaTwistFlavorType Twistflavor,
                                QudaPrecision Prec, int Pad, QudaSiteSubset siteSubset, QudaSiteOrder siteOrder,
                                QudaFieldOrder fieldOrder, QudaGammaBasis gammaBasis, QudaPCType pc_type,
                                QudaParity suggested_parity)
  {
    this->siteSubset = siteSubset;
    this->siteOrder = siteOrder;
    this->fieldOrder = fieldOrder;
    this->gammaBasis = gammaBasis;

    if (Ndim > QUDA_MAX_DIM){
      errorQuda("Number of dimensions nDim = %d too great", Ndim);
    }
    nDim = Ndim;
    nColor = Nc;
    nSpin = Ns;
    nVec = Nvec;
    twistFlavor = Twistflavor;

    this->pc_type = pc_type;
    this->suggested_parity = suggested_parity;

    precision = Prec;
    // Copy all data in X
    for (int d = 0; d < QUDA_MAX_DIM; d++) x[d] = X[d];
    volume = 1;
    for (int d=0; d<nDim; d++) {
      volume *= x[d];
    }
    volumeCB = siteSubset == QUDA_PARITY_SITE_SUBSET ? volume : volume/2;

   if((twistFlavor == QUDA_TWIST_NONDEG_DOUBLET || twistFlavor == QUDA_TWIST_DEG_DOUBLET) && x[4] != 2)
     errorQuda("Must be two flavors for non-degenerate twisted mass spinor (while provided with %d number of components)\n", x[4]);//two flavors

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
    if (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);

    if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
      norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET ? 2*stride : stride) * sizeof(float);
      if (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(norm_bytes/2) : ALIGNMENT_ADJUST(norm_bytes);
    } else {
      norm_bytes = 0;
    }

    init = true;

//! stuff for deflated solvers (eigenvector sets):
    if (composite_descr.is_composite) {

      if (composite_descr.is_component) errorQuda("\nComposite type is not implemented.\n");

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
    }  else if (composite_descr.is_component) {
      composite_descr.dim = 0;

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
      int aux_string_n = TuneKey::aux_n / 2;
      char aux_tmp[aux_string_n];
      int check = snprintf(aux_string, aux_string_n, "vol=%lu,stride=%lu,precision=%d,Ns=%d,Nc=%d", volume, stride,
                           precision, nSpin, nColor);
      if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
      if (twistFlavor != QUDA_TWIST_NO && twistFlavor != QUDA_TWIST_INVALID) {
        strcpy(aux_tmp, aux_string);
        check = snprintf(aux_string, aux_string_n, "%s,TwistFlavour=%d", aux_tmp, twistFlavor);
        if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
      }
    }
  }

  void ColorSpinorField::destroy() {
    init = false;
  }

  ColorSpinorField& ColorSpinorField::operator=(const ColorSpinorField &src) {
    if (&src != this) {
      if(src.composite_descr.is_composite){
        this->composite_descr.is_composite = true;
        this->composite_descr.dim          = src.composite_descr.dim;
        this->composite_descr.is_component = false;
        this->composite_descr.id           = 0;
      }
      else if(src.composite_descr.is_component){
        this->composite_descr.is_composite = false;
        this->composite_descr.dim          = 0;
        //this->composite_descr.is_component = false;
        //this->composite_descr.id           = 0;
      }

      create(src.nDim, src.x, src.nColor, src.nSpin, src.nVec, src.twistFlavor, src.precision, src.pad, src.siteSubset,
             src.siteOrder, src.fieldOrder, src.gammaBasis, src.pc_type, src.suggested_parity);
    }
    return *this;
  }

  // Resets the attributes of this field if param disagrees (and is defined)
  void ColorSpinorField::reset(const ColorSpinorParam &param)
  {
    if (param.nColor != 0) nColor = param.nColor;
    if (param.nSpin != 0) nSpin = param.nSpin;
    if (param.nVec != 0) nVec = param.nVec;
    if (param.twistFlavor != QUDA_TWIST_INVALID) twistFlavor = param.twistFlavor;

    if (param.pc_type != QUDA_PC_INVALID) pc_type = param.pc_type;
    if (param.suggested_parity != QUDA_INVALID_PARITY) suggested_parity = param.suggested_parity;

    if (param.Precision() != QUDA_INVALID_PRECISION) precision = param.Precision();
    if (param.GhostPrecision() != QUDA_INVALID_PRECISION) ghost_precision = param.GhostPrecision();
    if (param.nDim != 0) nDim = param.nDim;

    composite_descr.is_composite     = param.is_composite;
    composite_descr.is_component     = param.is_component;
    composite_descr.dim              = param.is_composite ? param.composite_dim : 0;
    composite_descr.id               = param.component_id;

    volume = 1;
    for (int d=0; d<nDim; d++) {
      if (param.x[d] != 0) x[d] = param.x[d];
      volume *= x[d];
    }
    volumeCB = param.siteSubset == QUDA_PARITY_SITE_SUBSET ? volume : volume/2;

    if((twistFlavor == QUDA_TWIST_NONDEG_DOUBLET || twistFlavor == QUDA_TWIST_DEG_DOUBLET) && x[4] != 2)
      errorQuda("Must be two flavors for non-degenerate twisted mass spinor (provided with %d)\n", x[4]);

    if (param.pad != 0) pad = param.pad;

    if (param.siteSubset == QUDA_FULL_SITE_SUBSET) {
      stride = volume/2 + pad;
      length = 2*stride*nColor*nSpin*2;
    } else if (param.siteSubset == QUDA_PARITY_SITE_SUBSET) {
      stride = volume + pad;
      length = stride*nColor*nSpin*2;
    } else {
      //errorQuda("SiteSubset not defined %d", param.siteSubset);
      //do nothing, not an error (can't remember why - need to document this sometime! )
    }

    if (param.siteSubset != QUDA_INVALID_SITE_SUBSET) siteSubset = param.siteSubset;
    if (param.siteOrder != QUDA_INVALID_SITE_ORDER) siteOrder = param.siteOrder;
    if (param.fieldOrder != QUDA_INVALID_FIELD_ORDER) fieldOrder = param.fieldOrder;
    if (param.gammaBasis != QUDA_INVALID_GAMMA_BASIS) gammaBasis = param.gammaBasis;

    real_length = volume*nColor*nSpin*2;

    bytes = (size_t)length * precision; // includes pads
    if (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);

    if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
      norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET ? 2*stride : stride) * sizeof(float);
      if (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(norm_bytes/2) : ALIGNMENT_ADJUST(norm_bytes);
    } else {
      norm_bytes = 0;
    }

    //! for deflated solvers:
    if (composite_descr.is_composite) {
      composite_descr.volume            = volume;
      composite_descr.stride            = stride;
      composite_descr.length            = length;
      composite_descr.real_length       = real_length;
      composite_descr.bytes             = bytes;
      composite_descr.norm_bytes        = norm_bytes;

      volume            *= composite_descr.dim;
      stride            *= composite_descr.dim;
      length            *= composite_descr.dim;
      real_length       *= composite_descr.dim;

      bytes      *= composite_descr.dim;
      norm_bytes *= composite_descr.dim;
    } else {
      composite_descr.volume            = 0;
      composite_descr.stride            = 0;
      composite_descr.length            = 0;
      composite_descr.real_length       = 0;
      composite_descr.bytes             = 0;
      composite_descr.norm_bytes        = 0;
    }

    if (!init) errorQuda("Shouldn't be resetting a non-inited field\n");

    setTuningString();
  }

  // Fills the param with the contents of this field
  void ColorSpinorField::fill(ColorSpinorParam &param) const {
    param.location = Location();
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
    // to false means we aggregate all cudaMemcpys which reduces
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
	    qudaMemcpy(send_back[i], sendbuf[2*i + 0], bytes[i], cudaMemcpyDeviceToHost);
	    qudaMemcpy(send_fwd[i],  sendbuf[2*i + 1], bytes[i], cudaMemcpyDeviceToHost);
	  }
	} else if (no_comms_fill) {
	  qudaMemcpy(ghost[2*i+1], sendbuf[2*i+0], bytes[i], cudaMemcpyDeviceToDevice);
	  qudaMemcpy(ghost[2*i+0], sendbuf[2*i+1], bytes[i], cudaMemcpyDeviceToDevice);
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
	qudaMemcpy(total_send, send_ptr, total_bytes, cudaMemcpyDeviceToHost);
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
	  qudaMemcpy(ghost[2*i+0], recv_back[i], bytes[i], cudaMemcpyHostToDevice);
	  qudaMemcpy(ghost[2*i+1], recv_fwd[i], bytes[i], cudaMemcpyHostToDevice);
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
	qudaMemcpy(ghost_ptr, total_recv, total_bytes, cudaMemcpyHostToDevice);
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

  bool ColorSpinorField::isNative() const {
    if (precision == QUDA_DOUBLE_PRECISION) {
      if (fieldOrder  == QUDA_FLOAT2_FIELD_ORDER) return true;
    } else if (precision == QUDA_SINGLE_PRECISION ||
	       (precision == QUDA_HALF_PRECISION && nColor == 3) ||
         (precision == QUDA_QUARTER_PRECISION && nColor == 3)) {
      if (nSpin == 4) {
	if (fieldOrder == QUDA_FLOAT4_FIELD_ORDER) return true;
      } else if (nSpin == 2) {
	if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) return true;
      } else if (nSpin == 1) {
	if (fieldOrder == QUDA_FLOAT2_FIELD_ORDER) return true;
      }
    }
    return false;
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

  ColorSpinorField& ColorSpinorField::Component(const int idx) {
    if (this->IsComposite()) {
      if (idx < this->CompositeDim()) {  //  setup eigenvector form the set
        return *(dynamic_cast<ColorSpinorField*>(components[idx]));
      }
      else{
        errorQuda("Incorrect component index...");
      }
    }
    errorQuda("Cannot get requested component");
    exit(-1);
  }

  ColorSpinorField& ColorSpinorField::Component(const int idx) const {
    if (this->IsComposite()) {
      if (idx < this->CompositeDim()) {  //  setup eigenvector form the set
        return *(dynamic_cast<ColorSpinorField*>(components[idx]));
      }
      else{
        errorQuda("Incorrect component index...");
      }
    }
    errorQuda("Cannot get requested component");
    exit(-1);
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

  ColorSpinorField* ColorSpinorField::Create(const ColorSpinorParam &param) {

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

  ColorSpinorField* ColorSpinorField::Create(const ColorSpinorField &src, const ColorSpinorParam &param) {

    ColorSpinorField *field = nullptr;
    if (param.location == QUDA_CPU_FIELD_LOCATION) {
      field = new cpuColorSpinorField(src, param);
    } else if (param.location== QUDA_CUDA_FIELD_LOCATION) {
      field = new cudaColorSpinorField(src, param);
    } else {
      errorQuda("Invalid field location %d", param.location);
    }

    return field;
  }

  ColorSpinorField* ColorSpinorField::CreateAlias(const ColorSpinorParam &param_)
  {
    if (param_.Precision() > precision) errorQuda("Cannot create an alias to source with lower precision than the alias");
    ColorSpinorParam param(param_);
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.v = V();

    // if norm field in the source exists, use it, else use the second
    // half of main field for norm storage, ensuring that the start of
    // the norm field is on an alignment boundary if we're using an
    // internal field
    if (param.Precision() < QUDA_SINGLE_PRECISION) {
      auto norm_offset = (isNative() || fieldOrder == QUDA_FLOAT2_FIELD_ORDER) ?
        (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(Bytes()/4) : ALIGNMENT_ADJUST(Bytes()/2) : 0;
      param.norm = Norm() ? Norm() : static_cast<char*>(V()) + norm_offset;
    }

    auto alias = ColorSpinorField::Create(param);

    if (alias->Bytes() > Bytes()) errorQuda("Alias footprint %lu greater than source %lu", alias->Bytes(), Bytes());
    if (alias->Precision() < QUDA_SINGLE_PRECISION) {
      // check that norm does not overlap with body
      if (static_cast<char*>(alias->V()) + alias->Bytes() > alias->Norm())
        errorQuda("Overlap between alias body and norm");
      // check that norm does fall off the end
      if (static_cast<char*>(alias->Norm()) + alias->NormBytes() > static_cast<char*>(V()) + Bytes())
        errorQuda("Norm is not contained in the srouce field");
    }

    return alias;
  }

  ColorSpinorField* ColorSpinorField::CreateCoarse(const int *geoBlockSize, int spinBlockSize, int Nvec,
                                                   QudaPrecision new_precision, QudaFieldLocation new_location,
                                                   QudaMemoryType new_mem_type) {
    ColorSpinorParam coarseParam(*this);
    for (int d=0; d<nDim; d++) coarseParam.x[d] = x[d]/geoBlockSize[d];
    coarseParam.nSpin = (nSpin == 1) ? 2 : (nSpin / spinBlockSize); // coarsening staggered check

    coarseParam.nColor = Nvec;
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
      fineParam.fieldOrder = (fineParam.nSpin==4 && fineParam.Precision()!= QUDA_DOUBLE_PRECISION) ?
	QUDA_FLOAT4_FIELD_ORDER : QUDA_FLOAT2_FIELD_ORDER;
    } else {
      fineParam.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    }

    fineParam.setPrecision(new_precision);

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

  std::ostream& operator<<(std::ostream &out, const ColorSpinorField &a) {
    out << "typedid = " << typeid(a).name() << std::endl;
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
