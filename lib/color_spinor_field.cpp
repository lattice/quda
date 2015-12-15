#include <color_spinor_field.h>
#include <string.h>
#include <iostream>
#include <typeinfo>
#include <face_quda.h>

namespace quda {

  /*ColorSpinorField::ColorSpinorField() : init(false) {

    }*/

  ColorSpinorParam::ColorSpinorParam(const ColorSpinorField &field) {
    field.fill(*this);
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorParam &param) 
    : LatticeField(param), init(false), v(0), norm(0), even(0), odd(0), composite_descr(param.is_composite, param.composite_dim, param.is_component, param.component_id), components(0)
  {
    create(param.nDim, param.x, param.nColor, param.nSpin, param.twistFlavor, 
	   param.precision, param.pad, param.siteSubset, param.siteOrder, 
	   param.fieldOrder, param.gammaBasis, param.PCtype);
  }

  ColorSpinorField::ColorSpinorField(const ColorSpinorField &field) 
    : LatticeField(field), init(false), v(0), norm(0), even(0), odd(0), composite_descr(field.composite_descr),  components(0)
  {
    create(field.nDim, field.x, field.nColor, field.nSpin, field.twistFlavor, 
	   field.precision, field.pad, field.siteSubset, field.siteOrder, 
	   field.fieldOrder, field.gammaBasis, field.PCtype);
  }

  ColorSpinorField::~ColorSpinorField() {
    destroy();
  }

  static bool createSpinorGhost = true;
  void setGhostSpinor(bool value) { createSpinorGhost = value; }

  void ColorSpinorField::createGhostZone() {

    if (!createSpinorGhost || typeid(*this) == typeid(cpuColorSpinorField)) {
      total_length = length;
      total_norm_length = (precision == QUDA_HALF_PRECISION) ? 2*stride : 0;
      ghost_length = 0;
      ghost_norm_length = 0;
      return;
    }

    if (getVerbosity() == QUDA_DEBUG_VERBOSE) 
      printfQuda("Precision = %d, Subset = %d\n", precision, siteSubset);

    // FIXME - The ghost zone is allocated before we know which
    // operator (and hence number of faces are needed), thus we
    // allocate a ghost zone large enough to cope with the maximum
    // number of faces.  All Wilson-like operators support only
    // involve the excahnge of one face so this is no problem.
    // However, for staggered fermions, we have either nFace=1 or 3,
    // thus we allocated using the latter.  This will artificially
    // raise the GPU memory requirements for naive staggered fermions.
    // One potential future solution may be to separate the ghost zone
    // memory allocation from the field itself, which has other
    // benefits (1. on multi-gpu machines with UVA, we can read the
    // ghost zone directly from the neighbouring field and 2.) we can
    // use a single contiguous buffer for the ghost zone and its norm
    // which will reduce latency for half precision and allow us to
    // enable GPU_COMMS support for half precision).
    int nFaceGhost = (nSpin == 1) ? 3 : 1;

    // For Wilson we have the number of effective faces since the
    // fields are spin projected.
    int num_faces = ((nSpin == 1) ? 2 : 1) * nFaceGhost;
    int num_norm_faces = 2*nFaceGhost;

    // calculate size of ghost zone required
    int ghostVolume = 0;
    //temporal hack
    int dims = nDim == 5 ? (nDim - 1) : nDim;
    int x5   = nDim == 5 ? x[4] : 1; ///includes DW  and non-degenerate TM ghosts
    for (int i=0; i<dims; i++) {
      ghostFace[i] = 0;
      if (commDimPartitioned(i)) {
	ghostFace[i] = 1;
	for (int j=0; j<dims; j++) {
	  if (i==j) continue;
	  ghostFace[i] *= x[j];
	}
	ghostFace[i] *= x5; ///temporal hack : extra dimension for DW ghosts
	if (i==0 && siteSubset != QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
	if (siteSubset == QUDA_FULL_SITE_SUBSET) ghostFace[i] /= 2;
	ghostVolume += ghostFace[i];
      }
      if(i==0){
	ghostOffset[i] = 0;
	ghostNormOffset[i] = 0;
      }else{
	ghostOffset[i] = ghostOffset[i-1] + num_faces*ghostFace[i-1];
	ghostNormOffset[i] = ghostNormOffset[i-1] + num_norm_faces*ghostFace[i-1];
      }

#ifdef MULTI_GPU
      if (getVerbosity() == QUDA_DEBUG_VERBOSE) 
	printfQuda("face %d = %6d commDimPartitioned = %6d ghostOffset = %6d ghostNormOffset = %6d\n", 
		   i, ghostFace[i], commDimPartitioned(i), ghostOffset[i], ghostNormOffset[i]);
#endif
    }//end of outmost for loop
    int ghostNormVolume = num_norm_faces * ghostVolume;
    ghostVolume *= num_faces;

    if (getVerbosity() == QUDA_DEBUG_VERBOSE) 
      printfQuda("Allocated ghost volume = %d, ghost norm volume %d\n", ghostVolume, ghostNormVolume);

    // ghost zones are calculated on c/b volumes
#ifdef MULTI_GPU
    ghost_length = ghostVolume*nColor*nSpin*2; 
    ghost_norm_length = (precision == QUDA_HALF_PRECISION) ? ghostNormVolume : 0;
#else
    ghost_length = 0;
    ghost_norm_length = 0;
#endif

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      total_length = length + 2*ghost_length; // 2 ghost zones in a full field
      total_norm_length = (precision == QUDA_HALF_PRECISION) ? 2*(stride + ghost_norm_length) : 0; // norm length = 2*stride
    } else {
      total_length = length + ghost_length;
      total_norm_length = (precision == QUDA_HALF_PRECISION) ? stride + ghost_norm_length : 0; // norm length = stride
    }

    if (getVerbosity() == QUDA_DEBUG_VERBOSE) {
      printfQuda("ghost length = %lu, ghost norm length = %lu\n", ghost_length, ghost_norm_length);
      printfQuda("total length = %lu, total norm length = %lu\n", total_length, total_norm_length);
    }

  } // createGhostZone

  void ColorSpinorField::create(int Ndim, const int *X, int Nc, int Ns, QudaTwistFlavorType Twistflavor, 
				QudaPrecision Prec, int Pad, QudaSiteSubset siteSubset, 
				QudaSiteOrder siteOrder, QudaFieldOrder fieldOrder, 
				QudaGammaBasis gammaBasis, QudaDWFPCType DWFPC) {
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
    twistFlavor = Twistflavor;

    PCtype = DWFPC;

    precision = Prec;
    volume = 1;
    for (int d=0; d<nDim; d++) {
      x[d] = X[d];
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

    createGhostZone();

    bytes = total_length * precision; // includes pads and ghost zones
    bytes = (siteSubset == QUDA_FULL_SITE_SUBSET && fieldOrder != QUDA_QDPJIT_FIELD_ORDER) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);

    norm_bytes = total_norm_length * sizeof(float);
    norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(norm_bytes/2) : ALIGNMENT_ADJUST(norm_bytes);

    init = true;

//! stuff for deflated solvers (eigenvector sets):
    if(composite_descr.is_composite){

      if(composite_descr.is_component) errorQuda("\nComposite type is not implemented.\n");

      composite_descr.volume   = volume;
      composite_descr.volumeCB = volumeCB;
      composite_descr.stride = stride;
      composite_descr.length = length;
      composite_descr.real_length = real_length;
//multi-gpu:
      composite_descr.total_length      = total_length;
      composite_descr.total_norm_length = total_norm_length;

      composite_descr.ghost_length      = ghost_length;
      composite_descr.ghost_norm_length = ghost_norm_length; 

      composite_descr.bytes       = bytes;
      composite_descr.norm_bytes  = norm_bytes; 
      
      volume *= composite_descr.dim;
      stride *= composite_descr.dim;
      length *= composite_descr.dim;
      real_length *= composite_descr.dim;
      
      total_length *= composite_descr.dim;
      total_norm_length *= composite_descr.dim;

      bytes *= composite_descr.dim;
      norm_bytes *= composite_descr.dim;
//won't be really used.
      ghost_length *= composite_descr.dim;
      ghost_norm_length *= composite_descr.dim;  
    }
    else if(composite_descr.is_component){
      composite_descr.dim = 0;

      composite_descr.volume      = 0;
      composite_descr.volumeCB    = 0;
      composite_descr.stride      = 0;
      composite_descr.length      = 0;
      composite_descr.real_length = 0;
//multi-gpu:
      composite_descr.total_length      = 0;
      composite_descr.total_norm_length = 0;

      composite_descr.ghost_length      = 0;
      composite_descr.ghost_norm_length = 0; 

      composite_descr.bytes       = 0;
      composite_descr.norm_bytes  = 0;
    }

    clearGhostPointers();
    setTuningString();
  }

  void ColorSpinorField::setTuningString() {
    char vol_tmp[TuneKey::volume_n];
    int check;
    check = snprintf(vol_string, TuneKey::volume_n, "%d", x[0]);
    if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
    for (int d=1; d<nDim; d++) {
      strcpy(vol_tmp, vol_string);
      check = snprintf(vol_string, TuneKey::volume_n, "%sx%d", vol_tmp, x[d]);
      if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
    }

    int aux_string_n = TuneKey::aux_n / 2;
    char aux_tmp[aux_string_n];
    check = snprintf(aux_string, aux_string_n, "vol=%d,stride=%d,precision=%d", volume, stride, precision);
    if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");

    if (twistFlavor != QUDA_TWIST_NO && twistFlavor != QUDA_TWIST_INVALID) {
      strcpy(aux_tmp, aux_string);
      check = snprintf(aux_string, aux_string_n, "%s,TwistFlavour=%d", aux_tmp, twistFlavor);
      if (check < 0 || check >= aux_string_n) errorQuda("Error writing aux string");
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

      create(src.nDim, src.x, src.nColor, src.nSpin, src.twistFlavor, 
	     src.precision, src.pad, src.siteSubset, 
	     src.siteOrder, src.fieldOrder, src.gammaBasis, src.PCtype);    
    }
    return *this;
  }

  // Resets the attributes of this field if param disagrees (and is defined)
  void ColorSpinorField::reset(const ColorSpinorParam &param) {

    if (param.nColor != 0) nColor = param.nColor;
    if (param.nSpin != 0) nSpin = param.nSpin;
    if (param.twistFlavor != QUDA_TWIST_INVALID) twistFlavor = param.twistFlavor;

    if (param.PCtype != QUDA_PC_INVALID) PCtype = param.PCtype;

    if (param.precision != QUDA_INVALID_PRECISION)  precision = param.precision;
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
    volumeCB = siteSubset == QUDA_PARITY_SITE_SUBSET ? volume : volume/2;

    if((twistFlavor == QUDA_TWIST_NONDEG_DOUBLET || twistFlavor == QUDA_TWIST_DEG_DOUBLET) && x[4] != 2)
      errorQuda("Must be two flavors for non-degenerate twisted mass spinor (provided with %d)\n", x[4]);

    if (param.pad != 0) pad = param.pad;

    if (param.siteSubset == QUDA_FULL_SITE_SUBSET){
      stride = volume/2 + pad;
      length = 2*stride*nColor*nSpin*2;
    } else if (param.siteSubset == QUDA_PARITY_SITE_SUBSET){
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

    createGhostZone();

    real_length = volume*nColor*nSpin*2;

    bytes = total_length * precision; // includes pads and ghost zones
    bytes = (siteSubset == QUDA_FULL_SITE_SUBSET && fieldOrder != QUDA_QDPJIT_FIELD_ORDER) ? 2*ALIGNMENT_ADJUST(bytes/2) : ALIGNMENT_ADJUST(bytes);

    if (precision == QUDA_HALF_PRECISION) {
      norm_bytes = total_norm_length * sizeof(float);
      norm_bytes = (siteSubset == QUDA_FULL_SITE_SUBSET) ? 2*ALIGNMENT_ADJUST(norm_bytes/2) : ALIGNMENT_ADJUST(norm_bytes);
    } else {
      norm_bytes = 0;
    }

//! for deflated solvers:
    if(composite_descr.is_composite){
        composite_descr.volume            = volume;
        composite_descr.stride            = stride;
        composite_descr.length            = length;
        composite_descr.real_length       = real_length;

        composite_descr.total_length      = total_length;
        composite_descr.total_norm_length = total_norm_length;

        composite_descr.ghost_length      = ghost_length;
        composite_descr.ghost_norm_length = ghost_norm_length;

        composite_descr.bytes             = bytes;
        composite_descr.norm_bytes        = norm_bytes;

        volume            *= composite_descr.dim;
        stride            *= composite_descr.dim;
        length            *= composite_descr.dim;
        real_length       *= composite_descr.dim;

        total_length      *= composite_descr.dim;
        total_norm_length *= composite_descr.dim;
        ghost_length      *= composite_descr.dim;
        ghost_norm_length *= composite_descr.dim;

        bytes      *= composite_descr.dim;
        norm_bytes *= composite_descr.dim;
    }
    else{
        composite_descr.volume            = 0;
        composite_descr.stride            = 0;
        composite_descr.length            = 0;
        composite_descr.real_length       = 0;

        composite_descr.total_length      = 0;
        composite_descr.total_norm_length = 0;

        composite_descr.ghost_length      = 0;
        composite_descr.ghost_norm_length = 0;

        composite_descr.bytes             = 0;
        composite_descr.norm_bytes        = 0;
    }

    if (!init) errorQuda("Shouldn't be resetting a non-inited field\n");

    if (getVerbosity() >= QUDA_DEBUG_VERBOSE) {
      printfQuda("\nPrinting out reset field\n");
      std::cout << *this << std::endl;
      printfQuda("\n");
    }

    setTuningString();
  }

  // Fills the param with the contents of this field
  void ColorSpinorField::fill(ColorSpinorParam &param) const {
    param.location = Location();
    param.nColor = nColor;
    param.nSpin = nSpin;
    param.twistFlavor = twistFlavor;
    param.precision = precision;
    param.nDim = nDim;

    param.is_composite  = composite_descr.is_composite;
    param.composite_dim = composite_descr.dim;
    param.is_component  = false;//always either a regular spinor or a composite object
    param.component_id  = 0;

    memcpy(param.x, x, QUDA_MAX_DIM*sizeof(int));
    param.pad = pad;
    param.siteSubset = siteSubset;
    param.siteOrder = siteOrder;
    param.fieldOrder = fieldOrder;
    param.gammaBasis = gammaBasis;
    param.PCtype = PCtype;
    param.create = QUDA_INVALID_FIELD_CREATE;
  }

  void ColorSpinorField::exchange(void **ghost, void **sendbuf, int nFace) const {

    // FIXME: use LatticeField MsgHandles
    MsgHandle *mh_send_fwd[4];
    MsgHandle *mh_from_back[4];
    MsgHandle *mh_from_fwd[4];
    MsgHandle *mh_send_back[4];
    size_t bytes[4];

    const int Ninternal = 2*nColor*nSpin;
    for (int i=0; i<nDimComms; i++) bytes[i] = siteSubset*nFace*surfaceCB[i]*Ninternal*precision;

    void *send_fwd[4];
    void *send_back[4];
    void *recv_fwd[4];
    void *recv_back[4];

    // leave this option in there just in case
    bool no_comms_fill = false;

    for (int i=0; i<nDimComms; i++) {
      if (Location() == QUDA_CPU_FIELD_LOCATION) {
	if (comm_dim_partitioned(i)) {
	  send_back[i] = sendbuf[2*i + 0];
	  send_fwd[i]  = sendbuf[2*i + 1];
	  recv_fwd[i]  =   ghost[2*i + 1];
	  recv_back[i] =   ghost[2*i + 0];
	} else if (no_comms_fill) {
	  memcpy(ghost[2*i+1], sendbuf[2*i+0], bytes[i]);
	  memcpy(ghost[2*i+0], sendbuf[2*i+1], bytes[i]);
	}

      } else { // FIXME add GPU_COMMS support
	if (comm_dim_partitioned(i)) {
	  send_back[i] = allocatePinned(bytes[i]);
	  send_fwd[i] = allocatePinned(bytes[i]);
	  recv_fwd[i] = allocatePinned(bytes[i]);
	  recv_back[i] = allocatePinned(bytes[i]);
	  cudaMemcpy(send_back[i], sendbuf[2*i + 0], bytes[i], cudaMemcpyDeviceToHost);
	  cudaMemcpy(send_fwd[i],  sendbuf[2*i + 1], bytes[i], cudaMemcpyDeviceToHost);
	} else if (no_comms_fill) {
	  cudaMemcpy(ghost[2*i+1], sendbuf[2*i+0], bytes[i], cudaMemcpyDeviceToDevice);
	  cudaMemcpy(ghost[2*i+0], sendbuf[2*i+1], bytes[i], cudaMemcpyDeviceToDevice);
	}
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
	cudaMemcpy(ghost[2*i+0], recv_back[i], bytes[i], cudaMemcpyHostToDevice);
	cudaMemcpy(ghost[2*i+1], recv_fwd[i], bytes[i], cudaMemcpyHostToDevice);
	freePinned(send_back[i]);
	freePinned(send_fwd[i]);
	freePinned(recv_fwd[i]);
	freePinned(recv_back[i]);
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
	       precision == QUDA_HALF_PRECISION) {
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

  // Set the ghost pointers to NULL.
  // This is a private initialisation routine. 
  void ColorSpinorField::clearGhostPointers() 
  {
    for(int dim=0; dim<QUDA_MAX_DIM; ++dim){
      ghost[dim] = NULL;
      ghostNorm[dim] = NULL;
    }
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
    return ghost_fixme;
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

    ColorSpinorField *field = NULL;
    if (param.location == QUDA_CPU_FIELD_LOCATION) {
      field = new cpuColorSpinorField(param);
    } else if (param.location== QUDA_CUDA_FIELD_LOCATION) {
      field = new cudaColorSpinorField(param);
    } else {
      errorQuda("Invalid field location %d", param.location);
    }

    return field;
  }

  ColorSpinorField* ColorSpinorField::CreateCoarse(const int *geoBlockSize, int spinBlockSize, int Nvec, 
						   QudaFieldLocation new_location) {
    ColorSpinorParam coarseParam(*this);
    for (int d=0; d<nDim; d++) coarseParam.x[d] = x[d]/geoBlockSize[d];
    coarseParam.nSpin = nSpin / spinBlockSize; //for staggered coarseParam.nSpin = nSpin 

    coarseParam.nColor = Nvec;
    coarseParam.siteSubset = QUDA_FULL_SITE_SUBSET; // coarse grid is always full
    coarseParam.create = QUDA_ZERO_FIELD_CREATE;
    
    // if new location is not set, use this->location
    new_location = (new_location == QUDA_INVALID_FIELD_LOCATION) ? Location(): new_location;

    // for GPU fields, always use native ordering to ensure coalescing
    if (new_location == QUDA_CUDA_FIELD_LOCATION) coarseParam.fieldOrder = QUDA_FLOAT2_FIELD_ORDER;

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
						 QudaFieldLocation new_location) {
    ColorSpinorParam fineParam(*this);
    for (int d=0; d<nDim; d++) fineParam.x[d] = x[d] * geoBlockSize[d];
    fineParam.nSpin = nSpin * spinBlockSize;
    fineParam.nColor = Nvec;
    fineParam.siteSubset = QUDA_FULL_SITE_SUBSET; // FIXME fine grid is always full
    fineParam.create = QUDA_ZERO_FIELD_CREATE;
    
    // if new location is not set, use this->location
    new_location = (new_location == QUDA_INVALID_FIELD_LOCATION) ? Location(): new_location;

    // for GPU fields, always use native ordering to ensure coalescing
    if (new_location == QUDA_CUDA_FIELD_LOCATION) {
      fineParam.fieldOrder = (fineParam.nSpin==4 && fineParam.precision!= QUDA_DOUBLE_PRECISION) ?
	QUDA_FLOAT4_FIELD_ORDER : QUDA_FLOAT2_FIELD_ORDER;
    }

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
    out << "precision = " << a.precision << std::endl;
    out << "pad = " << a.pad << std::endl;
    out << "stride = " << a.stride << std::endl;
    out << "real_length = " << a.real_length << std::endl;
    out << "length = " << a.length << std::endl;
    out << "ghost_length = " << a.ghost_length << std::endl;
    out << "total_length = " << a.total_length << std::endl;
    out << "ghost_norm_length = " << a.ghost_norm_length << std::endl;
    out << "total_norm_length = " << a.total_norm_length << std::endl;
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
    out << "PC type = " << a.PCtype << std::endl;
    return out;
  }

} // namespace quda
