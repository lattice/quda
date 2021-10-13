#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <string.h>
#include <iostream>
#include <limits>

#include <color_spinor_field.h>
#include <blas_quda.h>
#include <dslash_quda.h>
#include <device.h>

static bool zeroCopy = false;

namespace quda {

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) :
    ColorSpinorField(param),
    alloc(false),
    init(true)
  {
    // this must come before create
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      v = param.v;
      norm = param.norm;
    }

    create(param.create);

    switch (param.create) {
    case QUDA_NULL_FIELD_CREATE:
    case QUDA_REFERENCE_FIELD_CREATE: break; // do nothing;
    case QUDA_ZERO_FIELD_CREATE: zero(); break;
    case QUDA_COPY_FIELD_CREATE: errorQuda("Copy field create not implemented for this constructor"); break;
    default: errorQuda("Unexpected create type %d", param.create);
    }
  }

  cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) :
    ColorSpinorField(src),
    alloc(false),
    init(true)
  {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, const ColorSpinorParam &param) :
    ColorSpinorField(src),
    alloc(false),
    init(true)
  {
    // can only overide if we are not using a reference or parity special case
    if (param.create != QUDA_REFERENCE_FIELD_CREATE || 
	(param.create == QUDA_REFERENCE_FIELD_CREATE && 
	 src.SiteSubset() == QUDA_FULL_SITE_SUBSET && 
	 param.siteSubset == QUDA_PARITY_SITE_SUBSET && 
	 typeid(src) == typeid(cudaColorSpinorField) ) || 
         (param.create == QUDA_REFERENCE_FIELD_CREATE && (param.is_composite || param.is_component))) {
      reset(param);
    } else {
      errorQuda("Undefined behaviour"); // else silent bug possible?
    }

    // This must be set before create is called
    if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      if (typeid(src) == typeid(cudaColorSpinorField)) {
	v = (void*)src.V();
	norm = (void*)src.Norm();
      } else {
	errorQuda("Cannot reference a non-cuda field");
      }

      if (composite_descr.is_component && !(src.SiteSubset() == QUDA_FULL_SITE_SUBSET && this->SiteSubset() == QUDA_PARITY_SITE_SUBSET)) 
      {//setup eigenvector form the set
        v    = (void*)((char*)v    + composite_descr.id*bytes);         
        norm = (void*)((char*)norm + composite_descr.id*norm_bytes);         
      }
    }

    create(param.create);

    if (param.create == QUDA_NULL_FIELD_CREATE) {
      // do nothing
    } else if (param.create == QUDA_ZERO_FIELD_CREATE) {
      zero();
    } else if (param.create == QUDA_COPY_FIELD_CREATE) {
      copySpinorField(src);
    } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
      // do nothing
    } else {
      errorQuda("CreateType %d not implemented", param.create);
    }

  }

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) :
    ColorSpinorField(src),
    alloc(false),
    init(true)
  {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  ColorSpinorField& cudaColorSpinorField::operator=(const ColorSpinorField &src) {
    if (typeid(src) == typeid(cudaColorSpinorField)) {
      *this = (dynamic_cast<const cudaColorSpinorField&>(src));
    } else if (typeid(src) == typeid(cpuColorSpinorField)) {
      *this = (dynamic_cast<const cpuColorSpinorField&>(src));
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
    return *this;
  }

  cudaColorSpinorField& cudaColorSpinorField::operator=(const cudaColorSpinorField &src) {
    if (&src != this) {
      // keep current attributes unless unset
      if (!ColorSpinorField::init) { // note this will turn a reference field into a regular field
	destroy();
	destroyComms(); // not sure if this necessary
	ColorSpinorField::operator=(src);
	create(QUDA_COPY_FIELD_CREATE);
      }
      copySpinorField(src);
    }
    return *this;
  }

  cudaColorSpinorField& cudaColorSpinorField::operator=(const cpuColorSpinorField &src) {
    // keep current attributes unless unset
    if (!ColorSpinorField::init) { // note this will turn a reference field into a regular field
      destroy();
      ColorSpinorField::operator=(src);
      create(QUDA_COPY_FIELD_CREATE);
    }
    loadSpinorField(src);
    return *this;
  }

  cudaColorSpinorField::~cudaColorSpinorField() {
    destroyComms();
    destroy();
  }

  void cudaColorSpinorField::create(const QudaFieldCreate create) {

    if (siteSubset == QUDA_FULL_SITE_SUBSET && siteOrder != QUDA_EVEN_ODD_SITE_ORDER) {
      errorQuda("Subset not implemented");
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
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
      default:
	errorQuda("Unsupported memory type %d", mem_type);
      }
      alloc = true;
    }

    if (siteSubset == QUDA_FULL_SITE_SUBSET) {
      if(composite_descr.is_composite && (create != QUDA_REFERENCE_FIELD_CREATE)) {
	if(composite_descr.dim <= 0) errorQuda("\nComposite size is not defined\n");
	  
        ColorSpinorParam param;
        param.siteSubset = QUDA_FULL_SITE_SUBSET;
        param.nDim = nDim;
        memcpy(param.x, x, nDim*sizeof(int));
        param.create = QUDA_REFERENCE_FIELD_CREATE;
        param.v = v;
        param.norm = norm;
        param.is_composite   = false;
        param.composite_dim  = 0;
        param.is_component = true;
	param.mem_type = mem_type;

        components.reserve(composite_descr.dim);
        for(int cid = 0; cid < composite_descr.dim; cid++) {
	  param.component_id = cid;
	  components.push_back(new cudaColorSpinorField(*this, param));
        }
      } else {
        // create the associated even and odd subsets
        ColorSpinorParam param;
        param.siteSubset = QUDA_PARITY_SITE_SUBSET;
        param.nDim = nDim;
        memcpy(param.x, x, nDim*sizeof(int));
        param.x[0] /= 2; // set single parity dimensions
        param.create = QUDA_REFERENCE_FIELD_CREATE;
        param.v = v;
        param.norm = norm;
        param.is_composite  = false;
        param.composite_dim = 0;
        param.is_component  = composite_descr.is_component;
        param.component_id  = composite_descr.id;
	param.mem_type = mem_type;

        even = new cudaColorSpinorField(*this, param);
        odd = new cudaColorSpinorField(*this, param);

        // need this hackery for the moment (need to locate the odd pointers half way into the full field)
        // check for special metadata wrapper (look at reference comments in
        // createTexObject() below)
        if (!((uint64_t)v == (uint64_t)(void *)std::numeric_limits<uint64_t>::max()
              || (precision == QUDA_HALF_PRECISION
                  && (uint64_t)norm == (uint64_t)(void *)std::numeric_limits<uint64_t>::max()))) {
          (dynamic_cast<cudaColorSpinorField *>(odd))->v = (void *)((char *)v + bytes / 2);
          if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
            (dynamic_cast<cudaColorSpinorField *>(odd))->norm = (void *)((char *)norm + norm_bytes / 2);
        }
      }
    } else { //siteSubset == QUDA_PARITY_SITE_SUBSET

      //! setup an object for selected eigenvector (the 1st one as a default):
      if (composite_descr.is_composite && (create != QUDA_REFERENCE_FIELD_CREATE)) 
      {
         if(composite_descr.dim <= 0) errorQuda("\nComposite size is not defined\n");
         //if(bytes > 1811939328) warningQuda("\nCUDA API probably won't be able to create texture object for the eigenvector set... Object size is : %u bytes\n", bytes);
         // create the associated even and odd subsets
         ColorSpinorParam param;
         param.siteSubset = QUDA_PARITY_SITE_SUBSET;
         param.nDim = nDim;
         memcpy(param.x, x, nDim*sizeof(int));
         param.create = QUDA_REFERENCE_FIELD_CREATE;
         param.v = v;
         param.norm = norm;
         param.is_composite   = false;
         param.composite_dim  = 0;
         param.is_component = true;
	 param.mem_type = mem_type;

         //reserve eigvector set
         components.reserve(composite_descr.dim);
         //setup volume, [real_]length and stride for a single eigenvector
         for(int cid = 0; cid < composite_descr.dim; cid++)
         {
            param.component_id = cid;
            components.push_back(new cudaColorSpinorField(*this, param));
         }
      }
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if ( !(siteSubset == QUDA_FULL_SITE_SUBSET && composite_descr.is_composite) ) {
	zeroPad();
      } else { //temporary hack for the full spinor field sets, manual zeroPad for each component:
	for(int cid = 0; cid < composite_descr.dim; cid++) {
	  (dynamic_cast<cudaColorSpinorField&>(components[cid]->Even())).zeroPad();
	  (dynamic_cast<cudaColorSpinorField&>(components[cid]->Odd())).zeroPad();
	}
      }
    }
  }

  void cudaColorSpinorField::destroy()
  {
    if (alloc) {
      switch(mem_type) {
      case QUDA_MEMORY_DEVICE:
        pool_device_free(v);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) pool_device_free(norm);
        break;
      case QUDA_MEMORY_MAPPED:
        host_free(v_h);
        if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) host_free(norm_h);
        break;
      default:
        errorQuda("Unsupported memory type %d", mem_type);
      }
    }


    if (composite_descr.is_composite) 
    {
       CompositeColorSpinorField::iterator vec;
       for (vec = components.begin(); vec != components.end(); vec++) delete *vec;
    } 

    if ( siteSubset == QUDA_FULL_SITE_SUBSET && (!composite_descr.is_composite || composite_descr.is_component) ) {
      delete even;
      delete odd;
    }
  }

  void cudaColorSpinorField::backup() const {
    if (backed_up) errorQuda("ColorSpinorField already backed up");
    backup_h = new char[bytes];
    qudaMemcpy(backup_h, v, bytes, qudaMemcpyDefault);
    if (norm_bytes) {
      backup_norm_h = new char[norm_bytes];
      qudaMemcpy(backup_norm_h, norm, norm_bytes, qudaMemcpyDefault);
    }
    backed_up = true;
  }

  void cudaColorSpinorField::restore() const
  {
    if (!backed_up) errorQuda("Cannot restore since not backed up");
    qudaMemcpy(v, backup_h, bytes, qudaMemcpyDefault);
    delete []backup_h;
    if (norm_bytes) {
      qudaMemcpy(norm, backup_norm_h, norm_bytes, qudaMemcpyDefault);
      delete []backup_norm_h;
    }
    backed_up = false;
  }

  void cudaColorSpinorField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    // conditionals based on destructor
    if (is_prefetch_enabled() && alloc && mem_type == QUDA_MEMORY_DEVICE) {
      qudaMemPrefetchAsync(v, bytes, mem_space, stream);
      if ((precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) && norm_bytes > 0)
        qudaMemPrefetchAsync(norm, norm_bytes, mem_space, stream);
    }
  }

  // cuda's floating point format, IEEE-754, represents the floating point
  // zero as 4 zero bytes
  void cudaColorSpinorField::zero() {
    qudaMemsetAsync(v, 0, bytes, device::get_default_stream());
    if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
      qudaMemsetAsync(norm, 0, norm_bytes, device::get_default_stream());
  }

  void cudaColorSpinorField::zeroPad() {

    { // zero initialize the field pads
      size_t pad_bytes = (stride - volumeCB) * precision * fieldOrder;
      int Npad = nColor * nSpin * 2 / fieldOrder;

      if (composite_descr.is_composite && !composite_descr.is_component){//we consider the whole eigenvector set:
        Npad      *= composite_descr.dim;
        pad_bytes /= composite_descr.dim;
      }

      size_t pitch = ((!composite_descr.is_composite || composite_descr.is_component) ? stride : composite_descr.stride)*fieldOrder*precision;
      char   *dst  = (char*)v + ((!composite_descr.is_composite || composite_descr.is_component) ? volumeCB : composite_descr.volumeCB)*fieldOrder*precision;
      if (pad_bytes)
        for (int subset=0; subset<siteSubset; subset++) {
          qudaMemset2DAsync(dst + subset * bytes / siteSubset, pitch, 0, pad_bytes, Npad, device::get_default_stream());
        }
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

  void cudaColorSpinorField::copy(const cudaColorSpinorField &src)
  {
    checkField(*this, src);
    copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
  }

  void cudaColorSpinorField::copySpinorField(const ColorSpinorField &src)
  {
    if (typeid(src) == typeid(cudaColorSpinorField)) { // src is on the device
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else if (typeid(src) == typeid(cpuColorSpinorField)) { // src is on the host
      loadSpinorField(src);
    } else {
      errorQuda("Unknown input ColorSpinorField %s", typeid(src).name());
    }
  }

  void cudaColorSpinorField::loadSpinorField(const ColorSpinorField &src) {

    if ( reorder_location() == QUDA_CPU_FIELD_LOCATION && typeid(src) == typeid(cpuColorSpinorField)) {
      void *buffer = pool_pinned_malloc(bytes + norm_bytes);
      memset(buffer, 0, bytes+norm_bytes); // FIXME (temporary?) bug fix for padding

      copyGenericColorSpinor(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, 0, static_cast<char*>(buffer)+bytes, 0);

      qudaMemcpy(v, buffer, bytes, qudaMemcpyDefault);
      qudaMemcpy(norm, static_cast<char *>(buffer) + bytes, norm_bytes, qudaMemcpyDefault);

      pool_pinned_free(buffer);
    } else if (typeid(src) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(*this, src, QUDA_CUDA_FIELD_LOCATION);
    } else {

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
  }


  void cudaColorSpinorField::saveSpinorField(ColorSpinorField &dest) const {

    if ( reorder_location() == QUDA_CPU_FIELD_LOCATION && typeid(dest) == typeid(cpuColorSpinorField)) {
      void *buffer = pool_pinned_malloc(bytes+norm_bytes);
      qudaMemcpy(buffer, v, bytes, qudaMemcpyDefault);
      qudaMemcpy(static_cast<char *>(buffer) + bytes, norm, norm_bytes, qudaMemcpyDefault);

      copyGenericColorSpinor(dest, *this, QUDA_CPU_FIELD_LOCATION, 0, buffer, 0, static_cast<char*>(buffer)+bytes);
      pool_pinned_free(buffer);
    } else if (typeid(dest) == typeid(cudaColorSpinorField)) {
      copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION);
    } else {

      if (dest.FieldOrder() == QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER) {
	// special case where we use zero-copy memory to read/write directly from application's array
        void *dest_d = get_mapped_device_pointer(dest.V());
        copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dest_d, v);
      } else {
        void *dst = nullptr, *dstNorm = nullptr, *buffer = nullptr;
        if (!zeroCopy) {
          buffer = pool_device_malloc(dest.Bytes()+dest.NormBytes());
          dst = buffer;
          dstNorm = static_cast<char*>(dst) + dest.Bytes();
        } else {
          buffer = pool_pinned_malloc(dest.Bytes()+dest.NormBytes());
          dst = get_mapped_device_pointer(buffer);
          dstNorm = static_cast<char*>(dst)+dest.Bytes();
        }

        copyGenericColorSpinor(dest, *this, QUDA_CUDA_FIELD_LOCATION, dst, 0, dstNorm, 0);

        if (!zeroCopy) {
          qudaMemcpy(dest.V(), dst, dest.Bytes(), qudaMemcpyDefault);
          qudaMemcpy(dest.Norm(), dstNorm, dest.NormBytes(), qudaMemcpyDefault);
        } else {
          qudaDeviceSynchronize();
          memcpy(dest.V(), buffer, dest.Bytes());
          memcpy(dest.Norm(), static_cast<char*>(buffer) + dest.Bytes(), dest.NormBytes());
        }

        if (zeroCopy) pool_pinned_free(buffer);
        else pool_device_free(buffer);
      }
    }

    qudaDeviceSynchronize(); // need to sync before data can be used on CPU
  }

  void cudaColorSpinorField::allocateGhostBuffer(int nFace, bool spin_project) const
  {
    createGhostZone(nFace, spin_project);
    LatticeField::allocateGhostBuffer(ghost_bytes);
  }

  // pack the ghost zone into a contiguous buffer for communications
  void cudaColorSpinorField::packGhost(const int nFace, const QudaParity parity, const int dagger,
                                       const qudaStream_t &stream, MemoryLocation location[2 * QUDA_MAX_DIM],
                                       MemoryLocation location_label, bool spin_project, double a, double b, double c,
                                       int shmem)
  {
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
 
  // send the ghost zone to the host
  void cudaColorSpinorField::sendGhost(void *ghost_spinor, const int dim, const QudaDirection dir,
                                       const qudaStream_t &stream)
  {
    void *gpu_buf
      = (dir == QUDA_BACKWARDS) ? my_face_dim_dir_d[bufferIndex][dim][0] : my_face_dim_dir_d[bufferIndex][dim][1];
    qudaMemcpyAsync(ghost_spinor, gpu_buf, ghost_face_bytes[dim], qudaMemcpyDeviceToHost, stream);
  }

  void cudaColorSpinorField::unpackGhost(const void *ghost_spinor, const int dim, const QudaDirection dir,
                                         const qudaStream_t &stream)
  {
    const void *src = ghost_spinor;
    auto offset = (dir == QUDA_BACKWARDS) ? ghost_offset[dim][0] : ghost_offset[dim][1];
    void *ghost_dst = static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + offset;

    qudaMemcpyAsync(ghost_dst, src, ghost_face_bytes[dim], qudaMemcpyHostToDevice, stream);
  }

  void cudaColorSpinorField::createComms(int nFace, bool spin_project)
  {
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

  void cudaColorSpinorField::pack(int nFace, int parity, int dagger, const qudaStream_t &stream,
                                  MemoryLocation location[2 * QUDA_MAX_DIM], MemoryLocation location_label,
                                  bool spin_project, double a, double b, double c, int shmem)
  {
    createComms(nFace, spin_project); // must call this first

    packGhost(nFace, (QudaParity)parity, dagger, stream, location, location_label, spin_project, a, b, c, shmem);
  }

  void cudaColorSpinorField::gather(int dir, const qudaStream_t &stream)
  {
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

  void cudaColorSpinorField::recvStart(int d, const qudaStream_t &, bool gdr)
  {
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

  void cudaColorSpinorField::sendStart(int d, const qudaStream_t &stream, bool gdr, bool remote_write)
  {
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

  void cudaColorSpinorField::commsStart(int dir, const qudaStream_t &stream, bool gdr_send, bool gdr_recv)
  {
    recvStart(dir, stream, gdr_recv);
    sendStart(dir, stream, gdr_send);
  }

  static bool complete_recv_fwd[QUDA_MAX_DIM] = { };
  static bool complete_recv_back[QUDA_MAX_DIM] = { };
  static bool complete_send_fwd[QUDA_MAX_DIM] = { };
  static bool complete_send_back[QUDA_MAX_DIM] = { };

  int cudaColorSpinorField::commsQuery(int d, const qudaStream_t &, bool gdr_send, bool gdr_recv)
  {
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

  void cudaColorSpinorField::commsWait(int d, const qudaStream_t &, bool gdr_send, bool gdr_recv)
  {
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

  void cudaColorSpinorField::scatter(int dim_dir, const qudaStream_t &stream)
  {
    // note this is scatter centric, so input expects dir=0 (1) is send backwards
    // (forwards) and receive from forwards (backwards), so here we need flip to receive centric

    int dim = dim_dir/2;
    int dir = (dim_dir+1)%2; // dir = 1 - receive from forwards, dir == 0 recive from backwards
    if (!commDimPartitioned(dim)) return;
    if (comm_peer2peer_enabled(dir,dim)) return;

    unpackGhost(from_face_dim_dir_h[bufferIndex][dim][dir], dim, dir == 0 ? QUDA_BACKWARDS : QUDA_FORWARDS, stream);
  }

  void cudaColorSpinorField::exchangeGhost(QudaParity parity, int nFace, int dagger,
                                           const MemoryLocation *pack_destination_, const MemoryLocation *halo_location_,
                                           bool gdr_send, bool gdr_recv, QudaPrecision ghost_precision_) const
  {

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
    const_cast<cudaColorSpinorField&>(*this).createComms(nFace, false);

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
      const_cast<cudaColorSpinorField *>(this)->recvStart(i, device::get_default_stream(), gdr_recv);

    bool sync = pack_host ? true : false; // no p2p if pack_host so we need to synchronize
    // if not p2p in any direction then need to synchronize before MPI
    for (int i=0; i<nDimComms; i++) if (!comm_peer2peer_enabled(0,i) || !comm_peer2peer_enabled(1,i)) sync = true;
    if (sync) qudaDeviceSynchronize(); // need to make sure packing and/or memcpy has finished before kicking off MPI

    for (int p2p=0; p2p<2; p2p++) {
      for (int dim=0; dim<nDimComms; dim++) {
	for (int dir=0; dir<2; dir++) {
	  if ( (comm_peer2peer_enabled(dir,dim) + p2p) % 2 == 0 ) { // issue non-p2p transfers first
            const_cast<cudaColorSpinorField *>(this)->sendStart(2 * dim + dir, device::get_stream(2 * dim + dir),
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
            comms_complete[2 * dim + dir] = const_cast<cudaColorSpinorField *>(this)->commsQuery(
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

  std::ostream& operator<<(std::ostream &out, const cudaColorSpinorField &a) {
    out << (const ColorSpinorField&)a;
    out << "v = " << a.v << std::endl;
    out << "norm = " << a.norm << std::endl;
    out << "alloc = " << a.alloc << std::endl;
    out << "init = " << a.init << std::endl;
    return out;
  }

//! for composite fields:
  cudaColorSpinorField& cudaColorSpinorField::Component(const int idx) const {
    
    if (this->IsComposite()) {
      if (idx < this->CompositeDim()) {//setup eigenvector form the set
        return *(dynamic_cast<cudaColorSpinorField*>(components[idx])); 
      }
      else{
        errorQuda("Incorrect component index...");
      }
    }
    errorQuda("Cannot get requested component");
    exit(-1);
  }

  // copyCuda currently cannot not work with set of spinor fields..
  void cudaColorSpinorField::CopySubset(cudaColorSpinorField &, const int, const int) const
  {
#if 0
    if (first_element < 0) errorQuda("\nError: trying to set negative first element.\n");
    if (siteSubset == QUDA_PARITY_SITE_SUBSET && this->EigvId() == -1) {
      if (first_element == 0 && range == this->EigvDim())
      {
        if (range != dst.EigvDim())errorQuda("\nError: eigenvector range to big.\n");
        checkField(dst, *this);
        copyCuda(dst, *this);
      }
      else if ((first_element+range) < this->EigvDim()) 
      {//setup eigenvector subset

        cudaColorSpinorField *eigv_subset;

        ColorSpinorParam param;

        param.nColor = nColor;
        param.nSpin = nSpin;
        param.twistFlavor = twistFlavor;
        param.precision = precision;
        param.nDim = nDim;
        param.pad = pad;
        param.siteSubset = siteSubset;
        param.siteOrder = siteOrder;
        param.fieldOrder = fieldOrder;
        param.gammaBasis = gammaBasis;
        memcpy(param.x, x, nDim*sizeof(int));
        param.create = QUDA_REFERENCE_FIELD_CREATE;
 
        param.eigv_dim  = range;
        param.eigv_id   = -1;
        param.v = (void*)((char*)v + first_element*eigv_bytes);
        param.norm = (void*)((char*)norm + first_element*eigv_norm_bytes);

        eigv_subset = new cudaColorSpinorField(param);

        //Not really needed:
        eigv_subset->eigenvectors.reserve(param.eigv_dim);
        for (int id = first_element; id < (first_element+range); id++)
        {
            param.eigv_id = id;
            eigv_subset->eigenvectors.push_back(new cudaColorSpinorField(*this, param));
        }
        checkField(dst, *eigv_subset);
        copyCuda(dst, *eigv_subset);

        delete eigv_subset;
      } else {
        errorQuda("Incorrect eigenvector dimension...");
      }
    } else{
      errorQuda("Eigenvector must be a parity spinor");
      exit(-1);
    }
#endif
  }

  void cudaColorSpinorField::Source(const QudaSourceType sourceType, const int st, const int s, const int c) {
    ColorSpinorParam param(*this);
    param.fieldOrder = QUDA_SPACE_SPIN_COLOR_FIELD_ORDER;
    param.location = QUDA_CPU_FIELD_LOCATION;
    param.setPrecision((param.Precision() == QUDA_HALF_PRECISION || param.Precision() == QUDA_QUARTER_PRECISION) ?
                         QUDA_SINGLE_PRECISION :
                         param.Precision());
    param.create = (sourceType == QUDA_POINT_SOURCE ? QUDA_ZERO_FIELD_CREATE : QUDA_NULL_FIELD_CREATE);

    // since CPU fields cannot be low precision, use single precision instead
    if (precision < QUDA_SINGLE_PRECISION) param.setPrecision(QUDA_SINGLE_PRECISION, QUDA_INVALID_PRECISION, false);

    cpuColorSpinorField tmp(param);
    tmp.Source(sourceType, st, s, c);
    *this = tmp;
  }

  void cudaColorSpinorField::PrintVector(unsigned int i) const { genericCudaPrintVector(*this, i); }

  const void *cudaColorSpinorField::Ghost2() const
  {
    if (bufferIndex < 2) {
      return ghost_recv_buffer_d[bufferIndex];
    } else {
      return ghost_pinned_recv_buffer_hd[bufferIndex % 2];
    }
  }

  void cudaColorSpinorField::copy_to_buffer(void *buffer) const
  {
    qudaMemcpy(buffer, v, bytes, qudaMemcpyDeviceToHost);
    if (precision < QUDA_SINGLE_PRECISION) {
      qudaMemcpy(static_cast<char *>(buffer) + bytes, norm, norm_bytes, qudaMemcpyDeviceToHost);
    }
  }

  void cudaColorSpinorField::copy_from_buffer(void *buffer)
  {
    qudaMemcpy(v, buffer, bytes, qudaMemcpyHostToDevice);
    if (precision < QUDA_SINGLE_PRECISION) {
      qudaMemcpy(norm, static_cast<char *>(buffer) + bytes, norm_bytes, qudaMemcpyHostToDevice);
    }
  }

} // namespace quda
