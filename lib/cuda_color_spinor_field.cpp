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

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorParam &param) : ColorSpinorField(param)
  {
    create(param.create);

    switch (param.create) {
    case QUDA_NULL_FIELD_CREATE:
    case QUDA_REFERENCE_FIELD_CREATE: break; // do nothing;
    case QUDA_ZERO_FIELD_CREATE: zero(); break;
    case QUDA_COPY_FIELD_CREATE: errorQuda("Copy field create not implemented for this constructor"); break;
    default: errorQuda("Unexpected create type %d", param.create);
    }
  }

  cudaColorSpinorField::cudaColorSpinorField(const cudaColorSpinorField &src) : ColorSpinorField(src)
  {
    create(QUDA_COPY_FIELD_CREATE);
    copySpinorField(src);
  }

  // creates a copy of src, any differences defined in param
  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src, const ColorSpinorParam &param) : ColorSpinorField(src)
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

  cudaColorSpinorField::cudaColorSpinorField(const ColorSpinorField &src) : ColorSpinorField(src)
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

    if (composite_descr.is_composite) {
      CompositeColorSpinorField::iterator vec;
      for (vec = components.begin(); vec != components.end(); vec++) delete *vec;
    } 

    if (siteSubset == QUDA_FULL_SITE_SUBSET && (!composite_descr.is_composite || composite_descr.is_component) ) {
      delete even;
      delete odd;
    }
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

  // send the ghost zone to the host

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

} // namespace quda
