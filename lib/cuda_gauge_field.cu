#include <string.h>
#include <gauge_field.h>
#include <face_quda.h>
#include <typeinfo>
#include <misc_helpers.h>
#include <blas_quda.h>

namespace quda {

  cudaGaugeField::cudaGaugeField(const GaugeFieldParam &param) :
    GaugeField(param), gauge(0), even(0), odd(0)
  {
    if ((order == QUDA_QDP_GAUGE_ORDER || order == QUDA_QDPJIT_GAUGE_ORDER) &&
        create != QUDA_REFERENCE_FIELD_CREATE) {
      errorQuda("QDP ordering only supported for reference fields");
    }

    if (order == QUDA_QDP_GAUGE_ORDER ||
	order == QUDA_TIFR_GAUGE_ORDER || order == QUDA_TIFR_PADDED_GAUGE_ORDER ||
	order == QUDA_BQCD_GAUGE_ORDER || order == QUDA_CPS_WILSON_GAUGE_ORDER)
      errorQuda("Field ordering %d presently disabled for this type", order);

#ifdef MULTI_GPU
    if (link_type != QUDA_ASQTAD_MOM_LINKS &&
	ghostExchange == QUDA_GHOST_EXCHANGE_PAD &&
	isNative()) {
      bool pad_check = true;
      for (int i=0; i<nDim; i++) {
	// when we have coarse links we need to double the pad since we're storing forwards and backwards links
	int minimum_pad = nFace*surfaceCB[i] * (geometry == QUDA_COARSE_GEOMETRY ? 2 : 1);
	if (pad < minimum_pad) pad_check = false;
	if (!pad_check)
	  errorQuda("cudaGaugeField being constructed with insufficient padding (%d < %d)\n", pad, minimum_pad);
      }
    }
#endif

    if(create != QUDA_NULL_FIELD_CREATE &&
        create != QUDA_ZERO_FIELD_CREATE &&
        create != QUDA_REFERENCE_FIELD_CREATE){
      errorQuda("ERROR: create type(%d) not supported yet\n", create);
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      gauge = pool_device_malloc(bytes);
      if (create == QUDA_ZERO_FIELD_CREATE) cudaMemset(gauge, 0, bytes);
    } else {
      gauge = param.gauge;
    }

    if ( !isNative() ) {
      for (int i=0; i<nDim; i++) {
        size_t nbytes = nFace * surface[i] * nInternal * precision;
        ghost[i] = nbytes ? pool_device_malloc(nbytes) : nullptr;
	ghost[i+4] = (nbytes && geometry == QUDA_COARSE_GEOMETRY) ? pool_device_malloc(nbytes) : nullptr;
      }
    }

    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD) {
      if (create == QUDA_REFERENCE_FIELD_CREATE) exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);
    }

    even = gauge;
    odd = (char*)gauge + bytes/2; 

#ifdef USE_TEXTURE_OBJECTS
    createTexObject(tex, gauge, true);
    createTexObject(evenTex, even, false);
    createTexObject(oddTex, odd, false);
    if(reconstruct == QUDA_RECONSTRUCT_13 || reconstruct == QUDA_RECONSTRUCT_9)
    {  // Create texture objects for the phases
      bool isPhase = true;
      createTexObject(phaseTex, (char*)gauge + phase_offset, true, isPhase);
      createTexObject(evenPhaseTex, (char*)even + phase_offset, false, isPhase);
      createTexObject(oddPhaseTex, (char*)odd + phase_offset, false, isPhase);
    }
#endif

  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaGaugeField::createTexObject(cudaTextureObject_t &tex, void *field, bool full, bool isPhase) {

    if( isNative() ){
      // create the texture for the field components
      cudaChannelFormatDesc desc;
      memset(&desc, 0, sizeof(cudaChannelFormatDesc));
      if (precision == QUDA_SINGLE_PRECISION) desc.f = cudaChannelFormatKindFloat;
      else desc.f = cudaChannelFormatKindSigned; // half is short, double is int2

      if(isPhase){
        if(precision == QUDA_DOUBLE_PRECISION){
          desc.x = 8*sizeof(int);
          desc.y = 8*sizeof(int);
          desc.z = 0;
          desc.w = 0;
        }else{
          desc.x = 8*precision;
          desc.y = desc.z = desc.w = 0;
        }
      }else{
        // always four components regardless of precision
        if (precision == QUDA_DOUBLE_PRECISION) {
          desc.x = 8*sizeof(int);
          desc.y = 8*sizeof(int);
          desc.z = 8*sizeof(int);
          desc.w = 8*sizeof(int);
        } else {
          desc.x = 8*precision;
          desc.y = 8*precision;
          desc.z = (reconstruct == 18 || reconstruct == 10) ? 0 : 8*precision; // float2 or short2 for 18 reconstruct
          desc.w = (reconstruct == 18 || reconstruct == 10) ? 0 : 8*precision;
        }
      }

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = field;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = isPhase ? phase_bytes/(!full ? 2 : 1) : (bytes-phase_bytes)/(!full ? 2 : 1);

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
      else texDesc.readMode = cudaReadModeElementType;

      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
      checkCudaError();
    }
  }

  void cudaGaugeField::destroyTexObject() {
    if( isNative() ){
      cudaDestroyTextureObject(evenTex);
      cudaDestroyTextureObject(oddTex);
      if(reconstruct == QUDA_RECONSTRUCT_9 || reconstruct == QUDA_RECONSTRUCT_13){
        cudaDestroyTextureObject(evenPhaseTex);
        cudaDestroyTextureObject(oddPhaseTex);
      }
      checkCudaError();
    }
  }
#endif

  cudaGaugeField::~cudaGaugeField()
  {
#ifdef USE_TEXTURE_OBJECTS
    destroyTexObject();
#endif

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      if (gauge) pool_device_free(gauge);
    }

    if ( !isNative() ) {
      for (int i=0; i<nDim; i++) {
        if (ghost[i]) pool_device_free(ghost[i]);
        if (ghost[i+4] && geometry == QUDA_COARSE_GEOMETRY) pool_device_free(ghost[i]);
      }
    }

  }

  // This does the exchange of the gauge field ghost zone and places it
  // into the ghost array.
  void cudaGaugeField::exchangeGhost(QudaLinkDirection link_direction) {
    if (ghostExchange != QUDA_GHOST_EXCHANGE_PAD)
      errorQuda("Cannot call exchangeGhost with ghostExchange=%d",
		ghostExchange);

    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot exchange for %d geometry gauge field", geometry);

    if ( (link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == QUDA_LINK_FORWARDS) && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot request exchange of forward links on non-coarse geometry");

    void *ghost_[2*QUDA_MAX_DIM];
    void *send[2*QUDA_MAX_DIM];
    for (int d=0; d<nDim; d++) {
      ghost_[d] = isNative() ? pool_device_malloc(nFace*surface[d]*nInternal*precision) : ghost[d];
      send[d] = pool_device_malloc(nFace*surface[d]*nInternal*precision);
      if (geometry == QUDA_COARSE_GEOMETRY) { // bi-directional links
	ghost_[d+4] = isNative() ? pool_device_malloc(nFace*surface[d]*nInternal*precision) : ghost[d+4];
	send[d+4] = pool_device_malloc(nFace*surface[d]*nInternal*precision);
      }
    }

    if (link_direction == QUDA_LINK_BACKWARDS || link_direction == QUDA_LINK_BIDIRECTIONAL) {
      // get the links into contiguous buffers
      extractGaugeGhost(*this, send, true);

      // communicate between nodes
      exchange(ghost_, send, QUDA_FORWARDS);
    }

    // repeat if requested and links are bi-directional
    if (link_direction == QUDA_LINK_FORWARDS || link_direction == QUDA_LINK_BIDIRECTIONAL) {
      extractGaugeGhost(*this, send, true, nDim);
      exchange(ghost_+nDim, send+nDim, QUDA_FORWARDS);
    }

    for (int d=0; d<geometry; d++) pool_device_free(send[d]);

    if (isNative()) {
      // copy from ghost into the padded region in gauge
      if (link_direction == QUDA_LINK_BACKWARDS || link_direction == QUDA_LINK_BIDIRECTIONAL) copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, 0, ghost_, 1);

      // repeat for the second set if bi-directional
      if (link_direction == QUDA_LINK_FORWARDS || link_direction == QUDA_LINK_BIDIRECTIONAL) copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, 0, ghost_, 3);
    }

    if (isNative()) for (int d=0; d<geometry; d++) pool_device_free(ghost_[d]);
  }

  // This does the opposite of exchangeGhost and sends back the ghost
  // zone to the node from which it came and injects it back into the
  // field
  void cudaGaugeField::injectGhost(QudaLinkDirection link_direction) {
    if (ghostExchange != QUDA_GHOST_EXCHANGE_PAD)
      errorQuda("Cannot call exchangeGhost with ghostExchange=%d",
		ghostExchange);

    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot exchange for %d geometry gauge field", geometry);

    if (link_direction != QUDA_LINK_BACKWARDS)
      errorQuda("link_direction = %d not supported", link_direction);

    void *ghost_[QUDA_MAX_DIM];
    void *recv[QUDA_MAX_DIM];
    for (int d=0; d<nDim; d++) {
      ghost_[d] = isNative() ? pool_device_malloc(nFace*surface[d]*nInternal*precision) : ghost[d];
      recv[d] = pool_device_malloc(nFace*surface[d]*nInternal*precision);
    }

    if (isNative()) {
      // copy from padded region in gauge field into ghost
      copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, ghost_, 0, 1);
    }

    // communicate between nodes
    exchange(recv, ghost_, QUDA_BACKWARDS);

    // get the links into contiguous buffers
    extractGaugeGhost(*this, recv, false);

    for (int d=0; d<nDim; d++) {
      pool_device_free(recv[d]);
      if (isNative()) pool_device_free(ghost_[d]);
    }
  }

  void cudaGaugeField::exchangeExtendedGhost(const int *R, bool no_comms_fill) {
    
    void *send[QUDA_MAX_DIM];
    void *recv[QUDA_MAX_DIM];
    void *send_d[QUDA_MAX_DIM];
    void *recv_d[QUDA_MAX_DIM];
    size_t bytes[QUDA_MAX_DIM];

    for (int d=0; d<nDim; d++) {
      if ( !(commDimPartitioned(d) || (no_comms_fill && R[d])) ) continue;
      // store both parities and directions in each
      bytes[d] = surface[d] * R[d] * geometry * nInternal * precision;
      send_d[d] = pool_device_malloc(2 * bytes[d]);
      recv_d[d] = pool_device_malloc(2 * bytes[d]);
    }

#ifndef GPU_COMMS
    void *send_h[QUDA_MAX_DIM];
    void *recv_h[QUDA_MAX_DIM];
    size_t total_bytes = 0;
    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d)) continue;
      total_bytes += 4*bytes[d]; // (2 from send/recv) x (2 from fwd/back)
    }
    void *buffer = total_bytes > 0 ? pool_pinned_malloc(total_bytes) : nullptr;

    size_t offset = 0;
    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d)) continue;
      recv_h[d] = static_cast<char*>(buffer) + offset;
      send_h[d] = static_cast<char*>(recv_h[d]) + 2*bytes[d];
      offset += 4*bytes[d];
    }
#endif

    // do the exchange
    MsgHandle *mh_recv_back[QUDA_MAX_DIM];
    MsgHandle *mh_recv_fwd[QUDA_MAX_DIM];
    MsgHandle *mh_send_fwd[QUDA_MAX_DIM];
    MsgHandle *mh_send_back[QUDA_MAX_DIM];

    for (int d=0; d<nDim; d++) {
      if (!commDimPartitioned(d)) continue;
#ifdef GPU_COMMS
      recv[d] = recv_d[d];
      send[d] = send_d[d];
#else
      recv[d] = recv_h[d];
      send[d] = send_h[d];
#endif

      // look into storing these for later
      mh_recv_back[d] = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
      mh_recv_fwd[d]  = comm_declare_receive_relative(static_cast<char*>(recv[d])+bytes[d], 
						      d, +1, bytes[d]);
      mh_send_back[d] = comm_declare_send_relative(send[d], d, -1, bytes[d]);
      mh_send_fwd[d]  = comm_declare_send_relative(static_cast<char*>(send[d])+bytes[d], 
						   d, +1, bytes[d]);
    }

    for (int d=0; d<nDim; d++) {
      if ( !(commDimPartitioned(d) || (no_comms_fill && R[d])) ) continue;

      // FIXME why does this break if the order is switched?
      // prepost the receives
      if (commDimPartitioned(d)) {
	comm_start(mh_recv_fwd[d]);
	comm_start(mh_recv_back[d]);
      }

      //extract into a contiguous buffer
      extractExtendedGaugeGhost(*this, d, R, send_d, true);

      if (commDimPartitioned(d)) {
	
	// pipeline the forwards and backwards sending
#ifndef GPU_COMMS
	cudaMemcpyAsync(send_h[d], send_d[d], bytes[d], cudaMemcpyDeviceToHost, streams[0]);
	cudaMemcpyAsync(static_cast<char*>(send_h[d])+bytes[d], 
			static_cast<char*>(send_d[d])+bytes[d], bytes[d], cudaMemcpyDeviceToHost, streams[1]);
#endif      
	
#ifndef GPU_COMMS
	cudaStreamSynchronize(streams[0]);
#endif
	comm_start(mh_send_back[d]);
	
#ifndef GPU_COMMS
	cudaStreamSynchronize(streams[1]);
#endif
	comm_start(mh_send_fwd[d]);
	
	// forwards recv
	comm_wait(mh_send_back[d]);
	comm_wait(mh_recv_fwd[d]);
#ifndef GPU_COMMS
	cudaMemcpyAsync(static_cast<char*>(recv_d[d])+bytes[d], 
			static_cast<char*>(recv_h[d])+bytes[d], bytes[d], cudaMemcpyHostToDevice, streams[0]);
#endif      
	
	// backwards recv
	comm_wait(mh_send_fwd[d]);
	comm_wait(mh_recv_back[d]);
#ifndef GPU_COMMS
	cudaMemcpyAsync(recv_d[d], recv_h[d], bytes[d], cudaMemcpyHostToDevice, streams[1]);
#endif      
      } else { // if just doing a local exchange to fill halo then need to swap faces
	qudaMemcpy(static_cast<char*>(recv_d[d])+bytes[d], send_d[d], bytes[d], cudaMemcpyDeviceToDevice);
	qudaMemcpy(recv_d[d], static_cast<char*>(send_d[d])+bytes[d], bytes[d], cudaMemcpyDeviceToDevice);
      }

      // inject back into the gauge field
      extractExtendedGaugeGhost(*this, d, R, recv_d, false);
    }

#ifndef GPU_COMMS
    if (total_bytes > 0) pool_pinned_free(buffer);
#endif

    for (int d=0; d<nDim; d++) {
      if ( !(commDimPartitioned(d) || (no_comms_fill && R[d])) ) continue;

      if (commDimPartitioned(d)) {
	comm_free(mh_send_fwd[d]);
	comm_free(mh_send_back[d]);
	comm_free(mh_recv_back[d]);
	comm_free(mh_recv_fwd[d]);
      }

      pool_device_free(send_d[d]);
      pool_device_free(recv_d[d]);
    }

  }

  void cudaGaugeField::setGauge(void *gauge_)
  {
    if(create != QUDA_REFERENCE_FIELD_CREATE) {
      errorQuda("Setting gauge pointer is only allowed when create="
          "QUDA_REFERENCE_FIELD_CREATE type\n");
    }
    gauge = gauge_;
  }

  void *create_gauge_buffer(size_t bytes, QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {
    if (order == QUDA_QDP_GAUGE_ORDER) {
      void **buffer = new void*[geometry];
      for (int d=0; d<geometry; d++) buffer[d] = pool_device_malloc(bytes/geometry);
      return ((void*)buffer);
    } else {
      return pool_device_malloc(bytes);
    }

  }

  void **create_ghost_buffer(size_t bytes[], QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {

    if (order > 4) {
      void **buffer = new void*[geometry];
      for (int d=0; d<geometry; d++) buffer[d] = pool_device_malloc(bytes[d]);
      return buffer;
    } else {
      return 0;
    }

  }

  void free_gauge_buffer(void *buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {
    if (order == QUDA_QDP_GAUGE_ORDER) {
      for (int d=0; d<geometry; d++) pool_device_free(((void**)buffer)[d]);
      delete []((void**)buffer);
    } else {
      pool_device_free(buffer);
    }
  }

  void free_ghost_buffer(void **buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry) {
    if (order > 4) {
      for (int d=0; d<geometry; d++) pool_device_free(buffer[d]);
      delete []buffer;
    }
  }

  void cudaGaugeField::copy(const GaugeField &src) {
    if (this == &src) return;

    checkField(src);

    if (link_type == QUDA_ASQTAD_FAT_LINKS) {
      fat_link_max = src.LinkMax();
      if (precision == QUDA_HALF_PRECISION && fat_link_max == 0.0) 
        errorQuda("fat_link_max has not been computed");
    } else {
      fat_link_max = 1.0;
    }

    if (typeid(src) == typeid(cudaGaugeField)) {

      // copy field and ghost zone into this field
      copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, 
          static_cast<const cudaGaugeField&>(src).gauge);

      if (geometry == QUDA_COARSE_GEOMETRY)
	copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, static_cast<const cudaGaugeField&>(src).gauge, 0, 0, 3);

    } else if (typeid(src) == typeid(cpuGaugeField)) {
      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do reorder on the CPU
	void *buffer = pool_pinned_malloc(bytes);

	if (src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
	  // copy field and ghost zone into buffer
	  copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, static_cast<const cpuGaugeField&>(src).gauge);
	} else {
	  copyExtendedGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, static_cast<const cpuGaugeField&>(src).gauge);
	}

	// this copies over both even and odd
	qudaMemcpy(gauge, buffer, bytes, cudaMemcpyHostToDevice);
	pool_pinned_free(buffer);
      } else { // else on the GPU
	void *buffer = create_gauge_buffer(src.Bytes(), src.Order(), src.Geometry());
	size_t ghost_bytes[8];
	int srcNinternal = src.Reconstruct() != QUDA_RECONSTRUCT_NO ? src.Reconstruct() : 2*nColor*nColor;
	for (int d=0; d<geometry; d++) ghost_bytes[d] = nFace * surface[d%4] * srcNinternal * src.Precision();
	void **ghost_buffer = (nFace > 0) ? create_ghost_buffer(ghost_bytes, src.Order(), geometry) : nullptr;

	if (src.Order() == QUDA_QDP_GAUGE_ORDER) {
	  for (int d=0; d<geometry; d++) {
	    qudaMemcpy(((void**)buffer)[d], ((void**)src.Gauge_p())[d], src.Bytes()/geometry, cudaMemcpyHostToDevice);
	  }
	} else {
	  qudaMemcpy(buffer, src.Gauge_p(), src.Bytes(), cudaMemcpyHostToDevice);
	}

	if (src.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD &&
	    src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
	  for (int d=0; d<geometry; d++)
	    qudaMemcpy(ghost_buffer[d], src.Ghost()[d], ghost_bytes[d], cudaMemcpyHostToDevice);

	if (src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
	  copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer, 0, ghost_buffer);
	  if (geometry == QUDA_COARSE_GEOMETRY) copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer, 0, ghost_buffer, 3);
	} else {
	  copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer);
	}
	free_gauge_buffer(buffer, src.Order(), src.Geometry());
	if (nFace > 0) free_ghost_buffer(ghost_buffer, src.Order(), geometry);
      } // reorder_location
    } else {
      errorQuda("Invalid gauge field type");
    }

    // if we have copied from a source without a pad then we need to exchange
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD && src.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD)
      exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);

    staggeredPhaseApplied = src.StaggeredPhaseApplied();
    staggeredPhaseType = src.StaggeredPhase();

    checkCudaError();
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu) { copy(cpu); }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const
  {
    static_cast<LatticeField&>(cpu).checkField(*this);

    if (reorder_location() == QUDA_CUDA_FIELD_LOCATION) {

      void *buffer = create_gauge_buffer(cpu.Bytes(), cpu.Order(), cpu.Geometry());

      // Allocate space for ghost zone if required
      size_t ghost_bytes[8];
      int cpuNinternal = cpu.Reconstruct() != QUDA_RECONSTRUCT_NO ? cpu.Reconstruct() : 2*nColor*nColor;
      for (int d=0; d<geometry; d++) ghost_bytes[d] = nFace * surface[d%4] * cpuNinternal * cpu.Precision();
      void **ghost_buffer = (nFace > 0) ? create_ghost_buffer(ghost_bytes, cpu.Order(), geometry) : nullptr;

      if (cpu.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
	copyGenericGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, buffer, gauge, ghost_buffer, 0);
	if (geometry == QUDA_COARSE_GEOMETRY) copyGenericGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, buffer, gauge, ghost_buffer, 0, 3);
      } else {
	copyExtendedGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, buffer, gauge);
      }

      if (cpu.Order() == QUDA_QDP_GAUGE_ORDER) {
	for (int d=0; d<geometry; d++) qudaMemcpy(((void**)cpu.gauge)[d], ((void**)buffer)[d], cpu.Bytes()/geometry, cudaMemcpyDeviceToHost);
      } else {
	qudaMemcpy(cpu.gauge, buffer, cpu.Bytes(), cudaMemcpyDeviceToHost);
      }

      if (cpu.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD &&
	  cpu.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
	for (int d=0; d<geometry; d++)
	  qudaMemcpy(cpu.Ghost()[d], ghost_buffer[d], ghost_bytes[d], cudaMemcpyDeviceToHost);

      free_gauge_buffer(buffer, cpu.Order(), cpu.Geometry());
      if (nFace > 0) free_ghost_buffer(ghost_buffer, cpu.Order(), geometry);

    } else if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do copy then host-side reorder

      void *buffer = pool_pinned_malloc(bytes);
      qudaMemcpy(buffer, gauge, bytes, cudaMemcpyDeviceToHost);

      if (cpu.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
	copyGenericGauge(cpu, *this, QUDA_CPU_FIELD_LOCATION, cpu.gauge, buffer);
      } else {
	copyExtendedGauge(cpu, *this, QUDA_CPU_FIELD_LOCATION, cpu.gauge, buffer);
      }
      pool_pinned_free(buffer);

    } else {
      errorQuda("Invalid pack location %d", reorder_location());
    }

    cpu.staggeredPhaseApplied = staggeredPhaseApplied;
    cpu.staggeredPhaseType = staggeredPhaseType;
  }

  void cudaGaugeField::backup() const {
    if (backed_up) errorQuda("Gauge field already backed up");
    backup_h = new char[bytes];
    cudaMemcpy(backup_h, gauge, bytes, cudaMemcpyDeviceToHost);
    checkCudaError();
    backed_up = true;
  }

  void cudaGaugeField::restore() {
    if (!backed_up) errorQuda("Cannot restore since not backed up");
    cudaMemcpy(gauge, backup_h, bytes, cudaMemcpyHostToDevice);
    delete []backup_h;
    checkCudaError();
    backed_up = false;
  }

  void cudaGaugeField::zero() {
    cudaMemset(gauge, 0, bytes);
  }


} // namespace quda
