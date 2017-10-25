#include <string.h>
#include <gauge_field.h>
#include <typeinfo>
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

      int texel_size = 1;
      if (isPhase) {
        if (precision == QUDA_DOUBLE_PRECISION) {
          desc.x = 8*sizeof(int);
          desc.y = 8*sizeof(int);
          desc.z = 0;
          desc.w = 0;
          texel_size = 2*sizeof(int);
        } else {
          desc.x = 8*precision;
          desc.y = desc.z = desc.w = 0;
          texel_size = precision;
        }
      } else {
        // always four components regardless of precision
        if (precision == QUDA_DOUBLE_PRECISION) {
          desc.x = 8*sizeof(int);
          desc.y = 8*sizeof(int);
          desc.z = 8*sizeof(int);
          desc.w = 8*sizeof(int);
	  texel_size = 4*sizeof(int);
        } else {
          desc.x = 8*precision;
          desc.y = 8*precision;
          desc.z = (reconstruct == 18 || reconstruct == 10) ? 0 : 8*precision; // float2 or short2 for 18 reconstruct
          desc.w = (reconstruct == 18 || reconstruct == 10) ? 0 : 8*precision;
          texel_size = (reconstruct == 18 || reconstruct == 10 ? 2 : 4) * precision;
        }
      }

      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = field;
      resDesc.res.linear.desc = desc;
      resDesc.res.linear.sizeInBytes = isPhase ? phase_bytes/(!full ? 2 : 1) : (bytes-phase_bytes)/(!full ? 2 : 1);

      unsigned long texels = resDesc.res.linear.sizeInBytes / texel_size;
      if (texels > (unsigned)deviceProp.maxTexture1DLinear) {
	errorQuda("Attempting to bind too large a texture %lu > %d", texels, deviceProp.maxTexture1DLinear);
      }

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

    destroyComms();

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

  void cudaGaugeField::allocateGhostBuffer(const int *R, bool no_comms_fill) const
  {
    createGhostZone(R, no_comms_fill);
    LatticeField::allocateGhostBuffer(ghost_bytes);
  }

  void cudaGaugeField::createComms(const int *R, bool no_comms_fill)
  {
    allocateGhostBuffer(R, no_comms_fill); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs it comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face_h[0] != ghost_pinned_buffer_h[0]) || (my_face_h[1] != ghost_pinned_buffer_h[1]); // pinned buffers

    if (!initComms || comms_reset) LatticeField::createComms(no_comms_fill);

    if (ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  void cudaGaugeField::recvStart(int dim, int dir)
  {
    if (!comm_dim_partitioned(dim)) return;

    if (dir==0) { // sending backwards
      // receive from the processor in the +1 direction
      if (comm_peer2peer_enabled(1,dim)) {
	comm_start(mh_recv_p2p_fwd[bufferIndex][dim]);
      } else if (comm_gdr_enabled()) {
        comm_start(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_fwd[bufferIndex][dim]);
      }
    } else { //sending forwards
      // receive from the processor in the -1 direction
      if (comm_peer2peer_enabled(0,dim)) {
	comm_start(mh_recv_p2p_back[bufferIndex][dim]);
      } else if (comm_gdr_enabled()) {
        comm_start(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
        comm_start(mh_recv_back[bufferIndex][dim]);
      }
    }
  }

  void cudaGaugeField::sendStart(int dim, int dir, cudaStream_t* stream_p)
  {
    if (!comm_dim_partitioned(dim)) return;

    if (!comm_peer2peer_enabled(dir,dim)) {
      if (dir == 0)
	if (comm_gdr_enabled()) {
	  comm_start(mh_send_rdma_back[bufferIndex][dim]);
	} else {
	  comm_start(mh_send_back[bufferIndex][dim]);
	}
      else
	if (comm_gdr_enabled()) {
	  comm_start(mh_send_rdma_fwd[bufferIndex][dim]);
	} else {
	  comm_start(mh_send_fwd[bufferIndex][dim]);
	}
    } else { // doing peer-to-peer

      void* ghost_dst = static_cast<char*>(ghost_remote_send_buffer_d[bufferIndex][dim][dir])
	+ precision*ghostOffset[dim][(dir+1)%2];

      cudaMemcpyAsync(ghost_dst, my_face_dim_dir_d[bufferIndex][dim][dir],
		      ghost_face_bytes[dim], cudaMemcpyDeviceToDevice,
		      stream_p ? *stream_p : 0);

      if (dir == 0) {
	// record the event
	cudaEventRecord(ipcCopyEvent[bufferIndex][0][dim], stream_p ? *stream_p : 0);
	// send to the processor in the -1 direction
	comm_start(mh_send_p2p_back[bufferIndex][dim]);
      } else {
	cudaEventRecord(ipcCopyEvent[bufferIndex][1][dim], stream_p ? *stream_p : 0);
	// send to the processor in the +1 direction
	comm_start(mh_send_p2p_fwd[bufferIndex][dim]);
      }
    }
  }

  void cudaGaugeField::commsComplete(int dim, int dir)
  {
    if (!comm_dim_partitioned(dim)) return;

    if (dir==0) {
      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_recv_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][1][dim]);
      } else if (comm_gdr_enabled()) {
	comm_wait(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_fwd[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_send_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][0][dim]);
      } else if (comm_gdr_enabled()) {
	comm_wait(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_back[bufferIndex][dim]);
      }
    } else {
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_recv_p2p_back[bufferIndex][dim]);
	cudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][0][dim]);
      } else if (comm_gdr_enabled()) {
	comm_wait(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_back[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_send_p2p_fwd[bufferIndex][dim]);
	cudaEventSynchronize(ipcCopyEvent[bufferIndex][1][dim]);
      } else if (comm_gdr_enabled()) {
	comm_wait(mh_send_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_fwd[bufferIndex][dim]);
      }
    }
  }

  void cudaGaugeField::exchangeExtendedGhost(const int *R, bool no_comms_fill)
  {
    const int b = bufferIndex;
    void *send_d[QUDA_MAX_DIM], *recv_d[QUDA_MAX_DIM];

    createComms(R, no_comms_fill);

    size_t offset = 0;
    for (int dim=0; dim<nDim; dim++) {
      if ( !(comm_dim_partitioned(dim) || (no_comms_fill && R[dim])) ) continue;
      send_d[dim] = static_cast<char*>(ghost_send_buffer_d[b]) + offset;
      recv_d[dim] = static_cast<char*>(ghost_recv_buffer_d[b]) + offset;
      offset += 2*ghost_face_bytes[dim]; // factor of two from fwd/back
    }

    for (int dim=0; dim<nDim; dim++) {
      if ( !(comm_dim_partitioned(dim) || (no_comms_fill && R[dim])) ) continue;

      //extract into a contiguous buffer
      extractExtendedGaugeGhost(*this, dim, R, send_d, true);

      if (comm_dim_partitioned(dim)) {
	for (int dir=0; dir<2; dir++) recvStart(dim, dir);

	for (int dir=0; dir<2; dir++) {
	  // issue host-to-device copies if needed
	  if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) {
	    cudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
			    ghost_face_bytes[dim], cudaMemcpyDeviceToHost, streams[dir]);
	  }
	}

	// if neither direction is peer-to-peer then we need to synchronize
	if (!comm_peer2peer_enabled(0,dim) || !comm_peer2peer_enabled(1,dim)) cudaDeviceSynchronize();

	// if we pass a stream to sendStart then we must ensure that stream is synchronized
	for (int dir=0; dir<2; dir++) sendStart(dim, dir, &streams[dir]);
	for (int dir=0; dir<2; dir++) commsComplete(dim, dir);

	for (int dir=0; dir<2; dir++) {
	  // issue host-to-device copies if needed
	  if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) {
	    cudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][dir], from_face_dim_dir_h[bufferIndex][dim][dir],
			    ghost_face_bytes[dim], cudaMemcpyHostToDevice, streams[dir]);
	  }
	}

      } else { // if just doing a local exchange to fill halo then need to swap faces
	qudaMemcpy(from_face_dim_dir_d[b][dim][1], my_face_dim_dir_d[b][dim][0],
		   ghost_face_bytes[dim], cudaMemcpyDeviceToDevice);
	qudaMemcpy(from_face_dim_dir_d[b][dim][0], my_face_dim_dir_d[b][dim][1],
		   ghost_face_bytes[dim], cudaMemcpyDeviceToDevice);
      }

      // inject back into the gauge field
      extractExtendedGaugeGhost(*this, dim, R, recv_d, false);
    }

    bufferIndex = 1-bufferIndex;
    cudaDeviceSynchronize();
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

	if (src.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {
	  // special case where we use zero-copy memory to read/write directly from MILC's data
	  void *src_d;
	  cudaError_t error = cudaHostGetDevicePointer(&src_d, const_cast<void*>(src.Gauge_p()), 0);
	  if (error != cudaSuccess) errorQuda("Failed to get device pointer for MILC site array");

	  if (src.GhostExchange() == QUDA_GHOST_EXCHANGE_NO) {
	    copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, src_d);
	  } else {
	    errorQuda("Ghost copy not supported here");
	  }

	} else {
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
	}
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

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu) {
    copy(cpu);
    cudaDeviceSynchronize();
    checkCudaError();
  }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const
  {
    static_cast<LatticeField&>(cpu).checkField(*this);

    if (reorder_location() == QUDA_CUDA_FIELD_LOCATION) {

      if (cpu.Order() == QUDA_MILC_SITE_GAUGE_ORDER) {
	// special case where we use zero-copy memory to read/write directly from MILC's data
	void *cpu_d;
  cudaError_t error = cudaHostGetDevicePointer(&cpu_d, const_cast<void*>(cpu.Gauge_p()), 0);
  if (error != cudaSuccess) errorQuda("Failed to get device pointer for MILC site array");
	if (cpu.GhostExchange() == QUDA_GHOST_EXCHANGE_NO) {
	  copyGenericGauge(cpu, *this, QUDA_CUDA_FIELD_LOCATION, cpu_d, gauge);
	} else {
	  errorQuda("Ghost copy not supported here");
	}
      } else {
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
      }
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

    cudaDeviceSynchronize();
    checkCudaError();
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
