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
	  errorQuda("cudaGaugeField being constructed with insufficient padding in dim %d (%d < %d)\n", i, pad, minimum_pad);
      }
    }
#endif

    if (create != QUDA_NULL_FIELD_CREATE &&
        create != QUDA_ZERO_FIELD_CREATE &&
        create != QUDA_REFERENCE_FIELD_CREATE){
      errorQuda("ERROR: create type(%d) not supported yet\n", create);
    }

    if (create != QUDA_REFERENCE_FIELD_CREATE) {
      switch(mem_type) {
      case QUDA_MEMORY_DEVICE:
	gauge = pool_device_malloc(bytes);
	break;
      case QUDA_MEMORY_MAPPED:
        gauge_h = mapped_malloc(bytes);
	cudaHostGetDevicePointer(&gauge, gauge_h, 0); // set the matching device pointer
	break;
      default:
	errorQuda("Unsupported memory type %d", mem_type);
      }
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
    odd = static_cast<char*>(gauge) + bytes/2;
    if (create != QUDA_ZERO_FIELD_CREATE && isNative() && ghostExchange == QUDA_GHOST_EXCHANGE_PAD) zeroPad();

#ifdef USE_TEXTURE_OBJECTS
    createTexObject(tex, gauge, true);
    createTexObject(evenTex, even, false);
    createTexObject(oddTex, odd, false);
    if (reconstruct == QUDA_RECONSTRUCT_13 || reconstruct == QUDA_RECONSTRUCT_9) {
      // Create texture objects for the phases
      bool isPhase = true;
      createTexObject(phaseTex, (char*)gauge + phase_offset, true, isPhase);
      createTexObject(evenPhaseTex, (char*)even + phase_offset, false, isPhase);
      createTexObject(oddPhaseTex, (char*)odd + phase_offset, false, isPhase);
    }
#endif
  }

  void cudaGaugeField::zeroPad() {
    size_t pad_bytes = (stride - volumeCB) * precision * order;
    int Npad = (geometry * (reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : nColor * nColor * 2)) / order;

    size_t pitch = stride*order*precision;
    if (pad_bytes) {
      cudaMemset2D(static_cast<char*>(even) + volumeCB*order*precision, pitch, 0, pad_bytes, Npad);
      cudaMemset2D(static_cast<char*>(odd) + volumeCB*order*precision, pitch, 0, pad_bytes, Npad);
    }
  }

#ifdef USE_TEXTURE_OBJECTS
  void cudaGaugeField::createTexObject(cudaTextureObject_t &tex, void *field, bool full, bool isPhase) {

    if (isNative() && geometry != QUDA_COARSE_GEOMETRY) {
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
      resDesc.res.linear.sizeInBytes = (isPhase ? phase_bytes : bytes) / (!full ? 2 : 1);

      if (resDesc.res.linear.sizeInBytes % deviceProp.textureAlignment != 0
          || !is_aligned(resDesc.res.linear.devPtr, deviceProp.textureAlignment)) {
        errorQuda("Allocation size %lu does not have correct alignment for textures (%lu)",
                  resDesc.res.linear.sizeInBytes, deviceProp.textureAlignment);
      }

      unsigned long texels = resDesc.res.linear.sizeInBytes / texel_size;
      if (texels > (unsigned)deviceProp.maxTexture1DLinear) {
	errorQuda("Attempting to bind too large a texture %lu > %d", texels, deviceProp.maxTexture1DLinear);
      }

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) texDesc.readMode = cudaReadModeNormalizedFloat;
      else texDesc.readMode = cudaReadModeElementType;

      cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
      checkCudaError();
    }
  }

  void cudaGaugeField::destroyTexObject() {
    if ( isNative() && geometry != QUDA_COARSE_GEOMETRY ) {
      cudaDestroyTextureObject(tex);
      cudaDestroyTextureObject(evenTex);
      cudaDestroyTextureObject(oddTex);
      if (reconstruct == QUDA_RECONSTRUCT_9 || reconstruct == QUDA_RECONSTRUCT_13) {
        cudaDestroyTextureObject(phaseTex);
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
      switch(mem_type) {
      case QUDA_MEMORY_DEVICE:
        if (gauge) pool_device_free(gauge);
        break;
      case QUDA_MEMORY_MAPPED:
        if (gauge_h) host_free(gauge_h);
        break;
      default:
        errorQuda("Unsupported memory type %d", mem_type);
      }
    }

    if ( !isNative() ) {
      for (int i=0; i<nDim; i++) {
        if (ghost[i]) pool_device_free(ghost[i]);
        if (ghost[i+4] && geometry == QUDA_COARSE_GEOMETRY) pool_device_free(ghost[i+4]);
      }
    }

  }

  // This does the exchange of the forwards boundary gauge field ghost zone and places
  // it into the ghost array of the next node
  void cudaGaugeField::exchangeGhost(QudaLinkDirection link_direction) {

    if (ghostExchange != QUDA_GHOST_EXCHANGE_PAD) errorQuda("Cannot call exchangeGhost with ghostExchange=%d", ghostExchange);
    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY) errorQuda("Invalid geometry=%d", geometry);
    if ( (link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == QUDA_LINK_FORWARDS) && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot request exchange of forward links on non-coarse geometry");
    if (nFace == 0) errorQuda("nFace = 0");

    const int dir = 1; // sending forwards only
    const int R[] = {nFace, nFace, nFace, nFace};
    const bool no_comms_fill = true; // dslash kernels presently require this
    const bool bidir = false; // communication is only ever done in one direction at once
    createComms(R, true, bidir); // always need to allocate space for non-partitioned dimension for copyGenericGauge

    // loop over backwards and forwards links
    const QudaLinkDirection directions[] = {QUDA_LINK_BACKWARDS, QUDA_LINK_FORWARDS};
    for (int link_dir = 0; link_dir<2; link_dir++) {
      if (!(link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == directions[link_dir])) continue;

      void *send_d[2*QUDA_MAX_DIM] = { };
      void *recv_d[2*QUDA_MAX_DIM] = { };

      size_t offset = 0;
      for (int d=0; d<nDim; d++) {
	recv_d[d] = static_cast<char*>(ghost_recv_buffer_d[bufferIndex]) + offset; if (bidir) offset += ghost_face_bytes[d];
	send_d[d] = static_cast<char*>(ghost_send_buffer_d[bufferIndex]) + offset; offset += ghost_face_bytes[d];
      }

      extractGaugeGhost(*this, send_d, true, link_dir*nDim); // get the links into contiguous buffers

      // issue receive preposts and host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	recvStart(dim, dir); // prepost the receive
	if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) {
	  cudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
			  ghost_face_bytes[dim], cudaMemcpyDeviceToHost, streams[2*dim+dir]);
	}
      }

      // if gdr enabled then synchronize
      if (comm_gdr_enabled()) qudaDeviceSynchronize();

      // if the sending direction is not peer-to-peer then we need to synchronize before we start sending
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) qudaStreamSynchronize(streams[2*dim+dir]);
	sendStart(dim, dir, &streams[2*dim+dir]); // start sending
      }

      // complete communication and issue host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	commsComplete(dim, dir);
	if (!comm_peer2peer_enabled(1-dir,dim) && !comm_gdr_enabled()) {
	  cudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][1-dir], from_face_dim_dir_h[bufferIndex][dim][1-dir],
			  ghost_face_bytes[dim], cudaMemcpyHostToDevice, streams[2*dim+dir]);
	}
      }

      // fill in the halos for non-partitioned dimensions
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim) && no_comms_fill) {
	  qudaMemcpy(recv_d[dim], send_d[dim], ghost_face_bytes[dim], cudaMemcpyDeviceToDevice);
	}
      }

      if (isNative()) {
	copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, 0, recv_d, 1 + 2*link_dir); // 1, 3
      } else {
	// copy from receive buffer into ghost array
	for (int dim=0; dim<nDim; dim++)
	  qudaMemcpy(ghost[dim+link_dir*nDim], recv_d[dim], ghost_face_bytes[dim], cudaMemcpyDeviceToDevice);
      }

      bufferIndex = 1-bufferIndex;
    } // link_dir

    qudaDeviceSynchronize();
  }

  // This does the opposite of exchangeGhost and sends back the ghost
  // zone to the node from which it came and injects it back into the
  // field
  void cudaGaugeField::injectGhost(QudaLinkDirection link_direction) {

    if (ghostExchange != QUDA_GHOST_EXCHANGE_PAD) errorQuda("Cannot call exchangeGhost with ghostExchange=%d", ghostExchange);
    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY) errorQuda("Invalid geometry=%d", geometry);
    if (link_direction != QUDA_LINK_BACKWARDS) errorQuda("Invalid link_direction = %d", link_direction);
    if (nFace == 0) errorQuda("nFace = 0");

    const int dir = 0; // sending backwards only
    const int R[] = {nFace, nFace, nFace, nFace};
    const bool no_comms_fill = false; // injection never does no_comms_fill
    const bool bidir = false; // communication is only ever done in one direction at once
    createComms(R, true, bidir); // always need to allocate space for non-partitioned dimension for copyGenericGauge

    // loop over backwards and forwards links (forwards links never sent but leave here just in case)
    const QudaLinkDirection directions[] = {QUDA_LINK_BACKWARDS, QUDA_LINK_FORWARDS};
    for (int link_dir = 0; link_dir<2; link_dir++) {
      if (!(link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == directions[link_dir])) continue;

      void *send_d[2*QUDA_MAX_DIM] = { };
      void *recv_d[2*QUDA_MAX_DIM] = { };

      size_t offset = 0;
      for (int d=0; d<nDim; d++) {
	// send backwards is first half of each ghost_send_buffer
	send_d[d] = static_cast<char*>(ghost_send_buffer_d[bufferIndex]) + offset; if (bidir) offset += ghost_face_bytes[d];
	// receive from forwards is the second half of each ghost_recv_buffer
	recv_d[d] = static_cast<char*>(ghost_recv_buffer_d[bufferIndex]) + offset; offset += ghost_face_bytes[d];
      }

      if (isNative()) { // copy from padded region in gauge field into send buffer
	copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, send_d, 0, 1 + 2*link_dir);
      } else { // copy from receive buffer into ghost array
	for (int dim=0; dim<nDim; dim++) qudaMemcpy(send_d[dim], ghost[dim+link_dir*nDim], ghost_face_bytes[dim], cudaMemcpyDeviceToDevice);
      }

      // issue receive preposts and host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	recvStart(dim, dir); // prepost the receive
	if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) {
	  cudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
			  ghost_face_bytes[dim], cudaMemcpyDeviceToHost, streams[2*dim+dir]);
	}
      }

      // if gdr enabled then synchronize
      if (comm_gdr_enabled()) qudaDeviceSynchronize();

      // if the sending direction is not peer-to-peer then we need to synchronize before we start sending
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) qudaStreamSynchronize(streams[2*dim+dir]);
	sendStart(dim, dir, &streams[2*dim+dir]); // start sending
      }

      // complete communication and issue host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	commsComplete(dim, dir);
	if (!comm_peer2peer_enabled(1-dir,dim) && !comm_gdr_enabled()) {
	  cudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][1-dir], from_face_dim_dir_h[bufferIndex][dim][1-dir],
			  ghost_face_bytes[dim], cudaMemcpyHostToDevice, streams[2*dim+dir]);
	}
      }

      // fill in the halos for non-partitioned dimensions
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim) && no_comms_fill) {
	  qudaMemcpy(recv_d[dim], send_d[dim], ghost_face_bytes[dim], cudaMemcpyDeviceToDevice);
	}
      }

      // get the links into contiguous buffers
      extractGaugeGhost(*this, recv_d, false, link_dir*nDim);

      bufferIndex = 1-bufferIndex;
    } // link_dir

    qudaDeviceSynchronize();
  }

  void cudaGaugeField::allocateGhostBuffer(const int *R, bool no_comms_fill, bool bidir) const
  {
    createGhostZone(R, no_comms_fill, bidir);
    LatticeField::allocateGhostBuffer(ghost_bytes);
  }

  void cudaGaugeField::createComms(const int *R, bool no_comms_fill, bool bidir)
  {
    allocateGhostBuffer(R, no_comms_fill, bidir); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs it comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face_h[0] != ghost_pinned_send_buffer_h[0]) || (my_face_h[1] != ghost_pinned_send_buffer_h[1]) ||
      (from_face_h[0] != ghost_pinned_recv_buffer_h[0]) || (from_face_h[1] != ghost_pinned_recv_buffer_h[1]) ||
      ghost_bytes != ghost_bytes_old; // ghost buffer has been resized (e.g., bidir to unidir)

    if (!initComms || comms_reset) LatticeField::createComms(no_comms_fill, bidir);

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

  void cudaGaugeField::sendStart(int dim, int dir, qudaStream_t *stream_p)
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
	qudaEventRecord(ipcCopyEvent[bufferIndex][0][dim], stream_p ? *stream_p : 0);
	// send to the processor in the -1 direction
	comm_start(mh_send_p2p_back[bufferIndex][dim]);
      } else {
	qudaEventRecord(ipcCopyEvent[bufferIndex][1][dim], stream_p ? *stream_p : 0);
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
	qudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][1][dim]);
      } else if (comm_gdr_enabled()) {
	comm_wait(mh_recv_rdma_fwd[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_fwd[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_send_p2p_back[bufferIndex][dim]);
	qudaEventSynchronize(ipcCopyEvent[bufferIndex][0][dim]);
      } else if (comm_gdr_enabled()) {
	comm_wait(mh_send_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_send_back[bufferIndex][dim]);
      }
    } else {
      if (comm_peer2peer_enabled(0,dim)) {
	comm_wait(mh_recv_p2p_back[bufferIndex][dim]);
	qudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][0][dim]);
      } else if (comm_gdr_enabled()) {
	comm_wait(mh_recv_rdma_back[bufferIndex][dim]);
      } else {
	comm_wait(mh_recv_back[bufferIndex][dim]);
      }

      if (comm_peer2peer_enabled(1,dim)) {
	comm_wait(mh_send_p2p_fwd[bufferIndex][dim]);
	qudaEventSynchronize(ipcCopyEvent[bufferIndex][1][dim]);
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

      // silence cuda-memcheck initcheck errors that arise since we
      // have an oversized ghost buffer when doing the extended exchange
      qudaMemsetAsync(send_d[dim], 0, 2 * ghost_face_bytes[dim], 0);
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

	// if either direction is not peer-to-peer then we need to synchronize
	if (!comm_peer2peer_enabled(0,dim) || !comm_peer2peer_enabled(1,dim)) qudaDeviceSynchronize();

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
    qudaDeviceSynchronize();
  }

  void cudaGaugeField::exchangeExtendedGhost(const int *R, TimeProfile &profile, bool no_comms_fill) {
    profile.TPSTART(QUDA_PROFILE_COMMS);
    exchangeExtendedGhost(R, no_comms_fill);
    profile.TPSTOP(QUDA_PROFILE_COMMS);
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
      if (fat_link_max == 0.0 && precision < QUDA_SINGLE_PRECISION) fat_link_max = src.abs_max();
    } else {
      fat_link_max = 1.0;
    }

    if (typeid(src) == typeid(cudaGaugeField)) {

      if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
        // copy field and ghost zone into this field
        copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, static_cast<const cudaGaugeField&>(src).gauge);

        if (geometry == QUDA_COARSE_GEOMETRY)
          copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, static_cast<const cudaGaugeField&>(src).gauge, 0, 0, 3);
      } else {
        copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, static_cast<const cudaGaugeField&>(src).gauge);
        if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
      }

    } else if (typeid(src) == typeid(cpuGaugeField)) {
      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do reorder on the CPU
	void *buffer = pool_pinned_malloc(bytes);

	if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
	  // copy field and ghost zone into buffer
	  copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, static_cast<const cpuGaugeField&>(src).gauge);

          if (geometry == QUDA_COARSE_GEOMETRY)
            copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, static_cast<const cpuGaugeField &>(src).gauge,
                             0, 0, 3);
        } else {
	  copyExtendedGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, static_cast<const cpuGaugeField&>(src).gauge);
          if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
	}

	// this copies over both even and odd
	qudaMemcpy(gauge, buffer, bytes, cudaMemcpyDefault);
	pool_pinned_free(buffer);
      } else { // else on the GPU

        if (src.Order() == QUDA_MILC_SITE_GAUGE_ORDER ||
            src.Order() == QUDA_BQCD_GAUGE_ORDER      ||
            src.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	  // special case where we use zero-copy memory to read/write directly from application's array
	  void *src_d;
	  cudaError_t error = cudaHostGetDevicePointer(&src_d, const_cast<void*>(src.Gauge_p()), 0);
	  if (error != cudaSuccess) errorQuda("Failed to get device pointer for MILC site / BQCD array");

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
	      qudaMemcpy(((void**)buffer)[d], ((void**)src.Gauge_p())[d], src.Bytes()/geometry, cudaMemcpyDefault);
	    }
	  } else {
	    qudaMemcpy(buffer, src.Gauge_p(), src.Bytes(), cudaMemcpyDefault);
	  }

	  if (src.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD &&
	      src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
	    for (int d=0; d<geometry; d++)
	      qudaMemcpy(ghost_buffer[d], src.Ghost()[d], ghost_bytes[d], cudaMemcpyDefault);

	  if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
	    copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer, 0, ghost_buffer);
	    if (geometry == QUDA_COARSE_GEOMETRY) copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer, 0, ghost_buffer, 3);
	  } else {
	    copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer);
            if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
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

    qudaDeviceSynchronize(); // include sync here for accurate host-device profiling
    checkCudaError();
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu) {
    copy(cpu);
    qudaDeviceSynchronize();
    checkCudaError();
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu, TimeProfile &profile) {
    profile.TPSTART(QUDA_PROFILE_H2D);
    loadCPUField(cpu);
    profile.TPSTOP(QUDA_PROFILE_H2D);
  }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const
  {
    static_cast<LatticeField&>(cpu).checkField(*this);

    if (reorder_location() == QUDA_CUDA_FIELD_LOCATION) {

      if (cpu.Order() == QUDA_MILC_SITE_GAUGE_ORDER ||
          cpu.Order() == QUDA_BQCD_GAUGE_ORDER      ||
          cpu.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	// special case where we use zero-copy memory to read/write directly from application's array
	void *cpu_d;
	cudaError_t error = cudaHostGetDevicePointer(&cpu_d, const_cast<void*>(cpu.Gauge_p()), 0);
	if (error != cudaSuccess) errorQuda("Failed to get device pointer for MILC site / BQCD array");
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
	  for (int d=0; d<geometry; d++) qudaMemcpy(((void**)cpu.gauge)[d], ((void**)buffer)[d], cpu.Bytes()/geometry, cudaMemcpyDefault);
	} else {
	  qudaMemcpy(cpu.gauge, buffer, cpu.Bytes(), cudaMemcpyDefault);
	}

	if (cpu.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD &&
	    cpu.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
	  for (int d=0; d<geometry; d++)
	    qudaMemcpy(cpu.Ghost()[d], ghost_buffer[d], ghost_bytes[d], cudaMemcpyDefault);

	free_gauge_buffer(buffer, cpu.Order(), cpu.Geometry());
	if (nFace > 0) free_ghost_buffer(ghost_buffer, cpu.Order(), geometry);
      }
    } else if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do copy then host-side reorder

      void *buffer = pool_pinned_malloc(bytes);
      qudaMemcpy(buffer, gauge, bytes, cudaMemcpyDefault);

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

    qudaDeviceSynchronize();
    checkCudaError();
  }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, TimeProfile &profile) const {
    profile.TPSTART(QUDA_PROFILE_D2H);
    saveCPUField(cpu);
    profile.TPSTOP(QUDA_PROFILE_D2H);
  }

  void cudaGaugeField::backup() const {
    if (backed_up) errorQuda("Gauge field already backed up");
    backup_h = new char[bytes];
    cudaMemcpy(backup_h, gauge, bytes, cudaMemcpyDefault);
    checkCudaError();
    backed_up = true;
  }

  void cudaGaugeField::restore() const
  {
    if (!backed_up) errorQuda("Cannot restore since not backed up");
    cudaMemcpy(gauge, backup_h, bytes, cudaMemcpyDefault);
    delete []backup_h;
    checkCudaError();
    backed_up = false;
  }

  void cudaGaugeField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {

    if (is_prefetch_enabled() && mem_type == QUDA_MEMORY_DEVICE) {
      int dev_id = 0;
      if (mem_space == QUDA_CUDA_FIELD_LOCATION)
        dev_id = comm_gpuid();
      else if (mem_space == QUDA_CPU_FIELD_LOCATION)
        dev_id = cudaCpuDeviceId;
      else
        errorQuda("Invalid QudaFieldLocation.");

      if (gauge) cudaMemPrefetchAsync(gauge, bytes, dev_id, stream);
      if (!isNative()) {
        for (int i = 0; i < nDim; i++) {
          size_t nbytes = nFace * surface[i] * nInternal * precision;
          if (ghost[i] && nbytes) cudaMemPrefetchAsync(ghost[i], nbytes, dev_id, stream);
          if (ghost[i + 4] && nbytes && geometry == QUDA_COARSE_GEOMETRY)
            cudaMemPrefetchAsync(ghost[i + 4], nbytes, dev_id, stream);
        }
      }
    }
  }

  void cudaGaugeField::zero() { qudaMemset(gauge, 0, bytes); }

} // namespace quda
