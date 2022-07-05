#include <cstring>
#include <typeinfo>
#include <gauge_field.h>
#include <timer.h>
#include <blas_quda.h>
#include <device.h>

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
      case QUDA_MEMORY_DEVICE: gauge = bytes ? pool_device_malloc(bytes) : nullptr; break;
      case QUDA_MEMORY_MAPPED:
        gauge_h = bytes ? mapped_malloc(bytes) : nullptr;
        gauge = bytes ? get_mapped_device_pointer(gauge_h) : nullptr; // set the matching device pointer
        break;
      default:
	errorQuda("Unsupported memory type %d", mem_type);
      }
      if (create == QUDA_ZERO_FIELD_CREATE && bytes) qudaMemset(gauge, 0, bytes);
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

    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD) {
      if (isNative()) {
        if (create != QUDA_ZERO_FIELD_CREATE) zeroPad();
      } else {
        for (int i = 0; i < nDim; i++) {
          size_t nbytes = nFace * surface[i] * nInternal * precision;
          qudaMemset(ghost[i], 0, nbytes);
          if (nbytes && geometry == QUDA_COARSE_GEOMETRY) qudaMemset(ghost[i + 4], 0, nbytes);
        }
      }
    }
  }

  void cudaGaugeField::zeroPad() {
    size_t pad_bytes = (stride - volumeCB) * precision * order;
    int Npad = (geometry * (reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : nColor * nColor * 2)) / order;

    size_t pitch = stride*order*precision;
    if (pad_bytes) {
      qudaMemset2D(static_cast<char *>(even) + volumeCB * order * precision, pitch, 0, pad_bytes, Npad);
      qudaMemset2D(static_cast<char *>(odd) + volumeCB * order * precision, pitch, 0, pad_bytes, Npad);
    }
  }

  cudaGaugeField::~cudaGaugeField()
  {
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
        if (ghost[i + 4] && geometry == QUDA_COARSE_GEOMETRY) pool_device_free(ghost[i + 4]);
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
    const lat_dim_t R = {nFace, nFace, nFace, nFace};
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
        recv_d[d] = static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + offset;
        if (bidir) offset += ghost_face_bytes_aligned[d];
        send_d[d] = static_cast<char *>(ghost_send_buffer_d[bufferIndex]) + offset;
        offset += ghost_face_bytes_aligned[d];
      }

      extractGaugeGhost(*this, send_d, true, link_dir*nDim); // get the links into contiguous buffers
      qudaDeviceSynchronize(); // synchronize before issuing mem copies in different streams - could replace with event post and wait

      // issue receive preposts and host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	recvStart(dim, dir); // prepost the receive
	if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) {
          qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
                          ghost_face_bytes[dim], qudaMemcpyDeviceToHost, device::get_stream(2 * dim + dir));
        }
      }

      // if gdr enabled then synchronize
      if (comm_gdr_enabled()) qudaDeviceSynchronize();

      // if the sending direction is not peer-to-peer then we need to synchronize before we start sending
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
        if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled())
          qudaStreamSynchronize(device::get_stream(2 * dim + dir));
        sendStart(dim, dir, device::get_stream(2 * dim + dir)); // start sending
      }

      // complete communication and issue host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	commsComplete(dim, dir);
	if (!comm_peer2peer_enabled(1-dir,dim) && !comm_gdr_enabled()) {
          qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][1 - dir], from_face_dim_dir_h[bufferIndex][dim][1 - dir],
                          ghost_face_bytes[dim], qudaMemcpyHostToDevice, device::get_stream(2 * dim + dir));
        }
      }

      qudaDeviceSynchronize(); // synchronize before issuing kernels / copies in default stream - could replace with event post and wait

      // fill in the halos for non-partitioned dimensions
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim) && no_comms_fill) {
          qudaMemcpy(recv_d[dim], send_d[dim], ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
        }
      }

      if (isNative()) {
	copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, 0, recv_d, 1 + 2*link_dir); // 1, 3
      } else {
	// copy from receive buffer into ghost array
	for (int dim=0; dim<nDim; dim++)
          qudaMemcpy(ghost[dim + link_dir * nDim], recv_d[dim], ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
      }

      bufferIndex = 1-bufferIndex;
    } // link_dir

    qudaDeviceSynchronize();
  }

  // This does the opposite of exchangeGhost and sends back the ghost
  // zone to the node from which it came and injects it back into the
  // field
  void cudaGaugeField::injectGhost(QudaLinkDirection link_direction)
  {
    if (ghostExchange != QUDA_GHOST_EXCHANGE_PAD) errorQuda("Cannot call exchangeGhost with ghostExchange=%d", ghostExchange);
    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY) errorQuda("Invalid geometry=%d", geometry);
    if (link_direction != QUDA_LINK_BACKWARDS) errorQuda("Invalid link_direction = %d", link_direction);
    if (nFace == 0) errorQuda("nFace = 0");

    const int dir = 0; // sending backwards only
    const lat_dim_t R = {nFace, nFace, nFace, nFace};
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
        send_d[d] = static_cast<char *>(ghost_send_buffer_d[bufferIndex]) + offset;
        if (bidir) offset += ghost_face_bytes_aligned[d];
        // receive from forwards is the second half of each ghost_recv_buffer
        recv_d[d] = static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + offset;
        offset += ghost_face_bytes_aligned[d];
      }

      if (isNative()) { // copy from padded region in gauge field into send buffer
	copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, send_d, 0, 1 + 2*link_dir);
      } else { // copy from receive buffer into ghost array
        for (int dim = 0; dim < nDim; dim++)
          qudaMemcpy(send_d[dim], ghost[dim + link_dir * nDim], ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
      }
      qudaDeviceSynchronize(); // need to synchronize before issueing copies in different streams - could replace with event post and wait

      // issue receive preposts and host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	recvStart(dim, dir); // prepost the receive
	if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) {
          qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
                          ghost_face_bytes[dim], qudaMemcpyDeviceToHost, device::get_stream(2 * dim + dir));
        }
      }

      // if gdr enabled then synchronize
      if (comm_gdr_enabled()) qudaDeviceSynchronize();

      // if the sending direction is not peer-to-peer then we need to synchronize before we start sending
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
        if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled())
          qudaStreamSynchronize(device::get_stream(2 * dim + dir));
        sendStart(dim, dir, device::get_stream(2 * dim + dir)); // start sending
      }

      // complete communication and issue host-to-device copies if needed
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim)) continue;
	commsComplete(dim, dir);
	if (!comm_peer2peer_enabled(1-dir,dim) && !comm_gdr_enabled()) {
          qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][1 - dir], from_face_dim_dir_h[bufferIndex][dim][1 - dir],
                          ghost_face_bytes[dim], qudaMemcpyHostToDevice, device::get_stream(2 * dim + dir));
        }
      }

      qudaDeviceSynchronize(); // synchronize before issuing kernel / copies in default stream - could replace with event post and wait

      // fill in the halos for non-partitioned dimensions
      for (int dim=0; dim<nDim; dim++) {
	if (!comm_dim_partitioned(dim) && no_comms_fill) {
          qudaMemcpy(recv_d[dim], send_d[dim], ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
        }
      }

      // get the links into contiguous buffers
      extractGaugeGhost(*this, recv_d, false, link_dir*nDim);

      bufferIndex = 1-bufferIndex;
    } // link_dir

    qudaDeviceSynchronize();
  }

  void cudaGaugeField::allocateGhostBuffer(const lat_dim_t &R, bool no_comms_fill, bool bidir) const
  {
    createGhostZone(R, no_comms_fill, bidir);
    LatticeField::allocateGhostBuffer(ghost_bytes);
  }

  void cudaGaugeField::createComms(const lat_dim_t &R, bool no_comms_fill, bool bidir)
  {
    allocateGhostBuffer(R, no_comms_fill, bidir); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs it comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face_h[0] != ghost_pinned_send_buffer_h[0]) || (my_face_h[1] != ghost_pinned_send_buffer_h[1]) ||
      (from_face_h[0] != ghost_pinned_recv_buffer_h[0]) || (from_face_h[1] != ghost_pinned_recv_buffer_h[1]) ||
      ghost_bytes != ghost_bytes_old; // ghost buffer has been resized (e.g., bidir to unidir)

    if (!initComms || comms_reset) LatticeField::createComms(no_comms_fill);

    if (ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  void cudaGaugeField::recvStart(int dim, int dir)
  {
    if (!comm_dim_partitioned(dim)) return;

    // receive from neighboring the processor
    if (comm_peer2peer_enabled(1 - dir, dim)) {
      comm_start(mh_recv_p2p[bufferIndex][dim][1 - dir]);
    } else if (comm_gdr_enabled()) {
      comm_start(mh_recv_rdma[bufferIndex][dim][1 - dir]);
    } else {
      comm_start(mh_recv[bufferIndex][dim][1 - dir]);
    }
  }

  void cudaGaugeField::sendStart(int dim, int dir, const qudaStream_t &stream)
  {
    if (!comm_dim_partitioned(dim)) return;

    if (!comm_peer2peer_enabled(dir,dim)) {
      if (comm_gdr_enabled()) {
        comm_start(mh_send_rdma[bufferIndex][dim][dir]);
      } else {
        comm_start(mh_send[bufferIndex][dim][dir]);
      }
    } else { // doing peer-to-peer

      void *ghost_dst
        = static_cast<char *>(ghost_remote_send_buffer_d[bufferIndex][dim][dir]) + ghost_offset[dim][(dir + 1) % 2];

      qudaMemcpyP2PAsync(ghost_dst, my_face_dim_dir_d[bufferIndex][dim][dir], ghost_face_bytes[dim], stream);

      // record the event
      qudaEventRecord(ipcCopyEvent[bufferIndex][dim][dir], stream);
      // send to the neighboring processor
      comm_start(mh_send_p2p[bufferIndex][dim][dir]);
    }
  }

  void cudaGaugeField::commsComplete(int dim, int dir)
  {
    if (!comm_dim_partitioned(dim)) return;

    if (comm_peer2peer_enabled(1 - dir, dim)) {
      comm_wait(mh_recv_p2p[bufferIndex][dim][1 - dir]);
      qudaEventSynchronize(ipcRemoteCopyEvent[bufferIndex][dim][1 - dir]);
    } else if (comm_gdr_enabled()) {
      comm_wait(mh_recv_rdma[bufferIndex][dim][1 - dir]);
    } else {
      comm_wait(mh_recv[bufferIndex][dim][1 - dir]);
    }

    if (comm_peer2peer_enabled(dir, dim)) {
      comm_wait(mh_send_p2p[bufferIndex][dim][dir]);
      qudaEventSynchronize(ipcCopyEvent[bufferIndex][dim][dir]);
    } else if (comm_gdr_enabled()) {
      comm_wait(mh_send_rdma[bufferIndex][dim][dir]);
    } else {
      comm_wait(mh_send[bufferIndex][dim][dir]);
    }
  }

  void cudaGaugeField::exchangeExtendedGhost(const lat_dim_t &R, bool no_comms_fill)
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
      qudaMemsetAsync(send_d[dim], 0, 2 * ghost_face_bytes_aligned[dim], device::get_default_stream());
      offset += 2 * ghost_face_bytes_aligned[dim]; // factor of two from fwd/back
    }

    for (int dim=0; dim<nDim; dim++) {
      if ( !(comm_dim_partitioned(dim) || (no_comms_fill && R[dim])) ) continue;

      //extract into a contiguous buffer
      extractExtendedGaugeGhost(*this, dim, R, send_d, true);

      if (comm_dim_partitioned(dim)) {
        qudaDeviceSynchronize(); // synchronize before issuing mem copies in different streams - could replace with event post and wait

        for (int dir=0; dir<2; dir++) recvStart(dim, dir);

	for (int dir=0; dir<2; dir++) {
	  // issue host-to-device copies if needed
	  if (!comm_peer2peer_enabled(dir,dim) && !comm_gdr_enabled()) {
            qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
                            ghost_face_bytes[dim], qudaMemcpyDeviceToHost, device::get_stream(dir));
          }
        }

        // if either direction is not peer-to-peer then we need to synchronize
        if (!comm_peer2peer_enabled(0, dim) || !comm_peer2peer_enabled(1, dim)) qudaDeviceSynchronize();

        for (int dir = 0; dir < 2; dir++) sendStart(dim, dir, device::get_stream(dir));
        for (int dir = 0; dir < 2; dir++) commsComplete(dim, dir);

        for (int dir = 0; dir < 2; dir++) {
          // issue host-to-device copies if needed
          if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled()) {
            qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][dir], from_face_dim_dir_h[bufferIndex][dim][dir],
                            ghost_face_bytes[dim], qudaMemcpyHostToDevice, device::get_stream(dir));
          }
        }

      } else { // if just doing a local exchange to fill halo then need to swap faces
        qudaMemcpy(from_face_dim_dir_d[b][dim][1], my_face_dim_dir_d[b][dim][0], ghost_face_bytes[dim],
                   qudaMemcpyDeviceToDevice);
        qudaMemcpy(from_face_dim_dir_d[b][dim][0], my_face_dim_dir_d[b][dim][1], ghost_face_bytes[dim],
                   qudaMemcpyDeviceToDevice);
      }

      // inject back into the gauge field
      // need to synchronize the copy streams before rejoining the compute stream - could replace with event post and wait
      qudaDeviceSynchronize();
      extractExtendedGaugeGhost(*this, dim, R, recv_d, false);
    }

    bufferIndex = 1-bufferIndex;
    qudaDeviceSynchronize();
  }

  void cudaGaugeField::exchangeExtendedGhost(const lat_dim_t &R, TimeProfile &profile, bool no_comms_fill)
  {
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
        qudaMemcpy(gauge, buffer, bytes, qudaMemcpyDefault);
        pool_pinned_free(buffer);
      } else { // else on the GPU

        if (src.Order() == QUDA_MILC_SITE_GAUGE_ORDER ||
            src.Order() == QUDA_BQCD_GAUGE_ORDER      ||
            src.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
	  // special case where we use zero-copy memory to read/write directly from application's array
          void *src_d = get_mapped_device_pointer(src.Gauge_p());

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
              qudaMemcpy(((void **)buffer)[d], ((void **)src.Gauge_p())[d], src.Bytes() / geometry, qudaMemcpyDefault);
            }
          } else {
            qudaMemcpy(buffer, src.Gauge_p(), src.Bytes(), qudaMemcpyDefault);
          }

          if (src.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD
              && src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
            for (int d = 0; d < geometry; d++)
              qudaMemcpy(ghost_buffer[d], src.Ghost()[d], ghost_bytes[d], qudaMemcpyDefault);

          if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
            copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer, 0, ghost_buffer);
            if (geometry == QUDA_COARSE_GEOMETRY)
              copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, gauge, buffer, 0, ghost_buffer, 3);
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
  }

  void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu) {
    copy(cpu);
    qudaDeviceSynchronize();
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
        void *cpu_d = get_mapped_device_pointer(cpu.Gauge_p());
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
          for (int d = 0; d < geometry; d++)
            qudaMemcpy(((void **)cpu.gauge)[d], ((void **)buffer)[d], cpu.Bytes() / geometry, qudaMemcpyDefault);
        } else {
          qudaMemcpy(cpu.gauge, buffer, cpu.Bytes(), qudaMemcpyDefault);
        }

        if (cpu.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD
            && cpu.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
          for (int d = 0; d < geometry; d++)
            qudaMemcpy(cpu.Ghost()[d], ghost_buffer[d], ghost_bytes[d], qudaMemcpyDefault);

        free_gauge_buffer(buffer, cpu.Order(), cpu.Geometry());
        if (nFace > 0) free_ghost_buffer(ghost_buffer, cpu.Order(), geometry);
      }
    } else if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do copy then host-side reorder

      void *buffer = pool_pinned_malloc(bytes);
      qudaMemcpy(buffer, gauge, bytes, qudaMemcpyDefault);

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
  }

  void cudaGaugeField::saveCPUField(cpuGaugeField &cpu, TimeProfile &profile) const {
    profile.TPSTART(QUDA_PROFILE_D2H);
    saveCPUField(cpu);
    profile.TPSTOP(QUDA_PROFILE_D2H);
  }

  void cudaGaugeField::backup() const {
    if (backed_up) errorQuda("Gauge field already backed up");
    backup_h = new char[bytes];
    qudaMemcpy(backup_h, gauge, bytes, qudaMemcpyDefault);
    backed_up = true;
  }

  void cudaGaugeField::restore() const
  {
    if (!backed_up) errorQuda("Cannot restore since not backed up");
    qudaMemcpy(gauge, backup_h, bytes, qudaMemcpyDefault);
    delete []backup_h;
    backed_up = false;
  }

  void cudaGaugeField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    if (is_prefetch_enabled() && mem_type == QUDA_MEMORY_DEVICE) {
      if (gauge) qudaMemPrefetchAsync(gauge, bytes, mem_space, stream);
      if (!isNative()) {
        for (int i = 0; i < nDim; i++) {
          size_t nbytes = nFace * surface[i] * nInternal * precision;
          if (ghost[i] && nbytes) qudaMemPrefetchAsync(ghost[i], nbytes, mem_space, stream);
          if (ghost[i + 4] && nbytes && geometry == QUDA_COARSE_GEOMETRY)
            qudaMemPrefetchAsync(ghost[i + 4], nbytes, mem_space, stream);
        }
      }
    }
  }

  void cudaGaugeField::zero() { qudaMemset(gauge, 0, bytes); }

  void cudaGaugeField::copy_to_buffer(void *buffer) const
  {
    qudaMemcpy(buffer, Gauge_p(), Bytes(), qudaMemcpyDeviceToHost);
  }

  void cudaGaugeField::copy_from_buffer(void *buffer)
  {
    qudaMemcpy(Gauge_p(), buffer, Bytes(), qudaMemcpyHostToDevice);
  }

} // namespace quda
