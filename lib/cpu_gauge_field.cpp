#include <quda_internal.h>
#include <timer.h>
#include <gauge_field.h>
#include <assert.h>
#include <string.h>
#include <typeinfo>

namespace quda {

  cpuGaugeField::cpuGaugeField(const GaugeFieldParam &param) :
    GaugeField(param)
  {
    // exchange the boundaries if a non-trivial field
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD)
      if (create == QUDA_REFERENCE_FIELD_CREATE && (geometry == QUDA_VECTOR_GEOMETRY || geometry == QUDA_COARSE_GEOMETRY)) {
        exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);
      }

    // compute the fat link max now in case it is needed later (i.e., for half precision)
    if (param.compute_fat_link_max) fat_link_max = this->abs_max();
  }

  // This does the exchange of the gauge field ghost zone and places it
  // into the ghost array.
  void cpuGaugeField::exchangeGhost(QudaLinkDirection link_direction) {
    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot exchange for %d geometry gauge field", geometry);

    if ( (link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == QUDA_LINK_FORWARDS) && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot request exchange of forward links on non-coarse geometry");

    void *send[2 * QUDA_MAX_DIM];
    for (int d=0; d<nDim; d++) {
      send[d] = safe_malloc(nFace * surface[d] * nInternal * precision);
      if (geometry == QUDA_COARSE_GEOMETRY) send[d+4] = safe_malloc(nFace * surface[d] * nInternal * precision);
    }

    void *ghost_[2 * QUDA_MAX_DIM];
    for (auto i = 0; i < geometry; i++) ghost_[i] = ghost[i].data();

    // get the links into contiguous buffers
    if (link_direction == QUDA_LINK_BACKWARDS || link_direction == QUDA_LINK_BIDIRECTIONAL) {
      extractGaugeGhost(*this, send, true);

      // communicate between nodes
      exchange(ghost_, send, QUDA_FORWARDS);
    }

    // repeat if requested and links are bi-directional
    if (link_direction == QUDA_LINK_FORWARDS || link_direction == QUDA_LINK_BIDIRECTIONAL) {
      extractGaugeGhost(*this, send, true, nDim);
      exchange(ghost_+nDim, send+nDim, QUDA_FORWARDS);
    }

    for (int d = 0; d < geometry; d++) host_free(send[d]);
  }

  // This does the opposite of exchangeGhost and sends back the ghost
  // zone to the node from which it came and injects it back into the
  // field
  void cpuGaugeField::injectGhost(QudaLinkDirection link_direction) {
    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot exchange for %d geometry gauge field", geometry);

    if (link_direction != QUDA_LINK_BACKWARDS)
      errorQuda("link_direction = %d not supported", link_direction);

    void *recv[QUDA_MAX_DIM];
    for (int d=0; d<nDim; d++) recv[d] = safe_malloc(nFace*surface[d]*nInternal*precision);

    void *ghost_[] = {ghost[0].data(), ghost[1].data(), ghost[2].data(), ghost[3].data(),
                      ghost[4].data(), ghost[5].data(), ghost[6].data(), ghost[7].data()};

    // communicate between nodes
    exchange(recv, ghost_, QUDA_BACKWARDS);

    // get the links into contiguous buffers
    extractGaugeGhost(*this, recv, false);

    for (int d = 0; d < QUDA_MAX_DIM; d++) host_free(recv[d]);
  }

  void cpuGaugeField::exchangeExtendedGhost(const lat_dim_t &R, bool no_comms_fill)
  {

    void *send[QUDA_MAX_DIM];
    void *recv[QUDA_MAX_DIM];
    size_t bytes[QUDA_MAX_DIM];
    // store both parities and directions in each
    for (int d=0; d<nDim; d++) {
      if (!(comm_dim_partitioned(d) || (no_comms_fill && R[d])) ) continue;
      bytes[d] = surface[d] * R[d] * geometry * nInternal * precision;
      send[d] = safe_malloc(2 * bytes[d]);
      recv[d] = safe_malloc(2 * bytes[d]);
    }

    for (int d=0; d<nDim; d++) {
      if (!(comm_dim_partitioned(d) || (no_comms_fill && R[d])) ) continue;
      //extract into a contiguous buffer
      extractExtendedGaugeGhost(*this, d, R, send, true);

      if (comm_dim_partitioned(d)) {
	// do the exchange
	MsgHandle *mh_recv_back;
	MsgHandle *mh_recv_fwd;
	MsgHandle *mh_send_fwd;
	MsgHandle *mh_send_back;

	mh_recv_back = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
	mh_recv_fwd  = comm_declare_receive_relative(((char*)recv[d])+bytes[d], d, +1, bytes[d]);
	mh_send_back = comm_declare_send_relative(send[d], d, -1, bytes[d]);
	mh_send_fwd  = comm_declare_send_relative(((char*)send[d])+bytes[d], d, +1, bytes[d]);

	comm_start(mh_recv_back);
	comm_start(mh_recv_fwd);
	comm_start(mh_send_fwd);
	comm_start(mh_send_back);

	comm_wait(mh_send_fwd);
	comm_wait(mh_send_back);
	comm_wait(mh_recv_back);
	comm_wait(mh_recv_fwd);

	comm_free(mh_send_fwd);
	comm_free(mh_send_back);
	comm_free(mh_recv_back);
	comm_free(mh_recv_fwd);
      } else {
	memcpy(static_cast<char*>(recv[d])+bytes[d], send[d], bytes[d]);
	memcpy(recv[d], static_cast<char*>(send[d])+bytes[d], bytes[d]);
      }

      // inject back into the gauge field
      extractExtendedGaugeGhost(*this, d, R, recv, false);
    }

    for (int d=0; d<nDim; d++) {
      if (!(comm_dim_partitioned(d) || (no_comms_fill && R[d])) ) continue;
      host_free(send[d]);
      host_free(recv[d]);
    }
  }

  void cpuGaugeField::exchangeExtendedGhost(const lat_dim_t &R, TimeProfile &profile, bool no_comms_fill)
  {
    profile.TPSTART(QUDA_PROFILE_COMMS);
    exchangeExtendedGhost(R, no_comms_fill);
    profile.TPSTOP(QUDA_PROFILE_COMMS);
  }

  // defined in cudaGaugeField
  void *create_gauge_buffer(size_t bytes, QudaGaugeFieldOrder order, QudaFieldGeometry geometry);
  void **create_ghost_buffer(size_t bytes[], QudaGaugeFieldOrder order, QudaFieldGeometry geometry);
  void free_gauge_buffer(void *buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry);
  void free_ghost_buffer(void **buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry);

  void cpuGaugeField::copy(const GaugeField &src) {
    if (this == &src) return;

    checkField(src);

    if (link_type == QUDA_ASQTAD_FAT_LINKS) {
      fat_link_max = src.LinkMax();
      if (fat_link_max == 0.0 && precision < QUDA_SINGLE_PRECISION) fat_link_max = src.abs_max();
    } else {
      fat_link_max = 1.0;
    }

    if (typeid(src) == typeid(cudaGaugeField)) {

      if (reorder_location() == QUDA_CPU_FIELD_LOCATION) {

	if (!src.isNative()) errorQuda("Only native order is supported");
	void *buffer = pool_pinned_malloc(src.Bytes());
        qudaMemcpy(buffer, src.data(), src.Bytes(), qudaMemcpyDeviceToHost);

        copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, nullptr, buffer);
        pool_pinned_free(buffer);

      } else { // else on the GPU

	void *buffer = create_gauge_buffer(bytes, order, geometry);
	size_t ghost_bytes[8];
	int dstNinternal = reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : 2*nColor*nColor;
	for (int d=0; d<geometry; d++) ghost_bytes[d] = nFace * surface[d%4] * dstNinternal * precision;
	void **ghost_buffer = (nFace > 0) ? create_ghost_buffer(ghost_bytes, order, geometry) : nullptr;

	if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED) {
          copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, buffer, nullptr, ghost_buffer, nullptr);
          if (geometry == QUDA_COARSE_GEOMETRY)
            copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, buffer, nullptr, ghost_buffer, nullptr,
                             3); // forwards links if bi-directional
        } else {
	  copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, buffer, 0);
	}

	if (order == QUDA_QDP_GAUGE_ORDER) {
	  for (int d=0; d<geometry; d++) {
            qudaMemcpy(gauge_array[d].data(), ((void **)buffer)[d], bytes / geometry, qudaMemcpyDeviceToHost);
          }
	} else {
          qudaMemcpy(gauge.data(), buffer, bytes, qudaMemcpyHostToDevice);
        }

	if (order > 4 && ghostExchange == QUDA_GHOST_EXCHANGE_PAD && src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
	  for (int d=0; d<geometry; d++)
            qudaMemcpy(Ghost()[d].data(), ghost_buffer[d], ghost_bytes[d], qudaMemcpyDeviceToHost);

        free_gauge_buffer(buffer, order, geometry);
	if (nFace > 0) free_ghost_buffer(ghost_buffer, order, geometry);
      }

    } else if (typeid(src) == typeid(cpuGaugeField)) {
      // copy field and ghost zone directly
      copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION);
    } else {
      errorQuda("Invalid gauge field type");
    }

    // if we have copied from a source without a pad then we need to exchange
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD &&
	src.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD) {
      exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);
    }
  }

  void cpuGaugeField::copy_to_buffer(void *buffer) const
  {
    if (is_pointer_array(order)) {
      char *dst_buffer = reinterpret_cast<char *>(buffer);
      for (int d = 0; d < site_dim; d++) {
        std::memcpy(&dst_buffer[d * bytes / site_dim], gauge_array[d].data(), bytes / site_dim);
      }
    } else if (Order() == QUDA_CPS_WILSON_GAUGE_ORDER || Order() == QUDA_MILC_GAUGE_ORDER
               || Order() == QUDA_MILC_SITE_GAUGE_ORDER || Order() == QUDA_BQCD_GAUGE_ORDER
               || Order() == QUDA_TIFR_GAUGE_ORDER || Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
      std::memcpy(buffer, data(), Bytes());
    } else {
      errorQuda("Unsupported order = %d", Order());
    }
  }

  void cpuGaugeField::copy_from_buffer(void *buffer)
  {
    if (is_pointer_array(order)) {
      const char *dst_buffer = reinterpret_cast<const char *>(buffer);
      for (int d = 0; d < site_dim; d++) {
        std::memcpy(gauge_array[d].data(), &dst_buffer[d * bytes / site_dim], bytes / site_dim);
      }
    } else if (Order() == QUDA_CPS_WILSON_GAUGE_ORDER || Order() == QUDA_MILC_GAUGE_ORDER
               || Order() == QUDA_MILC_SITE_GAUGE_ORDER || Order() == QUDA_BQCD_GAUGE_ORDER
               || Order() == QUDA_TIFR_GAUGE_ORDER || Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
      std::memcpy(data(), buffer, Bytes());
    } else {
      errorQuda("Unsupported order = %d", Order());
    }
  }

} // namespace quda
