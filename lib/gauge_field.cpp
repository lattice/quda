#include <typeinfo>
#include <gauge_field.h>
#include <blas_quda.h>
#include <timer.h>

namespace quda {

  GaugeFieldParam::GaugeFieldParam(const GaugeField &u) : LatticeFieldParam(u) { u.fill(*this); }

  GaugeField::GaugeField(const GaugeFieldParam &param) : LatticeField(param)
  {
    create(param);

    switch (param.create) {
    case QUDA_NULL_FIELD_CREATE:
    case QUDA_REFERENCE_FIELD_CREATE: break; // do nothing
    case QUDA_ZERO_FIELD_CREATE: zero(); break;
    case QUDA_COPY_FIELD_CREATE: copy(*param.field); break;
    default: errorQuda("ERROR: create type(%d) not supported yet", param.create);
    }
  }

  GaugeField::GaugeField(const GaugeField &u) noexcept : LatticeField(u)
  {
    GaugeFieldParam param;
    u.fill(param);
    param.create = QUDA_COPY_FIELD_CREATE;
    create(param);
    copy(u);
  }

  GaugeField::GaugeField(GaugeField &&u) noexcept : LatticeField(std::move(u)) { move(std::move(u)); }

  GaugeField &GaugeField::operator=(const GaugeField &src)
  {
    if (src.empty()) errorQuda("Copying from empty field");
    if (&src != this) {
      if (!init) { // keep current attributes unless unset
        LatticeField::operator=(src);
        GaugeFieldParam param;
        src.fill(param);
        param.create = QUDA_COPY_FIELD_CREATE;
        create(param);
      }

      copy(src);
    }
    return *this;
  }

  GaugeField &GaugeField::operator=(GaugeField &&src)
  {
    if (&src != this) {
      // if field not already initialized then move the field
      if (!init || are_compatible(*this, src) || src.empty()) {
        LatticeField::operator=(std::move(src));
        move(std::move(src));
      } else {
        // we error if the field is not compatible with this
        errorQuda("Moving to already created field");
      }
    }
    return *this;
  }

  void GaugeField::create(const GaugeFieldParam &param)
  {
    if (param.siteSubset != QUDA_FULL_SITE_SUBSET) errorQuda("Unexpected siteSubset %d", param.siteSubset);
    if (param.order == QUDA_NATIVE_GAUGE_ORDER) errorQuda("Invalid gauge order %d", param.order);
    if (param.GhostPrecision() != param.Precision())
      errorQuda("Ghost precision %d doesn't match field precision %d", param.GhostPrecision(), param.Precision());
    if (param.link_type != QUDA_COARSE_LINKS && param.nColor != 3)
      errorQuda("nColor must be 3, not %d for this link type", param.nColor);
    if (param.nDim != 4) errorQuda("Number of dimensions must be 4 not %d", param.nDim);
    if (param.link_type != QUDA_WILSON_LINKS && param.anisotropy != 1.0)
      errorQuda("Anisotropy only supported for Wilson links");
    if (param.link_type != QUDA_WILSON_LINKS && param.fixed == QUDA_GAUGE_FIXED_YES)
      errorQuda("Temporal gauge fixing only supported for Wilson links");
    if ((param.reconstruct == QUDA_RECONSTRUCT_12 || param.reconstruct == QUDA_RECONSTRUCT_8)
        && param.link_type != QUDA_SU3_LINKS)
      errorQuda("Cannot request a 12/8 reconstruct type without SU(3) link type");
    if (param.reconstruct == QUDA_RECONSTRUCT_10 && param.link_type != QUDA_ASQTAD_MOM_LINKS)
      errorQuda("10-reconstruction only supported with momentum links");

    nColor = param.nColor;
    nFace = param.nFace;
    geometry = param.geometry;
    reconstruct = param.reconstruct;
    nInternal = reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : nColor * nColor * 2;
    order = param.order;
    fixed = param.fixed;
    link_type = param.link_type;
    t_boundary = param.t_boundary;
    anisotropy = param.anisotropy;
    tadpole = param.tadpole;
    fat_link_max = link_type == QUDA_ASQTAD_FAT_LINKS ? 0.0 : 1.0;
    staggeredPhaseType = param.staggeredPhaseType;
    staggeredPhaseApplied = param.staggeredPhaseApplied;
    i_mu = param.i_mu;
    site_offset = param.site_offset;
    site_size = param.site_size;

    if (geometry == QUDA_SCALAR_GEOMETRY) {
      real_length = volume*nInternal;
      length = 2*stride*nInternal; // two comes from being full lattice
    } else if (geometry == QUDA_VECTOR_GEOMETRY) {
      real_length = nDim*volume*nInternal;
      length = 2*nDim*stride*nInternal; // two comes from being full lattice
    } else if (geometry == QUDA_TENSOR_GEOMETRY) {
      real_length = (nDim*(nDim-1)/2)*volume*nInternal;
      length = 2*(nDim*(nDim-1)/2)*stride*nInternal; // two comes from being full lattice
    } else if (geometry == QUDA_COARSE_GEOMETRY) {
      real_length = 2*nDim*volume*nInternal;
      length = 2*2*nDim*stride*nInternal;  //two comes from being full lattice
    } else if (geometry == QUDA_KDINVERSE_GEOMETRY) {
      real_length = (1 << nDim) * volume * nInternal;
      length = 2 * (1 << nDim) * nDim * stride * nInternal; // two comes from being full lattice
    }

    switch (geometry) {
    case QUDA_SCALAR_GEOMETRY: site_dim = 1; break;
    case QUDA_VECTOR_GEOMETRY: site_dim = nDim; break;
    case QUDA_TENSOR_GEOMETRY: site_dim = nDim * (nDim - 1) / 2; break;
    case QUDA_COARSE_GEOMETRY: site_dim = 2 * nDim; break;
    case QUDA_KDINVERSE_GEOMETRY: site_dim = 1 << nDim; break;
    default: errorQuda("Unknown geometry type %d", geometry);
    }

    if (isNative()) {
      if (reconstruct == QUDA_RECONSTRUCT_9 || reconstruct == QUDA_RECONSTRUCT_13) {
        // Need to adjust the phase alignment as well.
        size_t half_phase_bytes
          = (length / (2 * reconstruct)) * precision; // bytes needed to store phases for a single parity
        size_t half_gauge_bytes = (length / 2) * precision
          - half_phase_bytes; // bytes needed to store the gauge field for a single parity excluding the phases
        // Adjust the alignments for the gauge and phase separately
        half_phase_bytes = ALIGNMENT_ADJUST(half_phase_bytes);
        half_gauge_bytes = ALIGNMENT_ADJUST(half_gauge_bytes);
        phase_offset = half_gauge_bytes;
        phase_bytes = half_phase_bytes * 2;
        bytes = (half_gauge_bytes + half_phase_bytes) * 2;
      } else {
        bytes = length * precision;
        bytes = 2 * ALIGNMENT_ADJUST(bytes / 2);
      }
    } else {
      // compute the correct bytes size for these padded field orders
      if (order == QUDA_TIFR_PADDED_GAUGE_ORDER) {
        bytes = site_dim * (x[0] * x[1] * (x[2] + 4) * x[3]) * nInternal * precision;
      } else if (order == QUDA_BQCD_GAUGE_ORDER) {
        bytes = site_dim * (x[0] + 4) * (x[1] + 2) * (x[2] + 2) * (x[3] + 2) * nInternal * precision;
      } else if (order == QUDA_MILC_SITE_GAUGE_ORDER) {
        bytes = volume * site_size;
      } else {
        bytes = length * precision;
      }
    }

    total_bytes = bytes;

    if (isNative() && ghostExchange == QUDA_GHOST_EXCHANGE_PAD) {
      bool pad_check = true;
      for (int i = 0; i < nDim; i++) {
        // when we have coarse links we need to double the pad since we're storing forwards and backwards links
        int minimum_pad = comm_dim_partitioned(i) ? nFace * surfaceCB[i] * (geometry == QUDA_COARSE_GEOMETRY ? 2 : 1) : 0;
        if (pad < minimum_pad) pad_check = false;
        if (!pad_check)
          errorQuda("GaugeField being constructed with insufficient padding in dim %d (%d < %d)", i, pad, minimum_pad);
      }
    }

    if (isNative()) {
      if (param.create != QUDA_REFERENCE_FIELD_CREATE) {
        gauge = quda_ptr(mem_type, bytes);
      } else {
        gauge = quda_ptr(param.gauge, mem_type);
      }
    } else if (is_pointer_array(order)) {

      size_t nbytes = volume * nInternal * precision;
      for (int d = 0; d < site_dim; d++) {
        if (param.create != QUDA_REFERENCE_FIELD_CREATE) {
          gauge_array[d] = quda_ptr(mem_type, nbytes);
        } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
          if (param.gauge) gauge_array[d] = quda_ptr(static_cast<void **>(param.gauge)[d], mem_type);
        } else {
          errorQuda("Unsupported creation type %d", param.create);
        }
      }

    } else if (order == QUDA_CPS_WILSON_GAUGE_ORDER || order == QUDA_MILC_GAUGE_ORDER || order == QUDA_BQCD_GAUGE_ORDER
               || order == QUDA_TIFR_GAUGE_ORDER || order == QUDA_TIFR_PADDED_GAUGE_ORDER
               || order == QUDA_MILC_SITE_GAUGE_ORDER) {
      // does not support device

      if (order == QUDA_MILC_SITE_GAUGE_ORDER && param.create != QUDA_REFERENCE_FIELD_CREATE) {
        errorQuda("MILC site gauge order only supported for reference fields");
      }

      if (param.create != QUDA_REFERENCE_FIELD_CREATE) {
        gauge = quda_ptr(mem_type, bytes);
      } else if (param.create == QUDA_REFERENCE_FIELD_CREATE) {
        gauge = quda_ptr(param.gauge, mem_type);
      } else {
        errorQuda("Unsupported creation type %d", param.create);
      }

    } else {
      errorQuda("Unsupported gauge order type %d", order);
    }

    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD) {
      if (!isNative()) {
        for (int i = 0; i < nDim; i++) {
          size_t nbytes = nFace * surface[i] * nInternal * precision;
          ghost[i] = quda_ptr(mem_type, nbytes);
          if (geometry == QUDA_COARSE_GEOMETRY) ghost[i + 4] = quda_ptr(mem_type, nbytes);

          qudaMemset(ghost[i], 0, nbytes);
          if (geometry == QUDA_COARSE_GEOMETRY) qudaMemset(ghost[i + 4], 0, nbytes);
        }
      } else {
        if (param.create != QUDA_ZERO_FIELD_CREATE) zeroPad();
      }
    }

    init = true;
    setTuningString();

    // exchange the boundaries if a non-trivial field
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD)
      if (param.create == QUDA_REFERENCE_FIELD_CREATE
          && (geometry == QUDA_VECTOR_GEOMETRY || geometry == QUDA_COARSE_GEOMETRY)) {
        exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);
      }

    // compute the fat link max now in case it is needed later (i.e., for half precision)
    if (param.compute_fat_link_max) fat_link_max = this->abs_max();
  }

  void GaugeField::move(GaugeField &&src)
  {
    init = std::exchange(src.init, {});
    if (src.gauge.is_reference()) errorQuda("Cannot move a reference allocation");
    gauge.exchange(src.gauge, {});
    for (auto i = 0; i < gauge_array.size(); i++) gauge_array[i].exchange(src.gauge_array[i], {});
    bytes = std::exchange(src.bytes, 0);
    phase_offset = std::exchange(src.phase_offset, 0);
    phase_bytes = std::exchange(src.phase_bytes, 0);
    length = std::exchange(src.length, 0);
    real_length = std::exchange(src.real_length, 0);
    nColor = std::exchange(src.nColor, 0);
    nFace = std::exchange(src.nFace, 0);
    geometry = std::exchange(src.geometry, QUDA_INVALID_GEOMETRY);
    site_dim = std::exchange(src.site_dim, 0);
    reconstruct = std::exchange(src.reconstruct, QUDA_RECONSTRUCT_INVALID);
    nInternal = std::exchange(src.nInternal, 0);
    order = std::exchange(src.order, QUDA_INVALID_GAUGE_ORDER);
    fixed = std::exchange(src.fixed, QUDA_GAUGE_FIXED_INVALID);
    link_type = std::exchange(src.link_type, QUDA_INVALID_LINKS);
    t_boundary = std::exchange(src.t_boundary, QUDA_INVALID_T_BOUNDARY);
    anisotropy = std::exchange(src.anisotropy, 0.0);
    tadpole = std::exchange(src.tadpole, 0.0);
    fat_link_max = std::exchange(src.fat_link_max, 0.0);
    for (auto i = 0; i < ghost.size(); i++) ghost[i].exchange(src.ghost[i], {});
    ghostFace = std::exchange(src.ghostFace, {});
    staggeredPhaseType = std::exchange(src.staggeredPhaseType, QUDA_STAGGERED_PHASE_INVALID);
    staggeredPhaseApplied = std::exchange(src.staggeredPhaseApplied, false);
    i_mu = std::exchange(src.i_mu, 0.0);
    site_offset = std::exchange(src.site_offset, 0);
    site_size = std::exchange(src.site_size, 0);
  }

  void GaugeField::fill(GaugeFieldParam &param) const
  {
    LatticeField::fill(param);
    param.gauge = nullptr;
    param.nColor = nColor;
    param.nFace = nFace;
    param.reconstruct = reconstruct;
    param.order = order;
    param.fixed = fixed;
    param.link_type = link_type;
    param.t_boundary = t_boundary;
    param.anisotropy = anisotropy;
    param.tadpole = tadpole;
    param.create = QUDA_NULL_FIELD_CREATE;
    param.geometry = geometry;
    param.compute_fat_link_max = false;
    param.staggeredPhaseType = staggeredPhaseType;
    param.staggeredPhaseApplied = staggeredPhaseApplied;
    param.i_mu = i_mu;
    param.site_offset = site_offset;
    param.site_size = site_size;
  }

  void GaugeField::setTuningString()
  {
    LatticeField::setTuningString();
    std::stringstream aux_ss;
    aux_ss << "vol=" << volume << ",stride=" << stride << ",precision=" << precision << ",geometry=" << geometry
           << ",Nc=" << nColor;
    if (ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED) aux_ss << ",r=" << r[0] << r[1] << r[2] << r[3];
    aux_string = aux_ss.str();
    if (aux_string.size() >= TuneKey::aux_n / 2) errorQuda("Aux string too large %lu", aux_string.size());
  }

  void GaugeField::zeroPad()
  {
    if (!isNative()) return;
    size_t pad_bytes = (stride - volumeCB) * precision * order;
    int Npad = (geometry * (reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : nColor * nColor * 2)) / order;

    size_t pitch = stride * order * precision;
    if (pad_bytes) {
      for (int parity = 0; parity < 2; parity++) {
        qudaMemset2DAsync(gauge, parity * (bytes / 2) + volumeCB * order * precision, pitch, 0, pad_bytes, Npad,
                          device::get_default_stream());
      }
    }
  }

  void GaugeField::createGhostZone(const lat_dim_t &R, bool no_comms_fill, bool bidir) const
  {
    if (location == QUDA_CPU_FIELD_LOCATION) return;

    // if this is not a bidirectional exchange then we are doing a
    // scalar exchange, e.g., only the link matrix in the direcion we
    // are exchanging is exchanged, and none of the orthogonal links
    QudaFieldGeometry geometry_comms = bidir ? (geometry == QUDA_COARSE_GEOMETRY ? QUDA_VECTOR_GEOMETRY : geometry) : QUDA_SCALAR_GEOMETRY;

    // calculate size of ghost zone required
    ghost_bytes_old = ghost_bytes; // save for subsequent resize checking
    ghost_bytes = 0;
    for (int i=0; i<nDim; i++) {
      ghost_face_bytes[i] = 0;
      if ( !(comm_dim_partitioned(i) || (no_comms_fill && R[i])) ) ghostFace[i] = 0;
      else ghostFace[i] = surface[i] * R[i]; // includes the radius (unlike ColorSpinorField)

      ghost_face_bytes[i] = ghostFace[i] * geometry_comms * nInternal * ghost_precision;
      ghost_face_bytes_aligned[i] = ghost_face_bytes[i];

      ghost_offset[i][0] = (i == 0) ? 0 : ghost_offset[i - 1][1] + ghost_face_bytes_aligned[i - 1];
      ghost_offset[i][1] = (bidir ? ghost_offset[i][0] + ghost_face_bytes_aligned[i] : ghost_offset[i][0]);

      ghost_bytes += (bidir ? 2 : 1) * ghost_face_bytes_aligned[i]; // factor of two from direction
    }

    if (isNative()) ghost_bytes = ALIGNMENT_ADJUST(ghost_bytes);
  } // createGhostZone

  void GaugeField::applyStaggeredPhase(QudaStaggeredPhase phase) {
    if (staggeredPhaseApplied) errorQuda("Staggered phases already applied");

    if (phase != QUDA_STAGGERED_PHASE_INVALID) staggeredPhaseType = phase;
    applyGaugePhase(*this);
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD) exchangeGhost();
    staggeredPhaseApplied = true;
  }

  void GaugeField::removeStaggeredPhase() {
    if (!staggeredPhaseApplied) errorQuda("No staggered phases to remove");
    applyGaugePhase(*this);
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD) exchangeGhost();
    staggeredPhaseApplied = false;
  }

  void GaugeField::createComms(const lat_dim_t &R, bool no_comms_fill, bool bidir)
  {
    allocateGhostBuffer(R, no_comms_fill, bidir); // allocate the ghost buffer if not yet allocated

    // ascertain if this instance needs it comms buffers to be updated
    bool comms_reset = ghost_field_reset || // FIXME add send buffer check
      (my_face_h[0] != ghost_pinned_send_buffer_h[0]) || (my_face_h[1] != ghost_pinned_send_buffer_h[1])
      || (from_face_h[0] != ghost_pinned_recv_buffer_h[0]) || (from_face_h[1] != ghost_pinned_recv_buffer_h[1])
      || ghost_bytes != ghost_bytes_old; // ghost buffer has been resized (e.g., bidir to unidir)

    if (!initComms || comms_reset) LatticeField::createComms(no_comms_fill);

    if (ghost_field_reset) destroyIPCComms();
    createIPCComms();
  }

  void GaugeField::allocateGhostBuffer(const lat_dim_t &R, bool no_comms_fill, bool bidir) const
  {
    createGhostZone(R, no_comms_fill, bidir);
    LatticeField::allocateGhostBuffer(ghost_bytes);
  }

  void GaugeField::recvStart(int dim, int dir)
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

  void GaugeField::sendStart(int dim, int dir, const qudaStream_t &stream)
  {
    if (!comm_dim_partitioned(dim)) return;

    if (!comm_peer2peer_enabled(dir, dim)) {
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

  void GaugeField::commsComplete(int dim, int dir)
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

  // This does the exchange of the forwards boundary gauge field ghost zone and places
  // it into the ghost array of the next node
  void GaugeField::exchangeGhost(QudaLinkDirection link_direction)
  {
    if (ghostExchange != QUDA_GHOST_EXCHANGE_PAD)
      errorQuda("Cannot call exchangeGhost with ghostExchange=%d", ghostExchange);
    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Invalid geometry=%d", geometry);
    if ((link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == QUDA_LINK_FORWARDS)
        && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Cannot request exchange of forward links on non-coarse geometry");
    if (nFace == 0) errorQuda("nFace = 0");

    if (location == QUDA_CUDA_FIELD_LOCATION) {
      const int dir = 1; // sending forwards only
      const lat_dim_t R = {nFace, nFace, nFace, nFace};
      const bool no_comms_fill = true; // dslash kernels presently require this
      const bool bidir = false;        // communication is only ever done in one direction at once
      createComms(R, true, bidir); // always need to allocate space for non-partitioned dimension for copyGenericGauge

      // loop over backwards and forwards links
      const QudaLinkDirection directions[] = {QUDA_LINK_BACKWARDS, QUDA_LINK_FORWARDS};
      for (int link_dir = 0; link_dir < 2; link_dir++) {
        if (!(link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == directions[link_dir])) continue;

        void *send_d[2 * QUDA_MAX_DIM] = {};
        void *recv_d[2 * QUDA_MAX_DIM] = {};

        size_t offset = 0;
        for (int d = 0; d < nDim; d++) {
          recv_d[d] = static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + offset;
          if (bidir) offset += ghost_face_bytes_aligned[d];
          send_d[d] = static_cast<char *>(ghost_send_buffer_d[bufferIndex]) + offset;
          offset += ghost_face_bytes_aligned[d];
        }

        extractGaugeGhost(*this, send_d, true, link_dir * nDim); // get the links into contiguous buffers
        qudaDeviceSynchronize(); // synchronize before issuing mem copies in different streams - could replace with event post and wait

        // issue receive preposts and host-to-device copies if needed
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim)) continue;
          recvStart(dim, dir); // prepost the receive
          if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled()) {
            qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
                            ghost_face_bytes[dim], qudaMemcpyDeviceToHost, device::get_stream(2 * dim + dir));
          }
        }

        // if gdr enabled then synchronize
        if (comm_gdr_enabled()) qudaDeviceSynchronize();

        // if the sending direction is not peer-to-peer then we need to synchronize before we start sending
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim)) continue;
          if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled())
            qudaStreamSynchronize(device::get_stream(2 * dim + dir));
          sendStart(dim, dir, device::get_stream(2 * dim + dir)); // start sending
        }

        // complete communication and issue host-to-device copies if needed
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim)) continue;
          commsComplete(dim, dir);
          if (!comm_peer2peer_enabled(1 - dir, dim) && !comm_gdr_enabled()) {
            qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][1 - dir],
                            from_face_dim_dir_h[bufferIndex][dim][1 - dir], ghost_face_bytes[dim],
                            qudaMemcpyHostToDevice, device::get_stream(2 * dim + dir));
          }
        }

        qudaDeviceSynchronize(); // synchronize before issuing kernels / copies in default stream - could replace with event post and wait

        // fill in the halos for non-partitioned dimensions
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim) && no_comms_fill) {
            qudaMemcpy(recv_d[dim], send_d[dim], ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
          }
        }

        if (isNative()) {
          copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, 0, recv_d, 1 + 2 * link_dir); // 1, 3
        } else {
          // copy from receive buffer into ghost array
          for (int dim = 0; dim < nDim; dim++)
            qudaMemcpy(ghost[dim + link_dir * nDim].data(), recv_d[dim], ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
        }

        bufferIndex = 1 - bufferIndex;
      } // link_dir

      qudaDeviceSynchronize();
    } else { // cpu field
      void *send[2 * QUDA_MAX_DIM];
      for (int d = 0; d < nDim; d++) {
        send[d] = safe_malloc(nFace * surface[d] * nInternal * precision);
        if (geometry == QUDA_COARSE_GEOMETRY) send[d + 4] = safe_malloc(nFace * surface[d] * nInternal * precision);
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
        exchange(ghost_ + nDim, send + nDim, QUDA_FORWARDS);
      }

      for (int d = 0; d < geometry; d++) host_free(send[d]);
    }
  }

  // This does the opposite of exchangeGhost and sends back the ghost
  // zone to the node from which it came and injects it back into the
  // field
  void GaugeField::injectGhost(QudaLinkDirection link_direction)
  {
    if (ghostExchange != QUDA_GHOST_EXCHANGE_PAD)
      errorQuda("Cannot call exchangeGhost with ghostExchange=%d", ghostExchange);
    if (geometry != QUDA_VECTOR_GEOMETRY && geometry != QUDA_COARSE_GEOMETRY)
      errorQuda("Invalid geometry=%d", geometry);
    if (link_direction != QUDA_LINK_BACKWARDS) errorQuda("Invalid link_direction = %d", link_direction);
    if (nFace == 0) errorQuda("nFace = 0");

    if (location == QUDA_CUDA_FIELD_LOCATION) {
      const int dir = 0; // sending backwards only
      const lat_dim_t R = {nFace, nFace, nFace, nFace};
      const bool no_comms_fill = false; // injection never does no_comms_fill
      const bool bidir = false;         // communication is only ever done in one direction at once
      createComms(R, true, bidir); // always need to allocate space for non-partitioned dimension for copyGenericGauge

      // loop over backwards and forwards links (forwards links never sent but leave here just in case)
      const QudaLinkDirection directions[] = {QUDA_LINK_BACKWARDS, QUDA_LINK_FORWARDS};
      for (int link_dir = 0; link_dir < 2; link_dir++) {
        if (!(link_direction == QUDA_LINK_BIDIRECTIONAL || link_direction == directions[link_dir])) continue;

        void *send_d[2 * QUDA_MAX_DIM] = {};
        void *recv_d[2 * QUDA_MAX_DIM] = {};

        size_t offset = 0;
        for (int d = 0; d < nDim; d++) {
          // send backwards is first half of each ghost_send_buffer
          send_d[d] = static_cast<char *>(ghost_send_buffer_d[bufferIndex]) + offset;
          if (bidir) offset += ghost_face_bytes_aligned[d];
          // receive from forwards is the second half of each ghost_recv_buffer
          recv_d[d] = static_cast<char *>(ghost_recv_buffer_d[bufferIndex]) + offset;
          offset += ghost_face_bytes_aligned[d];
        }

        if (isNative()) { // copy from padded region in gauge field into send buffer
          copyGenericGauge(*this, *this, QUDA_CUDA_FIELD_LOCATION, 0, 0, send_d, 0, 1 + 2 * link_dir);
        } else { // copy from receive buffer into ghost array
          for (int dim = 0; dim < nDim; dim++)
            qudaMemcpy(send_d[dim], ghost[dim + link_dir * nDim].data(), ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
        }
        qudaDeviceSynchronize(); // need to synchronize before issueing copies in different streams - could replace with event post and wait

        // issue receive preposts and host-to-device copies if needed
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim)) continue;
          recvStart(dim, dir); // prepost the receive
          if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled()) {
            qudaMemcpyAsync(my_face_dim_dir_h[bufferIndex][dim][dir], my_face_dim_dir_d[bufferIndex][dim][dir],
                            ghost_face_bytes[dim], qudaMemcpyDeviceToHost, device::get_stream(2 * dim + dir));
          }
        }

        // if gdr enabled then synchronize
        if (comm_gdr_enabled()) qudaDeviceSynchronize();

        // if the sending direction is not peer-to-peer then we need to synchronize before we start sending
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim)) continue;
          if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled())
            qudaStreamSynchronize(device::get_stream(2 * dim + dir));
          sendStart(dim, dir, device::get_stream(2 * dim + dir)); // start sending
        }

        // complete communication and issue host-to-device copies if needed
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim)) continue;
          commsComplete(dim, dir);
          if (!comm_peer2peer_enabled(1 - dir, dim) && !comm_gdr_enabled()) {
            qudaMemcpyAsync(from_face_dim_dir_d[bufferIndex][dim][1 - dir],
                            from_face_dim_dir_h[bufferIndex][dim][1 - dir], ghost_face_bytes[dim],
                            qudaMemcpyHostToDevice, device::get_stream(2 * dim + dir));
          }
        }

        qudaDeviceSynchronize(); // synchronize before issuing kernel / copies in default stream - could replace with event post and wait

        // fill in the halos for non-partitioned dimensions
        for (int dim = 0; dim < nDim; dim++) {
          if (!comm_dim_partitioned(dim) && no_comms_fill) {
            qudaMemcpy(recv_d[dim], send_d[dim], ghost_face_bytes[dim], qudaMemcpyDeviceToDevice);
          }
        }

        // get the links into contiguous buffers
        extractGaugeGhost(*this, recv_d, false, link_dir * nDim);

        bufferIndex = 1 - bufferIndex;
      } // link_dir

      qudaDeviceSynchronize();
    } else {
      void *recv[QUDA_MAX_DIM];
      for (int d = 0; d < nDim; d++) recv[d] = safe_malloc(nFace * surface[d] * nInternal * precision);

      void *ghost_[] = {ghost[0].data(), ghost[1].data(), ghost[2].data(), ghost[3].data(),
                        ghost[4].data(), ghost[5].data(), ghost[6].data(), ghost[7].data()};

      // communicate between nodes
      exchange(recv, ghost_, QUDA_BACKWARDS);

      // get the links into contiguous buffers
      extractGaugeGhost(*this, recv, false);

      for (int d = 0; d < QUDA_MAX_DIM; d++) host_free(recv[d]);
    }
  }

  void GaugeField::exchangeExtendedGhost(const lat_dim_t &R, bool no_comms_fill)
  {
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      const int b = bufferIndex;
      void *send_d[QUDA_MAX_DIM], *recv_d[QUDA_MAX_DIM];

      createComms(R, no_comms_fill);

      size_t offset = 0;
      for (int dim = 0; dim < nDim; dim++) {
        if (!(comm_dim_partitioned(dim) || (no_comms_fill && R[dim]))) continue;
        send_d[dim] = static_cast<char *>(ghost_send_buffer_d[b]) + offset;
        recv_d[dim] = static_cast<char *>(ghost_recv_buffer_d[b]) + offset;

        // silence cuda-memcheck initcheck errors that arise since we
        // have an oversized ghost buffer when doing the extended exchange
        qudaMemsetAsync(send_d[dim], 0, 2 * ghost_face_bytes_aligned[dim], device::get_default_stream());
        offset += 2 * ghost_face_bytes_aligned[dim]; // factor of two from fwd/back
      }

      for (int dim = 0; dim < nDim; dim++) {
        if (!(comm_dim_partitioned(dim) || (no_comms_fill && R[dim]))) continue;

        // extract into a contiguous buffer
        extractExtendedGaugeGhost(*this, dim, R, send_d, true);

        if (comm_dim_partitioned(dim)) {
          qudaDeviceSynchronize(); // synchronize before issuing mem copies in different streams - could replace with event post and wait

          for (int dir = 0; dir < 2; dir++) recvStart(dim, dir);

          for (int dir = 0; dir < 2; dir++) {
            // issue host-to-device copies if needed
            if (!comm_peer2peer_enabled(dir, dim) && !comm_gdr_enabled()) {
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

      bufferIndex = 1 - bufferIndex;
      qudaDeviceSynchronize();
    } else {
      void *send[QUDA_MAX_DIM];
      void *recv[QUDA_MAX_DIM];
      size_t bytes[QUDA_MAX_DIM];
      // store both parities and directions in each
      for (int d = 0; d < nDim; d++) {
        if (!(comm_dim_partitioned(d) || (no_comms_fill && R[d]))) continue;
        bytes[d] = surface[d] * R[d] * geometry * nInternal * precision;
        send[d] = safe_malloc(2 * bytes[d]);
        recv[d] = safe_malloc(2 * bytes[d]);
      }

      for (int d = 0; d < nDim; d++) {
        if (!(comm_dim_partitioned(d) || (no_comms_fill && R[d]))) continue;
        // extract into a contiguous buffer
        extractExtendedGaugeGhost(*this, d, R, send, true);

        if (comm_dim_partitioned(d)) {
          // do the exchange
          MsgHandle *mh_recv_back;
          MsgHandle *mh_recv_fwd;
          MsgHandle *mh_send_fwd;
          MsgHandle *mh_send_back;

          mh_recv_back = comm_declare_receive_relative(recv[d], d, -1, bytes[d]);
          mh_recv_fwd = comm_declare_receive_relative(((char *)recv[d]) + bytes[d], d, +1, bytes[d]);
          mh_send_back = comm_declare_send_relative(send[d], d, -1, bytes[d]);
          mh_send_fwd = comm_declare_send_relative(((char *)send[d]) + bytes[d], d, +1, bytes[d]);

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
          memcpy(static_cast<char *>(recv[d]) + bytes[d], send[d], bytes[d]);
          memcpy(recv[d], static_cast<char *>(send[d]) + bytes[d], bytes[d]);
        }

        // inject back into the gauge field
        extractExtendedGaugeGhost(*this, d, R, recv, false);
      }

      for (int d = 0; d < nDim; d++) {
        if (!(comm_dim_partitioned(d) || (no_comms_fill && R[d]))) continue;
        host_free(send[d]);
        host_free(recv[d]);
      }
    }
  }

  void GaugeField::exchangeExtendedGhost(const lat_dim_t &R, TimeProfile &profile, bool no_comms_fill)
  {
    profile.TPSTART(QUDA_PROFILE_COMMS);
    exchangeExtendedGhost(R, no_comms_fill);
    profile.TPSTOP(QUDA_PROFILE_COMMS);
  }

  void GaugeField::exchange(void **ghost_link, void **link_sendbuf, QudaDirection dir) const
  {
    MsgHandle *mh_send[4];
    MsgHandle *mh_recv[4];
    size_t bytes[4];

    for (int i=0; i<nDimComms; i++) bytes[i] = 2*nFace*surfaceCB[i]*nInternal*precision;

    // in general (standard ghost exchange) we always do the exchange
    // even if a dimension isn't partitioned.  However, this breaks
    // GaugeField::injectGhost(), so when transferring backwards we
    // only exchange if a dimension is partitioned.  FIXME: this
    // should probably be cleaned up.
    bool no_comms_fill = (dir == QUDA_BACKWARDS) ? false : true;

    void *send[4];
    void *receive[4];
    if (Location() == QUDA_CPU_FIELD_LOCATION) {
      for (int i=0; i<nDimComms; i++) {
	if (comm_dim_partitioned(i)) {
	  send[i] = link_sendbuf[i];
	  receive[i] = ghost_link[i];
	} else {
	  if (no_comms_fill) memcpy(ghost_link[i], link_sendbuf[i], bytes[i]);
	}
      }
    } else {
      errorQuda("Not supported");
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      if (dir == QUDA_FORWARDS) {
	mh_send[i] = comm_declare_send_relative(send[i], i, +1, bytes[i]);
	mh_recv[i] = comm_declare_receive_relative(receive[i], i, -1, bytes[i]);
      } else if (dir == QUDA_BACKWARDS) {
	mh_send[i] = comm_declare_send_relative(send[i], i, -1, bytes[i]);
	mh_recv[i] = comm_declare_receive_relative(receive[i], i, +1, bytes[i]);
      } else {
	errorQuda("Unsuported dir=%d", dir);
      }

    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_start(mh_send[i]);
      comm_start(mh_recv[i]);
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_wait(mh_send[i]);
      comm_wait(mh_recv[i]);
    }

    for (int i=0; i<nDimComms; i++) {
      if (!comm_dim_partitioned(i)) continue;
      comm_free(mh_send[i]);
      comm_free(mh_recv[i]);
    }
  }

  bool GaugeField::are_compatible_weak(const GaugeField &a, const GaugeField &b)
  {
    return (a.LinkType() == b.LinkType() && a.Ncolor() == b.Ncolor() && a.Nface() == b.Nface()
            && a.GaugeFixed() == b.GaugeFixed() && a.TBoundary() == b.TBoundary() && a.Anisotropy() == b.Anisotropy()
            && a.Tadpole() == b.Tadpole());
  }

  bool GaugeField::are_compatible(const GaugeField &a, const GaugeField &b)
  {
    return (a.Precision() == b.Precision() && a.Order() == b.Order() && are_compatible_weak(a, b));
  }

  void GaugeField::checkField(const LatticeField &l) const {
    LatticeField::checkField(l);
    try {
      const GaugeField &g = dynamic_cast<const GaugeField&>(l);
      if (g.link_type != link_type) errorQuda("link_type does not match %d %d", link_type, g.link_type);
      if (g.nColor != nColor) errorQuda("nColor does not match %d %d", nColor, g.nColor);
      if (g.nFace != nFace) errorQuda("nFace does not match %d %d", nFace, g.nFace);
      if (g.fixed != fixed) errorQuda("fixed does not match %d %d", fixed, g.fixed);
      if (g.t_boundary != t_boundary) errorQuda("t_boundary does not match %d %d", t_boundary, g.t_boundary);
      if (g.anisotropy != anisotropy) errorQuda("anisotropy does not match %e %e", anisotropy, g.anisotropy);
      if (g.tadpole != tadpole) errorQuda("tadpole does not match %e %e", tadpole, g.tadpole);
    }
    catch(std::bad_cast &e) {
      errorQuda("Failed to cast reference to GaugeField");
    }
  }

  void *create_gauge_buffer(size_t bytes, QudaGaugeFieldOrder order, QudaFieldGeometry geometry)
  {
    if (order == QUDA_QDP_GAUGE_ORDER) {
      void **buffer = new void *[geometry];
      for (int d = 0; d < geometry; d++) buffer[d] = pool_device_malloc(bytes / geometry);
      return ((void *)buffer);
    } else {
      return pool_device_malloc(bytes);
    }
  }

  void **create_ghost_buffer(size_t bytes[], QudaGaugeFieldOrder order, QudaFieldGeometry geometry)
  {
    if (order > 4) {
      void **buffer = new void *[geometry];
      for (int d = 0; d < geometry; d++) buffer[d] = pool_device_malloc(bytes[d]);
      return buffer;
    } else {
      return 0;
    }
  }

  void free_gauge_buffer(void *buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry)
  {
    if (order == QUDA_QDP_GAUGE_ORDER) {
      for (int d = 0; d < geometry; d++) pool_device_free(((void **)buffer)[d]);
      delete[]((void **)buffer);
    } else {
      pool_device_free(buffer);
    }
  }

  void free_ghost_buffer(void **buffer, QudaGaugeFieldOrder order, QudaFieldGeometry geometry)
  {
    if (order > 4) {
      for (int d = 0; d < geometry; d++) pool_device_free(buffer[d]);
      delete[] buffer;
    }
  }

  void GaugeField::copy(const GaugeField &src)
  {
    if (this == &src) return;

    if (src.Location() == QUDA_CUDA_FIELD_LOCATION && location == QUDA_CPU_FIELD_LOCATION) {
      getProfile().TPSTART(QUDA_PROFILE_D2H);
    } else if (src.Location() == QUDA_CPU_FIELD_LOCATION && location == QUDA_CUDA_FIELD_LOCATION) {
      getProfile().TPSTART(QUDA_PROFILE_H2D);
    }

    checkField(src);

    if (link_type == QUDA_ASQTAD_FAT_LINKS) {
      fat_link_max = src.LinkMax();
      if (fat_link_max == 0.0 && precision < QUDA_SINGLE_PRECISION) fat_link_max = src.abs_max();
    } else {
      fat_link_max = 1.0;
    }

    if (src.Location() == QUDA_CUDA_FIELD_LOCATION) {

      if (location == QUDA_CUDA_FIELD_LOCATION) {
        if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
          // copy field and ghost zone into this field
          copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION);

          if (geometry == QUDA_COARSE_GEOMETRY)
            copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, nullptr, nullptr, nullptr, 3);
        } else {
          copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, nullptr);
          if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
        }
      } else { // CPU location
        if (reorder_location() == QUDA_CPU_FIELD_LOCATION) {

          if (!src.isNative()) errorQuda("Only native order is supported");
          void *buffer = pool_pinned_malloc(src.Bytes());
          qudaMemcpy(buffer, src.data(), src.Bytes(), qudaMemcpyDeviceToHost);

          if (GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
            copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, nullptr, buffer);
          } else {
            copyExtendedGauge(*this, src, QUDA_CPU_FIELD_LOCATION, nullptr, buffer);
          }
          pool_pinned_free(buffer);

        } else { // else reorder on the GPU

          if (order == QUDA_MILC_SITE_GAUGE_ORDER || order == QUDA_BQCD_GAUGE_ORDER
              || order == QUDA_TIFR_PADDED_GAUGE_ORDER) {
            // special case where we use zero-copy memory to read/write directly from application's array
            void *data_d = get_mapped_device_pointer(data());
            if (GhostExchange() == QUDA_GHOST_EXCHANGE_NO) {
              copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, data_d, nullptr);
            } else {
              errorQuda("Ghost copy not supported here");
            }
            qudaDeviceSynchronize(); // synchronize to ensure visibility on the host
          } else {
            void *buffer = create_gauge_buffer(bytes, order, geometry);
            size_t ghost_bytes[8];
            int dstNinternal = reconstruct != QUDA_RECONSTRUCT_NO ? reconstruct : 2 * nColor * nColor;
            for (int d = 0; d < geometry; d++) ghost_bytes[d] = nFace * surface[d % 4] * dstNinternal * precision;
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
              for (int d = 0; d < geometry; d++) {
                qudaMemcpy(gauge_array[d].data(), ((void **)buffer)[d], bytes / geometry, qudaMemcpyDeviceToHost);
              }
            } else {
              qudaMemcpy(gauge.data(), buffer, bytes, qudaMemcpyDeviceToHost);
            }

            if (order > 4 && ghostExchange == QUDA_GHOST_EXCHANGE_PAD && src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD
                && nFace)
              for (int d = 0; d < geometry; d++)
                qudaMemcpy(Ghost()[d].data(), ghost_buffer[d], ghost_bytes[d], qudaMemcpyDeviceToHost);

            free_gauge_buffer(buffer, order, geometry);
            if (nFace > 0) free_ghost_buffer(ghost_buffer, order, geometry);
          } // order
        }
      }

    } else if (src.Location() == QUDA_CPU_FIELD_LOCATION) {

      if (location == QUDA_CPU_FIELD_LOCATION) {
        // copy field and ghost zone directly
        copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION);
      } else {
        if (reorder_location() == QUDA_CPU_FIELD_LOCATION) { // do reorder on the CPU
          void *buffer = pool_pinned_malloc(bytes);

          if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
            // copy field and ghost zone into buffer
            copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, nullptr);

            if (geometry == QUDA_COARSE_GEOMETRY)
              copyGenericGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, nullptr, 0, 0, 3);
          } else {
            copyExtendedGauge(*this, src, QUDA_CPU_FIELD_LOCATION, buffer, nullptr);
            if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
          }

          qudaMemcpy(gauge.data(), buffer, bytes, qudaMemcpyDefault);
          pool_pinned_free(buffer);
        } else { // else on the GPU

          if (src.Order() == QUDA_MILC_SITE_GAUGE_ORDER || src.Order() == QUDA_BQCD_GAUGE_ORDER
              || src.Order() == QUDA_TIFR_PADDED_GAUGE_ORDER) {
            // special case where we use zero-copy memory to read/write directly from application's array
            void *src_d = get_mapped_device_pointer(src.data());

            if (src.GhostExchange() == QUDA_GHOST_EXCHANGE_NO) {
              copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, data(), src_d);
            } else {
              errorQuda("Ghost copy not supported here");
            }

          } else {
            void *buffer = create_gauge_buffer(src.Bytes(), src.Order(), src.Geometry());
            size_t ghost_bytes[8];
            int srcNinternal = src.Reconstruct() != QUDA_RECONSTRUCT_NO ? src.Reconstruct() : 2 * nColor * nColor;
            for (int d = 0; d < geometry; d++) ghost_bytes[d] = nFace * surface[d % 4] * srcNinternal * src.Precision();
            void **ghost_buffer = (nFace > 0) ? create_ghost_buffer(ghost_bytes, src.Order(), geometry) : nullptr;

            if (src.Order() == QUDA_QDP_GAUGE_ORDER) {
              for (int d = 0; d < geometry; d++) {
                qudaMemcpy(((void **)buffer)[d], src.data(d), src.Bytes() / geometry, qudaMemcpyDefault);
              }
            } else {
              qudaMemcpy(buffer, src.data(), src.Bytes(), qudaMemcpyDefault);
            }

            if (src.Order() > 4 && GhostExchange() == QUDA_GHOST_EXCHANGE_PAD
                && src.GhostExchange() == QUDA_GHOST_EXCHANGE_PAD && nFace)
              for (int d = 0; d < geometry; d++)
                qudaMemcpy(ghost_buffer[d], src.Ghost()[d].data(), ghost_bytes[d], qudaMemcpyDefault);

            if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && src.GhostExchange() != QUDA_GHOST_EXCHANGE_EXTENDED) {
              copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, buffer, nullptr, ghost_buffer);
              if (geometry == QUDA_COARSE_GEOMETRY)
                copyGenericGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, buffer, nullptr, ghost_buffer, 3);
            } else {
              copyExtendedGauge(*this, src, QUDA_CUDA_FIELD_LOCATION, nullptr, buffer);
              if (geometry == QUDA_COARSE_GEOMETRY) errorQuda("Extended gauge copy for coarse geometry not supported");
            }
            free_gauge_buffer(buffer, src.Order(), src.Geometry());
            if (nFace > 0) free_ghost_buffer(ghost_buffer, src.Order(), geometry);
          }
        } // reorder_location
      }   // this location
    } else {
      errorQuda("Invalid gauge field type");
    }

    // if we have copied from a source without a pad then we need to exchange
    if (ghostExchange == QUDA_GHOST_EXCHANGE_PAD && src.GhostExchange() != QUDA_GHOST_EXCHANGE_PAD)
      exchangeGhost(geometry == QUDA_VECTOR_GEOMETRY ? QUDA_LINK_BACKWARDS : QUDA_LINK_BIDIRECTIONAL);

    staggeredPhaseApplied = src.StaggeredPhaseApplied();
    staggeredPhaseType = src.StaggeredPhase();

    if (src.Location() == QUDA_CUDA_FIELD_LOCATION && location == QUDA_CPU_FIELD_LOCATION) {
      getProfile().TPSTOP(QUDA_PROFILE_D2H);
    } else if (src.Location() == QUDA_CPU_FIELD_LOCATION && location == QUDA_CUDA_FIELD_LOCATION) {
      getProfile().TPSTOP(QUDA_PROFILE_H2D);
    }
  }

  std::ostream &operator<<(std::ostream &output, const GaugeFieldParam &param)
  {
    output << static_cast<const LatticeFieldParam &>(param);
    output << "nColor = " << param.nColor << std::endl;
    output << "nFace = " << param.nFace << std::endl;
    output << "reconstruct = " << param.reconstruct << std::endl;
    int nInternal = (param.reconstruct != QUDA_RECONSTRUCT_NO ? param.reconstruct : param.nColor * param.nColor * 2);
    output << "nInternal = " << nInternal << std::endl;
    output << "order = " << param.order << std::endl;
    output << "fixed = " << param.fixed << std::endl;
    output << "link_type = " << param.link_type << std::endl;
    output << "t_boundary = " << param.t_boundary << std::endl;
    output << "anisotropy = " << param.anisotropy << std::endl;
    output << "tadpole = " << param.tadpole << std::endl;
    output << "create = " << param.create << std::endl;
    output << "geometry = " << param.geometry << std::endl;
    output << "staggeredPhaseType = " << param.staggeredPhaseType << std::endl;
    output << "staggeredPhaseApplied = " << param.staggeredPhaseApplied << std::endl;

    return output;  // for multiple << operators.
  }

  std::ostream &operator<<(std::ostream &output, const GaugeField &field)
  {
    output << static_cast<const LatticeField &>(field);
    output << "init = " << field.init << std::endl;
    output << "gauge = " << field.gauge << std::endl;
    output << "gauge_array = " << field.gauge_array << std::endl;
    output << "bytes = " << field.bytes << std::endl;
    output << "phase_offset = " << field.phase_offset << std::endl;
    output << "phase_bytes = " << field.phase_bytes << std::endl;
    output << "length = " << field.length << std::endl;
    output << "real_length = " << field.real_length << std::endl;
    output << "nColor = " << field.nColor << std::endl;
    output << "nFace = " << field.nFace << std::endl;
    output << "geometry = " << field.geometry << std::endl;
    output << "site_dim = " << field.geometry << std::endl;
    output << "reconstruct = " << field.reconstruct << std::endl;
    output << "nInternal = " << field.nInternal << std::endl;
    output << "order = " << field.order << std::endl;
    output << "fixed = " << field.fixed << std::endl;
    output << "link_type = " << field.link_type << std::endl;
    output << "t_boundary = " << field.t_boundary << std::endl;
    output << "anisotropy = " << field.anisotropy << std::endl;
    output << "tadpole = " << field.tadpole << std::endl;
    output << "fat_link_max = " << field.fat_link_max << std::endl;
    output << "ghost = " << field.ghost << std::endl;
    output << "ghostFace = " << field.ghostFace << std::endl;
    output << "staggeredPhaseType = " << field.staggeredPhaseType << std::endl;
    output << "staggeredPhaseApplied = " << field.staggeredPhaseApplied << std::endl;
    output << "i_mu = " << field.i_mu << std::endl;
    output << "site_offset = " << field.site_offset << std::endl;
    output << "size_size = " << field.site_size << std::endl;
    return output; // for multiple << operators.
  }

  void GaugeField::zero()
  {
    if (order != QUDA_QDP_GAUGE_ORDER) {
      qudaMemset(gauge, 0, bytes);
    } else {
      for (int g = 0; g < geometry; g++) qudaMemset(gauge_array[g], 0, volume * nInternal * precision);
    }
  }

  ColorSpinorParam colorSpinorParam(const GaugeField &a) {
   if (a.FieldOrder() == QUDA_QDP_GAUGE_ORDER || a.FieldOrder() == QUDA_QDPJIT_GAUGE_ORDER)
     errorQuda("Not implemented for this order %d", a.FieldOrder());

    if (a.LinkType() == QUDA_COARSE_LINKS) errorQuda("Not implemented for coarse-link type");
    if (a.Ncolor() != 3) errorQuda("Not implemented for Ncolor = %d", a.Ncolor());

    if (a.Precision() == QUDA_HALF_PRECISION || a.Precision() == QUDA_QUARTER_PRECISION)
      errorQuda("Casting a GaugeField into ColorSpinorField not possible in half or quarter precision");

    ColorSpinorParam spinor_param;
    spinor_param.nColor = (a.Geometry()*a.Reconstruct())/2;
    spinor_param.nSpin = 1;
    spinor_param.nDim = a.Ndim();
    spinor_param.pc_type = QUDA_4D_PC;
    for (int d=0; d<a.Ndim(); d++) spinor_param.x[d] = a.X()[d];
    spinor_param.pad = a.Pad();
    spinor_param.siteSubset = QUDA_FULL_SITE_SUBSET;
    spinor_param.siteOrder = QUDA_EVEN_ODD_SITE_ORDER;
    spinor_param.setPrecision(a.Precision(), a.Precision(), true);
    spinor_param.gammaBasis = QUDA_UKQCD_GAMMA_BASIS;
    spinor_param.create = QUDA_REFERENCE_FIELD_CREATE;
    spinor_param.v = a.data();
    spinor_param.location = a.Location();
    return spinor_param;
  }

  // Return the L2 norm squared of the gauge field
  double norm2(const GaugeField &a)
  {
    ColorSpinorField b(colorSpinorParam(a));
    return blas::norm2(b);
  }

  // Return the L1 norm of the gauge field
  double norm1(const GaugeField &a)
  {
    ColorSpinorField b(colorSpinorParam(a));
    return blas::norm1(b);
  }

  // Scale the gauge field by the constant a
  void ax(const double &a, GaugeField &u)
  {
    ColorSpinorField b(colorSpinorParam(u));
    blas::ax(a, b);
  }

  uint64_t GaugeField::checksum(bool mini) const {
    return Checksum(*this, mini);
  }

  GaugeField *GaugeField::Create(const GaugeFieldParam &param) { return new GaugeField(param); }

  GaugeField GaugeField::create_alias(const GaugeFieldParam &param_)
  {
    if (param_.init && param_.Precision() > precision)
      errorQuda("Cannot create an alias to source with lower precision than the alias");
    GaugeFieldParam param = param_.init ? param_ : GaugeFieldParam(*this);
    param.create = QUDA_REFERENCE_FIELD_CREATE;
    param.gauge = gauge.data();
    return GaugeField(param);
  }

  // helper for creating extended gauge fields
  GaugeField *createExtendedGauge(const GaugeField &in, const lat_dim_t &R, TimeProfile &profile, bool redundant_comms,
                                  QudaReconstructType recon)
  {
    GaugeFieldParam gParamEx(in);
    gParamEx.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    gParamEx.pad = 0;
    gParamEx.nFace = 1;
    for (int d = 0; d < 4; d++) {
      gParamEx.x[d] += 2 * R[d];
      gParamEx.r[d] = R[d];
    }
    if (recon != QUDA_RECONSTRUCT_INVALID && recon != in.Reconstruct()) {
      gParamEx.reconstruct = recon;
      gParamEx.setPrecision(gParamEx.Precision());
    }

    auto *out = new GaugeField(gParamEx);

    // copy input field into the extended device gauge field
    copyExtendedGauge(*out, in, in.Location());

    // now fill up the halos
    out->exchangeExtendedGhost(R, profile, redundant_comms);

    return out;
  }

  // helper for creating extended (cpu) gauge fields
  GaugeField *createExtendedGauge(void **gauge, QudaGaugeParam &gauge_param, const lat_dim_t &R)
  {
    GaugeFieldParam gauge_field_param(gauge_param, gauge);
    GaugeField cpu(gauge_field_param);

    gauge_field_param.location = QUDA_CPU_FIELD_LOCATION;
    gauge_field_param.ghostExchange = QUDA_GHOST_EXCHANGE_EXTENDED;
    gauge_field_param.create = QUDA_ZERO_FIELD_CREATE;
    for (int d = 0; d < 4; d++) {
      gauge_field_param.x[d] += 2 * R[d];
      gauge_field_param.r[d] = R[d];
    }
    GaugeField *padded_cpu = new GaugeField(gauge_field_param);

    copyExtendedGauge(*padded_cpu, cpu, QUDA_CPU_FIELD_LOCATION);
    padded_cpu->exchangeExtendedGhost(R, true); // Do comm to fill halo = true

    return padded_cpu;
  }

  void GaugeField::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    if (location == QUDA_CUDA_FIELD_LOCATION && is_prefetch_enabled() && mem_type == QUDA_MEMORY_DEVICE) {
      if (gauge.data()) qudaMemPrefetchAsync(gauge.data(), bytes, mem_space, stream);
      if (!isNative()) {
        for (int i = 0; i < nDim; i++) {
          size_t nbytes = nFace * surface[i] * nInternal * precision;
          if (ghost[i].data() && nbytes) qudaMemPrefetchAsync(ghost[i].data(), nbytes, mem_space, stream);
          if (ghost[i + 4].data() && nbytes && geometry == QUDA_COARSE_GEOMETRY)
            qudaMemPrefetchAsync(ghost[i + 4].data(), nbytes, mem_space, stream);
        }
      }
    }
  }

  void GaugeField::backup() const
  {
    if (backup_h.size()) errorQuda("Gauge field already backed up");

    if (order == QUDA_QDP_GAUGE_ORDER) {
      backup_h.resize(geometry);
      for (int d = 0; d < geometry; d++) {
        backup_h[d] = quda_ptr(QUDA_MEMORY_HOST, bytes / geometry);
        qudaMemcpy(backup_h[d], gauge_array[d], bytes / geometry, qudaMemcpyDefault);
      }
    } else {
      backup_h.resize(1);
      backup_h[0] = quda_ptr(QUDA_MEMORY_HOST, bytes);
      qudaMemcpy(backup_h[0], gauge, bytes, qudaMemcpyDefault);
    }
  }

  void GaugeField::restore() const
  {
    if (!backup_h.size()) errorQuda("Cannot restore since not backed up");

    if (order == QUDA_QDP_GAUGE_ORDER) {
      for (int d = 0; d < geometry; d++) {
        qudaMemcpy(gauge_array[d], backup_h[d], bytes / geometry, qudaMemcpyDefault);
      }
    } else {
      qudaMemcpy(gauge, backup_h[0], bytes, qudaMemcpyDefault);
    }

    backup_h.resize(0);
  }

  void GaugeField::copy_to_buffer(void *buffer) const
  {
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(buffer, data(), Bytes(), qudaMemcpyDeviceToHost);
    } else {
      if (is_pointer_array(order)) {
        char *dst_buffer = static_cast<char *>(buffer);
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
  }

  void GaugeField::copy_from_buffer(void *buffer)
  {
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      qudaMemcpy(data(), buffer, Bytes(), qudaMemcpyHostToDevice);
    } else {
      if (is_pointer_array(order)) {
        const char *dst_buffer = static_cast<const char *>(buffer);
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
  }

  void GaugeField::PrintMatrix(int dim, int parity, unsigned int x_cb, int rank) const
  {
    genericPrintMatrix(*this, dim, parity, x_cb, rank);
  }

} // namespace quda
