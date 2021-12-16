#include <typeinfo>
#include <quda_internal.h>
#include <lattice_field.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>

namespace quda {

  bool LatticeField::initIPCComms = false;

  int LatticeField::buffer_send_p2p_fwd[2][QUDA_MAX_DIM] { };
  int LatticeField::buffer_recv_p2p_fwd[2][QUDA_MAX_DIM] { };
  int LatticeField::buffer_send_p2p_back[2][QUDA_MAX_DIM] { };
  int LatticeField::buffer_recv_p2p_back[2][QUDA_MAX_DIM] { };

  MsgHandle* LatticeField::mh_send_p2p_fwd[2][QUDA_MAX_DIM] { };
  MsgHandle* LatticeField::mh_send_p2p_back[2][QUDA_MAX_DIM] { };
  MsgHandle* LatticeField::mh_recv_p2p_fwd[2][QUDA_MAX_DIM] { };
  MsgHandle* LatticeField::mh_recv_p2p_back[2][QUDA_MAX_DIM] { };

  qudaEvent_t LatticeField::ipcCopyEvent[2][2][QUDA_MAX_DIM];
  qudaEvent_t LatticeField::ipcRemoteCopyEvent[2][2][QUDA_MAX_DIM];

  void *LatticeField::ghost_pinned_send_buffer_h[2] = {nullptr, nullptr};
  void *LatticeField::ghost_pinned_send_buffer_hd[2] = {nullptr, nullptr};

  void *LatticeField::ghost_pinned_recv_buffer_h[2] = {nullptr, nullptr};
  void *LatticeField::ghost_pinned_recv_buffer_hd[2] = {nullptr, nullptr};

  // gpu ghost receive buffer
  void *LatticeField::ghost_recv_buffer_d[2] = {nullptr, nullptr};

  // gpu ghost send buffer
  void *LatticeField::ghost_send_buffer_d[2] = {nullptr, nullptr};

  bool LatticeField::ghost_field_reset = false;

  void* LatticeField::ghost_remote_send_buffer_d[2][QUDA_MAX_DIM][2];

  bool LatticeField::initGhostFaceBuffer = false;

  size_t LatticeField::ghostFaceBytes = 0;

  int LatticeField::bufferIndex = 0;

  LatticeFieldParam::LatticeFieldParam(const LatticeField &field)
    : precision(field.Precision()), ghost_precision(field.Precision()),
      nDim(field.Ndim()), pad(field.Pad()),
      siteSubset(field.SiteSubset()), mem_type(field.MemType()),
      ghostExchange(field.GhostExchange()), scale(field.Scale())
  {
    for(int dir=0; dir<nDim; ++dir) {
      x[dir] = field.X()[dir];
      r[dir] = field.R()[dir];
    }
  }

  LatticeField::LatticeField(const LatticeFieldParam &param) :
    volume(1),
    localVolume(1),
    pad(param.pad),
    total_bytes(0),
    nDim(param.nDim),
    precision(param.Precision()),
    ghost_precision(param.GhostPrecision()),
    ghost_precision_reset(false),
    scale(param.scale),
    siteSubset(param.siteSubset),
    ghostExchange(param.ghostExchange),
    ghost_bytes(0),
    ghost_bytes_old(0),
    ghost_face_bytes {},
    ghost_face_bytes_aligned {},
    ghost_offset(),
    my_face_h {},
    my_face_hd {},
    my_face_d {},
    from_face_h {},
    from_face_hd {},
    from_face_d {},
    initComms(false),
    mem_type(param.mem_type),
    backup_h(nullptr),
    backup_norm_h(nullptr),
    backed_up(false)
  {
    precisionCheck();

    for (int dir = 0; dir < 2; dir++) { // XLC cannot do multi-dimensional array initialization
      for (int dim = 0; dim < QUDA_MAX_DIM; dim++) {

        for (int b = 0; b < 2; b++) {
          my_face_dim_dir_d[b][dim][dir] = nullptr;
          my_face_dim_dir_hd[b][dim][dir] = nullptr;
          my_face_dim_dir_h[b][dim][dir] = nullptr;

          from_face_dim_dir_d[b][dim][dir] = nullptr;
          from_face_dim_dir_hd[b][dim][dir] = nullptr;
          from_face_dim_dir_h[b][dim][dir] = nullptr;
        }

        mh_recv_fwd[dir][dim] = nullptr;
        mh_recv_back[dir][dim] = nullptr;
        mh_send_fwd[dir][dim] = nullptr;
        mh_send_back[dir][dim] = nullptr;

        mh_recv_rdma_fwd[dir][dim] = nullptr;
        mh_recv_rdma_back[dir][dim] = nullptr;
        mh_send_rdma_fwd[dir][dim] = nullptr;
        mh_send_rdma_back[dir][dim] = nullptr;
      }
    }

    for (int i=0; i<nDim; i++) {
      x[i] = param.x[i];
      r[i] = ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED ? param.r[i] : 0;
      local_x[i] = x[i] - 2 * r[i];
      volume *= x[i];
      localVolume *= local_x[i];
      surface[i] = 1;
      local_surface[i] = 1;
      for (int j=0; j<nDim; j++) {
	if (i==j) continue;
	surface[i] *= param.x[j];
        local_surface[i] *= param.x[j] - 2 * param.r[j];
      }
    }

    if (siteSubset == QUDA_INVALID_SITE_SUBSET) errorQuda("siteSubset is not set");
    volumeCB = (siteSubset == QUDA_FULL_SITE_SUBSET) ? volume / 2 : volume;
    localVolumeCB = (siteSubset == QUDA_FULL_SITE_SUBSET) ? localVolume / 2 : localVolume;
    stride = volumeCB + pad;

    // for parity fields the factor of half is present for all surfaces dimensions except x, so add it manually
    for (int i = 0; i < nDim; i++) {
      surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET || i==0) ? surface[i] / 2 : surface[i];
      local_surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET || i == 0) ? local_surface[i] / 2 : local_surface[i];
    }

    // for 5-dimensional fields, we only communicate in the space-time dimensions
    nDimComms = nDim == 5 ? 4 : nDim;

    switch (precision) {
    case QUDA_DOUBLE_PRECISION:
    case QUDA_SINGLE_PRECISION:
    case QUDA_HALF_PRECISION:
    case QUDA_QUARTER_PRECISION:
      break;
    default:
      errorQuda("Unknown precision %d", precision);
    }

    setTuningString();
  }

  LatticeField::LatticeField(const LatticeField &field) :
    volume(field.volume),
    volumeCB(field.volumeCB),
    localVolume(field.localVolume),
    localVolumeCB(field.localVolumeCB),
    stride(field.stride),
    pad(field.pad),
    total_bytes(0),
    nDim(field.nDim),
    precision(field.precision),
    ghost_precision(field.ghost_precision),
    ghost_precision_reset(false),
    scale(field.scale),
    siteSubset(field.siteSubset),
    ghostExchange(field.ghostExchange),
    nDimComms(field.nDimComms),
    ghost_bytes(0),
    ghost_bytes_old(0),
    ghost_face_bytes {},
    ghost_face_bytes_aligned {},
    ghost_offset(),
    my_face_h {},
    my_face_hd {},
    my_face_d {},
    from_face_h {},
    from_face_hd {},
    from_face_d {},
    initComms(false),
    mem_type(field.mem_type),
    backup_h(nullptr),
    backup_norm_h(nullptr),
    backed_up(false)
  {
    precisionCheck();

    for (int dir = 0; dir < 2; dir++) { // XLC cannot do multi-dimensional array initialization
      for (int dim = 0; dim < QUDA_MAX_DIM; dim++) {

        for (int b = 0; b < 2; b++) {
          my_face_dim_dir_d[b][dim][dir] = nullptr;
          my_face_dim_dir_hd[b][dim][dir] = nullptr;
          my_face_dim_dir_h[b][dim][dir] = nullptr;

          from_face_dim_dir_d[b][dim][dir] = nullptr;
          from_face_dim_dir_hd[b][dim][dir] = nullptr;
          from_face_dim_dir_h[b][dim][dir] = nullptr;
        }

        mh_recv_fwd[dir][dim] = nullptr;
        mh_recv_back[dir][dim] = nullptr;
        mh_send_fwd[dir][dim] = nullptr;
        mh_send_back[dir][dim] = nullptr;

        mh_recv_rdma_fwd[dir][dim] = nullptr;
        mh_recv_rdma_back[dir][dim] = nullptr;
        mh_send_rdma_fwd[dir][dim] = nullptr;
        mh_send_rdma_back[dir][dim] = nullptr;
      }
    }

    for (int i=0; i<nDim; i++) {
      x[i] = field.x[i];
      r[i] = field.r[i];
      local_x[i] = field.local_x[i];
      surface[i] = field.surface[i];
      surfaceCB[i] = field.surfaceCB[i];
      local_surface[i] = field.local_surface[i];
      local_surfaceCB[i] = field.local_surfaceCB[i];
    }

    setTuningString();
  }

  LatticeField::~LatticeField() { }

  void LatticeField::allocateGhostBuffer(size_t ghost_bytes) const
  {
    // only allocate if not already allocated or buffer required is bigger than previously
    if ( !initGhostFaceBuffer || ghost_bytes > ghostFaceBytes) {

      if (initGhostFaceBuffer) {
        if (ghostFaceBytes) {
          // remove potential for inter-process race conditions
          // ensures that all outstanding communication is complete
          // before we free any comms buffers
          qudaDeviceSynchronize();
          comm_barrier();
          for (int b=0; b<2; b++) {
            device_comms_pinned_free(ghost_recv_buffer_d[b]);
            device_comms_pinned_free(ghost_send_buffer_d[b]);
            host_free(ghost_pinned_send_buffer_h[b]);
            host_free(ghost_pinned_recv_buffer_h[b]);
          }
        }
      }

      if (ghost_bytes > 0) {
        for (int b = 0; b < 2; ++b) {
          // gpu receive buffer (use pinned allocator to avoid this being redirected, e.g., by QDPJIT)
          ghost_recv_buffer_d[b] = device_comms_pinned_malloc(ghost_bytes);
          // silence any false cuda-memcheck initcheck errors
          qudaMemset(ghost_recv_buffer_d[b], 0, ghost_bytes);

          // gpu send buffer (use pinned allocator to avoid this being redirected, e.g., by QDPJIT)
          ghost_send_buffer_d[b] = device_comms_pinned_malloc(ghost_bytes);
          // silence any false cuda-memcheck initcheck errors
          qudaMemset(ghost_send_buffer_d[b], 0, ghost_bytes);

          // pinned buffer used for sending
          ghost_pinned_send_buffer_h[b] = mapped_malloc(ghost_bytes);

          // set the matching device-mapped pointer
          ghost_pinned_send_buffer_hd[b] = get_mapped_device_pointer(ghost_pinned_send_buffer_h[b]);

          // pinned buffer used for receiving
          ghost_pinned_recv_buffer_h[b] = mapped_malloc(ghost_bytes);

          // set the matching device-mapped pointer
          ghost_pinned_recv_buffer_hd[b] = get_mapped_device_pointer(ghost_pinned_recv_buffer_h[b]);
        }

        initGhostFaceBuffer = true;
	ghostFaceBytes = ghost_bytes;
      }

      LatticeField::ghost_field_reset = true; // this signals that we must reset the IPC comms
    }

  }

  void LatticeField::freeGhostBuffer(void)
  {
    destroyIPCComms();

    if (!initGhostFaceBuffer) return;

    for (int b=0; b<2; b++) {
      // free receive buffer
      if (ghost_recv_buffer_d[b]) device_comms_pinned_free(ghost_recv_buffer_d[b]);
      ghost_recv_buffer_d[b] = nullptr;

      // free send buffer
      if (ghost_send_buffer_d[b]) device_comms_pinned_free(ghost_send_buffer_d[b]);
      ghost_send_buffer_d[b] = nullptr;

      // free pinned send memory buffer
      if (ghost_pinned_recv_buffer_h[b]) host_free(ghost_pinned_recv_buffer_h[b]);

      // free pinned send memory buffer
      if (ghost_pinned_send_buffer_h[b]) host_free(ghost_pinned_send_buffer_h[b]);

      ghost_pinned_recv_buffer_h[b] = nullptr;
      ghost_pinned_recv_buffer_hd[b] = nullptr;
      ghost_pinned_send_buffer_h[b] = nullptr;
      ghost_pinned_send_buffer_hd[b] = nullptr;
    }
    initGhostFaceBuffer = false;
  }

  void LatticeField::createComms(bool no_comms_fill)
  {
    destroyComms(); // if we are requesting a new number of faces destroy and start over

    // before allocating local comm handles, synchronize since the
    // comms buffers are static so remove potential for interferring
    // with any outstanding exchanges to the same buffers
    qudaDeviceSynchronize();
    comm_barrier();

    // initialize the ghost pinned buffers
    for (int b=0; b<2; b++) {
      my_face_h[b] = ghost_pinned_send_buffer_h[b];
      my_face_hd[b] = ghost_pinned_send_buffer_hd[b];
      my_face_d[b] = ghost_send_buffer_d[b];
      from_face_h[b] = ghost_pinned_recv_buffer_h[b];
      from_face_hd[b] = ghost_pinned_recv_buffer_hd[b];
      from_face_d[b] = ghost_recv_buffer_d[b];
    }

    // initialize ghost send pointers
    for (int i=0; i<nDimComms; i++) {
      if (!commDimPartitioned(i) && no_comms_fill==false) continue;

      for (int dir = 0; dir < 2; dir++) {
        for (int b = 0; b < 2; ++b) {
          my_face_dim_dir_h[b][i][dir] = static_cast<char *>(my_face_h[b]) + ghost_offset[i][dir];
          from_face_dim_dir_h[b][i][dir] = static_cast<char *>(from_face_h[b]) + ghost_offset[i][dir];

          my_face_dim_dir_hd[b][i][dir] = static_cast<char *>(my_face_hd[b]) + ghost_offset[i][dir];
          from_face_dim_dir_hd[b][i][dir] = static_cast<char *>(from_face_hd[b]) + ghost_offset[i][dir];

          my_face_dim_dir_d[b][i][dir] = static_cast<char *>(my_face_d[b]) + ghost_offset[i][dir];
          from_face_dim_dir_d[b][i][dir] = static_cast<char *>(from_face_d[b]) + ghost_offset[i][dir];
        } // loop over b
      }   // loop over direction
    } // loop over dimension

    bool gdr = comm_gdr_enabled(); // only allocate rdma buffers if GDR enabled

    // initialize the message handlers
    for (int i=0; i<nDimComms; i++) {
      if (!commDimPartitioned(i)) continue;

      for (int b=0; b<2; ++b) {
	mh_send_fwd[b][i] = comm_declare_send_relative(my_face_dim_dir_h[b][i][1], i, +1, ghost_face_bytes[i]);
	mh_send_back[b][i] = comm_declare_send_relative(my_face_dim_dir_h[b][i][0], i, -1, ghost_face_bytes[i]);

	mh_recv_fwd[b][i] = comm_declare_receive_relative(from_face_dim_dir_h[b][i][1], i, +1, ghost_face_bytes[i]);
	mh_recv_back[b][i] = comm_declare_receive_relative(from_face_dim_dir_h[b][i][0], i, -1, ghost_face_bytes[i]);

	mh_send_rdma_fwd[b][i] = gdr ? comm_declare_send_relative(my_face_dim_dir_d[b][i][1], i, +1, ghost_face_bytes[i]) : nullptr;
	mh_send_rdma_back[b][i] = gdr ? comm_declare_send_relative(my_face_dim_dir_d[b][i][0], i, -1, ghost_face_bytes[i]) : nullptr;

	mh_recv_rdma_fwd[b][i] = gdr ? comm_declare_receive_relative(from_face_dim_dir_d[b][i][1], i, +1, ghost_face_bytes[i]) : nullptr;
	mh_recv_rdma_back[b][i] = gdr ? comm_declare_receive_relative(from_face_dim_dir_d[b][i][0], i, -1, ghost_face_bytes[i]) : nullptr;
      } // loop over b

    } // loop over dimension

    initComms = true;
  }

  void LatticeField::destroyComms()
  {
    if (initComms) {

      // ensure that all processes bring down their communicators
      // synchronously so that we don't end up in an undefined state
      qudaDeviceSynchronize();
      comm_barrier();

      for (int b=0; b<2; ++b) {
	for (int i=0; i<nDimComms; i++) {
          if (mh_recv_fwd[b][i]) comm_free(mh_recv_fwd[b][i]);
          if (mh_recv_back[b][i]) comm_free(mh_recv_back[b][i]);
          if (mh_send_fwd[b][i]) comm_free(mh_send_fwd[b][i]);
          if (mh_send_back[b][i]) comm_free(mh_send_back[b][i]);

          if (mh_recv_rdma_fwd[b][i]) comm_free(mh_recv_rdma_fwd[b][i]);
          if (mh_recv_rdma_back[b][i]) comm_free(mh_recv_rdma_back[b][i]);
          if (mh_send_rdma_fwd[b][i]) comm_free(mh_send_rdma_fwd[b][i]);
          if (mh_send_rdma_back[b][i]) comm_free(mh_send_rdma_back[b][i]);
        }
      } // loop over b

      // local take down complete - now synchronize to ensure globally complete
      qudaDeviceSynchronize();
      comm_barrier();

      initComms = false;
    }

  }

  void LatticeField::createIPCComms()
  {
    if ( initIPCComms && !ghost_field_reset ) return;

    if (!initComms) errorQuda("Can only be called after create comms");
    if ((!ghost_recv_buffer_d[0] || !ghost_recv_buffer_d[1]) && comm_size() > 1)
      errorQuda("ghost_field appears not to be allocated");

    for (int b=0; b<2; b++) {
      // set remote send buffer to ghost receive buffers on neighboring processes
      comm_create_neighbor_memory(ghost_remote_send_buffer_d[b], ghost_recv_buffer_d[b]);
      // get remote events
      comm_create_neighbor_event(ipcRemoteCopyEvent[b], ipcCopyEvent[b]);
    }

    // Create message handles for IPC synchronization
    for (int dim=0; dim<4; ++dim) {
      if (comm_dim(dim)==1) continue;
      if (comm_peer2peer_enabled(1,dim)) {
	for (int b=0; b<2; b++) {
	  // send to processor in forward direction
	  mh_send_p2p_fwd[b][dim] = comm_declare_send_relative(&buffer_send_p2p_fwd[b][dim], dim, +1, sizeof(int));
	  // receive from processor in forward direction
	  mh_recv_p2p_fwd[b][dim] = comm_declare_receive_relative(&buffer_recv_p2p_fwd[b][dim], dim, +1, sizeof(int));
	}
      }

      if (comm_peer2peer_enabled(0,dim)) {
	for (int b=0; b<2; b++) {
	  // send to processor in backward direction
	  mh_send_p2p_back[b][dim] = comm_declare_send_relative(&buffer_send_p2p_back[b][dim], dim, -1, sizeof(int));
	  // receive from processor in backward direction
	  mh_recv_p2p_back[b][dim] = comm_declare_receive_relative(&buffer_recv_p2p_back[b][dim], dim, -1, sizeof(int));
	}
      }
    }

    initIPCComms = true;
    ghost_field_reset = false;
  }

  void LatticeField::destroyIPCComms()
  {
    if (!initIPCComms) return;

    // ensure that all processes bring down their communicators
    // synchronously so that we don't end up in an undefined state
    qudaDeviceSynchronize();
    comm_barrier();

    for (int b = 0; b < 2; b++) {
      comm_destroy_neighbor_memory(ghost_remote_send_buffer_d[b]);
      comm_destroy_neighbor_event(ipcRemoteCopyEvent[b], ipcCopyEvent[b]);
    }

    for (int dim=0; dim<4; ++dim) {

      if (comm_dim(dim)==1) continue;

      for (int b=0; b<2; b++) {
	if (comm_peer2peer_enabled(1,dim)) {
          if (mh_send_p2p_fwd[b][dim]) comm_free(mh_send_p2p_fwd[b][dim]);
          if (mh_recv_p2p_fwd[b][dim]) comm_free(mh_recv_p2p_fwd[b][dim]);
        }

	if (comm_peer2peer_enabled(0,dim)) {
          if (mh_send_p2p_back[b][dim]) comm_free(mh_send_p2p_back[b][dim]);
          if (mh_recv_p2p_back[b][dim]) comm_free(mh_recv_p2p_back[b][dim]);
        }
      } // buffer
    } // iterate over dim

    // local take down complete - now synchronize to ensure globally complete
    qudaDeviceSynchronize();
    comm_barrier();

    initIPCComms = false;
  }

  bool LatticeField::ipcCopyComplete(int dir, int dim) { return qudaEventQuery(ipcCopyEvent[bufferIndex][dir][dim]); }

  bool LatticeField::ipcRemoteCopyComplete(int dir, int dim)
  {
    return qudaEventQuery(ipcRemoteCopyEvent[bufferIndex][dir][dim]);
  }

  const qudaEvent_t &LatticeField::getIPCCopyEvent(int dir, int dim) const
  {
    return ipcCopyEvent[bufferIndex][dir][dim];
  }

  const qudaEvent_t &LatticeField::getIPCRemoteCopyEvent(int dir, int dim) const
  {
    return ipcRemoteCopyEvent[bufferIndex][dir][dim];
  }

  void *LatticeField::myFace_h(int dir, int dim) const { return my_face_dim_dir_h[bufferIndex][dim][dir]; }

  void *LatticeField::myFace_hd(int dir, int dim) const { return my_face_dim_dir_hd[bufferIndex][dim][dir]; }

  void *LatticeField::myFace_d(int dir, int dim) const { return my_face_dim_dir_d[bufferIndex][dim][dir]; }

  void *LatticeField::remoteFace_d(int dir, int dim) const { return ghost_remote_send_buffer_d[bufferIndex][dim][dir]; }

  void *LatticeField::remoteFace_r() const { return ghost_recv_buffer_d[bufferIndex]; }

  void LatticeField::setTuningString() {
    char vol_tmp[TuneKey::volume_n];
    int check  = snprintf(vol_string, TuneKey::volume_n, "%d", x[0]);
    if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
    for (int d=1; d<nDim; d++) {
      strcpy(vol_tmp, vol_string);
      check = snprintf(vol_string, TuneKey::volume_n, "%sx%d", vol_tmp, x[d]);
      if (check < 0 || check >= TuneKey::volume_n) errorQuda("Error writing volume string");
    }
  }

  void LatticeField::checkField(const LatticeField &a) const {
    if (a.nDim != nDim) errorQuda("nDim does not match %d %d", nDim, a.nDim);
    if (ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && a.ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED) {
      // if source is extended by I am not then we need to compare their interior volume to my volume
      size_t a_volume_interior = 1;
      for (int i=0; i<nDim; i++) {
	if (a.x[i]-2*a.r[i] != x[i]) errorQuda("x[%d] does not match %d %d", i, x[i], a.x[i]-2*a.r[i]);
	a_volume_interior *= a.x[i] - 2*a.r[i];
      }
      if (a_volume_interior != volume) errorQuda("Interior volume does not match %lu %lu", volume, a_volume_interior);
    } else if (a.ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED) {
      // if source is extended by I am not then we need to compare their interior volume to my volume
      size_t this_volume_interior = 1;
      for (int i=0; i<nDim; i++) {
	if (x[i]-2*r[i] != a.x[i]) errorQuda("x[%d] does not match %d %d", i, x[i]-2*r[i], a.x[i]);
	this_volume_interior *= x[i] - 2*r[i];
      }
      if (this_volume_interior != a.volume)
        errorQuda("Interior volume does not match %lu %lu", this_volume_interior, a.volume);
    } else {
      if (a.volume != volume) errorQuda("Volume does not match %lu %lu", volume, a.volume);
      if (a.volumeCB != volumeCB) errorQuda("VolumeCB does not match %lu %lu", volumeCB, a.volumeCB);
      for (int i=0; i<nDim; i++) {
	if (a.x[i] != x[i]) errorQuda("x[%d] does not match %d %d", i, x[i], a.x[i]);
	if (a.surface[i] != surface[i]) errorQuda("surface[%d] does not match %d %d", i, surface[i], a.surface[i]);
	if (a.surfaceCB[i] != surfaceCB[i]) errorQuda("surfaceCB[%d] does not match %d %d", i, surfaceCB[i], a.surfaceCB[i]);
        if (a.local_surface[i] != local_surface[i])
          errorQuda("local_surface[%d] does not match %d %d", i, local_surface[i], a.local_surface[i]);
        if (a.local_surfaceCB[i] != local_surfaceCB[i])
          errorQuda("local_surfaceCB[%d] does not match %d %d", i, local_surfaceCB[i], a.local_surfaceCB[i]);
      }
    }
  }

  QudaFieldLocation LatticeField::Location() const
  {
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;
    if (typeid(*this) == typeid(cudaGaugeField)) {
      location = QUDA_CUDA_FIELD_LOCATION;
    } else if (typeid(*this) == typeid(cpuGaugeField)) {
      location = QUDA_CPU_FIELD_LOCATION;
    } else {
      errorQuda("Unknown field %s, so cannot determine location", typeid(*this).name());
    }
    return location;
  }

  void LatticeField::read(char *) { errorQuda("Not implemented"); }

  void LatticeField::write(char *) { errorQuda("Not implemented"); }

  int LatticeField::Nvec() const {
    if (typeid(*this) == typeid(const ColorSpinorField)) {
      const ColorSpinorField &csField = static_cast<const ColorSpinorField&>(*this);
      if (csField.FieldOrder() == 2 || csField.FieldOrder() == 4)
	return static_cast<int>(csField.FieldOrder());
    } else if (typeid(*this) == typeid(const cudaGaugeField)) {
      const GaugeField &gField = static_cast<const GaugeField&>(*this);
      if (gField.Order() == 2 || gField.Order() == 4)
	return static_cast<int>(gField.Order());
    } else if (typeid(*this) == typeid(const CloverField)) {
      const CloverField &cField = static_cast<const CloverField&>(*this);
      if (cField.Order() == 2 || cField.Order() == 4)
	return static_cast<int>(cField.Order());
    }

    errorQuda("Unsupported field type");
    return -1;
  }

  // This doesn't really live here, but is fine for the moment
  std::ostream& operator<<(std::ostream& output, const LatticeFieldParam& param)
  {
    output << "nDim = " << param.nDim << std::endl;
    for (int i = 0; i < param.nDim; i++) { output << "x[" << i << "] = " << param.x[i] << std::endl; }
    output << "pad = " << param.pad << std::endl;
    output << "precision = " << param.Precision() << std::endl;
    output << "ghost_precision = " << param.GhostPrecision() << std::endl;
    output << "scale = " << param.scale << std::endl;

    output << "ghostExchange = " << param.ghostExchange << std::endl;
    for (int i=0; i<param.nDim; i++) {
      output << "r[" << i << "] = " << param.r[i] << std::endl;
    }

    return output;  // for multiple << operators.
  }

  static QudaFieldLocation reorder_location_ = QUDA_CUDA_FIELD_LOCATION;

  QudaFieldLocation reorder_location() { return reorder_location_; }
  void reorder_location_set(QudaFieldLocation _reorder_location) { reorder_location_ = _reorder_location; }

} // namespace quda
