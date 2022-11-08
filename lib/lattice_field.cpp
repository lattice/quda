#include <typeinfo>
#include <quda_internal.h>
#include <lattice_field.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>

namespace quda {

  LatticeFieldParam::LatticeFieldParam(const LatticeField &field) :
    location(field.Location()),
    precision(field.Precision()),
    ghost_precision(field.Precision()),
    nDim(field.Ndim()),
    pad(field.Pad()),
    siteSubset(field.SiteSubset()),
    mem_type(field.MemType()),
    ghostExchange(field.GhostExchange()),
    scale(field.Scale())
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
    location(param.location),
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
    my_face_dim_dir_h {},
    my_face_dim_dir_hd {},
    my_face_dim_dir_d {},
    from_face_h {},
    from_face_hd {},
    from_face_d {},
    from_face_dim_dir_h {},
    from_face_dim_dir_hd {},
    from_face_dim_dir_d {},
    mh_recv {},
    mh_send {},
    mh_recv_rdma {},
    mh_send_rdma {},
    initComms(false),
    mem_type(param.mem_type),
    backup_h(nullptr),
    backup_norm_h(nullptr),
    backed_up(false)
  {
    create(param);
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
    location(field.location),
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
    my_face_dim_dir_h {},
    my_face_dim_dir_hd {},
    my_face_dim_dir_d {},
    from_face_h {},
    from_face_hd {},
    from_face_d {},
    from_face_dim_dir_h {},
    from_face_dim_dir_hd {},
    from_face_dim_dir_d {},
    mh_recv {},
    mh_send {},
    mh_recv_rdma {},
    mh_send_rdma {},
    initComms(false),
    mem_type(field.mem_type),
    backup_h(nullptr),
    backup_norm_h(nullptr),
    backed_up(false)
  {
    LatticeFieldParam param;
    field.fill(param);
    create(param);
  }

  LatticeField::LatticeField(LatticeField &&field) { move(std::move(field)); }

  LatticeField::~LatticeField() { destroyComms(); }

  LatticeField &LatticeField::operator=(const LatticeField &src)
  {
    if (&src != this) {
      destroyComms();
      LatticeFieldParam param;
      src.fill(param);
      create(param);
    }
    return *this;
  }

  LatticeField &LatticeField::operator=(LatticeField &&src)
  {
    if (&src != this) {
      destroyComms();
      move(std::move(src));
    }
    return *this;
  }

  void LatticeField::create(const LatticeFieldParam &param)
  {
    if (param.location == QUDA_INVALID_FIELD_LOCATION) errorQuda("Invalid field location");
    location = param.location;
    precision = param.Precision();
    ghost_precision = param.ghost_precision;
    precisionCheck();

    if (param.nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions nDim = %d too great", param.nDim);
    nDim = param.nDim;

    volume = 1;
    localVolume = 1;
    for (int i = 0; i < nDim; i++) {
      x[i] = param.x[i];
      r[i] = ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED ? param.r[i] : 0;
      local_x[i] = x[i] - 2 * r[i];
      volume *= x[i];
      localVolume *= local_x[i];
      surface[i] = 1;
      local_surface[i] = 1;
      for (int j = 0; j < nDim; j++) {
        if (i == j) continue;
        surface[i] *= param.x[j];
        local_surface[i] *= param.x[j] - 2 * param.r[j];
      }
    }

    if (param.siteSubset == QUDA_INVALID_SITE_SUBSET) errorQuda("siteSubset is not set");
    siteSubset = param.siteSubset;
    volumeCB = (siteSubset == QUDA_FULL_SITE_SUBSET) ? volume / 2 : volume;
    localVolumeCB = (siteSubset == QUDA_FULL_SITE_SUBSET) ? localVolume / 2 : localVolume;
    stride = volumeCB + pad;

    // for parity fields the factor of half is present for all surfaces dimensions except x, so add it manually
    for (int i = 0; i < nDim; i++) {
      surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET || i == 0) ? surface[i] / 2 : surface[i];
      local_surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET || i == 0) ? local_surface[i] / 2 : local_surface[i];
    }

    // for 5-dimensional fields, we only communicate in the space-time dimensions
    nDimComms = nDim == 5 ? 4 : nDim;

    mem_type = param.mem_type;

    setTuningString();
  }

  void LatticeField::move(LatticeField &&src)
  {
    volume = std::exchange(src.volume, 0);
    volumeCB = std::exchange(src.volumeCB, 0);
    localVolume = std::exchange(src.localVolume, 0);
    localVolumeCB = std::exchange(src.localVolumeCB, 0);
    stride = std::exchange(src.stride, 0);
    pad = std::exchange(src.pad, 0);
    total_bytes = std::exchange(src.total_bytes, 0);
    nDim = std::exchange(src.nDim, 0);
    x = std::exchange(src.x, {});
    r = std::exchange(src.r, {});
    local_x = std::exchange(src.local_x, {});
    surface = std::exchange(src.surface, {});
    surfaceCB = std::exchange(src.surfaceCB, {});
    local_surface = std::exchange(src.local_surface, {});
    local_surfaceCB = std::exchange(src.local_surfaceCB, {});
    location = std::exchange(src.location, QUDA_INVALID_FIELD_LOCATION);
    precision = std::exchange(src.precision, QUDA_INVALID_PRECISION);
    ghost_precision = std::exchange(src.ghost_precision, QUDA_INVALID_PRECISION);
    ghost_precision_reset = std::exchange(src.ghost_precision_reset, false);
    scale = std::exchange(src.scale, 0.0);
    siteSubset = std::exchange(src.siteSubset, QUDA_INVALID_SITE_SUBSET);
    ghostExchange = std::exchange(src.ghostExchange, QUDA_GHOST_EXCHANGE_INVALID);
    nDimComms = std::exchange(src.nDimComms, 0);
    ghost_bytes = std::exchange(src.ghost_bytes, 0);
    ghost_bytes_old = std::exchange(src.ghost_bytes_old, {});
    ghost_face_bytes = std::exchange(src.ghost_face_bytes, {});
    ghost_face_bytes_aligned = std::exchange(src.ghost_face_bytes_aligned, {});
    ghost_offset = std::exchange(src.ghost_offset, {});
    my_face_h = std::exchange(src.my_face_h, {});
    my_face_hd = std::exchange(src.my_face_hd, {});
    my_face_d = std::exchange(src.my_face_d, {});
    my_face_dim_dir_h = std::exchange(src.my_face_dim_dir_h, {});
    my_face_dim_dir_hd = std::exchange(src.my_face_dim_dir_hd, {});
    my_face_dim_dir_d = std::exchange(src.my_face_dim_dir_d, {});
    from_face_h = std::exchange(src.from_face_h, {});
    from_face_hd = std::exchange(src.from_face_hd, {});
    from_face_d = std::exchange(src.from_face_d, {});
    from_face_dim_dir_h = std::exchange(src.from_face_dim_dir_h, {});
    from_face_dim_dir_hd = std::exchange(src.from_face_dim_dir_hd, {});
    from_face_dim_dir_d = std::exchange(src.from_face_dim_dir_d, {});
    mh_recv = std::exchange(src.mh_recv, {});
    mh_send = std::exchange(src.mh_send, {});
    mh_recv_rdma = std::exchange(src.mh_recv_rdma, {});
    mh_send_rdma = std::exchange(src.mh_send_rdma, {});
    initComms = std::exchange(src.initComms, false);
    vol_string = std::exchange(src.vol_string, {});
    aux_string = std::exchange(src.aux_string, {});
    mem_type = std::exchange(src.mem_type, QUDA_MEMORY_INVALID);
    backup_h = std::exchange(src.backup_h, nullptr);
    backup_norm_h = std::exchange(src.backup_norm_h, nullptr);
    backed_up = std::exchange(src.backed_up, false);
  }

  void LatticeField::fill(LatticeFieldParam &param) const
  {
    param.location = location;
    param.precision = precision;
    param.ghost_precision = ghost_precision;
    param.nDim = nDim;
    param.x = x;
    param.pad = pad;
    param.siteSubset = siteSubset;
    param.mem_type = mem_type;
    param.ghostExchange = ghostExchange;
    param.r = r;
    param.scale = scale;
  }

  void LatticeField::allocateGhostBuffer(size_t ghost_bytes) const
  {
    gb.allocate(ghost_bytes);
  }

  void LatticeField::freeGhostBuffer(void)
  {
    GhostBuffer::free();  
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
      my_face_h[b] = gb.ghost_pinned_send_buffer_h[b];
      my_face_hd[b] = gb.ghost_pinned_send_buffer_hd[b];
      my_face_d[b] = gb.ghost_send_buffer_d[b];
      from_face_h[b] = gb.ghost_pinned_recv_buffer_h[b];
      from_face_hd[b] = gb.ghost_pinned_recv_buffer_hd[b];
      from_face_d[b] = gb.ghost_recv_buffer_d[b];
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

      for (int dir = 0; dir < 2; dir++) {
        int hop = dir == 0 ? -1 : +1;
        for (int b = 0; b < 2; ++b) {
          mh_send[b][i][dir] = comm_declare_send_relative(my_face_dim_dir_h[b][i][dir], i, hop, ghost_face_bytes[i]);
          mh_recv[b][i][dir] = comm_declare_receive_relative(from_face_dim_dir_h[b][i][dir], i, hop, ghost_face_bytes[i]);
          mh_send_rdma[b][i][dir]
            = gdr ? comm_declare_send_relative(my_face_dim_dir_d[b][i][dir], i, hop, ghost_face_bytes[i]) : nullptr;
          mh_recv_rdma[b][i][dir]
            = gdr ? comm_declare_receive_relative(from_face_dim_dir_d[b][i][dir], i, hop, ghost_face_bytes[i]) : nullptr;
        }
      } // loop over b

    } // loop over dimension

    initComms = true;
  }

  void LatticeField::destroyComms()
  {
    if (Location() != QUDA_CUDA_FIELD_LOCATION) return;

    if (initComms) {

      // ensure that all processes bring down their communicators
      // synchronously so that we don't end up in an undefined state
      qudaDeviceSynchronize();
      comm_barrier();

      my_face_h = {};
      my_face_hd = {};
      my_face_d = {};
      from_face_h = {};
      from_face_hd = {};
      from_face_d = {};

      my_face_dim_dir_h = {};
      my_face_dim_dir_hd = {};
      my_face_dim_dir_d = {};
      from_face_dim_dir_h = {};
      from_face_dim_dir_hd = {};
      from_face_dim_dir_d = {};

      for (int b=0; b<2; ++b) {
	for (int i=0; i<nDimComms; i++) {
          for (int dir = 0; dir < 2; dir++) {
            if (mh_recv[b][i][dir]) comm_free(mh_recv[b][i][dir]);
            if (mh_send[b][i][dir]) comm_free(mh_send[b][i][dir]);
            if (mh_recv_rdma[b][i][dir]) comm_free(mh_recv_rdma[b][i][dir]);
            if (mh_send_rdma[b][i][dir]) comm_free(mh_send_rdma[b][i][dir]);
          }
        }
      } // loop over b

      mh_recv = {};
      mh_send = {};
      mh_recv_rdma = {};
      mh_send_rdma = {};

      // local take down complete - now synchronize to ensure globally complete
      qudaDeviceSynchronize();
      comm_barrier();

      initComms = false;
    }

  }

  void LatticeField::createIPCComms()
  {
    gb.createIPCComms();
   }

  void LatticeField::destroyIPCComms()
  {
    GhostBuffer::destroyIPCComms();
  }

  bool LatticeField::ipcCopyComplete(int dir, int dim) { return qudaEventQuery(gb.ipcCopyEvent[bufferIndex][dim][dir]); }

  bool LatticeField::ipcRemoteCopyComplete(int dir, int dim)
  {
    return qudaEventQuery(gb.ipcRemoteCopyEvent[bufferIndex][dim][dir]);
  }

  const qudaEvent_t &LatticeField::getIPCCopyEvent(int dir, int dim) const
  {
    return gb.ipcCopyEvent[bufferIndex][dim][dir];
  }

  const qudaEvent_t &LatticeField::getIPCRemoteCopyEvent(int dir, int dim) const
  {
    return gb.ipcRemoteCopyEvent[bufferIndex][dim][dir];
  }

  void *LatticeField::myFace_h(int dir, int dim) const { return my_face_dim_dir_h[bufferIndex][dim][dir]; }

  void *LatticeField::myFace_hd(int dir, int dim) const { return my_face_dim_dir_hd[bufferIndex][dim][dir]; }

  void *LatticeField::myFace_d(int dir, int dim) const { return my_face_dim_dir_d[bufferIndex][dim][dir]; }

  void *LatticeField::remoteFace_d(int dir, int dim) const { return gb.ghost_remote_send_buffer_d[bufferIndex][dim][dir]; }

  void *LatticeField::remoteFace_r() const { return gb.ghost_recv_buffer_d[bufferIndex]; }

  void LatticeField::setTuningString()
  {
    std::stringstream vol_ss;
    vol_ss << x[0];
    for (int d = 1; d < nDim; d++) vol_ss << "x" << x[d];
    vol_string = vol_ss.str();
    if (vol_string.size() >= TuneKey::volume_n) errorQuda("Vol string too large %lu", vol_string.size());
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
