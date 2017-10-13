#include <typeinfo>
#include <quda_internal.h>
#include <lattice_field.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <face_quda.h>

namespace quda {

  void* LatticeField::bufferPinned[] = {NULL};
  bool LatticeField::bufferPinnedInit[] = {false};
  size_t LatticeField::bufferPinnedBytes[] = {0};
  size_t LatticeField::bufferPinnedResizeCount = 0;

  void* LatticeField::bufferDevice = NULL;
  bool LatticeField::bufferDeviceInit = false;
  size_t LatticeField::bufferDeviceBytes = 0;

  bool LatticeField::initIPCComms = false;

  int LatticeField::buffer_send_p2p_fwd[2][QUDA_MAX_DIM] { };
  int LatticeField::buffer_recv_p2p_fwd[2][QUDA_MAX_DIM] { };
  int LatticeField::buffer_send_p2p_back[2][QUDA_MAX_DIM] { };
  int LatticeField::buffer_recv_p2p_back[2][QUDA_MAX_DIM] { };

  MsgHandle* LatticeField::mh_send_p2p_fwd[2][QUDA_MAX_DIM] { };
  MsgHandle* LatticeField::mh_send_p2p_back[2][QUDA_MAX_DIM] { };
  MsgHandle* LatticeField::mh_recv_p2p_fwd[2][QUDA_MAX_DIM] { };
  MsgHandle* LatticeField::mh_recv_p2p_back[2][QUDA_MAX_DIM] { };

  cudaEvent_t LatticeField::ipcCopyEvent[2][2][QUDA_MAX_DIM];
  cudaEvent_t LatticeField::ipcRemoteCopyEvent[2][2][QUDA_MAX_DIM];

  void *LatticeField::ghost_pinned_buffer_h[2] = {nullptr, nullptr};
  void *LatticeField::ghost_pinned_buffer_hd[2] = {nullptr, nullptr};

  // gpu ghost receive buffer
  void *LatticeField::ghost_recv_buffer_d[2] = {nullptr, nullptr};

  // gpu ghost send buffer
  void *LatticeField::ghost_send_buffer_d[2] = {nullptr, nullptr};

  bool LatticeField::ghost_field_reset = false;

  void* LatticeField::ghost_remote_send_buffer_d[2][QUDA_MAX_DIM][2];

  int LatticeField::bufferIndex = 0;

  LatticeFieldParam::LatticeFieldParam(const LatticeField &field)
    : nDim(field.Ndim()), pad(field.Pad()), precision(field.Precision()),
      siteSubset(field.SiteSubset()), mem_type(field.MemType()),  ghostExchange(field.GhostExchange())
  {
    for(int dir=0; dir<nDim; ++dir) {
      x[dir] = field.X()[dir];
      r[dir] = field.R()[dir];
    }
  }

  LatticeField::LatticeField(const LatticeFieldParam &param)
    : volume(1), pad(param.pad), total_bytes(0), nDim(param.nDim), precision(param.precision),
      siteSubset(param.siteSubset), ghostExchange(param.ghostExchange), initComms(false), mem_type(param.mem_type),
      backup_h(nullptr), backup_norm_h(nullptr), backed_up(false)
  {
    for (int i=0; i<nDim; i++) {
      x[i] = param.x[i];
      r[i] = ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED ? param.r[i] : 0;
      volume *= param.x[i];
      surface[i] = 1;
      for (int j=0; j<nDim; j++) {
	if (i==j) continue;
	surface[i] *= param.x[j];
      }
    }

    if (siteSubset == QUDA_INVALID_SITE_SUBSET) errorQuda("siteSubset is not set");
    volumeCB = (siteSubset == QUDA_FULL_SITE_SUBSET) ? volume / 2 : volume;
    stride = volumeCB + pad;

    // for parity fields the factor of half is present for all surfaces dimensions except x, so add it manually
    for (int i=0; i<nDim; i++)
      surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET || i==0) ? surface[i] / 2 : surface[i];

    // for 5-dimensional fields, we only communicate in the space-time dimensions
    nDimComms = nDim == 5 ? 4 : nDim;

    setTuningString();
  }

  LatticeField::LatticeField(const LatticeField &field)
    : volume(1), pad(field.pad), total_bytes(0), nDim(field.nDim), precision(field.precision),
      siteSubset(field.siteSubset), ghostExchange(field.ghostExchange), initComms(false), mem_type(field.mem_type),
      backup_h(nullptr), backup_norm_h(nullptr), backed_up(false)
  {
    for (int i=0; i<nDim; i++) {
      x[i] = field.x[i];
      r[i] = ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED ? field.r[i] : 0;
      volume *= field.x[i];
      surface[i] = 1;
      for (int j=0; j<nDim; j++) {
	if (i==j) continue;
	surface[i] *= field.x[j];
      }
    }

    if (siteSubset == QUDA_INVALID_SITE_SUBSET) errorQuda("siteSubset is not set");
    volumeCB = (siteSubset == QUDA_FULL_SITE_SUBSET) ? volume / 2 : volume;
    stride = volumeCB + pad;
  
    // for parity fields the factor of half is present for all surfaces dimensions except x, so add it manually
    for (int i=0; i<nDim; i++) 
      surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET || i==0) ? surface[i] / 2 : surface[i];

    // for 5-dimensional fields, we only communicate in the space-time dimensions
    nDimComms = nDim == 5 ? 4 : nDim;

    setTuningString();
  }

  LatticeField::~LatticeField() { }

  void LatticeField::createIPCComms() {
    if ( initIPCComms && !ghost_field_reset ) return;

    if (!initComms) errorQuda("Can only be called after create comms");
    if ( (!ghost_recv_buffer_d[0] || !ghost_recv_buffer_d[1]) && comm_size() > 1) errorQuda("ghost_field appears not to be allocated");

    // handles for obtained ghost pointers
    cudaIpcMemHandle_t ipcRemoteGhostDestHandle[2][2][QUDA_MAX_DIM];

    for (int b=0; b<2; b++) {
      for (int dim=0; dim<4; ++dim) {
	if (comm_dim(dim)==1) continue;
	for (int dir=0; dir<2; ++dir) {
	  MsgHandle* sendHandle = nullptr;
	  MsgHandle* receiveHandle = nullptr;
	  int disp = (dir == 1) ? +1 : -1;

	  // first set up receive
	  if (comm_peer2peer_enabled(1-dir,dim)) {
	    receiveHandle = comm_declare_receive_relative(&ipcRemoteGhostDestHandle[b][1-dir][dim],
							  dim, -disp,
							  sizeof(ipcRemoteGhostDestHandle[b][1-dir][dim]));
	  }
	  // now send
	  if (comm_peer2peer_enabled(dir,dim)) {
	    cudaIpcMemHandle_t ipcLocalGhostDestHandle;
	    cudaIpcGetMemHandle(&ipcLocalGhostDestHandle, ghost_recv_buffer_d[b]);
	    sendHandle = comm_declare_send_relative(&ipcLocalGhostDestHandle,
						    dim, disp,
						    sizeof(ipcLocalGhostDestHandle));
	  }
	  if (receiveHandle) comm_start(receiveHandle);
	  if (sendHandle) comm_start(sendHandle);

	  if (receiveHandle) comm_wait(receiveHandle);
	  if (sendHandle) comm_wait(sendHandle);

	  if (sendHandle) comm_free(sendHandle);
	  if (receiveHandle) comm_free(receiveHandle);
	}
      }

      checkCudaError();

      // open the remote memory handles and set the send ghost pointers
      for (int dim=0; dim<4; ++dim) {
	if (comm_dim(dim)==1) continue;
	const int num_dir = (comm_dim(dim) == 2) ? 1 : 2;
	for (int dir=0; dir<num_dir; ++dir) {
	  if (!comm_peer2peer_enabled(dir,dim)) continue;
	  void **ghostDest = &(ghost_remote_send_buffer_d[b][dim][dir]);
	  cudaIpcOpenMemHandle(ghostDest, ipcRemoteGhostDestHandle[b][dir][dim],
			       cudaIpcMemLazyEnablePeerAccess);
	}
	if (num_dir == 1) ghost_remote_send_buffer_d[b][dim][1] = ghost_remote_send_buffer_d[b][dim][0];
      }
    } // buffer index

    checkCudaError();

    // handles for obtained events
    cudaIpcEventHandle_t ipcRemoteEventHandle[2][2][QUDA_MAX_DIM];

    // Note that no b index is necessary here
    // Now communicate the event handles
    for (int dim=0; dim<4; ++dim) {
      if (comm_dim(dim)==1) continue;
      for (int dir=0; dir<2; ++dir) {
	for (int b=0; b<2; b++) {

	  MsgHandle* sendHandle = NULL;
	  MsgHandle* receiveHandle = NULL;
	  int disp = (dir == 1) ? +1 : -1;

	  // first set up receive
	  if (comm_peer2peer_enabled(1-dir,dim)) {
	    receiveHandle = comm_declare_receive_relative(&ipcRemoteEventHandle[b][1-dir][dim], dim, -disp,
							  sizeof(ipcRemoteEventHandle[b][1-dir][dim]));
	  }

	  // now send
	  if (comm_peer2peer_enabled(dir,dim)) {
	    cudaEventCreate(&ipcCopyEvent[b][dir][dim], cudaEventDisableTiming | cudaEventInterprocess);
	    cudaIpcEventHandle_t ipcLocalEventHandle;
	    cudaIpcGetEventHandle(&ipcLocalEventHandle, ipcCopyEvent[b][dir][dim]);

	    sendHandle = comm_declare_send_relative(&ipcLocalEventHandle, dim, disp,
						    sizeof(ipcLocalEventHandle));
	  }

	  if (receiveHandle) comm_start(receiveHandle);
	  if (sendHandle) comm_start(sendHandle);

	  if (receiveHandle) comm_wait(receiveHandle);
	  if (sendHandle) comm_wait(sendHandle);

	  if (sendHandle) comm_free(sendHandle);
	  if (receiveHandle) comm_free(receiveHandle);

	} // buffer index
      }
    }

    checkCudaError();

    for (int dim=0; dim<4; ++dim) {
      if (comm_dim(dim)==1) continue;
      for (int dir=0; dir<2; ++dir) {
	if (!comm_peer2peer_enabled(dir,dim)) continue;
	for (int b=0; b<2; b++) {
	  cudaIpcOpenEventHandle(&(ipcRemoteCopyEvent[b][dir][dim]), ipcRemoteEventHandle[b][dir][dim]);
	}
      }
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
	  mh_send_p2p_back[b][dim] = comm_declare_send_relative(&buffer_recv_p2p_back[b][dim], dim, -1, sizeof(int));
	  // receive from processor in backward direction
	  mh_recv_p2p_back[b][dim] = comm_declare_receive_relative(&buffer_recv_p2p_back[b][dim], dim, -1, sizeof(int));
	}
      }
    }
    checkCudaError();

    initIPCComms = true;
    ghost_field_reset = false;
  }

  void LatticeField::destroyIPCComms() {

    if (!initIPCComms) return;
    checkCudaError();

    for (int dim=0; dim<4; ++dim) {

      if (comm_dim(dim)==1) continue;
      const int num_dir = (comm_dim(dim) == 2) ? 1 : 2;

      for (int b=0; b<2; b++) {
	if (comm_peer2peer_enabled(1,dim)) {
	  comm_free(mh_send_p2p_fwd[b][dim]);
	  comm_free(mh_recv_p2p_fwd[b][dim]);
	  cudaEventDestroy(ipcCopyEvent[b][1][dim]);

	  // only close this handle if it doesn't alias the back ghost
	  if (num_dir == 2) cudaIpcCloseMemHandle(ghost_remote_send_buffer_d[b][dim][1]);
	}

	if (comm_peer2peer_enabled(0,dim)) {
	  comm_free(mh_send_p2p_back[b][dim]);
	  comm_free(mh_recv_p2p_back[b][dim]);
	  cudaEventDestroy(ipcCopyEvent[b][0][dim]);

	  cudaIpcCloseMemHandle(ghost_remote_send_buffer_d[b][dim][0]);
	}
      } // buffer
    } // iterate over dim

    checkCudaError();
    initIPCComms = false;
  }

  bool LatticeField::ipcCopyComplete(int dir, int dim)
  {
    return (cudaSuccess == cudaEventQuery(ipcCopyEvent[bufferIndex][dir][dim]) ? true : false);
  }

  bool LatticeField::ipcRemoteCopyComplete(int dir, int dim)
  {
    return (cudaSuccess == cudaEventQuery(ipcRemoteCopyEvent[bufferIndex][dir][dim]) ? true : false);
  }

  const cudaEvent_t& LatticeField::getIPCRemoteCopyEvent(int dir, int dim) const {
    return ipcRemoteCopyEvent[bufferIndex][dir][dim];
  }

  void LatticeField::setTuningString() {
    char vol_tmp[TuneKey::volume_n];
    int check;
    check = snprintf(vol_string, TuneKey::volume_n, "%d", x[0]);
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
      int a_volume_interior = 1;
      for (int i=0; i<nDim; i++) {
	if (a.x[i]-2*a.r[i] != x[i]) errorQuda("x[%d] does not match %d %d", i, x[i], a.x[i]-2*a.r[i]);
	a_volume_interior *= a.x[i] - 2*a.r[i];
      }
      if (a_volume_interior != volume) errorQuda("Interior volume does not match %d %d", volume, a_volume_interior);
    } else if (a.ghostExchange != QUDA_GHOST_EXCHANGE_EXTENDED && ghostExchange == QUDA_GHOST_EXCHANGE_EXTENDED) {
      // if source is extended by I am not then we need to compare their interior volume to my volume
      int this_volume_interior = 1;
      for (int i=0; i<nDim; i++) {
	if (x[i]-2*r[i] != a.x[i]) errorQuda("x[%d] does not match %d %d", i, x[i]-2*r[i], a.x[i]);
	this_volume_interior *= x[i] - 2*r[i];
      }
      if (this_volume_interior != a.volume) errorQuda("Interior volume does not match %d %d", this_volume_interior, a.volume);
    } else {
      if (a.volume != volume) errorQuda("Volume does not match %d %d", volume, a.volume);
      if (a.volumeCB != volumeCB) errorQuda("VolumeCB does not match %d %d", volumeCB, a.volumeCB);
      for (int i=0; i<nDim; i++) {
	if (a.x[i] != x[i]) errorQuda("x[%d] does not match %d %d", i, x[i], a.x[i]);
	if (a.surface[i] != surface[i]) errorQuda("surface[%d] does not match %d %d", i, surface[i], a.surface[i]);
	if (a.surfaceCB[i] != surfaceCB[i]) errorQuda("surfaceCB[%d] does not match %d %d", i, surfaceCB[i], a.surfaceCB[i]);
      }
    }
  }

  QudaFieldLocation LatticeField::Location() const { 
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;
    if (typeid(*this)==typeid(cudaCloverField) || 
	typeid(*this)==typeid(cudaColorSpinorField) ||
	typeid(*this)==typeid(cudaGaugeField)) {
      location = QUDA_CUDA_FIELD_LOCATION; 
    } else if (typeid(*this)==typeid(cpuCloverField) || 
	       typeid(*this)==typeid(cpuColorSpinorField) ||
	       typeid(*this)==typeid(cpuGaugeField)) {
      location = QUDA_CPU_FIELD_LOCATION;
      location = QUDA_CPU_FIELD_LOCATION;
    } else {
      errorQuda("Unknown field %s, so cannot determine location", typeid(*this).name());
    }
    return location;
}

  void LatticeField::read(char *filename) {
    errorQuda("Not implemented");
  }
  
  void LatticeField::write(char *filename) {
    errorQuda("Not implemented");
  }

  int LatticeField::Nvec() const {
    if (typeid(*this) == typeid(const cudaColorSpinorField)) {
      const ColorSpinorField &csField = static_cast<const ColorSpinorField&>(*this);
      if (csField.FieldOrder() == 2 || csField.FieldOrder() == 4)
	return static_cast<int>(csField.FieldOrder());
    } else if (typeid(*this) == typeid(const cudaGaugeField)) {
      const GaugeField &gField = static_cast<const GaugeField&>(*this);
      if (gField.Order() == 2 || gField.Order() == 4)
	return static_cast<int>(gField.Order());
    } else if (typeid(*this) == typeid(const cudaCloverField)) { 
      const CloverField &cField = static_cast<const CloverField&>(*this);
      if (cField.Order() == 2 || cField.Order() == 4)
	return static_cast<int>(cField.Order());
    }

    errorQuda("Unsupported field type");
    return -1;
  }

  void LatticeField::resizeBufferPinned(size_t bytes, const int idx) const {
    if ((bytes > bufferPinnedBytes[idx] || bufferPinnedInit[idx] == 0) && bytes > 0) {
      if (bufferPinnedInit[idx]) host_free(bufferPinned[idx]);
      bufferPinned[idx] = pinned_malloc(bytes);
      bufferPinnedBytes[idx] = bytes;
      bufferPinnedInit[idx] = true;
      bufferPinnedResizeCount++;
      if (bufferPinnedResizeCount == 0) bufferPinnedResizeCount = 1; // keep 0 as initialization state
    }
  }

  void LatticeField::resizeBufferDevice(size_t bytes) const {
    if ((bytes > bufferDeviceBytes || bufferDeviceInit == 0) && bytes > 0) {
      if (bufferDeviceInit) device_free(bufferDevice);
      bufferDevice = device_malloc(bytes);
      bufferDeviceBytes = bytes;
      bufferDeviceInit = true;
    }
  }

  void LatticeField::freeBuffer(int index) {
    if (bufferPinnedInit[index]) {
      host_free(bufferPinned[index]);
      bufferPinned[index] = NULL;
      bufferPinnedBytes[index] = 0;
      bufferPinnedInit[index] = false;
    }
    if (bufferDeviceInit) {
      device_free(bufferDevice);
      bufferDevice = NULL;
      bufferDeviceBytes = 0;
      bufferDeviceInit = false;
    }
  }

  // This doesn't really live here, but is fine for the moment
  std::ostream& operator<<(std::ostream& output, const LatticeFieldParam& param)
  {
    output << "nDim = " << param.nDim << std::endl;
    for (int i=0; i<param.nDim; i++) {
      output << "x[" << i << "] = " << param.x[i] << std::endl;    
    }
    output << "pad = " << param.pad << std::endl;
    output << "precision = " << param.precision << std::endl;

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
