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

  // cache of inactive allocations
  std::multimap<size_t, void *> LatticeField::pinnedCache;

  // sizes of active allocations
  std::map<void *, size_t> LatticeField::pinnedSize;

  // cache of inactive allocations
  std::multimap<size_t, void *> LatticeField::deviceCache;

  // sizes of active allocations
  std::map<void *, size_t> LatticeField::deviceSize;

  static bool pool_init = false;

  // whether to use a memory pool allocator for device memory
  static bool device_memory_pool = true;

  // whether to use a memory pool allocator for pinned memory
  static bool pinned_memory_pool = true;

  LatticeField::LatticeField(const LatticeFieldParam &param)
    : volume(1), pad(param.pad), total_bytes(0), nDim(param.nDim), precision(param.precision),
      siteSubset(param.siteSubset)
  {
    if (!pool_init) {
      // device memory pool
      char *enable_device_pool = getenv("QUDA_ENABLE_DEVICE_MEMORY_POOL");
      if (!enable_device_pool || strcmp(enable_device_pool,"0")!=0) {
	warningQuda("Using device memory pool allocator");
	device_memory_pool = true;
      } else {
	warningQuda("Not using device memory pool allocator");
	device_memory_pool = false;
      }

      // pinned memory pool
      char *enable_pinned_pool = getenv("QUDA_ENABLE_PINNED_MEMORY_POOL");
      if (!enable_pinned_pool || strcmp(enable_pinned_pool,"0")!=0) {
	warningQuda("Using pinned memory pool allocator");
	pinned_memory_pool = true;
      } else {
	warningQuda("Not using pinned memory pool allocator");
	pinned_memory_pool = false;
      }
      pool_init = true;
    }

    for (int i=0; i<nDim; i++) {
      x[i] = param.x[i];
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

  LatticeField::~LatticeField() {
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
    if (a.volume != volume) errorQuda("Volume does not match %d %d", volume, a.volume);
    if (a.volumeCB != volumeCB) errorQuda("VolumeCB does not match %d %d", volumeCB, a.volumeCB);
    if (a.nDim != nDim) errorQuda("nDim does not match %d %d", nDim, a.nDim);
    for (int i=0; i<nDim; i++) {
      if (a.x[i] != x[i]) errorQuda("x[%d] does not match %d %d", i, x[i], a.x[i]);
      if (a.surface[i] != surface[i]) errorQuda("surface[%d] does not match %d %d", i, surface[i], a.surface[i]);
      if (a.surfaceCB[i] != surfaceCB[i]) errorQuda("surfaceCB[%d] does not match %d %d", i, surfaceCB[i], a.surfaceCB[i]);  
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

  void *LatticeField::allocatePinned(size_t nbytes)
  {
    void *ptr = nullptr;
    if (pinned_memory_pool) {
      std::multimap<size_t, void *>::iterator it;

      if (pinnedCache.empty()) {
	ptr = pinned_malloc(nbytes);
      } else {
	it = pinnedCache.lower_bound(nbytes);
	if (it != pinnedCache.end()) { // sufficiently large allocation found
	  nbytes = it->first;
	  ptr = it->second;
	  pinnedCache.erase(it);
	} else { // sacrifice the smallest cached allocation
	  it = pinnedCache.begin();
	  ptr = it->second;
	  pinnedCache.erase(it);
	  host_free(ptr);
	  ptr = pinned_malloc(nbytes);
	}
      }
      pinnedSize[ptr] = nbytes;
    } else {
      ptr = pinned_malloc(nbytes);
    }
    return ptr;
  }

  void LatticeField::freePinned(void *ptr)
  {
    if (pinned_memory_pool) {
      if (!pinnedSize.count(ptr)) {
	errorQuda("Attempt to free invalid pointer");
      }
      pinnedCache.insert(std::make_pair(pinnedSize[ptr], ptr));
      pinnedSize.erase(ptr);
    } else {
      host_free(ptr);
    }
  }

  void LatticeField::flushPinnedCache()
  {
    if (pinned_memory_pool) {
      std::multimap<size_t, void *>::iterator it;
      for (it = pinnedCache.begin(); it != pinnedCache.end(); it++) {
	void *ptr = it->second;
	host_free(ptr);
      }
      pinnedCache.clear();
    }
  }

  void *LatticeField::allocateDevice(size_t nbytes) const
  {
    void *ptr = nullptr;
    if (device_memory_pool) {
      std::multimap<size_t, void *>::iterator it;

      if (deviceCache.empty()) {
	ptr = device_malloc(nbytes);
      } else {
	it = deviceCache.lower_bound(nbytes);
	if (it != deviceCache.end()) { // sufficiently large allocation found
	  nbytes = it->first;
	  ptr = it->second;
	  deviceCache.erase(it);
	} else { // sacrifice the smallest cached allocation
	  it = deviceCache.begin();
	  ptr = it->second;
	  deviceCache.erase(it);
	  device_free(ptr);
	  ptr = device_malloc(nbytes);
	}
      }
      deviceSize[ptr] = nbytes;
    } else {
      ptr = device_malloc(nbytes);
    }
    return ptr;
  }

  void LatticeField::freeDevice(void *ptr) const
  {
    if (device_memory_pool) {
      if (!deviceSize.count(ptr)) {
	errorQuda("Attempt to free invalid pointer");
      }
      deviceCache.insert(std::make_pair(deviceSize[ptr], ptr));
      deviceSize.erase(ptr);
    } else {
      device_free(ptr);
    }
  }

  void LatticeField::flushDeviceCache()
  {
    if (device_memory_pool) {
      std::multimap<size_t, void *>::iterator it;
      for (it = deviceCache.begin(); it != deviceCache.end(); it++) {
	void *ptr = it->second;
	device_free(ptr);
      }
      deviceCache.clear();
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

    return output;  // for multiple << operators.
  }

  static QudaFieldLocation reorder_location_ = QUDA_CUDA_FIELD_LOCATION;

  QudaFieldLocation reorder_location() { return reorder_location_; }
  void reorder_location_set(QudaFieldLocation _reorder_location) { reorder_location_ = _reorder_location; }

} // namespace quda
