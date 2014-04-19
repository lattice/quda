#include <typeinfo>
#include <quda_internal.h>
#include <lattice_field.h>
#include <color_spinor_field.h>
#include <gauge_field.h>
#include <clover_field.h>
#include <face_quda.h>

namespace quda {

  void* LatticeField::bufferPinned = NULL;
  bool LatticeField::bufferPinnedInit = false;
  size_t LatticeField::bufferPinnedBytes = 0;

  void* LatticeField::bufferDevice = NULL;
  bool LatticeField::bufferDeviceInit = false;
  size_t LatticeField::bufferDeviceBytes = 0;

  LatticeField::LatticeField(const LatticeFieldParam &param)
    : volume(1), pad(param.pad), total_bytes(0), nDim(param.nDim), precision(param.precision),
      siteSubset(param.siteSubset)
  {
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
  
    for (int i=0; i<nDim; i++) 
      surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET) ? surface[i] / 2 : surface[i];

    // for 5-dimensional fields, we only communicate in the space-time dimensions
    nDimComms = nDim == 5 ? 4 : nDim;
  }

  LatticeField::~LatticeField() {
  }

  void LatticeField::checkField(const LatticeField &a) {
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
	typeid(*this)==typeid(cudaGaugeField)) {
      location = QUDA_CUDA_FIELD_LOCATION; 
    } else if (typeid(*this)==typeid(cpuCloverField) || 
	       typeid(*this)==typeid(cpuGaugeField)) {
      location = QUDA_CPU_FIELD_LOCATION;
    } else {
      errorQuda("Unknown field %s, so cannot determine location", typeid(*this).name());
    }
    return location;
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

  void LatticeField::resizeBufferPinned(size_t bytes) const {
    if ((bytes > bufferPinnedBytes || bufferPinnedInit == 0) && bytes > 0) {
      if (bufferPinnedInit) host_free(bufferPinned);
      bufferPinned = pinned_malloc(bytes);
      bufferPinnedBytes = bytes;
      bufferPinnedInit = true;
    }
  }

  void LatticeField::resizeBufferDevice(size_t bytes) const {
    if ((bytes > bufferDeviceBytes || bufferDeviceInit == 0) && bytes > 0) {
      if (bufferDeviceInit) host_free(bufferPinned);
      bufferDevice = device_malloc(bytes);
      bufferDeviceBytes = bytes;
      bufferDeviceInit = true;
    }
  }

  void LatticeField::freeBuffer() {
    if (bufferPinnedInit) {
      host_free(bufferPinned);
      bufferPinned = NULL;
      bufferPinnedBytes = 0;
      bufferPinnedInit = false;
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

    return output;  // for multiple << operators.
  }

} // namespace quda
