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
      siteSubset(param.siteSubset), initComms(false)
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
    volumeCB = (siteSubset == QUDA_FULL_SITE_SUBSET) ? volume / 2 : volume;
    stride = volumeCB + pad;
  
    for (int i=0; i<nDim; i++) 
      surfaceCB[i] = (siteSubset == QUDA_FULL_SITE_SUBSET) ? surface[i] / 2 : surface[i];

    // for 5-dimensional fields, we only communicate in the space-time dimensions
    nDimComms = nDim == 5 ? 4 : nDim;
  }

  LatticeField::~LatticeField() {
    if (initComms) destroyComms();
  }


  void LatticeField::createComms() {
    if (!initComms) {

      // FIXME this only supports single parity dirac fields
      if (siteSubset != QUDA_PARITY_SITE_SUBSET) 
	errorQuda("Only supports single parity fields");

      // faceBytes is the sum of all face sizes 
      size_t faceBytes = 0;
      
      // nbytes is the size in bytes of each face
      size_t nbytes[QUDA_MAX_DIM];
      
      int Ndof;
      if (typeid(*this) == typeid(cudaColorSpinorField)) {
	cudaColorSpinorField &csField = static_cast<cudaColorSpinorField&>(*this);
	Ndof = 2 * csField.Nspin() * csField.Ncolor();
	Ndof /= csField.Nspin() == 4 ? 2 : 1;
      } else { // FIXME - generalize for all field types
	errorQuda("Not supported field type in createComms");
	Ndof = 0;
      }

      for (int i=0; i<nDimComms; i++) {
	nbytes[i] = maxNface*surfaceCB[i]*Ndof*precision;
	if (precision == QUDA_HALF_PRECISION) nbytes[i] += maxNface*surfaceCB[i]*sizeof(float);
	if (siteSubset == QUDA_PARITY_SITE_SUBSET && i==0) nbytes[i] /= 2;
	if (!commDimPartitioned(i)) continue;
	faceBytes += 2*nbytes[i];
      }
      
      // use static pinned memory for face buffers
      resizeBufferPinned(2*faceBytes);
      my_face = bufferPinned;
      from_face = static_cast<char*>(bufferPinned) + faceBytes;
      
      // assign pointers for each face - it's ok to alias for different Nface parameters
      size_t offset = 0;
      for (int i=0; i<nDimComms; i++) {
	if (!commDimPartitioned(i)) continue;
	
	my_back_face[i] = static_cast<char*>(my_face) + offset;
	from_back_face[i] = static_cast<char*>(from_face) + offset;
	offset += nbytes[i];
	
	my_fwd_face[i] = static_cast<char*>(my_face) + offset;
	from_fwd_face[i] = static_cast<char*>(from_face) + offset;
	offset += nbytes[i];
      }
      
      // create a different message handler for each direction and Nface
      mh_send_fwd = new MsgHandle**[maxNface];
      mh_send_back = new MsgHandle**[maxNface];
      mh_recv_fwd = new MsgHandle**[maxNface];
      mh_recv_back = new MsgHandle**[maxNface];
      for (int j=0; j<maxNface; j++) {
	mh_send_fwd[j] = new MsgHandle*[nDimComms];
	mh_send_back[j] = new MsgHandle*[nDimComms];
	mh_recv_fwd[j] = new MsgHandle*[nDimComms];
	mh_recv_back[j] = new MsgHandle*[nDimComms];
	for (int i=0; i<nDimComms; i++) {
	  size_t nbytes_Nface = (nbytes[i] / maxNface) * (j+1);
	  if (!commDimPartitioned(i)) continue;
	  mh_send_fwd[j][i] = comm_declare_send_relative(my_fwd_face[i], i, +1, nbytes_Nface);
	  mh_send_back[j][i] = comm_declare_send_relative(my_back_face[i], i, -1, nbytes_Nface);
	  mh_recv_fwd[j][i] = comm_declare_receive_relative(from_fwd_face[i], i, +1, nbytes_Nface);
	  mh_recv_back[j][i] = comm_declare_receive_relative(from_back_face[i], i, -1, nbytes_Nface);
	}
      }
      
      initComms = true;
    }
    checkCudaError();
  }
    
  void LatticeField::destroyComms() {
    for (int j=0; j<maxNface; j++) {
      for (int i=0; i<nDimComms; i++) {
	if (commDimPartitioned(i)) {
	  comm_free(mh_send_fwd[j][i]);
	  comm_free(mh_send_back[j][i]);
	  comm_free(mh_recv_fwd[j][i]);
	  comm_free(mh_recv_back[j][i]);
	}
      }
      delete []mh_recv_fwd[j];
      delete []mh_recv_back[j];
      delete []mh_send_fwd[j];
      delete []mh_send_back[j];
    }    
    delete []mh_recv_fwd;
    delete []mh_recv_back;
    delete []mh_send_fwd;
    delete []mh_send_back;

    for (int i=0; i<nDimComms; i++) {
      my_fwd_face[i] = NULL;
      my_back_face[i] = NULL;
      from_fwd_face[i] = NULL;
      from_back_face[i] = NULL;      
    }
    
    checkCudaError();    
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
