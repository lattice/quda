#include <typeinfo>
#include <quda_internal.h>
#include <lattice_field.h>
#include <gauge_field.h>
#include <clover_field.h>

LatticeField::LatticeField(const LatticeFieldParam &param)
  : volume(1), pad(param.pad), total_bytes(0), nDim(param.nDim), precision(param.precision)
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
  volumeCB = volume / 2;
  stride = volumeCB + pad;
  
  for (int i=0; i<nDim; i++) surfaceCB[i] = surface[i] / 2;
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

