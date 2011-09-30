#include <quda_internal.h>
#include <lattice_field.h>

void LatticeField::checkField(const LatticeField &a) {
  if (a.volume != volume) errorQuda("Volume does not match %d %d", volume, a.volume);
  if (a.volumeCB != volumeCB) errorQuda("VolumeCB does not match %d %d", volumeCB, a.volumeCB);
  if (a.nDim != nDim) errorQuda("nDim does not match %d %d", nDim, a.nDim);
  for (int i=0; i<nDim; i++) {
    if (a.x[i] != x[i]) errorQuda("x[%d] does not match %d %d", i, x[i], a.x[i]);
    if (a.surface[i] != surface[i]) errorQuda("surface does not match %d %d", i, surface[i], a.surface[i]);
    if (a.surfaceCB[i] != surfaceCB[i]) errorQuda("surfaceCB does not match %d %d", i, surfaceCB[i], a.surfaceCB[i]);  
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

