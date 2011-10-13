#include <quda_internal.h>
#include <lattice_field.h>

LatticeField::LatticeField(const LatticeFieldParam &param, const QudaFieldLocation &location)
  : volume(1), pad(param.pad), total_bytes(0), nDim(param.nDim),
    precision(param.precision), location(location), verbosity(param.verbosity) 
{
  if (location == QUDA_CPU_FIELD_LOCATION) {
    if (precision == QUDA_HALF_PRECISION) errorQuda("CPU fields do not support half precision");
    if (pad != 0) errorQuda("CPU fields do not support non-zero padding");
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
    if (a.surface[i] != surface[i]) errorQuda("surface %d does not match %d %d", i, surface[i], a.surface[i]);
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

