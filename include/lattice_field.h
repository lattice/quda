#ifndef _LATTICE_FIELD_H
#define _LATTICE_FIELD_H

// LatticeField is an abstract base clase for all Field objects.

// Forward declaration of all children
class ColorSpinorField;
class cudaColorSpinorField;
class cpuColorSpinorField;

class GaugeField;
class cpuGaugeField;
class cudaGaugeField;

class CloverField;
class cudaCloverField;
class cpuCloverField;

struct LatticeFieldParam {
  int nDim;
  int x[QUDA_MAX_DIM];
  int pad;

  QudaPrecision precision;
  QudaVerbosity verbosity;
};

class LatticeField {

 protected:
  int volume; // lattice volume
  int volumeCB; // the checkboarded volume
  int stride;
  int pad;
  
  size_t total_bytes;

  int nDim;
  int x[QUDA_MAX_DIM];

  int surface[QUDA_MAX_DIM];
  int surfaceCB[QUDA_MAX_DIM];

  QudaPrecision precision;
  QudaFieldLocation location;
  QudaVerbosity verbosity;

 public:
 LatticeField(const LatticeFieldParam &param, const QudaFieldLocation &location)
    : volume(1), pad(param.pad), total_bytes(0), nDim(param.nDim), precision(param.precision), 
    location(location), verbosity(param.verbosity) 
    {
      if (location == QUDA_CPU_FIELD_LOCATION && precision == QUDA_HALF_PRECISION)
	errorQuda("CPU fields do not support half precision");
      
      for (int i=0; i<nDim; i++) {
	x[i] = param.x[i];
	volume *= param.x[i];
	surfaceCB[i] = 1;
	for (int j=0; j<nDim; j++) {
	  if (i==j) continue;
	  surface[i] *= param.x[i];
	}
      }
      volumeCB = volume / 2;
      stride = volumeCB + pad;
  
      for (int i=0; i<nDim; i++) surfaceCB[i] = surface[i] / 2;
    }

  virtual ~LatticeField() { ; }

  int Volume() const { return volume; }
  int VolumeCB() const { return volumeCB; }
  int SurfaceCB(const int i) const { return surfaceCB[i]; }

  QudaPrecision Precision() const { return precision; }
  QudaFieldLocation Location() const { return location; }
  QudaVerbosity Verbosity() const { return verbosity; }
  size_t GBytes() const { return total_bytes / (1<<30); } // returns total storage allocated
};

#endif // _LATTICE_FIELD_H
