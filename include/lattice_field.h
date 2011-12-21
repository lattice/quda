#ifndef _LATTICE_FIELD_H
#define _LATTICE_FIELD_H

#include <quda.h>
#include <iostream>

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

  LatticeFieldParam() { ; }

  // constructor for creating a cpuGaugeField only
  LatticeFieldParam(const QudaGaugeParam &param) : nDim(4), pad(0), 
    precision(param.cpu_prec), verbosity(QUDA_SILENT)  {
    for (int i=0; i<nDim; i++) x[i] = param.X[i];
  }
};

std::ostream& operator<<(std::ostream& output, const LatticeFieldParam& param);

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
  LatticeField(const LatticeFieldParam &param, const QudaFieldLocation &Location);
  virtual ~LatticeField() { ; }

  const int* X() const { return x; }
  int Volume() const { return volume; }
  int VolumeCB() const { return volumeCB; }
  int SurfaceCB(const int i) const { return surfaceCB[i]; }
  int Stride() const { return stride; }

  QudaPrecision Precision() const { return precision; }
  QudaFieldLocation Location() const { return location; }
  QudaVerbosity Verbosity() const { return verbosity; }
  size_t GBytes() const { return total_bytes / (1<<30); } // returns total storage allocated

  void checkField(const LatticeField &);
};

#endif // _LATTICE_FIELD_H
