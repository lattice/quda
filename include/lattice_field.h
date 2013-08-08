#ifndef _LATTICE_FIELD_H
#define _LATTICE_FIELD_H

#include <quda.h>
#include <iostream>

namespace quda {

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

    LatticeFieldParam() 
    : nDim(0), pad(0), precision(QUDA_INVALID_PRECISION) {
      for (int i=0; i<nDim; i++) x[i] = 0; 
    }

  LatticeFieldParam(int nDim, const int *x, int pad, QudaPrecision precision)
    : nDim(nDim), pad(pad), precision(precision) { 
      if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
      for (int i=0; i<nDim; i++) this->x[i] = x[i]; 
    }
    
    // constructor for creating a cpuGaugeField only
    LatticeFieldParam(const QudaGaugeParam &param) 
    : nDim(4), pad(0), precision(param.cpu_prec) {
      for (int i=0; i<nDim; i++) this->x[i] = param.X[i];
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

    /** The precision of the field */
    QudaPrecision precision;

    /** Pinned-memory buffer that is used by all derived classes */
    static void *bufferPinned; 

    /** Whether the pinned-memory buffer has already been initialized or not */
    static bool bufferPinnedInit;

    /** The size in bytes of pinned-memory buffer */
    static size_t bufferPinnedBytes;

    /** Resize the pinned-memory buffer */
    void resizeBufferPinned(size_t bytes) const;

    /** Device-memory buffer that is used by all derived classes */
    static void *bufferDevice; 

    /** Whether the device-memory buffer has already been initialized or not */
    static bool bufferDeviceInit;

    /** The size in bytes of device-memory buffer */
    static size_t bufferDeviceBytes;

    /** Resize the device-memory buffer */
    void resizeBufferDevice(size_t bytes) const;

  public:
    LatticeField(const LatticeFieldParam &param);
    virtual ~LatticeField() { ; }

    /** Free the pinned-memory buffer */
    static void freeBuffer();

    int Ndim() const { return nDim; }
    const int* X() const { return x; }
    int Volume() const { return volume; }
    int VolumeCB() const { return volumeCB; }
    const int* SurfaceCB() const { return surfaceCB; }
    int SurfaceCB(const int i) const { return surfaceCB[i]; }
    int Stride() const { return stride; }
    int Pad() const { return pad; }

    QudaPrecision Precision() const { return precision; }
    QudaFieldLocation Location() const;
    size_t GBytes() const { return total_bytes / (1<<30); } // returns total storage allocated

    void checkField(const LatticeField &);
  };

} // namespace quda

#endif // _LATTICE_FIELD_H
