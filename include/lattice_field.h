#ifndef _LATTICE_FIELD_H
#define _LATTICE_FIELD_H

#include <quda.h>
#include <iostream>
#include <comm_quda.h>

namespace quda {

  /** The maximum number of faces that can be exchanged */
  const int maxNface = 3;
  
  // LatticeField is an abstract base clase for all Field objects.

  // Forward declaration of all children
  class ColorSpinorField;
  class cudaColorSpinorField;
  class cpuColorSpinorField;
  
  class EigValueSet;
  class cudaEigValueSet;
  class cpuEigValueSet;

  class EigVecSet;
  class cpuEigVecSet;
  class cudaEigVecSet;

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
    QudaSiteSubset siteSubset;
  
    LatticeFieldParam() 
    : nDim(0), pad(0), precision(QUDA_INVALID_PRECISION), siteSubset(QUDA_INVALID_SITE_SUBSET) {
      for (int i=0; i<nDim; i++) x[i] = 0; 
    }

    LatticeFieldParam(int nDim, const int *x, int pad, QudaPrecision precision)
    : nDim(nDim), pad(pad), precision(precision), siteSubset(QUDA_FULL_SITE_SUBSET) { 
      if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
      for (int i=0; i<nDim; i++) this->x[i] = x[i]; 
    }
    
    // constructor for creating a cpuGaugeField only
    LatticeFieldParam(const QudaGaugeParam &param) 
    : nDim(4), pad(0), precision(param.cpu_prec), siteSubset(QUDA_FULL_SITE_SUBSET) {
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

    /** The number field dimensions */
    int nDim;
    
    /** Array storing the length of dimension */
    int x[QUDA_MAX_DIM];

    int surface[QUDA_MAX_DIM];
    int surfaceCB[QUDA_MAX_DIM];

    /** The precision of the field */
    QudaPrecision precision;
    
    /** Whether the field is full or single parity */
    QudaSiteSubset siteSubset;

    /** Pinned-memory buffer that is used by all derived classes */
    static void *bufferPinned[2]; 

    /** Whether the pinned-memory buffer has already been initialized or not */
    static bool bufferPinnedInit[2];

    /** The size in bytes of pinned-memory buffer */
    static size_t bufferPinnedBytes[2];

    static int bufferIndex;

    /** Resize the pinned-memory buffer */
    void resizeBufferPinned(size_t bytes, const int index=0) const;

    /** Device-memory buffer that is used by all derived classes */
    static void *bufferDevice; 

    /** Whether the device-memory buffer has already been initialized or not */
    static bool bufferDeviceInit;

    /** The size in bytes of device-memory buffer */
    static size_t bufferDeviceBytes;

    /** Resize the device-memory buffer */
    void resizeBufferDevice(size_t bytes) const;

    // The below are additions for inter-GPU communication (merging FaceBuffer functionality)

    /** The number of dimensions we partition for communication */
    int nDimComms;

    /* 
       The need for persistent message handlers (for GPUDirect support)
       means that we allocate different message handlers for each number of
       faces we can send.
    */

    /** Memory buffer used for sending all messages (regardless of Nface) */
    void *my_face;
    void *my_fwd_face[QUDA_MAX_DIM];
    void *my_back_face[QUDA_MAX_DIM];

    /** Memory buffer used for sending all messages (regardless of Nface) */
    void *from_face;
    void *from_back_face[QUDA_MAX_DIM];
    void *from_fwd_face[QUDA_MAX_DIM];
    
    /** Message handles for receiving from forwards */
    MsgHandle ***mh_recv_fwd;

    /** Message handles for receiving from backwards */
    MsgHandle ***mh_recv_back;

    /** Message handles for sending forwards */
    MsgHandle ***mh_send_fwd;

    /** Message handles for sending backwards */
    MsgHandle ***mh_send_back;
    
  public:
    LatticeField(const LatticeFieldParam &param);
    virtual ~LatticeField();

    /** Free the pinned-memory buffer */
    static void freeBuffer(const int index=0);

    int Ndim() const { return nDim; }
    const int* X() const { return x; }
    int Volume() const { return volume; }
    int VolumeCB() const { return volumeCB; }
    const int* SurfaceCB() const { return surfaceCB; }
    int SurfaceCB(const int i) const { return surfaceCB[i]; }
    int Stride() const { return stride; }
    int Pad() const { return pad; }

    /**
       @return The vector storage length used for native fields , 2
       for Float2, 4 for Float4
     */
    int Nvec() const;

    QudaPrecision Precision() const { return precision; }
    QudaFieldLocation Location() const;
    size_t GBytes() const { return total_bytes / (1<<30); } // returns total storage allocated

    void checkField(const LatticeField &);

    virtual void pack(int nFace, int parity, int dagger, cudaStream_t *stream_p, bool zeroCopyPack,
		      double a=0, double b=0)
    { errorQuda("Not implemented"); }

    virtual void gather(int nFace, int dagger, int dir)
    { errorQuda("Not implemented"); }

    virtual void commsStart(int nFace, int dir, int dagger=0)
    { errorQuda("Not implemented"); }

    virtual int commsQuery(int nFace, int dir, int dagger=0)
    { errorQuda("Not implemented"); return 0; }

    virtual void scatter(int nFace, int dagger, int dir)
    { errorQuda("Not implemented"); }

  };

} // namespace quda

#endif // _LATTICE_FIELD_H
