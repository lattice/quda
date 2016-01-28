#ifndef _LATTICE_FIELD_H
#define _LATTICE_FIELD_H

#include <quda.h>
#include <iostream>
#include <comm_quda.h>
#include <map>

/**
 * @file lattice_field.h
 *
 * @section DESCRIPTION 
 *
 * LatticeField is an abstract base clase for all Field objects.
 */

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
    
    /**
       Constructor for creating a LatticeField from a QudaGaugeParam
       @param param Contains the metadate for creating the
       LatticeField
    */
    LatticeFieldParam(const QudaGaugeParam &param) 
    : nDim(4), pad(0), precision(param.cpu_prec), siteSubset(QUDA_FULL_SITE_SUBSET) {
      for (int i=0; i<nDim; i++) this->x[i] = param.X[i];
    }
  };

  std::ostream& operator<<(std::ostream& output, const LatticeFieldParam& param);

  class LatticeField : public Object {

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

    /**
       The precision of the field 
    */
    QudaPrecision precision;
    
    /** Whether the field is full or single parity */
    QudaSiteSubset siteSubset;

    /** Pinned-memory buffer that is used by all derived classes */
    static void *bufferPinned[2]; 

    /** Whether the pinned-memory buffer has already been initialized or not */
    static bool bufferPinnedInit[2];

    /** The size in bytes of pinned-memory buffer */
    static size_t bufferPinnedBytes[2];

    /** Resize the pinned-memory buffer */
    void resizeBufferPinned(size_t bytes, const int index=0) const;

    /** Keep track of resizes to the pinned memory buffers */
    static size_t bufferPinnedResizeCount;

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
    void *my_face[2];
    void *my_fwd_face[2][QUDA_MAX_DIM];
    void *my_back_face[2][QUDA_MAX_DIM];

    /** Memory buffer used for sending all messages (regardless of Nface) */
    void *from_face[2];
    void *from_back_face[2][QUDA_MAX_DIM];
    void *from_fwd_face[2][QUDA_MAX_DIM];
    
    /** Message handles for receiving from forwards */
    MsgHandle ***mh_recv_fwd[2];

    /** Message handles for receiving from backwards */
    MsgHandle ***mh_recv_back[2];

    /** Message handles for sending forwards */
    MsgHandle ***mh_send_fwd[2];

    /** Message handles for sending backwards */
    MsgHandle ***mh_send_back[2];
    
    /** Used as a label in the autotuner */
    char vol_string[TuneKey::volume_n];
    
    /** Sets the vol_string for use in tuning */
    virtual void setTuningString();

    /** Cache of inactive pinned-memory allocations.  We cache pinned
    memory allocations so that fields can reuse these with minimal
    overhead.*/
    static std::multimap<size_t, void *> pinnedCache;

    /** Sizes of active pinned-memory allocations.  For convenience,
     we keep track of the sizes of active allocations (i.e., those not
     in the cache). */
    static std::map<void *, size_t> pinnedSize;

    /**
       Allocate pinned-memory.  If free pre-existing allocation exists
       reuse this.
       @param bytes Size of allocation
       @return Pointer to allocated memory
     */
    void *allocatePinned(size_t nbytes) const;

    /**
       Virtual free of pinned-memory allocation.
       @param ptr Pointer to be (virtually) freed
     */
    void freePinned(void *ptr) const;

  public:

    /**
       Constructor for creating a LatticeField from a LatticeFieldParam
       @param param Contains the metadata for creating the LatticeField
    */
    LatticeField(const LatticeFieldParam &param);

    /**
       Destructor for LatticeField
    */
    virtual ~LatticeField();
    
    /** 
	Free the pinned-memory buffer 
    */
    static void freeBuffer(int index=0);

    /**
       Free all outstanding pinned-memory allocations.
     */
    static void flushPinnedCache();

    /**
       @return The dimension of the lattice 
    */
    int Ndim() const { return nDim; }
    
    /**
       @return The pointer to the lattice-dimension array
    */
    const int* X() const { return x; }
    
    /**
       @return The full-field volume
    */
    int Volume() const { return volume; }
    
    /**
       @return The single-parity volume
    */
    int VolumeCB() const { return volumeCB; }
    
    /**
       @param i The dimension of the requested surface 
       @return The single-parity surface of dimension i
    */
    const int* SurfaceCB() const { return surfaceCB; }
    
    /**
       @param i The dimension of the requested surface 
       @return The single-parity surface of dimension i
    */
    int SurfaceCB(const int i) const { return surfaceCB[i]; }
    
    /**
       @return The single-parity stride of the field     
    */
    int Stride() const { return stride; }
    
    /**
       @return The field padding
    */
    int Pad() const { return pad; }
    
    /**
       @return The field precision
    */
    QudaPrecision Precision() const { return precision; }

    /**
       @return The vector storage length used for native fields , 2
       for Float2, 4 for Float4
     */
    int Nvec() const;

    /**
       @return The location of the field
    */
    QudaFieldLocation Location() const;

    /**
       @return The total storage allocated
    */
    size_t GBytes() const { return total_bytes / (1<<30); }
    
    /**
       Check that the metadata of *this and a are compatible
       @param a The LatticeField to which we are comparing
    */
    void checkField(const LatticeField &a) const;
    
    /**
       Read in the field specified by filenemae
       @param filename The name of the file to read
    */
    virtual void read(char *filename);
    
    /**
       Write the field in the file specified by filename
       @param filename The name of the file to write
    */
    virtual void write(char *filename);
    
    virtual void pack(int nFace, int parity, int dagger, cudaStream_t *stream_p, bool zeroCopyPack,
		      double a=0, double b=0)
    { errorQuda("Not implemented"); }

    virtual void gather(int nFace, int dagger, int dir, cudaStream_t *stream_p=NULL)
    { errorQuda("Not implemented"); }

    virtual void commsStart(int nFace, int dir, int dagger=0)
    { errorQuda("Not implemented"); }

    virtual int commsQuery(int nFace, int dir, int dagger=0)
    { errorQuda("Not implemented"); return 0; }

    virtual void scatter(int nFace, int dagger, int dir)
    { errorQuda("Not implemented"); }

    /** Return the volume string used by the autotuner */
    const char *VolString() const { return vol_string; }
  };
  
  /**
     Helper function for determining if the location of the fields is the same.
     @param a Input field
     @param b Input field
     @return If location is unique return the location
   */
  inline QudaFieldLocation Location(const LatticeField &a, const LatticeField &b) {
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;
    if (a.Location() == b.Location()) location = a.Location();
    else errorQuda("Locations do not match");
    return location;
  }

  /**
     Helper function for determining if the location of the fields is the same.
     @param a Input field
     @param b Input field
     @param c Input field
     @return If location is unique return the location
   */
  inline QudaFieldLocation Location(const LatticeField &a, const LatticeField &b, 
				    const LatticeField &c) {
    return static_cast<QudaFieldLocation>(Location(a,b) & Location(b,c));
  }

  /**
     Helper function for determining if the location of the fields is the same.
     @param a Input field
     @param b Input field
     @param c Input field
     @param d Input field
     @return If location is unique return the location
   */
  inline QudaFieldLocation Location(const LatticeField &a, const LatticeField &b,
				    const LatticeField &c, const LatticeField &d) {
    return static_cast<QudaFieldLocation>(Location(a,b) & Location(a,c) & Location(a,d));
  }

  /**
     Helper function for determining if the location of the fields is the same.
     @param a Input field
     @param b Input field
     @param c Input field
     @param d Input field
     @param e Input field
     @return If location is unique return the location
   */
  inline QudaFieldLocation Location(const LatticeField &a, const LatticeField &b, 
				    const LatticeField &c, const LatticeField &d, 
				    const LatticeField &e) {
    return static_cast<QudaFieldLocation>(Location(a,b) & Location(a,c) & Location(a,d) & Location(a,e));
  }


} // namespace quda

#endif // _LATTICE_FIELD_H
