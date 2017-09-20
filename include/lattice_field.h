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
  
  // LatticeField is an abstract base clase for all Field objects.

  // Forward declaration of all children
  class LatticeField;

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

    /** Number of field dimensions */
    int nDim;

    /** Array storing the length of dimension */
    int x[QUDA_MAX_DIM];

    int pad;

    QudaPrecision precision;
    QudaSiteSubset siteSubset;

    QudaMemoryType mem_type; 
 
    /** The type of ghost exchange to be done with this field */
    QudaGhostExchange ghostExchange;

    /** The extended field radius (if applicable) */
    int r[QUDA_MAX_DIM];

    /**
       @brief Default constructor for LatticeFieldParam
    */
    LatticeFieldParam()
    : nDim(4), pad(0), precision(QUDA_INVALID_PRECISION), siteSubset(QUDA_INVALID_SITE_SUBSET), mem_type(QUDA_MEMORY_DEVICE),
      ghostExchange(QUDA_GHOST_EXCHANGE_PAD)
    {
      for (int i=0; i<nDim; i++) {
	x[i] = 0;
	r[i] = 0;
      }
    }

    /**
       @brief Constructor for creating a LatticeFieldParam from a set of parameters
       @param[in] nDim Number of field dimensions
       @param[in] x Array of dimension lengths
       @param[in] pad Field padding
       @param[in] precision Field Precision
       @param[in] ghostExchange Type of ghost exchange
    */
    LatticeFieldParam(int nDim, const int *x, int pad, QudaPrecision precision,
		      QudaGhostExchange ghostExchange=QUDA_GHOST_EXCHANGE_PAD)
    : nDim(nDim), pad(pad), precision(precision), siteSubset(QUDA_FULL_SITE_SUBSET), mem_type(QUDA_MEMORY_DEVICE),
      ghostExchange(ghostExchange)
    {
      if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
      for (int i=0; i<nDim; i++) {
	this->x[i] = x[i];
	this->r[i] = 0;
      }
    }
    
    /**
       @brief Constructor for creating a LatticeFieldParam from a
       QudaGaugeParam.  Used for wrapping around a CPU reference
       field.
       @param[in] param Contains the metadata for filling out the LatticeFieldParam
    */
    LatticeFieldParam(const QudaGaugeParam &param) 
    : nDim(4), pad(0), precision(param.cpu_prec), siteSubset(QUDA_FULL_SITE_SUBSET), mem_type(QUDA_MEMORY_DEVICE),
      ghostExchange(QUDA_GHOST_EXCHANGE_NO)
    {
      for (int i=0; i<nDim; i++) {
	this->x[i] = param.X[i];
	this->r[i] = 0;
      }
    }

    /**
       @brief Contructor for creating LatticeFieldParam from a LatticeField
    */
    LatticeFieldParam(const LatticeField &field);
  };

  std::ostream& operator<<(std::ostream& output, const LatticeFieldParam& param);

  class LatticeField : public Object {

  protected:
    /** Lattice volume */
    int volume;

    /** Checkerboarded volume */
    int volumeCB;

    int stride;
    int pad;

    size_t total_bytes;

    /** Number of field dimensions */
    int nDim;
    
    /** Array storing the length of dimension */
    int x[QUDA_MAX_DIM];

    int surface[QUDA_MAX_DIM];
    int surfaceCB[QUDA_MAX_DIM];

    /** The extended lattice radius (if applicable) */
    int r[QUDA_MAX_DIM];

    /** Precision of the field */
    QudaPrecision precision;
    
    /** Whether the field is full or single parity */
    QudaSiteSubset siteSubset;

    /** Type of ghost exchange to perform */
    QudaGhostExchange ghostExchange;

    // The below are additions for inter-GPU communication (merging FaceBuffer functionality)

    /** The number of dimensions we partition for communication */
    int nDimComms;

    /* 
       The need for persistent message handlers (for GPUDirect support)
       means that we allocate different message handlers for each number of
       faces we can send.
    */

    /**
       Double buffered static GPU halo send buffer
    */
    static void *ghost_send_buffer_d[2];

    /**
       Double buffered static GPU halo receive buffer
     */
    static void *ghost_recv_buffer_d[2];

    /**
       Double buffered static pinned send/recv buffers
    */
    static void *ghost_pinned_buffer_h[2];

    /**
       Mapped version of ghost_pinned
    */
    static void *ghost_pinned_buffer_hd[2];

    /**
       Remove ghost pointer for sending to
    */
    static void *ghost_remote_send_buffer_d[2][QUDA_MAX_DIM][2];

    /**
       The current size of the static ghost allocation
    */
    static size_t ghostFaceBytes;

    /**
       Whether the ghost buffers have been initialized
    */
    static bool initGhostFaceBuffer;

    /**
       Size in bytes of this ghost field
    */
    mutable size_t ghost_bytes;

    /**
       Size in bytes of the ghost in each dimension
    */
    mutable size_t ghost_face_bytes[QUDA_MAX_DIM];

    /**
       Real-number offsets to each ghost zone
    */
    mutable int ghostOffset[QUDA_MAX_DIM][2];

    /**
       Real-number (in floats) offsets to each ghost zone for norm field
    */
    mutable int ghostNormOffset[QUDA_MAX_DIM][2];

    /** Pinned memory buffer used for sending all messages */
    void *my_face_h[2];
    /** Mapped version of my_face_h */
    void *my_face_hd[2];

    /** Local pointers to the pinned my_face buffer */
    void *my_face_dim_dir_h[2][QUDA_MAX_DIM][2];

    /** Local pointers to the mapped my_face buffer */
    void *my_face_dim_dir_hd[2][QUDA_MAX_DIM][2];

    /** Local pointers to the device ghost_send buffer */
    void *my_face_dim_dir_d[2][QUDA_MAX_DIM][2];

    /** Memory buffer used for receiving all messages */
    void *from_face_h[2];
    /** Mapped version of from_face_h */
    void *from_face_hd[2];

    /** Local pointers to the pinned from_face buffer */
    void *from_face_dim_dir_h[2][QUDA_MAX_DIM][2];

    /** Local pointers to the mapped from_face buffer */
    void *from_face_dim_dir_hd[2][QUDA_MAX_DIM][2];

    /** Local pointers to the device ghost_recv buffer */
    void *from_face_dim_dir_d[2][QUDA_MAX_DIM][2];
    
    /** Message handles for receiving from forwards */
    MsgHandle *mh_recv_fwd[2][QUDA_MAX_DIM];

    /** Message handles for receiving from backwards */
    MsgHandle *mh_recv_back[2][QUDA_MAX_DIM];

    /** Message handles for sending forwards */
    MsgHandle *mh_send_fwd[2][QUDA_MAX_DIM];

    /** Message handles for sending backwards */
    MsgHandle *mh_send_back[2][QUDA_MAX_DIM];

    /** Message handles for rdma receiving from forwards */
    MsgHandle *mh_recv_rdma_fwd[2][QUDA_MAX_DIM];

    /** Message handles for rdma receiving from backwards */
    MsgHandle *mh_recv_rdma_back[2][QUDA_MAX_DIM];

    /** Message handles for rdma sending to forwards */
    MsgHandle *mh_send_rdma_fwd[2][QUDA_MAX_DIM];

    /** Message handles for rdma sending to backwards */
    MsgHandle *mh_send_rdma_back[2][QUDA_MAX_DIM];

    /** Peer-to-peer message handler for signaling event posting */
    static MsgHandle* mh_send_p2p_fwd[2][QUDA_MAX_DIM];

    /** Peer-to-peer message handler for signaling event posting */
    static MsgHandle* mh_send_p2p_back[2][QUDA_MAX_DIM];

    /** Peer-to-peer message handler for signaling event posting */
    static MsgHandle* mh_recv_p2p_fwd[2][QUDA_MAX_DIM];

    /** Peer-to-peer message handler for signaling event posting */
    static MsgHandle* mh_recv_p2p_back[2][QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_send_p2p_fwd[2][QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_recv_p2p_fwd[2][QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_send_p2p_back[2][QUDA_MAX_DIM];

    /** Buffer used by peer-to-peer message handler */
    static int buffer_recv_p2p_back[2][QUDA_MAX_DIM];

    /** Local copy of event used for peer-to-peer synchronization */
    static cudaEvent_t ipcCopyEvent[2][2][QUDA_MAX_DIM];

    /** Remote copy of event used for peer-to-peer synchronization */
    static cudaEvent_t ipcRemoteCopyEvent[2][2][QUDA_MAX_DIM];

    /** Whether we have initialized communication for this field */
    bool initComms;

    /** Whether we have initialized peer-to-peer communication */
    static bool initIPCComms;

    /** Used as a label in the autotuner */
    char vol_string[TuneKey::volume_n];
    
    /** Sets the vol_string for use in tuning */
    virtual void setTuningString();

    /** The type of allocation we are going to do for this field */
    QudaMemoryType mem_type;

    mutable char *backup_h;
    mutable char *backup_norm_h;
    mutable bool backed_up;

  public:

    /**
       Constructor for creating a LatticeField from a LatticeFieldParam
       @param param Contains the metadata for creating the LatticeField
    */
    LatticeField(const LatticeFieldParam &param);

    /**
       Constructor for creating a LatticeField from another LatticeField
       @param field Instance of LatticeField from which we are
       inheriting metadata
    */
    LatticeField(const LatticeField &field);

    /**
       Destructor for LatticeField
    */
    virtual ~LatticeField();
    
    /**
       @brief Allocate the static ghost buffers
       @param[in] ghost_bytes Size of the ghost buffer to allocate
    */
    void allocateGhostBuffer(size_t ghost_bytes) const;

    /**
       @brief Free statically allocated ghost buffers
    */
    static void freeGhostBuffer(void);

    /**
       Create the communication handlers (both host and device)
       @param[in] no_comms_fill Whether to allocate halo buffers for
       dimensions that are not partitioned
    */
    void createComms(bool no_comms_fill=false);

    /**
       Destroy the communication handlers
    */
    void destroyComms();

    /**
       Create the inter-process communication handlers
    */
    void createIPCComms();

    /**
       Destroy the statically allocated inter-process communication handlers
    */
    static void destroyIPCComms();

    /**
       Helper function to determine if local-to-remote (send) peer-to-peer copy is complete
    */
    inline bool ipcCopyComplete(int dir, int dim);

    /**
       Helper function to determine if local-to-remote (receive) peer-to-peer copy is complete
    */
    inline bool ipcRemoteCopyComplete(int dir, int dim);

    /**
       Handle to remote copy event used for peer-to-peer synchronization
    */
    const cudaEvent_t& getIPCRemoteCopyEvent(int dir, int dim) const;

    /**
       Static variable that is determined which ghost buffer we are using
     */
    static int bufferIndex;

    /**
       Bool which is triggered if the ghost field is reset
    */
    static bool ghost_field_reset;

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
       @return Extended field radius
    */
    const int* R() const { return r; }

    /**
       @return Type of ghost exchange
     */
    QudaGhostExchange GhostExchange() const { return ghostExchange; }

    /**
       @return The field precision
    */
    QudaPrecision Precision() const { return precision; }

    /**
       @return Field subset type
     */
    virtual QudaSiteSubset SiteSubset() const { return siteSubset; }

    /**
       @return Mem type
     */
    virtual QudaMemoryType MemType() const { return mem_type; }

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
    
    virtual void gather(int nFace, int dagger, int dir, cudaStream_t *stream_p=NULL)
    { errorQuda("Not implemented"); }

    virtual void commsStart(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false)
    { errorQuda("Not implemented"); }

    virtual int commsQuery(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false)
    { errorQuda("Not implemented"); return 0; }

    virtual void commsWait(int nFace, int dir, int dagger=0, cudaStream_t *stream_p=NULL, bool gdr=false)
    { errorQuda("Not implemented"); }

    virtual void scatter(int nFace, int dagger, int dir)
    { errorQuda("Not implemented"); }

    /** Return the volume string used by the autotuner */
    inline const char *VolString() const { return vol_string; }

    /** @brief Backs up the LatticeField */
    virtual void backup() const { errorQuda("Not implemented"); }

    /** @brief Restores the cpuGaugeField */
    virtual void restore() { errorQuda("Not implemented"); }
  };
  
  /**
     @brief Helper function for determining if the location of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If location is unique return the location
   */
  inline QudaFieldLocation Location_(const char *func, const char *file, int line,
				     const LatticeField &a, const LatticeField &b) {
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;
    if (a.Location() == b.Location()) location = a.Location();
    else errorQuda("Locations %d %d do not match  (%s:%d in %s())\n",
		   a.Location(), b.Location(), file, line, func);
    return location;
  }

  /**
     @brief Helper function for determining if the location of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check location on
     @return If location is unique return the location
   */
  template <typename... Args>
  inline QudaFieldLocation Location_(const char *func, const char *file, int line,
				     const LatticeField &a, const LatticeField &b, const Args &... args) {
    return static_cast<QudaFieldLocation>(Location_(func,file,line,a,b) & Location_(func,file,line,a,args...));
  }

#define checkLocation(...)Location_(__func__, __FILE__, __LINE__, __VA_ARGS__)

  /**
     @brief Helper function for determining if the precision of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If precision is unique return the precision
   */
  inline QudaPrecision Precision_(const char *func, const char *file, int line,
				  const LatticeField &a, const LatticeField &b) {
    QudaPrecision precision = QUDA_INVALID_PRECISION;
    if (a.Precision() == b.Precision()) precision = a.Precision();
    else errorQuda("Precisions %d %d do not match (%s:%d in %s())\n",
		   a.Precision(), b.Precision(), file, line, func);
    return precision;
  }

  /**
     @brief Helper function for determining if the precision of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check precision on
     @return If precision is unique return the precision
   */
  template <typename... Args>
  inline QudaPrecision Precision_(const char *func, const char *file, int line,
				  const LatticeField &a, const LatticeField &b,
				  const Args &... args) {
    return static_cast<QudaPrecision>(Precision_(func,file,line,a,b) & Precision_(func,file,line,a,args...));
  }

#define checkPrecision(...) Precision_(__func__, __FILE__, __LINE__, __VA_ARGS__)

  /**
     @brief Return whether data is reordered on the CPU or GPU.  This can set
     at QUDA initialization using the environment variable
     QUDA_REORDER_LOCATION.
     @return Reorder location
  */
  QudaFieldLocation reorder_location();

  /**
     @brief Set whether data is reorderd on the CPU or GPU.  This can set at
     QUDA initialization using the environment variable
     QUDA_REORDER_LOCATION.
     @param reorder_location_ The location to set where data will be reordered
  */
  void reorder_location_set(QudaFieldLocation reorder_location_);

} // namespace quda

#endif // _LATTICE_FIELD_H
