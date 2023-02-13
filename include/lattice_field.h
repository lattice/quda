#pragma once

#include <iostream>
#include <quda_internal.h>
#include <comm_quda.h>
#include <util_quda.h>
#include <object.h>
#include <quda_api.h>
#include <reference_wrapper_helper.h>

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

  enum class QudaOffsetCopyMode { COLLECT, DISPERSE };

  struct LatticeFieldParam {

    friend class LatticeField;

    /** Location of the field */
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;

  protected:
    /** Field precision */
    QudaPrecision precision = QUDA_INVALID_PRECISION;

    /** Ghost precision */
    QudaPrecision ghost_precision = QUDA_INVALID_PRECISION;

  public:
    /** Field precision */
    QudaPrecision Precision() const { return precision; }

    /** Ghost precision */
    QudaPrecision GhostPrecision() const { return ghost_precision; }

    /** indicate if the param has been initialized (created with a non trivial constructor) */
    bool init = false;

    /** Number of field dimensions */
    int nDim = 4;

    /** Array storing the length of dimension */
    lat_dim_t x = {};

    int pad = 0;

    QudaSiteSubset siteSubset = QUDA_INVALID_SITE_SUBSET;

    QudaMemoryType mem_type = QUDA_MEMORY_DEVICE;

    /** The type of ghost exchange to be done with this field */
    QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_PAD;

    /** The extended field radius (if applicable) */
    lat_dim_t r = {};

    /** For fixed-point fields that need a global scaling factor */
    double scale = 1.0;

    /**
       @brief Default constructor for LatticeFieldParam
    */
    LatticeFieldParam() = default;

    /**
       @brief Constructor for creating a LatticeFieldParam from a set of parameters
       @param[in] nDim Number of field dimensions
       @param[in] x Array of dimension lengths
       @param[in] pad Field padding
       @param[in] precision Field Precision
       @param[in] ghostExchange Type of ghost exchange
    */
    LatticeFieldParam(int nDim, const lat_dim_t &x, int pad, QudaFieldLocation location, QudaPrecision precision,
                      QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_PAD) :
      location(location),
      precision(precision),
      ghost_precision(precision),
      init(true),
      nDim(nDim),
      pad(pad),
      siteSubset(QUDA_FULL_SITE_SUBSET),
      mem_type(QUDA_MEMORY_DEVICE),
      ghostExchange(ghostExchange),
      scale(1.0)
    {
      if (nDim > QUDA_MAX_DIM) errorQuda("Number of dimensions too great");
      for (int i = 0; i < QUDA_MAX_DIM; i++) {
        this->x[i] = i < nDim ? x[i] : 0;
        this->r[i] = 0;
      }
    }
    
    /**
       @brief Constructor for creating a LatticeFieldParam from a
       QudaGaugeParam.  Used for wrapping around a CPU reference
       field.
       @param[in] param Contains the metadata for filling out the LatticeFieldParam
    */
    LatticeFieldParam(const QudaGaugeParam &param) :
      location(QUDA_CPU_FIELD_LOCATION),
      precision(param.cpu_prec),
      ghost_precision(param.cpu_prec),
      init(true),
      nDim(4),
      pad(0),
      siteSubset(QUDA_FULL_SITE_SUBSET),
      mem_type(QUDA_MEMORY_DEVICE),
      ghostExchange(QUDA_GHOST_EXCHANGE_NO),
      scale(param.scale)
    {
      for (int i = 0; i < QUDA_MAX_DIM; i++) {
        this->x[i] = i < nDim ? param.X[i] : 0;
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

    /**
       @brief Create the field as specified by the param
       @param[in] Parameter struct
    */
    void create(const LatticeFieldParam &param);

    /**
       @brief Move the contents of a field to this
       @param[in,out] other Field we are moving from
    */
    void move(LatticeField &&other);

  protected:
    /** Lattice volume */
    size_t volume = 0;

    /** Checkerboarded volume */
    size_t volumeCB = 0;

    /** Local lattice volume */
    size_t localVolume = 0;

    /** Checkerboarded local volume */
    size_t localVolumeCB = 0;

    size_t stride = 0;
    int pad = 0;

    size_t total_bytes = 0;

    /** Number of field dimensions */
    int nDim = 0;

    /** Array storing the length of dimension */
    lat_dim_t x = {};

    /** The extended lattice radius (if applicable) */
    lat_dim_t r = {};

    /** Array storing the local dimensions (x - 2 * r) */
    lat_dim_t local_x = {};

    /** Array storing the surface size in each dimension */
    lat_dim_t surface = {};

    /** Array storing the checkerboarded surface size in each dimension */
    lat_dim_t surfaceCB = {};

    /** Array storing the local surface size in each dimension */
    lat_dim_t local_surface = {};

    /** Array storing the local surface size in each dimension */
    lat_dim_t local_surfaceCB = {};

    /** Location of the field */
    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;

    /** Precision of the field */
    QudaPrecision precision = QUDA_INVALID_PRECISION;

    /** Precision of the ghost */
    mutable QudaPrecision ghost_precision = QUDA_INVALID_PRECISION;

    /** Bool which is triggered if the ghost precision is reset */
    mutable bool ghost_precision_reset = false;

    /** For fixed-point fields that need a global scaling factor */
    double scale = 0.0;

    /** Whether the field is full or single parity */
    QudaSiteSubset siteSubset = QUDA_INVALID_SITE_SUBSET;

    /** Type of ghost exchange to perform */
    QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_INVALID;

    /** The number of dimensions we partition for communication */
    int nDimComms = 0;

    /*
       The need for persistent message handlers (for GPUDirect support)
       means that we allocate different message handlers for each number of
       faces we can send.
    */

    /**
       Double buffered static GPU halo send buffer
    */
    inline static array<void *, 2> ghost_send_buffer_d = {};

    /**
       Double buffered static GPU halo receive buffer
     */
    inline static array<void *, 2> ghost_recv_buffer_d = {};

    /**
       Double buffered static pinned send buffers
    */
    inline static array<void *, 2> ghost_pinned_send_buffer_h = {};

    /**
       Double buffered static pinned recv buffers
    */
    inline static array<void *, 2> ghost_pinned_recv_buffer_h = {};

    /**
       Mapped version of pinned send buffers
    */
    inline static array<void *, 2> ghost_pinned_send_buffer_hd = {};

    /**
       Mapped version of pinned recv buffers
    */
    inline static array<void *, 2> ghost_pinned_recv_buffer_hd = {};

    /**
       Remove ghost pointer for sending to
    */
    inline static array_3d<void *, 2, QUDA_MAX_DIM, 2> ghost_remote_send_buffer_d;

    /**
       The current size of the static ghost allocation
    */
    inline static size_t ghostFaceBytes = 0;

    /**
       Whether the ghost buffers have been initialized
    */
    inline static bool initGhostFaceBuffer = false;

    /**
       Size in bytes of this ghost field
    */
    mutable size_t ghost_bytes = 0;

    /**
       Size in bytes of prior ghost allocation
    */
    mutable size_t ghost_bytes_old = 0;

    /**
       Size in bytes of the ghost in each dimension
    */
    mutable array<size_t, QUDA_MAX_DIM> ghost_face_bytes = {};

    /**
       Actual allocated size in bytes of the ghost in each dimension
    */
    mutable array<size_t, QUDA_MAX_DIM> ghost_face_bytes_aligned = {};

    /**
       Byte offsets to each ghost zone
    */
    mutable array_2d<size_t, QUDA_MAX_DIM, 2> ghost_offset = {};

    /**
       Pinned memory buffer used for sending messages
    */
    array<void *, 2> my_face_h = {};

    /**
       Mapped version of my_face_h
    */
    array<void *, 2> my_face_hd = {};

    /**
       Device memory buffer for sending messages
     */
    array<void *, 2> my_face_d = {};

    /**
       Local pointers to the pinned my_face buffer
    */
    array_3d<void *, 2, QUDA_MAX_DIM, 2> my_face_dim_dir_h = {};

    /**
       Local pointers to the mapped my_face buffer
    */
    array_3d<void *, 2, QUDA_MAX_DIM, 2> my_face_dim_dir_hd = {};

    /**
       Local pointers to the device ghost_send buffer
    */
    array_3d<void *, 2, QUDA_MAX_DIM, 2> my_face_dim_dir_d = {};

    /**
       Memory buffer used for receiving all messages
    */
    array<void *, 2> from_face_h = {};

    /**
       Mapped version of from_face_h
    */
    array<void *, 2> from_face_hd = {};

    /**
       Device memory buffer for receiving messages
     */
    array<void *, 2> from_face_d = {};

    /**
       Local pointers to the pinned from_face buffer
    */
    array_3d<void *, 2, QUDA_MAX_DIM, 2> from_face_dim_dir_h = {};

    /**
       Local pointers to the mapped from_face buffer
    */
    array_3d<void *, 2, QUDA_MAX_DIM, 2> from_face_dim_dir_hd = {};

    /**
       Local pointers to the device ghost_recv buffer
    */
    array_3d<void *, 2, QUDA_MAX_DIM, 2> from_face_dim_dir_d = {};

    /**
       Message handles for receiving
    */
    array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_recv = {};

    /**
       Message handles for sending
    */
    array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_send = {};

    /**
       Message handles for receiving
    */
    array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_recv_rdma = {};

    /**
       Message handles for sending
    */
    array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_send_rdma = {};

    /**
       Message handles for receiving
    */
    inline static array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_recv_p2p = {};

    /**
       Message handles for sending
    */
    inline static array_3d<MsgHandle *, 2, QUDA_MAX_DIM, 2> mh_send_p2p = {};

    /**
       Buffer used by peer-to-peer message handler
    */
    inline static array_3d<int, 2, QUDA_MAX_DIM, 2> buffer_send_p2p = {};

    /**
       Buffer used by peer-to-peer message handler
    */
    inline static array_3d<int, 2, QUDA_MAX_DIM, 2> buffer_recv_p2p = {};

    /**
       Local copy of event used for peer-to-peer synchronization
    */
    inline static array_3d<qudaEvent_t, 2, QUDA_MAX_DIM, 2> ipcCopyEvent = {};

    /**
       Remote copy of event used for peer-to-peer synchronization
    */
    inline static array_3d<qudaEvent_t, 2, QUDA_MAX_DIM, 2> ipcRemoteCopyEvent = {};

    /**
       Whether we have initialized communication for this field
    */
    bool initComms = false;

    /**
       Whether we have initialized peer-to-peer communication
    */
    inline static bool initIPCComms = false;

    /**
       Bool which is triggered if the ghost field is reset
    */
    inline static bool ghost_field_reset = false;

    /**
       Used as a label in the autotuner
    */
    std::string vol_string;

    /**
       Used as a label in the autotuner
    */
    std::string aux_string;

    /**
       Sets the vol_string for use in tuning
    */
    virtual void setTuningString();

    /**
       The type of allocation we are going to do for this field
    */
    QudaMemoryType mem_type = QUDA_MEMORY_INVALID;

    void precisionCheck()
    {
      switch (precision) {
      case QUDA_QUARTER_PRECISION:
      case QUDA_HALF_PRECISION:
      case QUDA_SINGLE_PRECISION:
      case QUDA_DOUBLE_PRECISION: break;
      default: errorQuda("Unknown precision %d", precision);
      }
    }

    mutable char *backup_h = nullptr;
    mutable char *backup_norm_h = nullptr;
    mutable bool backed_up = false;

  public:
    /**
       Static variable that is determined which ghost buffer we are using
     */
    inline static int bufferIndex = 0;

    /**
       @brief Default constructor
    */
    LatticeField() = default;

    /**
       @brief Copy constructor for creating a LatticeField from another LatticeField
       @param field Instance of LatticeField from which we are cloning
    */
    LatticeField(const LatticeField &field) noexcept;

    /**
       @brief Move constructor for creating a LatticeField from another LatticeField
       @param field Instance of LatticeField from which we are moving
    */
    LatticeField(LatticeField &&field) noexcept;

    /**
       @brief Constructor for creating a LatticeField from a LatticeFieldParam
       @param param Contains the metadata for creating the field
    */
    LatticeField(const LatticeFieldParam &param);

    /**
       @brief Destructor for LatticeField
    */
    virtual ~LatticeField();

    /**
       @brief Copy assignment operator
       @param[in] field Instance from which we are copying
       @return Reference to this field
     */
    LatticeField &operator=(const LatticeField &);

    /**
       @brief Move assignment operator
       @param[in] field Instance from which we are moving
       @return Reference to this field
     */
    LatticeField &operator=(LatticeField &&);

    /**
       @brief Fills the param with this field's meta data (used for
       creating a cloned field)
       @param[in] param The parameter we are filling
    */
    void fill(LatticeFieldParam &param) const;

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
    void createComms(bool no_comms_fill = false);

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
       Handle to local copy event used for peer-to-peer synchronization
    */
    const qudaEvent_t &getIPCCopyEvent(int dir, int dim) const;

    /**
       Handle to remote copy event used for peer-to-peer synchronization
    */
    const qudaEvent_t &getIPCRemoteCopyEvent(int dir, int dim) const;

    /**
       @return The dimension of the lattice 
    */
    int Ndim() const { return nDim; }
    
    /**
       @return The pointer to the lattice-dimension array
    */
    const auto &X() const { return x; }

    /**
       @return Extended field radius
    */
    const auto &R() const { return r; }

    /**
       @return Local checkboarded lattice dimensions
    */
    const auto &LocalX() const { return local_x; }

    /**
      @return The pointer to the **full** lattice-dimension array
    */
    virtual int full_dim(int d) const = 0;

    /**
       @return The full-field volume
    */
    size_t Volume() const { return volume; }

    /**
       @return The single-parity volume
    */
    size_t VolumeCB() const { return volumeCB; }

    /**
       @return The local full-field volume without any overlapping region
    */
    size_t LocalVolume() const { return localVolume; }

    /**
       @return The local single-parity volume without any overlapping region
    */
    size_t LocalVolumeCB() const { return localVolumeCB; }

    /**
       @param i The dimension of the requested surface 
       @return The single-parity surface of dimension i
    */
    const auto &SurfaceCB() const { return surfaceCB; }

    /**
       @param i The dimension of the requested surface 
       @return The single-parity surface of dimension i
    */
    int SurfaceCB(const int i) const { return surfaceCB[i]; }

    /**
       @return The single-parity local surface array
    */
    const auto &LocalSurfaceCB() const { return local_surfaceCB; }

    /**
       @param i The dimension of the requested local surface
       @return The single-parity local surface of dimension i
    */
    int LocalSurfaceCB(const int i) const { return local_surfaceCB[i]; }

    /**
       @return The single-parity stride of the field
    */
    size_t Stride() const { return stride; }

    /**
       @return The field padding
    */
    int Pad() const { return pad; }
    
    /**
       @return Type of ghost exchange
     */
    QudaGhostExchange GhostExchange() const { return ghostExchange; }

    /**
       @return The field precision
    */
    QudaPrecision Precision() const { return precision; }

    /**
       @return The ghost precision
    */
    QudaPrecision GhostPrecision() const { return ghost_precision; }

    /**
       @return The global scaling factor for a fixed-point field
    */
    double Scale() const { return scale; }

    /**
       @brief Set the scale factor for a fixed-point field
       @param[in] scale_ The new scale factor
    */
    void Scale(double scale_) { scale = scale_; }

    /**
       @return Field subset type
     */
    QudaSiteSubset SiteSubset() const { return siteSubset; }

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
    QudaFieldLocation Location() const { return location; }

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

    /**
       @brief Return pointer to the local pinned my_face buffer in a
       given direction and dimension
       @param[in] dir Direction we are requesting
       @param[in] dim Dimension we are requesting
       @return Pointer to pinned memory buffer
    */
    void *myFace_h(int dir, int dim) const;

    /**
       @brief Return pointer to the local mapped my_face buffer in a
       given direction and dimension
       @param[in] dir Direction we are requesting
       @param[in] dim Dimension we are requesting
       @return Pointer to pinned memory buffer
    */
    void *myFace_hd(int dir, int dim) const;

    /**
       @brief Return pointer to the device send buffer in a given
       direction and dimension
       @param[in] dir Direction we are requesting
       @param[in] dim Dimension we are requesting
       @return Pointer to pinned memory buffer
    */
    void *myFace_d(int dir, int dim) const;

    /**
       @brief Return base pointer to a remote device buffer for direct
       sending in a given direction and dimension.  Since this is a
       base pointer, one still needs to take care of offsetting to the
       correct point for each direction/dimension.
       @param[in] dir Direction we are requesting
       @param[in] dim Dimension we are requesting
       @return Pointer to remote memory buffer
    */
    void *remoteFace_d(int dir, int dim) const;

    /**
       @brief Return base pointer to the ghost recv buffer. Since this is a
       base pointer, one still needs to take care of offsetting to the
       correct point for each direction/dimension.
       @return Pointer to remote memory buffer
     */
    void *remoteFace_r() const;

    virtual void gather(int, const qudaStream_t &) { errorQuda("Not implemented"); }

    virtual void commsStart(int, const qudaStream_t &, bool, bool) { errorQuda("Not implemented"); }

    virtual int commsQuery(int, const qudaStream_t &, bool, bool)
    {
      errorQuda("Not implemented");
      return 0;
    }

    virtual void commsWait(int, const qudaStream_t &, bool, bool) { errorQuda("Not implemented"); }

    virtual void scatter(int, const qudaStream_t &) { errorQuda("Not implemented"); }

    /** Return the volume string used by the autotuner */
    auto VolString() const { return vol_string; }

    /** Return the aux string used by the autotuner */
    auto AuxString() const { return aux_string; }

    /** @brief Backs up the LatticeField */
    virtual void backup() const { errorQuda("Not implemented"); }

    /** @brief Restores the LatticeField */
    virtual void restore() const { errorQuda("Not implemented"); }

    /**
      @brief If managed memory and prefetch is enabled, prefetch
      all relevant memory fields to the current device or to the CPU.
      @param[in] mem_space Memory space we are prefetching to
    */
    virtual void prefetch(QudaFieldLocation, qudaStream_t = device::get_default_stream()) const { ; }

    virtual bool isNative() const = 0;

    /**
       @brief Return the number of bytes in the field allocation.
     */
    virtual size_t Bytes() const = 0;

    /**
      @brief Copy all contents of the field to a host buffer.
      @param[in] the host buffer to copy to.

      *** Currently `buffer` has to be a host pointer:
            passing in UVM or device pointer leads to undefined behavior. ***
    */
    virtual void copy_to_buffer(void *buffer) const = 0;

    /**
      @brief Copy all contents of the field from a host buffer to this field.
      @param[in] the host buffer to copy from.

      *** Currently `buffer` has to be a host pointer:
            passing in UVM or device pointer leads to undefined behavior. ***
    */
    virtual void copy_from_buffer(void *buffer) = 0;
  };

  /**
     @brief Helper function for determining if the location of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If location is unique return the location
   */
  template <typename T1, typename T2>
  inline QudaFieldLocation Location_(const char *func, const char *file, int line, const T1 &a_, const T2 &b_)
  {
    const unwrap_t<T1> &a(a_);
    const unwrap_t<T2> &b(b_);

    QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION;
    if (a.Location() == b.Location()) location = a.Location();
    else
      errorQuda("Locations %d %d do not match  (%s:%d in %s())", a.Location(), b.Location(), file, line, func);
    return location;
  }

  /**
     @brief Helper function for determining if the location of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check location on
     @return If location is unique return the location
   */
  template <typename T1, typename T2, typename... Args>
  inline QudaFieldLocation Location_(const char *func, const char *file, int line, const T1 &a, const T2 &b,
                                     const Args &...args)
  {
    return static_cast<QudaFieldLocation>(Location_(func,file,line,a,b) & Location_(func,file,line,a,args...));
  }

#define checkLocation(...) Location_(__func__, __FILE__, __LINE__, __VA_ARGS__)

  /**
     @brief Helper function for determining if the precision of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If precision is unique return the precision
   */
  template <typename T1, typename T2>
  inline QudaPrecision Precision_(const char *func, const char *file, int line, const T1 &a_, const T2 &b_)
  {
    const unwrap_t<T1> &a(a_);
    const unwrap_t<T2> &b(b_);
    QudaPrecision precision = QUDA_INVALID_PRECISION;
    if (a.Precision() == b.Precision()) precision = a.Precision();
    else
      errorQuda("Precisions %d %d do not match (%s:%d in %s())", a.Precision(), b.Precision(), file, line, func);
    return precision;
  }

  /**
     @brief Helper function for determining if the precision of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check precision on
     @return If precision is unique return the precision
   */
  template <typename T1, typename T2, typename... Args>
  inline QudaPrecision Precision_(const char *func, const char *file, int line, const T1 &a, const T2 &b,
                                  const Args &...args)
  {
    return static_cast<QudaPrecision>(Precision_(func,file,line,a,b) & Precision_(func,file,line,a,args...));
  }

#define checkPrecision(...) Precision_(__func__, __FILE__, __LINE__, __VA_ARGS__)

  /**
     @brief Helper function for determining if the field is in native order
     @param[in] a Input field
     @return true if field is in native order
   */
  template <typename T> inline bool Native_(const char *func, const char *file, int line, const T &a_)
  {
    const unwrap_t<T> &a(a_);
    if (!a.isNative()) errorQuda("Non-native field detected (%s:%d in %s())", file, line, func);
    return true;
  }

  /**
     @brief Helper function for determining if the fields are in native order
     @param[in] a Input field
     @param[in] args List of additional fields to check
     @return true if all fields are in native order
   */
  template <typename T, typename... Args>
  inline bool Native_(const char *func, const char *file, int line, const T &a, const Args &...args)
  {
    return (Native_(func, file, line, a) && Native_(func, file, line, args...));
  }

#define checkNative(...) Native_(__func__, __FILE__, __LINE__, __VA_ARGS__)

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

  /**
     @brief Helper function for setting auxilary string
     @param[in] meta LatticeField used for querying field location
     @return String containing location and compilation type
   */
  inline const char *compile_type_str(const LatticeField &meta, QudaFieldLocation location_ = QUDA_INVALID_FIELD_LOCATION)
  {
    QudaFieldLocation location = (location_ == QUDA_INVALID_FIELD_LOCATION ? meta.Location() : location_);
#ifdef JITIFY
    return location == QUDA_CUDA_FIELD_LOCATION ? "GPU-jitify," : "CPU,";
#else
    return location == QUDA_CUDA_FIELD_LOCATION ? "GPU-offline," : "CPU,";
#endif
  }

  /**
     @brief Helper function for setting auxilary string
     @return String containing location and compilation type
   */
  inline const char *compile_type_str(QudaFieldLocation location = QUDA_INVALID_FIELD_LOCATION)
  {
#ifdef JITIFY
    return location == QUDA_CUDA_FIELD_LOCATION ? "GPU-jitify," : "CPU,";
#else
    return location == QUDA_CUDA_FIELD_LOCATION ? "GPU-offline," : "CPU,";
#endif
  }

} // namespace quda
