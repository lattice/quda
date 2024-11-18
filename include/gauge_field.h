#pragma once

#include <quda_internal.h>
#include <quda.h>
#include <lattice_field.h>

#include <comm_key.h>

namespace quda {

  namespace gauge
  {

    inline bool isNative(QudaGaugeFieldOrder order, QudaPrecision precision, QudaReconstructType reconstruct)
    {
      if (precision == QUDA_DOUBLE_PRECISION) {
        if (order == QUDA_FLOAT2_GAUGE_ORDER) return true;
      } else if (precision == QUDA_SINGLE_PRECISION) {
        if (reconstruct == QUDA_RECONSTRUCT_NO || reconstruct == QUDA_RECONSTRUCT_10) {
          if (order == QUDA_FLOAT2_GAUGE_ORDER) return true;
        } else if (reconstruct == QUDA_RECONSTRUCT_12 || reconstruct == QUDA_RECONSTRUCT_13
                   || reconstruct == QUDA_RECONSTRUCT_8 || reconstruct == QUDA_RECONSTRUCT_9) {
          if (order == QUDA_FLOAT4_GAUGE_ORDER) return true;
        }
      } else if (precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION) {
        if (reconstruct == QUDA_RECONSTRUCT_NO || reconstruct == QUDA_RECONSTRUCT_10) {
          if (order == QUDA_FLOAT2_GAUGE_ORDER) return true;
        } else if (reconstruct == QUDA_RECONSTRUCT_12 || reconstruct == QUDA_RECONSTRUCT_13) {
          if (order == QUDA_FLOAT4_GAUGE_ORDER) return true;
        } else if (reconstruct == QUDA_RECONSTRUCT_8 || reconstruct == QUDA_RECONSTRUCT_9) {
          if (order == static_cast<QudaGaugeFieldOrder>(QUDA_ORDER_FP)) return true;
        }
      }
      return false;
    }

  } // namespace gauge

  struct GaugeFieldParam : public LatticeFieldParam {
    int nColor = 3;
    int nFace = 0;

    QudaGaugeFieldOrder order = QUDA_INVALID_GAUGE_ORDER;
    QudaGaugeFixed fixed = QUDA_GAUGE_FIXED_NO;
    QudaLinkType link_type = QUDA_WILSON_LINKS;
    QudaTboundary t_boundary = QUDA_INVALID_T_BOUNDARY;
    QudaReconstructType reconstruct = QUDA_RECONSTRUCT_NO;

    double anisotropy = 1.0;
    double tadpole = 1.0;
    GaugeField *field = nullptr; // pointer to a pre-allocated field
    void *gauge = nullptr;       // used when we use a reference to an external field

    QudaFieldCreate create = QUDA_REFERENCE_FIELD_CREATE; // used to determine the type of field created

    QudaFieldGeometry geometry = QUDA_VECTOR_GEOMETRY; // whether the field is a scalar, vector or tensor

    // whether we need to compute the fat link maxima
    // FIXME temporary flag until we have a kernel that can do this, then we just do this in copy()
    // always set to false, requires external override
    bool compute_fat_link_max = false;

    /** The staggered phase convention to use */
    QudaStaggeredPhase staggeredPhaseType = QUDA_STAGGERED_PHASE_NO;

    /** Whether the staggered phase factor has been applied */
    bool staggeredPhaseApplied = false;

    /** Imaginary chemical potential */
    double i_mu = 0.0;

    /** Offset into MILC site struct to the desired matrix field (only if gauge_order=MILC_SITE_GAUGE_ORDER) */
    size_t site_offset = 0;

    /** Size of MILC site struct (only if gauge_order=MILC_SITE_GAUGE_ORDER) */
    size_t site_size = 0;

    // Default constructor
    GaugeFieldParam(void *const h_gauge = nullptr) : gauge(h_gauge) { }

    GaugeFieldParam(const GaugeField &u);

    GaugeFieldParam(const lat_dim_t &x, QudaPrecision precision, QudaReconstructType reconstruct, int pad,
                    QudaFieldGeometry geometry, QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_PAD) :
      LatticeFieldParam(4, x, pad, QUDA_INVALID_FIELD_LOCATION, precision, ghostExchange),
      reconstruct(reconstruct),
      create(QUDA_NULL_FIELD_CREATE),
      geometry(geometry)
    {
    }

    GaugeFieldParam(const QudaGaugeParam &param, void *h_gauge = nullptr, QudaLinkType link_type_ = QUDA_INVALID_LINKS) :
      LatticeFieldParam(param),
      order(param.gauge_order),
      fixed(param.gauge_fix),
      link_type(link_type_ != QUDA_INVALID_LINKS ? link_type_ : param.type),
      t_boundary(link_type == QUDA_ASQTAD_MOM_LINKS ? QUDA_PERIODIC_T : param.t_boundary),
      // if we have momentum field and not using TIFR field, then we always have recon-10
      reconstruct(link_type == QUDA_ASQTAD_MOM_LINKS && order != QUDA_TIFR_GAUGE_ORDER
                      && order != QUDA_TIFR_PADDED_GAUGE_ORDER ?
                    QUDA_RECONSTRUCT_10 :
                    QUDA_RECONSTRUCT_NO),
      anisotropy(param.anisotropy),
      tadpole(param.tadpole_coeff),
      gauge(h_gauge),
      staggeredPhaseType(param.staggered_phase_type),
      staggeredPhaseApplied(param.staggered_phase_applied),
      i_mu(param.i_mu),
      site_offset(link_type == QUDA_ASQTAD_MOM_LINKS ? param.mom_offset : param.gauge_offset),
      site_size(param.site_size)
    {
      switch (link_type) {
      case QUDA_SU3_LINKS:
      case QUDA_GENERAL_LINKS:
      case QUDA_SMEARED_LINKS:
      case QUDA_MOMENTUM_LINKS: nFace = 1; break;
      case QUDA_THREE_LINKS: nFace = 3; break;
      default: errorQuda("Error: invalid link type(%d)\n", link_type);
      }
    }

    /**
       @brief Helper function for setting the precision and corresponding
       field order for QUDA internal fields.
       @param precision The precision to use
    */
    void setPrecision(QudaPrecision precision, bool force_native = false)
    {
      // is the current status in native field order?
      bool native = force_native ? true : gauge::isNative(order, this->precision, reconstruct);
      this->precision = precision;
      this->ghost_precision = precision;

      if (native) {
        if (precision == QUDA_DOUBLE_PRECISION || reconstruct == QUDA_RECONSTRUCT_NO
            || reconstruct == QUDA_RECONSTRUCT_10) {
          order = QUDA_FLOAT2_GAUGE_ORDER;
        } else if ((precision == QUDA_HALF_PRECISION || precision == QUDA_QUARTER_PRECISION)
                   && (reconstruct == QUDA_RECONSTRUCT_8 || reconstruct == QUDA_RECONSTRUCT_9)) {
          order = static_cast<QudaGaugeFieldOrder>(QUDA_ORDER_FP);
        } else {
          order = QUDA_FLOAT4_GAUGE_ORDER;
        }
      }
    }
  };

  std::ostream& operator<<(std::ostream& output, const GaugeFieldParam& param);
  std::ostream &operator<<(std::ostream &output, const GaugeField &param);

  class GaugeField : public LatticeField {

    friend std::ostream &operator<<(std::ostream &output, const GaugeField &param);

  private:
    /**
       @brief Create the field as specified by the param
       @param[in] Parameter struct
    */
    void create(const GaugeFieldParam &param);

    /**
       @brief Move the contents of a field to this
       @param[in,out] other Field we are moving from
    */
    void move(GaugeField &&other);

    /**
       @brief Fills the param with this field's meta data (used for
       creating a cloned field)
       @param[in] param The parameter we are filling
    */
    void fill(GaugeFieldParam &) const;

  protected:
    bool init = false;
    quda_ptr gauge = {};                 /** The gauge field allocation */
    array<quda_ptr, 8> gauge_array = {}; /** Array of pointers to each subset (e.g., QDP or QDPJITorder) */
    size_t bytes = 0;                    // bytes allocated per full field
    size_t phase_offset = 0;             // offset in bytes to gauge phases - useful to keep track of texture alignment
    size_t phase_bytes = 0;              // bytes needed to store the phases
    size_t length = 0;
    size_t real_length = 0;
    int nColor = 0;
    int nFace = 0;
    QudaFieldGeometry geometry = QUDA_INVALID_GEOMETRY; // whether the field is a scale, vector or tensor
    int site_dim = 0; // the dimensionality of each site (number of matrices per lattice site)

    QudaReconstructType reconstruct = QUDA_RECONSTRUCT_INVALID;
    int nInternal = 0; // number of degrees of freedom per link matrix
    QudaGaugeFieldOrder order = QUDA_INVALID_GAUGE_ORDER;
    QudaGaugeFixed fixed = QUDA_GAUGE_FIXED_INVALID;
    QudaLinkType link_type = QUDA_INVALID_LINKS;
    QudaTboundary t_boundary = QUDA_INVALID_T_BOUNDARY;

    double anisotropy = 0.0;
    double tadpole = 0.0;
    double fat_link_max = 0.0;

    mutable array<quda_ptr, 2 *QUDA_MAX_DIM> ghost
      = {}; // stores the ghost zone of the gauge field (non-native fields only)

    mutable array<int, QUDA_MAX_DIM> ghostFace = {}; // the size of each face

    /**
       The staggered phase convention to use
    */
    QudaStaggeredPhase staggeredPhaseType = QUDA_STAGGERED_PHASE_INVALID;

    /**
       Whether the staggered phase factor has been applied
    */
    bool staggeredPhaseApplied = false;

    /**
       Imaginary chemical potential
    */
    double i_mu = 0.0;

    /**
       Offset into MILC site struct to the desired matrix field (only if gauge_order=MILC_SITE_GAUGE_ORDER)
    */
    size_t site_offset = 0;

    /**
       Size of MILC site struct (only if gauge_order=MILC_SITE_GAUGE_ORDER)
    */
    size_t site_size = 0;

    /**
       @brief Exchange the buffers across all dimensions in a given direction
       @param[out] recv Receive buffer
       @param[in] send Send buffer
       @param[in] dir Direction in which we are sending (forwards OR backwards only)
    */
    void exchange(void **recv, void **send, QudaDirection dir) const;

    /**
       Compute the required extended ghost zone sizes and offsets
       @param[in] R Radius of the ghost zone
       @param[in] no_comms_fill If true we create a full halo
       regardless of partitioning
       @param[in] bidir Is this a bi-directional exchange - if not
       then we alias the fowards and backwards offsetss
    */
    void createGhostZone(const lat_dim_t &R, bool no_comms_fill, bool bidir = true) const;

    /**
       @brief Set the vol_string and aux_string for use in tuning
    */
    void setTuningString();

  public:
    /**
       @brief Initialize the padded region to 0
     */
    void zeroPad();

    /**
       @brief Default constructor
    */
    GaugeField() = default;

    /**
       @brief Copy constructor for creating a GaugeField from another GaugeField
       @param field Instance of GaugeField from which we are cloning
    */
    GaugeField(const GaugeField &field) noexcept;

    /**
       @brief Move constructor for creating a GaugeField from another GaugeField
       @param field Instance of GaugeField from which we are moving
    */
    GaugeField(GaugeField &&field) noexcept;

    /**
       @brief Constructor for creating a GaugeField from a GaugeFieldParam
       @param param Contains the metadata for creating the field
    */
    GaugeField(const GaugeFieldParam &param);

    /**
       @brief Copy assignment operator
       @param[in] field Instance from which we are copying
       @return Reference to this field
     */
    GaugeField &operator=(const GaugeField &field);

    /**
       @brief Move assignment operator
       @param[in] field Instance from which we are moving
       @return Reference to this field
     */
    GaugeField &operator=(GaugeField &&field);

    /**
       @brief Returns if the object is empty (not initialized)
       @return true if the object has not been allocated, otherwise false
    */
    bool empty() const { return !init; }

    /**
       @brief Create the communication handlers and buffers
       @param[in] R The thickness of the extended region in each dimension
       @param[in] no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
       @param[in] bidir Whether to allocate communication buffers to
       allow for simultaneous bi-directional exchange.  If false, then
       the forwards and backwards buffers will alias (saving memory).
    */
    void createComms(const lat_dim_t &R, bool no_comms_fill, bool bidir = true);

    /**
       @brief Allocate the ghost buffers
       @param[in] R The thickness of the extended region in each dimension
       @param[in] no_comms_fill Do local exchange to fill out the extended
       @param[in] bidir Is this a bi-directional exchange - if not
       then we alias the fowards and backwards offsetss
       region in non-partitioned dimensions
    */
    void allocateGhostBuffer(const lat_dim_t &R, bool no_comms_fill, bool bidir = true) const;

    /**
       @brief Start the receive communicators
       @param[in] dim The communication dimension
       @param[in] dir The communication direction (0=backwards, 1=forwards)
    */
    void recvStart(int dim, int dir);

    /**
       @brief Start the sending communicators
       @param[in] dim The communication dimension
       @param[in] dir The communication direction (0=backwards, 1=forwards)
       @param[in] stream_p Pointer to CUDA stream to post the
       communication in (if 0, then use null stream)
    */
    void sendStart(int dim, int dir, const qudaStream_t &stream_p);

    /**
       @brief Wait for communication to complete
       @param[in] dim The communication dimension
       @param[in] dir The communication direction (0=backwards, 1=forwards)
    */
    void commsComplete(int dim, int dir);

    /**
       @brief Exchange the ghost and store store in the padded region
       @param[in] link_direction Which links are we exchanging: this
       flag only applies to bi-directional coarse-link fields
     */
    void exchangeGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    /**
       @brief The opposite of exchangeGhost: take the ghost zone on x,
       send to node x-1, and inject back into the field
       @param[in] link_direction Which links are we injecting: this
       flag only applies to bi-directional coarse-link fields
     */
    void injectGhost(QudaLinkDirection link_direction = QUDA_LINK_BACKWARDS);

    size_t Length() const { return length; }
    int Ncolor() const { return nColor; }
    QudaReconstructType Reconstruct() const { return reconstruct; }
    QudaGaugeFieldOrder Order() const { return order; }
    double Anisotropy() const { return anisotropy; }
    double Tadpole() const { return tadpole; }
    QudaTboundary TBoundary() const { return t_boundary; }
    QudaLinkType LinkType() const { return link_type; }
    QudaGaugeFixed GaugeFixed() const { return fixed; }
    QudaGaugeFieldOrder FieldOrder() const { return order; }
    QudaFieldGeometry Geometry() const { return geometry; }
    QudaStaggeredPhase StaggeredPhase() const { return staggeredPhaseType; }
    bool StaggeredPhaseApplied() const { return staggeredPhaseApplied; }

    /**
     * Define the parameter type for this field.
     */
    using param_type = GaugeFieldParam;

    /**
       Apply the staggered phase factors to the gauge field.
       @param[in] phase The phase we will apply to the field.  If this
       is QUDA_STAGGERED_PHASE_INVALID, the default value, then apply
       the phase set internal to the field.
    */
    void applyStaggeredPhase(QudaStaggeredPhase phase=QUDA_STAGGERED_PHASE_INVALID);

    /**
       Remove the staggered phase factors from the gauge field.
    */
    void removeStaggeredPhase();

    /**
       Return the imaginary chemical potential applied to this field
    */
    double iMu() const { return i_mu; }

    const double& LinkMax() const { return fat_link_max; }
    int Nface() const { return nFace; }

    /**
       @brief This routine will populate the border / halo region of a
       gauge field that has been created using copyExtendedGauge.
       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const lat_dim_t &R, bool no_comms_fill = false);

    /**
       @brief This routine will populate the border / halo region
       of a gauge field that has been created using copyExtendedGauge.
       Overloaded variant that will start and stop a comms profile.
       @param R The thickness of the extended region in each dimension
       @param profile TimeProfile intance which will record the time taken
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const lat_dim_t &R, TimeProfile &profile, bool no_comms_fill = false);

    void checkField(const LatticeField &) const;

    /**
       This function returns true if the field is stored in an
       internal field order for the given precision.
    */
    bool isNative() const { return gauge::isNative(order, precision, reconstruct); }

    size_t Bytes() const { return bytes; }
    size_t PhaseBytes() const { return phase_bytes; }
    size_t PhaseOffset() const { return phase_offset; }

    size_t TotalBytes() const { return bytes; }

    /**
       @brief Helper function that returns true if the gauge order is an array of pointers
       @param[in] order The gauge order requested
       @return If the order is an array of pointers
     */
    constexpr bool is_pointer_array(QudaGaugeFieldOrder order) const
    {
      switch (order) {
      case QUDA_QDP_GAUGE_ORDER:
      case QUDA_QDPJIT_GAUGE_ORDER: return true;
      default: return false;
      }
    }

    /**
       @brief Return base pointer to the gauge field allocation.
       @tparam T Optional type to cast the pointer to (default is void*).
       @return Base pointer to the gauge field allocation
     */
    template <typename T = void *>
    std::enable_if_t<std::is_pointer_v<T> && !std::is_pointer_v<typename std::remove_pointer<T>::type>, T> data() const
    {
      if (is_pointer_array(order)) errorQuda("Non dim-array ordered field requested but order is %d", order);
      return static_cast<T>(gauge.data());
    }

    /**
       @brief Return base pointer to the gauge field allocation
       specified by the array index.  This is for geometry-array
       ordered fields, e.g., QDP or QDPJIT.

       @tparam T Optional type to cast the pointer to (default is void*)
       @param[in] d Dimension index when the allocation is an array type
       @return Base pointer to the gauge field allocation
     */
    template <typename T = void *> auto data(unsigned int d) const
    {
      static_assert(std::is_pointer_v<T> && !std::is_pointer_v<typename std::remove_pointer<T>::type>,
                    "data() requires a pointer cast type");
      if (d >= (unsigned)geometry) errorQuda("Invalid array index %d for geometry %d field", d, geometry);
      if (!is_pointer_array(order)) errorQuda("Dim-array ordered field requested but order is %d", order);
      return static_cast<T>(gauge_array[d].data());
    }

    void *raw_pointer() const
    {
      if (is_pointer_array(order)) {
        static void *data_array[8];
        for (int i = 0; i < site_dim; i++) data_array[i] = gauge_array[i].data();
        return data_array;
      } else {
        return gauge.data();
      }
    }

    /**
       @brief Return array of pointers to the per dimension gauge field allocation(s).
       @tparam T Optional type to cast the pointer to (default is
       void*).  this is for geometry-array ordered fields, e.g., QDP
       or QDPJIT.
       @return Array of pointers to the gauge field allocations
     */
    template <typename T = void *>
    std::enable_if_t<std::is_pointer_v<T> && !std::is_pointer_v<typename std::remove_pointer<T>::type>, array<T, QUDA_MAX_DIM>>
    data_array() const
    {
      if (!is_pointer_array(order)) errorQuda("Dim-array ordered field requested but order is %d", order);
      array<T, QUDA_MAX_DIM> u = {};
      for (auto d = 0; d < geometry; d++) u[d] = static_cast<T>(gauge_array[d].data());
      return u;
    }

    virtual int full_dim(int d) const { return x[d]; }

    auto &Ghost() const
    {
      if ( isNative() ) errorQuda("No ghost zone pointer for quda-native gauge fields");
      return ghost;
    }

    /**
       @return The offset into the struct to the start of the gauge
       field (only for order = QUDA_MILC_SITE_GAUGE_ORDER)
     */
    size_t SiteOffset() const { return site_offset; }

    /**
       @return The size of the struct into which the gauge
       field is packed (only for order = QUDA_MILC_SITE_GAUGE_ORDER)
     */
    size_t SiteSize() const { return site_size; }

    /**
       Set all field elements to zero
    */
    void zero();

    /**
     * Generic gauge field copy
     * @param[in] src Source from which we are copying
     */
    void copy(const GaugeField &src);

    /**
       @brief Compute the L1 norm of the field
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return L1 norm
     */
    double norm1(int dim = -1, bool fixed = false) const;

    /**
       @brief Compute the L2 norm squared of the field
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return L2 norm squared
     */
    double norm2(int dim = -1, bool fixed = false) const;

    /**
       @brief Compute the absolute maximum of the field (Linfinity norm)
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return Absolute maximum value
     */
    double abs_max(int dim = -1, bool fixed = false) const;

    /**
       @brief Compute the absolute minimum of the field
       @param[in] dim Which dimension we are taking the norm of (dim=-1 mean all dimensions)
       @return Absolute minimum value
     */
    double abs_min(int dim = -1, bool fixed = false) const;

    /**
       Compute checksum of this gauge field: this uses a XOR-based checksum method
       @param[in] mini Whether to compute a mini checksum or global checksum.
       A mini checksum only computes the checksum over a subset of the lattice
       sites and is to be used for online comparisons, e.g., checking
       a field has changed with a global update algorithm.
       @return checksum value
     */
    uint64_t checksum(bool mini = false) const;

    /**
       @brief Create the gauge field, with meta data specified in the
       parameter struct.
       @param param Parameter struct specifying the gauge field
       @return Pointer to allcoated gauge field
    */
    static GaugeField* Create(const GaugeFieldParam &param);

    /**
       @brief Create a field that aliases this field's storage.  The
       alias field can use a different precision than this field,
       though it cannot be greater.  This functionality is useful for
       the case where we have multiple temporaries in different
       precisions, but do not need them simultaneously.  Use this functionality with caution.
       @param[in] param Parameters for the alias field
    */
    GaugeField create_alias(const GaugeFieldParam &param = GaugeFieldParam());

    /**
      @brief If managed memory and prefetch is enabled, prefetch
      the gauge field and buffers to the CPU or the GPU
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in (default 0)
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream = device::get_default_stream()) const;

    /**
       @brief Backs up the GaugeField
    */
    void backup() const;

    /**
       @brief Restores the GaugeField
    */
    void restore() const;

    /**
      @brief Copy all contents of the field to a host buffer.
      @param[in] the host buffer to copy to.
    */
    void copy_to_buffer(void *buffer) const;

    /**
      @brief Copy all contents of the field from a host buffer to this field.
      @param[in] the host buffer to copy from.
    */
    void copy_from_buffer(void *buffer);

    /**
       @brief Check if two instances are compatible
       @param[in] a Input field
       @param[in] b Input field
       @return Return true if two fields are compatible
     */
    static bool are_compatible(const GaugeField &a, const GaugeField &b);

    /**
       @brief Check if two instances are weakly compatible (precision
       and order can differ)
       @param[in] a Input field
       @param[in] b Input field
       @return Return true if two fields are compatible
     */
    static bool are_compatible_weak(const GaugeField &a, const GaugeField &b);

    /**
     * @brief Helper function that returns the default tolerance used by SU(3) projection
     * @return The default tolerance, which is ~10x epsilon
     */
    double toleranceSU3() const
    {
      switch (precision) {
      case QUDA_DOUBLE_PRECISION: return 2e-15;
      case QUDA_SINGLE_PRECISION: return 1e-6;
      default: return 1e-6;
      }
    }

    /**
     * @brief Print the site data
     * @param[in] parity Parity index
     * @param[in] dim The dimension in which we are printing
     * @param[in] x_cb Checkerboard space-time index
     * @param[in] rank The rank we are requesting from (default is rank = 0)
     */
    void PrintMatrix(int dim, int parity, unsigned int x_cb, int rank = 0) const;

    friend struct GaugeFieldParam;
  };

  /**
     @brief Print the value of the field at the requested coordinates
     @param[in] a The field we are printing from
     @param[in] dim The dimension in which we are printing
     @param[in] parity Parity index
     @param[in] x_cb Checkerboard space-time index
     @param[in] rank The rank we are requesting from (default is rank = 0)
  */
  void genericPrintMatrix(const GaugeField &a, int dim, int parity, unsigned int x_cb, int rank = 0);

  /**
     @brief This is a debugging function, where we cast a gauge field
     into a spinor field so we can compute its L1 norm.
     @param u The gauge field that we want the norm of
     @return The L1 norm of the gauge field
  */
  double norm1(const GaugeField &u);

  /**
     @brief This is a debugging function, where we cast a gauge field
     into a spinor field so we can compute its L2 norm.
     @param u The gauge field that we want the norm of
     @return The L2 norm squared of the gauge field
  */
  double norm2(const GaugeField &u);

  /**
     @brief Scale the gauge field by the scalar a.
     @param[in] a scalar multiplier
     @param[in] u The gauge field we want to multiply
   */
  void ax(const double &a, GaugeField &u);

  /**
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in copy_gauge.cu.
     @param out The output field to which we are copying
     @param in The input field from which we are copying
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
     @param ghostOut The output ghost buffer (optional)
     @param ghostIn The input ghost buffer (optional)
     @param type The type of copy we doing (0 body and ghost else ghost only)
  */
  void copyGenericGauge(GaugeField &out, const GaugeField &in, QudaFieldLocation location, void *Out = 0, void *In = 0,
                        void **ghostOut = 0, void **ghostIn = 0, int type = 0);

  /**
    @brief This function is used for copying from a source gauge field to a destination gauge field
      with an offset.
    @param out The output field to which we are copying
    @param in The input field from which we are copying
    @param offset The offset for the larger field between out and in.
    @param pc_type Whether the field order uses 4d or 5d even-odd preconditioning.
 */
  void copyFieldOffset(GaugeField &out, const GaugeField &in, CommKey offset, QudaPCType pc_type);

  /**
     This function is used for copying the gauge field into an
     extended gauge field.  Defined in copy_extended_gauge.cu.
     @param out The extended output field to which we are copying
     @param in The input field from which we are copying
     @param location The location of where we are doing the copying (CPU or CUDA)
     @param Out The output buffer (optional)
     @param In The input buffer (optional)
  */
  void copyExtendedGauge(GaugeField &out, const GaugeField &in,
			 QudaFieldLocation location, void *Out=0, void *In=0);

  /**
     This function is used for creating an exteneded gauge field from the input,
     and copying the gauge field into the extended gauge field.  Defined in lib/gauge_field.cpp.
     @param in The input field from which we are extending
     @param R By how many do we want to extend the gauge field in each direction
     @param profile The `TimeProfile`
     @param redundant_comms
     @param recon The reconsturction type
     @return the pointer to the extended gauge field
  */
  GaugeField *createExtendedGauge(const GaugeField &in, const lat_dim_t &R, TimeProfile &profile = getProfile(),
                                  bool redundant_comms = false, QudaReconstructType recon = QUDA_RECONSTRUCT_INVALID);

  /**
     This function is used for creating an exteneded (cpu) gauge field from the input,
     and copying the gauge field into the extended gauge field.  Defined in lib/gauge_field.cpp.
     @param in The input field from which we are extending
     @param R By how many do we want to extend the gauge field in each direction
     @return the pointer to the extended gauge field
  */
  GaugeField *createExtendedGauge(void **gauge, QudaGaugeParam &gauge_param, const lat_dim_t &R);

  /**
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in extract_gauge_ghost.cu.
     @param u The gauge field from which we want to extract the ghost zone
     @param ghost The array where we want to pack the ghost zone into
     @param extract Where we are extracting into ghost or injecting from ghost
     @param offset By default we exchange the nDim site-vector of
     links in the first nDim dimensions; offset allows us to instead
     exchange the links in nDim+offset dimensions.  This is used to
     faciliate sending bi-directional links which is needed for the
     coarse links.
  */
  void extractGaugeGhost(const GaugeField &u, void **ghost, bool extract=true, int offset=0);

  /**
     This function is used for extracting the extended gauge ghost
     zone from a gauge field array.  Defined in
     extract_gauge_ghost_extended.cu.
     @param u The gauge field from which we want to extract/pack the ghost zone
     @param dim The dimension in which we are packing/unpacking
     @param R array holding the radius of the extended region
     @param ghost The array where we want to pack/unpack the ghost zone into/from
     @param extract Whether we are extracting into ghost or injecting from ghost
  */
  void extractExtendedGaugeGhost(const GaugeField &u, int dim, const lat_dim_t &R, void **ghost, bool extract);

  /**
     Apply the staggered phase factor to the gauge field.
     @param[in] u The gauge field to which we apply the staggered phase factors
  */
  void applyGaugePhase(GaugeField &u);

  /**
     Compute XOR-based checksum of this gauge field: each gauge field entry is
     converted to type uint64_t, and compute the cummulative XOR of these values.
     @param[in] mini Whether to compute a mini checksum or global checksum.
     A mini checksum only computes over a subset of the lattice
     sites and is to be used for online comparisons, e.g., checking
     a field has changed with a global update algorithm.
     @return checksum value
  */
  uint64_t Checksum(const GaugeField &u, bool mini=false);

  /**
     @brief Helper function for determining if the reconstruct of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @return If reconstruct is unique return the reconstruct
   */
  inline QudaReconstructType Reconstruct_(const char *func, const char *file, int line, const GaugeField &a,
                                          const GaugeField &b)
  {
    QudaReconstructType reconstruct = QUDA_RECONSTRUCT_INVALID;
    if (a.Reconstruct() == b.Reconstruct())
      reconstruct = a.Reconstruct();
    else
      errorQuda("Reconstruct %d %d do not match (%s:%d in %s())\n", a.Reconstruct(), b.Reconstruct(), file, line, func);
    return reconstruct;
  }

  /**
     @brief Helper function for determining if the reconstruct of the fields is the same.
     @param[in] a Input field
     @param[in] b Input field
     @param[in] args List of additional fields to check reconstruct on
     @return If reconstruct is unique return the reconstrict
   */
  template <typename... Args>
  inline QudaReconstructType Reconstruct_(const char *func, const char *file, int line, const GaugeField &a,
                                          const GaugeField &b, const Args &... args)
  {
    return static_cast<QudaReconstructType>(Reconstruct_(func, file, line, a, b)
                                            & Reconstruct_(func, file, line, a, args...));
  }

#define checkReconstruct(...) Reconstruct_(__func__, __FILE__, __LINE__, __VA_ARGS__)

} // namespace quda
