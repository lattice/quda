#ifndef _GAUGE_QUDA_H
#define _GAUGE_QUDA_H

#include <quda_internal.h>
#include <quda.h>
#include <lattice_field.h>

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
#ifdef FLOAT8
          if (order == QUDA_FLOAT8_GAUGE_ORDER) return true;
#else
          if (order == QUDA_FLOAT4_GAUGE_ORDER) return true;
#endif
        }
      }
      return false;
    }
  } // namespace gauge

  struct GaugeFieldParam : public LatticeFieldParam {

    QudaFieldLocation location; // where are we storing the field (CUDA or GPU)?
    int nColor;
    int nFace;

    QudaReconstructType reconstruct;
    QudaGaugeFieldOrder order;
    QudaGaugeFixed fixed;
    QudaLinkType link_type;
    QudaTboundary t_boundary;

    double anisotropy;
    double tadpole;
    void *gauge; // used when we use a reference to an external field

    QudaFieldCreate create; // used to determine the type of field created

    QudaFieldGeometry geometry; // whether the field is a scale, vector or tensor

    // whether we need to compute the fat link maxima
    // FIXME temporary flag until we have a kernel that can do this, then we just do this in copy()
    // always set to false, requires external override
    bool compute_fat_link_max;

    /** The staggered phase convention to use */
    QudaStaggeredPhase staggeredPhaseType;

    /** Whether the staggered phase factor has been applied */
    bool staggeredPhaseApplied;

    /** Imaginary chemical potential */
    double i_mu;

    /** Offset into MILC site struct to the desired matrix field (only if gauge_order=MILC_SITE_GAUGE_ORDER) */
    size_t site_offset;

    /** Size of MILC site struct (only if gauge_order=MILC_SITE_GAUGE_ORDER) */
    size_t site_size;

    // Default constructor
    GaugeFieldParam(void *const h_gauge = NULL) :
      LatticeFieldParam(),
      location(QUDA_INVALID_FIELD_LOCATION),
      nColor(3),
      nFace(0),
      reconstruct(QUDA_RECONSTRUCT_NO),
      order(QUDA_INVALID_GAUGE_ORDER),
      fixed(QUDA_GAUGE_FIXED_NO),
      link_type(QUDA_WILSON_LINKS),
      t_boundary(QUDA_INVALID_T_BOUNDARY),
      anisotropy(1.0),
      tadpole(1.0),
      gauge(h_gauge),
      create(QUDA_REFERENCE_FIELD_CREATE),
      geometry(QUDA_VECTOR_GEOMETRY),
      compute_fat_link_max(false),
      staggeredPhaseType(QUDA_STAGGERED_PHASE_NO),
      staggeredPhaseApplied(false),
      i_mu(0.0),
      site_offset(0),
      site_size(0)
    {
    }

    GaugeFieldParam(const GaugeField &u);

    GaugeFieldParam(const int *x, const QudaPrecision precision, const QudaReconstructType reconstruct, const int pad,
                    const QudaFieldGeometry geometry, const QudaGhostExchange ghostExchange = QUDA_GHOST_EXCHANGE_PAD) :
      LatticeFieldParam(4, x, pad, precision, ghostExchange),
      location(QUDA_INVALID_FIELD_LOCATION),
      nColor(3),
      nFace(0),
      reconstruct(reconstruct),
      order(QUDA_INVALID_GAUGE_ORDER),
      fixed(QUDA_GAUGE_FIXED_NO),
      link_type(QUDA_WILSON_LINKS),
      t_boundary(QUDA_INVALID_T_BOUNDARY),
      anisotropy(1.0),
      tadpole(1.0),
      gauge(0),
      create(QUDA_NULL_FIELD_CREATE),
      geometry(geometry),
      compute_fat_link_max(false),
      staggeredPhaseType(QUDA_STAGGERED_PHASE_NO),
      staggeredPhaseApplied(false),
      i_mu(0.0),
      site_offset(0),
      site_size(0)
    {
    }

    GaugeFieldParam(void *h_gauge, const QudaGaugeParam &param, QudaLinkType link_type_ = QUDA_INVALID_LINKS) :
      LatticeFieldParam(param),
      location(QUDA_CPU_FIELD_LOCATION),
      nColor(3),
      nFace(0),
      reconstruct(QUDA_RECONSTRUCT_NO),
      order(param.gauge_order),
      fixed(param.gauge_fix),
      link_type(link_type_ != QUDA_INVALID_LINKS ? link_type_ : param.type),
      t_boundary(param.t_boundary),
      anisotropy(param.anisotropy),
      tadpole(param.tadpole_coeff),
      gauge(h_gauge),
      create(QUDA_REFERENCE_FIELD_CREATE),
      geometry(QUDA_VECTOR_GEOMETRY),
      compute_fat_link_max(false),
      staggeredPhaseType(param.staggered_phase_type),
      staggeredPhaseApplied(param.staggered_phase_applied),
      i_mu(param.i_mu),
      site_offset(param.gauge_offset),
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
#ifdef FLOAT8
          order = QUDA_FLOAT8_GAUGE_ORDER;
#else
          order = QUDA_FLOAT4_GAUGE_ORDER;
#endif
        } else {
          order = QUDA_FLOAT4_GAUGE_ORDER;
        }
      }
    }
  };

  std::ostream& operator<<(std::ostream& output, const GaugeFieldParam& param);

  class GaugeField : public LatticeField {

  protected:
      size_t bytes;        // bytes allocated per full field
      size_t phase_offset; // offset in bytes to gauge phases - useful to keep track of texture alignment
      size_t phase_bytes;  // bytes needed to store the phases
      size_t length;
      size_t real_length;
      int nColor;
      int nFace;
      QudaFieldGeometry geometry; // whether the field is a scale, vector or tensor

      QudaReconstructType reconstruct;
      int nInternal; // number of degrees of freedom per link matrix
      QudaGaugeFieldOrder order;
      QudaGaugeFixed fixed;
      QudaLinkType link_type;
      QudaTboundary t_boundary;

      double anisotropy;
      double tadpole;
      double fat_link_max;

      QudaFieldCreate create; // used to determine the type of field created

      mutable void *ghost[2 * QUDA_MAX_DIM]; // stores the ghost zone of the gauge field (non-native fields only)

      mutable int ghostFace[QUDA_MAX_DIM]; // the size of each face

      /**
         The staggered phase convention to use
      */
      QudaStaggeredPhase staggeredPhaseType;

      /**
         Whether the staggered phase factor has been applied
      */
      bool staggeredPhaseApplied;

      /**
         @brief Exchange the buffers across all dimensions in a given direction
         @param[out] recv Receive buffer
         @param[in] send Send buffer
         @param[in] dir Direction in which we are sending (forwards OR backwards only)
      */
      void exchange(void **recv, void **send, QudaDirection dir) const;

      /**
         Imaginary chemical potential
      */
      double i_mu;

      /**
         Offset into MILC site struct to the desired matrix field (only if gauge_order=MILC_SITE_GAUGE_ORDER)
      */
      size_t site_offset;

      /**
         Size of MILC site struct (only if gauge_order=MILC_SITE_GAUGE_ORDER)
      */
      size_t site_size;

      /**
         Compute the required extended ghost zone sizes and offsets
         @param[in] R Radius of the ghost zone
         @param[in] no_comms_fill If true we create a full halo
         regardless of partitioning
         @param[in] bidir Is this a bi-directional exchange - if not
         then we alias the fowards and backwards offsetss
      */
      void createGhostZone(const int *R, bool no_comms_fill, bool bidir = true) const;

      /**
         @brief Set the vol_string and aux_string for use in tuning
      */
      void setTuningString();

  public:
    GaugeField(const GaugeFieldParam &param);
    virtual ~GaugeField();

    virtual void exchangeGhost(QudaLinkDirection = QUDA_LINK_BACKWARDS) = 0;
    virtual void injectGhost(QudaLinkDirection = QUDA_LINK_BACKWARDS) = 0;

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
    virtual void exchangeExtendedGhost(const int *R, bool no_comms_fill = false) = 0;

    /**
       @brief This routine will populate the border / halo region
       of a gauge field that has been created using copyExtendedGauge.
       Overloaded variant that will start and stop a comms profile.
       @param R The thickness of the extended region in each dimension
       @param profile TimeProfile intance which will record the time taken
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    virtual void exchangeExtendedGhost(const int *R, TimeProfile &profile, bool no_comms_fill = false) = 0;

    void checkField(const LatticeField &) const;

    /**
       This function returns true if the field is stored in an
       internal field order for the given precision.
    */
    bool isNative() const;

    size_t Bytes() const { return bytes; }
    size_t PhaseBytes() const { return phase_bytes; }
    size_t PhaseOffset() const { return phase_offset; }

    virtual void* Gauge_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Even_p() { errorQuda("Not implemented"); return (void*)0;}
    virtual void* Odd_p() { errorQuda("Not implemented"); return (void*)0;}

    virtual const void* Gauge_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Even_p() const { errorQuda("Not implemented"); return (void*)0;}
    virtual const void* Odd_p() const { errorQuda("Not implemented"); return (void*)0;}

    const void** Ghost() const {
      if ( isNative() ) errorQuda("No ghost zone pointer for quda-native gauge fields");
      return (const void**)ghost;
    }

    void** Ghost() {
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
       Set all field elements to zero (virtual)
    */
    virtual void zero() = 0;

    /**
     * Generic gauge field copy
     * @param[in] src Source from which we are copying
     */
    virtual void copy(const GaugeField &src) = 0;

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

  };

  class cudaGaugeField : public GaugeField {

  private:
    void *gauge;
    void *gauge_h; // mapped-memory pointer when allocating on the host
    void *even;
    void *odd;

    /**
       @brief Initialize the padded region to 0
     */
    void zeroPad();

#ifdef USE_TEXTURE_OBJECTS
    cudaTextureObject_t tex;
    cudaTextureObject_t evenTex;
    cudaTextureObject_t oddTex;
    cudaTextureObject_t phaseTex;
    cudaTextureObject_t evenPhaseTex;
    cudaTextureObject_t oddPhaseTex;
    void createTexObject(cudaTextureObject_t &tex, void *gauge, bool full, bool isPhase=false);
    void destroyTexObject();
#endif

  public:
    cudaGaugeField(const GaugeFieldParam &);
    virtual ~cudaGaugeField();

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

    /**
       @brief Create the communication handlers and buffers
       @param[in] R The thickness of the extended region in each dimension
       @param[in] no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
       @param[in] bidir Whether to allocate communication buffers to
       allow for simultaneous bi-directional exchange.  If false, then
       the forwards and backwards buffers will alias (saving memory).
    */
    void createComms(const int *R, bool no_comms_fill, bool bidir=true);

    /**
       @brief Allocate the ghost buffers
       @param[in] R The thickness of the extended region in each dimension
       @param[in] no_comms_fill Do local exchange to fill out the extended
       @param[in] bidir Is this a bi-directional exchange - if not
       then we alias the fowards and backwards offsetss
       region in non-partitioned dimensions
    */
    void allocateGhostBuffer(const int *R, bool no_comms_fill, bool bidir=true) const;

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
    void sendStart(int dim, int dir, qudaStream_t *stream_p = nullptr);

    /**
       @brief Wait for communication to complete
       @param[in] dim The communication dimension
       @param[in] dir The communication direction (0=backwards, 1=forwards)
    */
    void commsComplete(int dim, int dir);

    /**
       @brief This does routine will populate the border / halo region of a
       gauge field that has been created using copyExtendedGauge.
       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const int *R, bool no_comms_fill=false);

    /**
       @brief This does routine will populate the border / halo region
       of a gauge field that has been created using copyExtendedGauge.
       Overloaded variant that will start and stop a comms profile.
       @param R The thickness of the extended region in each dimension
       @param profile TimeProfile intance which will record the time taken
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const int *R, TimeProfile &profile, bool no_comms_fill=false);

    /**
     * Generic gauge field copy
     * @param[in] src Source from which we are copying
     */
    void copy(const GaugeField &src);

    /**
       @brief Download into this field from a CPU field
       @param[in] cpu The CPU field source
    */
    void loadCPUField(const cpuGaugeField &cpu);

    /**
       @brief Download into this field from a CPU field.  Overloaded
       variant that includes profiling
       @param[in] cpu The CPU field source
       @param[in] profile Time profile to record the transfer
    */
    void loadCPUField(const cpuGaugeField &cpu, TimeProfile &profile);

    /**
       @brief Upload from this field into a CPU field
       @param[out] cpu The CPU field source
    */
    void saveCPUField(cpuGaugeField &cpu) const;

    /**
       @brief Upload from this field into a CPU field.  Overloaded
       variant that includes profiling.
       @param[out] cpu The CPU field source
       @param[in] profile Time profile to record the transfer
    */
    void saveCPUField(cpuGaugeField &cpu, TimeProfile &profile) const;

    // (ab)use with care
    void* Gauge_p() { return gauge; }
    void* Even_p() { return even; }
    void* Odd_p() { return odd; }

    const void* Gauge_p() const { return gauge; }
    const void* Even_p() const { return even; }
    const void *Odd_p() const { return odd; }

#ifdef USE_TEXTURE_OBJECTS
    const cudaTextureObject_t& Tex() const { return tex; }
    const cudaTextureObject_t& EvenTex() const { return evenTex; }
    const cudaTextureObject_t& OddTex() const { return oddTex; }
    const cudaTextureObject_t& EvenPhaseTex() const { return evenPhaseTex; }
    const cudaTextureObject_t& OddPhaseTex() const { return oddPhaseTex; }
#endif

    void setGauge(void* _gauge); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE

    /**
       Set all field elements to zero
    */
    void zero();

    /**
       @brief Backs up the cudaGaugeField to CPU memory
    */
    void backup() const;

    /**
       @brief Restores the cudaGaugeField to CUDA memory
    */
    void restore() const;

    /**
      @brief If managed memory and prefetch is enabled, prefetch
      the gauge field and buffers to the CPU or the GPU
      @param[in] mem_space Memory space we are prefetching to
      @param[in] stream Which stream to run the prefetch in (default 0)
    */
    void prefetch(QudaFieldLocation mem_space, qudaStream_t stream = 0) const;
  };

  class cpuGaugeField : public GaugeField {

    friend void cudaGaugeField::copy(const GaugeField &cpu);
    friend void cudaGaugeField::loadCPUField(const cpuGaugeField &cpu);
    friend void cudaGaugeField::saveCPUField(cpuGaugeField &cpu) const;

  private:
    void **gauge; // the actual gauge field

  public:
    /**
       @brief Constructor for cpuGaugeField from a GaugeFieldParam
       @param[in,out] param Parameter struct - note that in the case
       that we are wrapping host-side extended fields, this param is
       modified for subsequent creation of fields that are not
       extended.
    */
    cpuGaugeField(const GaugeFieldParam &param);
    virtual ~cpuGaugeField();

    /**
       @brief Exchange the ghost and store store in the padded region
       @param[in] link_direction Which links are we extracting: this
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

    /**
       @brief This does routine will populate the border / halo region of a
       gauge field that has been created using copyExtendedGauge.

       @param R The thickness of the extended region in each dimension
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimenions
    */
    void exchangeExtendedGhost(const int *R, bool no_comms_fill=false);

    /**
       @brief This does routine will populate the border / halo region
       of a gauge field that has been created using copyExtendedGauge.
       Overloaded variant that will start and stop a comms profile.
       @param R The thickness of the extended region in each dimension
       @param profile TimeProfile intance which will record the time taken
       @param no_comms_fill Do local exchange to fill out the extended
       region in non-partitioned dimensions
    */
    void exchangeExtendedGhost(const int *R, TimeProfile &profile, bool no_comms_fill=false);

    /**
     * Generic gauge field copy
     * @param[in] src Source from which we are copying
     */
    void copy(const GaugeField &src);

    void* Gauge_p() { return gauge; }
    const void* Gauge_p() const { return gauge; }

    void setGauge(void** _gauge); //only allowed when create== QUDA_REFERENCE_FIELD_CREATE

    /**
       Set all field elements to zero
    */
    void zero();

    /**
       @brief Backs up the cpuGaugeField
    */
    void backup() const;

    /**
       @brief Restores the cpuGaugeField
    */
    void restore() const;
  };

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
  cudaGaugeField *createExtendedGauge(cudaGaugeField &in, const int *R, TimeProfile &profile,
                                      bool redundant_comms = false, QudaReconstructType recon = QUDA_RECONSTRUCT_INVALID);

  /**
     This function is used for creating an exteneded (cpu) gauge field from the input,
     and copying the gauge field into the extended gauge field.  Defined in lib/gauge_field.cpp.
     @param in The input field from which we are extending
     @param R By how many do we want to extend the gauge field in each direction
     @return the pointer to the extended gauge field
  */
  cpuGaugeField *createExtendedGauge(void **gauge, QudaGaugeParam &gauge_param, const int *R);

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
     This function is used for  extracting the gauge ghost zone from a
     gauge field array.  Defined in extract_gauge_ghost.cu.
     @param u The gauge field from which we want to extract/pack the ghost zone
     @param dim The dimension in which we are packing/unpacking
     @param ghost The array where we want to pack/unpack the ghost zone into/from
     @param extract Whether we are extracting into ghost or injecting from ghost
  */
  void extractExtendedGaugeGhost(const GaugeField &u, int dim, const int *R, void **ghost, bool extract);

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

#endif // _GAUGE_QUDA_H
